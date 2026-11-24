"""Batched GPU utilities / grenades: port of fpssim.grenades.GrenadeSystem for E independent matches.

    from gpusim.grenades import GpuGrenades, KINDS
    g = GpuGrenades(map_static, mesh_or_none, E, device)
    th = g.handle_throw(state, action, tick_mask)    # live ticks, after movement, before step()
    ev = g.step(state, tick, live_mask)               # projectiles, detonations, smoke/fire/decoy, damage, flash
    blk = g.smoke_blocks(p0[R,3], p1[R,3], env_idx[R], tick)   -> bool[R]
    g.carve_bullet(p0[R,3], p1[R,3], env_idx[R], tick)          (every bullet of the tick, before step())
    g.clear(env_mask)                                           (new round)

See docs/gpu_grenades.md for the tensor layout, caps and parity results.

state (dict of tensors, SPEC names; written in place):
    read : x y z vx vy vz yaw pitch [E,10] f64, crouch alive [E,10] bool, tick [E] i32 (if `tick` arg is None)
    write: hp armor [E,10] f64, alive [E,10] bool, blind_until blind_start throw_ready [E,10] i32,
           nades [E,10,6] i32 (counts in KINDS order), optional death_tick [E,10] i32, optional protection_until.
kills: damage that takes hp to <= 1e-9 sets hp=0, alive=False, death_tick and ev['ev_killer'][E,10] (killer seat, -1 none) / ev['ev_kill_kind'];
       the caller runs the death side effects (GpuSim: Rounds.kill_pass(alive_before, ev['ev_killer'])).
action (dict of tensors [E,10], any int dtype):
    throw_kind  (-1 none, 0..5 = KINDS index), throw_state (0 idle, 1 prime, 2 release, 3 cancel),
    throw_mode  (0 full, 1 lob, 2 drop)
"""
import numbers
import numpy as np
import torch

from fpssim import grenades as CG
from .bvh import build_bvh
from . import grenades_cuda as K

KINDS = CG.KINDS                       # ('hegrenade','flashbang','smokegrenade','molotov','incgrenade','decoy')
MODES = ('full', 'lob', 'drop')
STATES = ('idle', 'prime', 'release', 'cancel')


def _f32pad(a, sgn, pad=2e-3):
    return (np.asarray(a, np.float64) + sgn * pad).astype(np.float32)


class GpuWorld:
    """Grenade world contact on the GPU: TriMesh (Dust2/Mirage) via BVH, or whitebox BoxContact boxes."""

    def __init__(self, map_static=None, mesh=None, device='cuda'):
        dev = device
        z = lambda dt: torch.zeros(0, dtype=dt, device=dev)
        if mesh is not None:
            mc = CG.MeshContact(mesh)
            b = build_bvh(mesh.tri, leaf=4)
            order = b['tri_id'].astype(np.int64)
            # conservative float32 node boxes (padded 2 mm); triangles exactly as TriMesh (float64 v0/e1/e2)
            nlo = _f32pad(b['node_lo'], -1); nhi = _f32pad(b['node_hi'], +1)
            arr = dict(nlo=nlo, nhi=nhi, nleft=b['node_left'], nstart=b['node_start'], ncount=b['node_count'],
                       v0=mesh.v0[order], e1=mesh.e1[order], e2=mesh.e2[order], nrm=mc.n[order],
                       tid=b['tri_id'])
            self.t = {k: torch.as_tensor(np.ascontiguousarray(v), device=dev) for k, v in arr.items()}
            self.t['blo'] = z(torch.float64); self.t['bhi'] = z(torch.float64)
            self.wp = [0.0, 0.0, 0.0, -1e9, -1e9, 1e9, 1e9, mc.cs_offset[0], mc.cs_offset[1]]
            self.cpu = mc
        else:
            st = map_static.get('map', map_static) if isinstance(map_static, dict) else map_static
            bc = CG.BoxContact(st)
            self.t = dict(nlo=z(torch.float32), nhi=z(torch.float32), nleft=z(torch.int32), nstart=z(torch.int32),
                          ncount=z(torch.int32), v0=z(torch.float64), e1=z(torch.float64), e2=z(torch.float64),
                          nrm=z(torch.float64), tid=z(torch.int32),
                          blo=torch.as_tensor(np.ascontiguousarray(bc.lo), device=dev),
                          bhi=torch.as_tensor(np.ascontiguousarray(bc.hi), device=dev))
            self.wp = [1.0, float(len(bc.lo)), bc.ground, float(bc.gmin[0]), float(bc.gmin[1]), float(bc.gmax[0]),
                       float(bc.gmax[1]), bc.cs_offset[0], bc.cs_offset[1]]
            self.cpu = bc
        for k, ty in K.WD_FIELDS:
            assert self.t[k].dtype == K.TORCH_DT[ty], (k, self.t[k].dtype)
        self.list = [self.t[k].contiguous() for k, _ in K.WD_FIELDS]

    def contact(self, o, d, L):
        """[R,3],[R,3],[R] float64 -> [R,4] (dist or nan, normal); for tests."""
        return K.ext().contact_call(self.list, self.wp, o.double().contiguous(), d.double().contiguous(),
                                    L.double().contiguous())


class GpuGrenades:
    def __init__(self, map_static, mesh_or_none, E, device='cuda', P=24, S=12, F=12, D=12, C=40, NC=60, NI=4):
        """P projectiles, S smokes, F fires (NC cells each), D decoys, C carves per smoke, per match."""
        assert max(P, S, F, D) <= 64, 'tables are limited to 64 entries per match'
        self.E, self.P, self.S, self.F, self.D, self.C, self.NC, self.NI = E, P, S, F, D, C, NC, NI
        self.dev = device
        self.world = GpuWorld(map_static, mesh_or_none, device)
        self.dims = [E, P, S, C, F, NC, D, NI]
        U = CG.UTIL
        self.ui = torch.tensor([[int(U[k]['fuse']) for k in KINDS], [int(U[k].get('duration', 0)) for k in KINDS]],
                               dtype=torch.int32)
        self.ud = torch.tensor([[float(U[k]['radius']) for k in KINDS], [float(U[k].get('damage', 0.0)) for k in KINDS]],
                               dtype=torch.float64)
        shp = {'pr': (E, P), 'sm': (E, S), 'cv': (E, S, C), 'fi': (E, F), 'dc': (E, D), 'pm': (E, 10), 'ct': (E,), 'ig': (E,)}
        self.g = {}
        for n, ty in K.GS_FIELDS:
            pre = n[:2] if n[:3] != 'ctr' else 'ct'
            s = shp[pre]
            if n == 'ig_slot': s = (E, P)
            elif n == 'ig_proj': s = (E, NI)
            elif n == 'ig_ok': s = (E, NI, 64)
            elif n == 'ig_cell': s = (E, NI, 64, 3)
            if n in ('pr_pos', 'pr_vel', 'sm_pos', 'cv_a', 'cv_b', 'fi_pos', 'dc_pos'):
                s = s + (3,)
            if n == 'fi_cells':
                s = (E, F, NC, 3)
            if n == 'fi_cok':
                s = (E, F, NC)
            self.g[n] = torch.zeros(s, dtype=K.TORCH_DT[ty], device=device)
        self.g['pr_kind'].fill_(-1); self.g['pm_kind'].fill_(-1); self.g['ig_slot'].fill_(-1); self.g['ig_proj'].fill_(-1)
        self.glist = [self.g[n] for n, _ in K.GS_FIELDS]
        K.ext()

    # ----------------------------------------------------------------------------------------- helpers
    def _tick(self, state, tick):
        if tick is None:
            t = state['tick']
        elif isinstance(tick, numbers.Integral):
            t = torch.full((self.E,), int(tick), dtype=torch.int32, device=self.dev)
        else:
            t = torch.as_tensor(tick)
            if t.dim() == 0:
                t = t.expand(self.E)
        return t.to(device=self.dev, dtype=torch.int32).contiguous()

    def _mask(self, m):
        if m is None:
            return torch.ones(self.E, dtype=torch.bool, device=self.dev)
        if isinstance(m, bool):
            return torch.full((self.E,), m, dtype=torch.bool, device=self.dev)
        return m.to(device=self.dev, dtype=torch.bool).contiguous()

    def _pl(self, state):
        out = []
        for n, ty in K.PL_FIELDS:
            t = state.get(n)
            if t is None:
                if n in ('death_tick', 'protection_until'):
                    out.append(torch.zeros(0, dtype=K.TORCH_DT[ty], device=self.dev)); continue
                raise KeyError('state[%r] missing' % n)
            if t.dtype != K.TORCH_DT[ty] or not t.is_contiguous() or t.device.type != 'cuda':
                raise TypeError('state[%r] must be a contiguous %s CUDA tensor (got %s)' % (n, K.TORCH_DT[ty], t.dtype))
            out.append(t)
        return out

    def _ev(self, kind):
        E, P = self.E, self.P
        ev = {}
        for n, ty in K.EV_FIELDS:
            if kind == 'throw' and not n.startswith('th_') or kind == 'step' and n.startswith('th_'):
                ev[n] = None; continue
            s = {'ev_bounce': (E, P), 'ev_bounce_pos': (E, P, 3), 'ev_det': (E, P), 'ev_det_id': (E, P),
                 'ev_det_kind': (E, P), 'ev_det_pos': (E, P, 3), 'ev_fire_cells': (E, P), 'ev_fire_ext': (E,),
                 'ev_decoy_pulse': (E,), 'ev_decoy_exp': (E,), 'th_pos': (E, 10, 3)}.get(n, (E, 10))
            ev[n] = torch.zeros(s, dtype=K.TORCH_DT[ty], device=self.dev)
        for n in ('ev_det_kind', 'ev_killer', 'ev_kill_kind', 'th_kind', 'th_slot', 'th_id'):
            if ev[n] is not None:
                ev[n].fill_(-1)
        lst = [ev[n] if ev[n] is not None else torch.zeros(0, dtype=K.TORCH_DT[ty], device=self.dev)
               for n, ty in K.EV_FIELDS]
        return {k: v for k, v in ev.items() if v is not None}, lst

    # ----------------------------------------------------------------------------------------- API
    def handle_throw(self, state, action, tick_mask=None, tick=None):
        """Throw state machine for every alive player of the masked matches (seat order, like Arena).
        Returns dict th_flag/th_kind/th_slot/th_id [E,10] i32, th_pos [E,10,3] f64."""
        ev, lst = self._ev('throw')
        a = lambda k: action[k].to(device=self.dev, dtype=torch.int32).contiguous()
        K.ext().throw_call(self.glist, self._pl(state), lst, self.world.list, self.world.wp, self.dims, self.ui, self.ud,
                           a('throw_kind'), a('throw_state'), a('throw_mode'), self._mask(tick_mask),
                           self._tick(state, tick))
        return ev

    def step(self, state, tick=None, live_mask=None):
        """One live 60 Hz tick for the masked matches.  Returns event tensors (see docs/gpu_grenades.md)."""
        ev, lst = self._ev('step')
        K.ext().step_call(self.glist, self._pl(state), lst, self.world.list, self.world.wp, self.dims, self.ui, self.ud,
                          self._mask(live_mask), self._tick(state, tick))
        return ev

    def _bucket(self, env_idx):
        env = env_idx.to(device=self.dev, dtype=torch.int64)
        order = torch.sort(env, stable=True).indices
        cnt = torch.bincount(env, minlength=self.E)
        start = torch.zeros(self.E + 1, dtype=torch.int64, device=self.dev)
        start[1:] = torch.cumsum(cnt, 0)
        return order.contiguous(), start

    def carve_bullet(self, p0, p1, env_idx, tick):
        """Bullet traces (eye -> trace end) of this tick; per match applied in row order (= CPU call order)."""
        if p0.shape[0] == 0:
            return
        order, start = self._bucket(env_idx)
        K.ext().carve_call(self.glist, self.dims, self.ui, self.ud, p0.double().contiguous(), p1.double().contiguous(),
                           order, start, self._tick(None, tick) if tick is not None else None)

    def carve_rows(self, p0, p1, start, tick):
        """carve_bullet for rows already grouped by match: rows start[e] .. start[e+1]-1 belong to match e (row order = CPU call
        order); no host sync (gpusim.damage)."""
        if not hasattr(self, '_ident') or self._ident.numel() < p0.shape[0]:
            self._ident = torch.arange(max(p0.shape[0], 1), dtype=torch.int64, device=self.dev)
        K.ext().carve_call(self.glist, self.dims, self.ui, self.ud, p0.double().contiguous(), p1.double().contiguous(),
                           self._ident, start.contiguous(), self._tick(None, tick))

    def smoke_blocks(self, p0, p1, env_idx, tick):
        env = env_idx.to(device=self.dev, dtype=torch.int64).contiguous()
        return K.ext().smoke_call(self.glist, self.dims, self.ui, self.ud, p0.double().contiguous(),
                                  p1.double().contiguous(), env, self._tick(None, tick))

    def clear(self, env_mask=None):
        """New round: drop projectiles, smokes, fires, decoys and primes of the masked matches.
        (blind_until/blind_start live in the shared state and are reset by the round logic.)"""
        m = self._mask(env_mask)
        g = self.g
        mm = m[:, None]                                   # masked_fill_: no host sync (graph capturable)
        g['pr_kind'].masked_fill_(mm, -1); g['pm_kind'].masked_fill_(mm, -1); g['pm_rel'].masked_fill_(mm, 0)
        for n in ('sm_valid', 'sm_ncv', 'fi_valid', 'dc_valid'):
            g[n].masked_fill_(mm, 0)

    def add_smoke(self, env_idx, pos, created, expires=None, owner=0):
        """test / benchmark helper: write a smoke record (centre `pos` [R,3], already including the +1.3 m) into the first free smoke
        slot of every match in env_idx (long [R], unique).  created / expires: int tensors or ints (expires default created+1080)."""
        g = self.g; env = torch.as_tensor(env_idx, device=self.dev).long()
        pos = torch.as_tensor(pos, dtype=torch.float64, device=self.dev).reshape(-1, 3)
        cr = torch.as_tensor(created, dtype=torch.int32, device=self.dev).expand(len(env))
        ex = (cr + int(CG.UTIL['smokegrenade']['duration'])) if expires is None else torch.as_tensor(expires, dtype=torch.int32, device=self.dev).expand(len(env))
        free = g['sm_valid'][env] == 0
        ok = free.any(1)
        slot = free.int().argmax(1)
        env, slot, pos, cr, ex = env[ok], slot[ok], pos[ok], cr[ok], ex[ok]
        g['ctr_serial'][env] += 1; g['ctr_id'][env] += 1
        g['sm_valid'][env, slot] = 1; g['sm_created'][env, slot] = cr; g['sm_expires'][env, slot] = ex
        g['sm_serial'][env, slot] = g['ctr_serial'][env]; g['sm_id'][env, slot] = g['ctr_id'][env]; g['sm_owner'][env, slot] = owner
        g['sm_ncv'][env, slot] = 0; g['sm_pos'][env, slot] = pos
        return ok

    def is_primed(self):
        return self.g['pm_kind']

    def smoke_spheres(self, tick):
        """[E,S,4] (x,y,z,bloom radius) and valid [E,S] (coarse sphere list, like GrenadeSystem.smoke_spheres)."""
        g = self.g
        t = self._tick(None, tick).double()[:, None]
        age = ((t - g['sm_created'].double()) / 60).clamp(min=0)
        prog = (age / 1.8).clamp(max=1.0)
        r = CG.UTIL['smokegrenade']['radius'] * (0.15 + 0.85 * (1 - (1 - prog) ** 3))
        valid = (g['sm_valid'] > 0) & (g['sm_expires'] > t.int())
        return torch.cat([g['sm_pos'], r[..., None]], -1), valid

    def snapshot(self, e):
        """CPU dict of match e (projectiles / smokes / fires / decoys) for debugging and parity tests."""
        g = {k: v[e].cpu().numpy() for k, v in self.g.items() if v.dim() > 0 and v.shape[0] == self.E}
        pr = {int(g['pr_id'][p]): dict(kind=KINDS[g['pr_kind'][p]], pos=g['pr_pos'][p].copy(), vel=g['pr_vel'][p].copy(),
                                       resting=bool(g['pr_rest'][p]), bounces=int(g['pr_bounces'][p]))
              for p in range(self.P) if g['pr_kind'][p] >= 0}
        sm = {int(g['sm_id'][s]): dict(pos=g['sm_pos'][s].copy(), created=int(g['sm_created'][s]),
                                       expires=int(g['sm_expires'][s]), ncarves=int(g['sm_ncv'][s]))
              for s in range(self.S) if g['sm_valid'][s]}
        fi = {int(g['fi_id'][f]): dict(kind=KINDS[g['fi_kind'][f]], cells=g['fi_cells'][f][g['fi_cok'][f] > 0].copy(),
                                       expires=int(g['fi_expires'][f]), next=int(g['fi_next'][f]))
              for f in range(self.F) if g['fi_valid'][f]}
        dc = {int(g['dc_id'][d]): dict(pos=g['dc_pos'][d].copy(), expires=int(g['dc_expires'][d]))
              for d in range(self.D) if g['dc_valid'][d]}
        return dict(projectiles=pr, smokes=sm, fires=fi, decoys=dc)
