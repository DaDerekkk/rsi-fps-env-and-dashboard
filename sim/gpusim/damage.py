"""Shot -> damage pipeline for E matches (port of fpssim.arena._do_shot / _do_knife / _damage), see docs/gpu_rounds.md.

    D = Damage(rounds, shooter, grenades=None)
    D.run(out, tick)          # out = gpusim.weapons.tick(...) result of this tick; applies spread, bullets, damage, kills

Stages (no host sync: fixed-capacity row buffers of E*10*PMAX pellets, PMAX = most pellets of any weapon; the used rows are packed at
the front in CPU call order, the rest are marked env = -1 and skipped by every kernel):
  1. rows: one per pellet of every shot (ordered by match, seat, pellet = the CPU call order);
  2. rounds_cuda.build_rays: cs2-web accuracy from the pre-kick punch / penalty (`out['ap_pitch'/'ap_yaw'/'pen']`, zoom after the
     shot like the CPU), two-disc spread with the counter-based RNG u01(seed, 1, match tick, seat*16 + pellet, 0..3);
  3. gpusim.shoot.Shooter.trace: hit volumes + wall penetration, per pellet at most 4 player hits with their penetration scale;
  4. gpusim.grenades carve_bullet (smoke carving by bullets), if a grenade module is attached;
  5. rounds_cuda.settle: sequential per match over shooter seats 0..9: combat.damage formula x penetration scale, armor, kill
     processing (Rounds' `kill_one`: drops, rewards, CT bonus), knife swings (fpssim.melee), 'fire'/'hit' counters.
Known, documented differences to the CPU (see docs/gpu_rounds.md): a shooter killed by a lower seat in the same tick does not
fire (its weapon state was already advanced by the batched weapons tick); bullets are traced against the players alive at the
start of the tick, so a later pellet of the same tick still 'sees' a player that an earlier pellet / shooter just killed.
"""
import torch


class Damage:
    def __init__(self, rounds, shooter, grenades=None):
        self.R = rounds
        self.shooter = shooter
        self.grenades = grenades
        dv = rounds.dev; E = rounds.E
        self.PMAX = PM = int(rounds.T['pellets'].max())
        self.cap = cap = E * 10 * PM
        self.arange = torch.arange(E * 10, device=dv)
        self.pgrid = self.arange[:, None].expand(-1, PM).reshape(-1).contiguous()     # player index of every (player, pellet) slot
        self.jgrid = torch.arange(PM, device=dv)[None, :]
        f64 = lambda *sh: torch.zeros(*sh, dtype=torch.float64, device=dv)
        i64 = lambda *sh: torch.zeros(*sh, dtype=torch.long, device=dv)
        self.row_player = torch.full((cap + 1,), -1, dtype=torch.long, device=dv)     # last element: dump slot of the scatter
        self.org, self.dr, self.env, self.seat = f64(cap, 3), f64(cap, 3), i64(cap), i64(cap)
        self.wr, self.wp, self.wd = f64(cap), f64(cap), f64(cap)
        self.tr = None
        self._last = None
        self.keep = False; self.dbg = []          # test hook: keep the rays and trace results of every tick (debugging only, syncs)

    @property
    def last(self):
        """rows (pellets) and knife swings of the last run (reading it syncs)."""
        if self._last is None: return dict(rows=0, knives=0)
        tot, nk = torch.stack(self._last).tolist()
        return dict(rows=tot, knives=nk)

    def run(self, out, tick):
        """out: gpusim.weapons.tick result; tick: int or per-match tick tensor [E] (smoke carving clock)."""
        R = self.R; s = R.s; E = R.E; T = R.T; cap = self.cap
        wid = out['wid'].contiguous()                                    # [E,10] long: weapon fired this tick
        shot, kn = out['shot'], out['knife']
        npel = torch.where(shot, T['pellets'][wid.clamp(min=0)], torch.zeros_like(wid)).reshape(-1).contiguous()
        csum = npel.cumsum(0)
        row_start = (csum - npel).contiguous()
        self._last = (csum[-1], kn.sum())
        evm = [out[k].contiguous().reshape(-1) for k in ('ev_reload_start', 'ev_commit', 'ev_finish', 'ev_draw')]
        # pack the pellet rows: row row_start[p] + j belongs to player p (j < npel[p]); unused rows keep row_player = -1
        pos = torch.where(self.jgrid < npel[:, None], row_start[:, None] + self.jgrid, torch.full_like(row_start[:, None], cap)).reshape(-1)
        rp = self.row_player
        rp.fill_(-1)
        rp.scatter_(0, pos, self.pgrid)
        org, dr, env, seat, wr, wp, wd = self.org, self.dr, self.env, self.seat, self.wr, self.wp, self.wd
        R.k.build_rays(R.tab, rp[:cap], row_start, wid.reshape(-1), out['ap_pitch'].reshape(-1).contiguous(),
                       out['ap_yaw'].reshape(-1).contiguous(), out['pen'].reshape(-1).contiguous(), org, dr, env, seat, wr, wp, wd)
        ppos = torch.stack([s['x'], s['y'], s['z']], -1).contiguous()
        tr = self.tr = self.shooter.trace(org, dr, env, seat, wr, wp, wd, ppos, s['yaw'], s['crouch'], s['alive'], out=self.tr)
        if self.keep:
            n = int(csum[-1])
            self.dbg.append(dict(tick=tick, org=org[:n].cpu().numpy(), dir=dr[:n].cpu().numpy(), env=env[:n].cpu().numpy(),
                                 shooter=seat[:n].cpu().numpy(),
                                 **{k: v[:n].cpu().numpy() for k, v in tr.items() if k in ('end', 'seat', 'dist', 'part', 'scale', 'nhit')}))
        if self.grenades is not None:
            p1 = org + dr * tr['end'][:, None]
            start = torch.cat([row_start.view(E, 10)[:, 0], csum[-1:]])     # rows of match e: start[e] .. start[e+1]-1
            self.grenades.carve_rows(org, p1, start, tick)
        R.k.settle(R.tab, E, row_start, npel, kn.reshape(-1).contiguous(), wid.reshape(-1), tr['seat'], tr['dist'], tr['part'], tr['scale'],
                   tr['nhit'], *evm)
