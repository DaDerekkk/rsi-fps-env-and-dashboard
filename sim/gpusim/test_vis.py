"""Parity gpusim.vis.visible vs fpssim.visibility.visible_enemies on recorded frames, with and without smoke; throughput.

usage (server, from sim/; run in the background, the CPU mesh binning takes ~1 min):
  python3 -m gpusim.test_vis [--map dust2-cs2-v54|nevaio-r3-v54|borgo-r3-v54|molo-r3-v54] [--frames 600] [--seed 1] [--bench]

Frames: live-phase frames of recorded games on the map (frames.npz).  CPU reference: visible_enemies with LOS on the TriMesh
(Dust2) or on BoxContact (whitebox); the GPU side uses the exact Dust2 mesh / the triangulated whitebox (vis.whitebox_mesh).
Smoke section: random smokes (random age / expiry, between an observer and an enemy) and random bullet carves are written into
both a CPU GrenadeSystem per frame and GpuGrenades; GPU = visible(..., smoke_fn=GpuGrenades.smoke_blocks), CPU = visible_enemies(...,
GrenadeSystem.smoke_blocks).
"""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch

from fpssim import grenades as CG
from fpssim.visibility import visible_enemies, LOS
from gpusim.ray import GpuMesh
from gpusim.vis import visible, whitebox_mesh, candidates
from gpusim.grenades import GpuGrenades

ROOT = os.environ.get('FPS_ROOT', './fps_data')
DEV = 'cuda'
PN = ('head', 'chest', 'abdomen')


def load_world(map_id):
    d = json.load(open(f'{ROOT}/data/static/static_{map_id}.json'))
    st = d.get('map', d)
    if st.get('boxes'):
        return st, None, whitebox_mesh(st, DEV), LOS(CG.BoxContact(st))
    from fpssim.trimesh import TriMesh
    col = st['collision'][0]
    M = TriMesh(f'{ROOT}/ref/assets/map_positions.f32', col['basis'], col['offset'])
    return st, M, GpuMesh(M.tri, DEV), LOS(M)


def load_frames(map_id, nmax, step=5, rng=None):
    """live frames of the recorded games of this map -> dict of [N,10] arrays + tick [N]."""
    out = None
    for d in sorted(glob.glob(f'{ROOT}/data/rec/*/frames.npz')):
        job = os.path.dirname(d)
        try:
            sid = json.loads(open(job + '/static.json').readline())['map']['id']
        except Exception:
            continue
        if sid != map_id:
            continue
        z = np.load(d)
        P, Gf, cols = z['P'], z['G'], list(z['cols'])
        c = {k: i for i, k in enumerate(cols)}
        fr = [f for f in range(0, len(P), step) if Gf[f, 8] == 1]
        if rng is not None and len(fr) > nmax // 3:
            # half of the frames: some recorded enemy sight line exists (platform `vis` bitmask), half: random live frames
            em = np.array([0x3E0] * 5 + [0x1F] * 5)
            hot = [f for f in fr if (z['vis'][f] & em).any()]
            rest = [f for f in fr if f not in set(hot)]
            nh = min(len(hot), nmax // 6)
            fr = sorted(list(rng.choice(hot, nh, replace=False)) + list(rng.choice(rest, min(len(rest), nmax // 3 - nh), replace=False)))
        cur = {k: np.stack([P[f, :, c[k]] for f in fr]).astype(np.float64) for k in
               ('x', 'y', 'z', 'yaw', 'pitch', 'crouch', 'alive', 'blind_start', 'blind_until')}
        cur['tick'] = np.array([int(Gf[f, 0]) for f in fr], np.int64)
        out = cur if out is None else {k: np.concatenate([out[k], cur[k]]) for k in out}
        if len(out['tick']) >= nmax:
            break
    return {k: v[:nmax] for k, v in out.items()}


def gpu_state(fr):
    s = {k: torch.tensor(fr[k], dtype=torch.float64, device=DEV) for k in ('x', 'y', 'z', 'yaw', 'pitch')}
    s['crouch'] = torch.tensor(fr['crouch'] > .5, device=DEV); s['alive'] = torch.tensor(fr['alive'] > .5, device=DEV)
    s['blind_start'] = torch.tensor(fr['blind_start'], dtype=torch.int32, device=DEV)
    s['blind_until'] = torch.tensor(fr['blind_until'], dtype=torch.int32, device=DEV)
    return s


def cpu_players(fr, i):
    return [dict(seat=q, team=q // 5, alive=fr['alive'][i, q] > .5, x=float(fr['x'][i, q]), y=float(fr['y'][i, q]), z=float(fr['z'][i, q]),
                 yaw=float(fr['yaw'][i, q]), pitch=float(fr['pitch'][i, q]), crouch=fr['crouch'][i, q] > .5,
                 blind_start=float(fr['blind_start'][i, q]), blind_until=float(fr['blind_until'][i, q])) for q in range(10)]


def compare(out, cpu_all, fr, tag):
    tot = pos = agree = bad_parts = 0
    pts_cpu = 0
    bad = []
    for k, cpu in enumerate(cpu_all):
        for o in range(10):
            cm = {g: set(p) for g, p in cpu[o]}
            for g in range(10):
                if g // 5 == o // 5:
                    continue
                gp = {PN[j] for j in range(3) if out[k, o, g, j]}
                c = cm.get(g, set())
                tot += 1; pos += bool(c); agree += gp == c
                pts_cpu += len(c); bad_parts += len(gp ^ c)
                if gp != c and len(bad) < 5:
                    bad.append((k, o, g, sorted(c), sorted(gp)))
    print('[%s] enemy pairs %d (visible on CPU: %d, %d points), identical part sets %d (%.6f), differing points %d' %
          (tag, tot, pos, pts_cpu, agree, agree / max(1, tot), bad_parts))
    for b in bad:
        print('   mismatch frame %d obs %d tgt %d  cpu %s gpu %s' % b)
    return tot - agree


def inject_smokes(rng, fr, G, gss, tick):
    """random smokes between an observer and an enemy (+ random bullet carves) into the CPU systems and GpuGrenades."""
    N = len(gss)
    ids, poss, cre, exp = [], [], [], []
    for i in range(N):
        if rng.random() > 0.8:
            continue
        for _ in range(rng.integers(1, 4)):
            o = rng.integers(10); g = rng.integers(10)
            eye = np.array([fr['x'][i, o], fr['y'][i, o], fr['z'][i, o] + 1.62]); tg = np.array([fr['x'][i, g], fr['y'][i, g], fr['z'][i, g] + 1.0])
            p = eye + (tg - eye) * rng.uniform(0.15, 0.85) + rng.normal(0, 0.8, 3)
            age = int(rng.choice([0, 3, 12, 40, 100, 200, 600, 1090])) if rng.random() < 0.9 else -5
            c = int(tick[i]) - age; e_ = c + 1080
            gss[i].smokes.append(dict(id=len(gss[i].smokes) + 1, owner=0, team=0, kind='smokegrenade', position=[float(v) for v in p], created=c,
                                      expires=e_, radius=4.5, carves=[]))
            ids.append(i); poss.append(p); cre.append(c); exp.append(e_)
    # GPU: one injection pass per layer (env unique within a pass)
    ids = np.array(ids); poss = np.array(poss); cre = np.array(cre); exp = np.array(exp)
    seen = {}
    layer = np.array([seen.__setitem__(i, seen.get(i, -1) + 1) or seen[i] for i in ids])
    for L in range(layer.max() + 1 if len(layer) else 0):
        m = layer == L
        G.add_smoke(torch.tensor(ids[m], device=DEV), torch.tensor(poss[m], device=DEV), torch.tensor(cre[m], device=DEV), torch.tensor(exp[m], device=DEV))
    # bullet carves aimed at smokes, on random ticks in the last 0.8 s
    bl = []
    for i in range(N):
        for s in gss[i].smokes:
            if rng.random() < 0.6:
                c = np.array(s['position']); a = c + rng.normal(0, 5, 3); dv = c + rng.normal(0, 1.5, 3) - a
                b = a + dv / max(1e-9, np.linalg.norm(dv)) * rng.uniform(6, 25)
                tc = int(tick[i]) - int(rng.integers(0, 30))
                bl.append((i, tc, a, b))
    bl.sort(key=lambda x: x[1])
    for tc in sorted({b[1] for b in bl}):
        rows = [b for b in bl if b[1] == tc]
        for (i, _, a, b) in rows:
            gss[i].carve_bullet(tuple(a), tuple(b), tc)
        G.carve_bullet(torch.tensor(np.stack([r[2] for r in rows]), device=DEV), torch.tensor(np.stack([r[3] for r in rows]), device=DEV),
                       torch.tensor([r[0] for r in rows], device=DEV), tc)


def run_parity(map_id, nframes, seed, with_smoke=True):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    st, M, mesh, los = load_world(map_id)
    fr = load_frames(map_id, nframes, rng=rng)
    N = len(fr['tick'])
    print('map %s: %d frames, world ready (%.0fs)' % (map_id, N, time.time() - t0), flush=True)
    s = gpu_state(fr)
    tick = torch.tensor(fr['tick'], dtype=torch.int32, device=DEV)
    # ---- no smoke
    out = visible(s, mesh, tick).cpu().numpy()
    cpu_all = [visible_enemies(cpu_players(fr, i), los, None, float(fr['tick'][i])) for i in range(N)]
    nbad = compare(out, cpu_all, fr, 'no smoke')
    # ---- with smoke
    if with_smoke:
        contact = CG.MeshContact(M) if M is not None else CG.BoxContact(st)
        G = GpuGrenades(st, M, N, DEV)
        gss = [CG.GrenadeSystem(contact) for _ in range(N)]
        inject_smokes(rng, fr, G, gss, fr['tick'])
        out2 = visible(s, mesh, tick, G.smoke_blocks).cpu().numpy()
        cpu2 = [visible_enemies(cpu_players(fr, i), los, gss[i].smoke_blocks, int(fr['tick'][i])) for i in range(N)]
        nbad += compare(out2, cpu2, fr, 'smoke')
        nsm = sum(len(g.smokes) for g in gss); ncv = sum(len(sm['carves']) for g in gss for sm in g.smokes)
        hid = int((out & ~out2).sum()); print('   %d smokes, %d carves; points hidden by smoke (GPU): %d' % (nsm, ncv, hid))
        # smoke_blocks on raw segments
        R = 20000
        e = rng.integers(0, N, R); a = np.stack([fr['x'][e, 0], fr['y'][e, 0], fr['z'][e, 0]], 1) + rng.normal(0, 8, (R, 3)); b = a + rng.normal(0, 8, (R, 3))
        # re-anchor half of the segments on a random smoke centre so that many intersect
        for r in range(R // 2):
            if gss[e[r]].smokes:
                c = np.array(gss[e[r]].smokes[rng.integers(len(gss[e[r]].smokes))]['position']); a[r] = c + rng.normal(0, 4, 3); b[r] = c + rng.normal(0, 4, 3)
        gb = G.smoke_blocks(torch.tensor(a, device=DEV), torch.tensor(b, device=DEV), torch.tensor(e, device=DEV), tick).cpu().numpy()
        cb = np.array([gss[e[r]].smoke_blocks(a[r], b[r], int(fr['tick'][e[r]])) for r in range(R)])
        print('[smoke_blocks raw segments] %d queries, %d blocked on CPU, mismatches %d' % (R, cb.sum(), (cb != gb).sum()))
        nbad += int((cb != gb).sum())
    print('PARITY %s: %d differing pairs (%.0fs)' % (map_id, nbad, time.time() - t0))
    return nbad


def bench(map_id, seed=1):
    rng = np.random.default_rng(seed)
    st, M, mesh, los = load_world(map_id)
    fr = load_frames(map_id, 600, rng=rng)
    N = len(fr['tick'])
    for E in (4096, 16384):
        idx = rng.integers(0, N, E)
        ff = {k: v[idx] for k, v in fr.items()}
        s = gpu_state(ff); tick = torch.tensor(ff['tick'], dtype=torch.int32, device=DEV)
        G = GpuGrenades(st, M, E, DEV)
        # ~40% of the matches have 1-3 smokes near a player
        for L in range(3):
            envs = torch.nonzero(torch.rand(E, device=DEV) < (0.4 if L == 0 else 0.15)).squeeze(1)
            pos = torch.stack([s['x'][envs, 0], s['y'][envs, 0], s['z'][envs, 0] + 1.3], 1) + torch.randn(len(envs), 3, device=DEV, dtype=torch.float64) * 4
            G.add_smoke(envs, pos, tick[envs] - 100 * (L + 1))
        res = {}
        for name, fn in (('no smoke', None), ('smoke', G.smoke_blocks)):
            visible(s, mesh, tick, fn); torch.cuda.synchronize(); t0 = time.time()
            for _ in range(5):
                out = visible(s, mesh, tick, fn)
            torch.cuda.synchronize(); res[name] = (time.time() - t0) / 5
        cand, _, _ = candidates(s, tick)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(5):
            candidates(s, tick)
        torch.cuda.synchronize(); tp = (time.time() - t0) / 5
        print('E=%d: no smoke %.1f ms (prefilter %.1f ms, %d LOS segments), with smoke %.1f ms; %.0f K env-decisions/s (smoke)' %
              (E, res['no smoke'] * 1e3, tp * 1e3, int(cand.sum()), res['smoke'] * 1e3, E / res['smoke'] / 1e3), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='dust2-cs2-v54')
    ap.add_argument('--frames', type=int, default=600)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--no-smoke', action='store_true')
    a = ap.parse_args()
    if a.bench:
        bench(a.map, a.seed)
    else:
        run_parity(a.map, a.frames, a.seed, not a.no_smoke)


if __name__ == '__main__':
    main()
