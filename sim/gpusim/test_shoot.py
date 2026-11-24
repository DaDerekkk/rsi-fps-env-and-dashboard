"""Parity: gpusim.shoot vs fpssim (combat.ray_hit_player + penetration.trace_bullet) on Dust2 with real player poses from frames."""
import sys, os, json, math, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch
from fpssim.trimesh import TriMesh
from fpssim import penetration as PEN, combat as C
from fpssim.common import WEAPONS
from gpusim.ray import GpuMesh
from gpusim.shoot import Shooter
from gpusim import tables as TB
R = './fps_data'; dev = 'cuda'
st = json.load(open(f'{R}/data/static/static_dust2-cs2-v54.json'))['map']; col = st['collision'][0]
M = TriMesh(f'{R}/ref/assets/map_positions.f32', col['basis'], col['offset'], f'{R}/ref/assets/map_penetration-materials.u8')
G = GpuMesh(M.tri); S = Shooter(dev, G, M.tri, M.mat)
d = np.load(f'{R}/data/rec/09dd3817-9339-4287-8ac0-0e51facfe9c2/frames.npz'); P = d['P']; cols = list(d['cols']); c = {k: i for i, k in enumerate(cols)}
rng = np.random.default_rng(0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
frames = rng.integers(0, len(P), N * 3)
rows = []
for f in frames:
    al = np.where(P[f, :, c['alive']] > .5)[0]
    if len(al) < 2: continue
    sh = rng.choice(al); en = [q for q in al if q // 5 != sh // 5]
    if not en: continue
    tg = rng.choice(en)
    eye = P[f, sh, [c['x'], c['y'], c['z']]].astype(np.float64); eye[2] += 0.95 if P[f, sh, c['crouch']] > .5 else 1.62
    tp = P[f, tg, [c['x'], c['y'], c['z']]].astype(np.float64) + [0, 0, rng.uniform(0.2, 1.7)]
    dv = tp - eye; dv /= np.linalg.norm(dv); dv += rng.normal(0, 0.01, 3); dv /= np.linalg.norm(dv)
    wid = rng.choice(['ak47', 'm4a1', 'awp', 'usp', 'deagle', 'p250', 'mp9'])
    rows.append((f, sh, eye, dv, wid))
    if len(rows) >= N: break
E_ = len(rows)
ppos = torch.zeros(E_, 10, 3, dtype=torch.float64); pyaw = torch.zeros(E_, 10, dtype=torch.float64)
pcr = torch.zeros(E_, 10, dtype=torch.bool); pal = torch.zeros(E_, 10, dtype=torch.bool)
for k, (f, sh, eye, dv, wid) in enumerate(rows):
    ppos[k] = torch.tensor(P[f][:, [c['x'], c['y'], c['z']]].astype(np.float64)); pyaw[k] = torch.tensor(P[f, :, c['yaw']].astype(np.float64))
    pcr[k] = torch.tensor(P[f, :, c['crouch']] > .5); pal[k] = torch.tensor(P[f, :, c['alive']] > .5)
T = lambda x, dt=torch.float64: torch.tensor(np.array(x), dtype=dt, device=dev)
ws = [WEAPONS[r[4]] for r in rows]
out = S.trace(T([r[2] for r in rows]), T([r[3] for r in rows]), torch.arange(E_, device=dev), T([r[1] for r in rows], torch.long),
              T([w.get('range') or 200 for w in ws]), T([PEN.POWER.get(w['id'], 0) for w in ws]), T([w['damage'] for w in ws]),
              ppos.to(dev), pyaw.to(dev), pcr.to(dev), pal.to(dev))
o = {k: v.cpu().numpy() for k, v in out.items()}
bad = {}; nh = npn = 0
PART = {'head': 0, 'upper': 1, 'lower': 2}
for k, (f, sh, eye, dv, wid) in enumerate(rows):
    w = WEAPONS[wid]; rng_ = w.get('range') or 200
    ph = []
    for q in range(10):
        if q == sh or q // 5 == sh // 5 or P[f, q, c['alive']] < .5: continue
        r = C.ray_hit_player(eye, dv, tuple(P[f, q, [c['x'], c['y'], c['z']]].astype(np.float64)), float(P[f, q, c['yaw']]), P[f, q, c['crouch']] > .5, rng_)
        if r: ph.append((q, r[0], r[1]))
    tr = PEN.trace_bullet(eye, dv, w, ph, PEN.surfaces(M, eye, dv, rng_))
    nh += len(tr['hits']); npn += len(tr['segments'])
    def b(x): bad[x] = bad.get(x, 0) + 1
    if abs(tr['end_distance'] - o['end'][k]) > 1e-6: b('end')
    if len(tr['hits']) != o['nhit'][k]: b('nhit'); continue
    for j, h in enumerate(tr['hits']):
        if h[0] != o['seat'][k, j] or PART[h[2]] != o['part'][k, j]: b('hit_id')
        if abs(h[1] - o['dist'][k, j]) > 1e-9 or abs(h[3] - o['scale'][k, j]) > 1e-9: b('hit_dist_scale')
    if len(tr['segments']) != o['npen'][k]: b('npen')
print('shots', E_, 'player hits', nh, 'wall segments', npn, 'mismatches', bad if bad else 'NONE')
big = 200000
idx = torch.randint(0, E_, (big,), device=dev)
args = [T([r[2] for r in rows])[idx], T([r[3] for r in rows])[idx], idx, T([r[1] for r in rows], torch.long)[idx],
        T([w.get('range') or 200 for w in ws])[idx], T([PEN.POWER.get(w['id'], 0) for w in ws])[idx], T([w['damage'] for w in ws])[idx],
        ppos.to(dev), pyaw.to(dev), pcr.to(dev), pal.to(dev)]
S.trace(*args); torch.cuda.synchronize(); t0 = time.time(); S.trace(*args); torch.cuda.synchronize()
print('%d bullets in %.1f ms (%.2f M bullets/s)' % (big, (time.time() - t0) * 1e3, big / (time.time() - t0) / 1e6))
if len(sys.argv) > 2:
    for k, (f, sh, eye, dv, wid) in enumerate(rows):
        w = WEAPONS[wid]; rng_ = w.get('range') or 200
        cr = PEN.surfaces(M, eye, dv, rng_); tr = PEN.trace_bullet(eye, dv, w, [], cr)
        if len(tr['segments']) != o['npen'][k] and o['nhit'][k] == 0:
            t, ids = M.intersect(eye, eye + dv * rng_)
            print(wid, 'cpu end %.4f gpu %.4f' % (tr['end_distance'], o['end'][k]), 'cpu segs', [(round(s['entry'], 4), round(s['exit'], 4), s['material']) for s in tr['segments']],
                  'gpu npen', o['npen'][k], [(round(o['seg_entry'][k, j], 4), round(o['seg_exit'][k, j], 4), int(o['seg_mat'][k, j])) for j in range(o['npen'][k])],
                  'n raw hits', len(t), 'cpu crossings', [(round(x['distance'], 4), x['material']) for x in cr[:6]])
