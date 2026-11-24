"""Parity + throughput of the CUDA ray kernel vs fpssim.trimesh.TriMesh on Dust2 (random segments + visibility-like rays)."""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch
from fpssim.trimesh import TriMesh
from gpusim.ray import GpuMesh
R = './fps_data'
st = json.load(open(f'{R}/data/static/static_dust2-cs2-v54.json'))['map']; col = st['collision'][0]
M = TriMesh(f'{R}/ref/assets/map_positions.f32', col['basis'], col['offset'])
t0 = time.time(); G = GpuMesh(M.tri); torch.cuda.synchronize(); print('bvh nodes', G.n_nodes, 'build+upload %.1fs' % (time.time() - t0))
rng = np.random.default_rng(0)
n = 3000
a = rng.uniform(M.lo, M.hi, (n, 3)); b = a + rng.normal(0, 25, (n, 3))
tg, ig = G.first_hit(torch.tensor(a, device='cuda'), torch.tensor(b, device='cuda'))
tg = tg.cpu().numpy(); ig = ig.cpu().numpy()
agree = 0; terr = []
for k in range(n):
    t, ids = M.intersect(a[k], b[k])
    ch = len(t) > 0; gh = ig[k] >= 0
    if ch == gh:
        agree += 1
        if ch: terr.append(abs(t[0] - tg[k]) * np.linalg.norm(b[k] - a[k]))
print('hit/miss agreement %d/%d, hit distance |err| max %.2e m, p99 %.2e' % (agree, n, max(terr), np.percentile(terr, 99)))
for N in (10 ** 5, 10 ** 6):
    a = torch.tensor(rng.uniform(M.lo, M.hi, (N, 3)), device='cuda', dtype=torch.float32)
    b = a + torch.randn_like(a) * 25
    G.first_hit(a, b); torch.cuda.synchronize(); t0 = time.time()
    for _ in range(5): G.first_hit(a, b)
    torch.cuda.synchronize(); dt = (time.time() - t0) / 5
    print('%d rays: %.2f ms  (%.1f M rays/s)' % (N, dt * 1e3, N / dt / 1e6))
