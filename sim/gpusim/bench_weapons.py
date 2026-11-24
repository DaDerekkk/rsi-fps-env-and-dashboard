import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
from gpusim import tables as TB, weapons as GW
dev = 'cuda'; T = TB.build(dev)
for E in (1024, 4096, 16384):
    s = GW.new_state(E, dev); m = torch.ones(E, 10, dtype=torch.bool, device=dev)
    GW.give(s, T, m, 0, TB.WID['ak47']); GW.give(s, T, m, 1, TB.WID['usp']); GW.give(s, T, m, 2, TB.KNIFE); s['active'][:] = 0
    b = lambda p: torch.rand(E, 10, device=dev) < p
    args = lambda: (m, b(.9), b(.5), b(.02), b(.02), torch.full((E, 10), -1, device=dev), torch.full((E, 10), -1, device=dev), b(.3), b(.1))
    for t in range(1, 20): GW.tick(s, T, t, *args())
    torch.cuda.synchronize(); t0 = time.time(); N = 100
    for t in range(20, 20 + N): GW.tick(s, T, t, *args())
    torch.cuda.synchronize(); dt = (time.time() - t0) / N
    print('E=%d: %.2f ms/tick  -> %.1f M player-ticks/s' % (E, dt * 1e3, E * 10 / dt / 1e6))
