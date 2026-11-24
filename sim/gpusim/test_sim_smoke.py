"""Quick sanity run of GpuSim (no CPU reference): buy phase -> live, audio hook, visibility, invariants.
usage: python -m gpusim.test_sim_smoke [--map dust2-cs2-v54] [--E 64] [--steps 200] [--graph]"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
from gpusim.sim import GpuSim

ap = argparse.ArgumentParser()
ap.add_argument('--map', default='dust2-cs2-v54'); ap.add_argument('--E', type=int, default=64)
ap.add_argument('--steps', type=int, default=200); ap.add_argument('--no-graph', action='store_true')
a = ap.parse_args()
sim = GpuSim(a.map, a.E, 'cuda', seed=1, audio=True, graph=not a.no_graph)
sim.reset(1)
E = a.E
from gpusim.rounds import BID
for i in range(a.steps):
    act = dict(move_forward=torch.ones(E, 10), fire=torch.full((E, 10), i % 2 == 0, dtype=torch.bool))     # semi-auto guns need a release between shots
    if i in (0, 1): act['buy'] = torch.full((E, 10), BID['molotov' if i == 0 else 'incgrenade'], dtype=torch.long)
    if i >= 125:
        ph = (i - 125) % 4
        act['throw_kind'] = torch.full((E, 10), 3 if ph < 2 else 4, dtype=torch.long)
        act['throw_state'] = torch.full((E, 10), 1 if ph in (0, 2) else 2, dtype=torch.long)
    sim.step(act)
s = sim.s
assert not bool(sim.mover.errors()), 'mover error bits %d' % sim.mover.errors()
assert int(s['wi_overflow'].max()) == 0
assert bool((s['cash'] >= 0).all()) and bool((s['cash'] <= 16000).all())
assert bool((s['hp'][s['alive']] > 0).all())
print('diag: phase', s['phase'][0].item(), 'fire held', s['act_fire'][0].tolist(), 'shots', s['w_shots'][0, :, 1].tolist(), 'ammo', s['w_ammo'][0, :, 1].tolist(), 'can_act', s['w_can_act'][0].tolist(), 'alive_run', s['alive_run'][0].tolist(), 'switch', s['switch_until'][0].tolist(), 'active', s['active'][0].tolist(), 'tick', s['tick'][0].item(), 'rs', s['round_start_tick'][0].item())
print('fires', int((sim.grenades.g['fi_valid'] > 0).sum()) if sim.grenades is not None else 'no grenades')
em = sim.audio_emitters()
vis = sim.visibility()
obs = sim.observe()
assert torch.isfinite(obs).all()
print('ok: tick', sim.gtick, 'phase', s['phase'][0].item(), 'round', s['round_index'][0].item(), 'emitters max', int(em['n'].max()),
      'vis', tuple(vis.shape), 'obs', tuple(obs.shape), 'events(0)', {k: v for k, v in sim.events(0).items() if v})
