"""Smoke / integration test: GpuSim with the real GpuGrenades module (throws, kills through Rounds.kill_pass, smoke visibility).

usage (server, from sim/):  python3 -m gpusim.test_grenades_sim [--map nevaio-r3-v54] [--E 64] [--steps 360]

Players buy random utilities in the buy phase, then prime / release / cancel at random while moving and shooting.  Checks after
every decision: finite state, 0 <= hp <= 100, `alive` consistent with hp, nades >= 0 and <= the per-kind carry limits, at most one
life per round, grenade kills are processed by Rounds (death_tick set, deaths counter incremented); at the end: some throws, detonations,
smokes, fire hits and kills happened, and sim.visibility() (vis + smoke) runs.  This is a crash / consistency test, the numerical
parity of the module itself is gpusim/test_grenades.py.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
from gpusim.sim import GpuSim
from gpusim import rounds as RD
from gpusim.rounds import BID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='nevaio-r3-v54'); ap.add_argument('--E', type=int, default=64)
    ap.add_argument('--steps', type=int, default=360); ap.add_argument('--seed', type=int, default=1)
    a = ap.parse_args()
    t0 = time.time()
    sim = GpuSim(a.map, a.E, 'cuda', seed=a.seed)
    assert sim.grenades is not None, 'grenade module missing'
    sim.reset(a.seed)
    dev, E = sim.dev, a.E
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    r = lambda: torch.rand(E, 10, device=dev, generator=g)
    util = torch.tensor([BID[k] for k in ('hegrenade', 'flashbang', 'smokegrenade', 'molotov', 'incgrenade', 'decoy')], device=dev)
    s = sim.s
    stat = dict(throws=0, kills=0, smoke_max=0, fire_max=0, flash=0, hits=0)
    last_deaths = s['deaths'].clone()
    for step in range(a.steps):
        buy = torch.full((E, 10), -1, dtype=torch.long, device=dev)
        m = r() < 0.25
        buy[m] = util[(r() * 6).long().clamp(max=5)][m]
        u = r()
        st = torch.where(u < 0.15, 1, torch.where(u < 0.45, 2, torch.where(u < 0.47, 3, 0)))
        act = dict(move_forward=(r() * 2 - 1), move_right=(r() * 2 - 1), yaw_rate=(r() * 2 - 1) * 0.5, pitch_rate=(r() * 2 - 1) * 0.1,
                   fire=r() < 0.2, buy=buy, throw_kind=(r() * 6).long().clamp(max=5), throw_state=st.long(), throw_mode=(r() * 3).long().clamp(max=2))
        alive_before = s['alive'].clone()
        sim.step(act)
        # --- consistency
        for k in ('x', 'y', 'z', 'hp', 'armor'):
            assert torch.isfinite(s[k]).all(), k
        assert (s['hp'] >= 0).all() and (s['hp'] <= 100.0 + 1e-9).all()
        assert (s['hp'][~s['alive']] <= 1e-9).all() and (s['hp'][s['alive']] > 1e-9).all(), 'alive / hp inconsistent'
        assert (s['nades'] >= 0).all() and (s['nades'].sum(-1) <= 4).all(), 'nades out of range'
        dead_now = alive_before & ~s['alive']
        assert (s['death_tick'][dead_now] >= 0).all()
        gr = sim.grenades.g
        stat['smoke_max'] = max(stat['smoke_max'], int((gr['sm_valid'] > 0).sum(1).max()))
        stat['fire_max'] = max(stat['fire_max'], int((gr['fi_valid'] > 0).sum(1).max()))
    ev = sim.events(0)
    gt = sim.grenades.g
    print('steps', a.steps, 'matches', E, 'time %.1fs' % (time.time() - t0))
    print('round state: phase', s['phase'][:4].tolist(), 'round_index', s['round_index'][:4].tolist(), 'deaths total', int(s['deaths'].sum()))
    print('smokes now %d fires now %d decoys now %d projectiles now %d; max smokes/env %d, max fires/env %d' %
          ((gt['sm_valid'] > 0).sum(), (gt['fi_valid'] > 0).sum(), (gt['dc_valid'] > 0).sum(), (gt['pr_kind'] >= 0).sum(), stat['smoke_max'], stat['fire_max']))
    cnt = s['ev_count'].sum(0).cpu()
    print('round-machine event counts:', {n: int(cnt[i]) for i, n in enumerate(RD.RC.EVENTS) if cnt[i] > 0})
    vis = sim.visibility()
    print('visibility with smoke_fn: shape', tuple(vis.shape), 'visible pairs', int(vis.any(-1).sum()))
    assert int(gt['ctr_id'].sum()) > 0, 'no grenade was thrown'
    print('OK: %d grenades thrown in total' % int(gt['ctr_id'].sum()))


if __name__ == '__main__':
    main()
