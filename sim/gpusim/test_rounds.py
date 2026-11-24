"""Round machine / economy / bomb parity (GpuSim vs CPU Arena) on focused scenarios.  Needs a GPU and the CPU Arena data.

usage: python -m gpusim.test_rounds [--scenario economy|bomb|match|all] [--matches 4] [--seed 1]

  economy   buy phase only (120 decisions): random buys / refunds / drops / pick-ups of every catalogue item from every seat,
            cash settlement of the initial cash, all inventory fields and event counters compared after every decision.
  bomb      fights off, the carrier is moved into a plant zone and plants (180 ticks), then the bomb explodes (fuse 2400 ticks)
            or a defender is moved next to it and defuses (300 / 600 ticks): objective bookkeeping, round end, loss bonus,
            plant / defuse rewards, next round.
  match     freeze time shortened, duels every round (kills, rewards, drops, pick-ups), random plants: round machine incl.
            half-time swap at round 12, MR12 / first to 13, match end.
The CPU Arena is the reference (fpssim/arena.py) with the counter based random numbers injected (gpusim.test_sim.RefArena).
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from gpusim.test_sim import run_duels

SCENARIOS = dict(
    economy=dict(rounds=1, steps=125, kw=dict(fights=False, plant_prob=0.0, buy_activity=0.5)),
    bomb=dict(rounds=3, steps=4000, kw=dict(fights=False, plant_prob=1.0, defuse_prob=0.5, fast_buy=True)),
    match=dict(rounds=24, steps=12000, kw=dict(fast_buy=True, plant_prob=0.5)),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', default='all'); ap.add_argument('--matches', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--map', default='dust2-cs2-v54')
    ap.add_argument('-v', action='store_true')
    a = ap.parse_args()
    names = list(SCENARIOS) if a.scenario == 'all' else [a.scenario]
    ok = True
    for n in names:
        sc = SCENARIOS[n]
        print(f'=== scenario {n}', flush=True)
        pair = run_duels(a.map, a.matches, sc['rounds'], a.seed, False, sc['steps'], a.v, **sc['kw'])
        bad = [e for e in range(a.matches) if pair.first_bad[e] is not None]
        print(f'scenario {n}: {a.matches - len(bad)}/{a.matches} matches identical in every compared field'
              f' ({len(bad)} diverged: {bad})', flush=True)
        ok &= not bad
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
