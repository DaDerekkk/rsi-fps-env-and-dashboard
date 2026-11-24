"""Throughput of GpuSim (decision steps/s, per-stage cost, matches/s estimate).

usage: python -m gpusim.test_sim_bench [--E 4096] [--steps 60] [--warm 10] [--map dust2-cs2-v54] [--profile] [--policy random|idle]
                                       [--observe] [--no-graph]

--observe: the training loop.  The sim is built with audio=True and every decision is `sim.step(actions, observe=True)`
(3 ticks + visibility + audio flush + build_obs -> float32 [E,10,OBS_DIM]); also prints the split step / visibility() / observe()
and the peak memory of one observe().

Policy `random`: every step each player gets random movement / fire / reload / weapon-switch actions (a cheap on-GPU generator, a
fraction of the time firing), plus random buy attempts during the buy phase.  The state is whatever the sim reaches (rounds time out
because nobody plants), so the numbers are representative for 'busy' matches only in the weapon / damage / bullet stages.
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
from gpusim.sim import GpuSim
from gpusim import rounds as RD


def random_actions(sim, g):
    E, dev = sim.E, sim.dev
    r = lambda: torch.rand(E, 10, device=dev, generator=g)
    a = dict(move_forward=(r() * 2 - 1), move_right=(r() * 2 - 1), yaw_rate=(r() * 2 - 1) * 0.5, pitch_rate=(r() * 2 - 1) * 0.1,
             jump=r() < 0.02, crouch=r() < 0.1, fire=r() < 0.3, reload=r() < 0.03,
             weapon_slot=torch.where(r() < 0.05, (r() * 3).long(), torch.ones(E, 10, dtype=torch.long, device=dev)))
    a['buy'] = torch.where(r() < 0.1, (r() * RD.NB).long(), torch.full((E, 10), -1, dtype=torch.long, device=dev))
    return a


def stage_times(sim, g, n=10):
    """wall time (ms, synchronized) of every stage of GpuSim.tick, averaged over n ticks."""
    import collections
    from gpusim import weapons as GW
    R, s, T = sim.R, sim.s, sim.T
    acc = collections.OrderedDict()

    def tm(name, fn):
        torch.cuda.synchronize(); t = time.time(); r = fn(); torch.cuda.synchronize()
        acc[name] = acc.get(name, 0.0) + (time.time() - t) * 1e3
        return r
    sim.set_actions(random_actions(sim, g)); R.one_shots()
    for _ in range(n):
        sim.gtick += 1; gt = sim.gtick
        tm('tick_begin', lambda: R.tick_begin(gt))
        tm('move.step', lambda: sim.mover.step(sim._mstate, sim._mact, s['mv_maxspeed'], s['mv_locked'], gt))
        tm('move.resolve', lambda: sim.mover.resolve_players(sim._mstate, s['mv_locked']))
        def wt():
            R.snapshot_weapons()
            o = GW.tick(s, T, gt, s['alive_run'], s['w_can_act'], s['act_fire'], sim.no_fire2, s['act_reload'], s['w_want'], s['w_zoomreq'],
                        s['crouch'], ~s['grounded'])
            R.stabilize(); return o
        out = tm('weapons.tick', wt)
        tm('damage.run', lambda: sim.damage.run(out, gt))
        tm('obj_step', R.obj_step)
        if sim.grenades is not None:
            live = (s['phase'] == 1) & ~s['finished']
            tm('grenades.throw', lambda: sim.grenades.handle_throw(s, sim._gact, live))
            ab = s['alive'].clone()
            ev = tm('grenades.step', lambda: sim.grenades.step(s, None, live))
            tm('kill_pass', lambda: R.kill_pass(ab, ev['ev_killer']))
        tm('end_tick', R.end_tick)
        if sim.grenades is not None: tm('grenades.clear', lambda: sim.grenades.clear(s['round_started']))
    return {k: v / n for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--E', type=int, default=4096); ap.add_argument('--steps', type=int, default=60)
    ap.add_argument('--warm', type=int, default=10); ap.add_argument('--map', default='dust2-cs2-v54')
    ap.add_argument('--profile', action='store_true'); ap.add_argument('--no-graph', action='store_true'); ap.add_argument('--policy', default='random')
    ap.add_argument('--skip', type=int, default=0, help='steps to run first (e.g. 130 to leave the buy phase)')
    ap.add_argument('--observe', action='store_true', help='training loop: step(observe=True) = sim + visibility + observations')
    ap.add_argument('--no-stages', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    sim = GpuSim(a.map, a.E, 'cuda', seed=1, graph=not a.no_graph, audio=a.observe)
    sim.reset(1)
    g = torch.Generator(device='cuda'); g.manual_seed(1)
    print(f'init {time.time() - t0:.1f}s, E={a.E}', flush=True)
    for _ in range(a.skip + a.warm):
        sim.step(random_actions(sim, g) if a.policy == 'random' else {})
    torch.cuda.synchronize()
    acts = [random_actions(sim, g) for _ in range(a.steps)] if a.policy == 'random' else [{} for _ in range(a.steps)]
    torch.cuda.synchronize(); t = time.time()
    for k in range(a.steps):
        sim.step(acts[k])
    torch.cuda.synchronize(); dt = (time.time() - t) / a.steps
    print(f'E={a.E}: {dt * 1e3:.1f} ms per decision step ({dt / 3 * 1e3:.1f} ms per tick), {a.E / dt:.0f} env-steps/s')
    if a.observe:
        for k in range(3): sim.step(acts[k], observe=True)
        torch.cuda.synchronize(); t = time.time()
        for k in range(a.steps):
            obs = sim.step(acts[k], observe=True)
        torch.cuda.synchronize(); dto = (time.time() - t) / a.steps
        tv = tob = 0.0
        for k in range(a.steps):
            torch.cuda.synchronize(); t = time.time(); sim.visibility(); torch.cuda.synchronize(); tv += time.time() - t
            t = time.time(); sim.observe(); torch.cuda.synchronize(); tob += time.time() - t
        tv /= a.steps; tob /= a.steps
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
        o = sim.observe(); torch.cuda.synchronize(); pk = torch.cuda.max_memory_allocated() - m0
        print(f'TRAIN E={a.E} {a.map}: {dto * 1e3:.1f} ms per decision step with observations ({a.E / dto:.0f} env-steps/s); '
              f'sim only {dt * 1e3:.1f} ms; visibility() {tv * 1e3:.1f} ms; observe() (incl. visibility) {tob * 1e3:.1f} ms; '
              f'obs {tuple(o.shape)} {o.numel() * 4 / 2 ** 20:.0f} MB, peak extra memory of observe() {pk / 2 ** 30:.2f} GB', flush=True)
        del o
    ev = sim.s['ev_count'].sum(0).cpu()
    print('events per match so far:', {n: round(float(ev[i]) / a.E, 2) for i, n in enumerate(RD.RC.EVENTS) if ev[i] > 0})
    for steps_per_match in (20000, 40000):
        print(f'  matches/s if a match takes {steps_per_match} decision steps: {a.E / dt / steps_per_match:.3f}')
    if a.no_stages: return
    st = stage_times(sim, g)
    print('per-tick stage wall time (ms, synchronized):', {k: round(v, 2) for k, v in st.items()}, 'sum %.1f' % sum(st.values()))
    if a.profile:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for k in range(3): sim.step(acts[k])
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=25, max_name_column_width=60))


if __name__ == '__main__':
    main()
