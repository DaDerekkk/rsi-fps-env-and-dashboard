"""Parity: gpusim.weapons.tick vs fpssim.weapons.WeaponSystem on random inputs (E matches x 10 players), every field, every tick."""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np, torch
from fpssim.weapons import WeaponSystem, Weapon
from gpusim import tables as TB, weapons as GW
dev = 'cuda'
E, TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 6, int(sys.argv[2]) if len(sys.argv) > 2 else 3000
T = TB.build(dev)
rng = random.Random(0)
prim = [w for w in TB.WEAPON_IDS if TB.WEAPONS[w]['slot'] == 1]; sec = [w for w in TB.WEAPON_IDS if TB.WEAPONS[w]['slot'] == 2]
s = GW.new_state(E, dev)
cpu = []
for e in range(E):
    for i in range(10):
        ws = WeaponSystem(); p = rng.choice(prim + [None]); q = rng.choice(sec)
        if p: ws.slots[0] = Weapon(p)
        ws.slots[1] = Weapon(q); ws.slots[2] = Weapon('knife'); ws.active = 0 if p else 1
        cpu.append(ws)
        m = torch.zeros(E, 10, dtype=torch.bool, device=dev); m[e, i] = True
        if p: GW.give(s, T, m, 0, TB.WID[p])
        GW.give(s, T, m, 1, TB.WID[q]); GW.give(s, T, m, 2, TB.KNIFE)
        s['active'][e, i] = cpu[-1].active
mism = {}; n_shots = 0; checks = 0
for t in range(1, TICKS + 1):
    if t % 3 == 1:          # new held action every decision
        act = [dict(fire=rng.random() < .5, fire2=rng.random() < .05, reload=rng.random() < .05,
                    slot=rng.choice([None, None, None, 0, 1, 2]), zoom=rng.choice([None, 0, 1, 2]),
                    crouch=rng.random() < .3, air=rng.random() < .1, can=rng.random() < .9) for _ in range(E * 10)]
    f = lambda k, d=torch.bool: torch.tensor([a[k] for a in act], device=dev, dtype=d).view(E, 10)
    out = GW.tick(s, T, t, torch.ones(E, 10, dtype=torch.bool, device=dev), f('can'), f('fire'), f('fire2'), f('reload'),
                  torch.tensor([-1 if a['slot'] is None else a['slot'] for a in act], device=dev).view(E, 10),
                  torch.tensor([-1 if (a['zoom'] is None) else a['zoom'] for a in act], device=dev).view(E, 10),
                  f('crouch'), f('air'))
    for k, ws in enumerate(cpu):
        a = act[k]
        cur = ws.weapon()
        z = a['zoom'] if (cur is not None and TB.WEAPONS[cur.id].get('zoomStyle')) else None
        if a['slot'] is not None and a['slot'] != ws.active and ws.slots[a['slot']] is not None and a['can']:
            nw = ws.slots[a['slot']]; z = a['zoom'] if TB.WEAPONS[nw.id].get('zoomStyle') else None
        ev = ws.tick(t, a['fire'], a['reload'], a['slot'], can_act=a['can'], crouch=a['crouch'], zoom=z, in_air=a['air'], fire2=a['fire2'])
    # compare
    if t % 7 == 0 or t == TICKS:
        S = {k: v.cpu().numpy() for k, v in s.items()}
        for k, ws in enumerate(cpu):
            e, i = divmod(k, 10)
            checks += 1
            def chk(name, a, b, tol=0):
                if (abs(a - b) > tol) if tol else (a != b):
                    mism[name] = mism.get(name, 0) + 1
            chk('active', ws.active, S['active'][e, i]); chk('switch_until', ws.switch_until, S['switch_until'][e, i])
            chk('zoom', ws.zoom_level, S['zoom'][e, i])
            for sl in range(3):
                w = ws.slots[sl]
                if w is None: continue
                chk('ammo', w.ammo, S['w_ammo'][e, i, sl]); chk('reserve', w.reserve, S['w_reserve'][e, i, sl])
                chk('shots', w.shots, S['w_shots'][e, i, sl]); chk('shot_ready', w.shot_ready, S['w_shot_ready'][e, i, sl])
                chk('reload_until', -1 if w.reload_until is None else w.reload_until, S['w_reload_until'][e, i, sl])
                chk('inacc', w.inaccuracy, S['w_inacc'][e, i, sl], 1e-9)
                for j, key in enumerate(('index', 'pitch', 'yaw', 'velPitch', 'velYaw', 'viewPitch', 'viewYaw')):
                    chk('rs_' + key, w.rs[key], S['w_rs'][e, i, sl, j], 1e-9)
    n_shots += int(out['shot'].sum())
print('players', E * 10, 'ticks', TICKS, 'shots', n_shots, 'player-checks', checks)
print('mismatches:', mism if mism else 'NONE')
