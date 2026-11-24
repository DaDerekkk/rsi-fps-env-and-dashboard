"""Constant tables for gpusim, generated from the validated CPU modules (fpssim.common/recoil/accuracy/weapons/penetration).

W = number of catalogue weapons (knife included). All tables are torch tensors on the requested device.
"""
import math
import numpy as np
import torch
from fpssim.common import WEAPONS, CATALOG, UTIL
from fpssim import recoil as R, accuracy as A, penetration as PEN
from fpssim import weapons as FW

WEAPON_IDS = [w['id'] for w in CATALOG['weapons']]
WID = {k: i for i, k in enumerate(WEAPON_IDS)}
UTIL_IDS = ['hegrenade', 'flashbang', 'smokegrenade', 'molotov', 'incgrenade', 'decoy']
KNIFE = WID['knife']


def ticks(sec):
    return int(math.ceil(sec * 60.0 - 1e-6))


def build(device):
    W = len(WEAPON_IDS)
    f = lambda: np.zeros(W)
    T = dict(fire_ticks=f(), magazine=f(), reserve=f(), reload_ticks=f(), automatic=f(), pellets=f(), damage=f(),
             range_mod=f(), armor_ratio=f(), head_mult=f(), range=f(), max_speed=f(), scoped_max_speed=f(), n_zoom=f(),
             unzoom_after_shot=f(), kill_reward=f(), price=f(), slot=f(), shell=f(), commit_full=f(), commit_empty=f(),
             pen_power=f(), team_t=f(), team_ct=f(), inacc_fire=f(), rec_stand=f(), rec_crouch=f(), rec_stand_final=f(),
             rec_crouch_final=f(), trans_start=f(), trans_end=f(), recoil_mode=f(), acc_index_always1=f())
    acc = np.zeros((W, 5, 2))                      # spread, stand, crouch, move, jump  x  mode
    rtab = np.zeros((W, 64, 2))                    # (angle, magnitude) for the weapon's recoil mode
    for i, wid in enumerate(WEAPON_IDS):
        m = WEAPONS[wid]
        T['fire_ticks'][i] = ticks(m['fireInterval'])
        T['magazine'][i] = m.get('magazine', 0); T['reserve'][i] = m.get('reserve', 0)
        T['reload_ticks'][i] = ticks(m.get('reloadTime', 0) or 0)
        T['automatic'][i] = bool(m.get('automatic')); T['pellets'][i] = m.get('pellets', 1) or 1
        T['damage'][i] = m['damage']; T['range_mod'][i] = m.get('rangeModifier', 1.0); T['armor_ratio'][i] = m.get('armorRatio', 1.0)
        T['head_mult'][i] = m.get('headMultiplier', 1.0); T['range'][i] = m.get('range') or 200.0
        T['max_speed'][i] = m.get('maxSpeed', 6.0); T['scoped_max_speed'][i] = m.get('scopedMaxSpeed', m.get('maxSpeed', 6.0))
        T['n_zoom'][i] = len(m.get('zoomFovs') or [90]) - 1 if m.get('zoomStyle') else 0
        T['unzoom_after_shot'][i] = bool(m.get('unzoomsAfterShot')); T['kill_reward'][i] = m.get('killReward', 300)
        T['price'][i] = m.get('price', 0); T['slot'][i] = {1: 0, 2: 1, 3: 2}[m['slot']]
        T['shell'][i] = m.get('reloadStyle') == 'shell'
        if T['magazine'][i] > 0 and not T['shell'][i]:
            T['commit_full'][i] = FW.reload_commit_ticks(wid, False); T['commit_empty'][i] = FW.reload_commit_ticks(wid, True)
        T['pen_power'][i] = PEN.POWER.get(wid, 0)
        T['team_t'][i] = 'T' in m['teams']; T['team_ct'][i] = 'CT' in m['teams']
        if wid in R.WEAPON_PARAMS:
            p = R.WEAPON_PARAMS[wid]
            T['inacc_fire'][i] = R.inaccuracy_fire(wid)
            for k, key in (('rec_stand', 'recoveryStand'), ('rec_crouch', 'recoveryCrouch'), ('rec_stand_final', 'recoveryStandFinal'),
                           ('rec_crouch_final', 'recoveryCrouchFinal'), ('trans_start', 'transStart'), ('trans_end', 'transEnd'),
                           ('recoil_mode', 'recoilMode')):
                T[k][i] = p[key]
            rtab[i] = np.array(R.table(wid)[p['recoilMode']])
        else:
            T['rec_stand'][i] = T['rec_crouch'][i] = T['rec_stand_final'][i] = T['rec_crouch_final'][i] = 0.3
        src = A.ACC.get(wid)
        if src:
            for j, k in enumerate(('spread', 'stand', 'crouch', 'move', 'jump')):
                acc[i, j] = src[k]
        T['acc_index_always1'][i] = wid in ('m4a1', 'usp')
    out = {k: torch.tensor(v, dtype=torch.float64 if v.dtype.kind == 'f' else torch.int64, device=device) for k, v in T.items()}
    out['acc'] = torch.tensor(acc, dtype=torch.float64, device=device)
    out['recoil_table'] = torch.tensor(rtab, dtype=torch.float64, device=device)
    for k in ('fire_ticks', 'magazine', 'reserve', 'reload_ticks', 'pellets', 'n_zoom', 'kill_reward', 'price', 'slot',
              'commit_full', 'commit_empty', 'trans_start', 'trans_end', 'recoil_mode'):
        out[k] = out[k].round().long()
    for k in ('automatic', 'unzoom_after_shot', 'shell', 'team_t', 'team_ct', 'acc_index_always1'):
        out[k] = out[k] > 0.5
    out['knife_id'] = KNIFE
    out['has_recoil'] = torch.tensor([w in R.WEAPON_PARAMS for w in WEAPON_IDS], device=device)
    return out
