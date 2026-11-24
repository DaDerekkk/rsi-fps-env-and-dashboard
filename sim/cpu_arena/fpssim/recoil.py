"""Recoil / aim punch / accuracy penalty, ported from cs2-web shared/recoil-tables.js (commit e0e7e858).

The platform steps this state every 60 Hz physics tick (cs2-web decays lazily at shot time); the per-tick variant is
validated against recordings in validate/recoil.py.  Angles are radians; pitch > 0 kicks up, yaw > 0 kicks left.
"""
import math

RECOIL_SCALE = 2.0
VIEW_RECOIL_TRACKING = 0.45
VIEW_PUNCH_EXTRA = 0.055
DECAY2_EXP = 8.0
DECAY2_LIN = 18 * math.pi / 180
VEL_DECAY = 4.5
VIEW_PUNCH_DECAY = 18.0
INDEX_DECAY = math.log(10) * 2.0
RECOIL_DECAY_THRESHOLD = 1.10
VARIANCE = 0.55
SUPPRESSION_SHOTS = 4
SUPPRESSION_FACTOR = 0.75
DEG2RAD = math.pi / 180
TABLE_SIZE = 64

IA, IM, IQ, IR, NTAB = 16807, 2147483647, 127773, 2836, 32
NDIV = 1 + (IM - 1) // NTAB
AM = 1.0 / IM
RNMX = 1.0 - 1.2e-7


class UniformRandomStream:
    """vstdlib CUniformRandomStream (Park-Miller + Bays-Durham shuffle), as in cs2-web."""

    def __init__(self, seed):
        self.idum = seed if seed < 0 else -seed
        self.iy = 0
        self.iv = [0] * NTAB

    def gen(self):
        if self.idum <= 0 or not self.iy:
            self.idum = 1 if -self.idum < 1 else -self.idum
            for j in range(NTAB + 7, -1, -1):
                k = self.idum // IQ
                self.idum = IA * (self.idum - k * IQ) - IR * k
                if self.idum < 0:
                    self.idum += IM
                if j < NTAB:
                    self.iv[j] = self.idum
            self.iy = self.iv[0]
        k = self.idum // IQ
        self.idum = IA * (self.idum - k * IQ) - IR * k
        if self.idum < 0:
            self.idum += IM
        j = (self.iy // NDIV) & (NTAB - 1)
        self.iy = self.iv[j]
        self.iv[j] = self.idum
        return self.iy

    def random_float(self, lo, hi):
        fl = AM * self.gen()
        if fl > RNMX:
            fl = RNMX
        return fl * (hi - lo) + lo


def _p(seed, full, angle, angle_var, mag, mag_var, rs, rc, rsf, rcf, ts, te, inacc, mode=0):
    return dict(seed=seed, fullAuto=full, angle=angle, angleVar=angle_var, magnitude=mag, magVar=mag_var,
                recoveryStand=rs, recoveryCrouch=rc, recoveryStandFinal=rsf, recoveryCrouchFinal=rcf,
                transStart=ts, transEnd=te, inaccuracyFire=inacc, recoilMode=mode)


WEAPON_PARAMS = {
    'ak47': _p(223, True, [0, 0], [70, 70], [30, 30], [0, 0], .368, .305257, .506, .419728, 2, 5, [.0078, .0078]),
    'm4a4': _p(38965, True, [0, 0], [70, 70], [23, 23], [0, 0], .338941, .2421, .466044, .332888, 2, 5, [.007, .00634]),
    'm4a1': _p(38965, True, [0, 0], [65, 65], [25, 21], [3, 0], .338941, .2421, .466044, .332888, 2, 5, [.012, .007], 1),
    'galilar': _p(51191, True, [0, 0], [70, 70], [21, 21], [1, 1], .3, .15, .5, .47, 2, 5, [.007, .00585]),
    'sg553': _p(43500, True, [0, 0], [60, 60], [28, 19], [2, 2], .452886, .379204, .452886, .379204, 2, 5, [.00795, .0092]),
    'mp9': _p(50729, True, [0, 0], [70, 70], [21, 21], [1, 1], .25789, .184207, .25789, .184207, 2, 5, [.0037, .0037]),
    'mac10': _p(34079, True, [0, 0], [70, 70], [18, 18], [1, 1], .399729, .285521, .399729, .285521, 2, 5, [.00476, .00476]),
    'mp7': _p(61649, True, [0, 0], [70, 70], [16, 16], [1, 1], .437491, .312494, .437491, .312494, 2, 5, [.00218, .00218]),
    'bizon': _p(36387, True, [0, 0], [70, 70], [18, 18], [1, 1], .331572, .236837, .331572, .236837, 2, 5, [.00288, .00288]),
    'xm1014': _p(24862, True, [0, 0], [20, 20], [80, 80], [20, 20], .506569, .361835, .506569, .361835, 2, 5, [.00883, .00883]),
    'scar20': _p(19364, True, [0, 0], [30, 30], [31, 31], [4, 4], .544331, .388808, .544331, .388808, 2, 5, [.01861, .01861]),
    'awp': _p(4100, False, [0, 0], [20, 20], [78, 25], [15, 2], .34539, .24671, .34539, .24671, 2, 5, [.05385, .05385]),
    'ssg08': _p(1278, False, [0, 0], [20, 20], [33, 25], [15, 2], .142096, .055783, .142096, .055783, 2, 5, [.02292, .02292]),
    'deagle': _p(12345, True, [0, 3], [40, 50], [20, 45], [0, 6], .9, .7, .8112, .449927, 3, 10, [.05, .055]),
    'pistol': _p(4484, False, [0, 0], [20, 20], [18, 30], [0, 5], .2, .2, .33, .33, 0, 5, [.056, .045]),
    'usp': _p(5426, False, [0, 0], [0, 0], [29, 23], [0, 0], .349532, .291277, .349532, .291277, 3, 10, [.071, .052], 1),
    'p250': _p(9788, False, [0, 0], [10, 10], [26, 26], [3, 3], .345388, .287823, .345388, .287823, 3, 10, [.05245, .05245]),
    'fiveseven': _p(33244, False, [0, 0], [5, 5], [25, 25], [4, 4], .2, .2, .5, .5, 0, 5, [.025, .03245]),
    'elite': _p(24563, False, [0, 0], [20, 20], [27, 27], [4, 4], .524989, .437491, .524989, .437491, 3, 10, [.01116, .01196]),
    'tec9': _p(789, False, [0, 0], [60, 60], [23, 23], [3, 3], .391, .315, .391, .315, 3, 10, [.045, .03688]),
    'nova': _p(7763, False, [0, 0], [20, 20], [143, 143], [22, 22], .460517, .328941, .460517, .328941, 2, 5, [.00972, .00972]),
    'mag7': _p(12518, False, [0, 0], [20, 20], [165, 165], [25, 25], .399729, .285521, .399729, .285521, 2, 5, [.01119, .01119]),
    'sawedoff': _p(1089, False, [0, 0], [20, 20], [143, 143], [22, 22], .460517, .328941, .460517, .328941, 2, 5, [.00972, .00972]),
}


def params(wid):
    return WEAPON_PARAMS.get(wid, WEAPON_PARAMS['ak47'])


def _gen_table(p, mode):
    rng = UniformRandomStream(p['seed'])
    angle = magnitude = 0.0
    rows = []
    for j in range(TABLE_SIZE):
        a_new = p['angle'][mode] + rng.random_float(-p['angleVar'][mode], p['angleVar'][mode])
        m_new = p['magnitude'][mode] + rng.random_float(-p['magVar'][mode], p['magVar'][mode])
        if p['fullAuto'] and j > 0:
            angle = angle + (a_new - angle) * VARIANCE
            magnitude = magnitude + (m_new - magnitude) * VARIANCE
        else:
            angle, magnitude = a_new, m_new
        if p['fullAuto'] and j < SUPPRESSION_SHOTS:
            magnitude *= SUPPRESSION_FACTOR + (1.0 - SUPPRESSION_FACTOR) * (j / SUPPRESSION_SHOTS)
        rows.append((angle * DEG2RAD, magnitude * DEG2RAD))
    return rows


_TABLES = {}


def table(wid):
    if wid not in _TABLES:
        p = params(wid)
        _TABLES[wid] = (_gen_table(p, 0), _gen_table(p, 1))
    return _TABLES[wid]


def new_state():
    return dict(index=0.0, pitch=0.0, yaw=0.0, velPitch=0.0, velYaw=0.0, viewPitch=0.0, viewYaw=0.0)


def kick(s, wid):
    """CCSPlayer::KickBack: velocity kick from the table entry at floor(index), then index += 1."""
    if wid == 'knife':
        return s
    p = params(wid)
    ang, mag = table(wid)[p['recoilMode']][int(math.floor(s['index'])) % TABLE_SIZE]
    c, sn = math.cos(ang), math.sin(ang)
    s['velPitch'] += c * mag
    s['velYaw'] -= sn * mag
    vm = mag * VIEW_PUNCH_EXTRA
    s['viewPitch'] += c * vm
    s['viewYaw'] -= sn * vm
    s['index'] += 1
    return s


def decay_step(s, h):
    """One hybrid aim-punch decay step of length h seconds (cs2-web hybridDecayStep) + view punch decay."""
    f = math.exp(-DECAY2_EXP * h)
    s['pitch'] *= f
    s['yaw'] *= f
    mag = math.hypot(s['pitch'], s['yaw'])
    lin = DECAY2_LIN * h
    if mag > lin:
        k = 1 - lin / mag
        s['pitch'] *= k
        s['yaw'] *= k
    else:
        s['pitch'] = s['yaw'] = 0.0
    s['pitch'] += s['velPitch'] * h * 0.5
    s['yaw'] += s['velYaw'] * h * 0.5
    vf = math.exp(-VEL_DECAY * h)
    s['velPitch'] *= vf
    s['velYaw'] *= vf
    s['pitch'] += s['velPitch'] * h * 0.5
    s['yaw'] += s['velYaw'] * h * 0.5
    g = math.exp(-VIEW_PUNCH_DECAY * h)
    s['viewPitch'] *= g
    s['viewYaw'] *= g
    return s


def tick(s, dt=1 / 60):
    """Platform per-tick decay (validated: 60 Hz tick = one 1/64 s hybrid step + the remainder, then snap)."""
    decay_step(s, 1 / 64)
    decay_step(s, dt - 1 / 64)
    return snap_small(s)


def snap_small(s):
    if abs(s['velPitch']) < 1e-6 and abs(s['velYaw']) < 1e-6:
        s['velPitch'] = s['velYaw'] = 0.0
    if abs(s['pitch']) < 1e-6 and abs(s['yaw']) < 1e-6:
        s['pitch'] = s['yaw'] = 0.0
    if abs(s['viewPitch']) < 1e-6 and abs(s['viewYaw']) < 1e-6:
        s['viewPitch'] = s['viewYaw'] = 0.0
    return s


def decay_index(s, idle_seconds):
    """cs2-web only. NOT used by the platform: recorded aim_punch.index == shots fired this round, never decays."""
    if s['index'] <= 0:
        s['index'] = 0.0
        return s
    s['index'] *= math.exp(-INDEX_DECAY * min(max(idle_seconds, 0.0), 2.0))
    if s['index'] < 1e-3:
        s['index'] = 0.0
    return s


def _remap(x, a, b, c, d):
    if a == b:
        return d
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return c + (d - c) * t


def recovery_time(wid, crouch=False, in_air=False, on_ladder=False, recoil_index=0.0):
    if wid not in WEAPON_PARAMS:
        return 0.3
    p = WEAPON_PARAMS[wid]
    if on_ladder:
        return p['recoveryStand']
    if in_air:
        return p['recoveryCrouch'] * 4
    n = math.floor(recoil_index)
    if crouch:
        return _remap(n, p['transStart'], p['transEnd'], p['recoveryCrouch'], p['recoveryCrouchFinal'])
    return _remap(n, p['transStart'], p['transEnd'], p['recoveryStand'], p['recoveryStandFinal'])


def decay_penalty(pen, dt, rec_time):
    if not pen or pen <= 0:
        return 0.0
    v = pen * math.exp(-max(dt, 0.0) * math.log(10) / max(0.01, rec_time))
    return 0.0 if v < 1e-5 else v


def inaccuracy_fire(wid):
    if wid not in WEAPON_PARAMS:
        return 0.0
    p = WEAPON_PARAMS[wid]
    return p['inaccuracyFire'][p['recoilMode']]
