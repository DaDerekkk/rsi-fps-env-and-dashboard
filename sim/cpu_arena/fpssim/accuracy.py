"""Shot accuracy and spread sampling, ported from cs2-web shared/aim.js + weapon-accuracy-data.js (commit e0e7e858).

Platform conventions (z up): forward = (cos p cos y, -cos p sin y, sin p).  Validated in validate/spread.py.
"""
import json, math, os

ACC = json.load(open(os.path.join(os.path.dirname(__file__), 'accuracy_data.json')))
STANDING_EYE_HEIGHT = 1.62
CROUCH_EYE_HEIGHT = 0.95


def mode_index(w, zoom_level):
    """index 1 = scoped (multi-zoom weapons when zoomed) or the silenced variants m4a1 / usp."""
    return 1 if ((len(w.get('zoomFovs', [90])) > 1 and zoom_level > 0) or w['id'] in ('m4a1', 'usp')) else 0


def accuracy(w, speed, crouch, grounded, vz, zoom_level=0, penalty=0.0):
    src = ACC.get(w['id'])
    if not src:
        return dict(spread=0.0, inaccuracy=0.0, total=0.0, moving=0.0)
    i = mode_index(w, zoom_level)
    maxspeed = (w.get('scopedMaxSpeed') if zoom_level > 0 else w.get('maxSpeed')) or 6
    moving = max(0.0, min(1.0, (speed / maxspeed - .34) / (.95 - .34)))
    air = 0.0 if grounded else min(1.0, abs(vz) / 5.5)
    inacc = src['crouch' if crouch else 'stand'][i] + src['move'][i] * moving ** 3 + src['jump'][i] * air + (penalty or 0.0)
    sp = src['spread'][i]
    return dict(spread=sp, inaccuracy=inacc, total=sp + inacc, moving=moving)


def basis(yaw, pitch):
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    fwd = (cp * cy, -cp * sy, sp)
    right = (-sy, -cy, 0.0)               # d fwd / d yaw: yaw increases to the right
    up = (-sp * cy, sp * sy, cp)
    return fwd, right, up


def sample_direction(yaw, pitch, acc, rnd):
    """cs2-web sampleShotDirection: two independent uniform-radius discs (inaccuracy, spread) on the camera plane."""
    ra = rnd() * acc['inaccuracy']; aa = rnd() * math.pi * 2
    rb = rnd() * acc['spread']; ab = rnd() * math.pi * 2
    x = math.cos(aa) * ra + math.cos(ab) * rb
    y = math.sin(aa) * ra + math.sin(ab) * rb
    f, r, u = basis(yaw, pitch)
    d = [f[k] + r[k] * x + u[k] * y for k in range(3)]
    n = math.sqrt(sum(v * v for v in d))
    return tuple(v / n for v in d)
