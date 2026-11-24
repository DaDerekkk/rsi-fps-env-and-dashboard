"""Bullet tracing with wall penetration, ported from cs2-web server/bullet-penetration.js + physics.raycastWorldSurfaces.

Works on a TriMesh (Dust2/Mirage).  Triangle normals follow the cs2-web winding mapped into platform coordinates; the
platform basis has det -1, so the world normal is -(e1 x e2) of the transformed triangle.
"""
import json, os
import numpy as np

SURFACE_RULES = {0: ('concrete', .09, .48), 1: ('wood', .70, .14), 2: ('metal', .14, .34), 3: ('glass', .90, .08)}
POWER = json.load(open(os.path.join(os.path.dirname(__file__), 'penetration_power.json')))


def surfaces(mesh, origin, direction, max_dist=300.0, flip=True):
    o = np.asarray(origin, np.float64); d = np.asarray(direction, np.float64); d = d / np.linalg.norm(d)
    t, ids = mesh.intersect(o, o + d * max_dist)
    out = []
    for tt, tid in zip(t, ids):
        dist = tt * max_dist
        if out and dist - out[-1]['distance'] < .002:
            continue
        n = np.cross(mesh.e1[tid], mesh.e2[tid]); n /= (np.linalg.norm(n) or 1.0)
        if flip:
            n = -n
        mat = int(mesh.mat[tid]) if mesh.mat is not None else 0
        out.append(dict(distance=float(dist), normal=n, material=mat, triangle=int(tid)))
        if len(out) == 48:
            break
    return out


def trace_bullet(origin, direction, weapon, player_hits, crossings):
    """player_hits: list of (seat, distance, part) for enemies the ray intersects (any order).
    Returns dict(end_distance, hit_world, hits=[(seat, distance, part, damage_scale, wallbang, penetrations)], segments)."""
    rng = weapon.get('range') or 200
    power = POWER.get(weapon['id'], 0)
    d = np.asarray(direction, np.float64); d = d / np.linalg.norm(d)
    targets = sorted(player_hits, key=lambda h: h[1])
    scale, wallbang, pens, stop, hit_world = 1.0, False, 0, rng, False
    hits, segs = [], []
    ti = 0

    def hit_targets(until):
        nonlocal ti, scale, stop
        while ti < len(targets) and targets[ti][1] < until:
            seat, dist, part = targets[ti]; ti += 1
            hits.append((seat, dist, part, scale, wallbang, pens))
            scale *= .55
            if len(hits) >= 4 or scale * weapon['damage'] < 1:
                stop = dist
                return False
        return True

    i = 0
    while i < len(crossings):
        e = crossings[i]
        if e['distance'] > rng:
            break
        if not hit_targets(e['distance']):
            break
        name, depth, loss = SURFACE_RULES.get(e['material'], SURFACE_RULES[0])
        x = crossings[i + 1] if i + 1 < len(crossings) else None
        thick = x['distance'] - e['distance'] if x else float('inf')
        maxd = depth * power
        valid = x is not None and float(np.dot(e['normal'], d)) < -.01 and float(np.dot(x['normal'], d)) > .01 \
            and x['material'] == e['material'] and thick > .002
        if not power or pens >= 4 or not valid or thick > maxd:
            stop = e['distance']; hit_world = True
            break
        rem = scale * (1 - loss) - thick / max(.01, maxd) * .42
        if rem * weapon['damage'] < 1:
            stop = e['distance']; hit_world = True
            break
        segs.append(dict(entry=e['distance'], exit=x['distance'], material=name, thickness=thick))
        scale = rem; wallbang = True; pens += 1; i += 2
        while ti < len(targets) and targets[ti][1] < x['distance']:
            ti += 1
        continue
    if not hit_world and stop == rng:
        hit_targets(rng)
    if not hit_world and hits:
        stop = min(stop, hits[-1][1] + .15)
    return dict(end_distance=stop, hit_world=hit_world, hits=hits, segments=segs)
