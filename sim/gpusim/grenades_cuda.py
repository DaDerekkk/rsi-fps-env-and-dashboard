"""CUDA kernels of gpusim.grenades (torch load_inline, JIT).  See gpusim/grenades.py and docs/gpu_grenades.md.

All physics in float64 with FMA contraction disabled (-fmad=false) so that the arithmetic follows fpssim/grenades.py
operation by operation.  The BVH (gpusim/bvh.py layout) is traversed in float32 with node boxes padded by 2 mm
(conservative); the triangle test itself is the float64 Moller-Trumbore of fpssim.trimesh.TriMesh.intersect.
Whitebox maps use the analytic BoxContact model (float32-rounded boxes, stair steps, bounded ground plane) verbatim.

Struct layouts are generated from the field lists below so that Python and C++ agree on the tensor order.
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

# (name, ctype)   ctype: int -> torch.int32, double -> torch.float64, bool -> torch.bool
GS_FIELDS = [
    # projectiles [E,P]
    ('pr_kind', 'int'), ('pr_owner', 'int'), ('pr_born', 'int'), ('pr_det', 'int'), ('pr_rest', 'int'),
    ('pr_ign', 'int'), ('pr_lastb', 'int'), ('pr_hasb', 'int'), ('pr_bounces', 'int'), ('pr_serial', 'int'),
    ('pr_id', 'int'), ('pr_mode', 'int'), ('pr_pos', 'double'), ('pr_vel', 'double'),
    # smokes [E,S]
    ('sm_valid', 'int'), ('sm_created', 'int'), ('sm_expires', 'int'), ('sm_serial', 'int'), ('sm_id', 'int'),
    ('sm_owner', 'int'), ('sm_ncv', 'int'), ('sm_pos', 'double'),
    # carves [E,S,C] (list order, sm_ncv entries)
    ('cv_type', 'int'), ('cv_created', 'int'), ('cv_a', 'double'), ('cv_b', 'double'), ('cv_radius', 'double'),
    ('cv_dur', 'double'), ('cv_rec', 'double'),
    # fires [E,F], cells [E,F,NC]
    ('fi_valid', 'int'), ('fi_kind', 'int'), ('fi_owner', 'int'), ('fi_created', 'int'), ('fi_expires', 'int'),
    ('fi_next', 'int'), ('fi_serial', 'int'), ('fi_id', 'int'), ('fi_ncell', 'int'), ('fi_pos', 'double'),
    ('fi_cells', 'double'), ('fi_cok', 'int'),
    # decoys [E,D]
    ('dc_valid', 'int'), ('dc_owner', 'int'), ('dc_expires', 'int'), ('dc_next', 'int'), ('dc_serial', 'int'),
    ('dc_id', 'int'), ('dc_pos', 'double'),
    # primes [E,10]
    ('pm_kind', 'int'), ('pm_mode', 'int'), ('pm_tick', 'int'), ('pm_rel', 'int'),
    # counters [E]
    ('ctr_serial', 'int'), ('ctr_id', 'int'),
    # fire-cell scratch: ig_slot [E,P] (slot of an igniting projectile, -1), ig_proj [E,NI], ig_ok [E,NI,64], ig_cell [E,NI,64,3]
    ('ig_slot', 'int'), ('ig_proj', 'int'), ('ig_ok', 'int'), ('ig_cell', 'double'),
]
PL_FIELDS = [
    ('x', 'double'), ('y', 'double'), ('z', 'double'), ('vx', 'double'), ('vy', 'double'), ('vz', 'double'),
    ('yaw', 'double'), ('pitch', 'double'), ('hp', 'double'), ('armor', 'double'),
    ('alive', 'bool'), ('crouch', 'bool'),
    ('blind_until', 'int'), ('blind_start', 'int'), ('throw_ready', 'int'), ('nades', 'int'),
    ('death_tick', 'int'), ('protection_until', 'int'),        # optional (empty tensor -> nullptr)
]
EV_FIELDS = [
    # step events
    ('ev_bounce', 'int'), ('ev_bounce_pos', 'double'),                 # [E,P], [E,P,3]
    ('ev_det', 'int'), ('ev_det_id', 'int'), ('ev_det_kind', 'int'), ('ev_det_pos', 'double'),   # [E,P]
    ('ev_fire_cells', 'int'),                                          # [E,P] cells of a fire_started (0 none)
    ('ev_fire_ext', 'int'), ('ev_decoy_pulse', 'int'), ('ev_decoy_exp', 'int'),   # [E]
    ('ev_flash_dur', 'int'), ('ev_flash_n', 'int'),                    # [E,10]
    ('ev_dmg', 'double'), ('ev_hits', 'int'), ('ev_killer', 'int'), ('ev_kill_kind', 'int'),   # [E,10]
    # throw events [E,10]
    ('th_flag', 'int'), ('th_kind', 'int'), ('th_slot', 'int'), ('th_id', 'int'), ('th_pos', 'double'),
]
WD_FIELDS = [   # world arrays
    ('nlo', 'float'), ('nhi', 'float'), ('nleft', 'int'), ('nstart', 'int'), ('ncount', 'int'),
    ('v0', 'double'), ('e1', 'double'), ('e2', 'double'), ('nrm', 'double'), ('tid', 'int'),
    ('blo', 'double'), ('bhi', 'double'),
]
TORCH_DT = {'int': torch.int32, 'double': torch.float64, 'bool': torch.bool, 'float': torch.float32}


def _struct(name, fields):
    s = 'struct %s {\n' % name + ''.join('  %s* %s;\n' % (t, n) for n, t in fields) + '};\n'
    s += 'static %s make_%s(const std::vector<torch::Tensor>& v){ %s r;\n' % (name, name, name)
    for i, (n, t) in enumerate(fields):
        s += '  r.%s = v[%d].numel() ? v[%d].data_ptr<%s>() : nullptr;\n' % (n, i, i, t)
    return s + '  return r; }\n'


_KERNELS = r'''
#define NSEAT 10
#define TICK_HZ 60
#define SUBSTEP (1.0/120)
#define GRAVITY 8.128
#define BOUNCE 0.45
#define RADIUS 0.06
#define EYE_STAND 1.62
#define EYE_CROUCH 0.95
#define WALKABLE 0.7
#define THROW_READY_TICKS 24
#define PRIME_MIN_TICKS 12
#define MAX_FLIGHT_TICKS 720
#define FIRE_DAMAGE_TICKS 12
#define DECOY_PULSE_TICKS 54
#define PI_D 3.141592653589793
#define K_HE 0
#define K_FLASH 1
#define K_SMOKE 2
#define K_MOLO 3
#define K_INC 4
#define K_DECOY 5
#define CV_BULLET 1
#define CV_EXPL 2
#define BULLET_CARVE_MAX 30

struct Dims { int E, P, S, C, F, NC, D, NI; };
struct UT { int fuse[6]; int duration[6]; double radius[6]; double damage[6]; };
struct WorldP { int type; int nb; double ground, gmin0, gmin1, gmax0, gmax1, csox, csoy; };

// ------------------------------------------------------------------------------------------- world queries
__device__ __forceinline__ bool slab(const float* lo, const float* hi, const float* o, const float* inv,
                                     const bool* par, float tmax){
  float t0 = 0.f, t1 = tmax;
  #pragma unroll
  for (int a = 0; a < 3; a++){
    if (par[a]) { if (o[a] < lo[a] || o[a] > hi[a]) return false; continue; }
    float ta = (lo[a] - o[a]) * inv[a], tb = (hi[a] - o[a]) * inv[a];
    t0 = fmaxf(t0, fminf(ta, tb)); t1 = fminf(t1, fmaxf(ta, tb));
  }
  return t0 <= t1;
}

// first hit of segment p0->p1 (TriMesh.intersect, double sided): param t, BVH-order index k
__device__ bool mesh_first(const WD& W, const double* p0, const double* p1, double& tbest, int& kbest){
  double d[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
  float o32[3], inv[3]; bool par[3];
  for (int a = 0; a < 3; a++){ o32[a] = (float)p0[a]; float da = (float)d[a]; par[a] = (da == 0.f); inv[a] = par[a] ? 0.f : 1.f / da; }
  tbest = 2.0; kbest = -1; int bid = 0x7fffffff;
  int st[64]; int sp = 0; st[sp++] = 0;
  while (sp){
    int n = st[--sp];
    float tm = (kbest >= 0 ? (float)tbest : 1.f) + 1e-5f;
    if (!slab(W.nlo + 3*n, W.nhi + 3*n, o32, inv, par, tm)) continue;
    int l = W.nleft[n];
    if (l >= 0){ st[sp++] = l; st[sp++] = l + 1; continue; }
    int s = W.nstart[n], c = W.ncount[n];
    for (int k = s; k < s + c; k++){
      const double* v0 = W.v0 + 3*k; const double* e1 = W.e1 + 3*k; const double* e2 = W.e2 + 3*k;
      double hx = d[1]*e2[2] - d[2]*e2[1], hy = d[2]*e2[0] - d[0]*e2[2], hz = d[0]*e2[1] - d[1]*e2[0];
      double a = e1[0]*hx + e1[1]*hy + e1[2]*hz;
      if (!(fabs(a) > 1e-12)) continue;
      double f = 1.0 / a;
      double sx = p0[0] - v0[0], sy = p0[1] - v0[1], sz = p0[2] - v0[2];
      double u = f * (sx*hx + sy*hy + sz*hz);
      if (!(u >= 0.0)) continue;
      double qx = sy*e1[2] - sz*e1[1], qy = sz*e1[0] - sx*e1[2], qz = sx*e1[1] - sy*e1[0];
      double v = f * (qx*d[0] + qy*d[1] + qz*d[2]);
      if (!(v >= 0.0) || !(u + v <= 1.0)) continue;
      double t = f * (e2[0]*qx + e2[1]*qy + e2[2]*qz);
      if (!(t >= 0.0 && t <= 1.0)) continue;
      int id = W.tid[k];
      if (t < tbest || (t == tbest && id < bid)){ tbest = t; kbest = k; bid = id; }
    }
  }
  return kbest >= 0;
}

// BoxContact._hits
__device__ bool box_hits(const WD& W, const WorldP& P, const double* o, const double* d, double L, double& dist, double* n){
  bool found = false; double best = 0.0; int bax = 0; double bsign = 0.0;
  // entering
  int je = -1; double te = 0.0; int axe = 0;
  int jx = -1; double tx = 0.0; int axx = 0;
  double smin[3], smax[3];                 // bounds of the segment: a box that does not touch them cannot be entered or left inside [0, L]
  for (int a = 0; a < 3; a++){ double e1 = o[a] + d[a] * L; smin[a] = fmin(o[a], e1) - 1e-6; smax[a] = fmax(o[a], e1) + 1e-6; }
  for (int j = 0; j < P.nb; j++){
    if (smax[0] < W.blo[3*j] || smin[0] > W.bhi[3*j] || smax[1] < W.blo[3*j+1] || smin[1] > W.bhi[3*j+1] ||
        smax[2] < W.blo[3*j+2] || smin[2] > W.bhi[3*j+2]) continue;
    double tmin[3], tmax[3];
    for (int a = 0; a < 3; a++){
      double lo = W.blo[3*j + a], hi = W.bhi[3*j + a];
      if (d[a] == 0.0){
        bool inside = (o[a] >= lo) && (o[a] <= hi);
        tmin[a] = inside ? -INFINITY : INFINITY; tmax[a] = inside ? INFINITY : -INFINITY;
      } else {
        double inv = 1.0 / d[a];
        double ta = (lo - o[a]) * inv, tb = (hi - o[a]) * inv;
        tmin[a] = fmin(ta, tb); tmax[a] = fmax(ta, tb);
      }
    }
    double t0 = fmax(fmax(tmin[0], tmin[1]), tmin[2]);
    double t1 = fmin(fmin(tmax[0], tmax[1]), tmax[2]);
    if (!(t0 <= t1)) continue;
    if (t0 >= 0.0 && t0 <= L){
      if (je < 0 || t0 < te){
        je = j; te = t0;
        int ax = 0; if (tmin[1] > tmin[ax]) ax = 1; if (tmin[2] > tmin[ax]) ax = 2; axe = ax;
      }
    }
    if (t0 < 0.0 && t1 >= 0.0 && t1 <= L){
      if (jx < 0 || t1 < tx){
        jx = j; tx = t1;
        int ax = 0; if (tmax[1] < tmax[ax]) ax = 1; if (tmax[2] < tmax[ax]) ax = 2; axx = ax;
      }
    }
  }
  if (je >= 0){ found = true; best = te; bax = axe; bsign = d[axe] > 0 ? -1.0 : 1.0; }
  if (jx >= 0 && (!found || tx < best)){ found = true; best = tx; bax = axx; bsign = d[axx] > 0 ? -1.0 : 1.0; }
  n[0] = n[1] = n[2] = 0.0;
  if (found) n[bax] = bsign;
  if (d[2] != 0.0){
    double tg = (P.ground - o[2]) / d[2];
    if (0.0 <= tg && tg <= L && (!found || tg < best)){
      double px = o[0] + d[0] * tg, py = o[1] + d[1] * tg;
      if (P.gmin0 <= px && px <= P.gmax0 && P.gmin1 <= py && py <= P.gmax1){
        found = true; best = tg; n[0] = 0.0; n[1] = 0.0; n[2] = d[2] < 0 ? 1.0 : -1.0;
      }
    }
  }
  dist = best;
  return found;
}

// world.contact(o, d, L) -> (dist, normal facing the ray)
__device__ bool contact(const WD& W, const WorldP& P, const double* o, const double* d, double L, double& dist, double* n){
  if (P.type == 1) return box_hits(W, P, o, d, L, dist, n);
  double p1[3] = {o[0] + d[0] * L, o[1] + d[1] * L, o[2] + d[2] * L};
  double t; int k;
  if (!mesh_first(W, o, p1, t, k)) return false;
  dist = t * L;
  const double* nn = W.nrm + 3*k;
  if (nn[0]*d[0] + nn[1]*d[1] + nn[2]*d[2] > 0){ n[0] = -nn[0]; n[1] = -nn[1]; n[2] = -nn[2]; }
  else { n[0] = nn[0]; n[1] = nn[1]; n[2] = nn[2]; }
  return true;
}

__device__ __forceinline__ double dist3(const double* a, const double* b){
  double dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
  return sqrt(dx*dx + dy*dy + dz*dz);
}

// any triangle hit on the segment p0->p1 whose distance t*L is < Lthr (= first-hit distance < Lthr, see clear_sight); early exit
__device__ bool mesh_any_before(const WD& W, const double* p0, const double* p1, double L, double Lthr){
  if (!(Lthr > 0.0)) return false;
  double d[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
  float o32[3], inv[3]; bool par[3];
  for (int a = 0; a < 3; a++){ o32[a] = (float)p0[a]; float da = (float)d[a]; par[a] = (da == 0.f); inv[a] = par[a] ? 0.f : 1.f / da; }
  float tm = (float)(Lthr / L) + 1e-5f;
  if (tm > 1.00001f) tm = 1.00001f;
  int st[64]; int sp = 0; st[sp++] = 0;
  while (sp){
    int n = st[--sp];
    if (!slab(W.nlo + 3*n, W.nhi + 3*n, o32, inv, par, tm)) continue;
    int l = W.nleft[n];
    if (l >= 0){ st[sp++] = l; st[sp++] = l + 1; continue; }
    int s = W.nstart[n], c = W.ncount[n];
    for (int k = s; k < s + c; k++){
      const double* v0 = W.v0 + 3*k; const double* e1 = W.e1 + 3*k; const double* e2 = W.e2 + 3*k;
      double hx = d[1]*e2[2] - d[2]*e2[1], hy = d[2]*e2[0] - d[0]*e2[2], hz = d[0]*e2[1] - d[1]*e2[0];
      double a = e1[0]*hx + e1[1]*hy + e1[2]*hz;
      if (!(fabs(a) > 1e-12)) continue;
      double f = 1.0 / a;
      double sx = p0[0] - v0[0], sy = p0[1] - v0[1], sz = p0[2] - v0[2];
      double u = f * (sx*hx + sy*hy + sz*hz);
      if (!(u >= 0.0)) continue;
      double qx = sy*e1[2] - sz*e1[1], qy = sz*e1[0] - sx*e1[2], qz = sx*e1[1] - sy*e1[0];
      double v = f * (qx*d[0] + qy*d[1] + qz*d[2]);
      if (!(v >= 0.0) || !(u + v <= 1.0)) continue;
      double t = f * (e2[0]*qx + e2[1]*qy + e2[2]*qz);
      if (!(t >= 0.0 && t <= 1.0)) continue;
      if (t * L < Lthr) return true;
    }
  }
  return false;
}

__device__ bool clear_sight(const WD& W, const WorldP& P, const double* a, const double* b){
  double L = dist3(a, b);
  if (L < 0.01) return true;
  double d[3] = {(b[0] - a[0]) / L, (b[1] - a[1]) / L, (b[2] - a[2]) / L};
  if (P.type == 0){                      // mesh: blocked iff some hit has first-hit distance < L - 0.03 (CPU: h >= L - 0.03 is clear)
    double p1[3] = {a[0] + d[0] * L, a[1] + d[1] * L, a[2] + d[2] * L};
    return !mesh_any_before(W, a, p1, L, L - 0.03);
  }
  double h, n[3];
  if (!contact(W, P, a, d, L, h, n)) return true;
  return h >= L - 0.03;
}

// GrenadeSystem._sweep (7 rays, cs2 order mapped to world)
__device__ bool sweep(const WD& W, const WorldP& P, const double* o, const double* d, double L, double& bd, double* bn){
  const double OFF[7][3] = {{0,0,0},{0,RADIUS,0},{0,-RADIUS,0},{0,0,RADIUS},{0,0,-RADIUS},{-RADIUS,0,0},{RADIUS,0,0}};
  bool found = false;
  for (int k = 0; k < 7; k++){
    double oo[3] = {o[0] + OFF[k][0], o[1] + OFF[k][1], o[2] + OFF[k][2]};
    double dist, n[3];
    if (!contact(W, P, oo, d, L + RADIUS, dist, n)) continue;
    double closing = -(d[0]*n[0] + d[1]*n[1] + d[2]*n[2]);
    if (closing < 0.00001) continue;
    double dd = dist - (k == 0 ? RADIUS / closing : 0.0);
    if (dd < 0.0) dd = 0.0;
    if (dd <= L && (!found || dd < bd)){ found = true; bd = dd; bn[0] = n[0]; bn[1] = n[1]; bn[2] = n[2]; }
  }
  return found;
}

__device__ __forceinline__ void fwd(double yaw, double pitch, double* f){
  double c = cos(pitch);
  f[0] = c * cos(yaw); f[1] = -c * sin(yaw); f[2] = sin(pitch);
}

// ------------------------------------------------------------------------------------------- throw
__global__ void throw_k(GS g, PL pl, EV ev, WD W, WorldP WP, Dims D, UT U,
                        const int* __restrict__ akind, const int* __restrict__ astate, const int* __restrict__ amode,
                        const bool* __restrict__ mask, const int* __restrict__ tickv){
  int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= D.E) return;
  if (!mask[e]) return;
  int tick = tickv[e];
  for (int s = 0; s < NSEAT; s++){
    int i = e * NSEAT + s;
    if (!pl.alive[i]){ g.pm_kind[i] = -1; continue; }
    int kind = akind[i], st = astate[i], mode = amode[i];
    int* nd = pl.nades + i * 6;
    bool kvalid = kind >= 0 && kind < 6;
    if (st == 3){ g.pm_kind[i] = -1; }
    else if (st == 1 && kvalid){
      if (nd[kind] > 0){
        int cur = g.pm_kind[i];
        if (cur < 0 || cur != kind || g.pm_rel[i]){ g.pm_kind[i] = kind; g.pm_mode[i] = mode; g.pm_tick[i] = tick; g.pm_rel[i] = 0; }
        else g.pm_mode[i] = mode;
      }
    } else if (st == 2){
      int cur = g.pm_kind[i];
      if (cur >= 0 && (kind < 0 || kind == cur) && g.pm_tick[i] < tick){
        if (!g.pm_rel[i]){ g.pm_rel[i] = 1; g.pm_mode[i] = mode; }
      }
    }
    if (g.pm_kind[i] < 0 || !g.pm_rel[i]) continue;
    if (!(tick >= pl.throw_ready[i])) continue;
    if (tick < g.pm_tick[i] + PRIME_MIN_TICKS) continue;   // cs2-web readyAt = primedAt + 200 ms (fpssim.grenades.PRIME_MIN_TICKS)
    int k = g.pm_kind[i]; int md = g.pm_mode[i];
    g.pm_kind[i] = -1;
    if (nd[k] <= 0) continue;
    // free projectile slot (CPU: list length < MAX_PROJECTILES)
    int slot = -1;
    for (int p = 0; p < D.P; p++) if (g.pr_kind[e * D.P + p] < 0){ slot = p; break; }
    if (slot < 0) continue;                  // projectile table full: prime consumed, no throw (CPU: throw() -> None)
    double pitch = pl.pitch[i], yaw = pl.yaw[i];
    double raw = fmax(-1.48, fmin(1.48, pitch));
    double pa = raw + (PI_D / 18) * (1 - fabs(raw) / (PI_D / 2));
    double d[3]; fwd(yaw, pa, d);
    if (md < 0 || md > 2) md = 0;
    double strength = md == 0 ? 1.0 : (md == 1 ? 0.5 : 0.0);
    double speed = (750 * 0.0254) * 0.9 * (0.3 + 0.7 * strength);
    double o[3] = {pl.x[i], pl.y[i], pl.z[i] + (pl.crouch[i] ? EYE_CROUCH : EYE_STAND) - (1 - strength) * 0.3};
    double bd, bn[3];
    double off = 0.35;
    if (sweep(W, WP, o, d, 0.4, bd, bn)) off = fmax(0.0, bd - 0.01);
    int q = e * D.P + slot;
    g.ctr_serial[e] += 1; g.ctr_id[e] += 1;
    g.pr_kind[q] = k; g.pr_owner[q] = s; g.pr_mode[q] = md;
    g.pr_pos[3*q] = o[0] + d[0] * off; g.pr_pos[3*q+1] = o[1] + d[1] * off; g.pr_pos[3*q+2] = o[2] + d[2] * off;
    g.pr_vel[3*q] = d[0] * speed + pl.vx[i] * 1.25; g.pr_vel[3*q+1] = d[1] * speed + pl.vy[i] * 1.25;
    g.pr_vel[3*q+2] = d[2] * speed + pl.vz[i] * 1.25;
    g.pr_born[q] = tick; g.pr_det[q] = tick + U.fuse[k]; g.pr_rest[q] = 0; g.pr_ign[q] = 0; g.pr_hasb[q] = 0;
    g.pr_lastb[q] = 0; g.pr_bounces[q] = 0; g.pr_serial[q] = g.ctr_serial[e]; g.pr_id[q] = g.ctr_id[e];
    nd[k] -= 1;
    pl.throw_ready[i] = tick + THROW_READY_TICKS;
    ev.th_flag[i] = 1; ev.th_kind[i] = k; ev.th_slot[i] = slot; ev.th_id[i] = g.ctr_id[e];
    ev.th_pos[3*i] = g.pr_pos[3*q]; ev.th_pos[3*i+1] = g.pr_pos[3*q+1]; ev.th_pos[3*i+2] = g.pr_pos[3*q+2];
  }
}

// ------------------------------------------------------------------------------------------- projectile advance
__global__ void advance_k(GS g, EV ev, WD W, WorldP WP, Dims D, const bool* __restrict__ live, const int* __restrict__ tickv){
  int q = blockIdx.x * blockDim.x + threadIdx.x; if (q >= D.E * D.P) return;
  int e = q / D.P;
  if (!live[e]) return;
  int kind = g.pr_kind[q]; if (kind < 0) return;
  int now = tickv[e];
  double x = g.pr_pos[3*q], y = g.pr_pos[3*q+1], z = g.pr_pos[3*q+2];
  double vx = g.pr_vel[3*q], vy = g.pr_vel[3*q+1], vz = g.pr_vel[3*q+2];
  int rest = g.pr_rest[q], ign = g.pr_ign[q], hasb = g.pr_hasb[q], lastb = g.pr_lastb[q], nb = g.pr_bounces[q];
  bool fire = (kind == K_MOLO || kind == K_INC);
  const double dt = SUBSTEP;
  for (int sub = 0; sub < 2 && !ign; sub++){
    if (rest) continue;
    vz -= GRAVITY * dt * 0.5;
    double remaining = dt;
    bool done = false;
    for (int it = 0; it < 4; it++){
      if (remaining <= 1e-6) break;
      double sp = sqrt(vx*vx + vy*vy + vz*vz);
      if (sp < 1e-6) break;
      double d[3] = {vx / sp, vy / sp, vz / sp};
      double travel = sp * remaining;
      double o[3] = {x, y, z};
      double dist, n[3];
      if (!sweep(W, WP, o, d, travel, dist, n)){
        x += d[0] * travel; y += d[1] * travel; z += d[2] * travel;
        break;
      }
      x += d[0] * dist + n[0] * 0.002; y += d[1] * dist + n[1] * 0.002; z += d[2] * dist + n[2] * 0.002;
      if (n[2] > WALKABLE && fire){ ign = 1; vx = vy = vz = 0.0; done = true; break; }
      double dot = vx*n[0] + vy*n[1] + vz*n[2];
      vx = (vx - 2 * dot * n[0]) * BOUNCE;
      vy = (vy - 2 * dot * n[1]) * BOUNCE;
      vz = (vz - 2 * dot * n[2]) * BOUNCE;
      nb += 1;
      if (fabs(dot) > 1 && (!hasb || (now - lastb) * 1000.0 / TICK_HZ > 90)){
        hasb = 1; lastb = now;
        ev.ev_bounce[q] += 1; ev.ev_bounce_pos[3*q] = x; ev.ev_bounce_pos[3*q+1] = y; ev.ev_bounce_pos[3*q+2] = z;
      }
      if (n[2] > WALKABLE && sqrt(vx*vx + vy*vy + vz*vz) < 1.2){ vx = vy = vz = 0.0; rest = 1; done = true; break; }
      double r = 1 - dist / fmax(travel, 0.001);
      remaining *= fmax(0.0, r);
    }
    if (!done) vz -= GRAVITY * dt * 0.5;
  }
  g.pr_pos[3*q] = x; g.pr_pos[3*q+1] = y; g.pr_pos[3*q+2] = z;
  g.pr_vel[3*q] = vx; g.pr_vel[3*q+1] = vy; g.pr_vel[3*q+2] = vz;
  g.pr_rest[q] = rest; g.pr_ign[q] = ign; g.pr_hasb[q] = hasb; g.pr_lastb[q] = lastb; g.pr_bounces[q] = nb;
}

// ------------------------------------------------------------------------------------------- effects (per env)
struct Ctx { GS g; PL pl; EV ev; WD W; WorldP WP; Dims D; UT U; int e; int now; };

__device__ void damage(Ctx& c, int v, int attacker, double amount, int kind){
  int i = c.e * NSEAT + v;
  c.pl.hp[i] -= amount;
  double a = c.pl.armor[i] - 0.0; c.pl.armor[i] = a > 0.0 ? a : 0.0;     // HE/fire/decoy bypass armor (platform)
  c.ev.ev_dmg[i] += amount; c.ev.ev_hits[i] += 1;
  if (c.pl.hp[i] <= 1e-9 && c.pl.alive[i]){
    c.pl.hp[i] = 0.0; c.pl.alive[i] = false;
    if (c.pl.death_tick) c.pl.death_tick[i] = c.now;
    c.ev.ev_killer[i] = attacker; c.ev.ev_kill_kind[i] = kind;
    for (int k = 0; k < 6; k++) c.pl.nades[i * 6 + k] = 0;
    c.g.pm_kind[i] = -1;
  }
}

__device__ __forceinline__ bool hurtable(Ctx& c, int v, int owner){
  int i = c.e * NSEAT + v;
  if (!c.pl.alive[i]) return false;
  if (c.pl.protection_until && c.now < c.pl.protection_until[i]) return false;
  return !((v / 5) == (owner / 5) && v != owner);
}

__device__ void extinguish(Ctx& c){
  const Dims& D = c.D; int e = c.e;
  for (int pass = 0; pass < D.F; pass++){
    // fires in list (serial) order
    int f = -1; int best = 0x7fffffff;
    for (int j = 0; j < D.F; j++){
      int q = e * D.F + j;
      if (c.g.fi_valid[q] == 1 && c.g.fi_serial[q] < best){ bool seen = false; (void)seen; best = c.g.fi_serial[q]; f = j; }
    }
    if (f < 0) break;
    int q = e * D.F + f;
    c.g.fi_valid[q] = 2;                    // mark visited this pass
    int before = 0, after = 0;
    for (int k = 0; k < D.NC; k++){
      int ck = q * D.NC + k;
      if (!c.g.fi_cok[ck]) continue;
      before++;
      const double* cp = c.g.fi_cells + 3 * ck;
      double a[3] = {cp[0], cp[1], cp[2] + 0.4};
      bool ext = false;
      for (int s = 0; s < D.S && !ext; s++){
        int sq = e * D.S + s;
        if (!c.g.sm_valid[sq] || !(c.g.sm_expires[sq] > c.now)) continue;
        if (dist3(a, c.g.sm_pos + 3 * sq) < c.U.radius[K_SMOKE]) ext = true;
      }
      if (ext) c.g.fi_cok[ck] = 0; else after++;
    }
    if (before && !after) c.ev.ev_fire_ext[e] += 1;
    c.g.fi_ncell[q] = after;
  }
  for (int j = 0; j < D.F; j++){
    int q = e * D.F + j;
    if (c.g.fi_valid[q] == 2) c.g.fi_valid[q] = (c.g.fi_ncell[q] > 0 && c.g.fi_expires[q] > c.now) ? 1 : 0;
  }
}

__device__ void burn(Ctx& c, int q){
  int kind = c.g.fi_kind[q], owner = c.g.fi_owner[q];
  for (int v = 0; v < NSEAT; v++){
    if (!hurtable(c, v, owner)) continue;
    int i = c.e * NSEAT + v;
    double x = c.pl.x[i], y = c.pl.y[i], z = c.pl.z[i];
    for (int k = 0; k < c.D.NC; k++){
      int ck = q * c.D.NC + k;
      if (!c.g.fi_cok[ck]) continue;
      const double* cp = c.g.fi_cells + 3 * ck;
      double ddx = x - cp[0], ddy = y - cp[1];
      if (ddx*ddx + ddy*ddy > 0.6) continue;                                   // cheap reject (0.75^2 = 0.5625), exact test below
      if (hypot(x - cp[0], y - cp[1]) < 0.75 && z < cp[2] + 1.3 && z > cp[2] - 0.4){
        double a[3] = {cp[0], cp[1], cp[2] + 0.2}, b[3] = {x, y, z + 0.5};
        if (clear_sight(c.W, c.WP, a, b)){ damage(c, v, owner, c.U.damage[kind], kind); break; }
      }
    }
  }
}

__device__ void explode(Ctx& c, const double* pos, int owner, int kind){
  double R = c.U.radius[kind];
  for (int v = 0; v < NSEAT; v++){
    if (!hurtable(c, v, owner)) continue;
    int i = c.e * NSEAT + v;
    double tgt[3] = {c.pl.x[i], c.pl.y[i], c.pl.z[i] + 0.9};
    double d = dist3(pos, tgt);
    if (d >= R || !clear_sight(c.W, c.WP, pos, tgt)) continue;
    damage(c, v, owner, c.U.damage[kind] * fmax(0.0, 1 - d / R), kind);
  }
}

__device__ void carve_push(Ctx& c, int sq){
  // make room at the end of the carve list of smoke sq (drop the oldest when the table is full)
  int C = c.D.C; int n = c.g.sm_ncv[sq];
  if (n < C) return;
  int b = sq * C;
  for (int k = 1; k < n; k++){
    int a0 = b + k - 1, a1 = b + k;
    c.g.cv_type[a0] = c.g.cv_type[a1]; c.g.cv_created[a0] = c.g.cv_created[a1];
    c.g.cv_radius[a0] = c.g.cv_radius[a1]; c.g.cv_dur[a0] = c.g.cv_dur[a1]; c.g.cv_rec[a0] = c.g.cv_rec[a1];
    for (int t = 0; t < 3; t++){ c.g.cv_a[3*a0+t] = c.g.cv_a[3*a1+t]; c.g.cv_b[3*a0+t] = c.g.cv_b[3*a1+t]; }
  }
  c.g.sm_ncv[sq] = n - 1;
}

__device__ void disperse(Ctx& c, const double* pos){
  const double radius = 3.5, duration = 2.5, recover = 0.6;
  for (int s = 0; s < c.D.S; s++){
    int sq = c.e * c.D.S + s;
    if (!c.g.sm_valid[sq] || c.g.sm_expires[sq] <= c.now) continue;
    double d = dist3(c.g.sm_pos + 3 * sq, pos);
    double sr = c.U.radius[K_SMOKE];
    if (d < sr + radius){
      carve_push(c, sq);
      int k = sq * c.D.C + c.g.sm_ncv[sq];
      c.g.cv_type[k] = CV_EXPL; c.g.cv_created[k] = c.now;
      c.g.cv_radius[k] = d < 2.0 ? fmax(radius, sr + 0.5) : radius;
      c.g.cv_dur[k] = duration; c.g.cv_rec[k] = recover;
      for (int t = 0; t < 3; t++){ c.g.cv_a[3*k+t] = pos[t]; c.g.cv_b[3*k+t] = 0.0; }
      c.g.sm_ncv[sq] += 1;
    }
  }
}

__device__ void flash(Ctx& c, const double* gp, int owner){
  const double R = c.U.radius[K_FLASH];
  for (int v = 0; v < NSEAT; v++){
    int i = c.e * NSEAT + v;
    if (!c.pl.alive[i]) continue;
    double eye[3] = {c.pl.x[i], c.pl.y[i], c.pl.z[i] + (c.pl.crouch[i] ? EYE_CROUCH : EYE_STAND)};
    double d = dist3(eye, gp);
    if (d >= R || !clear_sight(c.W, c.WP, eye, gp)) continue;
    double fw[3]; fwd(c.pl.yaw[i], c.pl.pitch[i], fw);
    double dot = d < 0.01 ? 1.0 : (fw[0] * (gp[0] - eye[0]) + fw[1] * (gp[1] - eye[1]) + fw[2] * (gp[2] - eye[2])) / d;
    double df = fmax(0.0, 1 - d / R);
    double hold, fade;
    if (dot > 0.5){ hold = 2.2 * df; fade = 2.3 * df; }
    else if (dot > -0.2){ double fct = (dot + 0.2) / 0.7; hold = 0.8 * fct * df; fade = 1.8 * df; }
    else { hold = 0.0; fade = 0.6 * df; }
    double dur = hold + fade;
    if (dur > 0.05){
      int ticks = (int)ceil(dur * TICK_HZ - 1e-9);
      if (c.now + ticks > c.pl.blind_until[i]){ c.pl.blind_until[i] = c.now + ticks; c.pl.blind_start[i] = c.now; }
      if (ticks > c.ev.ev_flash_dur[i]) c.ev.ev_flash_dur[i] = ticks;
      c.ev.ev_flash_n[i] += 1;
    }
  }
}

// fire cells: candidate list of GrenadeSystem._ignite (rings 0.1/0.65/1.30/1.95/2.60/3.20 with 1/6/11/15/17/12 cells, the LCG draws two
// numbers per candidate whether or not it is kept), candidate index ci -> floor cell.  Independent of every other candidate, so the
// 62 candidates can be probed by parallel threads (ign_pre_k) or sequentially (fallback in ignite()).
__device__ int ign_ncand(double R){
  const double RR[6] = {0.1, 0.65, 1.30, 1.95, 2.60, 3.20}; const int CNT[6] = {1, 6, 11, 15, 17, 12};
  int n = 0; for (int k = 0; k < 6; k++) if (!(RR[k] > R)) n += CNT[k];
  return n;
}

__device__ bool ign_candidate(const WD& W, const WorldP& WP, double gx, double gy, double gz, double R, int ci, double* cell){
  const double RR[6] = {0.1, 0.65, 1.30, 1.95, 2.60, 3.20}; const int CNT[6] = {1, 6, 11, 15, 17, 12};
  double xc = gy - WP.csoy, zc = -(gx - WP.csox);
  double vabs = fabs(xc * 1000 + zc);
  unsigned int state = (unsigned int)(unsigned long long)trunc(vabs);
  if (state == 0u) state = 1u;
  int ring = -1, i = 0, acc = 0;
  for (int k = 0; k < 6; k++){
    if (RR[k] > R) continue;
    if (ci < acc + CNT[k]){ ring = k; i = ci - acc; break; }
    acc += CNT[k];
  }
  if (ring < 0) return false;
  double rnd1 = 0.0, rnd2 = 0.0;
  for (int k = 0; k <= ci; k++){
    state = state * 1664525u + 1013904223u; rnd1 = state / 4294967296.0;
    state = state * 1664525u + 1013904223u; rnd2 = state / 4294967296.0;
  }
  double rr = RR[ring]; int cnt = CNT[ring];
  double ang = ((double)i / cnt) * PI_D * 2 + rnd1 * 0.45;
  double r = rr + (rnd2 - 0.5) * 0.32;
  if (r > R) return false;
  double ox = cos(ang) * r, oz = sin(ang) * r;
  double o[3] = {gx - oz, gy + ox, gz + 0.45};
  double dn[3] = {0.0, 0.0, -1.0};
  double fd, fn[3];
  if (!contact(W, WP, o, dn, 1.3, fd, fn) || fn[2] < WALKABLE) return false;
  cell[0] = o[0]; cell[1] = o[1]; cell[2] = o[2] - fd + 0.025;
  double a[3] = {gx, gy, gz + 0.25}, b[3] = {cell[0], cell[1], cell[2] + 0.25};
  return clear_sight(W, WP, a, b);
}

// returns number of cells of the new fire after extinguish (0: no fire or extinguished)
__device__ int ignite(Ctx& c, int q){
  const Dims& D = c.D; int e = c.e;
  int kind = c.g.pr_kind[q];
  double gx = c.g.pr_pos[3*q], gy = c.g.pr_pos[3*q+1], gz = c.g.pr_pos[3*q+2];
  double R = c.U.radius[kind];
  // target fire slot: first free, else evict the oldest (CPU keeps <= 12)
  int f = -1;
  for (int j = 0; j < D.F; j++) if (!c.g.fi_valid[e * D.F + j]){ f = j; break; }
  if (f < 0){
    int best = 0x7fffffff;
    for (int j = 0; j < D.F; j++){ int qq = e * D.F + j; if (c.g.fi_serial[qq] < best){ best = c.g.fi_serial[qq]; f = j; } }
  }
  int fq = e * D.F + f;
  int ncell = 0;
  int cap = D.NC < 60 ? D.NC : 60;
  int ncand = ign_ncand(R);
  int slot = c.g.ig_slot[q];                    // >= 0: the candidates were probed in parallel by ign_pre_k this tick
  for (int ci = 0; ci < ncand && ncell < cap; ci++){
    double cell[3]; bool ok;
    if (slot >= 0){
      int si = (e * D.NI + slot) * 64 + ci;
      ok = c.g.ig_ok[si] != 0;
      if (ok){ cell[0] = c.g.ig_cell[3*si]; cell[1] = c.g.ig_cell[3*si+1]; cell[2] = c.g.ig_cell[3*si+2]; }
    } else ok = ign_candidate(c.W, c.WP, gx, gy, gz, R, ci, cell);
    if (!ok) continue;
    if (ncell == 0){ for (int k = 0; k < D.NC; k++) c.g.fi_cok[fq * D.NC + k] = 0; c.g.fi_valid[fq] = 0; }
    int ck = fq * D.NC + ncell;
    c.g.fi_cells[3*ck] = cell[0]; c.g.fi_cells[3*ck+1] = cell[1]; c.g.fi_cells[3*ck+2] = cell[2];
    c.g.fi_cok[ck] = 1;
    ncell++;
  }
  if (!ncell) return 0;
  c.g.ctr_serial[e] += 1;
  c.g.fi_valid[fq] = 1; c.g.fi_kind[fq] = kind; c.g.fi_owner[fq] = c.g.pr_owner[q];
  c.g.fi_created[fq] = c.now; c.g.fi_expires[fq] = c.now + c.U.duration[kind]; c.g.fi_next[fq] = c.now;
  c.g.fi_serial[fq] = c.g.ctr_serial[e]; c.g.fi_id[fq] = c.g.pr_id[q]; c.g.fi_ncell[fq] = ncell;
  c.g.fi_pos[3*fq] = gx; c.g.fi_pos[3*fq+1] = gy; c.g.fi_pos[3*fq+2] = gz;
  extinguish(c);
  return c.g.fi_valid[fq] == 1 && c.g.fi_id[fq] == c.g.pr_id[q] ? c.g.fi_ncell[fq] : 0;
}

__device__ void detonate(Ctx& c, int q){
  const Dims& D = c.D; int e = c.e;
  int kind = c.g.pr_kind[q];
  double pos[3] = {c.g.pr_pos[3*q], c.g.pr_pos[3*q+1], c.g.pr_pos[3*q+2]};
  int p = q - e * D.P;
  c.ev.ev_det[q] = 1; c.ev.ev_det_id[q] = c.g.pr_id[q]; c.ev.ev_det_kind[q] = kind;
  c.ev.ev_det_pos[3*q] = pos[0]; c.ev.ev_det_pos[3*q+1] = pos[1]; c.ev.ev_det_pos[3*q+2] = pos[2];
  (void)p;
  if (kind == K_SMOKE){
    int s = -1;
    for (int j = 0; j < D.S; j++) if (!c.g.sm_valid[e * D.S + j]){ s = j; break; }
    if (s < 0){ int best = 0x7fffffff; for (int j = 0; j < D.S; j++){ int sq = e * D.S + j; if (c.g.sm_serial[sq] < best){ best = c.g.sm_serial[sq]; s = j; } } }
    int sq = e * D.S + s;
    c.g.ctr_serial[e] += 1;
    c.g.sm_valid[sq] = 1; c.g.sm_created[sq] = c.now; c.g.sm_expires[sq] = c.now + c.U.duration[K_SMOKE];
    c.g.sm_serial[sq] = c.g.ctr_serial[e]; c.g.sm_id[sq] = c.g.pr_id[q]; c.g.sm_owner[sq] = c.g.pr_owner[q];
    c.g.sm_ncv[sq] = 0;
    c.g.sm_pos[3*sq] = pos[0]; c.g.sm_pos[3*sq+1] = pos[1]; c.g.sm_pos[3*sq+2] = pos[2] + 1.3;
    extinguish(c);
  } else if (kind == K_FLASH){
    flash(c, pos, c.g.pr_owner[q]);
  } else if (kind == K_MOLO || kind == K_INC){
    if (c.g.pr_ign[q]){
      int n = ignite(c, q);
      if (n > 0){ c.ev.ev_det[q] = 4; c.ev.ev_fire_cells[q] = n; }
    } else c.ev.ev_det[q] = 3;
  } else if (kind == K_DECOY){
    int s = -1;
    for (int j = 0; j < D.D; j++) if (!c.g.dc_valid[e * D.D + j]){ s = j; break; }
    if (s < 0){ int best = 0x7fffffff; for (int j = 0; j < D.D; j++){ int dq = e * D.D + j; if (c.g.dc_serial[dq] < best){ best = c.g.dc_serial[dq]; s = j; } } }
    int dq = e * D.D + s;
    c.g.ctr_serial[e] += 1;
    c.g.dc_valid[dq] = 1; c.g.dc_owner[dq] = c.g.pr_owner[q]; c.g.dc_expires[dq] = c.now + c.U.duration[K_DECOY];
    c.g.dc_next[dq] = c.now; c.g.dc_serial[dq] = c.g.ctr_serial[e]; c.g.dc_id[dq] = c.g.pr_id[q];
    c.g.dc_pos[3*dq] = pos[0]; c.g.dc_pos[3*dq+1] = pos[1]; c.g.dc_pos[3*dq+2] = pos[2];
  } else {
    disperse(c, pos);
    explode(c, pos, c.g.pr_owner[q], kind);
  }
}

// ------------------------------------------------------------------------------------------------- fire cell probing in parallel
__global__ void ign_rank_k(GS g, Dims D, const bool* __restrict__ live){
  int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= D.E) return;
  if (!live[e]) return;
  for (int k = 0; k < D.NI; k++) g.ig_proj[e * D.NI + k] = -1;
  int k = 0;
  for (int p = 0; p < D.P; p++){
    int q = e * D.P + p;
    g.ig_slot[q] = -1;
    int kind = g.pr_kind[q];
    if ((kind == K_MOLO || kind == K_INC) && g.pr_ign[q] && k < D.NI){ g.ig_slot[q] = k; g.ig_proj[e * D.NI + k] = p; k++; }
  }
}

__global__ void ign_pre_k(GS g, WD W, WorldP WP, Dims D, UT U, const bool* __restrict__ live){
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long long)D.E * D.NI * 64) return;
  int ci = (int)(idx % 64); int ni = (int)((idx / 64) % D.NI); int e = (int)(idx / (64LL * D.NI));
  if (!live[e]) return;
  int p = g.ig_proj[e * D.NI + ni]; if (p < 0) return;
  int q = e * D.P + p;
  int kind = g.pr_kind[q];
  double R = U.radius[kind];
  int si = (e * D.NI + ni) * 64 + ci;
  g.ig_ok[si] = 0;
  if (ci >= ign_ncand(R)) return;
  double cell[3];
  if (ign_candidate(W, WP, g.pr_pos[3*q], g.pr_pos[3*q+1], g.pr_pos[3*q+2], R, ci, cell)){
    g.ig_ok[si] = 1; g.ig_cell[3*si] = cell[0]; g.ig_cell[3*si+1] = cell[1]; g.ig_cell[3*si+2] = cell[2];
  }
}

// indices of the valid entries of a table (valid[q] != 0 or, for kind tables, kind >= 0), ordered by creation serial
#define MAXN 64
__device__ int collect_sorted(const int* tab, int base, int n, bool kind_mode, const int* serial, int* out, bool descending){
  int m = 0;
  for (int j = 0; j < n && m < MAXN; j++){
    int v = tab[base + j];
    if (kind_mode ? (v >= 0) : (v != 0)){
      int sj = serial[base + j];
      int k = m++;
      while (k > 0){
        int so = serial[base + out[k - 1]];
        if (descending ? (so < sj) : (so > sj)){ out[k] = out[k - 1]; k--; } else break;
      }
      out[k] = j;
    }
  }
  return m;
}

__global__ void effects_k(GS g, PL pl, EV ev, WD W, WorldP WP, Dims D, UT U, const bool* __restrict__ live,
                          const int* __restrict__ tickv){
  int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= D.E) return;
  if (!live[e]) return;
  Ctx c; c.g = g; c.pl = pl; c.ev = ev; c.W = W; c.WP = WP; c.D = D; c.U = U; c.e = e; c.now = tickv[e];
  int now = c.now;
  int idx[MAXN];
  // 1. expired smokes, 2. expired carves (list order kept)
  for (int s = 0; s < D.S; s++){
    int sq = e * D.S + s;
    if (!g.sm_valid[sq]) continue;
    if (!(g.sm_expires[sq] > now)){ g.sm_valid[sq] = 0; g.sm_ncv[sq] = 0; continue; }
    int n = g.sm_ncv[sq], m = 0;
    for (int k = 0; k < n; k++){
      int a = sq * D.C + k;
      if ((double)(now - g.cv_created[a]) / TICK_HZ < g.cv_dur[a]){
        if (m != k){
          int b = sq * D.C + m;
          g.cv_type[b] = g.cv_type[a]; g.cv_created[b] = g.cv_created[a]; g.cv_radius[b] = g.cv_radius[a];
          g.cv_dur[b] = g.cv_dur[a]; g.cv_rec[b] = g.cv_rec[a];
          for (int t = 0; t < 3; t++){ g.cv_a[3*b+t] = g.cv_a[3*a+t]; g.cv_b[3*b+t] = g.cv_b[3*a+t]; }
        }
        m++;
      }
    }
    g.sm_ncv[sq] = m;
  }
  // 3. fires: drop the expired ones.  (GrenadeSystem._extinguish() runs here on the CPU too, but cells can only be removed when a smoke or a
  //    fire is created, which both call extinguish() themselves, so at this point it only filters the expiry.)
  for (int f = 0; f < D.F; f++){
    int q = e * D.F + f;
    if (g.fi_valid[q] && !(g.fi_expires[q] > now)) g.fi_valid[q] = 0;
  }
  // 4. fire damage (list order)
  {
    int n = collect_sorted(g.fi_valid, e * D.F, D.F, false, g.fi_serial, idx, false);
    for (int a = 0; a < n; a++){
      int q = e * D.F + idx[a];
      if (now >= g.fi_next[q]){ g.fi_next[q] = now + FIRE_DAMAGE_TICKS; burn(c, q); }
    }
  }
  // 5. decoys (reverse list order)
  {
    int n = collect_sorted(g.dc_valid, e * D.D, D.D, false, g.dc_serial, idx, true);
    for (int a = 0; a < n; a++){
      int q = e * D.D + idx[a];
      if (now >= g.dc_expires[q]){
        ev.ev_decoy_exp[e] += 1;
        double pos[3] = {g.dc_pos[3*q], g.dc_pos[3*q+1], g.dc_pos[3*q+2]};
        explode(c, pos, g.dc_owner[q], K_DECOY);
        g.dc_valid[q] = 0;
      } else if (now >= g.dc_next[q]){
        g.dc_next[q] = now + DECOY_PULSE_TICKS; ev.ev_decoy_pulse[e] += 1;
      }
    }
  }
  // 6. projectiles (reverse list order); advance_k already moved them
  {
    int n = collect_sorted(g.pr_kind, e * D.P, D.P, true, g.pr_serial, idx, true);
    for (int a = 0; a < n; a++){
      int q = e * D.P + idx[a];
      double x = g.pr_pos[3*q], y = g.pr_pos[3*q+1], z = g.pr_pos[3*q+2];
      int kind = g.pr_kind[q];
      if (!isfinite(x + y + z) || z < -250){ ev.ev_det[q] = 2; ev.ev_det_id[q] = g.pr_id[q]; ev.ev_det_kind[q] = kind; g.pr_kind[q] = -1; continue; }
      bool delayed = (kind == K_SMOKE || kind == K_DECOY);
      if (g.pr_ign[q] || (now >= g.pr_det[q] && (!delayed || g.pr_rest[q]))){
        detonate(c, q); g.pr_kind[q] = -1;
      } else if (now - g.pr_born[q] > MAX_FLIGHT_TICKS){
        ev.ev_det[q] = 2; ev.ev_det_id[q] = g.pr_id[q]; ev.ev_det_kind[q] = kind; g.pr_kind[q] = -1;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------- bullets / smoke test
__device__ double pseg(const double* p, const double* s1, const double* s2){
  double dx = s2[0] - s1[0], dy = s2[1] - s1[1], dz = s2[2] - s1[2];
  double L = dx*dx + dy*dy + dz*dz;
  double t = 0.0;
  if (L > 0){ t = ((p[0] - s1[0]) * dx + (p[1] - s1[1]) * dy + (p[2] - s1[2]) * dz) / L; t = fmax(0.0, fmin(1.0, t)); }
  double ax = p[0] - (s1[0] + t * dx), ay = p[1] - (s1[1] + t * dy), az = p[2] - (s1[2] + t * dz);
  return sqrt(ax*ax + ay*ay + az*az);
}

__global__ void carve_k(GS g, Dims D, UT U, const double* __restrict__ p0, const double* __restrict__ p1,
                        const long* __restrict__ order, const long* __restrict__ start, const int* __restrict__ tickv){
  int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= D.E) return;
  int now = tickv[e];
  Ctx c; c.g = g; c.D = D; c.U = U; c.e = e; c.now = now;
  for (long r = start[e]; r < start[e + 1]; r++){
    long b = order[r];
    const double* a = p0 + 3 * b; const double* z = p1 + 3 * b;
    for (int s = 0; s < D.S; s++){
      int sq = e * D.S + s;
      if (!g.sm_valid[sq] || g.sm_expires[sq] <= now) continue;
      if (pseg(g.sm_pos + 3 * sq, a, z) < U.radius[K_SMOKE] + 0.24){
        carve_push(c, sq);
        int k = sq * D.C + g.sm_ncv[sq];
        g.cv_type[k] = CV_BULLET; g.cv_created[k] = now; g.cv_radius[k] = 0.24; g.cv_dur[k] = 0.7; g.cv_rec[k] = 0.25;
        for (int t = 0; t < 3; t++){ g.cv_a[3*k+t] = a[t]; g.cv_b[3*k+t] = z[t]; }
        g.sm_ncv[sq] += 1;
        if (g.sm_ncv[sq] > BULLET_CARVE_MAX){
          // pop(0)
          int n = g.sm_ncv[sq] + 0;  // shift down by one
          for (int kk = 1; kk < n; kk++){
            int a0 = sq * D.C + kk - 1, a1 = sq * D.C + kk;
            g.cv_type[a0] = g.cv_type[a1]; g.cv_created[a0] = g.cv_created[a1]; g.cv_radius[a0] = g.cv_radius[a1];
            g.cv_dur[a0] = g.cv_dur[a1]; g.cv_rec[a0] = g.cv_rec[a1];
            for (int t = 0; t < 3; t++){ g.cv_a[3*a0+t] = g.cv_a[3*a1+t]; g.cv_b[3*a0+t] = g.cv_b[3*a1+t]; }
          }
          g.sm_ncv[sq] = n - 1;
        }
      }
    }
  }
}

__device__ bool seg_smoke(const GS& g, const Dims& D, const UT& U, int sq, const double* a, const double* b, int now){
  double dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
  double L = dx*dx + dy*dy + dz*dz;
  if (L < 1e-6) return false;
  const double* sp = g.sm_pos + 3 * sq;
  double sx = sp[0], sy = sp[1], sz = sp[2];
  double tc = fmax(0.0, fmin(1.0, ((sx - a[0]) * dx + (sy - a[1]) * dy + (sz - a[2]) * dz) / L));
  double ex = a[0] + tc * dx - sx, ey = a[1] + tc * dy - sy, ez = a[2] + tc * dz - sz;
  double cd = sqrt(ex*ex + ey*ey + ez*ez);
  int created = g.sm_created[sq];
  double age = (now && created) ? fmax(0.0, (double)(now - created) / TICK_HZ) : 10.0;
  double prog = fmin(1.0, age / 1.8);
  double br = U.radius[K_SMOKE] * (0.15 + 0.85 * (1 - pow(1 - prog, 3.0)));
  if (cd >= br) return false;
  int n = g.sm_ncv[sq];
  if (!n || !now) return true;
  double fx = a[0] - sx, fy = a[1] - sy, fz = a[2] - sz;
  double pA = L;
  double pB = 2 * (fx * dx + fy * dy + fz * dz);
  double pC = (fx*fx + fy*fy + fz*fz) - br * br;
  double disc = pB * pB - 4 * pA * pC;
  if (disc <= 0) return false;
  double sq_ = sqrt(disc);
  double t0 = fmax(0.0, (-pB - sq_) / (2 * pA)), t1 = fmin(1.0, (-pB + sq_) / (2 * pA));
  if (t1 <= t0) return false;
  for (int step = 0; step < 5; step++){
    double st = t0 + (double)step / 4 * (t1 - t0);
    double p[3] = {a[0] + st * dx, a[1] + st * dy, a[2] + st * dz};
    bool carved = false;
    for (int k = 0; k < n && !carved; k++){
      int ci = sq * D.C + k;
      double cage = (double)(now - g.cv_created[ci]) / TICK_HZ;
      double dur = g.cv_dur[ci];
      if (cage < 0 || cage >= dur) continue;
      double rec = g.cv_rec[ci], rad = g.cv_radius[ci];
      double full = dur - rec;
      if (g.cv_type[ci] == CV_EXPL){
        double eff;
        if (cage < full) eff = rad;
        else { double w = (cage - full) / fmax(0.01, rec); eff = rad * fmax(0.0, 1 - pow(w, 2.0)); }
        if (eff > 0.001 && dist3(p, g.cv_a + 3 * ci) <= eff) carved = true;
      } else {
        double eff = cage < full ? rad : rad * fmax(0.0, 1 - (cage - full) / fmax(0.01, rec));
        if (eff > 0.001 && pseg(p, g.cv_a + 3 * ci, g.cv_b + 3 * ci) <= eff) carved = true;
      }
    }
    if (!carved) return true;
  }
  return false;
}

__global__ void smoke_blocks_k(GS g, Dims D, UT U, const double* __restrict__ p0, const double* __restrict__ p1,
                               const long* __restrict__ env, int R, const int* __restrict__ tickv, bool* __restrict__ out){
  int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= R) return;
  int e = (int)env[r]; int now = tickv[e];
  bool blk = false;
  for (int s = 0; s < D.S && !blk; s++){
    int sq = e * D.S + s;
    if (!g.sm_valid[sq] || !(g.sm_expires[sq] > now)) continue;
    if (seg_smoke(g, D, U, sq, p0 + 3 * r, p1 + 3 * r, now)) blk = true;
  }
  out[r] = blk;
}

// ------------------------------------------------------------------------------------------- host wrappers
static Dims mkdims(std::vector<int64_t> d){ Dims D; D.E=d[0]; D.P=d[1]; D.S=d[2]; D.C=d[3]; D.F=d[4]; D.NC=d[5]; D.D=d[6]; D.NI=d[7]; return D; }
static UT mkut(torch::Tensor ui, torch::Tensor ud){
  UT U; auto a = ui.cpu(); auto b = ud.cpu();
  for (int k = 0; k < 6; k++){ U.fuse[k] = a[0][k].item<int>(); U.duration[k] = a[1][k].item<int>();
    U.radius[k] = b[0][k].item<double>(); U.damage[k] = b[1][k].item<double>(); }
  return U;
}
static WorldP mkwp(std::vector<double> w){ WorldP P; P.type=(int)w[0]; P.nb=(int)w[1]; P.ground=w[2]; P.gmin0=w[3]; P.gmin1=w[4];
  P.gmax0=w[5]; P.gmax1=w[6]; P.csox=w[7]; P.csoy=w[8]; return P; }

void throw_call(std::vector<torch::Tensor> gs, std::vector<torch::Tensor> pl, std::vector<torch::Tensor> ev,
                std::vector<torch::Tensor> wd, std::vector<double> wp, std::vector<int64_t> dims,
                torch::Tensor ui, torch::Tensor ud, torch::Tensor ak, torch::Tensor as, torch::Tensor am,
                torch::Tensor mask, torch::Tensor tick){
  Dims D = mkdims(dims);
  throw_k<<<(D.E + 63) / 64, 64, 0, at::cuda::getCurrentCUDAStream()>>>(make_GS(gs), make_PL(pl), make_EV(ev), make_WD(wd), mkwp(wp), D, mkut(ui, ud),
      ak.data_ptr<int>(), as.data_ptr<int>(), am.data_ptr<int>(), mask.data_ptr<bool>(), tick.data_ptr<int>());
}
void step_call(std::vector<torch::Tensor> gs, std::vector<torch::Tensor> pl, std::vector<torch::Tensor> ev,
               std::vector<torch::Tensor> wd, std::vector<double> wp, std::vector<int64_t> dims,
               torch::Tensor ui, torch::Tensor ud, torch::Tensor live, torch::Tensor tick){
  Dims D = mkdims(dims); UT U = mkut(ui, ud); GS g = make_GS(gs); EV e = make_EV(ev); WD W = make_WD(wd); WorldP P = mkwp(wp);
  int n = D.E * D.P;
  advance_k<<<(n + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(g, e, W, P, D, live.data_ptr<bool>(), tick.data_ptr<int>());
  if (D.NI > 0){
    ign_rank_k<<<(D.E + 63) / 64, 64, 0, at::cuda::getCurrentCUDAStream()>>>(g, D, live.data_ptr<bool>());
    long long m = (long long)D.E * D.NI * 64;
    ign_pre_k<<<(int)((m + 63) / 64), 64, 0, at::cuda::getCurrentCUDAStream()>>>(g, W, P, D, U, live.data_ptr<bool>());
  }
  effects_k<<<(D.E + 63) / 64, 64, 0, at::cuda::getCurrentCUDAStream()>>>(g, make_PL(pl), e, W, P, D, U, live.data_ptr<bool>(), tick.data_ptr<int>());
}
void carve_call(std::vector<torch::Tensor> gs, std::vector<int64_t> dims, torch::Tensor ui, torch::Tensor ud,
                torch::Tensor p0, torch::Tensor p1, torch::Tensor order, torch::Tensor start, torch::Tensor tick){
  Dims D = mkdims(dims);
  carve_k<<<(D.E + 63) / 64, 64, 0, at::cuda::getCurrentCUDAStream()>>>(make_GS(gs), D, mkut(ui, ud), p0.data_ptr<double>(), p1.data_ptr<double>(),
      order.data_ptr<long>(), start.data_ptr<long>(), tick.data_ptr<int>());
}
torch::Tensor smoke_call(std::vector<torch::Tensor> gs, std::vector<int64_t> dims, torch::Tensor ui, torch::Tensor ud,
                         torch::Tensor p0, torch::Tensor p1, torch::Tensor env, torch::Tensor tick){
  Dims D = mkdims(dims); int R = p0.size(0);
  auto out = torch::zeros({R}, p0.options().dtype(torch::kBool));
  if (R) smoke_blocks_k<<<(R + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(make_GS(gs), D, mkut(ui, ud), p0.data_ptr<double>(), p1.data_ptr<double>(),
      env.data_ptr<long>(), R, tick.data_ptr<int>(), out.data_ptr<bool>());
  return out;
}
// world queries for tests: contact(o, d, L) per row -> [R,4] (dist or nan, normal)
__global__ void contact_k(WD W, WorldP P, const double* o, const double* d, const double* L, int R, double* out){
  int r = blockIdx.x * blockDim.x + threadIdx.x; if (r >= R) return;
  double dist, n[3];
  if (contact(W, P, o + 3*r, d + 3*r, L[r], dist, n)){ out[4*r] = dist; out[4*r+1] = n[0]; out[4*r+2] = n[1]; out[4*r+3] = n[2]; }
  else { out[4*r] = nan(""); out[4*r+1] = out[4*r+2] = out[4*r+3] = 0.0; }
}
torch::Tensor contact_call(std::vector<torch::Tensor> wd, std::vector<double> wp, torch::Tensor o, torch::Tensor d, torch::Tensor L){
  int R = o.size(0); auto out = torch::empty({R, 4}, o.options());
  if (R) contact_k<<<(R + 127) / 128, 128, 0, at::cuda::getCurrentCUDAStream()>>>(make_WD(wd), mkwp(wp), o.data_ptr<double>(), d.data_ptr<double>(), L.data_ptr<double>(), R, out.data_ptr<double>());
  return out;
}
'''

_CPP = '''
void throw_call(std::vector<torch::Tensor> gs, std::vector<torch::Tensor> pl, std::vector<torch::Tensor> ev, std::vector<torch::Tensor> wd, std::vector<double> wp, std::vector<int64_t> dims, torch::Tensor ui, torch::Tensor ud, torch::Tensor ak, torch::Tensor as, torch::Tensor am, torch::Tensor mask, torch::Tensor tick);
void step_call(std::vector<torch::Tensor> gs, std::vector<torch::Tensor> pl, std::vector<torch::Tensor> ev, std::vector<torch::Tensor> wd, std::vector<double> wp, std::vector<int64_t> dims, torch::Tensor ui, torch::Tensor ud, torch::Tensor live, torch::Tensor tick);
void carve_call(std::vector<torch::Tensor> gs, std::vector<int64_t> dims, torch::Tensor ui, torch::Tensor ud, torch::Tensor p0, torch::Tensor p1, torch::Tensor order, torch::Tensor start, torch::Tensor tick);
torch::Tensor smoke_call(std::vector<torch::Tensor> gs, std::vector<int64_t> dims, torch::Tensor ui, torch::Tensor ud, torch::Tensor p0, torch::Tensor p1, torch::Tensor env, torch::Tensor tick);
torch::Tensor contact_call(std::vector<torch::Tensor> wd, std::vector<double> wp, torch::Tensor o, torch::Tensor d, torch::Tensor L);
'''

_mod = None


def smoke_device_src():
    """Device source of the smoke line test (constants, Dims / UT, dist3, pseg, seg_smoke) for other extensions that test segments
    against the GpuGrenades smokes inline (gpusim/vis.py).  The caller adds the GS struct (_struct('GS', GS_FIELDS)) before it and
    compiles with -fmad=false like this module, so the arithmetic is the same as smoke_blocks_k."""
    k = _KERNELS
    defs = k[k.index('#define NSEAT'):k.index('struct WorldP')]
    d3 = k[k.index('__device__ __forceinline__ double dist3('):]
    d3 = d3[:d3.index('\n}\n') + 3]
    ps = k[k.index('__device__ double pseg('):k.index('__global__ void carve_k')]
    ss = k[k.index('__device__ bool seg_smoke('):k.index('__global__ void smoke_blocks_k')]
    return defs + d3 + ps + ss


def ext():
    global _mod
    if _mod is None:
        os.environ.setdefault('TORCH_EXTENSIONS_DIR', './fps_data/tmp/torch_ext')
        head = '#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_runtime.h>\n#include <math.h>\n'
        structs = _struct('GS', GS_FIELDS) + _struct('PL', PL_FIELDS) + _struct('EV', EV_FIELDS) + _struct('WD', WD_FIELDS)
        _mod = load_inline('fpsgren', cpp_sources=_CPP, cuda_sources=head + structs + _KERNELS,
                           functions=['throw_call', 'step_call', 'carve_call', 'smoke_call', 'contact_call'],
                           extra_cuda_cflags=['-O3', '-fmad=false'], verbose=False)
    return _mod
