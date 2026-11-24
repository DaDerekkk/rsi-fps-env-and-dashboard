// Player movement on a triangle mesh: CUDA port of fpssim/meshmove.py MeshMover (cs2-web stepPlayer + HullContact SAT), and the
// player-vs-player pass of fpssim/arena.py (Arena._resolve_players).  Kernels:
//   gather_kernel   thread per player: candidate triangles of the tick from the 2 m grid (as MeshWorld.gather)
//   step_kernel     warp per 32 players, converged: stepPlayer with warp-cooperative SAT / sweep (see below)
//   resolve_kernel  thread per match: two passes over the player pairs (+ optional wall check)
// Everything is float64 (float32 only as a conservative broad-phase filter), compiled with -fmad=false so that every arithmetic
// expression rounds exactly like the numpy / CPython reference (same operand order everywhere; see docs/gpu_move.md).
// Constants (MV_*) are #defined by gpusim/move.py from the Python reference values.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define TS 84            // doubles per triangle record
// record layout: 0-2 tlo, 3-5 thi, 6-8 leaf lo, 9-11 leaf hi, 12-14 surface normal, 15 valid-axis mask,
//                16+3k axis k (k=0..12), 55+k pmin[k], 68+k pmax[k]
#define O_LLO 6
#define O_LHI 9
#define O_SURF 12
#define O_MASK 15
#define O_AX 16
#define O_PMIN 55
#define O_PMAX 68

#define ERR_KMAX 1
#define ERR_CELLS 2
#define ERR_MAXC 4

struct World {
  const double* tri; const float4* tbox; const int* flat; const int* start;   // float32: tbox[3t] = (tlo, n.x), tbox[3t+1] = (thi, n.y), tbox[3t+2] = (n.z, pmin0, pmax0, 0), n = surface normal = SAT axis 0
  int nx, nz; double gx0, gz0, cell;
  double B[9], Bt[9], o[3];
  int kmax;                      // capacity of one candidate list (cbuf is [kmax, P])
  int nlad; const double* lad;   // per ladder: rail x, rail y, bottom, top, exit x, exit y
  double reach, climb_speed;
};
struct SP {   // state (in/out)
  double *x, *y, *z, *vx, *vy, *vz, *yaw, *pitch, *jbuf, *sdist, *height;
  bool *crouch, *grounded, *last_jump, *alive, *blocked;
  int8_t* mode; int16_t* lid; int32_t* reatt;
};
struct AP {   // held action + per-player inputs
  const double *fwd, *rgt, *yawr, *pitchr, *climb, *M;
  const bool *jump, *crouch, *walk, *locked, *freeze; const int8_t* ladder;
  const int32_t* tick; int tick_div;   // per-match (or per-player) tick: tick[i / tick_div]; nullptr: use the scalar argument
};

#define D_INF (__longlong_as_double(0x7ff0000000000000LL))
#ifdef MV_COUNT      // profiling build (GPUMOVE_COUNT=1): event counters
__device__ unsigned long long g_cnt[16];
__device__ long long g_wc[1 << 16]; __device__ int g_wn[1 << 16], g_wr[1 << 16], g_ws[1 << 16], g_wh[1 << 16];   // per warp: cycles, max N, SAT requests, sweep requests, hull calls
#define CNT(i) atomicAdd(&g_cnt[i], 1ull)
__device__ unsigned long long g_ph[8][1 << 13];
#define PH_INIT() long long ph_last = clock64()
#define PH(k) do { long long ph_now = clock64(); if ((threadIdx.x & 31) == 0) g_ph[k][(blockIdx.x * blockDim.x + threadIdx.x) >> 5] += ph_now - ph_last; ph_last = clock64(); } while (0)
#define WCNT(arr) do { if ((threadIdx.x & 31) == 0) atomicAdd(&arr[(blockIdx.x * blockDim.x + threadIdx.x) >> 5], 1); } while (0)
#else
#define CNT(i)
#define WCNT(arr)
#define PH_INIT()
#define PH(k)
#endif

// ---------------------------------------------------------------------------------------------- CPython helpers
__device__ __forceinline__ double py_min(double a, double b) { return (b < a) ? b : a; }   // min(a, b)
__device__ __forceinline__ double py_max(double a, double b) { return (b > a) ? b : a; }   // max(a, b)

// math.hypot(x, y) of CPython 3.12 (vector_norm: lossless scaled squares + one differential correction)
__device__ double py_hypot(double x, double y) {
  x = fabs(x); y = fabs(y);
  if (isinf(x) || isinf(y)) return D_INF;
  if (isnan(x) || isnan(y)) return x + y;
  double mx = 0.0;
  if (x > mx) mx = x;
  if (y > mx) mx = y;
  if (mx == 0.0) return mx;
  int e; frexp(mx, &e);
  if (e < -1023) return sqrt(x * x + y * y);   // subnormal range: irrelevant here
  double scale = ldexp(1.0, -e);
  double csum = 1.0, frac1 = 0.0, frac2 = 0.0, v[2] = {x, y};
  for (int i = 0; i < 2; i++) {
    double q = v[i] * scale;
    double ph = q * q, pl = fma(q, q, -ph);
    double sh = csum + ph, sl = (csum - sh) + ph;
    csum = sh; frac1 += pl; frac2 += sl;
  }
  double h = sqrt(csum - 1.0 + (frac1 + frac2));
  double ph = (-h) * h, pl = fma(-h, h, -ph);
  double sh = csum + ph, sl = (csum - sh) + ph;
  csum = sh; frac1 += pl; frac2 += sl;
  double q = csum - 1.0 + (frac1 + frac2);
  h += q / (2.0 * h);
  return h / scale;
}

// sin/cos correctly rounded via double-double evaluation (CUDA sin/cos are up to 2 ulp off; glibc's, used by
// CPython, are correctly rounded in practice).  Valid for moderate |x| (the yaw is wrapped to [-pi, pi)).
struct dd { double hi, lo; };
__device__ __forceinline__ dd two_sum(double a, double b) { double s = a + b, bb = s - a; return {s, (a - (s - bb)) + (b - bb)}; }
__device__ __forceinline__ dd qsum(double a, double b) { double s = a + b; return {s, b - (s - a)}; }
__device__ __forceinline__ dd dd_add(dd a, dd b) {
  dd s = two_sum(a.hi, b.hi), t = two_sum(a.lo, b.lo);
  s.lo += t.hi; s = qsum(s.hi, s.lo); s.lo += t.lo; return qsum(s.hi, s.lo);
}
__device__ __forceinline__ dd dd_mul(dd a, dd b) {
  double p = a.hi * b.hi, e = fma(a.hi, b.hi, -p);
  e += a.hi * b.lo + a.lo * b.hi; return qsum(p, e);
}
__device__ __forceinline__ dd dd_muld(dd a, double b) {
  double p = a.hi * b, e = fma(a.hi, b, -p);
  e += a.lo * b; return qsum(p, e);
}
__constant__ double c_sin[2 * MV_NTAYLOR] = MV_SIN_INIT;   // (hi, lo) of (-1)^k/(2k+1)!
__constant__ double c_cos[2 * MV_NTAYLOR] = MV_COS_INIT;   // (hi, lo) of (-1)^k/(2k)!
__constant__ double c_pio2[3] = MV_PIO2_INIT;               // pi/2 = p0 + p1 + p2
__device__ void dd_sincos(double x, double& so, double& co) {
  double k = rint(x * 0.6366197723675814);
  dd r = two_sum(x, -(k * c_pio2[0]));
  { double p = k * c_pio2[0]; r.lo -= fma(k, c_pio2[0], -p); r = qsum(r.hi, r.lo); }
  r = dd_add(r, dd_muld({-c_pio2[1], 0.0}, k));
  r = dd_add(r, dd_muld({-c_pio2[2], 0.0}, k));
  dd r2 = dd_mul(r, r);
  // Horner from the highest term: terms k >= 5 are below 3e-8 of the result, plain double is enough there (error < 1e-23)
  double ts = c_sin[2 * (MV_NTAYLOR - 1)], tc = c_cos[2 * (MV_NTAYLOR - 1)];
  for (int j = MV_NTAYLOR - 2; j >= 5; j--) { ts = c_sin[2 * j] + ts * r2.hi; tc = c_cos[2 * j] + tc * r2.hi; }
  dd ps = {ts, 0.0}, pc = {tc, 0.0};
  for (int j = 4; j >= 0; j--) {
    ps = dd_add(dd_mul(ps, r2), dd{c_sin[2 * j], c_sin[2 * j + 1]});
    pc = dd_add(dd_mul(pc, r2), dd{c_cos[2 * j], c_cos[2 * j + 1]});
  }
  ps = dd_mul(ps, r);
  double s = ps.hi + ps.lo, c = pc.hi + pc.lo;
  int q = ((int)k) & 3;
  if (q == 0) { so = s; co = c; } else if (q == 1) { so = c; co = -s; }
  else if (q == 2) { so = -s; co = -c; } else { so = -c; co = s; }
}

// Python float a % b and a // b
__device__ double py_mod(double a, double b) {
  double m = fmod(a, b);
  if (m != 0.0) { if ((b < 0) != (m < 0)) m += b; }
  else m = copysign(0.0, b);
  return m;
}
__device__ double py_floordiv(double a, double b) {
  double m = fmod(a, b);
  double div = (a - m) / b;
  if (m != 0.0) { if ((b < 0) != (m < 0)) { m += b; div -= 1.0; } }
  double fd;
  if (div != 0.0) { fd = floor(div); if (div - fd > 0.5) fd += 1.0; }
  else fd = copysign(0.0, a / b);
  return fd;
}
__device__ __forceinline__ double py_wrap(double a) {
  const double PI = 3.141592653589793, TWO_PI = 6.283185307179586;
  return py_mod(a + PI, TWO_PI) - PI;
}

// ---------------------------------------------------------------------------------------------- SAT
struct Box { double mn[3], mx[3]; };

// float32 broad-phase filter.  The FP64 pipe of these GPUs runs at 1/64 of the FP32 rate and a warp pays for every lane, so the
// per-candidate tests of the long loops are done with float32 compares of the triangle AABB (float32 values already: the mesh
// is a float32 file) against a conservatively rounded float32 copy of the box (outward, +1e-7 m, far above the ~1e-13 m
// rounding of the FP64 SAT it replaces).  A candidate that passes is then re-tested with the original FP64 code, so the result
// is the same as without the filter.
struct FB { float lo[3], hi[3]; };
__device__ __forceinline__ void fb_make(const double* mn, const double* mx, FB& f) {
#pragma unroll
  for (int i = 0; i < 3; i++) { f.lo[i] = __double2float_rd(mn[i] - 1e-7); f.hi[i] = __double2float_ru(mx[i] + 1e-7); }
}
__device__ __forceinline__ bool fb_hit(const float4& tl, const float4& th, const FB& f) {
  return (tl.x <= f.hi[0]) & (th.x >= f.lo[0]) & (tl.y <= f.hi[1]) & (th.y >= f.lo[1]) & (tl.z <= f.hi[2]) & (th.z >= f.lo[2]);
}

// float32 test of SAT axis 0 (the triangle's own plane): true if the box is separated from the triangle plane by more than 1e-3 m,
// so that the exact FP64 SAT would reject it as well (float32 error of the projections is < 1e-4 m at 150 m).  Only a shortcut
// for the cooperative SAT, which would return "no contact" for such a candidate.
struct FC { float c[3], h[3]; };
__device__ __forceinline__ void fc_make(const Box& b, FC& f) {
#pragma unroll
  for (int i = 0; i < 3; i++) { f.c[i] = (float)((b.mn[i] + b.mx[i]) * .5); f.h[i] = (float)((b.mx[i] - b.mn[i]) * .5); }
}
__device__ __forceinline__ bool fsep(const float4& tl, const float4& th, const float4& t2, const FC& f) {
  const float cc = tl.w * f.c[0] + th.w * f.c[1] + t2.x * f.c[2];
  const float r = fabsf(tl.w) * f.h[0] + fabsf(th.w) * f.h[1] + fabsf(t2.x) * f.h[2];
  return (t2.z - cc + r < -1e-3f) || (cc + r - t2.y < -1e-3f);
}

__device__ __forceinline__ void cenhf(const Box& b, double* c, double* h) {
#pragma unroll
  for (int i = 0; i < 3; i++) { c[i] = (b.mn[i] + b.mx[i]) * .5; h[i] = (b.mx[i] - b.mn[i]) * .5; }
}
__device__ __forceinline__ void proj(const double* t, int k, const double* c, const double* h, double& pos, double& neg) {
  const double* a = t + O_AX + 3 * k;
  double cc = a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
  double r = fabs(a[0]) * h[0] + fabs(a[1]) * h[1] + fabs(a[2]) * h[2];
  pos = t[O_PMAX + k] - cc + r;
  neg = cc + r - t[O_PMIN + k];
}
// Local.overlap_mask for one triangle (no separating axis with overlap <= 1e-8).  The boolean does not depend on the axis
// order, so the three world axes are tested first from the triangle AABB (identical numbers: pmin/pmax of those axes
// are the vertex min/max and c, r reduce exactly to cen, hf).
__device__ __forceinline__ bool overlap(const double* __restrict__ t, const double* c, const double* h) {
#pragma unroll
  for (int i = 0; i < 3; i++) {
    double pos = t[3 + i] - c[i] + h[i];
    double neg = c[i] + h[i] - t[i];
    if (pos <= 1e-8 || neg <= 1e-8) return false;
  }
  int mask = (int)t[O_MASK];
  for (int k = 0; k < 13; k++) {
    if (k >= 1 && k <= 3) continue;
    if (!((mask >> k) & 1)) continue;
    double pos, neg; proj(t, k, c, h, pos, neg);
    if (pos <= 1e-8 || neg <= 1e-8) return false;
  }
  return true;
}
// ---------------------------------------------------------------------------------------------- warp-cooperative SAT
// The FP64 pipe of these GPUs issues one warp instruction per 16 cycles whatever the number of active lanes, and the per-player
// work is dominated by the 13-axis SAT of a few triangles per tick that every lane needs at a different time.  Executed one lane
// at a time it keeps ~2 of the 32 lanes busy.  Here the whole warp stays converged: all lanes scan their own candidate lists
// together and, whenever lanes need a SAT (a triangle vs. a box), the warp handles those requests one after the other with the
// 13 axes spread over 13 lanes (one axis each, the same expressions as the scalar code), reduced with ballots/shuffles.  All
// functions below with the suffix _w must be called by all 32 lanes of a warp together (act = false: the lane only helps).
#define FULL 0xffffffffu
#define SB 8                      // candidates per scan batch
__device__ __forceinline__ double shfl_d(double v, int src) { return __shfl_sync(FULL, v, src); }
__device__ __forceinline__ unsigned warp_max_u32(unsigned v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __reduce_max_sync(FULL, v);
#else
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    unsigned o = __shfl_xor_sync(FULL, v, off);
    if (o > v) v = o;
  }
  return v;
#endif
}

// Local.contact tail: normal / (flipped) surface / walkable from the chosen axis and sign
struct Ct { double n[3], s[3], depth; bool walk; };
__device__ __forceinline__ void ct_from_axis(const double* __restrict__ t, int& best, bool neg_sign, double depth, Ct& ct, bool& flip) {
  double sign = neg_sign ? -1.0 : 1.0;
  if (best < 0) best = 12;     // python axes[j, -1]; unreachable (world axes are always valid)
  const double* a = t + O_AX + 3 * best;
  ct.n[0] = a[0] * sign; ct.n[1] = a[1] * sign; ct.n[2] = a[2] * sign;
  ct.s[0] = t[O_SURF]; ct.s[1] = t[O_SURF + 1]; ct.s[2] = t[O_SURF + 2];
  flip = ct.s[0] * ct.n[0] + ct.s[1] * ct.n[1] + ct.s[2] * ct.n[2] < 0;
  if (flip) { ct.s[0] = -ct.s[0]; ct.s[1] = -ct.s[1]; ct.s[2] = -ct.s[2]; }
  ct.depth = depth; ct.walk = (ct.s[1] >= .7 && ct.n[1] >= .7);
}

// contact record for collide(): triangle id + axis/sign/flip/walk bits (normal and surface are rebuilt exactly)
struct CRec { int tri; int bits; };
__device__ __forceinline__ void crec_vec(const double* __restrict__ TR, const CRec& r, double* n, double* s, bool& walk) {
  const double* t = TR + (size_t)r.tri * TS;
  int best = r.bits & 15; double sign = (r.bits & 16) ? -1.0 : 1.0;
  const double* a = t + O_AX + 3 * best;
  n[0] = a[0] * sign; n[1] = a[1] * sign; n[2] = a[2] * sign;
  s[0] = t[O_SURF]; s[1] = t[O_SURF + 1]; s[2] = t[O_SURF + 2];
  if (r.bits & 32) { s[0] = -s[0]; s[1] = -s[1]; s[2] = -s[2]; }
  walk = (r.bits & 64) != 0;
}

struct Hit { double n[3]; double depth; bool walk; bool deep; int nc; };
struct Cx { const double* TR; const float4* TB; const int* cand; int cs; int N; };    // candidate list of one player-tick
struct SatOut { bool ov; int best; bool neg; double depth; };

// order-preserving int64 key of a double (a < b  <=>  dkey(a) < dkey(b) for non-NaN values), its own inverse as `k ^ ((k >> 63) & 0x7fff..)`
__device__ __forceinline__ long long dkey(double d) { long long b = __double_as_longlong(d); return b ^ ((b >> 63) & 0x7fffffffffffffffLL); }

// Local.overlap_mask + Local.contact for the requests of all lanes (need = this lane has a request: triangle id vs box c/h).
// out.ov: no separating axis (every overlap > 1e-8); then out.best/neg/depth = minimum-overlap axis in the fixed axis order with
// the 1e-7 tie rule (axis 0 = surface normal, 1..3 = y, x, z, 4..12 edge crosses), sign, and overlap.
// Two requests are served per round: lanes 0-15 take the first pending lane, lanes 16-31 the second, 13 lanes (= axes) each.
__device__ __forceinline__ void coop_sat(bool need, const double* __restrict__ TR, int id, const double* c, const double* h, SatOut& out) {
  out.ov = false; out.best = -1; out.neg = false; out.depth = 0.0;
  const int lane = threadIdx.x & 31, g = lane >> 4, gl = lane & 15;
  unsigned m = __ballot_sync(FULL, need);
  while (m) {
    const int L0 = __ffs(m) - 1; m &= m - 1;
    const bool has1 = m != 0;
    const int L1 = has1 ? __ffs(m) - 1 : L0; if (has1) m &= m - 1;
    WCNT(g_wr);
    const int myL = g ? L1 : L0;
    const bool act = (g == 0) || has1;
    const int idL = __shfl_sync(FULL, id, myL);
    double cc3[3], hh3[3];
#pragma unroll
    for (int q = 0; q < 3; q++) { cc3[q] = shfl_d(c[q], myL); hh3[q] = shfl_d(h[q], myL); }
    const double* t = TR + (size_t)idL * TS;
    const int mask = (int)t[O_MASK];
    const bool valid = act && gl < 13 && ((mask >> gl) & 1);
    double pos = 1.0, neg = 1.0;
    if (valid) {
      const double* a = t + O_AX + 3 * gl;
      double cc = a[0] * cc3[0] + a[1] * cc3[1] + a[2] * cc3[2];
      double r = fabs(a[0]) * hh3[0] + fabs(a[1]) * hh3[1] + fabs(a[2]) * hh3[2];
      pos = t[O_PMAX + gl] - cc + r;
      neg = cc + r - t[O_PMIN + gl];
    }
    const bool sep = valid && (pos <= 1e-8 || neg <= 1e-8);
    const unsigned sepm = __ballot_sync(FULL, sep), vm = __ballot_sync(FULL, valid);
    const bool gsep = ((sepm >> (16 * g)) & 0xffffu) != 0;
    const unsigned vmg = (vm >> (16 * g)) & 0xffffu;
    bool gov = act && !gsep;
    double depth = D_INF; int best = -1; int sneg = 0;
    if (__any_sync(FULL, gov)) {
      // `ov < depth - 1e-7` with depth = the overlap of the last axis that won: threshold per axis precomputed, order-preserving
      // int64 keys instead of FP64 compares (the FP64 pipe is the scarce resource)
      const double ov = valid ? (pos < neg ? pos : neg) : D_INF;
      const double th = ov - 1e-7;
      const int ng = !(pos < neg);
      const long long ovk = dkey(ov), thk = dkey(th);
      long long thr = dkey(D_INF);
      long long dk = 0;
#pragma unroll
      for (int k = 0; k < 13; k++) {
        const long long o = __shfl_sync(FULL, ovk, k, 16), tk = __shfl_sync(FULL, thk, k, 16);
        const int kn = __shfl_sync(FULL, ng, k, 16);
        if (((vmg >> k) & 1) && o < thr) { dk = o; thr = tk; best = k; sneg = kn; }
      }
      depth = best >= 0 ? __longlong_as_double(dk ^ ((dk >> 63) & 0x7fffffffffffffffLL)) : D_INF;
    }
    // owners fetch their group's result (lane 0 / lane 16 hold the group's values)
    const int src = (lane == L0) ? 0 : 16;
    const int r_ov = __shfl_sync(FULL, (int)gov, src), r_best = __shfl_sync(FULL, best, src), r_neg = __shfl_sync(FULL, sneg, src);
    const double r_depth = shfl_d(depth, src);
    if (lane == L0 || (has1 && lane == L1)) { out.ov = r_ov != 0; out.best = r_best; out.neg = r_neg != 0; out.depth = r_depth; }
  }
}

// MapCollision.hullIntersect over the candidate list (ascending slot = BVH visit order), warp-cooperative
__device__ bool hull_w(bool act, const Cx& X, const Box& b0, Hit& H, CRec* rec, bool record, int* err) {
  const double* __restrict__ TR = X.TR; const float4* __restrict__ TB = X.TB; const int* __restrict__ cand = X.cand;
  const int cs = X.cs, N = act ? X.N : 0;
  H.nc = 0; H.walk = false; H.deep = false; H.depth = 0.0;
  const int Nmax = (int)warp_max_u32((unsigned)N);
  if (Nmax == 0) return false;
  CNT(0); WCNT(g_wh);
  Box r = b0;
  double c0[3], h0[3], c[3], h[3];
  cenhf(b0, c0, h0); cenhf(r, c, h);
  FB fr, f0; fb_make(b0.mn, b0.mx, f0); fr = f0;
  FC gr, g0; fc_make(b0, g0); gr = g0;
  bool any = false, walk = false, deep = false; int nc = 0;
  // The candidates are scanned in batches of SB: all their loads are issued together (the per-candidate chain id -> box is a
  // global-memory round trip each), a batch in which no lane has a candidate near its box (the usual case) costs one vote.
  for (int k0 = 0; k0 < Nmax; k0 += SB) {
    int ids[SB]; float4 tls[SB], ths[SB], t2s[SB];
    bool bn = false;
#pragma unroll
    for (int j = 0; j < SB; j++) ids[j] = (k0 + j < N) ? cand[(size_t)(k0 + j) * cs] : 0;
#pragma unroll
    for (int j = 0; j < SB; j++) {
      const bool in = k0 + j < N;
      tls[j] = in ? TB[3 * (size_t)ids[j]] : make_float4(0, 0, 0, 0); ths[j] = in ? TB[3 * (size_t)ids[j] + 1] : make_float4(0, 0, 0, 0);
      t2s[j] = in ? TB[3 * (size_t)ids[j] + 2] : make_float4(0, 0, 0, 0);
      bn = bn || (in && (fb_hit(tls[j], ths[j], fr) || (any && fb_hit(tls[j], ths[j], f0))));
    }
    if (!__any_sync(FULL, bn)) continue;
#pragma unroll 1
    for (int j = 0; j < SB; j++) {
    const bool in = k0 + j < N;
    const int id = ids[j]; const float4 tl = tls[j], th = ths[j], t2 = t2s[j];
    const bool nearA = in && fb_hit(tl, th, fr) && !fsep(tl, th, t2, gr);
    SatOut sa;
    coop_sat(nearA, TR, id, c, h, sa);
    if (sa.ov) {
      const double* t = TR + (size_t)id * TS;
      Ct ct; bool fl; int best = sa.best;
      ct_from_axis(t, best, sa.neg, sa.depth, ct, fl);
      walk = walk || ct.walk; if (ct.depth > .0003) deep = true;
      if (record) {
        if (nc < MV_MAXC) { rec[nc].tri = id; rec[nc].bits = best | (sa.neg ? 16 : 0) | (fl ? 32 : 0) | (ct.walk ? 64 : 0); nc++; }
        else atomicOr(err, ERR_MAXC);
      }
      any = true;
      double d = ct.depth + .00001;
#pragma unroll
      for (int i = 0; i < 3; i++) { double tt = ct.n[i] * d; r.mn[i] += tt; r.mx[i] += tt; }
      cenhf(r, c, h); fb_make(r.mn, r.mx, fr); fc_make(r, gr);
    }
    // support contacts: triangles visited between pushes, tested against the ORIGINAL box, if their BVH leaf is visited
    bool needB = false;
    if (in && !sa.ov && any && fb_hit(tl, th, f0) && !fsep(tl, th, t2, g0)) {
      const double* t = TR + (size_t)id * TS;
      bool vis = true;
#pragma unroll
      for (int i = 0; i < 3; i++) {
        double blo = r.mn[i] - .002, bhi = r.mx[i] + .002;
        vis = vis && (t[O_LLO + i] <= bhi) && (t[O_LHI + i] >= blo);
      }
      needB = vis;
    }
    SatOut sb;
    coop_sat(needB, TR, id, c0, h0, sb);
    if (sb.ov) {
      const double* t = TR + (size_t)id * TS;
      Ct ct; bool fl; int best = sb.best;
      ct_from_axis(t, best, sb.neg, sb.depth, ct, fl);
      if (ct.walk) {
        walk = true;
        if (record) {
          if (nc < MV_MAXC) { rec[nc].tri = id; rec[nc].bits = best | (sb.neg ? 16 : 0) | (fl ? 32 : 0) | 64; nc++; }
          else atomicOr(err, ERR_MAXC);
        }
      }
    }
    }
  }
  double dn0 = r.mn[0] - b0.mn[0], dn1 = r.mn[1] - b0.mn[1], dn2 = r.mn[2] - b0.mn[2];
  double depth = sqrt(dn0 * dn0 + dn1 * dn1 + dn2 * dn2);
  H.nc = nc; H.walk = walk; H.deep = deep;
  if (act && depth > 1e-7) {
    double inv = 1 / depth;
    H.n[0] = dn0 * inv; H.n[1] = dn1 * inv; H.n[2] = dn2 * inv; H.depth = depth;
    return true;
  }
  return false;
}

// MapCollision.sweepDown, warp-cooperative: distance to the first walkable support below the box within max_drop
__device__ bool sweep_w(bool act, const Cx& X, const Box& b, double max_drop, double& dist_out) {
  const double* __restrict__ TR = X.TR; const float4* __restrict__ TB = X.TB; const int* __restrict__ cand = X.cand;
  const int cs = X.cs, N = act ? X.N : 0;
  const int Nmax = (int)warp_max_u32((unsigned)N);
  if (Nmax == 0) return false;
  CNT(6);
  const int lane = threadIdx.x & 31;
  double blo[3] = {b.mn[0], b.mn[1] - max_drop, b.mn[2]};
  FB fs; fb_make(blo, b.mx, fs);     // necessary for the exact AABB test below (float32 filter, see fb_hit)
  double c[3], h[3]; cenhf(b, c, h);
  double distance = D_INF; bool walkable = false;
  const double nmd = -max_drop;
  for (int k0 = 0; k0 < Nmax; k0 += SB) {
    int ids[SB]; float4 tls[SB], ths[SB];
    bool bn = false;
#pragma unroll
    for (int j = 0; j < SB; j++) ids[j] = (k0 + j < N) ? cand[(size_t)(k0 + j) * cs] : 0;
#pragma unroll
    for (int j = 0; j < SB; j++) {
      const bool in = k0 + j < N;
      tls[j] = in ? TB[3 * (size_t)ids[j]] : make_float4(0, 0, 0, 0); ths[j] = in ? TB[3 * (size_t)ids[j] + 1] : make_float4(0, 0, 0, 0);
      bn = bn || (in && fb_hit(tls[j], ths[j], fs));
    }
    if (!__any_sync(FULL, bn)) continue;
#pragma unroll 1
    for (int j = 0; j < SB; j++) {
    const bool in = k0 + j < N;
    const int id = ids[j]; const float4 tl = tls[j], th = ths[j];
    bool sel = false;
    if (in && fb_hit(tl, th, fs)) {
      const double* t = TR + (size_t)id * TS;
      sel = true;
#pragma unroll
      for (int i = 0; i < 3; i++)
        sel = sel && (t[O_LLO + i] <= b.mx[i]) && (t[O_LHI + i] >= blo[i]) && (t[i] <= b.mx[i]) && (t[3 + i] >= blo[i]);
    }
    unsigned m = __ballot_sync(FULL, sel);
    while (m) {
      const int L0 = __ffs(m) - 1; m &= m - 1;
      const bool has1 = m != 0;
      const int L1 = has1 ? __ffs(m) - 1 : L0; if (has1) m &= m - 1;
      WCNT(g_ws);
      const int g = lane >> 4, gl = lane & 15;
      const int myL = g ? L1 : L0;
      const bool actg = (g == 0) || has1;
      const int idL = __shfl_sync(FULL, id, myL);
      double cc3[3], hh3[3];
#pragma unroll
      for (int q = 0; q < 3; q++) { cc3[q] = shfl_d(c[q], myL); hh3[q] = shfl_d(h[q], myL); }
      const double* t = TR + (size_t)idL * TS;
      const int mask = (int)t[O_MASK];
      const bool valid = actg && gl < 13 && ((mask >> gl) & 1);
      double first = -D_INF, last = D_INF, nyj = 0.0; bool kill = false, mv = false;
      if (valid) {
        const double* ax = t + O_AX + 3 * gl;
        double cc = ax[0] * cc3[0] + ax[1] * cc3[1] + ax[2] * cc3[2];
        double r = fabs(ax[0]) * hh3[0] + fabs(ax[1]) * hh3[1] + fabs(ax[2]) * hh3[2];
        double lo = t[O_PMIN + gl] - r - cc;
        double hi = t[O_PMAX + gl] + r - cc;
        double sp = nmd * ax[1];
        if (fabs(sp) < 1e-10) { kill = (lo >= -1e-8 || hi <= 1e-8); }
        else {
          double a1 = lo / sp, a2 = hi / sp;
          first = fmin(a1, a2); last = fmax(a1, a2); mv = true;
          nyj = sp < 0 ? ax[1] : -ax[1];
        }
      }
      const unsigned killm = __ballot_sync(FULL, kill);
      double en = first, ex = last;
#pragma unroll
      for (int off = 8; off > 0; off >>= 1) {          // within the 16-lane group
        double o1 = __shfl_xor_sync(FULL, en, off, 16); en = o1 > en ? o1 : en;
        double o2 = __shfl_xor_sync(FULL, ex, off, 16); ex = o2 < ex ? o2 : ex;
      }
      ex = fmin(1.0, ex);
      // the sequential loop keeps the FIRST axis that reached the maximum "enter" (strict > update)
      const unsigned am = (__ballot_sync(FULL, mv && first == en) >> (16 * g)) & 0xffffu;
      const double ny = __shfl_sync(FULL, nyj, am ? __ffs(am) - 1 : 0, 16);
      const bool gkill = ((killm >> (16 * g)) & 0xffffu) != 0;
      // lane L0 / L1 take the result of their group (lane 0 / lane 16 hold it)
      const int src = (lane == L0) ? 0 : 16;
      const double r_en = shfl_d(en, src), r_ex = shfl_d(ex, src), r_ny = shfl_d(ny, src);
      const int r_kill = __shfl_sync(FULL, (int)gkill, src);
      if (lane == L0 || (has1 && lane == L1)) {
        const double* tt = TR + (size_t)id * TS;
        const bool alive = !r_kill && !(r_en > r_ex + 1e-8);
        if (alive && !(r_en < -.00001 || r_en > 1 || r_ex < 0)) {
          double d = fmax(0.0, r_en) * max_drop;
          bool wk = (tt[O_SURF + 1] >= .7) && (r_ny >= .7);
          if (d < distance - .00001) { distance = d; walkable = wk; }
          else if (fabs(d - distance) <= .00001) walkable = walkable || wk;
        }
      }
    }
    }
  }
  if (act && distance < D_INF && walkable) { dist_out = distance; return true; }
  return false;
}

// ---------------------------------------------------------------------------------------------- player tick
struct PS { double x, y, z, vx, vy, vz, yaw, height, jbuf; bool grounded, crouch, last_jump; };

__device__ __forceinline__ Box pbox(const PS& p) {
  const double R = MV_R; Box b;
  b.mn[0] = p.x - R; b.mn[1] = p.y; b.mn[2] = p.z - R;
  b.mx[0] = p.x + R; b.mx[1] = p.y + p.height; b.mx[2] = p.z + R;
  return b;
}
__device__ __forceinline__ void tr(Box& b, double dx, double dy, double dz) {
  b.mn[0] = b.mn[0] + dx; b.mn[1] = b.mn[1] + dy; b.mn[2] = b.mn[2] + dz;
  b.mx[0] = b.mx[0] + dx; b.mx[1] = b.mx[1] + dy; b.mx[2] = b.mx[2] + dz;
}

// MeshMover._update_crouch for the lanes with run && a stance change; returns stand-blocked.  All lanes call it together.
__device__ bool update_crouch_w(bool run, const Cx& X, PS& p, bool wanted, int* err) {
  const bool ch = run && (wanted != p.crouch);
  if (!__any_sync(FULL, ch)) return false;
  double nh = 0.0, y = 0.0; Box tb = pbox(p);
  if (ch) {
    double oh = p.crouch ? MV_CROUCH_H : MV_STAND_H; nh = wanted ? MV_CROUCH_H : MV_STAND_H;
    double d = oh - nh;
    double lift = d != 0 ? copysign(1.0, d) * py_min(fabs(d), MV_CJLIFT) : 0.0;
    y = p.y + (p.grounded ? 0.0 : lift);
    const double R = MV_R;
    tb.mn[0] = p.x - R; tb.mn[1] = y; tb.mn[2] = p.z - R; tb.mx[0] = p.x + R; tb.mx[1] = y + nh; tb.mx[2] = p.z + R;
    if (!wanted) tr(tb, 0.0, .0002, 0.0);
  }
  const bool test = ch && !wanted;
  Hit H; bool hit = hull_w(test, X, tb, H, nullptr, false, err);
  const bool blk = test && hit && H.deep;
  if (ch && !blk) { p.y = y; p.crouch = wanted; p.height = nh; }
  return blk;
}

// meshmove.MeshWorld.gather into cand[k*cs] (k-way merge of the ascending per-cell lists, dedup, AABB filter)
__device__ int gather(const World& W, const double* lo, const double* hi, int* cand, int cs, int* err) {
  const double c = W.cell;
  double fx0 = py_floordiv(lo[0] - W.gx0, c), fx1 = py_floordiv(hi[0] - W.gx0, c);
  double fz0 = py_floordiv(lo[2] - W.gz0, c), fz1 = py_floordiv(hi[2] - W.gz0, c);
  // clamp in double before converting (python ints are unbounded)
  int ix0 = (int)py_max(0.0, py_min(fx0, 1e9)), ix1 = (int)py_min((double)(W.nx - 1), py_max(fx1, -1e9));
  int iz0 = (int)py_max(0.0, py_min(fz0, 1e9)), iz1 = (int)py_min((double)(W.nz - 1), py_max(fz1, -1e9));
  if (ix1 < ix0 || iz1 < iz0) return 0;
  int ncell = (ix1 - ix0 + 1) * (iz1 - iz0 + 1);
  if (ncell > MV_MAXCELLS) { atomicOr(err, ERR_CELLS); ncell = 0; }
  int ptr[MV_MAXCELLS], end[MV_MAXCELLS];
  int q = 0;
  for (int ix = ix0; ix <= ix1 && q < ncell; ix++)
    for (int iz = iz0; iz <= iz1; iz++) { int cid = ix * W.nz + iz; ptr[q] = W.start[cid]; end[q] = W.start[cid + 1]; q++; }
  // the AABB filter (t.lo <= hi && t.hi >= lo, exact) in float32: the triangle bounds are float32 values, so comparing them to
  // hi rounded down / lo rounded up gives the identical answer to the FP64 comparison
  float fhi[3], flo[3];
#pragma unroll
  for (int i = 0; i < 3; i++) { fhi[i] = __double2float_rd(hi[i]); flo[i] = __double2float_ru(lo[i]); }
  int n = 0;
  while (true) {
    int m = 0x7fffffff;
    for (int j = 0; j < ncell; j++) if (ptr[j] < end[j]) { int v = W.flat[ptr[j]]; if (v < m) m = v; }
    if (m == 0x7fffffff) break;
    for (int j = 0; j < ncell; j++) if (ptr[j] < end[j] && W.flat[ptr[j]] == m) ptr[j]++;
    float4 tl = W.tbox[3 * (size_t)m], th = W.tbox[3 * (size_t)m + 1];
    bool ok = (tl.x <= fhi[0]) & (th.x >= flo[0]) & (tl.y <= fhi[1]) & (th.y >= flo[1]) & (tl.z <= fhi[2]) & (th.z >= flo[2]);
    CNT(11);
    if (ok) {
      CNT(10);
      if (n < W.kmax) cand[(size_t)n * cs] = m;
      else atomicOr(err, ERR_KMAX);
      n++;
    }
  }
  return n < W.kmax ? n : W.kmax;
}

// candidate triangles of the whole tick (meshmove.MeshMover.step: box around the player grown by the tick's travel), per player
__global__ void gather_kernel(World W, SP S, AP A, int P, int* __restrict__ cbuf, int* __restrict__ nout, int* err) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P) return;
  int N = 0;
  if (S.alive[i] && S.mode[i] != 2) {
    const bool frz = A.freeze && A.freeze[i];
    const double R = MV_R, dt = MV_DT;
    double d0 = S.x[i] - W.o[0], d1 = S.y[i] - W.o[1], d2 = S.z[i] - W.o[2];
    double py = W.Bt[3] * d0 + W.Bt[4] * d1 + W.Bt[5] * d2, px = W.Bt[0] * d0 + W.Bt[1] * d1 + W.Bt[2] * d2, pz = W.Bt[6] * d0 + W.Bt[7] * d1 + W.Bt[8] * d2;
    double v0 = frz ? 0.0 : S.vx[i], v1 = frz ? 0.0 : S.vy[i], v2 = S.vz[i];
    double vx = W.Bt[0] * v0 + W.Bt[1] * v1 + W.Bt[2] * v2, vy = W.Bt[3] * v0 + W.Bt[4] * v1 + W.Bt[5] * v2, vz = W.Bt[6] * v0 + W.Bt[7] * v1 + W.Bt[8] * v2;
    double sp = sqrt(vx * vx + vz * vz);
    double mh = sp * dt + 0.45, vdn = py_max(0.0, -vy) * dt + 1.2, vup = py_max(0.0, vy) * dt + 0.7;
    double lo[3] = {px - R - mh, py - vdn, pz - R - mh};
    double hi[3] = {px + R + mh, py + MV_STAND_H + vup, pz + R + mh};
    N = gather(W, lo, hi, cbuf + (size_t)i * W.kmax, 1, err);
  }
  nout[i] = N;
}

// One warp = 32 players that stay converged (see the cooperative SAT above): per-player scalar code is predicated by `run`, the
// calls that need the whole warp (hull_w / sweep_w / update_crouch_w) are made by every lane, guarded only by warp-uniform
// conditions (__any_sync).  Lanes that are dead / on a ladder / past the end of the array just help.
__global__ void __launch_bounds__(MV_BLOCK) step_kernel(World W, SP S, AP A, int P, int tick0, const int* __restrict__ nin,
                                                       int* __restrict__ cbuf, int* err) {
  const int t_ = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = t_ < P ? t_ : 0;
#ifdef MV_COUNT
  const long long t_start = clock64();
#endif
  const bool alive = t_ < P && S.alive[i];
  bool done = !alive;
  PH_INIT();
  const double dt = MV_DT;
  const int tick = (alive && A.tick) ? A.tick[i / A.tick_div] : tick0;
  if (alive) CNT(12);
  // buy-phase freeze (Arena._tick_inner): vx = vy = 0, no forward/right input, no jump
  const bool frz = alive && A.freeze && A.freeze[i];
  double fwd_raw = 0.0, fwd = 0.0, rgt = 0.0, yaw = 0.0;
  bool jump = false, want_crouch = false, walk = false, locked = false;
  if (alive) {
    if (frz) { S.vx[i] = 0.0; S.vy[i] = 0.0; }
    fwd_raw = frz ? 0.0 : A.fwd[i];
    fwd = py_max(-1.0, py_min(1.0, fwd_raw));
    rgt = py_max(-1.0, py_min(1.0, frz ? 0.0 : A.rgt[i]));
    jump = A.jump[i] && !frz; want_crouch = A.crouch[i]; walk = A.walk[i]; locked = A.locked[i];
    yaw = py_wrap(S.yaw[i] + A.yawr[i] * MV_YAW_RATE * dt);
    S.yaw[i] = yaw;
    S.pitch[i] = py_max(-MV_PITCH_LIM, py_min(MV_PITCH_LIM, S.pitch[i] + A.pitchr[i] * MV_PITCH_RATE * dt));
    const int8_t lad = A.ladder[i];
    // ---------------- ladders (movement.Mover._ladder_step / _try_attach, world coordinates)
    if (S.mode[i] == 2) {
      int id = S.lid[i];
      double climb = A.climb[i];
      if (climb == 0.0) climb = fwd_raw;
      climb = py_max(-1.0, py_min(1.0, climb));
      S.vx[i] = 0.0; S.vy[i] = 0.0;
      if (id < 0 || id >= W.nlad || lad == 2) {
        S.mode[i] = 1; S.lid[i] = -1; S.grounded[i] = false; S.vz[i] = 0.0;
        S.reatt[i] = tick + MV_REATTACH;
      } else {
        const double* L = W.lad + 6 * id;
        double vz = climb >= 0 ? climb * W.climb_speed : climb * W.climb_speed * 1.5;
        S.vz[i] = vz;
        double z = py_min(L[3], S.z[i] + vz * dt);
        S.z[i] = z;
        if (z <= L[2] + 1e-9) {
          S.x[i] = L[4]; S.y[i] = L[5]; S.z[i] = L[2] + 0.00011;
          S.mode[i] = 0; S.lid[i] = -1; S.vz[i] = 0.0; S.grounded[i] = true;
          S.reatt[i] = tick + MV_REATTACH;
        }
      }
      done = true;
    } else if (lad == 1 && tick >= S.reatt[i] && W.nlad > 0) {
      int best = -1; double bd = 0.0;
      double sx = S.x[i], sy = S.y[i], sz = S.z[i];
      for (int l = 0; l < W.nlad; l++) {
        const double* L = W.lad + 6 * l;
        double d = py_hypot(sx - L[0], sy - L[1]);
        if (d <= W.reach && L[2] - 0.3 <= sz && sz <= L[3] + 0.1) {
          if (best < 0 || d < bd) { best = l; bd = d; }
        }
      }
      if (best >= 0) {
        const double* L = W.lad + 6 * best;
        S.x[i] = L[0]; S.y[i] = L[1];
        S.vx[i] = 0.0; S.vy[i] = 0.0; S.vz[i] = 0.0;
        S.mode[i] = 2; S.lid[i] = best; S.grounded[i] = false;
        done = true;
      }
    }
  }
  // ---------------- to raw (cs2-web, y up) coordinates, candidate triangles for the whole tick
  PS p;
  p.x = p.y = p.z = p.vx = p.vy = p.vz = p.jbuf = 0.0; p.yaw = yaw; p.height = MV_STAND_H;
  p.grounded = p.crouch = p.last_jump = false;
  double x0w = 0.0, y0w = 0.0;
  int* cand = cbuf + (size_t)i * W.kmax; const int cs = 1;
  int N = 0;
  if (!done) {
    x0w = S.x[i]; y0w = S.y[i];
    double d0 = S.x[i] - W.o[0], d1 = S.y[i] - W.o[1], d2 = S.z[i] - W.o[2];
    p.x = W.Bt[0] * d0 + W.Bt[1] * d1 + W.Bt[2] * d2;
    p.y = W.Bt[3] * d0 + W.Bt[4] * d1 + W.Bt[5] * d2;
    p.z = W.Bt[6] * d0 + W.Bt[7] * d1 + W.Bt[8] * d2;
    double v0 = S.vx[i], v1 = S.vy[i], v2 = S.vz[i];
    p.vx = W.Bt[0] * v0 + W.Bt[1] * v1 + W.Bt[2] * v2;
    p.vy = W.Bt[3] * v0 + W.Bt[4] * v1 + W.Bt[5] * v2;
    p.vz = W.Bt[6] * v0 + W.Bt[7] * v1 + W.Bt[8] * v2;
    p.grounded = S.grounded[i]; p.crouch = S.crouch[i];
    p.height = p.crouch ? MV_CROUCH_H : MV_STAND_H;
    p.last_jump = S.last_jump[i]; p.jbuf = S.jbuf[i];
    N = nin[i];                       // gather_kernel: same box, same list
    if (locked) {
      p.vx = 0.0; p.vy = 0.0; p.vz = 0.0;
      p.crouch = true; p.height = MV_CROUCH_H;
      p.jbuf = 0.0; p.last_jump = jump;
    }
  }
  const Cx X = {W.tri, W.tbox, cand, cs, N};
  bool blocked = false;
  const bool run = !done && !locked;
  if (__any_sync(FULL, run)) {
    bool jumped = false;
    double incoming = 0.0;
#define TRY_JUMP() do { if (p.grounded && p.jbuf > 0) { p.vy = MV_JUMP_V; p.grounded = false; p.jbuf = 0.0; jumped = true; } } while (0)
    if (run) {
      if (p.vy > .1) p.grounded = false;
      bool pressed = jump && !p.last_jump;
      if (pressed) p.jbuf = MV_JBUF_T;
      p.last_jump = jump;
      p.jbuf = py_max(0.0, p.jbuf);
      TRY_JUMP();
    }
    PH(0);
    blocked = update_crouch_w(run, X, p, want_crouch, err);
    PH(1);
    if (run) {
      if (p.grounded && !jumped) {
        double speed = py_hypot(p.vx, p.vz);
        if (speed < .00254) { p.vx = 0.0; p.vz = 0.0; }
        else {
          double control = speed < MV_STOP ? MV_STOP : speed;
          double drop = control * MV_FRIC * dt;
          double ns = py_max(0.0, speed - drop);
          if (ns != speed) { double f = ns / speed; p.vx *= f; p.vz *= f; }
        }
      }
      double mag = py_hypot(fwd, rgt);
      if (mag > 0) {
        double sy, cy; dd_sincos(p.yaw, sy, cy);
        double wdx = (-sy * fwd + cy * rgt) / mag;
        double wdz = (-cy * fwd - sy * rgt) / mag;
        double base = A.M[i];
        double fac = p.crouch ? MV_CROUCH_F : (walk ? MV_WALK_F : 1.0);
        double wish = fac * base * py_min(1.0, mag);
        if (p.grounded && !jumped) {
          double cur = p.vx * wdx + p.vz * wdz;
          double add = wish - cur;
          if (add > 0) { double acc = py_min(add, MV_ACC * base * dt); p.vx += acc * wdx; p.vz += acc * wdz; }
        } else {
          double aw = py_min(wish, MV_AIRCAP);
          double cur = p.vx * wdx + p.vz * wdz;
          double add = aw - cur;
          if (add > 0) { double acc = py_min(add, MV_AIRACC * wish * dt); p.vx += acc * wdx; p.vz += acc * wdz; }
        }
      }
      incoming = py_hypot(p.vx, p.vz);
    }
    // ---------------- substeps
    const double hsub = MV_HSUB;
    for (int sub = 0; sub < MV_NSUB; sub++) {
      bool was_g = false; double hs = 0.0, mx_ = 0.0, mz_ = 0.0;
      Box c = pbox(p), before = c;
      if (run) {
        TRY_JUMP();
        was_g = p.grounded;
        hs = py_hypot(p.vx, p.vz);
        p.vy -= MV_GRAV * hsub * .5;
        c = pbox(p); before = c;
        mx_ = p.vx * hsub; mz_ = p.vz * hsub; double my_ = p.vy * hsub;
        tr(c, mx_, my_, mz_);
      }
      PH(2);
      Hit H; CRec rec[MV_MAXC];
      const bool hit = hull_w(run, X, c, H, rec, true, err);     // also the first pass of collide() unless the step-up replaces c
      PH(3);
      bool reuse = true;
      bool ground = false;
      const bool stepc = run && hit && H.depth * py_hypot(H.n[0], H.n[2]) > .0001 && was_g && !jumped && py_hypot(mx_, mz_) > .001;
      if (__any_sync(FULL, stepc)) {
        Box st = before;
        if (stepc) tr(st, mx_, MV_STEP_H, mz_);
        Hit H2; const bool hit2 = hull_w(stepc, X, st, H2, nullptr, false, err);
        const bool c2 = stepc && !hit2;
        if (__any_sync(FULL, c2)) {
          Box sup = st; double dd = 0.0;
          const bool okS = sweep_w(c2, X, sup, MV_STEP_DROP, dd);       // sweep_snap
          if (c2 && okS) {
            tr(sup, 0.0, -py_max(0.0, dd - .0001), 0.0);
            if (sup.mn[1] >= before.mn[1] - .001) { c = sup; p.vy = 0.0; ground = true; reuse = false; }
          }
        }
      }
      PH(4);
      // collide(): up to 3 hull passes, per-contact velocity projection, push out
      bool g2 = false, brk = !run;
#pragma unroll 1
      for (int it = 0; it < 3; it++) {
        const bool use0 = (it == 0) && reuse;
        const bool need = !brk && !use0;
        Hit Hc; bool hitc = false;
        if (__any_sync(FULL, need)) hitc = hull_w(need, X, c, Hc, rec, true, err);
        if (!brk && use0) { Hc = H; hitc = hit; }
        if (!brk && !hitc) brk = true;
        if (!brk) {
          if (Hc.walk && p.vy <= .1) { g2 = true; p.vy = 0.0; }
          for (int q = 0; q < Hc.nc; q++) {
            double n[3], s[3]; bool wk;
            crec_vec(X.TR, rec[q], n, s, wk);
            if (wk && p.vy <= .1) { g2 = true; p.vy = 0.0; }
            else {
              const double* u = wk ? s : n;
              double dot = p.vx * u[0] + p.vy * u[1] + p.vz * u[2];
              if (dot < 0) { p.vx -= u[0] * dot; p.vy -= u[1] * dot; p.vz -= u[2] * dot; }
            }
          }
          double t = Hc.depth + 0.0001;
          tr(c, Hc.n[0] * t, Hc.n[1] * t, Hc.n[2] * t);
        }
      }
      ground = g2 || ground;
      PH(5);
      const bool snapc = run && !ground && was_g && !jumped && p.vy <= 0;
      if (__any_sync(FULL, snapc)) {
        Box sup = c; double dd = 0.0;
        const bool okS = sweep_w(snapc, X, sup, MV_STEP_H, dd);
        if (snapc && okS) { tr(sup, 0.0, -py_max(0.0, dd - .0001), 0.0); c = sup; ground = true; }
      }
      PH(6);
      if (run) {
        p.x = (c.mn[0] + c.mx[0]) / 2; p.y = c.mn[1]; p.z = (c.mn[2] + c.mx[2]) / 2;
        p.grounded = ground;
        if (ground) {
          p.vy = 0.0;
          double a2 = py_hypot(p.vx, p.vz);
          if (a2 > hs && a2 > 0) { double k = hs / a2; p.vx *= k; p.vz *= k; }
        } else {
          p.vy -= MV_GRAV * hsub * .5;
        }
        p.jbuf = py_max(0.0, p.jbuf - hsub);
      }
    }
#undef TRY_JUMP
    if (run && p.grounded) {
      double sp2 = py_hypot(p.vx, p.vz);
      if (sp2 > incoming && sp2 > 0) { double k = incoming / sp2; p.vx *= k; p.vz *= k; }
    }
  }
#ifdef MV_COUNT
  { int nm = (int)warp_max_u32((unsigned)N); if ((threadIdx.x & 31) == 0) { int w = t_ >> 5; g_wc[w] = clock64() - t_start; g_wn[w] = nm; } }
#endif
  // ---------------- write back (MeshMover._write)
  if (!done) {
    double wx = W.B[0] * p.x + W.B[1] * p.y + W.B[2] * p.z + W.o[0];
    double wy = W.B[3] * p.x + W.B[4] * p.y + W.B[5] * p.z + W.o[1];
    double wz = W.B[6] * p.x + W.B[7] * p.y + W.B[8] * p.z + W.o[2];
    S.x[i] = wx; S.y[i] = wy; S.z[i] = wz;
    S.vx[i] = W.B[0] * p.vx + W.B[1] * p.vy + W.B[2] * p.vz;
    S.vy[i] = W.B[3] * p.vx + W.B[4] * p.vy + W.B[5] * p.vz;
    S.vz[i] = W.B[6] * p.vx + W.B[7] * p.vy + W.B[8] * p.vz;
    S.grounded[i] = p.grounded; S.crouch[i] = p.crouch; S.last_jump[i] = p.last_jump; S.jbuf[i] = p.jbuf;
    S.sdist[i] = S.sdist[i] + py_hypot(wx - x0w, wy - y0w);
    if (S.height) S.height[i] = p.height;
    S.blocked[i] = blocked;
    if (!locked) S.mode[i] = p.grounded ? 0 : 1;
  }
}

__global__ void hypot_kernel(const double* a, const double* b, double* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = py_hypot(a[i], b[i]);
}
__global__ void trig_kernel(const double* a, double* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { dd_sincos(a[i], out[2 * i], out[2 * i + 1]); }
}

// ---------------------------------------------------------------------------------------------- player vs player
// Arena._resolve_players (cs2-web resolveAllPlayerCollisions): two passes over all pairs of live players in seat order, one thread
// per match.  World coordinates (z up).  See docs/gpu_move.md.
#define MAXP 16
struct CS { double *x, *y, *z, *vx, *vy, *vz; const double* yaw; bool* grounded; const bool *crouch, *alive, *locked; };

// BoxWorld.blockers(x0,x1,y0,y1,zlo,zhi) != empty, on the triangle mesh: does any triangle overlap the world box
// [x+-HW] x [y+-HW] x [zlo, zhi], shrunk by BoxWorld.EPS (1e-6) on every side (less the SAT's own 1e-8 slack)?
__device__ bool blocked_box(const World& W, double xc, double yc, double zlo, double zhi, int* cand, int cs, int* err) {
  const double e = MV_WALL_SHRINK;
  double lo[3] = {xc - MV_HW + e, yc - MV_HW + e, zlo + e}, hi[3] = {xc + MV_HW - e, yc + MV_HW - e, zhi - e};
  double rlo[3] = {D_INF, D_INF, D_INF}, rhi[3] = {-D_INF, -D_INF, -D_INF};
  for (int k = 0; k < 8; k++) {            // raw = Bt (w - o); the basis is a signed permutation, so the image is a box
    double d0 = ((k & 1) ? hi[0] : lo[0]) - W.o[0], d1 = ((k & 2) ? hi[1] : lo[1]) - W.o[1], d2 = ((k & 4) ? hi[2] : lo[2]) - W.o[2];
    for (int a = 0; a < 3; a++) {
      double v = W.Bt[3 * a] * d0 + W.Bt[3 * a + 1] * d1 + W.Bt[3 * a + 2] * d2;
      rlo[a] = v < rlo[a] ? v : rlo[a]; rhi[a] = v > rhi[a] ? v : rhi[a];
    }
  }
  int N = gather(W, rlo, rhi, cand, cs, err);
  Box b; for (int a = 0; a < 3; a++) { b.mn[a] = rlo[a]; b.mx[a] = rhi[a]; }
  double c[3], h[3]; cenhf(b, c, h);
  FB fb; fb_make(b.mn, b.mx, fb);
  for (int k = 0; k < N; k++) {
    int id = cand[(size_t)k * cs];
    if (fb_hit(W.tbox[3 * (size_t)id], W.tbox[3 * (size_t)id + 1], fb) && overlap(W.tri + (size_t)id * TS, c, h)) return true;
  }
  return false;
}
// The push from (x0,y0) to (x1,y1): the hull at the new position (what BoxWorld.blockers is asked), and with `swept`, for pushes
// longer than 0.3 m, also the hulls 1/3 and 2/3 of the way.  BoxWorld's solid boxes make the new hull alone enough; a triangle surface
// has no inside, so a push that carries the hull through a wall face into the solid (only possible when it is longer than half the
// hull width: a locked partner leaves the other player the whole overlap, up to 0.7 m) would go unnoticed, and the mover cannot get a
// player out of a solid.  Shorter pushes cannot do that and are tested at the new position only, like the CPU.
__device__ bool blocked_at(const World& W, double x0, double y0, double x1, double y1, bool swept, double zlo, double zhi,
                           int* cand, int cs, int* err) {
  if (blocked_box(W, x1, y1, zlo, zhi, cand, cs, err)) return true;
  if (!swept || (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) <= 0.3 * 0.3) return false;
  if (blocked_box(W, x0 + (x1 - x0) * (1.0 / 3.0), y0 + (y1 - y0) * (1.0 / 3.0), zlo, zhi, cand, cs, err)) return true;
  return blocked_box(W, x0 + (x1 - x0) * (2.0 / 3.0), y0 + (y1 - y0) * (2.0 / 3.0), zlo, zhi, cand, cs, err);
}

__global__ void resolve_kernel(World W, CS S, int E, int NP, int wall, int swept, int* __restrict__ cbuf, int* err) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;
  const int base = e * NP;
  double X[MAXP], Y[MAXP], Z[MAXP], VX[MAXP], VY[MAXP], VZ[MAXP], Hh[MAXP];
  bool G[MAXP], LK[MAXP]; int live[MAXP]; int n = 0;
  for (int s = 0; s < NP; s++) {
    int i = base + s;
    if (!S.alive[i]) continue;
    live[n++] = s;
    X[s] = S.x[i]; Y[s] = S.y[i]; Z[s] = S.z[i]; VX[s] = S.vx[i]; VY[s] = S.vy[i]; VZ[s] = S.vz[i];
    Hh[s] = S.crouch[i] ? MV_CROUCH_H : MV_STAND_H; G[s] = S.grounded[i]; LK[s] = S.locked[i];
  }
  if (n < 2) return;
  const double R = 0.70, R2 = 0.70 * 0.70;
  for (int pass = 0; pass < 2; pass++)
    for (int ia = 0; ia < n; ia++)
      for (int ib = ia + 1; ib < n; ib++) {
        const int a = live[ia], b = live[ib];
        double dx = X[a] - X[b], dy = Y[a] - Y[b];
        double d2 = dx * dx + dy * dy;
        if (d2 >= R2) continue;
        double ta = Z[a] + Hh[a], tb = Z[b] + Hh[b];
        if (Z[a] >= tb - 0.22 && VZ[a] <= 0.1) { Z[a] = tb; G[a] = true; VZ[a] = py_max(0.0, VZ[a]); continue; }
        if (Z[b] >= ta - 0.22 && VZ[b] <= 0.1) { Z[b] = ta; G[b] = true; VZ[b] = py_max(0.0, VZ[b]); continue; }
        if (py_min(ta, tb) - py_max(Z[a], Z[b]) <= 0.05) continue;
        bool la = LK[a], lb = LK[b];
        if (la && lb) continue;
        double d = sqrt(d2), ov = R - d, nx, ny;
        if (d > 1e-5) { nx = dx / d; ny = dy / d; }
        else { double sy, cy; dd_sincos(S.yaw[base + a], sy, cy); nx = cy; ny = -sy; }
        double pa = ov * .5, pb = ov * .5, fa = .5, fb = .5;
        if (la) { pa = 0.0; pb = ov; fa = 0.0; fb = 1.0; }
        else if (lb) { pa = ov; pb = 0.0; fa = 1.0; fb = 0.0; }
        for (int k = 0; k < 2; k++) {
          const int st = k ? b : a; const double sg = k ? -1.0 : 1.0, pu = k ? pb : pa;
          double nxp = X[st] + sg * nx * pu, nyp = Y[st] + sg * ny * pu;
          bool ok = true;
          if (wall) ok = !blocked_at(W, X[st], Y[st], nxp, nyp, swept != 0, Z[st] + 0.46, Z[st] + Hh[st], cbuf + (size_t)e * W.kmax, 1, err);
          if (ok) { X[st] = nxp; Y[st] = nyp; }
        }
        double vn = (VX[a] - VX[b]) * nx + (VY[a] - VY[b]) * ny;
        if (vn < 0) {
          VX[a] -= vn * fa * nx; VY[a] -= vn * fa * ny;
          VX[b] += vn * fb * nx; VY[b] += vn * fb * ny;
        }
      }
  for (int q = 0; q < n; q++) {
    int s = live[q], i = base + s;
    S.x[i] = X[s]; S.y[i] = Y[s]; S.z[i] = Z[s]; S.vx[i] = VX[s]; S.vy[i] = VY[s]; S.vz[i] = VZ[s]; S.grounded[i] = G[s];
  }
}

// ---------------------------------------------------------------------------------------------- host wrappers
template <class T> static T* P_(const torch::Tensor& t) { return t.numel() ? (T*)t.data_ptr() : nullptr; }

// wt = [tri f64 [T,84], tbox f32 [T,8], flat i32, start i32, ladders f64 [L,6]]
static World make_world(const std::vector<torch::Tensor>& wt, const std::vector<double>& geo, int kmax) {
  World W; W.kmax = kmax;
  W.tri = (const double*)wt[0].data_ptr(); W.tbox = (const float4*)wt[1].data_ptr();
  W.flat = (const int*)wt[2].data_ptr(); W.start = (const int*)wt[3].data_ptr();
  // geo: nx, nz, gx0, gz0, cell, B[9], Bt[9], o[3], reach, climb_speed
  W.nx = (int)geo[0]; W.nz = (int)geo[1]; W.gx0 = geo[2]; W.gz0 = geo[3]; W.cell = geo[4];
  for (int k = 0; k < 9; k++) { W.B[k] = geo[5 + k]; W.Bt[k] = geo[14 + k]; }
  for (int k = 0; k < 3; k++) W.o[k] = geo[23 + k];
  W.reach = geo[26]; W.climb_speed = geo[27];
  W.nlad = (int)(wt[4].numel() / 6); W.lad = W.nlad ? (const double*)wt[4].data_ptr() : nullptr;
  return W;
}

void step(std::vector<torch::Tensor> wt, std::vector<double> geo, std::vector<torch::Tensor> st, std::vector<torch::Tensor> ac,
          torch::Tensor tick_t, int64_t tick_div, torch::Tensor cbuf, torch::Tensor nbuf, torch::Tensor err, int64_t tick, int64_t kmax) {
  c10::cuda::CUDAGuard guard(st[0].device());
  World W = make_world(wt, geo, (int)kmax);
  SP S;
  S.x = P_<double>(st[0]); S.y = P_<double>(st[1]); S.z = P_<double>(st[2]);
  S.vx = P_<double>(st[3]); S.vy = P_<double>(st[4]); S.vz = P_<double>(st[5]);
  S.yaw = P_<double>(st[6]); S.pitch = P_<double>(st[7]); S.jbuf = P_<double>(st[8]); S.sdist = P_<double>(st[9]);
  S.height = P_<double>(st[10]);
  S.crouch = P_<bool>(st[11]); S.grounded = P_<bool>(st[12]); S.last_jump = P_<bool>(st[13]); S.alive = P_<bool>(st[14]);
  S.blocked = P_<bool>(st[15]); S.mode = P_<int8_t>(st[16]); S.lid = P_<int16_t>(st[17]); S.reatt = P_<int32_t>(st[18]);
  AP A;
  A.fwd = P_<double>(ac[0]); A.rgt = P_<double>(ac[1]); A.yawr = P_<double>(ac[2]); A.pitchr = P_<double>(ac[3]);
  A.climb = P_<double>(ac[4]); A.M = P_<double>(ac[5]);
  A.jump = P_<bool>(ac[6]); A.crouch = P_<bool>(ac[7]); A.walk = P_<bool>(ac[8]); A.locked = P_<bool>(ac[9]);
  A.freeze = P_<bool>(ac[10]); A.ladder = P_<int8_t>(ac[11]);
  A.tick = P_<int32_t>(tick_t); A.tick_div = (int)tick_div;
  int P = (int)st[0].numel();
  if (P == 0) return;
  int bs = MV_BLOCK;
  auto stream = c10::cuda::getCurrentCUDAStream();
  gather_kernel<<<(P + 127) / 128, 128, 0, stream>>>(W, S, A, P, (int*)cbuf.data_ptr(), (int*)nbuf.data_ptr(), (int*)err.data_ptr());
  step_kernel<<<(P + bs - 1) / bs, bs, 0, stream>>>(W, S, A, P, (int)tick, (const int*)nbuf.data_ptr(), (int*)cbuf.data_ptr(),
                                                   (int*)err.data_ptr());
}

// st = [x y z vx vy vz yaw grounded crouch alive locked]; E matches of NP players each
void resolve(std::vector<torch::Tensor> wt, std::vector<double> geo, std::vector<torch::Tensor> st, torch::Tensor cbuf,
             torch::Tensor err, int64_t E, int64_t NP, bool wall, bool swept, int64_t kmax) {
  c10::cuda::CUDAGuard guard(st[0].device());
  TORCH_CHECK(NP <= MAXP, "at most 16 players per match");
  World W = make_world(wt, geo, (int)kmax);
  CS S;
  S.x = P_<double>(st[0]); S.y = P_<double>(st[1]); S.z = P_<double>(st[2]);
  S.vx = P_<double>(st[3]); S.vy = P_<double>(st[4]); S.vz = P_<double>(st[5]); S.yaw = P_<double>(st[6]);
  S.grounded = P_<bool>(st[7]); S.crouch = P_<bool>(st[8]); S.alive = P_<bool>(st[9]); S.locked = P_<bool>(st[10]);
  if (E == 0) return;
  int bs = 64;
  resolve_kernel<<<(int)((E + bs - 1) / bs), bs, 0, c10::cuda::getCurrentCUDAStream()>>>(
      W, S, (int)E, (int)NP, wall ? 1 : 0, swept ? 1 : 0, (int*)cbuf.data_ptr(), (int*)err.data_ptr());
}

#ifdef MV_COUNT
torch::Tensor dbg_counters(bool reset) {
  auto o = torch::empty({16}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  unsigned long long h[16]; cudaMemcpyFromSymbol(h, g_cnt, sizeof(h));
  for (int i = 0; i < 16; i++) o[i] = (int64_t)h[i];
  if (reset) { unsigned long long z[16] = {0}; cudaMemcpyToSymbol(g_cnt, z, sizeof(z)); }
  return o;
}
torch::Tensor dbg_phases(int64_t nw) {
  auto o = torch::zeros({8, nw}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  static unsigned long long ph[8][1 << 13];
  cudaMemcpyFromSymbol(ph, g_ph, sizeof(ph));
  for (int k = 0; k < 8; k++) for (int64_t i = 0; i < nw; i++) o[k][i] = (int64_t)ph[k][i];
  static unsigned long long z[8][1 << 13]; cudaMemcpyToSymbol(g_ph, z, sizeof(z));
  return o;
}
// per warp: cycles, max candidates, SAT requests, sweep requests, hull calls   (first nw warps)
torch::Tensor dbg_warps(int64_t nw, bool reset) {
  auto o = torch::zeros({5, nw}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  static long long wc[1 << 16]; static int wn[1 << 16], wr[1 << 16], ws[1 << 16], wh[1 << 16];
  cudaMemcpyFromSymbol(wc, g_wc, sizeof(wc)); cudaMemcpyFromSymbol(wn, g_wn, sizeof(wn)); cudaMemcpyFromSymbol(wr, g_wr, sizeof(wr));
  cudaMemcpyFromSymbol(ws, g_ws, sizeof(ws)); cudaMemcpyFromSymbol(wh, g_wh, sizeof(wh));
  for (int64_t i = 0; i < nw; i++) { o[0][i] = wc[i]; o[1][i] = wn[i]; o[2][i] = wr[i]; o[3][i] = ws[i]; o[4][i] = wh[i]; }
  if (reset) { static long long z8[1 << 16]; static int z4[1 << 16]; cudaMemcpyToSymbol(g_wc, z8, sizeof(z8)); cudaMemcpyToSymbol(g_wn, z4, sizeof(z4)); cudaMemcpyToSymbol(g_wr, z4, sizeof(z4)); cudaMemcpyToSymbol(g_ws, z4, sizeof(z4)); cudaMemcpyToSymbol(g_wh, z4, sizeof(z4)); }
  return o;
}
#endif
torch::Tensor dbg_hypot(torch::Tensor a, torch::Tensor b) {
  auto o = torch::empty_like(a); int n = a.numel();
  if (n) hypot_kernel<<<(n + 255) / 256, 256>>>((double*)a.data_ptr(), (double*)b.data_ptr(), (double*)o.data_ptr(), n);
  return o;
}
torch::Tensor dbg_trig(torch::Tensor a) {
  auto o = torch::empty({a.numel(), 2}, a.options()); int n = a.numel();
  if (n) trig_kernel<<<(n + 255) / 256, 256>>>((double*)a.data_ptr(), (double*)o.data_ptr(), n);
  return o;
}

// candidate count of step_kernel's tick-level gather, for players pv[P,6] = (x, y, z, vx, vy, vz) in world coordinates
__global__ void cand_kernel(World W, const double* pv, int P, int* cbuf, int* err, int* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= P) return;
  const double* q = pv + 6 * i; const double R = MV_R, dt = MV_DT;
  double d0 = q[0] - W.o[0], d1 = q[1] - W.o[1], d2 = q[2] - W.o[2];
  double px = W.Bt[0] * d0 + W.Bt[1] * d1 + W.Bt[2] * d2, py = W.Bt[3] * d0 + W.Bt[4] * d1 + W.Bt[5] * d2, pz = W.Bt[6] * d0 + W.Bt[7] * d1 + W.Bt[8] * d2;
  double v0 = q[3], v1 = q[4], v2 = q[5];
  double vx = W.Bt[0] * v0 + W.Bt[1] * v1 + W.Bt[2] * v2, vy = W.Bt[3] * v0 + W.Bt[4] * v1 + W.Bt[5] * v2, vz = W.Bt[6] * v0 + W.Bt[7] * v1 + W.Bt[8] * v2;
  double sp = sqrt(vx * vx + vz * vz);
  double mh = sp * dt + 0.45, vdn = py_max(0.0, -vy) * dt + 1.2, vup = py_max(0.0, vy) * dt + 0.7;
  double lo[3] = {px - R - mh, py - vdn, pz - R - mh}, hi[3] = {px + R + mh, py + MV_STAND_H + vup, pz + R + mh};
  out[i] = gather(W, lo, hi, cbuf + (size_t)i * W.kmax, 1, err);
}
torch::Tensor dbg_cand(std::vector<torch::Tensor> wt, std::vector<double> geo, torch::Tensor pv, torch::Tensor cbuf, torch::Tensor err, int64_t kmax) {
  c10::cuda::CUDAGuard guard(pv.device());
  World W = make_world(wt, geo, (int)kmax);
  int P = (int)(pv.numel() / 6);
  auto o = torch::empty({P}, torch::TensorOptions().dtype(torch::kInt32).device(pv.device()));
  if (P) cand_kernel<<<(P + 127) / 128, 128, 0, c10::cuda::getCurrentCUDAStream()>>>(W, (const double*)pv.data_ptr(), P, (int*)cbuf.data_ptr(), (int*)err.data_ptr(), (int*)o.data_ptr());
  return o;
}
