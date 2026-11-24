"""Batched enemy visibility (port of fpssim.visibility.visible_enemies; validated 0.99997-0.999999 per pair on recordings).

Rule: observer alive and not blind (blind_start <= t < blind_until); target alive enemy; |horizontal angle to the target's feet|
<= 55 deg; per body point (head h-0.16, chest (upper_lo + h-0.18)/2, abdomen 0.3h; h = 1.8 / 1.1 crouched, upper_lo = 0.75 / 0.45):
|elevation(eye -> point) - pitch| <= 45 deg, static line of sight clear (float64 any-hit segment test, same as TriMesh.intersect), and
not blocked by smoke (`smoke_fn`, e.g. GpuGrenades.smoke_blocks).  Eye = feet + 1.62 (0.95 crouched).

    from gpusim.vis import visible, whitebox_mesh
    parts = visible(state, mesh, tick, smoke_fn=None)      # bool [E,10 obs,10 tgt,3 (head, chest, abdomen)]
    seen  = parts.any(-1)                                   # bool [E,10,10]

`mesh`: gpusim.ray.GpuMesh of the exact Dust2 mesh, or of the triangulated whitebox (`whitebox_mesh(static, device)`: float32 boxes,
stair-stepped ramps, ground plane at -0.002; the same geometry as fpssim.grenades.BoxContact).
`smoke_fn(p0[R,3], p1[R,3], env_idx[R], tick) -> bool[R]`.  `tick`: int, or a [E] tensor (per match clock).

Implementation (docs/gpu_sim.md "性能优化"): on CUDA, with smoke_fn None or a GpuGrenades.smoke_blocks bound method, everything runs in
three kernels without a host sync: a candidate kernel (the float64 cone tests of `candidates`, one thread per (match, observer,
target)) appends the surviving (match, observer, target, part) segments to a work list (warp-aggregated atomics), a trace kernel runs
the float64 any-hit BVH test on them (a float32 pre-test with a rigorous error bound rejects most triangles; every triangle it cannot
reject gets the original float64 Moller-Trumbore test, so the result is the float64 one) and the smoke test inline (the seg_smoke code of
gpusim.grenades_cuda).  `trace(..., points=...)` does the same for points (bomb / world items of gpusim.obs).  Any other smoke_fn, CPU
tensors or GPUSIM_VIS_TORCH=1 use the original torch path (`visible_torch`).
"""
import math, os
import torch
from torch.utils.cpp_extension import load_inline

HALF_H = math.radians(55.0)
HALF_V = math.radians(45.0)
_C = {}


def _consts(dev):
    c = _C.get(dev)
    if c is None:
        f = lambda v: torch.tensor(v, dtype=torch.float64, device=dev)
        c = _C[dev] = dict(eye=(f(0.95), f(1.62)), h=(f(1.1), f(1.8)), ulo=(f(0.45), f(0.75)), c016=f(0.16), c018=f(0.18), c03=f(0.3),
                           team=torch.arange(10, device=dev) // 5, seat=torch.arange(10, device=dev))
    return c


def whitebox_mesh(static, device='cuda'):
    """GpuMesh of a whitebox map: tools.meshmove_wb_validate.triangulate (= BoxContact geometry) in world coordinates."""
    import numpy as np
    from tools.meshmove_wb_validate import triangulate, BASIS
    from .ray import GpuMesh
    raw = triangulate(static).astype(np.float64)                  # cs2-web raw coordinates (y up), float32 values
    tri = np.einsum('ij,tkj->tki', np.asarray(BASIS, np.float64), raw)
    return GpuMesh(tri, device)


def candidates(s, t, teammates=False):
    """prefilter (alive / blind / team / horizontal cone / vertical cone per point).  Returns (cand bool [E,10,10,3], geometry tensors)."""
    dev = s['x'].device; k = _consts(dev)
    cr = s['crouch']
    ez = s['z'] + torch.where(cr, k['eye'][0], k['eye'][1])
    h = torch.where(cr, k['h'][0], k['h'][1]); ulo = torch.where(cr, k['ulo'][0], k['ulo'][1])
    hz = torch.stack([h - k['c016'], (ulo + h - k['c018']) / 2, k['c03'] * h], -1)           # [E,10,3] per target
    dx = s['x'][:, None, :] - s['x'][:, :, None]; dy = s['y'][:, None, :] - s['y'][:, :, None]   # [E,obs,tgt] (target - observer)
    az = torch.atan2(-dy, dx)
    da = az - s['yaw'][:, :, None]
    hok = torch.atan2(torch.sin(da), torch.cos(da)).abs() <= HALF_H
    team = k['team']
    enemy = team[None, :, None] != team[None, None, :]
    if teammates:
        enemy = k['seat'][None, :, None] != k['seat'][None, None, :]
    if torch.is_tensor(t) and t.dim() == 1:
        t = t[:, None]
    obs_ok = s['alive'] & ~((s['blind_start'] <= t) & (t < s['blind_until']))
    pair = obs_ok[:, :, None] & s['alive'][:, None, :] & enemy & hok                        # [E,10,10]
    tz = s['z'][:, None, :, None] + hz[:, None, :, :]                                      # [E,1->obs,tgt,3]
    dxy = torch.hypot(dx, dy)[..., None]
    el = torch.atan2(tz - ez[:, :, None, None], dxy) - s['pitch'][:, :, None, None]
    cand = pair[..., None] & (el.abs() <= HALF_V)                                            # [E,10,10,3]
    return cand, ez, hz


def visible(s, mesh, t, smoke_fn=None, teammates=False):
    """s: state dict with x,y,z,yaw,pitch [E,10] f64; crouch, alive [E,10] bool; blind_start, blind_until [E,10] int.
    Returns parts bool [E,10(obs),10(tgt),3(head,chest,abdomen)]."""
    G = smoke_owner(smoke_fn)
    if not fused_ok(s, mesh) or (smoke_fn is not None and G is None):
        return visible_torch(s, mesh, t, smoke_fn, teammates)
    return trace(s, mesh, t, G, teammates=teammates)[0]


def visible_torch(s, mesh, t, smoke_fn=None, teammates=False):
    """original torch implementation of `visible` (reference for the fused kernels; any smoke_fn)."""
    E = s['x'].shape[0]; dev = s['x'].device
    cand, ez, hz = candidates(s, t, teammates)
    idx = cand.nonzero(as_tuple=False)
    out = torch.zeros(E, 10, 10, 3, dtype=torch.bool, device=dev)
    if len(idx) == 0:
        return out
    e, o, g, k = idx.unbind(1)
    p0 = torch.stack([s['x'][e, o], s['y'][e, o], ez[e, o]], -1)
    p1 = torch.stack([s['x'][e, g], s['y'][e, g], (s['z'][e, g] + hz[e, g, k])], -1)
    clear = ~mesh.blocked(p0, p1)
    if smoke_fn is not None:
        clear &= ~smoke_fn(p0, p1, e, t)
    out[e, o, g, k] = clear
    return out


# ============================================================================================================== fused CUDA path
def smoke_owner(fn):
    """the GpuGrenades behind `fn` if fn is its bound smoke_blocks method, else None."""
    o = getattr(fn, '__self__', None)
    if o is not None and getattr(fn, '__name__', '') == 'smoke_blocks' and hasattr(o, 'glist') and hasattr(o, 'dims'):
        return o
    return None


def fused_ok(s, mesh):
    return s['x'].is_cuda and (mesh is None or hasattr(mesh, 'tri64')) and not os.environ.get('GPUSIM_VIS_TORCH')


_VBODY = r"""
#define VSTACK 64
#define EPSK 1.9073486328125e-06f
struct TF { const void* p; int t; };
__device__ __forceinline__ long long gi(const TF& f, long long i){
  switch(f.t){ case 0: return (long long)((const double*)f.p)[i]; case 1: return (long long)((const float*)f.p)[i];
    case 2: return ((const long long*)f.p)[i]; case 3: return ((const int*)f.p)[i]; case 4: return ((const short*)f.p)[i];
    case 5: return ((const signed char*)f.p)[i]; default: return ((const unsigned char*)f.p)[i]; } }
struct VP { const double *x,*y,*z,*yaw,*pitch; const bool *crouch,*alive; TF bs, bu; };
struct MSH { const float *nlo,*nhi; const int *nleft,*nstart,*ncount; const double* tri; const float* e12; int has; };
struct PTS { const double* wpos; TF wvalid; const double* bpos; TF bstate; int K; };

__device__ __forceinline__ bool aabb_v(float lx,float ly,float lz,float hx,float hy,float hz,float3 o,float3 inv,float tmax){
  float t0=0.f, t1=tmax, a, b;
  a=(lx-o.x)*inv.x; b=(hx-o.x)*inv.x; t0=fmaxf(t0,fminf(a,b)); t1=fminf(t1,fmaxf(a,b));
  a=(ly-o.y)*inv.y; b=(hy-o.y)*inv.y; t0=fmaxf(t0,fminf(a,b)); t1=fminf(t1,fmaxf(a,b));
  a=(lz-o.z)*inv.z; b=(hz-o.z)*inv.z; t0=fmaxf(t0,fminf(a,b)); t1=fminf(t1,fmaxf(a,b));
  return t0<=t1;
}
// float64 any-hit of the segment o -> q over the BVH (traversal and float64 test of gpusim.ray any_hit64_k = TriMesh.intersect).
// Float32 pre-test: e1/e2 rounded once from the float64 edges, s = o - v0 rounded from float64; every float32 quantity carries the bound
// EPSK = 32 ulp x (sum of |products|) (the worst case of these depth-3 expressions is about 8 ulp); a triangle is skipped only when the
// bound proves that the float64 test rejects it, otherwise the float64 test decides (so the result is the float64 one).
__device__ bool blocked64(const MSH& M, double ox, double oy, double oz, double qx, double qy, double qz){
  double Dx=qx-ox, Dy=qy-oy, Dz=qz-oz;
  float3 o=make_float3((float)ox,(float)oy,(float)oz);
  float3 inv=make_float3(1.f/(float)Dx,1.f/(float)Dy,1.f/(float)Dz);
  float dx=(float)Dx, dy=(float)Dy, dz=(float)Dz;
  int st[VSTACK]; int sp=0; st[sp++]=0;
  while(sp){
    int n=st[--sp];
    if(!aabb_v(M.nlo[3*n]-1e-4f,M.nlo[3*n+1]-1e-4f,M.nlo[3*n+2]-1e-4f,M.nhi[3*n]+1e-4f,M.nhi[3*n+1]+1e-4f,M.nhi[3*n+2]+1e-4f,o,inv,1.0f+1e-6f)) continue;
    int l=M.nleft[n]; if(l>=0){ st[sp++]=l; st[sp++]=l+1; continue; }
    int k1=M.nstart[n]+M.ncount[n];
    for(int k=M.nstart[n];k<k1;k++){
      const double* v=M.tri+9*k;
      {
        const float* E=M.e12+6*k;
        float e1x=E[0],e1y=E[1],e1z=E[2],e2x=E[3],e2y=E[4],e2z=E[5];
        float sx=(float)(ox-v[0]), sy=(float)(oy-v[1]), sz=(float)(oz-v[2]);
        float hx=dy*e2z-dz*e2y, hy=dz*e2x-dx*e2z, hz=dx*e2y-dy*e2x;
        float ahx=fabsf(dy*e2z)+fabsf(dz*e2y), ahy=fabsf(dz*e2x)+fabsf(dx*e2z), ahz=fabsf(dx*e2y)+fabsf(dy*e2x);
        float a=e1x*hx+e1y*hy+e1z*hz;
        float Ba=EPSK*(fabsf(e1x)*ahx+fabsf(e1y)*ahy+fabsf(e1z)*ahz);
        float aa=fabsf(a);
        if(aa>Ba+1e-12f){
          float sg=a>0.f?1.f:-1.f;
          float un=(sx*hx+sy*hy+sz*hz)*sg, Bu=EPSK*(fabsf(sx)*ahx+fabsf(sy)*ahy+fabsf(sz)*ahz);
          if(un<-Bu || un-aa>Bu+Ba) continue;
          float qx_=sy*e1z-sz*e1y, qy_=sz*e1x-sx*e1z, qz_=sx*e1y-sy*e1x;
          float aqx=fabsf(sy*e1z)+fabsf(sz*e1y), aqy=fabsf(sz*e1x)+fabsf(sx*e1z), aqz=fabsf(sx*e1y)+fabsf(sy*e1x);
          float vn=(dx*qx_+dy*qy_+dz*qz_)*sg, Bv=EPSK*(fabsf(dx)*aqx+fabsf(dy)*aqy+fabsf(dz)*aqz);
          if(vn<-Bv || un+vn-aa>Bu+Bv+Ba+EPSK*(fabsf(un)+fabsf(vn))) continue;
          float tn=(e2x*qx_+e2y*qy_+e2z*qz_)*sg, Bt=EPSK*(fabsf(e2x)*aqx+fabsf(e2y)*aqy+fabsf(e2z)*aqz);
          if(tn<-Bt || tn-aa>Bt+Ba) continue;
        }
      }
      double e1x=v[3]-v[0],e1y=v[4]-v[1],e1z=v[5]-v[2],e2x=v[6]-v[0],e2y=v[7]-v[1],e2z=v[8]-v[2];
      double hx=Dy*e2z-Dz*e2y, hy=Dz*e2x-Dx*e2z, hz=Dx*e2y-Dy*e2x; double a=e1x*hx+e1y*hy+e1z*hz;
      if(fabs(a)<=1e-12) continue; double f=1.0/a; double sx=ox-v[0],sy=oy-v[1],sz=oz-v[2];
      double u=f*(sx*hx+sy*hy+sz*hz); if(u<0||u>1) continue;
      double qx_=sy*e1z-sz*e1y,qy_=sz*e1x-sx*e1z,qz_=sx*e1y-sy*e1x; double vv=f*(Dx*qx_+Dy*qy_+Dz*qz_); if(vv<0||u+vv>1) continue;
      double t=f*(e2x*qx_+e2y*qy_+e2z*qz_); if(t>=0 && t<=1) return true;
    }
  }
  return false;
}
__device__ __forceinline__ void push(bool w, long long item, long long* work, int* cnt){
  unsigned m=__ballot_sync(0xffffffffu, w);
  if(!m) return;
  int lane=threadIdx.x&31, leader=__ffs(m)-1; int base=0;
  if(lane==leader) base=atomicAdd(cnt, __popc(m));
  base=__shfl_sync(0xffffffffu, base, leader);
  if(w) work[base+__popc(m&((1u<<lane)-1u))]=item;
}
__device__ __forceinline__ double eyez(const VP& P, long long pi){ return P.z[pi]+(P.crouch[pi]?0.95:1.62); }
__device__ __forceinline__ void hzpart(const VP& P, long long pi, double* hz){
  bool cr=P.crouch[pi]; double h=cr?1.1:1.8, ulo=cr?0.45:0.75;
  hz[0]=h-0.16; hz[1]=((ulo+h)-0.18)*0.5; hz[2]=0.3*h;
}
// gpusim.vis.candidates for one (match, observer, target): up to 3 part segments go to the work list
__global__ void vis_cand_k(VP P, int E, int teammates, const int* __restrict__ tick, bool* __restrict__ out, long long* work, int* cnt){
  long long gid=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  int want=0; int e=(int)(gid/100), o=(int)((gid/10)%10), g=(int)(gid%10);
  if(gid<(long long)E*100){
    out[gid*3]=0; out[gid*3+1]=0; out[gid*3+2]=0;
    long long po=(long long)e*10+o, pg=(long long)e*10+g;
    bool enemy= teammates? (o!=g) : ((o/5)!=(g/5));
    long long t=tick[e];
    bool obs_ok=P.alive[po] && !(gi(P.bs,po)<=t && t<gi(P.bu,po));
    if(enemy && obs_ok && P.alive[pg]){
      double dx=P.x[pg]-P.x[po], dy=P.y[pg]-P.y[po];
      double da=atan2(-dy,dx)-P.yaw[po];
      if(fabs(atan2(sin(da),cos(da)))<=HALF_H){
        double hz[3]; hzpart(P,pg,hz);
        double ez=eyez(P,po), dxy=hypot(dx,dy);
        for(int k=0;k<3;k++){
          double el=atan2((P.z[pg]+hz[k])-ez,dxy)-P.pitch[po];
          if(fabs(el)<=HALF_V) want|=1<<k;
        }
      }
    }
  }
  for(int k=0;k<3;k++) push((want>>k)&1, ((long long)e<<16)|((long long)o<<12)|((long long)g<<4)|((long long)k<<2), work, cnt);
}
__device__ __forceinline__ const double* ptpos(const PTS& Q, int e, int k){ return k<Q.K ? Q.wpos+((long long)e*Q.K+k)*3 : Q.bpos+(long long)e*3; }
// gpusim.obs.point_visible prefilter for one (match, observer, point k): k < K world item, k == K bomb
__global__ void pt_cand_k(VP P, int E, PTS Q, int has_mesh, bool* __restrict__ out, long long* work, int* cnt){
  int K1=Q.K+1;
  long long gid=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  bool want=false; int e=(int)(gid/(10*K1)), o=(int)((gid/K1)%10), k=(int)(gid%K1);
  if(gid<(long long)E*10*K1){
    long long po=(long long)e*10+o;
    bool valid= k<Q.K ? (gi(Q.wvalid,(long long)e*Q.K+k)!=0) : (gi(Q.bstate,e)>=1);
    bool c=false;
    if(valid && P.alive[po]){
      const double* p=ptpos(Q,e,k);
      double ex=P.x[po], ey=P.y[po], ez=eyez(P,po);
      double dx=p[0]-ex, dy=p[1]-ey, dz=p[2]-ez;
      // float32 pre-decision of the two cone tests (margin 1e-4 rad >> the float32 error of a few 1e-7 rad); undecided -> float64
      int dec=-1;                                    // 1 inside, 0 outside, -1 undecided
      double yw=P.yaw[po], pt=P.pitch[po];
      float fx=(float)dx, fy=(float)dy, fz=(float)dz, fh=sqrtf(fx*fx+fy*fy);
      if(fh>1e-3f && fabs(yw)<20.0 && fabs(pt)<20.0){        // |yaw| < 20: the 2 pi reduction stays within ~1e-5 rad
        float da=atan2f(-fy,fx)-(float)yw;
        while(da>3.14159265f) da-=6.28318531f; while(da<-3.14159265f) da+=6.28318531f;
        float h=fabsf(da)-(float)HALF_H, v=fabsf(atan2f(fz,fh)-(float)pt)-(float)HALF_V;
        if(h>1e-4f || v>1e-4f) dec=0; else if(h<-1e-4f && v<-1e-4f) dec=1;
      }
      if(dec>=0) c=dec;
      else {
        double dh=hypot(dx,dy);
        double da=atan2(-dy,dx)-yw;
        double wrap=atan2(sin(da),cos(da));
        bool in0=(dh<1e-9)&&(fabs(dz)<1e-9);
        c= in0 || ((fabs(wrap)<=HALF_H) && (fabs(atan2(dz,dh)-pt)<=HALF_V));
      }
    }
    out[gid]= c && !has_mesh;
    want= c && has_mesh;
  }
  push(want, ((long long)e<<16)|((long long)o<<12)|((long long)k<<4)|1LL, work, cnt);
}
__device__ __forceinline__ double norm3(double a, double b, double c){ return sqrt(NORM3(a,b,c)); }
__global__ void trace_k(VP P, MSH M, GS g, Dims D, UT U, int has_smoke, PTS Q, const long long* __restrict__ work, const int* __restrict__ cnt,
                        const int* __restrict__ tick_v, const int* __restrict__ tick_p, bool* __restrict__ out_vis, bool* __restrict__ out_pv){
  int n=*cnt;
  for(long long w=(long long)blockIdx.x*blockDim.x+threadIdx.x; w<n; w+=(long long)gridDim.x*blockDim.x){
    long long it=work[w];
    int typ=(int)(it&1), k=(int)((it>>2)&3), g_=(int)((it>>4)&255), o=(int)((it>>12)&15), e=(int)(it>>16);
    long long po=(long long)e*10+o;
    double a[3], b[3]; int now;
    a[0]=P.x[po]; a[1]=P.y[po]; a[2]=eyez(P,po);
    if(typ==0){
      long long pg=(long long)e*10+g_; double hz[3]; hzpart(P,pg,hz);
      b[0]=P.x[pg]; b[1]=P.y[pg]; b[2]=P.z[pg]+hz[k]; now=tick_v[e];
    } else {
      const double* p=ptpos(Q,e,g_);
      double dx=p[0]-a[0], dy=p[1]-a[1], dz=p[2]-a[2];
      double nn=norm3(dx,dy,dz); if(nn==0) nn=1.0;
      b[0]=p[0]-(0.03*dx)/nn; b[1]=p[1]-(0.03*dy)/nn; b[2]=p[2]-(0.03*dz)/nn; now=tick_p[e];
    }
    bool clear=!blocked64(M,a[0],a[1],a[2],b[0],b[1],b[2]);
    if(clear && has_smoke){
      for(int s=0;s<D.S;s++){
        int sq=e*D.S+s;
        if(!g.sm_valid[sq] || !(g.sm_expires[sq]>now)) continue;
        if(seg_smoke(g,D,U,sq,a,b,now)){ clear=false; break; }
      }
    }
    if(typ==0) out_vis[(po*10+g_)*3+k]=clear;
    else out_pv[po*(Q.K+1)+g_]=clear;
  }
}
__global__ void norm_test_k(const double* d, int R, double* out){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=R) return;
  double a=d[3*r], b=d[3*r+1], c=d[3*r+2];
  out[3*r]=sqrt(fma(c,c,fma(b,b,a*a))); out[3*r+1]=sqrt((a*a+b*b)+c*c); out[3*r+2]=sqrt(a*a+(b*b+c*c));
}

static TF tf(const torch::Tensor& t){
  TF f; f.p=t.numel()? t.data_ptr() : nullptr;
  auto d=t.scalar_type();
  f.t= d==torch::kFloat64?0: d==torch::kFloat32?1: d==torch::kInt64?2: d==torch::kInt32?3: d==torch::kInt16?4: d==torch::kInt8?5:6;
  return f;
}
std::vector<torch::Tensor> run(std::vector<torch::Tensor> pl, torch::Tensor tick_v, torch::Tensor tick_p, int64_t do_vis, int64_t teammates,
    int64_t K, torch::Tensor wpos, torch::Tensor wvalid, torch::Tensor bpos, torch::Tensor bstate,
    std::vector<torch::Tensor> mesh, int64_t has_mesh, std::vector<torch::Tensor> gs, std::vector<int64_t> dims,
    torch::Tensor ui, torch::Tensor ud, int64_t has_smoke, torch::Tensor work, torch::Tensor cnt){
  auto stream=at::cuda::getCurrentCUDAStream();
  int E=(int)pl[0].size(0);
  VP P; P.x=pl[0].data_ptr<double>(); P.y=pl[1].data_ptr<double>(); P.z=pl[2].data_ptr<double>(); P.yaw=pl[3].data_ptr<double>();
  P.pitch=pl[4].data_ptr<double>(); P.crouch=pl[5].data_ptr<bool>(); P.alive=pl[6].data_ptr<bool>(); P.bs=tf(pl[7]); P.bu=tf(pl[8]);
  MSH M; M.has=(int)has_mesh; M.nlo=M.nhi=nullptr; M.nleft=M.nstart=M.ncount=nullptr; M.tri=nullptr; M.e12=nullptr;
  if(has_mesh){ M.nlo=mesh[0].data_ptr<float>(); M.nhi=mesh[1].data_ptr<float>(); M.nleft=mesh[2].data_ptr<int>(); M.nstart=mesh[3].data_ptr<int>();
    M.ncount=mesh[4].data_ptr<int>(); M.tri=mesh[5].data_ptr<double>(); M.e12=mesh[6].data_ptr<float>(); }
  GS g=make_GS(gs); Dims D; D.E=dims[0]; D.P=dims[1]; D.S=dims[2]; D.C=dims[3]; D.F=dims[4]; D.NC=dims[5]; D.D=dims[6]; D.NI=dims[7];
  UT U; for(int k=0;k<6;k++){ U.fuse[k]=0; U.duration[k]=0; U.radius[k]=0; U.damage[k]=0; }
  if(has_smoke){ auto a=ui.accessor<int,2>(); auto b=ud.accessor<double,2>();
    for(int k=0;k<6;k++){ U.fuse[k]=a[0][k]; U.duration[k]=a[1][k]; U.radius[k]=b[0][k]; U.damage[k]=b[1][k]; } }
  PTS Q; Q.K=(int)K; Q.wpos=nullptr; Q.bpos=nullptr; Q.wvalid=tf(wvalid); Q.bstate=tf(bstate);
  if(K>0) Q.wpos=wpos.data_ptr<double>();
  if(K>=0) Q.bpos=bpos.data_ptr<double>();
  auto ob=pl[0].options().dtype(torch::kBool);
  torch::Tensor ov= do_vis? torch::empty({E,10,10,3},ob) : torch::empty({0},ob);
  torch::Tensor op= K>=0? torch::empty({E,10,K+1},ob) : torch::empty({0},ob);
  int* c=cnt.data_ptr<int>(); long long* w=(long long*)work.data_ptr<int64_t>();
  long long cap=(do_vis? (long long)E*300 : 0) + (K>=0? (long long)E*10*(K+1) : 0);
  TORCH_CHECK(work.numel()>=cap, "work buffer too small");
  cudaMemsetAsync(c,0,sizeof(int),stream);
  if(do_vis && E){ long long n=(long long)E*100; vis_cand_k<<<(int)((n+127)/128),128,0,stream>>>(P,E,(int)teammates,tick_v.data_ptr<int>(),ov.data_ptr<bool>(),w,c); }
  if(K>=0 && E){ long long n=(long long)E*10*(K+1); pt_cand_k<<<(int)((n+127)/128),128,0,stream>>>(P,E,Q,(int)has_mesh,op.data_ptr<bool>(),w,c); }
  if(has_mesh && cap){
    int dev; cudaGetDevice(&dev); int nsm; cudaDeviceGetAttribute(&nsm,cudaDevAttrMultiProcessorCount,dev);
    long long nb=(cap+127)/128; if(nb>(long long)nsm*16) nb=(long long)nsm*16;
    trace_k<<<(int)nb,128,0,stream>>>(P,M,g,D,U,(int)has_smoke,Q,w,c,tick_v.data_ptr<int>(),tick_p.data_ptr<int>(),
        do_vis? ov.data_ptr<bool>():nullptr, K>=0? op.data_ptr<bool>():nullptr);
  }
  return {ov, op};
}
torch::Tensor norm_test(torch::Tensor d){
  int R=(int)d.size(0); auto out=torch::empty({R,3},d.options());
  if(R) norm_test_k<<<(R+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(d.data_ptr<double>(),R,out.data_ptr<double>());
  return out;
}
"""

_VCPP = """
std::vector<torch::Tensor> run(std::vector<torch::Tensor> pl, torch::Tensor tick_v, torch::Tensor tick_p, int64_t do_vis, int64_t teammates,
    int64_t K, torch::Tensor wpos, torch::Tensor wvalid, torch::Tensor bpos, torch::Tensor bstate,
    std::vector<torch::Tensor> mesh, int64_t has_mesh, std::vector<torch::Tensor> gs, std::vector<int64_t> dims,
    torch::Tensor ui, torch::Tensor ud, int64_t has_smoke, torch::Tensor work, torch::Tensor cnt);
torch::Tensor norm_test(torch::Tensor d);
"""
# torch.linalg.vector_norm over a last dim of 3 (float64, CUDA) combines the squares as ((a^2 + c^2) + b^2) (two strided accumulators; checked on CUDA, torch 2.10)
NORM3 = '((a*a+c*c)+b*b)'
_vmod = None


def _vext():
    global _vmod
    if _vmod is None:
        import hashlib
        from . import grenades_cuda as GC
        os.environ.setdefault('TORCH_EXTENSIONS_DIR', './fps_data/tmp/torch_ext')
        src = ('#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_runtime.h>\n#include <math.h>\n' +
               GC._struct('GS', GC.GS_FIELDS) + GC.smoke_device_src() +
               '#define HALF_H %r\n#define HALF_V %r\n#define NORM3(a,b,c) %s\n' % (HALF_H, HALF_V, NORM3) + _VBODY)
        key = hashlib.sha1(src.encode()).hexdigest()[:10]
        _vmod = load_inline('fpsvis_' + key, cpp_sources=_VCPP, cuda_sources=src, functions=['run', 'norm_test'],
                            extra_cuda_cflags=['-O3', '-fmad=false'], verbose=False)
    return _vmod


def _mesh_args(mesh, dev):
    if mesh is None:
        return [torch.zeros(1, device=dev)] * 7
    c = getattr(mesh, '_vis_args', None)
    if c is None:
        t = mesh.t; tri = mesh.tri64
        e12 = torch.cat([tri[:, 3:6] - tri[:, 0:3], tri[:, 6:9] - tri[:, 0:3]], 1).float().contiguous()   # float64 edges rounded once
        c = mesh._vis_args = [t['node_lo'].contiguous(), t['node_hi'].contiguous(), t['node_left'].contiguous(), t['node_start'].contiguous(),
                              t['node_count'].contiguous(), tri.contiguous(), e12]
    return c


_WORK = {}


def _work(dev, n):
    w = _WORK.get(dev)
    if w is None or w[0].numel() < n:
        w = _WORK[dev] = (torch.empty(max(n, 1), dtype=torch.int64, device=dev), torch.zeros(1, dtype=torch.int32, device=dev))
    return w


def _tick_e(t, E, dev):
    if torch.is_tensor(t):
        t = t.to(device=dev, dtype=torch.int32)
        return (t.expand(E) if t.dim() == 0 else t).contiguous()
    return torch.full((E,), int(t), dtype=torch.int32, device=dev)


def trace(s, mesh, t, grenades=None, teammates=False, points=None, tick_p=None, vis=True):
    """fused visibility -> (parts bool [E,10,10,3] or None, point visibility bool [E,10,K+1] or None).  No host sync.
    points = (wi_pos [E,K,3] f64 or None, wi_valid [E,K], bomb_pos [E,3] f64, bomb_state [E]): point k < K is world item k (valid =
    wi_valid), k = K the bomb (valid = bomb_state >= 1); value = gpusim.obs.point_visible (cone, alive observer, mesh LOS pulled back
    3 cm, smoke).  `t`: tick of the enemy test (blind window and smokes), `tick_p`: tick of the point test (default `t`), int or [E].
    grenades: GpuGrenades whose smokes block (or None).  mesh None: points get the cone test only (as point_visible)."""
    E = s['x'].shape[0]; dev = s['x'].device
    d64 = lambda k: s[k] if s[k].dtype == torch.float64 and s[k].is_contiguous() else s[k].double().contiguous()
    b = lambda k: s[k] if s[k].dtype == torch.bool and s[k].is_contiguous() else s[k].bool().contiguous()
    pl = [d64('x'), d64('y'), d64('z'), d64('yaw'), d64('pitch'), b('crouch'), b('alive'), s['blind_start'].contiguous(),
          s['blind_until'].contiguous()]
    tv = _tick_e(t, E, dev)
    tp = tv if tick_p is None else _tick_e(tick_p, E, dev)
    z = torch.zeros(1, device=dev)
    if points is not None:
        wpos, wvalid, bpos, bstate = points
        K = 0 if wpos is None else wpos.shape[1]
        wpos = wpos.double().contiguous() if K else z.double()
        wvalid = wvalid.contiguous() if K else z
        bpos = bpos.double().contiguous(); bstate = bstate.contiguous()
    else:
        K = -1; wpos = bpos = z.double(); wvalid = bstate = z
    if vis and mesh is None:
        raise ValueError('visible() needs a mesh')
    n = (E * 300 if vis else 0) + (E * 10 * (K + 1) if K >= 0 else 0)
    work, cnt = _work(dev, n)
    if grenades is not None:
        gs, dims, ui, ud, hs = grenades.glist, grenades.dims, grenades.ui, grenades.ud, 1
    else:
        from . import grenades_cuda as GC
        gs = [torch.zeros(0, device=dev)] * len(GC.GS_FIELDS); dims = [0] * 8; hs = 0
        ui = torch.zeros(2, 6, dtype=torch.int32); ud = torch.zeros(2, 6, dtype=torch.float64)
    ov, op = _vext().run(pl, tv, tp, int(vis), int(teammates), K, wpos, wvalid, bpos, bstate, _mesh_args(mesh, dev), int(mesh is not None),
                         gs, list(dims), ui, ud, hs, work, cnt)
    return (ov if vis else None), (op if K >= 0 else None)
