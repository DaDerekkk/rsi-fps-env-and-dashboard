"""CUDA segment-vs-mesh queries over a flat BVH (torch cpp extension, JIT compiled).

first_hit(seg_p0[R,3], seg_p1[R,3]) -> t[R] (1.0e30 if no hit), tri_id[R]   (parametric t in [0,1] along p0->p1)
Moller-Trumbore in float32 with a small epsilon; double-sided like the CPU TriMesh.intersect."""
import os, torch
from torch.utils.cpp_extension import load_inline
from .bvh import build_bvh

_SRC = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#define STACK 64
__device__ __forceinline__ bool aabb(const float* lo, const float* hi, float3 o, float3 inv, float tmax){
  float t0=0.f, t1=tmax;
  float a,b;
  a=(lo[0]-o.x)*inv.x; b=(hi[0]-o.x)*inv.x; t0=fmaxf(t0,fminf(a,b)); t1=fminf(t1,fmaxf(a,b));
  a=(lo[1]-o.y)*inv.y; b=(hi[1]-o.y)*inv.y; t0=fmaxf(t0,fminf(a,b)); t1=fminf(t1,fmaxf(a,b));
  a=(lo[2]-o.z)*inv.z; b=(hi[2]-o.z)*inv.z; t0=fmaxf(t0,fminf(a,b)); t1=fminf(t1,fmaxf(a,b));
  return t0<=t1;
}
__global__ void first_hit_k(const float* __restrict__ p0, const float* __restrict__ p1, int R,
    const float* __restrict__ nlo, const float* __restrict__ nhi, const int* __restrict__ nleft,
    const int* __restrict__ nstart, const int* __restrict__ ncount, const float* __restrict__ tri,
    const int* __restrict__ tid, float* __restrict__ out_t, int* __restrict__ out_id){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=R) return;
  float3 o=make_float3(p0[3*r],p0[3*r+1],p0[3*r+2]);
  float3 d=make_float3(p1[3*r]-o.x,p1[3*r+1]-o.y,p1[3*r+2]-o.z);
  float3 inv=make_float3(1.f/d.x,1.f/d.y,1.f/d.z);
  float best=1.0f; int bid=-1;
  int st[STACK]; int sp=0; st[sp++]=0;
  while(sp){
    int n=st[--sp];
    if(!aabb(nlo+3*n,nhi+3*n,o,inv,best)) continue;
    int l=nleft[n];
    if(l>=0){ st[sp++]=l; st[sp++]=l+1; continue; }
    int s=nstart[n], c=ncount[n];
    for(int k=s;k<s+c;k++){
      const float* v=tri+9*k;
      float e1x=v[3]-v[0],e1y=v[4]-v[1],e1z=v[5]-v[2], e2x=v[6]-v[0],e2y=v[7]-v[1],e2z=v[8]-v[2];
      float hx=d.y*e2z-d.z*e2y, hy=d.z*e2x-d.x*e2z, hz=d.x*e2y-d.y*e2x;
      float a=e1x*hx+e1y*hy+e1z*hz; if(fabsf(a)<1e-12f) continue;
      float f=1.f/a; float sx=o.x-v[0], sy=o.y-v[1], sz=o.z-v[2];
      float u=f*(sx*hx+sy*hy+sz*hz); if(u<0.f||u>1.f) continue;
      float qx=sy*e1z-sz*e1y, qy=sz*e1x-sx*e1z, qz=sx*e1y-sy*e1x;
      float vv=f*(d.x*qx+d.y*qy+d.z*qz); if(vv<0.f||u+vv>1.f) continue;
      float t=f*(e2x*qx+e2y*qy+e2z*qz);
      if(t>=0.f && (t<best || (t==best && tid[k]<bid))){ best=t; bid=tid[k]; }
    }
  }
  out_t[r]= bid>=0? best : 1e30f; out_id[r]=bid;
}
__global__ void any_hit64_k(const double* __restrict__ p0, const double* __restrict__ p1, int R,
    const float* __restrict__ nlo, const float* __restrict__ nhi, const int* __restrict__ nleft,
    const int* __restrict__ nstart, const int* __restrict__ ncount, const double* __restrict__ tri, bool* __restrict__ out){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=R) return;
  double ox=p0[3*r],oy=p0[3*r+1],oz=p0[3*r+2]; double Dx=p1[3*r]-ox,Dy=p1[3*r+1]-oy,Dz=p1[3*r+2]-oz;
  float3 o=make_float3((float)ox,(float)oy,(float)oz); float3 inv=make_float3(1.f/(float)Dx,1.f/(float)Dy,1.f/(float)Dz);
  int st[STACK]; int sp=0; st[sp++]=0; bool hit=false;
  while(sp && !hit){
    int n=st[--sp];
    float lo[3]={nlo[3*n]-1e-4f,nlo[3*n+1]-1e-4f,nlo[3*n+2]-1e-4f}, hi[3]={nhi[3*n]+1e-4f,nhi[3*n+1]+1e-4f,nhi[3*n+2]+1e-4f};
    if(!aabb(lo,hi,o,inv,1.0f+1e-6f)) continue;
    int l=nleft[n]; if(l>=0){ st[sp++]=l; st[sp++]=l+1; continue; }
    for(int k=nstart[n];k<nstart[n]+ncount[n];k++){
      const double* v=tri+9*k;
      double e1x=v[3]-v[0],e1y=v[4]-v[1],e1z=v[5]-v[2],e2x=v[6]-v[0],e2y=v[7]-v[1],e2z=v[8]-v[2];
      double hx=Dy*e2z-Dz*e2y, hy=Dz*e2x-Dx*e2z, hz=Dx*e2y-Dy*e2x; double a=e1x*hx+e1y*hy+e1z*hz;
      if(fabs(a)<=1e-12) continue; double f=1.0/a; double sx=ox-v[0],sy=oy-v[1],sz=oz-v[2];
      double u=f*(sx*hx+sy*hy+sz*hz); if(u<0||u>1) continue;
      double qx=sy*e1z-sz*e1y,qy=sz*e1x-sx*e1z,qz=sx*e1y-sy*e1x; double vv=f*(Dx*qx+Dy*qy+Dz*qz); if(vv<0||u+vv>1) continue;
      double t=f*(e2x*qx+e2y*qy+e2z*qz); if(t>=0 && t<=1){ hit=true; break; }
    }
  }
  out[r]=hit;
}
torch::Tensor any_hit64(torch::Tensor p0, torch::Tensor p1, torch::Tensor nlo, torch::Tensor nhi, torch::Tensor nleft,
    torch::Tensor nstart, torch::Tensor ncount, torch::Tensor tri){
  int R=p0.size(0); auto out=torch::empty({R},p0.options().dtype(torch::kBool));
  if(R) any_hit64_k<<<(R+127)/128,128>>>(p0.data_ptr<double>(),p1.data_ptr<double>(),R,nlo.data_ptr<float>(),nhi.data_ptr<float>(),
     nleft.data_ptr<int>(),nstart.data_ptr<int>(),ncount.data_ptr<int>(),tri.data_ptr<double>(),out.data_ptr<bool>());
  return out;
}
std::vector<torch::Tensor> first_hit(torch::Tensor p0, torch::Tensor p1, torch::Tensor nlo, torch::Tensor nhi,
    torch::Tensor nleft, torch::Tensor nstart, torch::Tensor ncount, torch::Tensor tri, torch::Tensor tid){
  int R=p0.size(0);
  auto t=torch::empty({R},p0.options()); auto id=torch::empty({R},nleft.options());
  if(R) first_hit_k<<<(R+127)/128,128>>>(p0.data_ptr<float>(),p1.data_ptr<float>(),R,nlo.data_ptr<float>(),nhi.data_ptr<float>(),
     nleft.data_ptr<int>(),nstart.data_ptr<int>(),ncount.data_ptr<int>(),tri.data_ptr<float>(),tid.data_ptr<int>(),
     t.data_ptr<float>(),id.data_ptr<int>());
  return {t,id};
}
'''
_CPP = "torch::Tensor any_hit64(torch::Tensor p0, torch::Tensor p1, torch::Tensor nlo, torch::Tensor nhi, torch::Tensor nleft, torch::Tensor nstart, torch::Tensor ncount, torch::Tensor tri); std::vector<torch::Tensor> first_hit(torch::Tensor p0, torch::Tensor p1, torch::Tensor nlo, torch::Tensor nhi, torch::Tensor nleft, torch::Tensor nstart, torch::Tensor ncount, torch::Tensor tri, torch::Tensor tid);"
_mod = None


def _ext():
    global _mod
    if _mod is None:
        os.environ.setdefault('TORCH_EXTENSIONS_DIR', './fps_data/tmp/torch_ext')
        _mod = load_inline('fpsray2', cpp_sources=_CPP, cuda_sources=_SRC, functions=['first_hit', 'any_hit64'],
                           extra_cuda_cflags=['-O3', '--use_fast_math'] if False else ['-O3'], verbose=False)
    return _mod


class GpuMesh:
    def __init__(self, tri, device='cuda', leaf=4):
        b = build_bvh(tri, leaf)
        self.n_nodes = len(b['node_left'])
        self.t = {k: torch.as_tensor(v, device=device).contiguous() for k, v in b.items()}
        import numpy as _np
        self.tri64 = torch.as_tensor(_np.asarray(tri, _np.float64)[b['tri_id']].reshape(-1, 9), device=device).contiguous()

    def blocked(self, p0, p1):
        """float64 any-hit on the segment p0->p1 (same test as fpssim TriMesh.intersect: t in [0,1], |a| > 1e-12)."""
        t = self.t
        return _ext().any_hit64(p0.double().contiguous(), p1.double().contiguous(), t['node_lo'], t['node_hi'], t['node_left'],
                                t['node_start'], t['node_count'], self.tri64)

    def first_hit(self, p0, p1):
        t = self.t
        return _ext().first_hit(p0.float().contiguous(), p1.float().contiguous(), t['node_lo'], t['node_hi'], t['node_left'],
                                t['node_start'], t['node_count'], t['tri'], t['tri_id'])
