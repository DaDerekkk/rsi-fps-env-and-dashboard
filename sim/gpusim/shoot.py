"""Batched bullet resolution on the GPU (port of fpssim.arena._do_shot + fpssim.penetration + fpssim.combat.ray_hit_player).

One CUDA thread per pellet, all in float64: enemy hit volumes (cs2-web rayHitPlayer, platform-axis version validated 4226/4226),
world crossings along the ray via the BVH (all hits <= range, deduplicated at 2 mm, max 48), cs2-web penetration rules
(Dust2/Mirage meshes) or first-wall-only (whitebox maps, as the CPU arena does). Damage/armor is applied afterwards in torch,
sequentially by shooter seat (same order as the CPU arena).
"""
import math, os
import torch
from torch.utils.cpp_extension import load_inline

_SRC = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#define MAXC 64
#define NKEEP 10
__device__ inline double sphere_hit(double ox,double oy,double oz,double dx,double dy,double dz,double cx,double cy,double cz,double r){
  double px=ox-cx,py=oy-cy,pz=oz-cz; double b=px*dx+py*dy+pz*dz; double c=px*px+py*py+pz*pz-r*r; double h=b*b-c;
  if(h<0) return -1; double s=sqrt(h); double t=-b-s; if(t<0) t=-b+s; return t>=0? t:-1;
}
__device__ inline double box_hit(double o0,double o1,double o2,double d0,double d1,double d2,double l0,double l1,double l2,double h0,double h1,double h2){
  double tn=-1e300,tf=1e300; double o[3]={o0,o1,o2},d[3]={d0,d1,d2},lo[3]={l0,l1,l2},hi[3]={h0,h1,h2};
  for(int k=0;k<3;k++){
    if(fabs(d[k])<1e-12){ if(o[k]<lo[k]||o[k]>hi[k]) return -1; continue; }
    double a=(lo[k]-o[k])/d[k], b=(hi[k]-o[k])/d[k]; if(a>b){double t=a;a=b;b=t;}
    tn=fmax(tn,a); tf=fmin(tf,b); if(tn>tf) return -1;
  }
  if(tf<0) return -1; return tn>=0? tn: tf;
}
// returns distance (or -1) and part (0 head, 1 upper, 2 lower)
__device__ inline double ray_player(const double* o,const double* d,double px,double py,double pz,double yaw,bool crouch,double tmax,int* part){
  double cy=cos(yaw), sy=sin(yaw); double fx=cy,fy=sy, rx=sy,ry=-cy;
  double ox=o[0]-px, oy=o[1]-py;
  double l0=ox*rx+oy*ry, l1=o[2]-pz, l2=ox*fx+oy*fy;
  double d0=d[0]*rx+d[1]*ry, d1=d[2], d2=d[0]*fx+d[1]*fy;
  double h=crouch?1.10:1.80;
  double head=sphere_hit(l0,l1,l2,d0,d1,d2,0.0,h-0.16,-0.02,0.23);
  double up=box_hit(l0,l1,l2,d0,d1,d2,-0.38,crouch?0.45:0.75,-0.22,0.38,h-0.18,0.22);
  double low=box_hit(l0,l1,l2,d0,d1,d2,-0.30,0.0,-0.22,0.30,crouch?0.50:0.85,0.22);
  double body=-1; int bp=-1;
  if(up>=0 && (low<0 || up<=low)){body=up;bp=1;} else if(low>=0){body=low;bp=2;}
  if(head>=0 && head<=tmax && (body<0 || head<=body)){*part=0; return head;}
  if(body>=0 && body<=tmax){*part=bp; return body;}
  return -1;
}
// float32 pre-test of the float64 Moller-Trumbore below (same bound as gpusim.vis blocked64): true only when the float64 test
// certainly rejects the triangle (t outside [0,1], u / v outside the triangle or |a| <= 1e-12); otherwise the float64 test decides.
#define EPSK 1.9073486328125e-06f
__device__ __forceinline__ bool tri_reject(double ox,double oy,double oz,float dx,float dy,float dz,const double* v,const float* E){
  float e1x=E[0],e1y=E[1],e1z=E[2],e2x=E[3],e2y=E[4],e2z=E[5];
  float sx=(float)(ox-v[0]), sy=(float)(oy-v[1]), sz=(float)(oz-v[2]);
  float hx=__fsub_rn(__fmul_rn(dy,e2z),__fmul_rn(dz,e2y)), hy=__fsub_rn(__fmul_rn(dz,e2x),__fmul_rn(dx,e2z)), hz=__fsub_rn(__fmul_rn(dx,e2y),__fmul_rn(dy,e2x));
  float ahx=fabsf(dy*e2z)+fabsf(dz*e2y), ahy=fabsf(dz*e2x)+fabsf(dx*e2z), ahz=fabsf(dx*e2y)+fabsf(dy*e2x);
  float a=e1x*hx+e1y*hy+e1z*hz;
  float Ba=EPSK*(fabsf(e1x)*ahx+fabsf(e1y)*ahy+fabsf(e1z)*ahz);
  float aa=fabsf(a);
  if(!(aa>Ba+1e-12f)) return false;
  float sg=a>0.f?1.f:-1.f;
  float un=(sx*hx+sy*hy+sz*hz)*sg, Bu=EPSK*(fabsf(sx)*ahx+fabsf(sy)*ahy+fabsf(sz)*ahz);
  if(un<-Bu || un-aa>Bu+Ba) return true;
  float qx=__fsub_rn(__fmul_rn(sy,e1z),__fmul_rn(sz,e1y)), qy=__fsub_rn(__fmul_rn(sz,e1x),__fmul_rn(sx,e1z)), qz=__fsub_rn(__fmul_rn(sx,e1y),__fmul_rn(sy,e1x));
  float aqx=fabsf(sy*e1z)+fabsf(sz*e1y), aqy=fabsf(sz*e1x)+fabsf(sx*e1z), aqz=fabsf(sx*e1y)+fabsf(sy*e1x);
  float vn=(dx*qx+dy*qy+dz*qz)*sg, Bv=EPSK*(fabsf(dx)*aqx+fabsf(dy)*aqy+fabsf(dz)*aqz);
  if(vn<-Bv || un+vn-aa>Bu+Bv+Ba+EPSK*(fabsf(un)+fabsf(vn))) return true;
  float tn=(e2x*qx+e2y*qy+e2z*qz)*sg, Bt=EPSK*(fabsf(e2x)*aqx+fabsf(e2y)*aqy+fabsf(e2z)*aqz);
  return tn<-Bt || tn-aa>Bt+Ba;
}
__global__ void shoot_k(int P, const double* __restrict__ org, const double* __restrict__ dir, const long* __restrict__ env,
    const long* __restrict__ seat, const double* __restrict__ wrange, const double* __restrict__ wpower, const double* __restrict__ wdamage,
    const double* __restrict__ ppos, const double* __restrict__ pyaw, const bool* __restrict__ pcrouch, const bool* __restrict__ palive,
    int has_mesh, const float* __restrict__ nlo, const float* __restrict__ nhi, const int* __restrict__ nleft, const int* __restrict__ nstart,
    const int* __restrict__ ncount, const double* __restrict__ tri, const int* __restrict__ tid, const unsigned char* __restrict__ mat,
    int whitebox, const double* __restrict__ wb_tri, int wb_n, const float* __restrict__ e12,
    const float* __restrict__ wlo, const float* __restrict__ whi, const int* __restrict__ wleft, const int* __restrict__ wstart,
    const int* __restrict__ wcount, const float* __restrict__ wbe12,
    double* __restrict__ out_end, long* __restrict__ out_seat, double* __restrict__ out_dist, long* __restrict__ out_part,
    double* __restrict__ out_scale, long* __restrict__ out_nhit, long* __restrict__ out_npen,
    double* __restrict__ seg_entry, double* __restrict__ seg_exit, long* __restrict__ seg_mat){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=P) return;
  if(env[i]<0){ out_end[i]=0; out_nhit[i]=0; out_npen[i]=0; return; }     // unused row (gpusim.damage fixed-capacity buffer)
  const double* o=org+3*i; const double* d=dir+3*i; long e=env[i], sh=seat[i];
  double rng=wrange[i], power=wpower[i], dmg=wdamage[i];
  // ---- enemy hit volumes (team = seat/5)
  double hd[10]; int hs[10], hp[10]; int nh=0;
  for(int q=0;q<10;q++){
    if(q==sh || q/5==sh/5 || !palive[e*10+q]) continue;
    int part; double t=ray_player(o,d,ppos[(e*10+q)*3],ppos[(e*10+q)*3+1],ppos[(e*10+q)*3+2],pyaw[e*10+q],pcrouch[e*10+q],rng,&part);
    if(t>=0){ int k=nh++; while(k>0 && hd[k-1]>t){hd[k]=hd[k-1];hs[k]=hs[k-1];hp[k]=hp[k-1];k--;} hd[k]=t;hs[k]=q;hp[k]=part; }
  }
  // ---- world crossings
  double cd[MAXC]; int ct[MAXC]; int nc=0;
  double p1x=o[0]+d[0]*rng, p1y=o[1]+d[1]*rng, p1z=o[2]+d[2]*rng; double Dx=p1x-o[0],Dy=p1y-o[1],Dz=p1z-o[2];
  if(has_mesh){
    float ix=1.f/(float)Dx, iy=1.f/(float)Dy, iz=1.f/(float)Dz;
    // Near-first traversal with culling: crossings are kept sorted by (distance, triangle id) as they are found; once the deduplicated
    // list has NKEEP entries, nodes / crossings beyond the NKEEP-th kept crossing cannot change the first NKEEP kept crossings (the
    // dedupe of a sorted prefix is the prefix of the dedupe), and the trace below reads at most cd[0..9].  Only the far part of long
    // rays is skipped; the kept crossings and so the result are those of the full search (with more than MAXC raw crossings the
    // nearest MAXC are kept, the old code kept the first MAXC in traversal order).
    float fdx=(float)Dx, fdy=(float)Dy, fdz=(float)Dz;
    double cullT=1e300; float cullp=1.f;
    int st[64]; float sn[64]; int sp=0;
    #define NODE_BOX(N, LIM, TN, HIT) { const float* lo=nlo+3*(N); const float* hi=nhi+3*(N); float t0=0.f,t1=(LIM),a,b; \
      a=(lo[0]-1e-4f-(float)o[0])*ix; b=(hi[0]+1e-4f-(float)o[0])*ix; t0=fmaxf(t0,fminf(a,b)); t1=fminf(t1,fmaxf(a,b)); \
      a=(lo[1]-1e-4f-(float)o[1])*iy; b=(hi[1]+1e-4f-(float)o[1])*iy; t0=fmaxf(t0,fminf(a,b)); t1=fminf(t1,fmaxf(a,b)); \
      a=(lo[2]-1e-4f-(float)o[2])*iz; b=(hi[2]+1e-4f-(float)o[2])*iz; t0=fmaxf(t0,fminf(a,b)); t1=fminf(t1,fmaxf(a,b)); \
      HIT=!(t0>t1+1e-6f); TN=t0; }
    { float tn; bool h; NODE_BOX(0,1.f,tn,h); if(h){ st[0]=0; sn[0]=tn; sp=1; } }
    while(sp){
      sp--; int n=st[sp]; if(sn[sp]>cullp) continue;
      int l=nleft[n];
      if(l>=0){
        float tl,tr; bool hl,hr; NODE_BOX(l,cullp,tl,hl); NODE_BOX(l+1,cullp,tr,hr);
        if(hl && hr){ if(tl<=tr){ st[sp]=l+1; sn[sp++]=tr; st[sp]=l; sn[sp++]=tl; } else { st[sp]=l; sn[sp++]=tl; st[sp]=l+1; sn[sp++]=tr; } }
        else if(hl){ st[sp]=l; sn[sp++]=tl; } else if(hr){ st[sp]=l+1; sn[sp++]=tr; }
        continue;
      }
      for(int k=nstart[n];k<nstart[n]+ncount[n];k++){
        const double* v=tri+9*k;
        if(tri_reject(o[0],o[1],o[2],fdx,fdy,fdz,v,e12+6*k)) continue;
        double e1x=v[3]-v[0],e1y=v[4]-v[1],e1z=v[5]-v[2],e2x=v[6]-v[0],e2y=v[7]-v[1],e2z=v[8]-v[2];
        double hx=Dy*e2z-Dz*e2y, hy=Dz*e2x-Dx*e2z, hz=Dx*e2y-Dy*e2x; double aa=e1x*hx+e1y*hy+e1z*hz;
        if(fabs(aa)<=1e-12) continue; double f=1.0/aa; double sx=o[0]-v[0],sy=o[1]-v[1],sz=o[2]-v[2];
        double u=f*(sx*hx+sy*hy+sz*hz); if(u<0||u>1) continue;
        double qx=sy*e1z-sz*e1y,qy=sz*e1x-sx*e1z,qz=sx*e1y-sy*e1x; double vv=f*(Dx*qx+Dy*qy+Dz*qz); if(vv<0||u+vv>1) continue;
        double t=f*(e2x*qx+e2y*qy+e2z*qz); if(t<0||t>1) continue;
        double dd=t*rng; if(dd>cullT) continue;
        int kt=tid[k];
        if(nc==MAXC){ if(cd[MAXC-1]<dd || (cd[MAXC-1]==dd && tid[ct[MAXC-1]]<kt)) continue; nc--; }
        int p=nc++;
        while(p>0 && (cd[p-1]>dd || (cd[p-1]==dd && tid[ct[p-1]]>kt))){ cd[p]=cd[p-1]; ct[p]=ct[p-1]; p--; }
        cd[p]=dd; ct[p]=k;
        int mk=0; double lk=0;
        for(int a2=0;a2<nc;a2++){ if(mk && cd[a2]-lk<.002) continue; lk=cd[a2]; if(++mk==NKEEP){ cullT=cd[a2]; cullp=fminf(1.f,(float)(cullT/rng)*1.0001f+1e-5f); break; } }
      }
    }
    // sort ascending by distance, ties by original triangle id (CPU: argsort of t over unique candidate ids)
    for(int a2=1;a2<nc;a2++){ double x=cd[a2]; int y=ct[a2]; int b2=a2-1;
      while(b2>=0 && (cd[b2]>x || (cd[b2]==x && tid[ct[b2]]>tid[y]))){cd[b2+1]=cd[b2];ct[b2+1]=ct[b2];b2--;} cd[b2+1]=x; ct[b2+1]=y; }
    // dedupe < 2 mm
    int m=0; for(int a2=0;a2<nc && m<48;a2++){ if(m && cd[a2]-cd[m-1]<.002) continue; cd[m]=cd[a2]; ct[m]=ct[a2]; m++; } nc=m;
  }
  // ---- trace (cs2-web traceBullet); whitebox: first wall only
  const double DEPTH[4]={.09,.70,.14,.90}, LOSS[4]={.48,.14,.34,.08};
  double scale=1, stop=rng; bool wall=false, hitw=false; int pens=0, ti=0, nout=0;
  double wbt=-1;
  if(whitebox){ // first hit on the whitebox triangle soup: BVH (built over the same triangles; the minimum is order independent)
    float ix=1.f/(float)Dx, iy=1.f/(float)Dy, iz=1.f/(float)Dz; float fdx=(float)Dx, fdy=(float)Dy, fdz=(float)Dz;
    int st[64]; int sp=0; st[sp++]=0;
    while(sp){
      int n=st[--sp];
      const float* lo=wlo+3*n; const float* hi=whi+3*n;
      float t0=0.f,t1= wbt>=0? fminf(1.f,(float)(wbt/rng)*1.0001f+1e-5f) : 1.f, a1,b1;
      a1=(lo[0]-1e-4f-(float)o[0])*ix; b1=(hi[0]+1e-4f-(float)o[0])*ix; t0=fmaxf(t0,fminf(a1,b1)); t1=fminf(t1,fmaxf(a1,b1));
      a1=(lo[1]-1e-4f-(float)o[1])*iy; b1=(hi[1]+1e-4f-(float)o[1])*iy; t0=fmaxf(t0,fminf(a1,b1)); t1=fminf(t1,fmaxf(a1,b1));
      a1=(lo[2]-1e-4f-(float)o[2])*iz; b1=(hi[2]+1e-4f-(float)o[2])*iz; t0=fmaxf(t0,fminf(a1,b1)); t1=fminf(t1,fmaxf(a1,b1));
      if(t0>t1+1e-6f) continue;
      int l=wleft[n]; if(l>=0){ st[sp++]=l; st[sp++]=l+1; continue; }
     for(int k=wstart[n];k<wstart[n]+wcount[n];k++){
      const double* v=wb_tri+9*k;
      if(tri_reject(o[0],o[1],o[2],fdx,fdy,fdz,v,wbe12+6*k)) continue;
      double e1x=v[3]-v[0],e1y=v[4]-v[1],e1z=v[5]-v[2],e2x=v[6]-v[0],e2y=v[7]-v[1],e2z=v[8]-v[2];
      double hx=Dy*e2z-Dz*e2y, hy=Dz*e2x-Dx*e2z, hz=Dx*e2y-Dy*e2x; double aa=e1x*hx+e1y*hy+e1z*hz;
      if(fabs(aa)<=1e-12) continue; double f=1.0/aa; double sx=o[0]-v[0],sy=o[1]-v[1],sz=o[2]-v[2];
      double u=f*(sx*hx+sy*hy+sz*hz); if(u<0||u>1) continue;
      double qx=sy*e1z-sz*e1y,qy=sz*e1x-sx*e1z,qz=sx*e1y-sy*e1x; double vv=f*(Dx*qx+Dy*qy+Dz*qz); if(vv<0||u+vv>1) continue;
      double t=f*(e2x*qx+e2y*qy+e2z*qz); if(t<0||t>1) continue;
      if(wbt<0 || t*rng<wbt) wbt=t*rng;
     }
    }
    double wd= wbt>=0? wbt: rng;
    if(nh>0 && hd[0]<wd){ out_seat[i*4]=hs[0]; out_dist[i*4]=hd[0]; out_part[i*4]=hp[0]; out_scale[i*4]=1; nout=1; stop=hd[0]; }
    else stop=wd;
    out_end[i]=stop; out_nhit[i]=nout; out_npen[i]=0; return;
  }
  bool done=false;
  #define HIT_UNTIL(U) { while(ti<nh && hd[ti]<(U)){ out_seat[i*4+nout]=hs[ti]; out_dist[i*4+nout]=hd[ti]; out_part[i*4+nout]=hp[ti]; out_scale[i*4+nout]=scale; nout++; double hdist=hd[ti]; ti++; scale*=.55; if(nout>=4 || scale*dmg<1){ stop=hdist; done=true; break; } } }
  int c=0;
  while(c<nc){
    double ed=cd[c]; if(ed>rng) break;
    HIT_UNTIL(ed); if(done) break;
    const double* v=tri+9*ct[c]; int mi=mat? mat[tid[ct[c]]] : 0; if(mi>3) mi=0;
    double maxd=DEPTH[mi]*power;
    bool valid=false; double thick=1e300;
    if(c+1<nc){
      thick=cd[c+1]-ed;
      const double* w=tri+9*ct[c+1]; int mx=mat? mat[tid[ct[c+1]]]:0; if(mx>3) mx=0;
      double nx1=-((v[4]-v[1])*(v[8]-v[2])-(v[5]-v[2])*(v[7]-v[1])), ny1=-((v[5]-v[2])*(v[6]-v[0])-(v[3]-v[0])*(v[8]-v[2])), nz1=-((v[3]-v[0])*(v[7]-v[1])-(v[4]-v[1])*(v[6]-v[0]));
      double nx2=-((w[4]-w[1])*(w[8]-w[2])-(w[5]-w[2])*(w[7]-w[1])), ny2=-((w[5]-w[2])*(w[6]-w[0])-(w[3]-w[0])*(w[8]-w[2])), nz2=-((w[3]-w[0])*(w[7]-w[1])-(w[4]-w[1])*(w[6]-w[0]));
      double n1=sqrt(nx1*nx1+ny1*ny1+nz1*nz1), n2=sqrt(nx2*nx2+ny2*ny2+nz2*nz2); if(n1==0)n1=1; if(n2==0)n2=1;
      double dn1=(nx1*d[0]+ny1*d[1]+nz1*d[2])/n1, dn2=(nx2*d[0]+ny2*d[1]+nz2*d[2])/n2;
      valid= dn1<-.01 && dn2>.01 && mx==mi && thick>.002;
    }
    if(power==0 || pens>=4 || !valid || thick>maxd){ stop=ed; hitw=true; break; }
    double rem=scale*(1-LOSS[mi])-thick/fmax(.01,maxd)*.42;
    if(rem*dmg<1){ stop=ed; hitw=true; break; }
    if(pens<4){ seg_entry[i*4+pens]=ed; seg_exit[i*4+pens]=cd[c+1]; seg_mat[i*4+pens]=mi; }
    scale=rem; pens++; double xd=cd[c+1]; c+=2;
    while(ti<nh && hd[ti]<xd) ti++;
  }
  if(!done && !hitw && stop==rng){ HIT_UNTIL(rng); }
  if(!hitw && nout>0) stop=fmin(stop, out_dist[i*4+nout-1]+.15);
  out_end[i]=stop; out_nhit[i]=nout; out_npen[i]=pens;
}
void shoot(torch::Tensor org, torch::Tensor dir, torch::Tensor env, torch::Tensor seat, torch::Tensor wrange, torch::Tensor wpower,
    torch::Tensor wdamage, torch::Tensor ppos, torch::Tensor pyaw, torch::Tensor pcrouch, torch::Tensor palive, int64_t has_mesh,
    torch::Tensor nlo, torch::Tensor nhi, torch::Tensor nleft, torch::Tensor nstart, torch::Tensor ncount, torch::Tensor tri,
    torch::Tensor tid, torch::Tensor mat, int64_t has_mat, int64_t whitebox, torch::Tensor wb_tri, torch::Tensor e12, torch::Tensor wlo,
    torch::Tensor whi, torch::Tensor wleft, torch::Tensor wstart, torch::Tensor wcount, torch::Tensor wbe12, torch::Tensor out_end, torch::Tensor out_seat,
    torch::Tensor out_dist, torch::Tensor out_part, torch::Tensor out_scale, torch::Tensor out_nhit, torch::Tensor out_npen,
    torch::Tensor seg_entry, torch::Tensor seg_exit, torch::Tensor seg_mat){
  int P=org.size(0); if(!P) return;
  shoot_k<<<(P+63)/64,64,0,at::cuda::getCurrentCUDAStream()>>>(P,org.data_ptr<double>(),dir.data_ptr<double>(),env.data_ptr<long>(),seat.data_ptr<long>(),
    wrange.data_ptr<double>(),wpower.data_ptr<double>(),wdamage.data_ptr<double>(),ppos.data_ptr<double>(),pyaw.data_ptr<double>(),
    pcrouch.data_ptr<bool>(),palive.data_ptr<bool>(),(int)has_mesh,nlo.data_ptr<float>(),nhi.data_ptr<float>(),nleft.data_ptr<int>(),
    nstart.data_ptr<int>(),ncount.data_ptr<int>(),tri.data_ptr<double>(),tid.data_ptr<int>(),has_mat? mat.data_ptr<unsigned char>():nullptr,
    (int)whitebox,wb_tri.data_ptr<double>(),(int)wb_tri.size(0),e12.data_ptr<float>(),wlo.data_ptr<float>(),whi.data_ptr<float>(),
    wleft.data_ptr<int>(),wstart.data_ptr<int>(),wcount.data_ptr<int>(),wbe12.data_ptr<float>(),out_end.data_ptr<double>(),out_seat.data_ptr<long>(),out_dist.data_ptr<double>(),
    out_part.data_ptr<long>(),out_scale.data_ptr<double>(),out_nhit.data_ptr<long>(),out_npen.data_ptr<long>(),
    seg_entry.data_ptr<double>(),seg_exit.data_ptr<double>(),seg_mat.data_ptr<long>());
}
'''
_CPP = "void shoot(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);"
_mod = None


def _ext():
    global _mod
    if _mod is None:
        os.environ.setdefault('TORCH_EXTENSIONS_DIR', './fps_data/tmp/torch_ext')
        _mod = load_inline('fpsshoot2', cpp_sources=_CPP, cuda_sources=_SRC, functions=['shoot'], extra_cuda_cflags=['-O3'])
    return _mod


class Shooter:
    """mesh: gpusim.ray.GpuMesh (Dust2/Mirage) built from the float64 world triangles, with materials; or whitebox triangles."""

    def __init__(self, device, mesh=None, tri64=None, materials=None, wb_tri=None):
        self.dev = device
        self.has_mesh = mesh is not None
        z = lambda dt: torch.zeros(1, dtype=dt, device=device)
        if mesh is not None:
            t = mesh.t
            self.nodes = (t['node_lo'], t['node_hi'], t['node_left'], t['node_start'], t['node_count'])
            self.tri = torch.as_tensor(tri64, dtype=torch.float64, device=device)[t['tri_id'].long()].reshape(-1, 9).contiguous()
            self.tid = t['tri_id']
            self.mat = torch.as_tensor(materials, dtype=torch.uint8, device=device) if materials is not None else z(torch.uint8)
            self.has_mat = materials is not None
        else:
            self.nodes = (z(torch.float32), z(torch.float32), z(torch.int32), z(torch.int32), z(torch.int32))
            self.tri = z(torch.float64).reshape(1, 1).expand(1, 9).contiguous(); self.tid = z(torch.int32); self.mat = z(torch.uint8); self.has_mat = False
        self.whitebox = wb_tri is not None
        e12 = lambda t9: torch.cat([t9[:, 3:6] - t9[:, 0:3], t9[:, 6:9] - t9[:, 0:3]], 1).float().contiguous()   # float64 edges, rounded once
        self.e12 = e12(self.tri)
        if wb_tri is not None:                          # BVH over the whitebox triangles (first-hit search; was brute force)
            from .bvh import build_bvh
            import numpy as np
            wt = np.asarray(wb_tri, np.float64).reshape(-1, 3, 3)
            b = build_bvh(wt, 4)
            self.wb = torch.as_tensor(wt[b['tri_id']].reshape(-1, 9), device=device).contiguous()
            self.wbvh = [torch.as_tensor(b[k], device=device).contiguous() for k in ('node_lo', 'node_hi', 'node_left', 'node_start', 'node_count')]
        else:
            self.wb = torch.zeros(1, 9, dtype=torch.float64, device=device)
            self.wbvh = [z(torch.float32), z(torch.float32), z(torch.int32), z(torch.int32), z(torch.int32)]
        self.wbe12 = e12(self.wb)

    def trace(self, org, dir, env, seat, wrange, wpower, wdamage, ppos, pyaw, pcrouch, palive, out=None):
        """rows with env < 0 are skipped (nhit 0).  out: optional dict of preallocated result tensors (as returned before) to reuse."""
        P = org.shape[0]; d = self.dev
        if out is None or out['end'].shape[0] != P: out = dict(end=torch.zeros(P, dtype=torch.float64, device=d), seat=torch.full((P, 4), -1, dtype=torch.long, device=d),
                   dist=torch.zeros(P, 4, dtype=torch.float64, device=d), part=torch.zeros(P, 4, dtype=torch.long, device=d),
                   scale=torch.zeros(P, 4, dtype=torch.float64, device=d), nhit=torch.zeros(P, dtype=torch.long, device=d),
                   npen=torch.zeros(P, dtype=torch.long, device=d), seg_entry=torch.zeros(P, 4, dtype=torch.float64, device=d),
                   seg_exit=torch.zeros(P, 4, dtype=torch.float64, device=d), seg_mat=torch.zeros(P, 4, dtype=torch.long, device=d))
        _ext().shoot(org.contiguous(), dir.contiguous(), env.contiguous(), seat.contiguous(), wrange.contiguous(), wpower.contiguous(),
                     wdamage.contiguous(), ppos.contiguous(), pyaw.contiguous(), pcrouch.contiguous(), palive.contiguous(), int(self.has_mesh),
                     *self.nodes, self.tri, self.tid, self.mat, int(self.has_mat), int(self.whitebox), self.wb, self.e12, *self.wbvh, self.wbe12,
                     out['end'], out['seat'], out['dist'], out['part'], out['scale'], out['nhit'], out['npen'],
                     out['seg_entry'], out['seg_exit'], out['seg_mat'])
        return out
