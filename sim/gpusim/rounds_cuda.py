"""CUDA kernels for gpusim.rounds / gpusim.damage (round machine, economy, buy/refund/drop/pickup, bomb + objective,
kill handling, bullet spread rays, damage settlement).  One thread per match (E threads) for everything that is
sequential by seat in the CPU Arena; one thread per pellet for the ray builder.

All state is reached through a table of raw device pointers (`struct S`, generated from the tensor registry in
gpusim/rounds.py), so a kernel launch costs one tensor argument.  Everything is float64 / int; compile flags `-fmad=false`
keep the arithmetic identical to the python reference (fpssim.arena / combat / accuracy / economy).

Random numbers are counter based: u01(seed, purpose, a, b, c) = splitmix64 chain (python twin: gpusim.rounds.u01_py).
"""
import os
import torch
from torch.utils.cpp_extension import load_inline

EVENTS = ['fire', 'hit', 'kill', 'kill_reward', 'buy', 'refund', 'drop', 'pickup', 'item_create', 'item_destroy', 'cash',
          'plant_start', 'plant_cancel', 'plant_complete', 'defuse_start', 'defuse_cancel', 'defuse_complete',
          'bomb_explode', 'round_start', 'round_end', 'match_end', 'side_swap', 'economy_round_start', 'round_economy',
          'reload_start', 'reload_commit', 'reload_finish', 'draw', 'void_shot', 'dead_hit']
RECEIPT_KINDS = ['buy', 'refund', 'drop', 'pickup']
RECEIPT_REASONS = ['none', 'buy_unavailable', 'wrong_side', 'already_owned', 'limit', 'round_limit', 'insufficient_cash',
                   'unknown_item', 'refund_unavailable', 'not_refundable', 'unavailable', 'no_bomb', 'nothing', 'no_id',
                   'not_found', 'too_far', 'unsupported']
ROUND_REASONS = ['elimination', 'timeout', 'bomb_explode', 'defuse']

CT = {torch.float64: 'double', torch.int64: 'long long', torch.int32: 'int', torch.int16: 'short', torch.int8: 'signed char',
      torch.bool: 'bool', torch.float32: 'float', torch.uint8: 'unsigned char'}


def header(fields):
    """fields: list of (name, torch dtype).  Returns the C++ definitions shared by every kernel."""
    out = ['#include <torch/extension.h>', '#include <ATen/cuda/CUDAContext.h>', '#include <stdint.h>', '#include <math.h>',
           'typedef unsigned long long u64;']
    for i, n in enumerate(EVENTS):
        out.append(f'#define EV_{n.upper()} {i}')
    out.append(f'#define NEV {len(EVENTS)}')
    for i, n in enumerate(RECEIPT_REASONS):
        out.append(f'#define RS_{n.upper()} {i}')
    out.append('struct S {')
    out += [f'  {CT[dt]}* {n};' for n, dt in fields]
    out.append('};')
    return '\n'.join(out) + '\n'


BODY = r'''
#define PHASE_BUY 0
#define PHASE_LIVE 1
#define PHASE_POST 2
#define PHASE_FIN 3
#define B_HELD 0
#define B_WORLD 1
#define B_PLANTED 2
#define B_DEFUSED 3
#define B_EXPLODED 4
#define R_ELIM 0
#define R_TIMEOUT 1
#define R_EXPLODE 2
#define R_DEFUSE 3
#define OBJ_PLANT 1
#define OBJ_DEFUSE 2
#define CAP 16000
#define MAXPAD 16
#define MAXZ 4
#define MAXV 128
// cfg_i: 0 K, 1 npad0, 2 npad1, 3 nzones, 4 has_mesh, 5 whitebox(z offset), 6 pz_circle(1: sites instead of polygons)
// cfg_d: 0 z offset, 1 buy radius, 2 buy vertical limit
#define CK ((int)s.cfg_i[0])
#define PF(f,i) s.f[(long long)e*10+(i)]
#define WF(f,i,k) s.f[((long long)e*10+(i))*3+(k)]
#define EF(f) s.f[e]
#define EVN(k) (s.ev_count[(long long)e*NEV+(k)]++)
#define WI(f,k) s.f[(long long)e*CK+(k)]
#define WI3(f,k,c) s.f[((long long)e*CK+(k))*3+(c)]
#define WI7(f,k,c) s.f[((long long)e*CK+(k))*7+(c)]
#define PUR(f,i,k) s.f[((long long)e*10+(i))*NB+(k)]
#define NADE(f,i,k) s.f[((long long)e*10+(i))*6+(k)]

// ---------------------------------------------------------------- RNG (python twin: gpusim.rounds.u01_py)
__device__ __forceinline__ u64 mix64(u64 z){
  z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31);
}
__device__ __forceinline__ double u01(long long seed,int purpose,long long a,long long b,long long c){
  u64 h=mix64((u64)seed ^ 0xA0761D6478BD642FULL);
  h=mix64(h + (u64)purpose*0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL);
  h=mix64(h + (u64)a*0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL);
  h=mix64(h + (u64)b*0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL);
  h=mix64(h + (u64)c*0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL);
  return (double)(h>>11) * (1.0/9007199254740992.0);
}
#define P_SPREAD 1
#define P_SPAWN 2
#define P_CARRIER 3
#define P_ATTACK 4

// ---------------------------------------------------------------- geometry helpers
__device__ inline double eye_z(const S& s,int e,int i){ return PF(z,i) + (PF(crouch,i) ? 0.95 : 1.62); }
__device__ inline bool is_attack(const S& s,int e,int i){ return (i/5)==(int)EF(attack_team); }
__device__ inline double dist3(double ax,double ay,double az,double bx,double by,double bz){
  double dx=ax-bx,dy=ay-by,dz=az-bz; return sqrt(dx*dx+dy*dy+dz*dz);
}

__device__ bool poly_in(const double* pts,int n,double x,double y){
  bool inside=false; int j=n-1;
  for(int i=0;i<n;i++){
    double xi=pts[2*i],yi=pts[2*i+1],xj=pts[2*j],yj=pts[2*j+1];
    if((yi>y)!=(yj>y) && x < (xj-xi)*(y-yi)/(yj-yi+1e-300)+xi) inside=!inside;
    j=i;
  }
  return inside;
}
// returns site index (0 = first zone) or -1
__device__ int in_plant_zone(const S& s,double x,double y,double z){
  int nz=(int)s.cfg_i[3];
  if(s.cfg_i[6]){   // sites as circles
    for(int k=0;k<nz;k++){ const double* c=s.pz_circ+4*k; if(hypot(x-c[0],y-c[1])<=c[2]) return k; }
    return -1;
  }
  for(int k=0;k<nz;k++){
    if(s.pz_z[2*k]<=z && z<=s.pz_z[2*k+1] && poly_in(s.pz_poly+(long long)k*MAXV*2,(int)s.pz_n[k],x,y)) return k;
  }
  return -1;
}
__device__ bool in_buy_region(const S& s,int e,int i){
  int side=is_attack(s,e,i)?0:1; int n=(int)s.cfg_i[1+side];
  double x=PF(x,i),y=PF(y,i),z=PF(z,i);
  for(int k=0;k<n;k++){
    const double* p=s.pad+((side*MAXPAD)+k)*4;
    if(hypot(x-p[0],y-p[1])<s.cfg_d[1] && fabs(z-p[2])<s.cfg_d[2]) return true;
  }
  return false;
}

// float64 any-hit of segment p0->p1 against the BVH (same test as gpusim.ray.GpuMesh.blocked / fpssim TriMesh.intersect)
__device__ bool seg_blocked(const S& s,double ox,double oy,double oz,double px,double py,double pz){
  if(!s.cfg_i[4]) return false;
  double Dx=px-ox,Dy=py-oy,Dz=pz-oz;
  float fo0=(float)ox,fo1=(float)oy,fo2=(float)oz; float i0=1.f/(float)Dx,i1=1.f/(float)Dy,i2=1.f/(float)Dz;
  int st[64]; int sp=0; st[sp++]=0;
  while(sp){
    int n=st[--sp];
    float t0=0.f,t1=1.0f+1e-6f,a,b;
    a=(s.bv_lo[3*n]-1e-4f-fo0)*i0; b=(s.bv_hi[3*n]+1e-4f-fo0)*i0; t0=fmaxf(t0,fminf(a,b)); t1=fminf(t1,fmaxf(a,b));
    a=(s.bv_lo[3*n+1]-1e-4f-fo1)*i1; b=(s.bv_hi[3*n+1]+1e-4f-fo1)*i1; t0=fmaxf(t0,fminf(a,b)); t1=fminf(t1,fmaxf(a,b));
    a=(s.bv_lo[3*n+2]-1e-4f-fo2)*i2; b=(s.bv_hi[3*n+2]+1e-4f-fo2)*i2; t0=fmaxf(t0,fminf(a,b)); t1=fminf(t1,fmaxf(a,b));
    if(!(t0<=t1)) continue;
    int l=s.bv_left[n]; if(l>=0){ st[sp++]=l; st[sp++]=l+1; continue; }
    for(int k=s.bv_start[n];k<s.bv_start[n]+s.bv_count[n];k++){
      const double* v=s.bv_tri+9*(long long)k;
      double e1x=v[3]-v[0],e1y=v[4]-v[1],e1z=v[5]-v[2],e2x=v[6]-v[0],e2y=v[7]-v[1],e2z=v[8]-v[2];
      double hx=Dy*e2z-Dz*e2y, hy=Dz*e2x-Dx*e2z, hz=Dx*e2y-Dy*e2x; double aa=e1x*hx+e1y*hy+e1z*hz;
      if(fabs(aa)<=1e-12) continue; double f=1.0/aa; double sx=ox-v[0],sy=oy-v[1],sz=oz-v[2];
      double u=f*(sx*hx+sy*hy+sz*hz); if(u<0||u>1) continue;
      double qx=sy*e1z-sz*e1y,qy=sz*e1x-sx*e1z,qz=sx*e1y-sy*e1x; double vv=f*(Dx*qx+Dy*qy+Dz*qz); if(vv<0||u+vv>1) continue;
      double t=f*(e2x*qx+e2y*qy+e2z*qz); if(t>=0 && t<=1) return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------- receipts / economy
__device__ inline void rcpt(const S& s,int e,int i,int kind,int ok,int reason,int item){
  long long b=(((long long)e*10+i)*4+kind)*3; s.rcpt[b]=ok; s.rcpt[b+1]=reason; s.rcpt[b+2]=item;
}
__device__ void cash_add(const S& s,int e,int i,long long delta){
  long long before=PF(cash,i), after=before+delta; if(after>CAP) after=CAP; if(after<0) after=0;
  PF(cash,i)=(int)after; EVN(EV_CASH);
}
__device__ inline void pur_clear(const S& s,int e,int i,int k){ PUR(pur_has,i,k)=0; PUR(pur_price,i,k)=0; PUR(pur_count,i,k)=0; }

// ---------------------------------------------------------------- weapons inventory
__device__ void give_slot(const S& s,int e,int i,int slot,int wid){
  WF(w_wid,i,slot)=wid; WF(w_ammo,i,slot)=s.T_magazine[wid]; WF(w_reserve,i,slot)=s.T_reserve[wid];
  WF(w_shots,i,slot)=0; WF(w_shot_ready,i,slot)=0; WF(w_reload_until,i,slot)=-1; WF(w_reload_commit,i,slot)=-1;
  WF(w_last_shot,i,slot)=-1; WF(w_reload_committed,i,slot)=0; WF(w_reload_shell,i,slot)=0; WF(w_pressed,i,slot)=0;
  WF(w_inacc,i,slot)=0.0;
  for(int c=0;c<7;c++) s.w_rs[(((long long)e*10+i)*3+slot)*7+c]=0.0;
}
__device__ void clear_slots(const S& s,int e,int i){ for(int k=0;k<3;k++) WF(w_wid,i,k)=-1; }
__device__ void give_default(const S& s,int e,int i){
  give_slot(s,e,i,1,is_attack(s,e,i)? (int)s.cfg_i[7] : (int)s.cfg_i[8]); EVN(EV_ITEM_CREATE);
  give_slot(s,e,i,2,(int)s.knife_id[0]); EVN(EV_ITEM_CREATE);
}
__device__ void switch_to(const S& s,int e,int i,int slot){
  if(WF(w_wid,i,slot)<0 || PF(active,i)==slot) return;
  PF(active,i)=slot; PF(zoom,i)=0; PF(switch_until,i)=EF(tick)+11;
}
__device__ void drop_weapon(const S& s,int e,int i,int slot,bool death){
  int w=(int)WF(w_wid,i,slot); if(w<0) return;
  if(w==(int)s.knife_id[0] && !death) return;
  int k=EF(wi_count);
  if(k>=CK){ k=-1; for(int j=0;j<CK;j++) if(!WI(wi_valid,j)){ k=j; break; } }    // full: reuse a free slot
  else EF(wi_count)=k+1;
  if(k>=0){
    WI(wi_valid,k)=1; WI(wi_kind,k)=w; WI(wi_owner,k)=i; WI(wi_tick,k)=EF(tick);
    WI3(wi_pos,k,0)=PF(x,i); WI3(wi_pos,k,1)=PF(y,i); WI3(wi_pos,k,2)=PF(z,i);
    WI(wi_ammo,k)=WF(w_ammo,i,slot); WI(wi_reserve,k)=WF(w_reserve,i,slot); WI(wi_shots,k)=WF(w_shots,i,slot);
    WI(wi_shot_ready,k)=WF(w_shot_ready,i,slot); WI(wi_reload_until,k)=WF(w_reload_until,i,slot);
    WI(wi_reload_commit,k)=WF(w_reload_commit,i,slot); WI(wi_last_shot,k)=WF(w_last_shot,i,slot);
    WI(wi_committed,k)=WF(w_reload_committed,i,slot); WI(wi_shell,k)=WF(w_reload_shell,i,slot);
    WI(wi_pressed,k)=WF(w_pressed,i,slot); WI(wi_inacc,k)=WF(w_inacc,i,slot);
    for(int c=0;c<7;c++) WI7(wi_rs,k,c)=s.w_rs[(((long long)e*10+i)*3+slot)*7+c];
  } else EF(wi_overflow)++;
  WF(w_wid,i,slot)=-1;
  EVN(EV_DROP);
  if(PF(active,i)==slot){ PF(active,i)=(WF(w_wid,i,0)>=0)?0:1; PF(switch_until,i)=EF(tick)+11; }
  pur_clear(s,e,i,w);
}
__device__ void take_item(const S& s,int e,int i,int slot,int k){
  WF(w_wid,i,slot)=WI(wi_kind,k); WF(w_ammo,i,slot)=WI(wi_ammo,k); WF(w_reserve,i,slot)=WI(wi_reserve,k);
  WF(w_shots,i,slot)=WI(wi_shots,k); WF(w_shot_ready,i,slot)=WI(wi_shot_ready,k); WF(w_reload_until,i,slot)=WI(wi_reload_until,k);
  WF(w_reload_commit,i,slot)=WI(wi_reload_commit,k); WF(w_last_shot,i,slot)=WI(wi_last_shot,k);
  WF(w_reload_committed,i,slot)=WI(wi_committed,k); WF(w_reload_shell,i,slot)=WI(wi_shell,k);
  WF(w_pressed,i,slot)=WI(wi_pressed,k); WF(w_inacc,i,slot)=WI(wi_inacc,k);
  for(int c=0;c<7;c++) s.w_rs[(((long long)e*10+i)*3+slot)*7+c]=WI7(wi_rs,k,c);
}

// ---------------------------------------------------------------- bomb / objective
__device__ void drop_bomb(const S& s,int e,int i,bool voluntary){
  if(!PF(has_bomb,i)) return;
  PF(has_bomb,i)=0;
  EF(bomb_state)=B_WORLD; EF(bomb_carrier)=-1;
  s.bomb_pos[e*3]=PF(x,i); s.bomb_pos[e*3+1]=PF(y,i); s.bomb_pos[e*3+2]=PF(z,i);
  EF(bomb_drop_tick)=EF(tick); EF(bomb_dropper)=voluntary? i : -1;
  EVN(EV_DROP);
}
__device__ void cancel_obj(const S& s,int e,int i){
  if(PF(objective,i)==OBJ_PLANT) EVN(EV_PLANT_CANCEL); else if(PF(objective,i)==OBJ_DEFUSE) EVN(EV_DEFUSE_CANCEL);
  PF(objective,i)=0; PF(obj_progress,i)=0;
  if(EF(bomb_actor)==i) EF(bomb_actor)=-1;
}
__device__ bool try_pick_bomb(const S& s,int e,int i,bool explicit_){
  double bx=s.bomb_pos[e*3],by=s.bomb_pos[e*3+1],bz=s.bomb_pos[e*3+2];
  double ex=PF(x,i),ey=PF(y,i),ez=eye_z(s,e,i);
  if(dist3(PF(x,i),PF(y,i),PF(z,i),bx,by,bz)>=2.3 && dist3(ex,ey,ez,bx,by,bz)>=2.3) return false;
  if(EF(bomb_dropper)==i && EF(tick)-EF(bomb_drop_tick)<90) return false;
  if(seg_blocked(s,ex,ey,ez,bx,by,bz+0.4)) return false;
  PF(has_bomb,i)=1; EF(bomb_state)=B_HELD; EF(bomb_carrier)=i;
  EVN(EV_PICKUP);
  if(explicit_) rcpt(s,e,i,3,1,RS_NONE,s.cfg_i[0]);
  return true;
}
__device__ inline bool can_obj(const S& s,int e,int i){
  bool slow = hypot(PF(vx,i),PF(vy,i)) <= 0.1+1e-9;
  return (PF(alive,i) && PF(grounded,i) && slow && PF(zoom,i)==0) || (PF(alive,i) && PF(grounded,i) && PF(objective,i)!=0 && slow);
}

// ---------------------------------------------------------------- kills (Arena._kill)
// weapon < 0: not a catalogue weapon (grenades etc.) -> reward 300.  restore: the victim has not had its weapon-tick turn yet in
// this tick of the CPU loop (it is never ticked there): put back the pre-tick weapon state before dropping its weapons.
__device__ void restore_weapons(const S& s,int e,int v){
  for(int k=0;k<3;k++){
    WF(w_ammo,v,k)=WF(ps_w_ammo,v,k); WF(w_reserve,v,k)=WF(ps_w_reserve,v,k); WF(w_shots,v,k)=WF(ps_w_shots,v,k);
    WF(w_shot_ready,v,k)=WF(ps_w_shot_ready,v,k); WF(w_reload_until,v,k)=WF(ps_w_reload_until,v,k);
    WF(w_reload_commit,v,k)=WF(ps_w_reload_commit,v,k); WF(w_last_shot,v,k)=WF(ps_w_last_shot,v,k);
    WF(w_reload_committed,v,k)=WF(ps_w_reload_committed,v,k); WF(w_reload_shell,v,k)=WF(ps_w_reload_shell,v,k);
    WF(w_pressed,v,k)=WF(ps_w_pressed,v,k); WF(w_inacc,v,k)=WF(ps_w_inacc,v,k);
    for(int c=0;c<7;c++){ long long ix=(((long long)e*10+v)*3+k)*7+c; s.w_rs[ix]=s.ps_w_rs[ix]; }
  }
  PF(active,v)=PF(ps_active,v); PF(switch_until,v)=PF(ps_switch_until,v); PF(zoom,v)=PF(ps_zoom,v);
}
__device__ void kill_one(const S& s,int e,int v,int killer,int weapon,bool restore=false){
  if(restore) restore_weapons(s,e,v);
  PF(alive,v)=0; PF(death_tick,v)=EF(tick); PF(deaths,v)++;
  EVN(EV_KILL);
  if(PF(objective,v)) cancel_obj(s,e,v);
  if(PF(has_bomb,v)) drop_bomb(s,e,v,false);
  for(int slot=0;slot<3;slot++) if(WF(w_wid,v,slot)>=0) drop_weapon(s,e,v,slot,true);
  if(PF(kit,v)) PF(kit,v)=0;
  for(int k=0;k<6;k++) NADE(nades,v,k)=0;
  if(killer>=0 && (killer/5)!=(v/5) && EF(phase)==PHASE_LIVE){
    long long reward = weapon>=0 ? s.T_kill_reward[weapon] : 300;
    cash_add(s,e,killer,reward);
    if(!is_attack(s,e,killer)){
      for(int p=0;p<10;p++) if(!is_attack(s,e,p)) cash_add(s,e,p,50);
    }
    EVN(EV_KILL_REWARD); PF(kills,killer)++;
  }
}

// ---------------------------------------------------------------- round end (Arena._end_round)
__device__ void end_round(const S& s,int e,int winner,int reason){
  if(EF(round_over)) return;
  EF(round_over)=1; EF(phase)=PHASE_POST; EF(post_end)=EF(tick)+300;
  for(int i=0;i<10;i++) if(PF(objective,i)) cancel_obj(s,e,i);
  int bs=EF(bomb_state);
  bool planted_lost=(bs==B_PLANTED||bs==B_DEFUSED) && reason==R_DEFUSE;
  bool objective_win = reason==R_EXPLODE || reason==R_DEFUSE;
  int win_amt = objective_win?3500:3250;
  int lb0=s.loss_level[e*2],lb1=s.loss_level[e*2+1]; int lvl_before[2]={lb0,lb1};
  int loser=1-winner;
  bool attackers_lost = loser==(int)EF(attack_team);
  const int LOSS[5]={1400,1900,2400,2900,3400};
  for(int p=0;p<10;p++) if((p/5)==winner) cash_add(s,e,p,win_amt);
  for(int p=0;p<10;p++) if((p/5)==loser){
    bool survive_timeout = (reason==R_TIMEOUT && attackers_lost && PF(alive,p));
    if(!survive_timeout) cash_add(s,e,p,LOSS[lvl_before[loser]]);
    if(planted_lost && attackers_lost) cash_add(s,e,p,600);
  }
  int ll=s.loss_level[e*2+loser]+1; s.loss_level[e*2+loser]= ll>4?4:ll;
  int lw=s.loss_level[e*2+winner]-1; s.loss_level[e*2+winner]= lw<0?0:lw;
  s.round_wins[e*2+winner]++;
  EVN(EV_ROUND_ECONOMY); EVN(EV_ROUND_END);
  int ri=EF(round_index); if(ri>=0 && ri<32){ s.hist_winner[(long long)e*32+ri]=winner; s.hist_reason[(long long)e*32+ri]=reason; s.hist_tick[(long long)e*32+ri]=EF(tick); }
  int w0=s.round_wins[e*2],w1=s.round_wins[e*2+1];
  bool done = (w0>=13||w1>=13) || (w0+w1>=24);
  if(done){
    EF(finished)=1; EF(phase)=PHASE_FIN;
    EF(winner)= (w0==w1)? -1 : (w1>w0 ? 1 : 0);
    EVN(EV_MATCH_END);
  }
}

// ---------------------------------------------------------------- rounds (Arena._respawn/_half_reset/_start_round)
__device__ void reset_held(const S& s,int e,int i){
  PF(act_move_forward,i)=0; PF(act_move_right,i)=0; PF(act_yaw_rate,i)=0; PF(act_pitch_rate,i)=0; PF(act_climb,i)=0;
  PF(act_jump,i)=0; PF(act_crouch,i)=0; PF(act_walk,i)=0; PF(act_fire,i)=0; PF(act_fire2,i)=0; PF(act_reload,i)=0;
  PF(act_use_plant,i)=0; PF(act_use_defuse,i)=0; PF(act_weapon_slot,i)=1; PF(act_zoom,i)=0; PF(act_ladder,i)=0;
  PF(act_buy,i)=-1; PF(act_refund,i)=-1; PF(act_drop,i)=0; PF(act_pickup,i)=-1;
  PF(act_throw_kind,i)=-1; PF(act_throw_state,i)=0; PF(act_throw_mode,i)=0;
}
__device__ void respawn(const S& s,int e,int i,const double* pad){
  bool was_alive=PF(alive,i);
  if(!was_alive){
    clear_slots(s,e,i); PF(armor,i)=0.0; PF(helmet,i)=0; PF(kit,i)=0; for(int k=0;k<6;k++) NADE(nades,i,k)=0;
    give_default(s,e,i);
  } else {
    for(int k=0;k<3;k++) if(WF(w_wid,i,k)>=0) give_slot(s,e,i,k,(int)WF(w_wid,i,k));
  }
  PF(alive,i)=1; PF(hp,i)=100.0;
  PF(x,i)=pad[0]; PF(y,i)=pad[1]; PF(z,i)=pad[2]+s.cfg_d[0];
  PF(vx,i)=0.0; PF(vy,i)=0.0; PF(vz,i)=0.0; PF(yaw,i)=pad[3]; PF(pitch,i)=0.0;
  PF(crouch,i)=0; PF(grounded,i)=1; PF(movement_mode,i)=0; PF(ladder_id,i)=-1; PF(last_jump,i)=0;
  PF(mv_reattach_until,i)=0; PF(mv_stand_blocked,i)=0;
  PF(active,i)=(WF(w_wid,i,0)>=0)?0:1; PF(switch_until,i)=0; PF(zoom,i)=0;
  for(int k=0;k<NB;k++) pur_clear(s,e,i,k);
  for(int k=0;k<6;k++) NADE(nade_bought,i,k)=0;
  PF(has_bomb,i)=0; PF(objective,i)=0; PF(obj_progress,i)=0; PF(blind_until,i)=0;
  reset_held(s,e,i);
  for(int k=0;k<12;k++) s.rcpt[((long long)e*10+i)*12+k]=(k%3==0)?-1:0;
  PF(death_tick,i)=-1; PF(pickup_ready_tick,i)=0;
}
__device__ void half_reset(const S& s,int e,bool initial){
  for(int i=0;i<10;i++){
    PF(cash,i)=800;
    if(!initial){ clear_slots(s,e,i); PF(armor,i)=0.0; PF(helmet,i)=0; PF(kit,i)=0; for(int k=0;k<6;k++) NADE(nades,i,k)=0; }
  }
  s.loss_level[e*2]=1; s.loss_level[e*2+1]=1;
  if(!initial){ for(int i=0;i<10;i++) give_default(s,e,i); }
  else { for(int i=0;i<10;i++) EVN(EV_CASH); }
}
__device__ void start_round(const S& s,int e){
  EF(round_index)=EF(round_index)+1;
  int ri=EF(round_index);
  if(ri==12){
    EF(attack_team)=1-EF(attack_team);
    EVN(EV_SIDE_SWAP);
    for(int i=0;i<10;i++) EVN(EV_CASH);       // period_reset (before = current cash)
    half_reset(s,e,false);
  }
  EF(round_start_tick)=EF(tick); EF(phase)=PHASE_BUY; EF(live_end)=EF(tick)+360+6900; EF(post_end)=-1; EF(round_over)=0;
  EF(bomb_state)=B_HELD; EF(bomb_carrier)=-1; EF(bomb_site)=-1; EF(bomb_planted_tick)=-1; EF(bomb_fuse_end)=-1;
  EF(bomb_drop_tick)=-1; EF(bomb_dropper)=-1; EF(bomb_actor)=-1; EF(bomb_duration)=0; EF(bomb_planter)=-1;
  s.bomb_pos[e*3]=0; s.bomb_pos[e*3+1]=0; s.bomb_pos[e*3+2]=0;
  EF(wi_count)=0; for(int k=0;k<CK;k++) WI(wi_valid,k)=0;
  EVN(EV_ECONOMY_ROUND_START); EVN(EV_ROUND_START);
  EF(round_started)=1;
  long long seed=s.rng_seed[e];
  for(int side=0;side<2;side++){
    int team = side==0 ? (int)EF(attack_team) : 1-(int)EF(attack_team);
    int n=(int)s.cfg_i[1+side];
    double key[MAXPAD]; for(int k=0;k<n;k++) key[k]=u01(seed,P_SPAWN,ri,side*16+k,0);
    // first five of the ascending (key, index) order
    bool used[MAXPAD]; for(int k=0;k<n;k++) used[k]=false;
    for(int j=0;j<5;j++){
      int best=-1; for(int k=0;k<n;k++) if(!used[k] && (best<0 || key[k]<key[best])) best=k;
      used[best]=true;
      respawn(s,e,team*5+j,s.pad+((side*MAXPAD)+best)*4);
    }
  }
  double u=u01(seed,P_CARRIER,ri,0,0); int c=(int)(u*5); if(c>4) c=4;
  int cs=(int)EF(attack_team)*5+c;
  PF(has_bomb,cs)=1; EF(bomb_carrier)=cs;
  s.bomb_pos[e*3]=PF(x,cs); s.bomb_pos[e*3+1]=PF(y,cs); s.bomb_pos[e*3+2]=PF(z,cs);
}

// ---------------------------------------------------------------- init (Arena.reset) for masked envs
__global__ void k_init(const S* sp,const bool* mask,int E){
  int e=blockIdx.x*blockDim.x+threadIdx.x; if(e>=E||!mask[e]) return; const S& s=*sp;
  long long seed=s.rng_seed[e];
  EF(attack_team)=(signed char)(int)(u01(seed,P_ATTACK,0,0,0)*2.0);
  EF(round_index)=-1;
  for(int i=0;i<10;i++) give_default(s,e,i);
  half_reset(s,e,true);
  start_round(s,e);
}

// ---------------------------------------------------------------- tick begin (phase clock, movement / weapon inputs)
__global__ void k_tick_begin(const S* sp,int E,int gtick,const long long* gtp){
  int e=blockIdx.x*blockDim.x+threadIdx.x; if(e>=E) return; const S& s=*sp;
  if(gtp) gtick=(int)*gtp;            // device clock (whole-decision CUDA graph of GpuSim)
  if(EF(finished)){ for(int i=0;i<10;i++){ PF(alive_run,i)=0; PF(mv_locked,i)=0; } return; }
  EF(round_started)=0;
  EF(tick)=gtick; int t=gtick;
  if(EF(phase)==PHASE_BUY && t-EF(round_start_tick)>=360) EF(phase)=PHASE_LIVE;
  int ph=EF(phase); bool live=ph==PHASE_LIVE, buy=ph==PHASE_BUY;
  for(int i=0;i<10;i++){
    bool al=PF(alive,i);
    PF(alive_run,i)=al;
    if(!al){ PF(mv_locked,i)=0; PF(w_can_act,i)=0; continue; }
    PF(mv_fwd,i)=buy?0.0:PF(act_move_forward,i);
    PF(mv_right,i)=buy?0.0:PF(act_move_right,i);
    PF(mv_jump,i)=buy?0:PF(act_jump,i);
    if(buy){ PF(vx,i)=0.0; PF(vy,i)=0.0; }
    PF(mv_locked,i)=PF(objective,i)!=0;
    int a=(int)PF(active,i); int w=(int)WF(w_wid,i,a);
    double ms=6.0;
    if(w>=0) ms = PF(zoom,i)>0 ? s.T_scoped_max_speed[w] : s.T_max_speed[w];
    PF(mv_maxspeed,i)=ms;
    PF(w_can_act,i)=live && PF(objective,i)==0;   // Arena: p.throw_primed is never set -> priming does not block actions
    PF(w_want,i)= ph!=PHASE_POST ? (long long)PF(act_weapon_slot,i) : -1;
    long long zr=PF(act_zoom,i);
    PF(w_zoomreq,i)= (w>=0 && s.T_n_zoom[w]>0) ? zr : -1;
  }
}

// ---------------------------------------------------------------- decision-boundary one-shots (Arena._one_shots)
__device__ bool buy_ok(const S& s,int e,int i){
  int ph=EF(phase);
  return PF(alive,i) && (ph==PHASE_BUY||ph==PHASE_LIVE) && (EF(tick)-EF(round_start_tick))<=1860 && in_buy_region(s,e,i);
}
__device__ void do_buy(const S& s,int e,int i,int id){
  if(id<0||id>=NB) return;
  if(!buy_ok(s,e,i)){ rcpt(s,e,i,0,0,RS_BUY_UNAVAILABLE,id); return; }
  bool att=is_attack(s,e,i);
  bool teamok = att ? s.buy_t[id] : s.buy_ct[id];
  if(id>=NW){
    if(id==ID_C4 || !teamok){ rcpt(s,e,i,0,0,RS_WRONG_SIDE,id); return; }
    int price=(int)s.buy_price[id];
    if(id==ID_ARMOR){ if(PF(armor,i)>=100){ rcpt(s,e,i,0,0,RS_ALREADY_OWNED,id); return; } }
    else if(id==ID_HELMET){
      if(PF(armor,i)>=100 && PF(helmet,i)){ rcpt(s,e,i,0,0,RS_ALREADY_OWNED,id); return; }
      price = PF(armor,i)>=100 ? 350 : 1000;
    }
    else if(id==ID_KIT){ if(PF(kit,i)){ rcpt(s,e,i,0,0,RS_ALREADY_OWNED,id); return; } }
    else {
      int gi=id-ID_UTIL0; int tot=0; for(int k=0;k<6;k++) tot+=NADE(nades,i,k);
      if(NADE(nades,i,gi)>=(int)s.buy_maxc[id] || tot>=4){ rcpt(s,e,i,0,0,RS_LIMIT,id); return; }
      if(NADE(nade_bought,i,gi)>=(id==ID_UTIL0+1?2:1)){ rcpt(s,e,i,0,0,RS_ROUND_LIMIT,id); return; }
    }
    if(PF(cash,i)<price){ rcpt(s,e,i,0,0,RS_INSUFFICIENT_CASH,id); return; }
    if(id==ID_ARMOR) PF(armor,i)=100.0;
    else if(id==ID_HELMET){ PF(armor,i)=100.0; PF(helmet,i)=1; }
    else if(id==ID_KIT){ PF(kit,i)=1; EVN(EV_ITEM_CREATE); }
    else { int gi=id-ID_UTIL0; EVN(EV_ITEM_CREATE); NADE(nades,i,gi)++; NADE(nade_bought,i,gi)++; }
    cash_add(s,e,i,-price); EVN(EV_BUY);
    int prev_price=PUR(pur_price,i,id), prev_count=PUR(pur_count,i,id);
    PUR(pur_price,i,id)=price+(id==ID_HELMET?prev_price:0); PUR(pur_count,i,id)=prev_count+1; PUR(pur_has,i,id)=1;
    rcpt(s,e,i,0,1,RS_NONE,id);
    return;
  }
  int slot=(int)s.T_slot[id];
  if(slot==2){ rcpt(s,e,i,0,0,RS_UNKNOWN_ITEM,id); return; }
  if(!teamok){ rcpt(s,e,i,0,0,RS_WRONG_SIDE,id); return; }
  int price=(int)s.buy_price[id];
  if(PF(cash,i)<price){ rcpt(s,e,i,0,0,RS_INSUFFICIENT_CASH,id); return; }
  if(WF(w_wid,i,slot)>=0) drop_weapon(s,e,i,slot,false);
  give_slot(s,e,i,slot,id); EVN(EV_ITEM_CREATE);
  cash_add(s,e,i,-price); EVN(EV_BUY);
  PUR(pur_price,i,id)=price; PUR(pur_count,i,id)=1; PUR(pur_has,i,id)=1;
  switch_to(s,e,i,slot);
  rcpt(s,e,i,0,1,RS_NONE,id);
}
__device__ void do_refund(const S& s,int e,int i,int id){
  if(id<0||id>=NB) return;
  if(!buy_ok(s,e,i)){ rcpt(s,e,i,1,0,RS_REFUND_UNAVAILABLE,id); return; }
  if(!PUR(pur_has,i,id)){ rcpt(s,e,i,1,0,RS_NOT_REFUNDABLE,id); return; }
  if(id<NW){
    int slot=(int)s.T_slot[id]; int w=(int)WF(w_wid,i,slot);
    if(slot==2 || w!=id || WF(w_shots,i,slot)){ rcpt(s,e,i,1,0,RS_NOT_REFUNDABLE,id); return; }
    WF(w_wid,i,slot)=-1; EVN(EV_ITEM_DESTROY);
    if(PF(active,i)==slot) PF(active,i)=(WF(w_wid,i,0)>=0)?0:1;
  } else if(id==ID_ARMOR || id==ID_HELMET){
    if(PF(armor,i)<=0){ rcpt(s,e,i,1,0,RS_NOT_REFUNDABLE,id); return; }
    PF(armor,i)=0.0; PF(helmet,i)=0;
  } else if(id==ID_KIT){
    if(!PF(kit,i)){ rcpt(s,e,i,1,0,RS_NOT_REFUNDABLE,id); return; }
    PF(kit,i)=0;
  } else {
    int gi=id-ID_UTIL0; int n=NADE(nades,i,gi); int c=PUR(pur_count,i,id); if(c<n) n=c;
    if(n<=0){ rcpt(s,e,i,1,0,RS_NOT_REFUNDABLE,id); return; }
    NADE(nades,i,gi)-=n; int nb=NADE(nade_bought,i,gi)-n; NADE(nade_bought,i,gi)= nb<0?0:nb;
  }
  int price=PUR(pur_price,i,id);
  pur_clear(s,e,i,id);
  cash_add(s,e,i,price); EVN(EV_REFUND);
  rcpt(s,e,i,1,1,RS_NONE,id);
}
__device__ void do_drop(const S& s,int e,int i,int code){
  if(code<=0) return;
  if(!PF(alive,i)||EF(phase)==PHASE_POST){ rcpt(s,e,i,2,0,RS_UNAVAILABLE,code); return; }
  if(code==3){
    if(PF(has_bomb,i)){ drop_bomb(s,e,i,true); rcpt(s,e,i,2,1,RS_NONE,code); }
    else rcpt(s,e,i,2,0,RS_NO_BOMB,code);
    return;
  }
  int slot=code-1; if(slot<0||slot>1) return;
  if(WF(w_wid,i,slot)<0){ rcpt(s,e,i,2,0,RS_NOTHING,code); return; }
  drop_weapon(s,e,i,slot,false);
  rcpt(s,e,i,2,1,RS_NONE,code);
}
__device__ void do_pickup(const S& s,int e,int i,int pid){
  if(!PF(alive,i)) return;
  int K=CK;
  if(EF(bomb_state)==B_WORLD && pid==K && is_attack(s,e,i)){ try_pick_bomb(s,e,i,true); return; }
  if(pid<0||pid>=K||!WI(wi_valid,pid)){ rcpt(s,e,i,3,0,RS_NOT_FOUND,pid); return; }
  double ex=PF(x,i),ey=PF(y,i),ez=eye_z(s,e,i);
  double px=WI3(wi_pos,pid,0),py=WI3(wi_pos,pid,1),pz=WI3(wi_pos,pid,2);
  double d=dist3(ex,ey,ez,px,py,pz);
  if(d>2.8 && dist3(PF(x,i),PF(y,i),PF(z,i),px,py,pz)>2.8){ rcpt(s,e,i,3,0,RS_TOO_FAR,pid); return; }
  int kind=(int)WI(wi_kind,pid); int slot=(int)s.T_slot[kind];
  if(slot==2){ rcpt(s,e,i,3,0,RS_UNSUPPORTED,pid); return; }
  if(WF(w_wid,i,slot)>=0) drop_weapon(s,e,i,slot,false);
  WI(wi_valid,pid)=0;
  take_item(s,e,i,slot,pid);
  switch_to(s,e,i,slot);
  EVN(EV_PICKUP);
  rcpt(s,e,i,3,1,RS_NONE,pid);
}
__global__ void k_one_shots(const S* sp,int E){
  int e=blockIdx.x*blockDim.x+threadIdx.x; if(e>=E) return; const S& s=*sp;
  if(EF(finished)) return;
  for(int i=0;i<10;i++){
    for(int k=0;k<12;k++) s.rcpt[((long long)e*10+i)*12+k]=(k%3==0)?-1:0;
  }
  if(EF(phase)==PHASE_POST) return;
  for(int i=0;i<10;i++){
    if(!PF(alive,i)) continue;
    int b=(int)PF(act_buy,i); if(b>=0) do_buy(s,e,i,b);
    int r=(int)PF(act_refund,i); if(r>=0) do_refund(s,e,i,r);
    int d=(int)PF(act_drop,i); if(d>0) do_drop(s,e,i,d);
    int p=(int)PF(act_pickup,i); if(p>=0) do_pickup(s,e,i,p);
  }
}

// ---------------------------------------------------------------- objective step (Arena._obj_step)
__global__ void k_obj_step(const S* sp,int E){
  int e=blockIdx.x*blockDim.x+threadIdx.x; if(e>=E) return; const S& s=*sp;
  if(EF(finished)) return;
  if(EF(phase)!=PHASE_LIVE){ for(int i=0;i<10;i++) if(PF(objective,i)) cancel_obj(s,e,i); return; }
  int t=EF(tick);
  if(EF(bomb_state)==B_HELD){
    int c=EF(bomb_carrier);
    s.bomb_pos[e*3]=PF(x,c); s.bomb_pos[e*3+1]=PF(y,c); s.bomb_pos[e*3+2]=PF(z,c);
    if(!PF(alive,c)) drop_bomb(s,e,c,false);
  }
  if(EF(bomb_state)==B_WORLD){
    for(int i=0;i<10;i++) if(PF(alive,i) && is_attack(s,e,i) && !PF(has_bomb,i)){ if(try_pick_bomb(s,e,i,false)) break; }
  }
  int actor=-1,kind=0;
  if(EF(bomb_state)==B_HELD){
    int c=EF(bomb_carrier);
    if(PF(alive,c) && PF(act_use_plant,c) && can_obj(s,e,c) && in_plant_zone(s,PF(x,c),PF(y,c),PF(z,c))>=0){ actor=c; kind=OBJ_PLANT; }
  } else if(EF(bomb_state)==B_PLANTED){
    int cur=EF(bomb_actor);
    double bx=s.bomb_pos[e*3],by=s.bomb_pos[e*3+1],bz=s.bomb_pos[e*3+2];
    bool cand[10];
    for(int i=0;i<10;i++) cand[i]=PF(alive,i) && !is_attack(s,e,i) && PF(act_use_defuse,i) && can_obj(s,e,i)
                                   && dist3(PF(x,i),PF(y,i),PF(z,i),bx,by,bz)<2.8;
    if(cur>=0 && cand[cur]){ actor=cur; kind=OBJ_DEFUSE; }
    else {
      for(int i=0;i<10;i++) if(cand[i]){
        if(!seg_blocked(s,PF(x,i),PF(y,i),eye_z(s,e,i),bx,by,bz+0.3)){ actor=i; kind=OBJ_DEFUSE; break; }
      }
    }
  }
  for(int i=0;i<10;i++) if(PF(objective,i) && i!=actor) cancel_obj(s,e,i);
  if(actor<0){ EF(bomb_actor)=-1; return; }
  if(PF(objective,actor)!=kind){
    PF(objective,actor)=kind; PF(obj_progress,actor)=0; EF(bomb_actor)=actor;
    int dur = kind==OBJ_PLANT ? 180 : (PF(kit,actor)?300:600);
    EF(bomb_duration)=dur;
    if(kind==OBJ_PLANT) EVN(EV_PLANT_START); else EVN(EV_DEFUSE_START);
    if(PF(zoom,actor)) PF(zoom,actor)=0;
    int a=(int)PF(active,actor);
    if(WF(w_wid,actor,a)>=0 && WF(w_reload_until,actor,a)>=0){
      WF(w_reload_until,actor,a)=-1; WF(w_reload_commit,actor,a)=-1; WF(w_reload_committed,actor,a)=0; WF(w_reload_shell,actor,a)=0;
    }
  }
  PF(obj_progress,actor)++;
  PF(vx,actor)=0.0; PF(vy,actor)=0.0; PF(vz,actor)=0.0;
  if(PF(obj_progress,actor)>=EF(bomb_duration)){
    PF(objective,actor)=0;
    if(kind==OBJ_PLANT){
      int site=in_plant_zone(s,PF(x,actor),PF(y,actor),PF(z,actor));
      PF(has_bomb,actor)=0;
      EF(bomb_state)=B_PLANTED; EF(bomb_carrier)=-1;
      s.bomb_pos[e*3]=PF(x,actor); s.bomb_pos[e*3+1]=PF(y,actor); s.bomb_pos[e*3+2]=PF(z,actor);
      EF(bomb_site)=site; EF(bomb_planted_tick)=t; EF(bomb_fuse_end)=t+2400; EF(bomb_actor)=-1; EF(bomb_planter)=actor;
      EVN(EV_PLANT_COMPLETE);
      cash_add(s,e,actor,300);
    } else {
      EF(bomb_state)=B_DEFUSED;
      EVN(EV_DEFUSE_COMPLETE);
      cash_add(s,e,actor,300);
      end_round(s,e,1-(int)EF(attack_team),R_DEFUSE);
    }
  }
}

// ---------------------------------------------------------------- bomb timer, win conditions, next round
__global__ void k_end_tick(const S* sp,int E){
  int e=blockIdx.x*blockDim.x+threadIdx.x; if(e>=E) return; const S& s=*sp;
  if(EF(finished)) return;
  int t=EF(tick);
  if(EF(bomb_state)==B_PLANTED && t>=EF(bomb_fuse_end) && !EF(round_over)){
    EF(bomb_state)=B_EXPLODED; EVN(EV_BOMB_EXPLODE);
    end_round(s,e,EF(attack_team),R_EXPLODE);
  }
  if(EF(phase)==PHASE_LIVE && !EF(round_over)){
    int aa=0,da=0;
    for(int i=0;i<10;i++) if(PF(alive,i)){ if(is_attack(s,e,i)) aa++; else da++; }
    bool planted=EF(bomb_state)==B_PLANTED;
    if(da==0) end_round(s,e,EF(attack_team),R_ELIM);
    else if(aa==0 && !planted) end_round(s,e,1-(int)EF(attack_team),R_ELIM);
    else if(t>=EF(live_end) && !planted) end_round(s,e,1-(int)EF(attack_team),R_TIMEOUT);
  }
  if(EF(phase)==PHASE_POST && t>=EF(post_end) && !EF(finished)) start_round(s,e);
}

// ---------------------------------------------------------------- kills caused by grenades (damage was applied by gpusim.grenades)
__global__ void k_kill_pass(const S* sp,int E,const bool* alive_before,const int* killer){
  int e=blockIdx.x*blockDim.x+threadIdx.x; if(e>=E) return; const S& s=*sp;
  for(int i=0;i<10;i++){
    long long ix=(long long)e*10+i;
    if(alive_before[ix] && !PF(alive,i)) kill_one(s,e,i,killer[ix],-1);
  }
}

// ---------------------------------------------------------------- bullet rays (Arena._do_shot: accuracy + two-disc spread)
// row r: player index pi = row_player[r] (= e*10+seat), pellet j = r - row_start[pi]
__global__ void k_build_rays(const S* sp,int P,const long long* row_player,const long long* row_start,const long long* wid_in,
    const double* ap_pitch,const double* ap_yaw,const double* pen,
    double* org,double* dir,long long* env,long long* seat,double* wrange,double* wpower,double* wdamage){
  int r=blockIdx.x*blockDim.x+threadIdx.x; if(r>=P) return; const S& s=*sp;
  long long pi=row_player[r];
  if(pi<0){ env[r]=-1; seat[r]=0; return; }        // unused row of the fixed-capacity buffer (gpusim.damage): skipped by shoot_k / settle
  int e=(int)(pi/10); int i=(int)(pi%10); int j=(int)(r-row_start[pi]);
  int w=(int)wid_in[pi];
  double eyex=PF(x,i),eyey=PF(y,i),eyez=eye_z(s,e,i);
  double ra=0,aa=0,rb=0,ab=0,spread=0,inacc=0;
  // accuracy
  if(s.T_acc_has[w]){
    int zl=(int)PF(zoom,i);
    int mode=((s.T_n_zoom[w]>0 && zl>0) || s.T_acc_always1[w])?1:0;
    double ms=zl>0 ? s.T_scoped_max_speed[w] : s.T_max_speed[w]; if(ms==0.0) ms=6.0;
    double speed=hypot(PF(vx,i),PF(vy,i));
    double mv=(speed/ms-.34)/(.95-.34); mv= mv<0.0?0.0:(mv>1.0?1.0:mv);
    double air= PF(grounded,i) ? 0.0 : fmin(1.0,fabs(PF(vz,i))/5.5);
    const double* A=s.T_acc+(long long)w*10;     // [5][2]: spread, stand, crouch, move, jump
    inacc=A[(PF(crouch,i)?2:1)*2+mode]+A[3*2+mode]*pow(mv,3.0)+A[4*2+mode]*air+pen[pi];
    spread=A[0*2+mode];
  }
  long long seed=s.rng_seed[e]; long long mt=EF(tick)-EF(t0);
  long long b=(long long)i*16+j;
  ra=u01(seed,P_SPREAD,mt,b,0)*inacc; aa=u01(seed,P_SPREAD,mt,b,1)*3.141592653589793*2;
  rb=u01(seed,P_SPREAD,mt,b,2)*spread; ab=u01(seed,P_SPREAD,mt,b,3)*3.141592653589793*2;
  double x=cos(aa)*ra+cos(ab)*rb, y=sin(aa)*ra+sin(ab)*rb;
  double yaw=PF(yaw,i)+2.0*ap_yaw[pi];
  double pitch=PF(pitch,i)+2.0*ap_pitch[pi]; pitch=fmax(-1.48,fmin(1.48,pitch));
  double cy=cos(yaw),sy=sin(yaw),cp=cos(pitch),sp_=sin(pitch);
  double f[3]={cp*cy,-cp*sy,sp_}, rr[3]={-sy,-cy,0.0}, uu[3]={-sp_*cy,sp_*sy,cp};
  double d[3]; for(int k=0;k<3;k++) d[k]=f[k]+rr[k]*x+uu[k]*y;
  double n=sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
  org[3*r]=eyex; org[3*r+1]=eyey; org[3*r+2]=eyez;
  dir[3*r]=d[0]/n; dir[3*r+1]=d[1]/n; dir[3*r+2]=d[2]/n;
  env[r]=e; seat[r]=i; wrange[r]=s.T_range[w]; wpower[r]=s.T_pen_power[w]; wdamage[r]=s.T_damage[w];
}

// ---------------------------------------------------------------- damage settlement (Arena._do_shot tail, _do_knife, _damage)
__device__ bool seg_box(const double* o,const double* d,const double* lo,const double* hi,double rng,double* out){
  double enter=0.0,exit_=rng;
  for(int k=0;k<3;k++){
    if(fabs(d[k])<1e-8){ if(o[k]<lo[k]||o[k]>hi[k]) return false; continue; }
    double a=(lo[k]-o[k])/d[k], b=(hi[k]-o[k])/d[k];
    enter=fmax(enter,fmin(a,b)); exit_=fmin(exit_,fmax(a,b));
    if(enter>exit_) return false;
  }
  *out=enter; return true;
}
__device__ void do_knife(const S& s,int e,int i,const bool* alive0,const double* armor0){
  // fpssim.melee.trace_knife with raycast=None (Arena passes none), first=True, pitch 0
  const double SLASH=48*.0254, STAB=32*.0254, RAD=16*.0254;
  bool heavy=PF(act_fire2,i);
  double ox=PF(x,i),oy=PF(y,i),oz=eye_z(s,e,i);
  double yaw=PF(yaw,i); double cp=cos(0.0);
  double dv[3]={cp*cos(yaw),-cp*sin(yaw),sin(0.0)};
  double o[3]={ox,oy,oz};
  double rng=heavy?STAB:SLASH;
  EVN(EV_FIRE);
  double nearest=rng; int target=-1;
  for(int q=0;q<10;q++){
    if(!alive0[q] || q==i || (q/5)==(i/5)) continue;
    double qx=PF(x,q),qy=PF(y,q),qz=PF(z,q);
    if((qx-ox)*dv[0]+(qy-oy)*dv[1]<=.02) continue;
    double h=PF(crouch,q)?1.1:1.8;
    double lo[3]={qx-.28,qy-.28,qz+.06}, hi[3]={qx+.28,qy+.28,qz+h};
    double elo[3]={lo[0]-RAD,lo[1]-RAD,lo[2]-RAD}, ehi[3]={hi[0]+RAD,hi[1]+RAD,hi[2]+RAD};
    double dist; if(!seg_box(o,dv,elo,ehi,rng,&dist)) continue;
    if(dist>nearest) continue;
    nearest=dist; target=q;
  }
  if(target<0) return;
  double dx=PF(x,target)-PF(x,i),dy=PF(y,target)-PF(y,i),L=hypot(dx,dy); double ty=PF(yaw,target);
  bool back = L>0 && (cos(ty)*dx-sin(ty)*dy)/L>.475;
  double raw = back ? (heavy?180.0:90.0) : (heavy?65.0:40.0);      // first=True always (Arena default)
  double hp=floor(raw*(armor0[target]>0?0.85:1.0));
  EVN(EV_HIT);
  PF(armor,target)=fmax(0.0,PF(armor,target)-0.0);
  PF(hp,target)-=hp;
  if(PF(hp,target)<=1e-9 && PF(alive,target)){ PF(hp,target)=0.0; kill_one(s,e,target,i,(int)s.knife_id[0],target>i && PF(alive_run,target)); }
}
__global__ void k_settle(const S* sp,int E,const long long* row_start,const long long* npel,const bool* knife,const long long* wid_in,
    const long long* t_seat,const double* t_dist,const long long* t_part,const double* t_scale,const long long* t_nhit,
    const bool* ev_rs,const bool* ev_cm,const bool* ev_fi,const bool* ev_dr){
  int e=blockIdx.x*blockDim.x+threadIdx.x; if(e>=E) return; const S& s=*sp;
  if(EF(finished)) return;
  bool alive0[10]; double armor0[10];
  for(int i=0;i<10;i++){ alive0[i]=PF(alive,i); armor0[i]=PF(armor,i); }     // Arena._targets() snapshot (knife targets)
  for(int i=0;i<10;i++){
    long long pi=(long long)e*10+i;
    // weapon-tick events in the seat order of the CPU loop (a player killed earlier in this tick is never ticked there)
    if(PF(alive,i)){
      if(ev_rs[pi]) EVN(EV_RELOAD_START);
      if(ev_cm[pi]) EVN(EV_RELOAD_COMMIT);
      if(ev_fi[pi]) EVN(EV_RELOAD_FINISH);
      if(ev_dr[pi]) EVN(EV_DRAW);
    } else if(ev_rs[pi]||ev_cm[pi]||ev_fi[pi]||ev_dr[pi]) EVN(EV_VOID_SHOT);
    if(knife[pi]){ if(PF(alive,i)) do_knife(s,e,i,alive0,armor0); else EVN(EV_VOID_SHOT); continue; }
    long long n=npel[pi]; if(n<=0) continue;
    if(!PF(alive,i)){ EVN(EV_VOID_SHOT); continue; }   // shooter was killed earlier this tick by a lower seat: CPU never ticks it
    int w=(int)wid_in[pi];
    for(long long j=0;j<n;j++){
      long long r=row_start[pi]+j;
      EVN(EV_FIRE); PF(last_fire_tick,i)=EF(tick);
      int nh=(int)t_nhit[r];
      for(int h=0;h<nh;h++){
        int v=(int)t_seat[r*4+h]; if(!PF(alive,v)){ EVN(EV_DEAD_HIT); continue; }       // traced against the tick-start players (gpu_rounds.md, difference 3)
        double dist=t_dist[r*4+h]; int part=(int)t_part[r*4+h]; double scale=t_scale[r*4+h];
        // combat.damage
        double mult = part==0 ? s.T_head_mult[w] : (part==1 ? 1.0 : 0.75);
        double d=fmin(dist,s.T_range[w]);
        double amount=s.T_damage[w]*pow(s.T_range_mod[w],d/12.7)*mult;
        bool prot = PF(armor,v)>0 && (part==1 || (part==0 && PF(helmet,v)));
        double hpd=amount, ab=0.0;
        if(prot){
          hpd=amount*s.T_armor_ratio[w]*0.5; ab=(amount-hpd)*0.5;
          if(ab>PF(armor,v)){ ab=PF(armor,v); hpd=amount-ab/0.5; }
        }
        if(scale!=1.0){ amount*=scale; hpd*=scale; ab*=scale; }
        EVN(EV_HIT);
        PF(armor,v)=fmax(0.0,PF(armor,v)-ab);
        PF(hp,v)-=hpd;
        if(PF(hp,v)<=1e-9 && PF(alive,v)){ PF(hp,v)=0.0; kill_one(s,e,v,i,w,v>i && PF(alive_run,v)); }
      }
    }
  }
}

// ---------------------------------------------------------------- host entry points
#define LAUNCH(k,n,...) { int _n=(n); if(_n>0) k<<<(_n+63)/64,64,0,at::cuda::getCurrentCUDAStream()>>>(__VA_ARGS__); }
#define LLP(t) ((long long*)(t).data_ptr<int64_t>())
#define SP(t) ((const S*)LLP(t))
void init(torch::Tensor tab,torch::Tensor mask,int64_t E){ LAUNCH(k_init,E,SP(tab),mask.data_ptr<bool>(),(int)E); }
void tick_begin(torch::Tensor tab,int64_t E,int64_t gtick){ LAUNCH(k_tick_begin,E,SP(tab),(int)E,(int)gtick,(const long long*)nullptr); }
void tick_begin_dev(torch::Tensor tab,int64_t E,torch::Tensor gt){ LAUNCH(k_tick_begin,E,SP(tab),(int)E,0,LLP(gt)); }
void one_shots(torch::Tensor tab,int64_t E){ LAUNCH(k_one_shots,E,SP(tab),(int)E); }
void obj_step(torch::Tensor tab,int64_t E){ LAUNCH(k_obj_step,E,SP(tab),(int)E); }
void end_tick(torch::Tensor tab,int64_t E){ LAUNCH(k_end_tick,E,SP(tab),(int)E); }
void kill_pass(torch::Tensor tab,int64_t E,torch::Tensor alive_before,torch::Tensor killer){
  LAUNCH(k_kill_pass,E,SP(tab),(int)E,alive_before.data_ptr<bool>(),killer.data_ptr<int>()); }
void build_rays(torch::Tensor tab,torch::Tensor row_player,torch::Tensor row_start,torch::Tensor wid,torch::Tensor ap_pitch,
    torch::Tensor ap_yaw,torch::Tensor pen,torch::Tensor org,torch::Tensor dir,torch::Tensor env,torch::Tensor seat,
    torch::Tensor wrange,torch::Tensor wpower,torch::Tensor wdamage){
  int P=(int)row_player.size(0);
  LAUNCH(k_build_rays,P,SP(tab),P,LLP(row_player),LLP(row_start),LLP(wid),
    ap_pitch.data_ptr<double>(),ap_yaw.data_ptr<double>(),pen.data_ptr<double>(),org.data_ptr<double>(),dir.data_ptr<double>(),
    LLP(env),LLP(seat),wrange.data_ptr<double>(),wpower.data_ptr<double>(),wdamage.data_ptr<double>()); }
void settle(torch::Tensor tab,int64_t E,torch::Tensor row_start,torch::Tensor npel,torch::Tensor knife,torch::Tensor wid,
    torch::Tensor t_seat,torch::Tensor t_dist,torch::Tensor t_part,torch::Tensor t_scale,torch::Tensor t_nhit,
    torch::Tensor ev_rs,torch::Tensor ev_cm,torch::Tensor ev_fi,torch::Tensor ev_dr){
  LAUNCH(k_settle,E,SP(tab),(int)E,LLP(row_start),LLP(npel),knife.data_ptr<bool>(),
    LLP(wid),LLP(t_seat),t_dist.data_ptr<double>(),LLP(t_part),
    t_scale.data_ptr<double>(),LLP(t_nhit),ev_rs.data_ptr<bool>(),ev_cm.data_ptr<bool>(),ev_fi.data_ptr<bool>(),ev_dr.data_ptr<bool>()); }
'''

CPP = '''
void init(torch::Tensor tab,torch::Tensor mask,int64_t E);
void tick_begin(torch::Tensor tab,int64_t E,int64_t gtick);
void tick_begin_dev(torch::Tensor tab,int64_t E,torch::Tensor gt);
void one_shots(torch::Tensor tab,int64_t E);
void obj_step(torch::Tensor tab,int64_t E);
void end_tick(torch::Tensor tab,int64_t E);
void kill_pass(torch::Tensor tab,int64_t E,torch::Tensor alive_before,torch::Tensor killer);
void build_rays(torch::Tensor tab,torch::Tensor row_player,torch::Tensor row_start,torch::Tensor wid,torch::Tensor ap_pitch,torch::Tensor ap_yaw,torch::Tensor pen,torch::Tensor org,torch::Tensor dir,torch::Tensor env,torch::Tensor seat,torch::Tensor wrange,torch::Tensor wpower,torch::Tensor wdamage);
void settle(torch::Tensor tab,int64_t E,torch::Tensor row_start,torch::Tensor npel,torch::Tensor knife,torch::Tensor wid,torch::Tensor t_seat,torch::Tensor t_dist,torch::Tensor t_part,torch::Tensor t_scale,torch::Tensor t_nhit,torch::Tensor ev_rs,torch::Tensor ev_cm,torch::Tensor ev_fi,torch::Tensor ev_dr);
'''

_mods = {}


def ext(fields, consts):
    """fields: ordered [(name, dtype)] of the pointer table; consts: dict of #define name -> int (NB, NW, ID_*...)."""
    import hashlib
    hdr = header(fields) + ''.join(f'#define {k} {v}\n' for k, v in sorted(consts.items()))
    key = hashlib.sha1((hdr + BODY).encode()).hexdigest()[:10]
    if key not in _mods:
        os.environ.setdefault('TORCH_EXTENSIONS_DIR', './fps_data/tmp/torch_ext')
        _mods[key] = load_inline(f'fpsrounds_{key}', cpp_sources=CPP, cuda_sources=hdr + BODY,
                                 functions=['init', 'tick_begin', 'tick_begin_dev', 'one_shots', 'obj_step', 'end_tick', 'kill_pass', 'build_rays',
                                            'settle'],
                                 extra_cuda_cflags=['-O3', '-fmad=false', '--prec-div=true', '--prec-sqrt=true'], verbose=False)
    return _mods[key]
