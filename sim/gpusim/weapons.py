"""Batched weapon state machine + recoil/inaccuracy for E matches x 10 players x 3 slots (port of fpssim/weapons.py + recoil.py).

State keys (SPEC): w_wid w_ammo w_reserve w_shots w_shot_ready w_reload_until w_reload_commit w_last_shot (long [E,10,3]),
w_reload_committed w_reload_shell w_pressed (bool [E,10,3]), w_inacc (f64 [E,10,3]), w_rs (f64 [E,10,3,7]),
active (long [E,10]), switch_until (long [E,10]), zoom (long [E,10]), last_knife (long [E,10]).
Order inside tick t (identical to WeaponSystem.tick): decay + reload progress -> switch -> zoom -> reload request -> fire.
"""
import math
import torch
from fpssim import recoil as R

DRAW_TICKS = 11
KNIFE_FIRST_GAP = 60
I_INDEX, I_PITCH, I_YAW, I_VP, I_VY, I_VWP, I_VWY = range(7)
NONE = -1


def new_state(E, device):
    z = lambda *s, dt=torch.long, v=0: torch.full((E, 10) + s, v, dtype=dt, device=device)
    return dict(w_wid=z(3, v=-1), w_ammo=z(3), w_reserve=z(3), w_shots=z(3), w_shot_ready=z(3), w_reload_until=z(3, v=-1),
                w_reload_commit=z(3, v=-1), w_last_shot=z(3, v=-1), w_reload_committed=z(3, dt=torch.bool),
                w_reload_shell=z(3, dt=torch.bool), w_pressed=z(3, dt=torch.bool), w_inacc=z(3, dt=torch.float64),
                w_rs=z(3, 7, dt=torch.float64), active=z(v=1), switch_until=z(), zoom=z(), last_knife=z(v=-10 ** 6))


def give(s, T, mask, slot, wid):
    """mask [E,10] bool: put a fresh weapon `wid` (long [E,10] or int) into `slot`."""
    wid = torch.as_tensor(wid, device=mask.device).expand(mask.shape)
    sl = (slice(None), slice(None), slot)
    for k, v in (('w_wid', wid), ('w_ammo', T['magazine'][wid.clamp(min=0)]), ('w_reserve', T['reserve'][wid.clamp(min=0)]),
                 ('w_shots', 0), ('w_shot_ready', 0), ('w_reload_until', -1), ('w_reload_commit', -1), ('w_last_shot', -1)):
        s[k][sl] = torch.where(mask, torch.as_tensor(v, device=mask.device, dtype=torch.long), s[k][sl])
    for k in ('w_reload_committed', 'w_reload_shell', 'w_pressed'):
        s[k][sl] &= ~mask
    s['w_inacc'][sl] = torch.where(mask, 0.0, s['w_inacc'][sl])
    s['w_rs'][sl] = torch.where(mask[..., None], 0.0, s['w_rs'][sl])


def _decay_step(rs, h):
    p, y = rs[..., I_PITCH], rs[..., I_YAW]
    f = math.exp(-R.DECAY2_EXP * h)
    p = p * f; y = y * f
    mag = torch.hypot(p, y); lin = R.DECAY2_LIN * h
    k = torch.where(mag > lin, 1 - lin / mag.clamp(min=1e-300), torch.zeros_like(mag))
    p = p * k; y = y * k
    vp, vy = rs[..., I_VP], rs[..., I_VY]
    p = p + vp * h * 0.5; y = y + vy * h * 0.5
    vf = math.exp(-R.VEL_DECAY * h)
    vp = vp * vf; vy = vy * vf
    p = p + vp * h * 0.5; y = y + vy * h * 0.5
    g = math.exp(-R.VIEW_PUNCH_DECAY * h)
    rs[..., I_PITCH] = p; rs[..., I_YAW] = y; rs[..., I_VP] = vp; rs[..., I_VY] = vy
    rs[..., I_VWP] *= g; rs[..., I_VWY] *= g


def _snap(rs):
    for a, b in ((I_VP, I_VY), (I_PITCH, I_YAW), (I_VWP, I_VWY)):
        m = (rs[..., a].abs() < 1e-6) & (rs[..., b].abs() < 1e-6)
        rs[..., a] = torch.where(m, 0.0, rs[..., a]); rs[..., b] = torch.where(m, 0.0, rs[..., b])


def _recovery(T, wid, crouch, in_air, index):
    w = wid.clamp(min=0)
    n = torch.floor(index)
    def remap(a, b, c, d):
        t = ((n - a) / (b - a).clamp(min=1e-9)).clamp(0, 1)
        return torch.where(a == b, d, c + (d - c) * t)
    ts, te = T['trans_start'][w].double(), T['trans_end'][w].double()
    rc = remap(ts, te, T['rec_crouch'][w], T['rec_crouch_final'][w])
    rsd = remap(ts, te, T['rec_stand'][w], T['rec_stand_final'][w])
    rt = torch.where(crouch, rc, rsd)
    rt = torch.where(in_air, T['rec_crouch'][w] * 4, rt)
    return torch.where(T['has_recoil'][w], rt, torch.full_like(rt, 0.3))


def _gather(x, idx):
    return x.gather(2, idx[..., None]).squeeze(2)


def _scatter(x, idx, v, mask):
    cur = _gather(x, idx)
    x.scatter_(2, idx[..., None], torch.where(mask, v, cur)[..., None])


def tick_torch(s, T, t, alive, can_act, fire, fire2, reload, want_slot, zoom_req, crouch, in_air):
    """(torch reference of `tick`) One tick for all players. Inputs [E,10]: alive/can_act/fire/fire2/reload/crouch/in_air bool; want_slot long (-1 none);
    zoom_req long (-1 = leave as is). t: int. Returns dict(shot, knife, knife_heavy, knife_first, ap_pitch, ap_yaw, pen, wid, ...)."""
    dev = alive.device
    wid3 = s['w_wid']; has = wid3 >= 0
    # ---------------- decay all slots (alive players only; dead players' weapons are not ticked in the CPU arena)
    rs = s['w_rs']
    live3 = (alive[..., None] & has)
    rs_new = rs.clone()
    _decay_step(rs_new, 1 / 64); _decay_step(rs_new, 1 / 60 - 1 / 64); _snap(rs_new)
    rs.copy_(torch.where(live3[..., None], rs_new, rs))
    rt = _recovery(T, wid3, crouch[..., None].expand_as(wid3), in_air[..., None].expand_as(wid3), rs[..., I_INDEX])
    pen = s['w_inacc'] * torch.exp(-(1 / 60) * math.log(10) / rt.clamp(min=0.01))
    pen = torch.where(pen < 1e-5, 0.0, pen)
    s['w_inacc'] = torch.where(live3 & (s['w_inacc'] > 0), pen, torch.where(live3, 0.0, s['w_inacc']))
    # ---------------- reload progress (active weapon)
    a = s['active']
    wid = _gather(wid3, a); w = wid.clamp(min=0)
    ru = _gather(s['w_reload_until'], a); rc = _gather(s['w_reload_commit'], a)
    rcm = _gather(s['w_reload_committed'], a); rsh = _gather(s['w_reload_shell'], a)
    ammo = _gather(s['w_ammo'], a); res = _gather(s['w_reserve'], a)
    mag = T['magazine'][w]
    rel = alive & (wid >= 0) & (ru >= 0)
    ev_commit = torch.zeros_like(alive); ev_finish = torch.zeros_like(alive)
    # shell: one insert per reload period (period >= 28 ticks -> at most one per tick)
    sh = rel & rsh & (t >= ru)
    ins = sh & (res > 0) & (ammo < mag)
    ammo = ammo + ins.long(); res = res - ins.long(); ev_commit |= ins
    cont = sh & (res > 0) & (ammo < mag)
    ru = torch.where(cont, ru + T['reload_ticks'][w], ru)
    fin_sh = sh & ~cont
    # magazine
    mg = rel & ~rsh
    do_commit = mg & ~rcm & (((rc >= 0) & (t >= rc)) | (t >= ru))
    take = torch.minimum(mag - ammo, res)
    ammo = torch.where(do_commit, ammo + take, ammo); res = torch.where(do_commit, res - take, res); rcm = rcm | do_commit
    ev_commit |= do_commit
    fin_mg = mg & (t >= ru)
    fin = fin_sh | fin_mg; ev_finish |= fin
    ru = torch.where(fin, -1, ru); rc = torch.where(fin_mg, -1, rc); rcm = rcm & ~fin; rsh = rsh & ~fin
    # ---------------- switch
    ws = want_slot
    tgt_has = (ws >= 0) & (_gather(has.long(), ws.clamp(0, 2)) > 0)
    sw = alive & can_act & tgt_has & (ws != a)
    cancel = sw & (ru >= 0)
    ru = torch.where(cancel, -1, ru); rc = torch.where(cancel, -1, rc); rcm = rcm & ~cancel; rsh = rsh & ~cancel
    # write back reload/ammo of the old active slot before switching
    for k, v in (('w_reload_until', ru), ('w_reload_commit', rc), ('w_reload_committed', rcm), ('w_reload_shell', rsh),
                 ('w_ammo', ammo), ('w_reserve', res)):
        _scatter(s[k], a, v, alive & (wid >= 0))
    newa = torch.where(sw, ws, a)
    _scatter(s['w_pressed'], newa, torch.zeros_like(sw), sw)
    s['active'] = newa; s['zoom'] = torch.where(sw, 0, s['zoom']); s['switch_until'] = torch.where(sw, t + DRAW_TICKS, s['switch_until'])
    a = newa
    wid = _gather(wid3, a); w = wid.clamp(min=0); valid = alive & (wid >= 0)
    ru = _gather(s['w_reload_until'], a); ammo = _gather(s['w_ammo'], a); res = _gather(s['w_reserve'], a); mag = T['magazine'][w]
    # ---------------- zoom
    zr = zoom_req
    zset = valid & (zr >= 0)
    zval = torch.where(ru >= 0, 0, torch.minimum(zr.clamp(min=0), T['n_zoom'][w]))
    s['zoom'] = torch.where(zset, zval, s['zoom'])
    # ---------------- reload request
    can_rel = valid & can_act & reload & (ru < 0) & (mag > 0) & (ammo < mag) & (res > 0) & (t >= s['switch_until'])
    s['zoom'] = torch.where(can_rel, 0, s['zoom'])
    shell = T['shell'][w]
    ru = torch.where(can_rel, t + T['reload_ticks'][w], ru)
    commit_t = torch.where(ammo == 0, T['commit_empty'][w], T['commit_full'][w])
    _scatter(s['w_reload_until'], a, ru, can_rel)
    _scatter(s['w_reload_commit'], a, torch.where(shell, -1, t + commit_t), can_rel)
    _scatter(s['w_reload_committed'], a, torch.zeros_like(can_rel), can_rel)
    _scatter(s['w_reload_shell'], a, shell, can_rel)
    rsh = _gather(s['w_reload_shell'], a)
    # ---------------- fire
    knife = wid == T['knife_id'] if 'knife_id' in T else torch.zeros_like(valid)
    trig = fire | (knife & fire2)
    pressed = _gather(s['w_pressed'], a)
    edge = trig & ~pressed
    _scatter(s['w_pressed'], a, trig, valid)
    sr = _gather(s['w_shot_ready'], a)
    base = valid & can_act & trig & (t >= sr) & (t >= s['switch_until'])
    gun_ok = (~((ru >= 0) & ~(rsh & (ammo > 0)))) & (ammo > 0) & (T['automatic'][w] | edge)
    shot = base & ~knife & gun_ok
    kn = base & knife
    heavy = kn & fire2
    first = kn & ((t - s['last_knife']) > KNIFE_FIRST_GAP)
    s['last_knife'] = torch.where(kn, t, s['last_knife'])
    _scatter(s['w_shot_ready'], a, t + torch.where(heavy, 60, 30), kn)
    # gun shot
    _scatter(s['w_reload_until'], a, torch.full_like(ru, -1), shot & (ru >= 0))
    _scatter(s['w_reload_shell'], a, torch.zeros_like(shot), shot & (ru >= 0))
    _scatter(s['w_reload_commit'], a, torch.full_like(ru, -1), shot & (ru >= 0))
    _scatter(s['w_ammo'], a, ammo - 1, shot)
    _scatter(s['w_shots'], a, _gather(s['w_shots'], a) + 1, shot)
    _scatter(s['w_last_shot'], a, torch.full_like(ru, t), shot)
    _scatter(s['w_shot_ready'], a, t + T['fire_ticks'][w], shot)
    rsa = rs.gather(2, a[..., None, None].expand(-1, -1, 1, 7)).squeeze(2)          # [E,10,7]
    pena = _gather(s['w_inacc'], a)
    ap_pitch, ap_yaw = rsa[..., I_PITCH].clone(), rsa[..., I_YAW].clone()
    # kick
    idx = (torch.floor(rsa[..., I_INDEX]).long() % 64)
    ent = T['recoil_table'][w, idx]                                                 # [E,10,2]
    c, sn, m = torch.cos(ent[..., 0]), torch.sin(ent[..., 0]), ent[..., 1]
    kick = shot & T['has_recoil'][w]
    rsa2 = rsa.clone()
    rsa2[..., I_VP] += c * m; rsa2[..., I_VY] -= sn * m
    rsa2[..., I_VWP] += c * m * R.VIEW_PUNCH_EXTRA; rsa2[..., I_VWY] -= sn * m * R.VIEW_PUNCH_EXTRA
    rsa2[..., I_INDEX] += 1
    rsa = torch.where(kick[..., None], rsa2, rsa)
    rs.scatter_(2, a[..., None, None].expand(-1, -1, 1, 7), rsa[:, :, None, :])
    _scatter(s['w_inacc'], a, pena + T['inacc_fire'][w], shot)
    s['zoom'] = torch.where(shot & T['unzoom_after_shot'][w], 0, s['zoom'])
    return dict(shot=shot, knife=kn, knife_heavy=heavy, knife_first=first, ap_pitch=ap_pitch, ap_yaw=ap_yaw, pen=pena,
                wid=wid, ev_commit=ev_commit, ev_finish=ev_finish, ev_reload_start=can_rel, ev_draw=sw)


# ============================================================================================================== fused CUDA tick
# `tick` on CUDA runs as one kernel, one thread per player, with the operations of `tick_torch` in the same order and the same float64
# arithmetic (libdevice exp / hypot / cos / sin like torch, -fmad=false, the python-side constants math.exp(...) passed in as literals).
# GPUSIM_WEAPONS_TORCH=1 (or CPU tensors) uses tick_torch.  `t` may be an int or an int64 device scalar (CUDA graphs).
_WK = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>
typedef long long ll;
struct WS { ll *wid,*ammo,*reserve,*shots,*shot_ready,*reload_until,*reload_commit,*last_shot; bool *committed,*shell,*pressed;
            double *inacc,*rs; ll *active,*switch_until,*zoom,*last_knife; };
struct WI { const bool *alive,*can_act,*fire,*fire2,*reload,*crouch,*in_air; const ll *want,*zoomreq; };
struct WT { const ll *magazine,*reload_ticks,*n_zoom,*commit_empty,*commit_full,*fire_ticks,*trans_start,*trans_end;
            const bool *shell,*automatic,*has_recoil,*unzoom; const double *recoil,*inacc_fire,*rec_stand,*rec_crouch,*rec_stand_f,*rec_crouch_f; };
struct WO { bool *shot,*knife,*heavy,*first,*ev_commit,*ev_finish,*ev_rs,*ev_draw; double *ap_pitch,*ap_yaw,*pen; ll *wid; };
__device__ __forceinline__ void decay_step(double* r, double f, double lin, double hh, double vf, double g){
  double p=r[1]*f, y=r[2]*f;
  double mag=hypot(p,y);
  double k= mag>lin ? 1.0-(1.0/(mag<1e-300?1e-300:mag))*lin : 0.0;     // torch: python scalar / tensor = reciprocal(tensor) * scalar
  p=p*k; y=y*k;
  double vp=r[3], vy=r[4];
  p=p+(vp*hh)*0.5; y=y+(vy*hh)*0.5;
  vp=vp*vf; vy=vy*vf;
  p=p+(vp*hh)*0.5; y=y+(vy*hh)*0.5;
  r[1]=p; r[2]=y; r[3]=vp; r[4]=vy; r[5]*=g; r[6]*=g;
}
__device__ __forceinline__ void snap(double* r){
  const int A[3]={3,1,5}, B[3]={4,2,6};
  for(int q=0;q<3;q++){ if(fabs(r[A[q]])<1e-6 && fabs(r[B[q]])<1e-6){ r[A[q]]=0.0; r[B[q]]=0.0; } }
}
__device__ __forceinline__ double remap(double n, double a, double b, double c, double d){
  double den=b-a; if(den<1e-9) den=1e-9;
  double t=(n-a)/den; t= t<0.0?0.0:(t>1.0?1.0:t);
  return a==b ? d : c+(d-c)*t;
}
__global__ void wtick_k(WS S, WI I, WT T, WO O, int N, ll t0, const ll* tp){
  int pi=blockIdx.x*blockDim.x+threadIdx.x; if(pi>=N) return;
  ll t= tp? *tp : t0;
  bool alive=I.alive[pi];
  ll* W3=S.wid+(ll)pi*3;
  // ---- decay all slots (alive players, slots holding a weapon), inaccuracy recovery
  for(int sl=0;sl<3;sl++){
    ll ws=W3[sl]; bool live3=alive && ws>=0;
    double* r=S.rs+((ll)pi*3+sl)*7;
    if(live3){ decay_step(r,F1,LIN1,H1,VF1,G1); decay_step(r,F2,LIN2,H2,VF2,G2); snap(r); }
    int w= ws<0? 0 : (int)ws;
    double n=floor(r[0]);
    double ts=(double)T.trans_start[w], te=(double)T.trans_end[w];
    double rc=remap(n,ts,te,T.rec_crouch[w],T.rec_crouch_f[w]);
    double rsd=remap(n,ts,te,T.rec_stand[w],T.rec_stand_f[w]);
    double rt= I.crouch[pi]? rc : rsd;
    if(I.in_air[pi]) rt=T.rec_crouch[w]*4;
    if(!T.has_recoil[w]) rt=0.3;
    double* ia=S.inacc+(ll)pi*3+sl;
    double pen=(*ia)*exp((1.0/(rt<0.01?0.01:rt))*CREC);                // (scalar / tensor, as torch)
    if(pen<1e-5) pen=0.0;
    if(live3) *ia= (*ia>0)? pen : 0.0;
  }
  // ---- reload progress (active weapon)
  ll a=S.active[pi]; ll ia3=(ll)pi*3;
  ll wid=W3[a]; int w= wid<0? 0 : (int)wid;
  ll ru=S.reload_until[ia3+a], rc=S.reload_commit[ia3+a];
  bool rcm=S.committed[ia3+a], rsh=S.shell[ia3+a];
  ll ammo=S.ammo[ia3+a], res=S.reserve[ia3+a];
  ll mag=T.magazine[w];
  bool rel= alive && wid>=0 && ru>=0;
  bool evc=false, evf=false;
  bool sh= rel && rsh && t>=ru;
  bool ins= sh && res>0 && ammo<mag;
  if(ins){ ammo+=1; res-=1; evc=true; }
  bool cont= sh && res>0 && ammo<mag;
  if(cont) ru=ru+T.reload_ticks[w];
  bool fin_sh= sh && !cont;
  bool mg= rel && !rsh;
  bool do_commit= mg && !rcm && ((rc>=0 && t>=rc) || t>=ru);
  ll take= mag-ammo<res? mag-ammo : res;
  if(do_commit){ ammo+=take; res-=take; }
  rcm= rcm || do_commit; evc= evc || do_commit;
  bool fin_mg= mg && t>=ru;
  bool fin= fin_sh || fin_mg; evf= evf || fin;
  if(fin) ru=-1; if(fin_mg) rc=-1; if(fin){ rcm=false; rsh=false; }
  // ---- switch
  ll wsl=I.want[pi];
  bool tgt_has= wsl>=0 && W3[wsl<0?0:(wsl>2?2:wsl)]>=0;
  bool sw= alive && I.can_act[pi] && tgt_has && wsl!=a;
  bool cancel= sw && ru>=0;
  if(cancel){ ru=-1; rc=-1; rcm=false; rsh=false; }
  if(alive && wid>=0){ S.reload_until[ia3+a]=ru; S.reload_commit[ia3+a]=rc; S.committed[ia3+a]=rcm; S.shell[ia3+a]=rsh;
    S.ammo[ia3+a]=ammo; S.reserve[ia3+a]=res; }
  ll na= sw? wsl : a;
  if(sw){ S.pressed[ia3+na]=false; S.zoom[pi]=0; S.switch_until[pi]=t+DRAW_TICKS; }
  S.active[pi]=na;
  a=na; wid=W3[a]; w= wid<0? 0 : (int)wid; bool valid= alive && wid>=0;
  ru=S.reload_until[ia3+a]; ammo=S.ammo[ia3+a]; res=S.reserve[ia3+a]; mag=T.magazine[w];
  // ---- zoom
  ll zr=I.zoomreq[pi];
  if(valid && zr>=0){ ll z0= zr<0?0:zr; ll nz=T.n_zoom[w]; S.zoom[pi]= ru>=0? 0 : (z0<nz? z0 : nz); }
  // ---- reload request
  bool can_rel= valid && I.can_act[pi] && I.reload[pi] && ru<0 && mag>0 && ammo<mag && res>0 && t>=S.switch_until[pi];
  bool shellw=T.shell[w];
  if(can_rel){
    S.zoom[pi]=0;
    ru=t+T.reload_ticks[w];
    ll ct= ammo==0? T.commit_empty[w] : T.commit_full[w];
    S.reload_until[ia3+a]=ru; S.reload_commit[ia3+a]= shellw? -1 : t+ct; S.committed[ia3+a]=false; S.shell[ia3+a]=shellw;
  }
  rsh=S.shell[ia3+a];
  // ---- fire
  bool knife= wid==KNIFE_ID;
  bool trig= I.fire[pi] || (knife && I.fire2[pi]);
  bool edge= trig && !S.pressed[ia3+a];
  if(valid) S.pressed[ia3+a]=trig;
  ll sr=S.shot_ready[ia3+a];
  bool base= valid && I.can_act[pi] && trig && t>=sr && t>=S.switch_until[pi];
  bool gun_ok= !(ru>=0 && !(rsh && ammo>0)) && ammo>0 && (T.automatic[w] || edge);
  bool shot= base && !knife && gun_ok;
  bool kn= base && knife;
  bool heavy= kn && I.fire2[pi];
  bool first= kn && (t-S.last_knife[pi])>KNIFE_FIRST_GAP;
  if(kn){ S.last_knife[pi]=t; S.shot_ready[ia3+a]=t+(heavy?60:30); }
  if(shot){
    if(ru>=0){ S.reload_until[ia3+a]=-1; S.shell[ia3+a]=false; S.reload_commit[ia3+a]=-1; }
    S.ammo[ia3+a]=ammo-1; S.shots[ia3+a]+=1; S.last_shot[ia3+a]=t; S.shot_ready[ia3+a]=t+T.fire_ticks[w];
  }
  double* r=S.rs+(ia3+a)*7;
  double pena=S.inacc[ia3+a];
  O.ap_pitch[pi]=r[1]; O.ap_yaw[pi]=r[2];
  if(shot && T.has_recoil[w]){
    ll ix=((ll)floor(r[0]))%64; if(ix<0) ix+=64;
    const double* ent=T.recoil+((ll)w*64+ix)*2;
    double c=cos(ent[0]), sn=sin(ent[0]), m=ent[1];
    r[3]+=c*m; r[4]-=sn*m; r[5]+=c*m*VPX; r[6]-=sn*m*VPX; r[0]+=1;
  }
  if(shot) S.inacc[ia3+a]=pena+T.inacc_fire[w];
  if(shot && T.unzoom[w]) S.zoom[pi]=0;
  O.shot[pi]=shot; O.knife[pi]=kn; O.heavy[pi]=heavy; O.first[pi]=first; O.pen[pi]=pena; O.wid[pi]=wid;
  O.ev_commit[pi]=evc; O.ev_finish[pi]=evf; O.ev_rs[pi]=can_rel; O.ev_draw[pi]=sw;
}
#define LP(x) ((ll*)(x).data_ptr<int64_t>())
#define CLP(x) ((const ll*)(x).data_ptr<int64_t>())
void wtick(std::vector<torch::Tensor> st, std::vector<torch::Tensor> in, std::vector<torch::Tensor> tb, std::vector<torch::Tensor> out,
           int64_t t, torch::Tensor tdev){
  WS S; S.wid=LP(st[0]); S.ammo=LP(st[1]); S.reserve=LP(st[2]); S.shots=LP(st[3]); S.shot_ready=LP(st[4]); S.reload_until=LP(st[5]);
  S.reload_commit=LP(st[6]); S.last_shot=LP(st[7]); S.committed=st[8].data_ptr<bool>(); S.shell=st[9].data_ptr<bool>();
  S.pressed=st[10].data_ptr<bool>(); S.inacc=st[11].data_ptr<double>(); S.rs=st[12].data_ptr<double>(); S.active=LP(st[13]);
  S.switch_until=LP(st[14]); S.zoom=LP(st[15]); S.last_knife=LP(st[16]);
  WI I; I.alive=in[0].data_ptr<bool>(); I.can_act=in[1].data_ptr<bool>(); I.fire=in[2].data_ptr<bool>(); I.fire2=in[3].data_ptr<bool>();
  I.reload=in[4].data_ptr<bool>(); I.crouch=in[5].data_ptr<bool>(); I.in_air=in[6].data_ptr<bool>(); I.want=CLP(in[7]); I.zoomreq=CLP(in[8]);
  WT T; T.magazine=CLP(tb[0]); T.reload_ticks=CLP(tb[1]); T.n_zoom=CLP(tb[2]); T.commit_empty=CLP(tb[3]); T.commit_full=CLP(tb[4]);
  T.fire_ticks=CLP(tb[5]); T.trans_start=CLP(tb[6]); T.trans_end=CLP(tb[7]); T.shell=tb[8].data_ptr<bool>(); T.automatic=tb[9].data_ptr<bool>();
  T.has_recoil=tb[10].data_ptr<bool>(); T.unzoom=tb[11].data_ptr<bool>(); T.recoil=tb[12].data_ptr<double>(); T.inacc_fire=tb[13].data_ptr<double>();
  T.rec_stand=tb[14].data_ptr<double>(); T.rec_crouch=tb[15].data_ptr<double>(); T.rec_stand_f=tb[16].data_ptr<double>();
  T.rec_crouch_f=tb[17].data_ptr<double>();
  WO O; O.shot=out[0].data_ptr<bool>(); O.knife=out[1].data_ptr<bool>(); O.heavy=out[2].data_ptr<bool>(); O.first=out[3].data_ptr<bool>();
  O.ap_pitch=out[4].data_ptr<double>(); O.ap_yaw=out[5].data_ptr<double>(); O.pen=out[6].data_ptr<double>(); O.wid=LP(out[7]);
  O.ev_commit=out[8].data_ptr<bool>(); O.ev_finish=out[9].data_ptr<bool>(); O.ev_rs=out[10].data_ptr<bool>(); O.ev_draw=out[11].data_ptr<bool>();
  int N=(int)st[13].numel();
  const ll* tp= tdev.numel()? CLP(tdev) : nullptr;
  if(N) wtick_k<<<(N+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(S,I,T,O,N,(ll)t,tp);
}
"""
_wmod = None
_WST = ('w_wid', 'w_ammo', 'w_reserve', 'w_shots', 'w_shot_ready', 'w_reload_until', 'w_reload_commit', 'w_last_shot', 'w_reload_committed',
        'w_reload_shell', 'w_pressed', 'w_inacc', 'w_rs', 'active', 'switch_until', 'zoom', 'last_knife')
_WTB = ('magazine', 'reload_ticks', 'n_zoom', 'commit_empty', 'commit_full', 'fire_ticks', 'trans_start', 'trans_end', 'shell', 'automatic',
        'has_recoil', 'unzoom_after_shot', 'recoil_table', 'inacc_fire', 'rec_stand', 'rec_crouch', 'rec_stand_final', 'rec_crouch_final')
_WDT = dict(w_inacc=torch.float64, w_rs=torch.float64, w_reload_committed=torch.bool, w_reload_shell=torch.bool, w_pressed=torch.bool)


def _wext():
    global _wmod
    if _wmod is None:
        import hashlib, os
        from torch.utils.cpp_extension import load_inline
        h1, h2 = 1 / 64, 1 / 60 - 1 / 64
        c = dict(F1=math.exp(-R.DECAY2_EXP * h1), LIN1=R.DECAY2_LIN * h1, H1=h1, VF1=math.exp(-R.VEL_DECAY * h1), G1=math.exp(-R.VIEW_PUNCH_DECAY * h1),
                 F2=math.exp(-R.DECAY2_EXP * h2), LIN2=R.DECAY2_LIN * h2, H2=h2, VF2=math.exp(-R.VEL_DECAY * h2), G2=math.exp(-R.VIEW_PUNCH_DECAY * h2),
                 CREC=-(1 / 60) * math.log(10), VPX=R.VIEW_PUNCH_EXTRA)
        hdr = ''.join('#define %s %r\n' % kv for kv in c.items())
        from .tables import KNIFE as _KN
        hdr += '#define DRAW_TICKS %d\n#define KNIFE_FIRST_GAP %d\n#define KNIFE_ID %d\n' % (DRAW_TICKS, KNIFE_FIRST_GAP, _KN)
        src = hdr + _WK
        key = hashlib.sha1(src.encode()).hexdigest()[:10]
        os.environ.setdefault('TORCH_EXTENSIONS_DIR', './fps_data/tmp/torch_ext')
        _wmod = load_inline('fpsweap_' + key, cpp_sources='void wtick(std::vector<torch::Tensor> st, std::vector<torch::Tensor> in, '
                            'std::vector<torch::Tensor> tb, std::vector<torch::Tensor> out, int64_t t, torch::Tensor tdev);',
                            cuda_sources=src, functions=['wtick'], extra_cuda_cflags=['-O3', '-fmad=false'], verbose=False)
    return _wmod


def _fused_ok(s, alive):
    import os
    if not alive.is_cuda or os.environ.get('GPUSIM_WEAPONS_TORCH'):
        return False
    for k in _WST:
        v = s[k]
        if v.dtype != _WDT.get(k, torch.int64) or not v.is_contiguous():
            return False
    return True


def tick(s, T, t, alive, can_act, fire, fire2, reload, want_slot, zoom_req, crouch, in_air):
    """One tick for all players. Inputs [E,10]: alive/can_act/fire/fire2/reload/crouch/in_air bool; want_slot long (-1 none);
    zoom_req long (-1 = leave as is). t: int (or int64 device scalar). Returns dict(shot, knife, knife_heavy, knife_first, ap_pitch,
    ap_yaw, pen, wid, ...).  State is updated in place (the fused kernel) or partly re-bound (tick_torch)."""
    if not _fused_ok(s, alive):
        return tick_torch(s, T, t, alive, can_act, fire, fire2, reload, want_slot, zoom_req, crouch, in_air)
    dev = alive.device; sh = alive.shape
    tbl = T.get('_wtab')
    if tbl is None:
        tbl = T['_wtab'] = [T[k].contiguous() for k in _WTB]
    b = lambda x: x.to(device=dev, dtype=torch.bool).expand(sh).contiguous()
    l = lambda x: x.to(device=dev, dtype=torch.int64).expand(sh).contiguous()
    inp = [b(alive), b(can_act), b(fire), b(fire2), b(reload), b(crouch), b(in_air), l(want_slot), l(zoom_req)]
    B = lambda: torch.empty(sh, dtype=torch.bool, device=dev)
    D = lambda: torch.empty(sh, dtype=torch.float64, device=dev)
    out = dict(shot=B(), knife=B(), knife_heavy=B(), knife_first=B(), ap_pitch=D(), ap_yaw=D(), pen=D(),
               wid=torch.empty(sh, dtype=torch.int64, device=dev), ev_commit=B(), ev_finish=B(), ev_reload_start=B(), ev_draw=B())
    ol = [out[k] for k in ('shot', 'knife', 'knife_heavy', 'knife_first', 'ap_pitch', 'ap_yaw', 'pen', 'wid', 'ev_commit', 'ev_finish',
                           'ev_reload_start', 'ev_draw')]
    if torch.is_tensor(t):
        _wext().wtick([s[k] for k in _WST], inp, tbl, ol, 0, t.to(device=dev, dtype=torch.int64).reshape(1))
    else:
        _wext().wtick([s[k] for k in _WST], inp, tbl, ol, int(t), torch.zeros(0, dtype=torch.int64, device=dev))
    return out
