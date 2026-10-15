import os, sys, json, pickle, numpy as np, mujoco
sys.path.insert(0, "/home/zym/roboparty_train/robo_newton")
V="/data/zym/roboparty_train/videos/amp_path_4mps_upperfix_20261011"
M="/data/zym/roboparty_train/robo_newton_data/motions/v0"
XML="/data/zym/roboparty_train/robo_newton_data/robots/roboparty/rp1.3/mjcf/rp1_flat.xml"
MOTION_JOINT_NAMES = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_roll_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_yaw_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_joint", "right_wrist_joint",
]
# verify against the live cfg file text
src=open("/home/zym/roboparty_train/robo_newton/robolab/tasks/manager_based/amp/rp1_amp_env_cfg.py").read()
assert all(n in src for n in MOTION_JOINT_NAMES)
ARM=["shoulder_pitch","shoulder_roll","shoulder_yaw","elbow","wrist"]
D=np.rad2deg
model=mujoco.MjModel.from_xml_path(XML); data=mujoco.MjData(model)
def jrange(n):
    j=model.joint(n); return (float(j.range[0]), float(j.range[1])) if model.jnt_limited[j.id] else (None,None)

def stats(x, fps=50):
    dx=np.diff(x)
    # jitter: RMS of 2nd difference (deg/step^2) and high-freq power fraction >8Hz
    dd=np.diff(x,2)
    X=np.abs(np.fft.rfft(x-x.mean()))**2; f=np.fft.rfftfreq(len(x),1/fps)
    hf=float(X[f>8].sum()/max(X[f>0.5].sum(),1e-12))
    return dict(mean=D(x.mean()),std=D(x.std()),min=D(x.min()),max=D(x.max()),
                rms_dd=D(np.sqrt((dd**2).mean())), hf=hf)

# demo
demo={}
for name in ["run_loop_180_R_003__A326","jog_forward_A021","jog_forward_A045"]:
    d=pickle.load(open(f"{M}/{name}.pkl","rb")); q=np.asarray(d["dof_pos"])
    demo[name]={n:q[:,MOTION_JOINT_NAMES.index(n)] for n in MOTION_JOINT_NAMES}
    demo[name]["_kb"]=np.asarray(d["key_body_pos"]); demo[name]["_root"]=(np.asarray(d["root_pos"]),np.asarray(d["root_rot"]))

def quat_rot_inv(q, v):  # q wxyz? assume xyzw if |w| ambiguous -> handled by calle
    w,x,y,z=q; R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    return R.T@v

def wrist_y_policy(z):
    names=[str(n) for n in z["joint_names"]]; q=z["joint_position"]; rp=z["root_position_w"]; rq=z["root_quaternion_wxyz"]
    lb=model.body("left_wrist_link").id; rb=model.body("right_wrist_link").id; base=model.body(1).id
    L=[];R=[]
    for t in range(len(q)):
        data.qpos[:3]=rp[t]; data.qpos[3:7]=rq[t]
        for i,n in enumerate(names): data.qpos[model.jnt_qposadr[model.joint(n).id]]=q[t,i]
        mujoco.mj_kinematics(model,data)
        Rb=data.xmat[base].reshape(3,3); pb=data.xpos[base]
        L.append((Rb.T@(data.xpos[lb]-pb))[1]); R.append((Rb.T@(data.xpos[rb]-pb))[1])
    return np.array(L),np.array(R)

def wrist_y_demo(dm):
    kb=dm["_kb"]; rp,rr=dm["_root"]
    # key bodies order: l_ankle,r_ankle,l_knee,r_knee,l_wrist,r_wrist ; try wxyz then xyzw, pick the one giving L>R mostly
    out={}
    for conv in ("wxyz","xyzw"):
        L=[];R=[]
        for t in range(len(kb)):
            q=rr[t] if conv=="wxyz" else np.r_[rr[t][3],rr[t][:3]]
            # key_body_pos may be world or root-relative
            lw=kb[t,4]; rw=kb[t,5]
            if np.linalg.norm(lw-rp[t])<np.linalg.norm(lw): lw=lw-rp[t]; rw=rw-rp[t]
            L.append(quat_rot_inv(q,lw)[1]); R.append(quat_rot_inv(q,rw)[1])
        out[conv]=(np.array(L),np.array(R))
    conv=max(out,key=lambda c:(out[c][0]-out[c][1]).mean())
    return out[conv]

pol={}
for tag,fn in [("step4","c15_30000_step_vx4p0_10s.npz"),("step2","c15_30000_step_vx2p0_10s.npz"),("ramp4","c15_30000_ramp_0to4p0_2s_10s.npz")]:
    z=np.load(f"{V}/{fn}",allow_pickle=True); names=[str(n) for n in z["joint_names"]]; m=z["time"]>=2.0
    pol[tag]={n:z["joint_position"][m,names.index(n)] for n in names}
    L,R=wrist_y_policy(z); pol[tag]["_wy"]=(L[m],R[m])

res={"limits":{},"policy":{},"demo":{},"wrist_y":{}}
for side in ("left","right"):
    for a in ARM:
        n=f"{side}_{a}_joint"; lo,hi=jrange(n); res["limits"][n]=(D(lo) if lo is not None else None, D(hi) if hi is not None else None)
for tag in pol:
    res["policy"][tag]={n:stats(pol[tag][n]) for n in res["limits"]}
    L,R=pol[tag]["_wy"]; res["wrist_y"][tag]=dict(L_min=float(L.min()),L_mean=float(L.mean()),R_max=float(R.max()),R_mean=float(R.mean()))
for dn in demo:
    res["demo"][dn]={n:stats(demo[dn][n]) for n in res["limits"]}
    L,R=wrist_y_demo(demo[dn]); res["wrist_y"][dn]=dict(L_min=float(L.min()),L_mean=float(L.mean()),R_max=float(R.max()),R_mean=float(R.mean()))
json.dump(res,open(f"{V}/c15_30000_arm_anomaly.json","w"),indent=1,default=float)

# print table
def f(x): return "  n/a" if x is None else f"{x:6.1f}"
for side in ("left","right"):
    for a in ARM:
        n=f"{side}_{a}_joint"; lo,hi=res["limits"][n]
        print(f"\n## {n}  limits [{f(lo)},{f(hi)}] deg")
        print("   src       mean    std    min    max  rms_dd   hf>8Hz")
        for src,S in [("step4",res["policy"]["step4"][n]),("step2",res["policy"]["step2"][n]),("ramp4",res["policy"]["ramp4"][n])]+[(d[:10],res["demo"][d][n]) for d in res["demo"]]:
            print(f"   {src:10s}{S['mean']:6.1f} {S['std']:6.1f} {S['min']:6.1f} {S['max']:6.1f} {S['rms_dd']:6.2f}  {S['hf']:.3f}")
print("\n## wrist lateral y in base frame (m): +y=left. crossing if L_min<0 or R_max>0")
for k,v in res["wrist_y"].items(): print(f"   {k:28s}"+"  ".join(f"{a}={b:+.3f}" for a,b in v.items()))
