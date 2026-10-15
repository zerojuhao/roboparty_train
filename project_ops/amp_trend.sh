#!/bin/bash
# usage: amp_trend.sh LOGFILE [every]
L=$1; E=${2:-50}
awk -v E=$E '
/Learning iteration/ { if (it!="") emit(); split($0,a,"iteration "); split(a[2],b,"/"); it=b[1]+0; delete m }
/Mean episode length:/ {m["eplen"]=$NF}
/Mean reward:/ {m["rew"]=$NF}
/Steps per second:/ {m["sps"]=$NF}
/Episode_Termination\/time_out:/ {m["to"]=$NF}
/Episode_Termination\/base_contact:/ {m["bc"]=$NF}
/Episode_Termination\/bad_orientation:/ {m["bo"]=$NF}
/Episode_Reward\/lean_orientation_l2:/ {m["lean"]=$NF}
/Episode_Reward\/track_lin_vel_xy_exp:/ {m["trk"]=$NF}
/Episode_Reward\/track_ang_vel_z_exp:/ {m["yaw"]=$NF}
/Episode_Reward\/flat_orientation_l2:/ {m["flat"]=$NF}
/AMP\/rollout:/ {split($0,c,"style_reward="); split(c[2],d," "); m["amp"]=d[1]}
/AMP\/general:/ {split($0,c,"accuracy="); split(c[2],d," "); m["dacc"]=d[1]}
/max_velocity:/ {m["vmax"]=$NF}
/success_rate:/ {m["succ"]=$NF}
/error_vel_xy:/ {m["evxy"]=$NF}
function emit() { if (it%E==0 || last) printf "%6d eplen=%7s to=%s bc=%s bo=%s lean=%s flat=%s trk=%s yaw=%s amp=%s dacc=%s vmax=%s succ=%s evxy=%s sps=%s\n", it,m["eplen"],m["to"],m["bc"],m["bo"],m["lean"],m["flat"],m["trk"],m["yaw"],m["amp"],m["dacc"],m["vmax"],m["succ"],m["evxy"],m["sps"] }
END { last=1; emit() }' "$L"