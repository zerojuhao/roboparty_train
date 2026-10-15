import re
p='/data/zym/roboparty_train/logs/rsl_rl/rp1_amp_path/2026-10-11_19-11-34_amp_path_c22_flight_leangate15_clamp030_w4_from36500_20261012/params/env.yaml'
t=open(p).read()
m=re.search(r'high_speed_double_flight:.*?^  weight:\s*([0-9.]+)', t, re.S|re.M)
# also try weight nested differently
m2=re.search(r'high_speed_double_flight:[\s\S]*?weight:\s*([0-9.]+)', t)
print('flight_weight', (m or m2).group(1) if (m or m2) else 'MISSING')
# show snippet around weight after high_speed
idx=t.find('high_speed_double_flight:')
print(t[idx:idx+1200][:800])
