import numpy as np

N_DIM = 2   # 计算维数
N_C = 4     # 计算守恒量个数(二维欧拉4个）
GHOST_LAYER = 2  #虚网格层数



GAMMA = 1.4              # 比热比
R_GAS = 287.05           # 空气气体常数 J/(kg·K)



#  远场条件
P_FAR= 127238.62  # 来流压力
T_FAR = 288.15     # 来流温度
MA_FAR = 0.3       # 来流马赫数
DR_FAR = [0.999390827, 0.0348994967]  # 来流方向


#  参考值
DR_FAR = DR_FAR / (np.linalg.norm(DR_FAR))  # 来流方向归一化
A_ANGLE = np.arctan(DR_FAR(1) / DR_FAR(0))  # 来流攻角
V_FAR = np.sqrt(GAMMA * R_GAS * T_FAR) * MA_FAR  # 远场速度
V_SOUND_FAR = np.sqrt(GAMMA * R_GAS * T_FAR)     # 远场声速
# 取来流声速与来流速度之中的较大值为参考速度
if V_FAR > V_SOUND_FAR:
    V_REF = V_FAR
else:
    V_REF = V_SOUND_FAR
L_REF = 1.0  # 网格参考长度
PHO_REF = P_FAR / (R_GAS * T_FAR)  # 参考密度
P_REF = PHO_REF * V_REF * V_REF    # 参考压强(2倍动压）
R_REF = (V_REF * V_REF) / T_FAR    # 参考气体常数


# 远场无量纲值
R_DL = R_GAS / R_REF  # 无量纲气体常数
VI_DL_FAR = (V_REF * np.cos(A_ANGLE)) / V_REF        # 无量纲速度ui
VJ_DL_FAR = (V_REF * np.sin(A_ANGLE)) / V_REF        # 无量纲速度uj
PHO_DL_FAR = 1  # 无量纲密度
T_DL_FAR = 1    # 无量纲温度

# 远场无量纲守恒量
