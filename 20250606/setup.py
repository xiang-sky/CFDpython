import numpy as np

def setup_farfield():
    # 参数（常量）
    a = 1.458000623e-6      # Sutherland公式中的μ0
    Ts = 110.4              # Sutherland公式中的Ts

    global Uinf_c, m_c, Mainf_c, alpinf_c, gamma0_c, R_c, Tinf_c
    global Vref_c, rhoinf_c, Pinf_c, Pref_c, Pinf2ref_c
    global Rdl_c, cpdl_c, gamma0m1_c, mall_c, FlowType_c
    global miuinf_c, Reref_c, miu2Re_c, Ts_c, sigma_c, sigmadl_c

    if Mainf_c != 0:
        # 速度方向归一化
        norm = np.linalg.norm(Uinf_c[1:m_c-1])
        Uinf_c[1:m_c-1] = Uinf_c[1:m_c-1] / norm
        # 来流迎角
        alpinf_c = np.arctan(Uinf_c[2] / Uinf_c[1])
    else:
        Uinf_c[1:m_c-1] = 0.0

    # 来流声速
    Vref_c = np.sqrt(gamma0_c * R_c * Tinf_c)
    if Mainf_c != 0.0:
        Vref_c *= Mainf_c
    Vref2 = Vref_c ** 2

    # 来流密度、参考压力、来流压力/参考压力
    rhoinf_c = Pinf_c / (R_c * Tinf_c)
    Pref_c = rhoinf_c * Vref2
    Pinf2ref_c = Pinf_c / Pref_c

    # 参考气体常数 & 无量纲参数
    Rref = Vref2 / Tinf_c
    Rdl_c = R_c / Rref
    cpdl_c = gamma0_c * Rdl_c / gamma0m1_c

    # 设置来流守恒变量
    Uinf_c[0] = 1.0
    Uinf_c[m_c - 1] = Pinf2ref_c / gamma0m1_c + 0.5

    # 若为粘性流动（FlowType_c > 1）
    if FlowType_c > 1:
        # Sutherland公式计算来流粘度
        miuinf = a * np.sqrt(Tinf_c ** 3) / (Tinf_c + Ts)
        # 参考Re数（特征长度为1）
        Reref_c = rhoinf_c * Vref_c / miuinf
        # μ / Re 的系数
        miu2Re_c = a * np.sqrt(Tinf_c) / (miuinf * Reref_c)
        Ts_c = Ts / Tinf_c

        # 来流分子粘性用于无量纲化
        miuinf_c[0] = miuinf
        CP_Conserv2Primit(Uinf_c[:m_c], miuL2Re=miuinf_c[0])  # 自定义函数，需要你自己实现

        if FlowType_c == 3:  # SA湍流模型
            Uinf_c[mall_c - 1] = 0.1
            miuinf_c[1] = 2.793982878e-6 * miuinf_c[0]
        elif FlowType_c == 4:  # k-omega SST模型
            Uinf_c[m_c] = 9.0e-9      # rho*K
            Uinf_c[mall_c - 1] = 1.0e-6  # rho*omega
            miuinf_c[1] = 1.0e-5 * miuinf_c[0]

        # 无量纲辐射强度项
        sigmadl_c = sigma_c * Tinf_c ** 4 / (Pref_c * Vref_c)
