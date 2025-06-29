import numpy as np
from .residual import compute_residual_ausm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import boundary.boundary as bd
from post_output.output_tecplot import output_tecplot
import pickle


class RK3Solver:
    def __init__(self, blocks, gamma=config.GAMMA, cfl=0.1):
        """
        参数:
            blocks: list of dict，包含每个 block 的网格、守恒量、边界等信息
            gamma: 比热比
            CFL: CFL 数
        """
        self.blocks = blocks
        self.gamma = gamma
        self.cfl = cfl
        self.residuals = []  # 收敛历史
        self.iteration = 0

        # 初始化每个 block 的残差矩阵
        for blk in self.blocks:
            shape = blk['fluid'].shape  # shape = (ni, nj, 4)
            blk['res'] = np.zeros(shape)

    def compute_primitive_variables(self, blk):
        """由守恒量计算原始变量：rho, u, v, p
        """
        U = blk['fluid']  # shape (ni, nj, 4)
        rho = U[:, :, 0]
        u = U[:, :, 1] / rho
        v = U[:, :, 2] / rho
        E = U[:, :, 3] / rho
        p = (self.gamma - 1) * rho * (E - 0.5 * (u**2 + v**2))
        return rho, u, v, p

    def compute_time_step(self, blk):
        """基于最大特征速度和体积计算每个点的时间步后取全局最小"""
        rho, u, v, p = self.compute_primitive_variables(blk)
        a = np.sqrt(self.gamma * p / rho)  # 声速
        v1 = blk['geo'][:, :, 3:5]
        v2 = blk['geo'][:, :, 5:7]
        v3 = blk['geo'][:, :, 7:9]
        v4 = blk['geo'][:, :, 9:11]
        len1 = np.linalg.norm(v1, axis=2)  # shape = (ni, nj)
        len2 = np.linalg.norm(v2, axis=2)
        len3 = np.linalg.norm(v3, axis=2)
        len4 = np.linalg.norm(v4, axis=2)
        n1 = v1 / len1[:, :, None]  # shape = (ni, nj, 2)
        n2 = v2 / len2[:, :, None]
        n3 = v3 / len3[:, :, None]
        n4 = v4 / len4[:, :, None]

        ni = 0.5 * (n2 - n4)
        nj = 0.5 * (n3 - n1)
        leni = 0.5 * (len2 + len4)
        lenj = 0.5 * (len1 + len3)

        s1 = (np.abs(u * ni[:, :, 0]) + a) * leni
        s2 = (np.abs(v * nj[:, :, 1]) + a) * lenj

        vol = blk['geo'][:, :, 2]  # 控制体积 |Ω|

        dt_local = self.cfl * vol / (s1 + s2)  # 每个点自己的 dt
    #    dt = np.min(dt_local)  # 全局最小时间步  注释掉已使用全局时间步长
        return dt_local

    def compute_residual(self, blk):
        """计算残差"""
        res = compute_residual_ausm(blk, m=4, gamma=self.gamma)
        return res

    def apply_boundary_conditions(self):
        """统一调用边界条件模块"""
        bd.boundary_farfeild(self.blocks)
        bd.boundary_wall_inviscid(self.blocks)
        bd.boundary_interface(self.blocks)

    def iterate(self):
        """执行一次迭代，采用 3 阶 Runge-Kutta 显式格式"""

        # 1. 保存初始状态 W^(n)
        for blk in self.blocks:
            blk['U0'] = blk['fluid'].copy()  # 保存初始状态
            blk['dt_local'] = self.compute_time_step(blk)  # 计算并缓存局部时间步（数组）

        # === RK Step 1 ===
        self.apply_boundary_conditions()
        for blk in self.blocks:
            U0 = blk['U0']
            dt_local = blk['dt_local']
            vol = blk['geo'][:, :, 2]
            res1 = self.compute_residual(blk)
            W1 = U0 - (dt_local[:, :, None] / vol[:, :, None]) * res1
            blk['fluid'] = W1

        # === RK Step 2 ===
        self.apply_boundary_conditions()
        for blk in self.blocks:
            U0 = blk['U0']
            W1 = blk['fluid']
            dt_local = blk['dt_local']
            vol = blk['geo'][:, :, 2]
            res2 = self.compute_residual(blk)
            W2 = (3 / 4) * U0 + (1 / 4) * (W1 - (dt_local[:, :, None] / vol[:, :, None]) * res2)
            blk['fluid'] = W2

        # === RK Step 3 ===
        self.apply_boundary_conditions()
        for blk in self.blocks:
            U0 = blk['U0']
            W2 = blk['fluid']
            dt_local = blk['dt_local']
            vol = blk['geo'][:, :, 2]
            res3 = self.compute_residual(blk)
            W3 = (1 / 3) * U0 + (2 / 3) * (W2 - (dt_local[:, :, None] / vol[:, :, None]) * res3)
            blk['fluid'] = W3
            blk['res'] = res3  # 最终残差

        for blk in self.blocks:
            blk.pop('U0', None)
            blk.pop('dt_local', None)

        self.iteration += 1

    def compute_global_residual_norm(self):
        """计算所有块中残差范数的最大值"""
        norm_max = 0.0
        for blk in self.blocks:
            res = blk['res']
            norm_blk = np.linalg.norm(res)
            norm_max = max(norm_max, norm_blk)
        return norm_max

    def run(self, max_iter=10000, tol=1e-3):
        """主求解循环"""
        for _ in range(max_iter):
            self.iterate()
            res_norm = self.compute_global_residual_norm()
            self.residuals.append(res_norm)
            if self.iteration % 20 == 0:
                print(f"[Iter {self.iteration}] Residual = {res_norm:.3e}")

            if self.iteration % 100 == 0:
                tecplot_filename = f"solution_iter_{self.iteration}.dat"
                pkl_filename = f"blocks_result_iter_{self.iteration}.pkl"

                # 保存 Tecplot 文件
                output_tecplot(self.blocks, tecplot_filename)

                # 保存 Python 对象
                with open(pkl_filename, 'wb') as f:
                    pickle.dump(self.blocks, f)

            if res_norm < tol:
                print("收敛达到停止条件")
                break
