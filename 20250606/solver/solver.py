import numpy as np
from residual import conflux_ausm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class RK1Solver:
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
        """计算局部时间步长 Δt，简单基于最大特征速度"""
        rho, u, v, p = self.compute_primitive_variables(blk)
        a = np.sqrt(self.gamma * p / rho)  # 声速
        S = np.abs(u) + a
        dt_x = blk['geo'].shape[0] / np.max(S)  # 简化估计：dx / max speed
        dt_y = blk['geo'].shape[1] / np.max(S)
        dt = self.cfl * min(dt_x, dt_y)
        return dt

    def compute_residual(self, blk):
        """计算残差，调用你已有的compute_residual_roe或类似函数"""
        # 简单示范，假设你有compute_residual_roe函数
        res = compute_residual_roe(blk, m=4, gamma=self.gamma)
        return res

    def apply_boundary_conditions(self):
        """统一调用边界条件模块"""
        # 这里调用已有的边界处理代码
        import boundary_conditions as bc
        for blk in self.blocks:
            bc.apply_all_boundary_conditions(blk)

    def apply_update(self, blk, deltaU):
        """更新守恒变量"""
        blk['fluid'] += deltaU

    def step(self):
        """执行一次迭代，显式RK1步骤"""
        self.apply_boundary_conditions()
        for blk in self.blocks:
            dt = self.compute_time_step(blk)
            res = self.compute_residual(blk)
            deltaU = -dt * res  # 显式欧拉
            self.apply_update(blk, deltaU)
        self.iteration += 1

    def compute_global_residual_norm(self):
        """计算所有块残差范数，用于监控收敛"""
        norm_total = 0.0
        for blk in self.blocks:
            res = self.compute_residual(blk)
            norm_total += np.linalg.norm(res)
        return norm_total

    def run(self, max_iter=10000, tol=1e-6):
        """主求解循环"""
        for _ in range(max_iter):
            self.step()
            res_norm = self.compute_global_residual_norm()
            self.residuals.append(res_norm)
            print(f"[Iter {self.iteration}] Residual = {res_norm:.3e}")
            if res_norm < tol:
                print("收敛达到停止条件")
                break
