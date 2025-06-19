import numpy as np


class LUSGSSolver:
    def __init__(self, blocks, gamma=1.4, CFL=1.5):
        """
        参数:
            blocks: list of dict，包含每个 block 的网格、守恒量、边界等信息
            gamma: 比热比
            CFL: CFL 数
        """
        self.blocks = blocks
        self.gamma = gamma
        self.CFL = CFL
        self.residuals = []  # 收敛历史
        self.iteration = 0

    def compute_primitive_variables(self, blk):
        """由守恒量计算原始变量：rho, u, v, p"""
        pass

    def compute_time_step(self, blk):
        """计算局部时间步长 Δt"""
        pass

    def compute_residual(self, blk):
        """计算每个块的残差（有限体积通量差）"""
        pass

    def forward_sweep(self, blk, dt, res):
        """LU-SGS 的前向步"""
        pass

    def backward_sweep(self, blk, dt, deltaU):
        """LU-SGS 的后向步"""
        pass

    def apply_update(self, blk, deltaU):
        """更新守恒变量"""
        pass

    def apply_boundary_conditions(self):
        """统一调用边界条件模块"""
        pass  # 可调用已有的边界处理代码

    def step(self):
        """执行一次迭代"""
        self.apply_boundary_conditions()
        for blk in self.blocks:
            prim = self.compute_primitive_variables(blk)
            dt = self.compute_time_step(blk)
            res = self.compute_residual(blk)
            deltaU = self.forward_sweep(blk, dt, res)
            deltaU = self.backward_sweep(blk, dt, deltaU)
            self.apply_update(blk, deltaU)
        self.iteration += 1

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

    def compute_global_residual_norm(self):
        """返回所有块的残差范数，用于监控收敛"""
        pass
