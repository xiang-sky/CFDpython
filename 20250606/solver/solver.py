import numpy as np
from .residual import compute_residual_ausm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import boundary.boundary as bd
from post_output.output_tecplot import output_tecplot
from post_output.output_tecplot import output_tecplot_series
from post_output.output_tecplot import output_forces
import pickle
import time

class CFDSolver:
    def __init__(self, blocks, gamma=config.GAMMA, cfl=0.1):
        self.blocks = blocks  # list of BlockData
        self.gamma = gamma
        self.cfl = cfl
        self.residuals = []
        self.iteration = 0
        self.temporal_discrete = 1
        self.if_localdt = 1

        for blk in self.blocks:
            blk.res = np.zeros_like(blk.fluid)

    def compute_primitive_variables(self, blk):
        U = blk.fluid
        rho = U[:, :, 0]
        u = U[:, :, 1] / rho
        v = U[:, :, 2] / rho
        E = U[:, :, 3] / rho
        p = (self.gamma - 1) * rho * (E - 0.5 * (u**2 + v**2))
        return rho, u, v, p

    def compute_time_step(self, blk):
        rho, u, v, p = self.compute_primitive_variables(blk)
        a = np.sqrt(self.gamma * p / rho)

        s = blk.s  # 法向量 (ni,nj,4,2)
        len_s = np.linalg.norm(s, axis=3)  # shape = (ni,nj,4)

        n1, n2, n3, n4 = [s[:, :, i, :] / (len_s[:, :, i:i+1] + 1e-12) for i in range(4)]
        len1, len2, len3, len4 = len_s[:, :, 0], len_s[:, :, 1], len_s[:, :, 2], len_s[:, :, 3]

        ni = 0.5 * (n2 - n4)
        nj = 0.5 * (n3 - n1)
        leni = 0.5 * (len2 + len4)
        lenj = 0.5 * (len1 + len3)

        s1 = (np.abs(u * ni[:, :, 0]) + a) * leni
        s2 = (np.abs(v * nj[:, :, 1]) + a) * lenj

        vol = blk.geo[:, :, 2]

        if self.if_localdt == 1:
            # 当地时间步长
            dt_local = self.cfl * vol / (s1 + s2)
        else:
            # 全局时间步长
            dt_min = np.min(self.cfl * vol / (s1 + s2))
            dt_local = np.full_like(vol, dt_min)

        return dt_local

    def compute_residual(self, blk):
        return compute_residual_ausm(blk, m=4, gamma=self.gamma)

    def apply_boundary_conditions(self):
        bd.boundary_farfeild(self.blocks)
        bd.boundary_wall_inviscid(self.blocks)
        bd.boundary_interface(self.blocks)

    def rk4_iterate(self):
        for blk in self.blocks:
            blk.U0 = blk.fluid.copy()
            blk.dt_local = self.compute_time_step(blk)
            blk.vol = blk.geo[:, :, 2]

        # === RK1 ===
        self.apply_boundary_conditions()
        for blk in self.blocks:
            blk.k1 = self.compute_residual(blk)

        # === RK2 ===
        for blk in self.blocks:
            dt, vol = blk.dt_local[:, :, None], blk.vol[:, :, None]
            blk.fluid = blk.U0 - 0.5 * (dt / vol) * blk.k1

        self.apply_boundary_conditions()
        for blk in self.blocks:
            blk.k2 = self.compute_residual(blk)

        # === RK3 ===
        for blk in self.blocks:
            dt, vol = blk.dt_local[:, :, None], blk.vol[:, :, None]
            blk.fluid = blk.U0 - 0.5 * (dt / vol) * blk.k2

        self.apply_boundary_conditions()
        for blk in self.blocks:
            blk.k3 = self.compute_residual(blk)

        # === RK4 ===
        for blk in self.blocks:
            dt, vol = blk.dt_local[:, :, None], blk.vol[:, :, None]
            blk.fluid = blk.U0 - (dt / vol) * blk.k3

        self.apply_boundary_conditions()
        for blk in self.blocks:
            blk.k4 = self.compute_residual(blk)

        # 合成解
        for blk in self.blocks:
            dt, vol = blk.dt_local[:, :, None], blk.vol[:, :, None]
            blk.fluid = blk.U0 - (dt / vol) * (
                (1/6) * blk.k1 + (1/3) * blk.k2 + (1/3) * blk.k3 + (1/6) * blk.k4
            )
            blk.res = blk.k4

        # 清除临时量
        for blk in self.blocks:
            for attr in ['U0', 'dt_local', 'vol', 'k1', 'k2', 'k3', 'k4']:
                delattr(blk, attr)

        self.iteration += 1

    def lu_sgs_iterate(self):
        gamma = self.gamma
        pass

        self.iteration += 1

    def compute_global_residual_norm(self):
        return max(np.linalg.norm(blk.res) for blk in self.blocks)

    def run(self, max_iter=10000, tol=1e-3):
        with open("history.dat", "w") as f:
            f.write("Iter\tResidual\tFx\tFy\tTime(s)\n")

        # 初始化计时器
        start_time = time.time()

        for _ in range(max_iter):

            if self.temporal_discrete == 1:
                self.rk4_iterate()
            elif self.temporal_discrete == 2:
                self.lu_sgs_iterate()

            res_norm = self.compute_global_residual_norm()
            self.residuals.append(res_norm)

            if self.iteration % 10 == 0:
                fx, fy = output_forces(self.blocks)

                # 记录从上一次10次迭代起的时间
                elapsed = time.time() - start_time
                start_time = time.time()  # 重置计时器

                print(f"[Iter {self.iteration}] Residual = {res_norm:.3e}")
                print(f"Forces = {fx:.6f}, {fy:.6f}")
                print(f"Time for last 10 iters = {elapsed:.2f} s")

                with open("history.dat", "a") as f:
                    f.write(f"{self.iteration}\t\t\t{res_norm:.6e}\t\t\t\t{fx:.6f}\t\t\t{fy:.6f}\t\t\t{elapsed:.2f}\n")

            os.makedirs("results", exist_ok=True)
            if self.iteration % 10 == 0:
                tecplot_filename = os.path.join("results", f"solution_iter_{self.iteration}.dat")
                pkl_filename = os.path.join("results", f"blocks_result_iter_{self.iteration}.pkl")
                #output_tecplot(self.blocks, tecplot_filename)
                output_tecplot_series(self.blocks, self.iteration, tecplot_filename)
                with open(pkl_filename, 'wb') as f:
                    pickle.dump(self.blocks, f)

            if res_norm < tol:
                print("收敛达到停止条件")
                break