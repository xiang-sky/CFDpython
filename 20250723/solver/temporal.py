from numba import njit
import numpy as np


@njit(parallel=False, cache=True)
def forward_sweep_numba(fluid, res_in, lambdax, lambday, D, deltaU, s, gamma, w):
    nx, ny, _ = fluid.shape
    for diag in range(nx + ny - 1):
        for i in range(nx):
            j = diag - i
            if 0 <= j < ny:
                len_s = np.sqrt(np.sum(s[i, j] ** 2, axis=1))
                n = s[i, j] / (len_s[:, None] + 1e-12)
                n1, n2, n3, n4 = n

                u = fluid[i, j]
                res = -res_in[i, j].copy()
                lambx = lambdax[i, j]
                lamby = lambday[i, j]

                acx = conjacobian_numba(u, n4, gamma)
                acy = conjacobian_numba(u, n1, gamma)
                lx = 0.5 * (acx * len_s[3] - (w * lambx) * np.eye(4))
                ly = 0.5 * (acy * len_s[0] - (w * lamby) * np.eye(4))

                if i > 0:
                    res -= lx @ deltaU[i - 1, j]
                if j > 0:
                    res -= ly @ deltaU[i, j - 1]

                deltaU[i, j] = np.linalg.solve(D[i, j], res)


@njit(parallel=False, cache=True)
def backward_sweep_numba(fluid, deltaU, lambdax, lambday, D, s, gamma, w):
    nx, ny, _ = fluid.shape
    for diag in range(nx + ny - 1, -1, -1):
        for i in range(nx):
            j = diag - i
            if 0 <= j < ny:
                len_s = np.sqrt(np.sum(s[i, j] ** 2, axis=1))
                n = s[i, j] / (len_s[:, None] + 1e-12)
                n1, n2, n3, n4 = n

                u = fluid[i, j]
                res = D[i, j] @ deltaU[i, j].copy()
                lambx = lambdax[i, j]
                lamby = lambday[i, j]

                acx = conjacobian_numba(u, n2, gamma)
                ux = 0.5 * (acx * len_s[1] - (w * lambx) * np.eye(4))
                acy = conjacobian_numba(u, n3, gamma)
                uy = 0.5 * (acy * len_s[2] - (w * lamby) * np.eye(4))

                if i < nx - 1:
                    res -= ux @ deltaU[i + 1, j]
                if j < ny - 1:
                    res -= uy @ deltaU[i, j + 1]

                deltaU[i, j] = np.linalg.solve(D[i, j], res)


@njit(cache=True)
def compute_lusgs_d_numba(dt_local, vol, lambdax, lambday, w):
    nx, ny = dt_local.shape
    D = np.zeros((nx, ny, 4, 4), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            D[i, j] = np.eye(4) * ((vol[i, j] / dt_local[i, j]) + w * (lambdax[i, j] + lambday[i, j]))
    return D


@njit(cache=True)
def conjacobian_numba(U, n, gamma):
    rho = U[0]
    u = U[1] / rho
    v = U[2] / rho
    E_total = U[3] / rho
    p = (gamma - 1) * (U[3] - 0.5 * rho * (u ** 2 + v ** 2))

    vn = u * n[0] + v * n[1]
    rha = 0.5 * (gamma - 1) * (u ** 2 + v ** 2)
    E = (p / ((gamma - 1) * rho)) + 0.5 * (u ** 2 + v ** 2)
    a1 = gamma * E - rha
    a2 = gamma - 1
    a3 = gamma - 2

    ac = np.empty((4, 4), dtype=np.float64)
    ac[0, :] = [0, n[0], n[1], 0]
    ac[1, :] = [n[0] * rha - u * vn, vn - a3 * n[0] * u, n[1] * u - a2 * n[0] * v, a2 * n[0]]
    ac[2, :] = [n[1] * rha - v * vn, n[0] * v - a2 * n[1] * u, vn - a3 * n[1] * v, a2 * n[1]]
    ac[3, :] = [vn * (rha - a1), n[0] * a1 - a2 * u * vn, n[1] * a1 - a2 * v * vn, gamma * vn]
    return ac