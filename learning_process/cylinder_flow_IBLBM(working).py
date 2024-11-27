# exmaple: flow over an obstacle from book Mohamad 2011 Lattice Boltzmann Method
# LBM- 2-D2Q9, flow over an obstacle, Re=400, note that c2=1/3, w9=4/9, w1-4=1/9, and w5-w8, 1/36
# Add immersed boundary method for a cylinder obstacle based on Wu & Shu 2009

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

Nx = 801
Ny = 201
u0 = 0.04  # characteristic flow velocity (comparing to the speed of sound)
f = np.zeros((Nx, Ny, 9))  # distribution function
feq = np.zeros((Nx, Ny, 9))  # distribution function at equilibrium
Nt = 90
utim = np.zeros(Nt)
count = np.zeros(Nt)
u = u0 * np.ones((Nx, Ny))
v = np.zeros((Nx, Ny))
rho = 2 * np.ones((Nx, Ny))
x = np.zeros(Nx)
y = np.zeros(Ny)
Tm = np.zeros(Nx)
Tvm = np.zeros(Nx)
w = np.array([1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 4 / 9])
cx = np.array([1, 0, -1, 0, 1, -1, -1, 1, 0])
cy = np.array([0, 1, 0, -1, 1, 1, -1, -1, 0])
c2 = 1 / 3
dx = 1
dy = 1
dt = 1
x1 = (Nx - 1) / (Ny - 1)
y1 = 1
x = np.linspace(0, Nx - 1, Nx)
y = np.linspace(0, Ny - 1, Ny)
alpha = 0.05
ReH = u0 * (Ny - 1) / alpha
x_center = Nx / 4
y_center = Ny / 2
radius = 5
n_res = 50
# print(np.linspace(0, 2 * np.pi, n_res, endpoint=False))
x_lag = radius * np.cos(np.linspace(0, 2 * np.pi, n_res, endpoint=False)) + x_center
y_lag = radius * np.sin(np.linspace(0, 2 * np.pi, n_res, endpoint=False)) + y_center

ds = 2 * np.pi * radius / n_res

xmesh, ymesh = np.meshgrid(x, y)


IB_force = np.zeros((Nx, Ny, 2))

ReD = u0 * radius * 2 / alpha
print(ReH, ReD)
omega = 1 / (3 * alpha + 0.5)

count[0] = 0
# setting velocity
u[0, 1:-1] = u0
order_delta = 2


def kernel(x_lag, y_lag, x_mesh, y_mesh, order):
    deltax = x_mesh - x_lag
    deltay = y_mesh - y_lag

    mask = np.logical_and(np.abs(deltax) < 2, np.abs(deltay) < 2)
    return (
        mask
        / 16
        * (1 + np.cos(np.pi / 2 * np.abs(deltax)))
        * (1 + np.cos(np.pi / 2 * np.abs(deltay)))
    )
    # if np.abs(deltax) >= 2 or np.abs(deltay) >= 2:
    #     return 0
    # else:
    #     return (
    #         1
    #         / 16
    #         * (1 + np.cos(np.pi / 2 * np.abs(deltax)))
    #         * (1 + np.cos(np.pi / 2 * np.abs(deltay)))
    #     )
    # if order == 2:

    #     if np.abs(deltax) > dx or np.abs(deltay) > dy:
    #         return 0
    #     else:
    #         return (1 - np.abs(deltax)) * (1 - np.abs(deltay))
    # elif order == 3:
    #     if np.abs(deltax) > 1.5 * dx or np.abs(deltay) > 1.5 * dy:
    #         return 0
    #     else:
    #         res = 1
    #         if np.abs(deltax) < dx * 0.5:
    #             res = res * 1 / 3 * (1 + np.sqrt(1 - 3 * deltax**2))
    #         else:
    #             res = (
    #                 res
    #                 * 1
    #                 / 6
    #                 * (
    #                     5
    #                     - 3 * np.abs(deltax)
    #                     - np.sqrt(-2 + 6 * np.abs(deltax) - 3 * deltax**2)
    #                 )
    #             )
    #         if np.abs(deltay) < dy * 0.5:
    #             res = res * 1 / 3 * (1 + np.sqrt(1 - 3 * deltay**2))
    #         else:
    #             res = (
    #                 res
    #                 * 1
    #                 / 6
    #                 * (
    #                     5
    #                     - 3 * np.abs(deltay)
    #                     - np.sqrt(-2 + 6 * np.abs(deltay) - 3 * deltay**2)
    #                 )
    #             )
    #         return res
    # elif order == 4:
    #     if np.abs(deltax) > 2 * dx or np.abs(deltay) > 2 * dy:
    #         return 0
    #     else:
    #         res = 1
    #         if np.abs(deltax) < dx:
    #             res = (
    #                 res
    #                 * 1
    #                 / 8
    #                 * (
    #                     3
    #                     - 2 * np.abs(deltax)
    #                     + np.sqrt(1 + 4 * np.abs(deltax) - 4 * deltax**2)
    #                 )
    #             )
    #         else:
    #             res = (
    #                 res
    #                 * 1
    #                 / 8
    #                 * (
    #                     5
    #                     - 2 * np.abs(deltax)
    #                     - np.sqrt(-7 + 12 * np.abs(deltax) - 4 * deltax**2)
    #                 )
    #             )
    #         if np.abs(deltay) < dy:
    #             res = (
    #                 res
    #                 * 1
    #                 / 8
    #                 * (
    #                     3
    #                     - 2 * np.abs(deltay)
    #                     + np.sqrt(1 + 4 * np.abs(deltay) - 4 * deltay**2)
    #                 )
    #             )
    #         else:
    #             res = (
    #                 res
    #                 * 1
    #                 / 8
    #                 * (
    #                     5
    #                     - 2 * np.abs(deltay)
    #                     - np.sqrt(-7 + 12 * np.abs(deltay) - 4 * deltay**2)
    #                 )
    #             )
    #         return res
    # else:
    #     raise NotImplementedError("order not implemented")


# if os.path.exists("matrix_A_inv.npy"):
#     print("loading matrix_A_inv.npy")
#     matrix_A_inv = np.load("matrix_A_inv.npy")
# else:
#     matrix_A = np.zeros((n_res, n_res))
#     for lag_idx_i in tqdm(range(n_res)):
#         for lag_idx_j in range(lag_idx_i, n_res):
#             # mark=np.zeros((Nx, Ny))
#             # print(lag_idx_i, lag_idx_j)
#             # for i in range(Nx):
#             #     for j in range(Ny):
#             #         value= np.sum(
#             #             kernel(
#             #                 x_lag[lag_idx_i], y_lag[lag_idx_i],x[i],y[j], order_delta
#             #             )
#             #             * kernel(
#             #                 x_lag[lag_idx_j], y_lag[lag_idx_j], x[i],y[j], order_delta
#             #             )
#             #             * ds
#             #             * dx
#             #             * dy
#             #         )
#             #         matrix_A[lag_idx_i, lag_idx_j] += value
#             #         if lag_idx_j != lag_idx_i:
#             #             matrix_A[lag_idx_j, lag_idx_i] += value
#             value = np.sum(
#                 kernel(x_lag[lag_idx_i], y_lag[lag_idx_i], xmesh, ymesh, order_delta)
#                 * kernel(x_lag[lag_idx_j], y_lag[lag_idx_j], xmesh, ymesh, order_delta)
#                 * ds
#                 * dx
#                 * dy
#             )
#             matrix_A[lag_idx_i, lag_idx_j] += value
#             if lag_idx_j != lag_idx_i:
#                 matrix_A[lag_idx_j, lag_idx_i] += value
#                 # if value > 0:
#                 #     mark[i,j]=1
#             # print(matrix_A[lag_idx_i, lag_idx_j] )
#             # plt.figure()
#             # plt.scatter(x_lag[lag_idx_i], y_lag[lag_idx_i])
#             # plt.scatter(x_lag[lag_idx_j], y_lag[lag_idx_j])
#             # plt.imshow(mark.T)
#             # plt.colorbar()
#             # plt.axis("equal")
#             # plt.show()
#         # print(matrix_A[lag_idx_i, :])
#     # eig_val, eig_vec = np.linalg.eig(matrix_A)
#     # print("eigen value of matrix A: ", np.sort(eig_val))
#     # print(np.linalg.det(matrix_A * 1000))
#     # input()
#     matrix_A_inv = np.linalg.inv(matrix_A)
#     # matrix_A_inv = np.linalg.inv(matrix_A)
#     # np.save("matrix_A_inv.npy", matrix_A_inv)
#     # print("matrix_A_inv.npy saved")
B_x_dir = np.zeros(n_res)
B_y_dir = np.zeros(n_res)


# def boundary(Nx, Ny, f, u, v, u0, rho):
#     # right hand boundary

#     f[Nx - 1, :, 2] = f[Nx - 2, :, 2]
#     f[Nx - 1, :, 6] = f[Nx - 2, :, 6]
#     f[Nx - 1, :, 5] = f[Nx - 2, :, 5]
#     # bottom, and top boundary Neumann boundary condition
#     for i in range(Nx):
#         # f[i, 0, 1] = f[i, 0, 3]
#         # f[i, 0, 4] = f[i, 0, 6]
#         # f[i, 0, 5] = f[i, 0, 7]
#         # f[i, Ny - 1, 3] = f[i, Ny - 1, 1]
#         # f[i, Ny - 1, 6] = f[i, Ny - 1, 4]
#         # f[i, Ny - 1, 7] = f[i, Ny - 1, 5]
#         u[i, 0] = u[i, 1]
#         v[i, 0] = v[i, 1]
#         u[i, Ny - 1] = u[i, Ny - 2]
#         v[i, Ny - 1] = u[i, Ny - 2]
#     # left boundary, velocity is given as u0
#     for j in range(1, Ny - 1):
#         f[0, j, 0] = f[0, j, 2] + 2 * rho[0, j] * u0 / 3
#         f[0, j, 4] = f[0, j, 6] - 0.5 * (f[0, j, 1] - f[0, j, 3]) + rho[0, j] * u0 / 6
#         f[0, j, 7] = f[0, j, 5] + 0.5 * (f[0, j, 1] - f[0, j, 3]) + rho[0, j] * u0 / 6
#         u[0, j] = u0
#         v[0, j] = 0


def boundary(Nx, Ny, f, u, v, u0, rho):
    # right hand boundary

    f[Nx - 1, :, 2] = f[Nx - 2, :, 2]
    f[Nx - 1, :, 6] = f[Nx - 2, :, 6]
    f[Nx - 1, :, 5] = f[Nx - 2, :, 5]
    # bottom, and top boundary, bounce back
    for i in range(Nx):
        f[i, 0, 1] = f[i, 0, 3]
        f[i, 0, 4] = f[i, 0, 6]
        f[i, 0, 5] = f[i, 0, 7]
        f[i, Ny - 1, 3] = f[i, Ny - 1, 1]
        f[i, Ny - 1, 6] = f[i, Ny - 1, 4]
        f[i, Ny - 1, 7] = f[i, Ny - 1, 5]
        u[i, 0] = 0
        v[i, 0] = 0
        u[i, Ny - 1] = 0
        v[i, Ny - 1] = 0
    # left boundary, velocity is given as u0
    for j in range(1, Ny - 1):
        f[0, j, 0] = f[0, j, 2] + 2 * rho[0, j] * u0 / 3
        f[0, j, 4] = f[0, j, 6] - 0.5 * (f[0, j, 1] - f[0, j, 3]) + rho[0, j] * u0 / 6
        f[0, j, 7] = f[0, j, 5] + 0.5 * (f[0, j, 1] - f[0, j, 3]) + rho[0, j] * u0 / 6
        u[0, j] = u0
        v[0, j] = 0


def collition(Nx, Ny, u, v, cx, cy, omega, f, rho, w, with_BI=False):
    for j in range(Ny):
        for i in range(Nx):
            t1 = u[i, j] ** 2 + v[i, j] ** 2
            # print(t1)
            for k in range(9):
                t2 = u[i, j] * cx[k] + v[i, j] * cy[k]
                # print(t2)
                feq[i, j, k] = rho[i, j] * w[k] * (1 + 3 * t2 + 4.5 * t2**2 - 1.5 * t1)

                f[i, j, k] = (
                    (1 - omega) * f[i, j, k]
                    + omega * feq[i, j, k]
                    + (1 - 2 * omega) * w[k]
                )
                if with_BI:
                    f[i, j, k] += (
                        IB_force[i, j, 0] * (cx[k] - u[i, j])
                        + IB_force[i, j, 1] * (cy[k] - v[i, j])
                    ) / c2 + (
                        t2 * (IB_force[i, j, 0] * cx[k] + IB_force[i, j, 1] * cy[k])
                    ) / c2**2


def immersed_boundary(x_lag, y_lag):

    for idx_lag in range(n_res):
        B_x_dir[idx_lag] = 0
        B_y_dir[idx_lag] = 0
        for i in range(Nx):
            for j in range(Ny):
                B_x_dir[idx_lag] -= (
                    kernel(x_lag[idx_lag], y_lag[idx_lag], x[i], y[j], order_delta)
                    * u[i, j]
                    * dx
                    * dy
                )
                B_y_dir[idx_lag] -= (
                    kernel(x_lag[idx_lag], y_lag[idx_lag], x[i], y[j], order_delta)
                    * v[i, j]
                    * dx
                    * dy
                )
    # delta_ux = np.matmul(matrix_A_inv, B_x_dir)
    # delta_uy = np.matmul(matrix_A_inv, B_y_dir)
    # delta_ux = np.linalg.solve(matrix_A, B_x_dir)
    # delta_uy = np.linalg.solve(matrix_A, B_y_dir)
    # print(np.linalg.det(matrix_A_inv))
    print(np.mean(np.square(B_x_dir)), np.mean(np.square(B_y_dir)))
    for i in range(Nx):
        for j in range(Ny):
            IB_force[i, j, 0] = 0
            IB_force[i, j, 1] = 0
            for idx_lag in range(n_res):
                IB_force[i, j, 0] += (
                    kernel(x_lag[idx_lag], y_lag[idx_lag], x[i], y[j], order_delta)
                    * B_x_dir[idx_lag]
                    * ds
                )
                IB_force[i, j, 1] += (
                    kernel(x_lag[idx_lag], y_lag[idx_lag], x[i], y[j], order_delta)
                    * B_y_dir[idx_lag]
                    * ds
                )
    IB_force[:, :, 0] = IB_force[:, :, 0] * rho * 2 / dt
    IB_force[:, :, 1] = IB_force[:, :, 1] * rho * 2 / dt
    # print(np.max(IB_force[:, :, 0]))
    # print(np.min(IB_force[:, :, 0]))
    # plt.figure()
    # plt.imshow(IB_force[:, :, 0].T)
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(IB_force[:, :, 1].T)
    # plt.colorbar()
    # plt.show()


# def obstc(Nx, Ny, f, u, v, u0, rho):
#     # length of obsticale= nx/5, and has sides of 10 units
#     nxb = int((Nx - 1) / 5) - 1
#     nxe = nxb + 10
#     nyb = int((Ny - 1 - 10) / 2) - 1
#     # nyb=35
#     nye = nyb + 10
#     f[nxb:nxe, nyb, 3] = f[nxb:nxe, nyb, 1]
#     f[nxb:nxe, nyb, 6] = f[nxb:nxe, nyb, 4]
#     f[nxb:nxe, nyb, 7] = f[nxb:nxe, nyb, 5]
#     f[nxb:nxe, nye, 1] = f[nxb:nxe, nye, 3]
#     f[nxb:nxe, nye, 4] = f[nxb:nxe, nye, 6]
#     f[nxb:nxe, nye, 5] = f[nxb:nxe, nye, 7]
#     # bottom, and top boundary, bounce back
#     f[nxb, nyb:nye, 2] = f[nxb, nyb:nye, 0]
#     f[nxb, nyb:nye, 6] = f[nxb, nyb:nye, 4]
#     f[nxb, nyb:nye, 5] = f[nxb, nyb:nye, 7]
#     f[nxe, nyb:nye, 0] = f[nxe, nyb:nye, 2]
#     f[nxe, nyb:nye, 4] = f[nxe, nyb:nye, 6]
#     f[nxe, nyb:nye, 7] = f[nxe, nyb:nye, 7]
#     # inside the obstacle
#     u[nxb:nxe, nyb:nye] = 0
#     v[nxb:nxe, nyb:nye] = 0


# Plots for channel flow
def result(Nx, Ny, x, y, u, v, u0, rho, count, utim):
    Tm1 = u[50, :] / u0
    Tm2 = u[100, :] / u0
    Tm3 = u[260, :] / u0
    Tm4 = u[300, :] / u0
    umx = u[:, int((Ny - 1) / 2) - 1] / u0
    vmx = v[:, int((Ny - 1) / 2) - 1] / u0
    plt.figure(1)
    plt.plot(x / (Nx - 1), umx, label="u")
    plt.plot(x / (Nx - 1), vmx, label="v")
    plt.xlabel("x/L")
    plt.legend()
    plt.title("Velocity profile at mid channel")
    plt.figure(2)
    plt.plot(Tm1, y, label="x/L=0.1")
    plt.plot(Tm2, y, label="x/L=0.2")
    plt.plot(Tm3, y, label="x/L=0.5")
    plt.plot(Tm4, y, label="x/L=0.6")
    plt.xlabel("u/u0")
    plt.ylabel("y/L")
    plt.legend()
    plt.title("Velocity profile at different x/L")
    plt.figure(3)
    plt.plot(count, utim)
    # strema function calculation
    sx = np.zeros((Nx, Ny))
    sy = np.zeros((Nx, Ny))
    sx[:, :] = x[:, np.newaxis]
    sy[:, :] = y[np.newaxis, :]
    str = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(1, Ny):
            str[i, j] = str[i, j - 1] + 0.5 * (u[i, j] + u[i, j - 1])
    plt.figure(4)
    plt.contour(sx, sy, str)
    plt.colorbar()
    plt.axis("equal")
    plt.title("Stream function")
    plt.figure(5)
    plt.contour(sx, sy, u)
    plt.colorbar()
    plt.axis("equal")
    plt.title("Velocity profile")
    plt.show()


def ruv(Nx, Ny, f, rho, u, v, with_BI=False):
    rho[:, :] = np.sum(f, axis=2)
    for i in range(Nx):
        rho[i, Ny - 1] = (
            f[i, Ny - 1, 8]
            + f[i, Ny - 1, 0]
            + f[i, Ny - 1, 2]
            + 2 * (f[i, Ny - 1, 1] + f[i, Ny - 1, 5] + f[i, Ny - 1, 4])
        )
    u[:, :] = (
        np.sum(f[:, :, (0, 4, 7)], axis=2) - np.sum(f[:, :, (2, 5, 6)], axis=2)
    ) / rho

    v[:, :] = (
        np.sum(f[:, :, (1, 4, 5)], axis=2) - np.sum(f[:, :, (3, 6, 7)], axis=2)
    ) / rho
    if with_BI:
        u[:, :] += 0.5 * IB_force[:, :, 0] / rho * dt
        v[:, :] += 0.5 * IB_force[:, :, 1] / rho * dt


def stream(f):
    # Periodic streaming using numpy's roll operation
    for i in range(9):
        f[:, :, i] = np.roll(np.roll(f[:, :, i], cx[i], axis=1), cy[i], axis=0)
    # for k in range(9):
    #     f[:, :, k] = np.roll(f[:, :, k], shift=(cx[k], cy[k]), axis=(0, 1))


plt.figure(figsize=(19, 4.07))
for time_index in tqdm(range(Nt)):
    f_old = np.copy(f)

    for iter in range(10):
        f_before_iter = np.copy(f)

        if iter == 0:
            collition(Nx, Ny, u, v, cx, cy, omega, f, rho, w)
        ruv(Nx, Ny, f, rho, u, v)
        immersed_boundary(x_lag, y_lag)

        collition(Nx, Ny, u, v, cx, cy, omega, f, rho, w, with_BI=True)
        ruv(Nx, Ny, f, rho, u, v, with_BI=True)
        error = np.mean(np.square(f - f_before_iter))

        print(error)
        if error < 1e-6:
            break
    # obstc(Nx, Ny, f, u, v, u0, rho)
    stream(f)
    # boundary(Nx, Ny, f, u, v, u0, rho)

    count[time_index] = time_index
    utim[time_index] = rho[int((Nx - 1) / 2), int((Ny - 1) / 2)]
    # print(xmesh.shape, u.shape)

    # calculate vorticity from velocity field
    vorticity = np.zeros((Nx, Ny))
    for i in range(1, Nx - 1):
        for j in range(1, Ny - 1):
            vorticity[i, j] = (
                (v[i + 1, j] - v[i - 1, j] - u[i, j + 1] + u[i, j - 1]) / 2 / dx
            )
    plt.imshow(vorticity.T, cmap="coolwarm", origin="lower", extent=[0, Nx, 0, Ny])
    plt.plot(x_lag, y_lag, "k")
    plt.clim([-1, 1])
    plt.colorbar()
    plt.axis("equal")
    plt.pause(0.1)
    plt.clf()

    # plt.imshow(u.T, cmap="coolwarm", origin="lower", extent=[0, Nx, 0, Ny])
    # plt.plot(x_lag, y_lag, "k")
    # plt.clim([-0.1, 0.3])
    # plt.colorbar()
    # plt.axis("equal")
    # plt.pause(0.1)
    # plt.clf()
result(Nx, Ny, x, y, u, v, u0, rho, count, utim)
