# exmaple: flow over an obstacle from book Mohamad 2011 Lattice Boltzmann Method
# LBM- 2-D2Q9, flow over an obstacle, Re=400, note that c2=1/3, w9=4/9, w1-4=1/9, and w5-w8, 1/36
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

Nx = 501
Ny = 81
u0 = 0.1  # characteristic flow velocity (comparing to the speed of sound)
f = np.zeros((Nx, Ny, 9))  # distribution function
feq = np.zeros((Nx, Ny, 9))  # distribution function at equilibrium
Nt = 2090
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
x1 = (Nx - 1) / (Ny - 1)
y1 = 1
x = np.linspace(0, Nx - 1, Nx)
y = np.linspace(0, Ny - 1, Ny)
alpha = 0.01
ReH = u0 * (Ny - 1) / alpha
ReD = u0 * 10 / alpha
print(ReH, ReD)
omega = 1 / (3 * alpha + 0.5)
count[0] = 0
# setting velocity
u[0, 1:-1] = u0


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


def collition(Nx, Ny, u, v, cx, cy, omega, f, rho, w):
    for j in range(Ny):
        for i in range(Nx):
            t1 = u[i, j] ** 2 + v[i, j] ** 2
            # print(t1)
            for k in range(9):
                t2 = u[i, j] * cx[k] + v[i, j] * cy[k]
                # print(t2)
                feq[i, j, k] = rho[i, j] * w[k] * (1 + 3 * t2 + 4.5 * t2**2 - 1.5 * t1)
                f[i, j, k] = (1 - omega) * f[i, j, k] + omega * feq[i, j, k]


def obstc(Nx, Ny, f, u, v, u0, rho):
    # length of obsticale= nx/5, and has sides of 10 units
    nxb = int((Nx - 1) / 5) - 1
    nxe = nxb + 10
    nyb = int((Ny - 1 - 10) / 2) - 1
    # nyb=35
    nye = nyb + 10
    f[nxb:nxe, nyb, 3] = f[nxb:nxe, nyb, 1]
    f[nxb:nxe, nyb, 6] = f[nxb:nxe, nyb, 4]
    f[nxb:nxe, nyb, 7] = f[nxb:nxe, nyb, 5]
    f[nxb:nxe, nye, 1] = f[nxb:nxe, nye, 3]
    f[nxb:nxe, nye, 4] = f[nxb:nxe, nye, 6]
    f[nxb:nxe, nye, 5] = f[nxb:nxe, nye, 7]
    # bottom, and top boundary, bounce back
    f[nxb, nyb:nye, 2] = f[nxb, nyb:nye, 0]
    f[nxb, nyb:nye, 6] = f[nxb, nyb:nye, 4]
    f[nxb, nyb:nye, 5] = f[nxb, nyb:nye, 7]
    f[nxe, nyb:nye, 0] = f[nxe, nyb:nye, 2]
    f[nxe, nyb:nye, 4] = f[nxe, nyb:nye, 6]
    f[nxe, nyb:nye, 7] = f[nxe, nyb:nye, 7]
    # inside the obstacle
    u[nxb:nxe, nyb:nye] = 0
    v[nxb:nxe, nyb:nye] = 0


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


def ruv(Nx, Ny, f, rho, u, v):
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


def stream(f):
    for k in range(9):
        f[:, :, k] = np.roll(f[:, :, k], shift=(cx[k], cy[k]), axis=(0, 1))


for time_index in tqdm(range(Nt)):
    collition(Nx, Ny, u, v, cx, cy, omega, f, rho, w)
    stream(f)
    boundary(Nx, Ny, f, u, v, u0, rho)
    obstc(Nx, Ny, f, u, v, u0, rho)
    ruv(Nx, Ny, f, rho, u, v)
    count[time_index] = time_index
    utim[time_index] = rho[int((Nx - 1) / 2), int((Ny - 1) / 2)]
result(Nx, Ny, x, y, u, v, u0, rho, count, utim)
