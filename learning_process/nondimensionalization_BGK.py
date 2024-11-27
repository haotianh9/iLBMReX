# criterion introduced in Kruger 2017 Chapter 7.2

import numpy as np

# physical quantities
Re = 200
U = 1  # characteristic velocity
U_max = 3  # estimated maximum velocity
L = 1  # characteristic length
size = 32  # domain size
rho = 1  # characteristic density

# lattice quantities
cs_star = 1 / np.sqrt(3)  # sound speed in lattice units
Delta_x = 1  # lattice spacing
Delta_t = 1  # lattice time step

# goal is to find characteristic length and velocty in lattice units

U_star = 0.1
# tau_star = 0.51
# tau_star = 1
L_star = 20
# rho_star = 1

try:
    tau_star
except NameError:
    nu_star = U_star * L_star / Re
    tau_star = nu_star * Delta_t / Delta_x**2 / cs_star**2 + 0.5
try:
    L_star
except NameError:
    nu_star = cs_star**2 * (tau_star - 0.5) * Delta_x**2 / Delta_t
    L_star = Re * nu_star / U_star
print("L_star: {}, nu_star: {}, tau_star: {}".format(L_star, nu_star, tau_star))


U_max_star = U_max / U * U_star
# grid Reynolds number
Re_g = U_max_star * Delta_x / nu_star
Ma = U_star / cs_star
alpha = 1 / 8  # constant

if U_star > 0.1:
    print("U_star is too large: {}".format(U_star))
if U_star > 0.03:
    print("U_star is a little large: {}".format(U_star))
if Re_g > 10:
    print("Re_g is too large: {}".format(Re_g))
if tau_star > 0.5 + alpha * U_max_star:
    print("tau_star is too large: {}, {}".format(tau_star, 0.5 + alpha * U_max_star))
if Ma > 0.3:
    print("Ma is too large: {}".format(Ma))
