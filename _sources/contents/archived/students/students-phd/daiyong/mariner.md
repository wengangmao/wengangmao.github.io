# MSS-Mariner python code

+++
---

```python
import math
import numpy as np
from copy import deepcopy
# sigma1 = 0.000
# sigma2 = 0.00
# sigma3 = 0.00
# beta1 = 0.0001
# beta2 = 0.001
# beta3 = 0.0002
# sigma1 = 0.0001
# sigma2 = 0.0001
# sigma3 = 0.001
# beta1 = 0.000
# beta2 = 0.0008
# beta3 = 0.0002


def mariner(x, ui, U0):
    # need to check the parameters, like the x, max_rudder_speed, ship's speed!!
    """
    x = [u v r x y psi delta] discribe state of the ship, numpy array data
    u     = pertubed surge velocity about Uo (m/s)
    v     = pertubed sway velocity about zero (m/s)
    r     = pertubed yaw velocity about zero (rad/s)
    x     = position in x-direction (m)
    y     = position in y-direction (m)
    psi   = pertubed yaw angle about zero (rad)
    delta = actual rudder angle (rad)

    rudder_angle: rudder angel (-40*pi/180 ~ 40*pi/180 rad)
    max_rudder_speed: max rudder speed (0~5 °/s)
    ship_speed: speed of the ship (0~7.5 m/s)
    return: dotx = [u, v,r,x,y,psi,delta]
    """
    # Normalization variables
    # U0 = 7.7175
    L = 160.93
    U = math.sqrt(math.pow(U0 + x[0], 2) + math.pow(x[1], 2))

    # Non - dimensional states and inputs
    # change rudder
    delta_c = -ui   # delta_c = -rudder_angle such that positive delta_c -> positive r

    u = x[0] / U
    v = x[1] / U
    r = x[2] * L / U 
    psi = x[5]
    delta = x[6]

    # Parameters, hydrodynamic derivatives and main dimensions
    delta_max = 40  # max rudder angle(deg)
    Ddelta_max = 5  # max rudder derivative(deg / s)

    m = 798e-5
    Iz = 39.2e-5
    xG = -0.023

    Xudot = -42e-5
    Yvdot = -748e-5
    Nvdot = 4.646e-5
    Xu = -184e-5
    Yrdot = -9.354e-5
    Nrdot = -43.8e-5
    Xuu = -110e-5
    Yv = -1160e-5
    Nv = -264e-5
    Xuuu = -215e-5
    Yr = -499e-5
    Nr = -166e-5
    Xvv = -899e-5
    Yvvv = -8078e-5
    Nvvv = 1636e-5
    Xrr = 18e-5
    Yvvr = 15356e-5
    Nvvr = -5483e-5
    Xdd = -95e-5
    Yvu = -1160e-5
    Nvu = -264e-5
    Xudd = -190e-5
    Yru = -499e-5
    Nru = -166e-5
    Xrv = 798e-5
    Yd = 278e-5
    Nd = -139e-5
    Xvd = 93e-5
    Yddd = -90e-5
    Nddd = 45e-5
    Xuvd = 93e-5
    Yud = 556e-5
    Nud = -278e-5
    Yuud = 278e-5
    Nuud = -139e-5
    Yvdd = -4e-5
    Nvdd = 13e-5
    Yvvd = 1190e-5
    Nvvd = -489e-5
    Y0 = -4e-5
    N0 = 3e-5
    Y0u = -8e-5
    N0u = 6e-5
    Y0uu = -4e-5
    N0uu = 3e-5

    # Masses and moments of inertia
    m11 = m - Xudot
    m22 = m - Yvdot
    m23 = m * xG - Yrdot
    m32 = m * xG - Nvdot
    m33 = Iz - Nrdot

    # Rudder saturation and dynamics
    if abs(delta_c) >= delta_max * math.pi / 180:
        delta_c = np.sign(delta_c) * delta_max * math.pi / 180

    delta_dot = delta_c - delta
    if abs(delta_dot) >= Ddelta_max * math.pi / 180:
        delta_dot = np.sign(delta_dot) * Ddelta_max * math.pi / 180

    # Forces and moments
    X = Xu * u + Xuu * u ** 2 + Xuuu * u ** 3 + Xvv * v ** 2 + Xrr * r ** 2 + Xrv * r * v + Xdd * delta ** 2 \
        + Xudd * u * delta ** 2 + Xvd * v * delta + Xuvd * u * v * delta
         # + sigma1*(-1+2*np.random.rand(1)[0]) + beta1

    Y = Yv * v + Yr * r + Yvvv * v ** 3 + Yvvr * v ** 2 * r + Yvu * v * u + Yru * r * u + Yd * delta \
        + Yddd * delta ** 3 + Yud * u * delta + Yuud * u ** 2 * delta + Yvdd * v * delta ** 2 \
        + Yvvd * v ** 2 * delta + (Y0 + Y0u * u + Y0uu * u ** 2)
         # + sigma2*(-1+2*np.random.rand(1)[0]) + beta2

    N = Nv * v + Nr * r + Nvvv * v ** 3 + Nvvr * v ** 2 * r + Nvu * v * u + Nru * r * u + Nd * delta \
        + Nddd * delta ** 3 + Nud * u * delta + Nuud * u ** 2 * delta + Nvdd * v * delta ** 2 \
        + Nvvd * v ** 2 * delta + (N0 + N0u * u + N0uu * u ** 2)
         # + sigma3*(-1+2*np.random.rand(1)[0]) + beta3

    # Dimensional tate derivative
    detM22 = m22 * m33 - m23 * m32

    xdot = np.array([X * (U ** 2 / L) / m11,
                    -(-m33 * Y + m23 * N) * (U ** 2 / L) / detM22,
                    (-m32 * Y + m22 * N) * (U ** 2 / L ** 2) / detM22,
# change x y raw:(xy) --> (yx)
                    (math.cos(psi) * (U0 / U + u) - math.sin(psi) * v) * U,
                    (math.sin(psi) * (U0 / U + u) + math.cos(psi) * v) * U,
   #                 (math.sin(psi) * (U0 / U + u) + math.cos(psi) * v) * U,
   #                 (math.cos(psi) * (U0 / U + u) - math.sin(psi) * v) * U,
                    r * (U / L),
                    delta_dot])

    return xdot,U


def euler2(xdot, x, delta_t):
    """
    xdot: delta variables of the ship's parameters
    x: state of the ship
    delta_t: delta time of each step
    """
    xnext = x + delta_t * xdot
    return xnext


# def differentiator(state, real_value, sample_t, state_num=2):
#     x_k = np.zeros(state_num)
#     for i in range(state_num):
#         x_k[i] = deepcopy(state[i])
#     tao1 = sample_t
#     tao2 = sample_t
#     state[0] = x_k[0] + sample_t * x_k[1]
#     state[1] = x_k[1] - sample_t * (1 / (tao1 * tao2) * (x_k[0] - real_value) + (tao1 + tao2) / (tao1 * tao2) * x_k[1])
#     return state


# def linear_estimator(z, y, u):
#     e = (y-z[0][0])
#     b0 = 1.0
#     w = 12.
#     A = np.array([[0,1,0],[0,0,1],[0,0,0]])
#     B = np.array([[0],[b0],[0]])
#     # C = np.array([1,0,0])
#     # E = np.array([0,0,1])
#     L = np.array([[3*w],[3*math.pow(w,2)],[math.pow(w,3)]])
#     dz = np.matmul(A,z)+B*u+L*e
#     z_ = z+.05*dz
#     # z_ = z_*math.pi/180

#     return z_


# def linear_estimator_2(z, y, u):
#     e = y
#     b0 = 1.0
#     w = 8.
#     A = np.array([[0,1,0],[0,0,1],[0,0,0]])
#     B = np.array([[0],[b0],[0]])
#     # C = np.array([1,0,0])
#     # E = np.array([0,0,1])
#     L = np.array([[3*w],[3*math.pow(w,2)],[math.pow(w,3)]])
#     dz = np.matmul(A,z)+B*u+L*e
#     z_ = z+.05*dz
#     # z_ = z_*math.pi/180

#     return z_


```