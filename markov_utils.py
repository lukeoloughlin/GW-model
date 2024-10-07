from typing import Any

import numpy as np
import numpy.typing as npt


def RyR_Q(params: Any, CaSS: float) -> npt.NDArray:
    Q = np.zeros((6, 6))

    Q[0, 1] = params.k12 * CaSS**2
    Q[0, 0] = -Q[0, 1]

    Q[1, 0] = params.k21
    Q[1, 2] = params.k23 * CaSS**2
    Q[1, 4] = params.k25 * CaSS**2
    Q[1, 1] = -(Q[1, 0] + Q[1, 2] + Q[1, 4])

    Q[2, 1] = params.k32
    Q[2, 3] = params.k34 * CaSS**2
    Q[2, 2] = -(Q[2, 1] + Q[2, 3])

    Q[3, 2] = params.k43
    Q[3, 4] = params.k45
    Q[3, 3] = -(Q[3, 2] + Q[3, 4])

    Q[4, 1] = params.k52
    Q[4, 3] = params.k54 * CaSS**2
    Q[4, 5] = params.k56 * CaSS**2
    Q[4, 4] = -(Q[4, 1] + Q[4, 3] + Q[4, 5])

    Q[5, 4] = params.k65
    Q[5, 5] = -Q[5, 4]
    return Q


def LCC_Q(params: Any, V: float, CaSS: float) -> npt.NDArray:
    Q = np.zeros((12, 12))

    alpha = 2.0 * np.exp(0.012 * (V - 35))
    beta = 0.0882 * np.exp(-0.05 * (V - 35))
    gamma = params.gamma0 * CaSS

    Q[0, 1] = 4 * alpha
    Q[0, 6] = gamma
    Q[0, 0] = -(Q[0, 1] + Q[0, 6])

    Q[1, 0] = beta
    Q[1, 2] = 3 * alpha
    Q[1, 7] = gamma * params.a
    Q[1, 1] = -(Q[1, 0] + Q[1, 2] + Q[1, 7])

    Q[2, 1] = 2 * beta
    Q[2, 3] = 2 * alpha
    Q[2, 8] = gamma * params.a**2
    Q[2, 2] = -(Q[2, 1] + Q[2, 3] + Q[2, 8])

    Q[3, 2] = 3 * beta
    Q[3, 4] = alpha
    Q[3, 9] = gamma * params.a**3
    Q[3, 3] = -(Q[3, 2] + Q[3, 4] + Q[3, 9])

    Q[4, 3] = 4 * beta
    Q[4, 5] = params.f
    Q[4, 10] = gamma * params.a**4
    Q[4, 4] = -(Q[4, 3] + Q[4, 5] + Q[4, 10])

    Q[5, 4] = params.g
    Q[5, 5] = -Q[5, 4]

    Q[6, 7] = 4 * alpha * params.a
    Q[6, 1] = params.omega
    Q[6, 6] = -(Q[6, 1] + Q[6, 7])

    Q[7, 6] = beta / params.b
    Q[7, 8] = 3 * alpha * params.a
    Q[7, 1] = params.omega / params.b
    Q[7, 7] = -(Q[7, 6] + Q[7, 8] + Q[7, 1])

    Q[8, 7] = 2 * beta / params.b
    Q[8, 9] = 2 * alpha * params.a
    Q[8, 2] = params.omega / (params.b**2)
    Q[8, 8] = -(Q[8, 7] + Q[8, 9] + Q[8, 2])

    Q[9, 8] = 3 * beta / params.b
    Q[9, 10] = alpha * params.a
    Q[9, 3] = params.omega / (params.b**3)
    Q[9, 9] = -(Q[9, 8] + Q[9, 10] + Q[9, 3])

    Q[10, 9] = 4 * beta / params.b
    Q[10, 11] = params.f1
    Q[10, 4] = params.omega / (params.b**4)
    Q[10, 10] = -(Q[10, 9] + Q[10, 11] + Q[10, 4])

    Q[11, 10] = params.g1
    Q[11, 11] = -Q[11, 10]

    return Q
