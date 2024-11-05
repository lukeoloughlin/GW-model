import math

import numpy as np
import numpy.typing as npt
from numba import float32
from numba import cuda


@cuda.jit(device=True, inline=True)
def square(val: float) -> float:
    """Return val^2"""
    return val * val


@cuda.jit(device=True, inline=True)
def cube(val: float) -> float:
    """Return val^3"""
    return val * val * val


@cuda.jit(device=True, inline=True)
def pow4(val: float) -> float:
    """Return val^4"""
    return val * val * val * val


@cuda.jit(device=True, inline=True)
def calculate_rho(cjsr: float, K: float, rho_inf: float) -> float:
    """Calculate rho(cjsr)"""
    log_cjsr_K = math.log(cjsr) - math.log(K)
    return rho_inf / (float32(1.0) + math.exp(float32(23.0) * log_cjsr_K))


@cuda.jit(device=True, inline=True)
def calculate_Mhat(rho: float, BCSQN: float) -> float:
    """Calculate Mhat from rho"""
    return (math.sqrt(float32(1.0) + float32(8.0) * rho * BCSQN) - float32(1.0)) / (
        float32(4.0) * rho * BCSQN
    )


@cuda.jit(device=True, inline=True)
def get_boundary_val(arr: npt.NDArray, x: int, y: int, Nx: int, Ny: int):
    """Get values on boundaries from flattened array arr. Assumes arr is organised in the order top, bottom, left, right"""
    if x == 0:
        return arr[y]
    elif x == (Nx - 1):
        return arr[y + Ny]
    elif y == 0:
        return arr[2 * Ny + x - 1]
    elif y == (Ny - 1):
        return arr[2 * Ny + Nx + x - 3]
    else:
        return float32(0.0)


@cuda.jit(device=True, inline=True)
def bubble_sort_ryr(RyR, RyR_sorted, x, y):
    RyR_sorted[x, y, 0] = RyR[x, y, 0]
    RyR_sorted[x, y, 1] = RyR[x, y, 1]
    RyR_sorted[x, y, 2] = RyR[x, y, 2]
    RyR_sorted[x, y, 3] = RyR[x, y, 3]

    swapped = False
    tmp = float32(0.0)
    for i in range(3):
        swapped = False
        for j in range(3 - i):
            if RyR_sorted[x, y, j] > RyR_sorted[x, y, j + 1]:
                tmp = RyR_sorted[x, y, j]
                RyR_sorted[x, y, j] = RyR_sorted[x, y, j + 1]
                RyR_sorted[x, y, j + 1] = tmp
                swapped = True

        if not swapped:
            break
