import math

import numpy as np
import numpy.typing as npt
from numba import float32
from numba import cuda


def constants_struct_array(dt: float, sqrtdt: float) -> npt.ArrayLike:
    """Create a structured array of constants"""
    values = (dt, sqrtdt)
    names = ("dt", "sqrtdt")
    offsets = 4 * np.arange(len(values))
    dtype = np.dtype(
        (
            np.record,
            dict(
                names=names,
                formats=[np.dtype("float32")] * len(names),
                offsets=offsets,
                itemsize=offsets[-1] + 4,
            ),
        ),
        align=True,
    )
    return np.rec.array(values, dtype=dtype, aligned=True)[None]


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
def calculate_rho(cjsr: float, K: float, rho_inf: float, h: float) -> float:
    """Calculate rho(cjsr)"""
    K_cjsr_h = math.pow(K / cjsr, h)
    return rho_inf / (float32(1.0) + K_cjsr_h)


@cuda.jit(device=True, inline=True)
def calculate_Mhat(rho: float, BCSQN: float) -> float:
    """Calculate Mhat from rho"""
    rhoBCSQN = rho * BCSQN
    if rhoBCSQN < float32(1e-10):
        # Use a first order Taylor approximation for small rho, otherwise numerical instability becomes an issue
        return float32(1.0) - float32(2.0) * rhoBCSQN
    return (math.sqrt(float32(1.0) + float32(8.0) * rhoBCSQN) - float32(1.0)) / (
        float32(4.0) * rhoBCSQN
    )


@cuda.jit(device=True, inline=True)
def boundary_from_flattened(arr: npt.NDArray, x: int, y: int, Nx: int, Ny: int):
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
def flattened_from_boundary(arr: npt.NDArray, Nx: int, Ny: int, idx: int):
    """Convert idx to appropriate 2d index on boundary of arr and return the corresponding value of arr
    For now I will assume that the values are arranged according to top, bottom, left, right
    """
    if idx < Ny:
        return arr[0, idx]
    elif idx < 2 * Ny:
        return arr[Nx - 1, idx - Ny]
    elif idx < 2 * Ny + Nx - 2:
        return arr[idx - 2 * Ny + 1, 0]
    else:
        return arr[idx - (2 * Ny + Nx - 2) + 1, Nx - 1]


@cuda.jit(device=True, inline=True)
def bubble_sort_ryr(RyR: npt.NDArray, RyR_sorted: npt.NDArray, x: int, y: int):
    """Sort RyR values and store in RyR sorted using bubble sort."""
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
