import math

import numpy as np
import numpy.typing as npt
from numba import float32
from numba import cuda

f32 = np.float32
i32 = np.int32


def constants_struct_array(dt: f32, sqrtdt: f32, Nai3: f32) -> npt.ArrayLike:
    """Create a structured array of constants"""
    values = (dt, sqrtdt, Nai3)
    names = ("dt", "sqrtdt", "Nai3")
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
def square(val: f32) -> f32:
    """Return val^2"""
    return val * val


@cuda.jit(device=True, inline=True)
def cube(val: f32) -> f32:
    """Return val^3"""
    return val * val * val


@cuda.jit(device=True, inline=True)
def pow4(val: f32) -> f32:
    """Return val^4"""
    return val * val * val * val


@cuda.jit(device=True, inline=True)
def calculate_rho(cjsr: f32, K: f32, rho_inf: f32, h: f32) -> f32:
    """Calculate rho(cjsr)"""
    K_cjsr_h = math.pow(K / cjsr, h)
    return rho_inf / (float32(1.0) + K_cjsr_h)


@cuda.jit(device=True, inline=True)
def calculate_Mhat(rho: f32, BCSQN: f32) -> f32:
    """Calculate Mhat from rho"""
    rhoBCSQN = rho * BCSQN
    if rhoBCSQN < float32(1e-10):
        # Use a first order Taylor approximation for small rho, otherwise numerical instability becomes an issue
        return float32(1.0) - float32(2.0) * rhoBCSQN
    return (math.sqrt(float32(1.0) + float32(8.0) * rhoBCSQN) - float32(1.0)) / (
        float32(4.0) * rhoBCSQN
    )


@cuda.jit(device=True, inline=True)
def boundary_from_flattened(
    arr: npt.NDArray[f32], x: i32, y: i32, Nx: i32, Ny: i32
) -> f32:
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
def flattened_from_boundary(arr: npt.NDArray[f32], Nx: i32, Ny: i32, idx: i32) -> f32:
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
def bubble_sort_ryr(
    RyR: npt.NDArray[f32], RyR_sorted: npt.NDArray[f32], x: i32, y: i32
) -> None:
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


@cuda.jit(device=True, inline=True)
def bubble_sort_ryr_3d(
    RyR: npt.NDArray[f32],
    RyR_sorted: npt.NDArray[f32],
    x: i32,
    y: i32,
    z: i32,
) -> None:
    """Sort RyR values and store in RyR sorted using bubble sort."""
    RyR_sorted[x, y, z, 0] = RyR[x, y, z, 0]
    RyR_sorted[x, y, z, 1] = RyR[x, y, z, 1]
    RyR_sorted[x, y, z, 2] = RyR[x, y, z, 2]
    RyR_sorted[x, y, z, 3] = RyR[x, y, z, 3]

    swapped = False
    tmp = float32(0.0)
    for i in range(3):
        swapped = False
        for j in range(3 - i):
            if RyR_sorted[x, y, z, j] > RyR_sorted[x, y, z, j + 1]:
                tmp = RyR_sorted[x, y, z, j]
                RyR_sorted[x, y, z, j] = RyR_sorted[x, y, z, j + 1]
                RyR_sorted[x, y, z, j + 1] = tmp
                swapped = True

        if not swapped:
            break


@cuda.jit(device=True, inline=True)
def time_const_up(tau_x: f32, tau_x_bdy: f32, x: i32, y: i32, Nx: i32, Ny: i32) -> f32:
    """Get the correct diffusion time constant for the above neighbours"""
    if x == 0:
        return float32(0.0)
    elif x == 1 or x == (Nx - 1) or y == 0 or y == (Ny - 1):
        return float32(1.0) / tau_x_bdy
    else:
        return float32(1.0) / tau_x


@cuda.jit(device=True, inline=True)
def time_const_down(
    tau_x: f32, tau_x_bdy: f32, x: i32, y: i32, Nx: i32, Ny: i32
) -> f32:
    """Get the correct diffusion time constant for the below neighbours"""
    if x == (Nx - 1):
        return float32(0.0)
    elif x == 0 or x == (Nx - 2) or y == 0 or y == (Ny - 1):
        return float32(1.0) / tau_x_bdy
    else:
        return float32(1.0) / tau_x


@cuda.jit(device=True, inline=True)
def time_const_left(
    tau_y: f32, tau_y_bdy: f32, x: i32, y: i32, Nx: i32, Ny: i32
) -> f32:
    """Get the correct diffusion time constant for the below neighbours"""
    if y == 0:
        return float32(0.0)
    elif y == 1 or y == (Ny - 1) or x == 0 or x == (Nx - 1):
        return float32(1.0) / tau_y_bdy
    else:
        return float32(1.0) / tau_y


@cuda.jit(device=True, inline=True)
def time_const_right(
    tau_y: f32, tau_y_bdy: f32, x: i32, y: i32, Ny: i32, Nx: i32
) -> f32:
    """Get the correct diffusion time constant for the below neighbours"""
    if y == (Ny - 1):
        return float32(0.0)
    elif y == 0 or y == (Ny - 2) or x == 0 or x == (Nx - 1):
        return float32(1.0) / tau_y_bdy
    else:
        return float32(1.0) / tau_y


@cuda.jit(device=True, inline=True)
def time_const_forward_3d(
    tau_x: f32,
    tau_x_bdy: f32,
    x: i32,
    y: i32,
    z: i32,
    Nx: i32,
    junctional: npt.NDArray[np.bool_],
) -> f32:
    if x == (Nx - 1):
        return float32(0.0)
    elif junctional[x, y, z] or junctional[x + 1, y, z]:
        return float32(1.0) / tau_x_bdy
    else:
        return float32(1.0) / tau_x


@cuda.jit(device=True, inline=True)
def time_const_backward_3d(
    tau_x: f32,
    tau_x_bdy: f32,
    x: i32,
    y: i32,
    z: i32,
    Nx: i32,
    junctional: npt.NDArray[np.bool_],
) -> f32:
    if x == 0:
        return float32(0.0)
    elif junctional[x, y, z] or junctional[x - 1, y, z]:
        return float32(1.0) / tau_x_bdy
    else:
        return float32(1.0) / tau_x


@cuda.jit(device=True, inline=True)
def time_const_left_3d(
    tau_y: f32,
    tau_y_bdy: f32,
    x: i32,
    y: i32,
    z: i32,
    Ny: i32,
    junctional: npt.NDArray[np.bool_],
) -> f32:
    if y == (Ny - 1):
        return float32(0.0)
    elif junctional[x, y, z] or junctional[x, y + 1, z]:
        return float32(1.0) / tau_y_bdy
    else:
        return float32(1.0) / tau_y


@cuda.jit(device=True, inline=True)
def time_const_right_3d(
    tau_y: f32,
    tau_y_bdy: f32,
    x: i32,
    y: i32,
    z: i32,
    Ny: i32,
    junctional: npt.NDArray[np.bool_],
) -> f32:
    if y == 0:
        return float32(0.0)
    elif junctional[x, y, z] or junctional[x, y - 1, z]:
        return float32(1.0) / tau_y_bdy
    else:
        return float32(1.0) / tau_y


@cuda.jit(device=True, inline=True)
def time_const_up_3d(
    tau_z: f32,
    tau_z_bdy: f32,
    x: i32,
    y: i32,
    z: i32,
    Nz: i32,
    junctional: npt.NDArray[np.bool_],
) -> f32:
    if z == (Nz - 1):
        return float32(0.0)
    elif junctional[x, y, z] or junctional[x, y, z + 1]:
        return float32(1.0) / tau_z_bdy
    else:
        return float32(1.0) / tau_z


@cuda.jit(device=True, inline=True)
def time_const_down_3d(
    tau_z: f32,
    tau_z_bdy: f32,
    x: i32,
    y: i32,
    z: i32,
    Nz: i32,
    junctional: npt.NDArray[np.bool_],
) -> f32:
    if z == 0:
        return float32(0.0)
    elif junctional[x, y, z] or junctional[x, y, z - 1]:
        return float32(1.0) / tau_z_bdy
    else:
        return float32(1.0) / tau_z


def truncated_normal(
    std: float, lower: float, upper: float, Nx: int, Ny: int
) -> npt.NDArray:
    out = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            while True:
                sample = std * np.random.normal()
                if sample > lower and sample < upper:
                    out[i, j] = sample
                    break

    return out
