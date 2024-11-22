import math

import numpy as np
import numpy.typing as npt
from numba import float32
from numba import cuda

from params import RestrepoParams
from .utils import *

# f32 = np.float32
# i32 = np.int32


@cuda.jit(device=True, inline=True)
def ITCa(c: f32, CaT: f32, kon: f32, koff: f32, BT: f32) -> f32:
    return kon * c * (BT - CaT) - koff * CaT


@cuda.jit(device=True, inline=True)
def Ileak(cnsr: f32, ci: f32, gleak: f32, Knsr2: f32) -> f32:
    cnsr2 = square(cnsr)
    return gleak * (cnsr2 / (cnsr2 + Knsr2)) * (cnsr - ci)


@cuda.jit(device=True, inline=True)
def Iup(ci: f32, cnsr: f32, Ki: f32, Knsr: f32, vup: f32, H: f32) -> f32:
    ci_term = math.pow(ci / Ki, H)
    cnsr_term = math.pow(cnsr / Knsr, H)

    return vup * (ci_term - cnsr_term) / (float32(1.0) + ci_term + cnsr_term)


@cuda.jit(device=True, inline=True)
def Ir(cp: f32, cjsr: f32, ryr_open: f32, Jmax: f32, vp: f32) -> f32:
    return Jmax * ryr_open * (cjsr - cp) / vp


@cuda.jit(device=True, inline=True)
def luminal_buffer(
    cjsr: f32,
    rho_inf: f32,
    K: f32,
    BCSQN: f32,
    nM: f32,
    nD: f32,
    KC: f32,
    h: f32,
) -> f32:
    """Calculate the luminal buffering term"""

    rho = calculate_rho(cjsr, K, rho_inf, h)
    Mhat = calculate_Mhat(rho, BCSQN)

    ncjsr = Mhat * nM + (float32(1.0) - Mhat) * nD

    drho = (h * rho / cjsr) * (float32(1.0) - rho / rho_inf)
    dMhat = (
        (-float32(2.0) * BCSQN * square(Mhat))
        / (float32(1.0) + float32(4.0) * Mhat * rho * BCSQN)
    ) * drho
    dn = dMhat * (nM - nD)

    return float32(1.0) / (
        float32(1.0)
        + BCSQN * (KC * ncjsr + dn * (cjsr * KC + square(cjsr))) / (square(KC + cjsr))
    )


@cuda.jit(device=True, inline=True)
def update_ICa(
    ICa: npt.NDArray[f32],
    LCC: npt.NDArray[f32],
    cp: f32,
    PCa: f32,
    z: f32,
    gamma: f32,
    Cao: f32,
    idx: i32,
) -> None:
    """Update array of ICa values. This is only calclated along the boundaries in a flattened 1d array"""
    F = float32(96.5)
    exp2z = math.exp(float32(2.0) * z)
    NLCC = float32(0.0)
    for i in range(4):
        if LCC[idx, i] == 7:
            NLCC += float32(1.0)
    if math.fabs(z) > float32(0.01):
        ICa[idx] = (
            NLCC
            * float32(4.0)
            * PCa
            * z
            * F
            * gamma
            * (cp * float32(1e-3) * exp2z - Cao)
            / (exp2z - float32(1.0))
        )
    else:
        # Approximate z/(exp(2z)-1) by 1/(2+2z) for small z
        ICa[idx] = (
            NLCC
            * float32(2.0)
            * PCa
            * F
            * gamma
            * (cp * float32(1e-3) * exp2z - Cao)
            / (float32(1.0) + z)
        )


@cuda.jit(device=True, inline=True)
def update_ICa_3d(
    ICa: npt.NDArray[f32],
    LCC: npt.NDArray[f32],
    cp: f32,
    PCa: f32,
    VF_RT: f32,
    gamma: f32,
    Cao: f32,
    junctional: bool,
    x: i32,
    y: i32,
    z: i32,
) -> None:
    """Update array of ICa values. This is only calclated along the boundaries in a flattened 1d array"""
    F = float32(96.5)
    exp2VF_RT = math.exp(float32(2.0) * VF_RT)
    NLCC = float32(0.0)
    for k in range(4):
        if LCC[x, y, z, k] == 7:
            NLCC += float32(1.0)
    if math.fabs(VF_RT) > float32(0.01):
        ICa[x, y, z] = (
            (
                NLCC
                * float32(4.0)
                * PCa
                * VF_RT
                * F
                * gamma
                * (cp * float32(1e-3) * exp2VF_RT - Cao)
                / (exp2VF_RT - float32(1.0))
            )
            if junctional
            else float32(0.0)
        )
    else:
        # Approximate z/(exp(2z)-1) by 1/(2+2z) for small z
        ICa[x, y, z] = (
            (
                NLCC
                * float32(2.0)
                * PCa
                * F
                * gamma
                * (cp * float32(1e-3) * exp2VF_RT - Cao)
                / (float32(1.0) + VF_RT)
            )
            if junctional
            else float32(0.0)
        )


@cuda.jit(device=True, inline=True)
def update_INaCa(
    INaCa: npt.NDArray[f32],
    cs: f32,
    z: f32,
    Nai3: f32,
    params: RestrepoParams,
    idx: i32,
) -> None:
    # Careful with units here
    Nao3 = cube(params.Nao[0])
    KmNao3 = cube(params.KmNao[0])
    KmNai3 = cube(params.KmNai[0])
    Ka = float32(1.0) / (float32(1.0) + cube(params.Kda[0] / cs))

    cs_mM = cs * float32(1e-3)  # convert cs to mM
    KmCai_mM = params.KmCai[0] * float32(1e-3)  # convert KmCai to mM

    t1 = KmCai_mM * Nao3 * (float32(1.0) + Nai3 / KmNai3)
    t2 = KmNao3 * cs_mM * (float32(1.0) + (cs_mM / KmCai_mM))
    t3 = params.KmCao[0] * Nai3 + Nai3 * params.Cao[0] + Nao3 * cs_mM
    exp_etaz = math.exp(params.eta[0] * z)
    exp_etam1z = math.exp((params.eta[0] - float32(1.0)) * z)

    INaCa[idx] = (
        Ka
        * params.vNaCa[0]
        * (exp_etaz * Nai3 * params.Cao[0] - exp_etam1z * Nao3 * cs_mM)
        / ((t1 + t2 + t3) * (float32(1.0) + params.ksat[0] * exp_etam1z))
    )


# Using the form given in Shiferaw et al (2018).
@cuda.jit(device=True, inline=True)
def update_INaCa_3d(
    INaCa: npt.NDArray[f32],
    cs: f32,
    VF_RT: f32,
    Nai3: f32,
    params: RestrepoParams,
    junctional: bool,
    x: i32,
    y: i32,
    z: i32,
) -> None:
    # Careful with units here
    Nao3 = cube(params.Nao[0])
    KmNao3 = cube(params.KmNao[0])
    KmNai3 = cube(params.KmNai[0])
    Ka = float32(1.0) / (float32(1.0) + cube(params.Kda[0] / cs))
    # ANaCa = float32(1.0) / (float32(1.0) + cube(params.cNaCa[0] / cs))

    cs_mM = cs * float32(1e-3)  # convert cs to mM
    # KmCai_mM = params.KmCai[0] * float32(1e-3)  # convert KmCai to mM

    t1 = params.KmCai[0] * Nao3 * (float32(1.0) + Nai3 / KmNai3)
    t2 = KmNao3 * cs_mM * (float32(1.0) + (cs_mM / params.KmCai[0]))
    t3 = params.KmCao[0] * Nai3 + Nai3 * params.Cao[0] + Nao3 * cs_mM
    # U = (
    #    params.KmCao[0] * Nai3
    #    + KmNao3 * cs_mM
    #    + KmNai3 * params.Cao[0] * (float32(1.0) + cs_mM / params.KmCai[0])
    #    + params.KmCai[0] * Nao3 * (float32(1.0) + Nai3 / KmNai3)
    #    + Nai3 * params.Cao[0]
    #    + Nao3 * cs_mM
    # )
    exp_etaz = math.exp(params.eta[0] * VF_RT)
    exp_etam1z = math.exp((params.eta[0] - float32(1.0)) * VF_RT)

    INaCa[x, y, z] = (
        (
            params.vNaCa[0]
            * Ka
            * (exp_etaz * Nai3 * params.Cao[0] - exp_etam1z * Nao3 * cs_mM)
            / ((t1 + t2 + t3) * (float32(1.0) + params.ksat[0] * exp_etam1z))
        )
        if junctional
        else float32(0.0)
    )


@cuda.jit(device=True, inline=True)
def update_diffusive_fluxes(
    Delta_ci: npt.NDArray[f32],
    Delta_cnsr: npt.NDArray[f32],
    Delta_cs: npt.NDArray[f32],
    ci: npt.NDArray[f32],
    cnsr: npt.NDArray[f32],
    cs: npt.NDArray[f32],
    params: RestrepoParams,
    x: i32,
    y: i32,
) -> None:
    Nx, Ny = ci.shape
    # Calculate the inverse time constants here to avoid dealing with all the checks at the boundaries
    _1_tau_i_l = time_const_left(params.tau_iT_i[0], params.tau_iT_p[0], x, y, Nx, Ny)
    _1_tau_i_r = time_const_right(params.tau_iT_i[0], params.tau_iT_p[0], x, y, Nx, Ny)
    _1_tau_i_u = time_const_up(params.tau_iL_i[0], params.tau_iL_p[0], x, y, Nx, Ny)
    _1_tau_i_d = time_const_down(params.tau_iL_i[0], params.tau_iL_p[0], x, y, Nx, Ny)

    _1_tau_s_l = time_const_left(params.tau_sT_i[0], params.tau_sT_p[0], x, y, Nx, Ny)
    _1_tau_s_r = time_const_right(params.tau_sT_i[0], params.tau_sT_p[0], x, y, Nx, Ny)
    _1_tau_s_u = time_const_up(params.tau_sL_i[0], params.tau_sL_p[0], x, y, Nx, Ny)
    _1_tau_s_d = time_const_down(params.tau_sL_i[0], params.tau_sL_p[0], x, y, Nx, Ny)

    _1_tau_nsr_l = time_const_left(
        params.tau_nsrT_i[0], params.tau_nsrT_p[0], x, y, Nx, Ny
    )
    _1_tau_nsr_r = time_const_right(
        params.tau_nsrT_i[0], params.tau_nsrT_p[0], x, y, Nx, Ny
    )
    _1_tau_nsr_u = time_const_up(
        params.tau_nsrL_i[0], params.tau_nsrL_p[0], x, y, Nx, Ny
    )
    _1_tau_nsr_d = time_const_down(
        params.tau_nsrL_i[0], params.tau_nsrL_p[0], x, y, Nx, Ny
    )

    above = x - 1 if x > 0 else x
    below = x + 1 if x < (Nx - 1) else x
    left = y - 1 if y > 0 else y
    right = y + 1 if y < (Ny - 1) else y

    Delta_ci[x, y] = (
        _1_tau_i_l * (ci[x, left] - ci[x, y])
        + _1_tau_i_r * (ci[x, right] - ci[x, y])
        + _1_tau_i_u * (ci[above, y] - ci[x, y])
        + _1_tau_i_d * (ci[below, y] - ci[x, y])
    )

    Delta_cnsr[x, y] = (
        _1_tau_nsr_l * (cnsr[x, left] - cnsr[x, y])
        + _1_tau_nsr_r * (cnsr[x, right] - cnsr[x, y])
        + _1_tau_nsr_u * (cnsr[above, y] - cnsr[x, y])
        + _1_tau_nsr_d * (cnsr[below, y] - cnsr[x, y])
    )

    Delta_cs[x, y] = (
        _1_tau_s_l * (cs[x, left] - cs[x, y])
        + _1_tau_s_r * (cs[x, right] - cs[x, y])
        + _1_tau_s_u * (cs[above, y] - cs[x, y])
        + _1_tau_s_d * (cs[below, y] - cs[x, y])
    )


@cuda.jit(device=True, inline=True)
def update_diffusive_fluxes_3d(
    Delta_ci: npt.NDArray[f32],
    Delta_cnsr: npt.NDArray[f32],
    Delta_cs: npt.NDArray[f32],
    ci: npt.NDArray[f32],
    cnsr: npt.NDArray[f32],
    cs: npt.NDArray[f32],
    junctional: npt.NDArray[np.bool_],
    params: RestrepoParams,
    x: i32,
    y: i32,
    z: i32,
) -> None:
    Nx, Ny, Nz = ci.shape
    # Calculate the inverse time constants here to avoid dealing with all the checks at the boundaries
    _1_tau_i_f = time_const_forward_3d(
        params.tau_iL_i[0], params.tau_iL_p[0], x, y, z, Nx, junctional
    )
    _1_tau_i_b = time_const_backward_3d(
        params.tau_iL_i[0], params.tau_iL_p[0], x, y, z, Nx, junctional
    )
    _1_tau_i_l = time_const_left_3d(
        params.tau_iT_i[0], params.tau_iT_p[0], x, y, z, Ny, junctional
    )
    _1_tau_i_r = time_const_right_3d(
        params.tau_iT_i[0], params.tau_iT_p[0], x, y, z, Ny, junctional
    )
    _1_tau_i_u = time_const_up_3d(
        params.tau_iT_i[0], params.tau_iT_p[0], x, y, z, Nz, junctional
    )
    _1_tau_i_d = time_const_down_3d(
        params.tau_iT_i[0], params.tau_iT_p[0], x, y, z, Nz, junctional
    )

    _1_tau_s_f = time_const_forward_3d(
        params.tau_sL_i[0], params.tau_sL_p[0], x, y, z, Nx, junctional
    )
    _1_tau_s_b = time_const_backward_3d(
        params.tau_sL_i[0], params.tau_sL_p[0], x, y, z, Nx, junctional
    )
    _1_tau_s_l = time_const_left_3d(
        params.tau_sT_i[0], params.tau_sT_p[0], x, y, z, Ny, junctional
    )
    _1_tau_s_r = time_const_right_3d(
        params.tau_sT_i[0], params.tau_sT_p[0], x, y, z, Ny, junctional
    )
    _1_tau_s_u = time_const_up_3d(
        params.tau_sT_i[0], params.tau_sT_p[0], x, y, z, Nz, junctional
    )
    _1_tau_s_d = time_const_down_3d(
        params.tau_sT_i[0], params.tau_sT_p[0], x, y, z, Nz, junctional
    )

    _1_tau_nsr_f = time_const_forward_3d(
        params.tau_nsrL_i[0], params.tau_nsrL_p[0], x, y, z, Nx, junctional
    )
    _1_tau_nsr_b = time_const_backward_3d(
        params.tau_nsrL_i[0], params.tau_nsrL_p[0], x, y, z, Nx, junctional
    )
    _1_tau_nsr_l = time_const_left_3d(
        params.tau_nsrT_i[0], params.tau_nsrT_p[0], x, y, z, Ny, junctional
    )
    _1_tau_nsr_r = time_const_right_3d(
        params.tau_nsrT_i[0], params.tau_nsrT_p[0], x, y, z, Ny, junctional
    )
    _1_tau_nsr_u = time_const_up_3d(
        params.tau_nsrT_i[0], params.tau_nsrT_p[0], x, y, z, Nz, junctional
    )
    _1_tau_nsr_d = time_const_down_3d(
        params.tau_nsrT_i[0], params.tau_nsrT_p[0], x, y, z, Nz, junctional
    )

    backward = x - 1 if x > 0 else x
    forward = x + 1 if x < (Nx - 1) else x
    left = y - 1 if y > 0 else y
    right = y + 1 if y < (Ny - 1) else y
    down = z - 1 if z > 0 else z
    up = z + 1 if z < (Nz - 1) else z

    ci_centre = ci[x, y, z]
    cs_centre = cs[x, y, z]
    cnsr_centre = cnsr[x, y, z]

    Delta_ci[x, y, z] = (
        _1_tau_i_b * (ci[backward, y, z] - ci_centre)
        + _1_tau_i_f * (ci[forward, y, z] - ci_centre)
        + _1_tau_i_l * (ci[x, left, z] - ci_centre)
        + _1_tau_i_r * (ci[x, right, z] - ci_centre)
        + _1_tau_i_d * (ci[x, y, down] - ci_centre)
        + _1_tau_i_u * (ci[x, y, up] - ci_centre)
    )

    Delta_cs[x, y, z] = (
        _1_tau_s_b * (cs[backward, y, z] - cs_centre)
        + _1_tau_s_f * (cs[forward, y, z] - cs_centre)
        + _1_tau_s_l * (cs[x, left, z] - cs_centre)
        + _1_tau_s_r * (cs[x, right, z] - cs_centre)
        + _1_tau_s_d * (cs[x, y, down] - cs_centre)
        + _1_tau_s_u * (cs[x, y, up] - cs_centre)
    )

    Delta_cnsr[x, y, z] = (
        _1_tau_nsr_b * (cnsr[backward, y, z] - cnsr_centre)
        + _1_tau_nsr_f * (cnsr[forward, y, z] - cnsr_centre)
        + _1_tau_nsr_l * (cnsr[x, left, z] - cnsr_centre)
        + _1_tau_nsr_r * (cnsr[x, right, z] - cnsr_centre)
        + _1_tau_nsr_d * (cnsr[x, y, down] - cnsr_centre)
        + _1_tau_nsr_u * (cnsr[x, y, up] - cnsr_centre)
    )
