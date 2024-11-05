import math

import numpy as np
import numpy.typing as npt
from numba import float32
from numba import cuda

from utils import square, calculate_rho, calculate_Mhat


@cuda.jit(device=True, inline=True)
def ITCa(c: float, CaT: float, kon: float, koff: float, BT: float) -> float:
    return kon * c * (BT - CaT) - koff * CaT


@cuda.jit(device=True, inline=True)
def Ileak(cjsr: float, cnsr: float, ci: float, gleak: float, Kjsr2: float) -> float:
    cjsr2 = square(cjsr)
    return gleak * cjsr2 / (cjsr2 + Kjsr2) * (cnsr - ci)


@cuda.jit(device=True, inline=True)
def Iup(ci: float, cnsr: float, Ki: float, Knsr: float, vup: float) -> float:
    ci_term = math.pow(ci / Ki, float32(1.787))
    cnsr_term = math.pow(cnsr / Knsr, float32(1.787))

    return vup * (ci_term - cnsr_term) / (float32(1.0) + ci_term + cnsr_term)


@cuda.jit(device=True, inline=True)
def Ir(cp: float, cjsr: float, RyR_open: float, Jmax: float, vp: float) -> float:
    return Jmax * RyR_open * (cjsr - cp) / vp


@cuda.jit(device=True, inline=True)
def luminal_buffer(
    cjsr: float, rho_inf: float, K: float, BCSQN: float, nM: float, nD: float, KC: float
):
    """Calculate the luminal buffering term"""

    rho = calculate_rho(cjsr, K, rho_inf)
    Mhat = calculate_Mhat(rho, BCSQN)

    ncjsr = Mhat * nM + (float32(1.0) - Mhat) * nD

    dlog_rho = float32(23.0) * (float32(1.0) - rho / rho_inf) / cjsr
    dMhat = (
        -(float32(0.25) / BCSQN)
        * dlog_rho
        * (
            float32(1.0)
            + (float32(1.0) + float32(4.0) * rho * BCSQN)
            / math.sqrt(float32(1.0) + float32(8.0) * rho * BCSQN)
        )
        / rho
    )

    dn = dMhat * (nM - nD)

    return float32(1.0) / (
        float32(1.0)
        + (KC * BCSQN * ncjsr + dn * (cjsr * KC + square(cjsr))) / (square(KC + cjsr))
    )


@cuda.jit(device=True, inline=True)
def update_ICa(
    ICa: npt.NDArray,
    LCC: npt.NDArray,
    cp: float,
    PCa: float,
    z: float,
    gamma: float,
    Cao: float,
    idx: int,
):
    """Update array of ICa values. This is only calclated along the boundaries in a flattened 1d array"""
    F = float32(96.5)
    exp2z = math.exp(float32(2.0) * z)
    NLCC = float32(0.0)
    for i in range(4):
        if LCC[idx, i, 6] == 1:
            NLCC += float32(1.0)
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


@cuda.jit(device=True, inline=True)
def update_INaCa(
    INaCa: npt.NDArray,
    cs: float,
    vNaCa: float,
    z: float,
    eta: float,
    Nai3: float,
    Cao: float,
    Nao3: float,
    ksat: float,
    KmNao3: float,  # KmNao^3
    KmCao: float,
    KmCai: float,
    Kda: float,
    t1: float,
    idx: int,
):
    Ka = float32(1.0) / (float32(1.0) + (Kda / cs) * (Kda / cs) * (Kda / cs))
    t2 = KmNao3 * cs * (float32(1.0) + (cs / KmCai))
    t3 = KmCao * Nai3 + Nai3 * Cao + Nao3 * cs
    exp_etaz = math.exp(eta * z)
    exp_etam1z = math.exp((eta - float32(1.0)) * z)

    INaCa[idx] = (
        Ka
        * vNaCa
        * (exp_etaz * Nai3 * Cao - exp_etam1z * Nao3 * cs)
        / ((t1 + t2 + t3) * (float32(1.0) + ksat * exp_etam1z))
    )


@cuda.jit(device=True, inline=True)
def update_diffusive_fluxes(
    Delta_ci: npt.NDArray,
    Delta_cnsr: npt.NDArray,
    sum_cs_nn: npt.NDArray,
    ci: npt.NDArray,
    cnsr: npt.NDArray,
    cs: npt.NDArray,
    tau_iT: float,
    tau_iL: float,
    tau_nsrT: float,
    tau_nsrL: float,
    tau_sL: float,
    tau_sT: float,
    x: int,
    y: int,
) -> None:
    if x == 0 and y == 0:
        Delta_ci[x, y] = (ci[x + 1, y] - ci[x, y]) / tau_iL + (
            ci[x, y + 1] - ci[x, y] / tau_iT
        )
        Delta_cnsr[x, y] = (cnsr[x + 1, y] - cnsr[x, y]) / tau_nsrL + (
            cnsr[x, y + 1] - cnsr[x, y] / tau_nsrT
        )
        sum_cs_nn[x, y] = cs[x + 1, y] / tau_sL + cs[x, y + 1] / tau_sT
    elif x == 0 and y == (ci.shape[1] - 1):
        Delta_ci[x, y] = (ci[x + 1, y] - ci[x, y]) / tau_iL + (
            ci[x, y - 1] - ci[x, y]
        ) / tau_iT
        Delta_cnsr[x, y] = (cnsr[x + 1, y] - cnsr[x, y]) / tau_nsrL + (
            cnsr[x, y - 1] - cnsr[x, y]
        ) / tau_iT
        sum_cs_nn[x, y] = cs[x + 1, y] / tau_sL + cs[x, y - 1] / tau_sT
    elif x == (ci.shape[0] - 1) and y == 0:
        Delta_ci[x, y] = (ci[x - 1, y] - ci[x, y]) / tau_iL + (
            ci[x, y + 1] - ci[x, y]
        ) / tau_iT
        Delta_cnsr[x, y] = (cnsr[x - 1, y] - cnsr[x, y]) / tau_nsrL + (
            cnsr[x, y + 1] - cnsr[x, y]
        ) / tau_nsrT
        sum_cs_nn[x, y] = cs[x - 1, y] / tau_sL + cs[x, y + 1] / tau_sT
    elif x == (ci.shape[0] - 1) and y == (ci.shape[1] - 1):
        Delta_ci[x, y] = (ci[x - 1, y] - ci[x, y]) / tau_iL + (
            ci[x, y - 1] - ci[x, y]
        ) / tau_iT
        Delta_cnsr[x, y] = (cnsr[x - 1, y] - cnsr[x, y]) / tau_nsrL + (
            cnsr[x, y - 1] - cnsr[x, y]
        ) / tau_nsrT
        sum_cs_nn[x, y] = cs[x - 1, y] / tau_sL + cs[x, y - 1] / tau_sT
    elif x == 0:
        Delta_ci[x, y] = (ci[x + 1, y] - ci[x, y]) / tau_iL + (
            ci[x, y - 1] + ci[x, y + 1] - float32(2.0) * ci[x, y]
        ) / tau_iT
        Delta_cnsr[x, y] = (cnsr[x + 1, y] - cnsr[x, y]) / tau_nsrL + (
            cnsr[x, y - 1] + cnsr[x, y + 1] - float32(2.0) * cnsr[x, y]
        ) / tau_nsrT
        sum_cs_nn[x, y] = cs[x + 1, y] / tau_sL + (cs[x, y - 1] + cs[x, y + 1]) / tau_sT
    elif x == (ci.shape[0] - 1):
        Delta_ci[x, y] = (ci[x - 1, y] - ci[x, y]) / tau_iL + (
            ci[x, y - 1] + ci[x, y + 1] - float32(2.0) * ci[x, y]
        ) / tau_iT
        Delta_cnsr[x, y] = (cnsr[x - 1, y] - cnsr[x, y]) / tau_nsrL + (
            cnsr[x, y - 1] + cnsr[x, y + 1] - float32(2.0) * cnsr[x, y]
        ) / tau_nsrT
        sum_cs_nn[x, y] = cs[x - 1, y] / tau_sL + (cs[x, y - 1] + cs[x, y + 1]) / tau_sT
    elif y == 0:
        Delta_ci[x, y] = (
            ci[x - 1, y] + ci[x + 1, y] - float32(2.0) * ci[x, y]
        ) / tau_iL + (ci[x, y + 1] - ci[x, y]) / tau_iT
        Delta_cnsr[x, y] = (
            cnsr[x - 1, y] + cnsr[x + 1, y] - float32(2.0) * cnsr[x, y]
        ) / tau_nsrL + (cnsr[x, y + 1] - cnsr[x, y]) / tau_nsrT
        sum_cs_nn[x, y] = (cs[x - 1, y] + cs[x + 1, y]) / tau_sL + cs[x, y + 1] / tau_sT
    elif y == (ci.shape[1] - 1):
        Delta_ci[x, y] = (
            ci[x - 1, y] + ci[x + 1, y] - float32(2.0) * ci[x, y]
        ) / tau_iL + (ci[x, y - 1] - ci[x, y]) / tau_iT
        Delta_cnsr[x, y] = (
            cnsr[x - 1, y] + cnsr[x + 1, y] - float32(2.0) * cnsr[x, y]
        ) / tau_nsrL + (cnsr[x, y - 1] - cnsr[x, y]) / tau_nsrT
        sum_cs_nn[x, y] = (cs[x - 1, y] + cs[x + 1, y]) / tau_sL + cs[x, y - 1] / tau_sT
    else:
        Delta_ci[x, y] = (
            ci[x - 1, y] + ci[x + 1, y] - float32(2.0) * ci[x, y]
        ) / tau_iL + (ci[x, y + 1] + ci[x, y - 1] - float32(2.0) * ci[x, y]) / tau_iT
        Delta_cnsr[x, y] = (
            cnsr[x - 1, y] + cnsr[x + 1, y] - float32(2.0) * cnsr[x, y]
        ) / tau_nsrL + (
            cnsr[x, y + 1] + cnsr[x, y - 1] - float32(2.0) * cnsr[x, y]
        ) / tau_nsrT
        sum_cs_nn[x, y] = (cs[x - 1, y] + cs[x + 1, y]) / tau_sL + (
            cs[x, y + 1] + cs[x, y - 1]
        ) / tau_sT


@cuda.jit(device=True, inline=True)
def cp_cs_iter(
    cp: npt.NDArray,
    cs: npt.NDArray,
    cp_prev: float,
    cs_prev: float,
    cjsr: float,
    ci: float,
    ryr_open: float,
    ICa: float,
    INaCa: float,
    ITCs: float,
    sum_cs_nn: float,
    vp: float,
    vs: float,
    tau_p: float,
    tau_si: float,
    tau_sT: float,
    tau_sL: float,
    Jmax: float,
    x: int,
    y: int,
    Nx: int,
    Ny: int,
) -> None:
    """Update cp and cs under rapid equilibrium approximation."""
    nn_rate = float32(0.0)
    # Get the sum nearest neighbours term, accounting for boundaries
    if (
        (x == 0 and y == 0)
        or (x == 0 and y == Ny - 1)
        or (x == Nx - 1 and y == 0)
        or (x == Nx - 1 and y == Ny - 1)
    ):
        nn_rate = float32(1.0) / tau_sL + float32(1.0) / tau_sT
    elif x == 0 or x == Nx - 1:
        nn_rate = float32(1.0) / tau_sL + float32(2.0) / tau_sT
    elif y == 0 or y == Ny - 1:
        nn_rate = float32(2.0) / tau_sL + float32(1.0) / tau_sT
    else:
        nn_rate = float32(2.0) / tau_sL + float32(2.0) / tau_sT

    kr = Jmax / vp
    cp[x, y] = (cs_prev + tau_p * (kr * cjsr - ICa)) / (
        float32(1.0) + tau_p * kr * ryr_open
    )

    kp = vp / (vs * tau_p)

    cs[x, y] = (cp_prev * kp + INaCa + ci / tau_si - ITCs + sum_cs_nn) / (
        kp + float32(1.0) / tau_si + nn_rate
    )
