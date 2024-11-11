import math

import numpy as np
import numpy.typing as npt
from numba import float32
from numba import cuda

from params import RestrepoParams
from src.restrepo.cuda.utils import square, cube, calculate_rho, calculate_Mhat


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
        if LCC[idx, i] == 7:
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
    z: float,
    Nai3: float,
    params: RestrepoParams,
    idx: int,
):
    Nao3 = cube(params.Nao[0])
    KmNao3 = cube(params.KmNao[0])
    KmNai3 = cube(params.KmNai[0])
    Ka = float32(1.0) / (
        float32(1.0)
        + (params.Kda[0] / cs) * (params.Kda[0] / cs) * (params.Kda[0] / cs)
    )
    t1 = params.KmCai[0] * Nao3 * (float32(1.0) + Nai3 / KmNai3)
    t2 = KmNao3 * cs * (float32(1.0) + (cs / params.KmCai[0]))
    t3 = params.KmCao[0] * Nai3 + Nai3 * params.Cao[0] + Nao3 * cs
    exp_etaz = math.exp(params.eta[0] * z)
    exp_etam1z = math.exp((params.eta[0] - float32(1.0)) * z)

    INaCa[idx] = (
        Ka
        * params.vNaCa[0]
        * (exp_etaz * Nai3 * params.Cao[0] - exp_etam1z * Nao3 * cs)
        / ((t1 + t2 + t3) * (float32(1.0) + params.ksat[0] * exp_etam1z))
    )


@cuda.jit(device=True, inline=True)
def update_diffusive_fluxes(
    Delta_ci: npt.NDArray,
    Delta_cnsr: npt.NDArray,
    sum_cs_nn: npt.NDArray,
    ci: npt.NDArray,
    cnsr: npt.NDArray,
    cs: npt.NDArray,
    params: RestrepoParams,
    x: int,
    y: int,
) -> None:
    if x == 0 and y == 0:
        Delta_ci[x, y] = (ci[x + 1, y] - ci[x, y]) / params.tau_iL[0] + (
            ci[x, y + 1] - ci[x, y]
        ) / params.tau_iT[0]
        Delta_cnsr[x, y] = (cnsr[x + 1, y] - cnsr[x, y]) / params.tau_nsrL[0] + (
            cnsr[x, y + 1] - cnsr[x, y]
        ) / params.tau_nsrT[0]
        sum_cs_nn[x, y] = (
            cs[x + 1, y] / params.tau_sL[0] + cs[x, y + 1] / params.tau_sT[0]
        )
    elif x == 0 and y == (ci.shape[1] - 1):
        Delta_ci[x, y] = (ci[x + 1, y] - ci[x, y]) / params.tau_iL[0] + (
            ci[x, y - 1] - ci[x, y]
        ) / params.tau_iT[0]
        Delta_cnsr[x, y] = (cnsr[x + 1, y] - cnsr[x, y]) / params.tau_nsrL[0] + (
            cnsr[x, y - 1] - cnsr[x, y]
        ) / params.tau_nsrT[0]
        sum_cs_nn[x, y] = (
            cs[x + 1, y] / params.tau_sL[0] + cs[x, y - 1] / params.tau_sT[0]
        )
    elif x == (ci.shape[0] - 1) and y == 0:
        Delta_ci[x, y] = (ci[x - 1, y] - ci[x, y]) / params.tau_iL[0] + (
            ci[x, y + 1] - ci[x, y]
        ) / params.tau_iT[0]
        Delta_cnsr[x, y] = (cnsr[x - 1, y] - cnsr[x, y]) / params.tau_nsrL[0] + (
            cnsr[x, y + 1] - cnsr[x, y]
        ) / params.tau_nsrT[0]
        sum_cs_nn[x, y] = (
            cs[x - 1, y] / params.tau_sL[0] + cs[x, y + 1] / params.tau_sT[0]
        )
    elif x == (ci.shape[0] - 1) and y == (ci.shape[1] - 1):
        Delta_ci[x, y] = (ci[x - 1, y] - ci[x, y]) / params.tau_iL[0] + (
            ci[x, y - 1] - ci[x, y]
        ) / params.tau_iT[0]
        Delta_cnsr[x, y] = (cnsr[x - 1, y] - cnsr[x, y]) / params.tau_nsrL[0] + (
            cnsr[x, y - 1] - cnsr[x, y]
        ) / params.tau_nsrT[0]
        sum_cs_nn[x, y] = (
            cs[x - 1, y] / params.tau_sL[0] + cs[x, y - 1] / params.tau_sT[0]
        )
    elif x == 0:
        Delta_ci[x, y] = (ci[x + 1, y] - ci[x, y]) / params.tau_iL[0] + (
            ci[x, y - 1] + ci[x, y + 1] - float32(2.0) * ci[x, y]
        ) / params.tau_iT[0]
        Delta_cnsr[x, y] = (cnsr[x + 1, y] - cnsr[x, y]) / params.tau_nsrL[0] + (
            cnsr[x, y - 1] + cnsr[x, y + 1] - float32(2.0) * cnsr[x, y]
        ) / params.tau_nsrT[0]
        sum_cs_nn[x, y] = (
            cs[x + 1, y] / params.tau_sL[0]
            + (cs[x, y - 1] + cs[x, y + 1]) / params.tau_sT[0]
        )
    elif x == (ci.shape[0] - 1):
        Delta_ci[x, y] = (ci[x - 1, y] - ci[x, y]) / params.tau_iL[0] + (
            ci[x, y - 1] + ci[x, y + 1] - float32(2.0) * ci[x, y]
        ) / params.tau_iT[0]
        Delta_cnsr[x, y] = (cnsr[x - 1, y] - cnsr[x, y]) / params.tau_nsrL[0] + (
            cnsr[x, y - 1] + cnsr[x, y + 1] - float32(2.0) * cnsr[x, y]
        ) / params.tau_nsrT[0]
        sum_cs_nn[x, y] = (
            cs[x - 1, y] / params.tau_sL[0]
            + (cs[x, y - 1] + cs[x, y + 1]) / params.tau_sT[0]
        )
    elif y == 0:
        Delta_ci[x, y] = (
            ci[x - 1, y] + ci[x + 1, y] - float32(2.0) * ci[x, y]
        ) / params.tau_iL[0] + (ci[x, y + 1] - ci[x, y]) / params.tau_iT[0]
        Delta_cnsr[x, y] = (
            cnsr[x - 1, y] + cnsr[x + 1, y] - float32(2.0) * cnsr[x, y]
        ) / params.tau_nsrL[0] + (cnsr[x, y + 1] - cnsr[x, y]) / params.tau_nsrT[0]
        sum_cs_nn[x, y] = (cs[x - 1, y] + cs[x + 1, y]) / params.tau_sL[0] + cs[
            x, y + 1
        ] / params.tau_sT[0]
    elif y == (ci.shape[1] - 1):
        Delta_ci[x, y] = (
            ci[x - 1, y] + ci[x + 1, y] - float32(2.0) * ci[x, y]
        ) / params.tau_iL[0] + (ci[x, y - 1] - ci[x, y]) / params.tau_iT[0]
        Delta_cnsr[x, y] = (
            cnsr[x - 1, y] + cnsr[x + 1, y] - float32(2.0) * cnsr[x, y]
        ) / params.tau_nsrL[0] + (cnsr[x, y - 1] - cnsr[x, y]) / params.tau_nsrT[0]
        sum_cs_nn[x, y] = (cs[x - 1, y] + cs[x + 1, y]) / params.tau_sL[0] + cs[
            x, y - 1
        ] / params.tau_sT[0]
    else:
        Delta_ci[x, y] = (
            ci[x - 1, y] + ci[x + 1, y] - float32(2.0) * ci[x, y]
        ) / params.tau_iL[0] + (
            ci[x, y + 1] + ci[x, y - 1] - float32(2.0) * ci[x, y]
        ) / params.tau_iT[
            0
        ]
        Delta_cnsr[x, y] = (
            cnsr[x - 1, y] + cnsr[x + 1, y] - float32(2.0) * cnsr[x, y]
        ) / params.tau_nsrL[0] + (
            cnsr[x, y + 1] + cnsr[x, y - 1] - float32(2.0) * cnsr[x, y]
        ) / params.tau_nsrT[
            0
        ]
        sum_cs_nn[x, y] = (cs[x - 1, y] + cs[x + 1, y]) / params.tau_sL[0] + (
            cs[x, y + 1] + cs[x, y - 1]
        ) / params.tau_sT[0]


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

    kr = (Jmax / vp) * ryr_open
    cp[x, y] = (cs_prev + tau_p * (kr * cjsr - ICa)) / (float32(1.0) + tau_p * kr)

    kp = vp / (vs * tau_p)

    cs[x, y] = (cp_prev * kp + INaCa + ci / tau_si - ITCs + sum_cs_nn) / (
        kp + float32(1.0) / tau_si + nn_rate
    )
