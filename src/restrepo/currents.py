import math

import numpy as np
import numpy.typing as npt
from numba import float32
from numba import cuda


@cuda.jit(device=True, inline=True)
def get_ICa(ICa: npt.NDArray, x: int, y: int, Nx: int, Ny: int):
    """Get ICa values on boundaries from flattened array in the order top, bottom, left, right"""
    if x == 0:
        return ICa[y]
    elif x == (Nx - 1):
        return ICa[y + Ny]
    elif y == 0:
        return ICa[2 * Ny + x - 1]
    elif y == (Ny - 1):
        return ICa[2 * Ny + Nx + x - 3]
    else:
        return float32(0.0)


@cuda.jit(device=True, inline=True)
def get_INaCa(INaCa: npt.NDArray, x: int, y: int, Nx: int, Ny: int):
    """Get ICa values on boundaries from flattened array in the order top, bottom, left, right"""
    if y == 0:
        return INaCa[x]
    elif y == (Ny - 1):
        return INaCa[x + Nx]
    elif x == 0:
        return INaCa[2 * Nx + y]
    elif x == (Nx - 1):
        return INaCa[2 * Nx + Ny + y]
    else:
        return float32(0.0)


@cuda.jit(device=True, inline=True)
def luminal_buffer(
    cjsr: float, rho_inf: float, K: float, BCSQN: float, nM: float, nD: float, KC: float
):
    """Calculate the luminal buffering term"""
    logK_cjsr = math.log(cjsr / K)
    rho = rho_inf / (float32(1.0) + float32(23.0) * math.exp(logK_cjsr))
    sqrt_term = math.sqrt(float32(1.0) + float32(8.0) * rho * BCSQN)
    Mhat = (sqrt_term - float32(1.0)) / (float32(4.0) * rho * BCSQN)

    ncjsr = Mhat * nM + (float32(1.0) - Mhat) * nD

    dlog_rho = float32(23.0) * (float32(1.0) - rho / rho_inf) / cjsr
    dMhat = (
        -(float32(0.25) / BCSQN)
        * dlog_rho
        * (float32(1.0) + (float32(1.0) + float32(4.0) * rho * BCSQN) / sqrt_term)
        / rho
    )

    dn = dMhat * (nM - nD)

    return float32(1.0) / (
        float32(1.0)
        + (KC * BCSQN * ncjsr + dn * (cjsr * KC + cjsr * cjsr))
        / ((KC + cjsr) * (KC + cjsr))
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
def update_ITC(ITCi, ITCs, ci, cs, CaTi, CaTs, kon, koff, BT, x, y):
    ITCi[x, y] = kon * ci[x, y] * (BT - CaTi[x, y]) - koff * CaTi[x, y]
    ITCs[x, y] = kon * cs[x, y] * (BT - CaTs[x, y]) - koff * CaTs[x, y]


@cuda.jit(device=True, inline=True)
def update_diffusive_fluxes(
    Delta_ci, Delta_cnsr, ci, cnsr, tau_iT, tau_iL, tau_nsrT, tau_nsrL, x, y
):
    if x == 0 and y == 0:
        Delta_ci[x, y] = (ci[x + 1, y] - ci[x, y]) / tau_iL + (
            ci[x, y + 1] - ci[x, y] / tau_iT
        )
        Delta_cnsr[x, y] = (cnsr[x + 1, y] - cnsr[x, y]) / tau_nsrL + (
            cnsr[x, y + 1] - cnsr[x, y] / tau_nsrT
        )
    elif x == 0 and y == (ci.shape[1] - 1):
        Delta_ci[x, y] = (ci[x + 1, y] - ci[x, y]) / tau_iL + (
            ci[x, y - 1] - ci[x, y]
        ) / tau_iT
        Delta_cnsr[x, y] = (cnsr[x + 1, y] - cnsr[x, y]) / tau_nsrL + (
            cnsr[x, y - 1] - cnsr[x, y]
        ) / tau_iT
    elif x == (ci.shape[0] - 1) and y == 0:
        Delta_ci[x, y] = (ci[x - 1, y] - ci[x, y]) / tau_iL + (
            ci[x, y + 1] - ci[x, y]
        ) / tau_iT
        Delta_cnsr[x, y] = (cnsr[x - 1, y] - cnsr[x, y]) / tau_nsrL + (
            cnsr[x, y + 1] - cnsr[x, y]
        ) / tau_nsrT
    elif x == (ci.shape[0] - 1) and y == (ci.shape[1] - 1):
        Delta_ci[x, y] = (ci[x - 1, y] - ci[x, y]) / tau_iL + (
            ci[x, y - 1] - ci[x, y]
        ) / tau_iT
        Delta_cnsr[x, y] = (cnsr[x - 1, y] - cnsr[x, y]) / tau_nsrL + (
            cnsr[x, y - 1] - cnsr[x, y]
        ) / tau_nsrT
    elif x == 0:
        Delta_ci[x, y] = (ci[x + 1, y] - ci[x, y]) / tau_iL + (
            ci[x, y - 1] + ci[x, y + 1] - float32(2.0) * ci[x, y]
        ) / tau_iT
        Delta_cnsr[x, y] = (cnsr[x + 1, y] - cnsr[x, y]) / tau_nsrL + (
            cnsr[x, y - 1] + cnsr[x, y + 1] - float32(2.0) * cnsr[x, y]
        ) / tau_nsrT
    elif x == (ci.shape[0] - 1):
        Delta_ci[x, y] = (ci[x - 1, y] - ci[x, y]) / tau_iL + (
            ci[x, y - 1] + ci[x, y + 1] - float32(2.0) * ci[x, y]
        ) / tau_iT
        Delta_cnsr[x, y] = (cnsr[x - 1, y] - cnsr[x, y]) / tau_nsrL + (
            cnsr[x, y - 1] + cnsr[x, y + 1] - float32(2.0) * cnsr[x, y]
        ) / tau_nsrT
    elif y == 0:
        Delta_ci[x, y] = (
            ci[x - 1, y] + ci[x + 1, y] - float32(2.0) * ci[x, y]
        ) / tau_iL + (ci[x, y + 1] - ci[x, y]) / tau_iT
        Delta_cnsr[x, y] = (
            cnsr[x - 1, y] + cnsr[x + 1, y] - float32(2.0) * cnsr[x, y]
        ) / tau_nsrL + (cnsr[x, y + 1] - cnsr[x, y]) / tau_nsrT
    elif y == (ci.shape[1] - 1):
        Delta_ci[x, y] = (
            ci[x - 1, y] + ci[x + 1, y] - float32(2.0) * ci[x, y]
        ) / tau_iL + (ci[x, y - 1] - ci[x, y]) / tau_iT
        Delta_cnsr[x, y] = (
            cnsr[x - 1, y] + cnsr[x + 1, y] - float32(2.0) * cnsr[x, y]
        ) / tau_nsrL + (cnsr[x, y - 1] - cnsr[x, y]) / tau_nsrT
    else:
        Delta_ci[x, y] = (
            ci[x - 1, y] + ci[x + 1, y] - float32(2.0) * ci[x, y]
        ) / tau_iL + (ci[x, y + 1] + ci[x, y - 1] - float32(2.0) * ci[x, y]) / tau_iT
        Delta_cnsr[x, y] = (
            cnsr[x - 1, y] + cnsr[x + 1, y] - float32(2.0) * cnsr[x, y]
        ) / tau_nsrL + (
            cnsr[x, y + 1] + cnsr[x, y - 1] - float32(2.0) * cnsr[x, y]
        ) / tau_nsrT


@cuda.jit(device=True, inline=True)
def update_cp_and_cs(
    cp,
    cs,
    cs_tmp,
    cjsr,
    ci,
    RyR,
    ITCs,
    vp,
    ICa,
    INaCa,
    vs,
    kr,
    Jmax,
    tau_p,
    tau_ps,
    tau_si,
    tau_sT,
    tau_sL,
    x,
    y,
):
    """Update cp[x,y] and cs[x,y] under rapid equilibrium approximation."""
    cs_nn = float32(0.0)
    tau_diff = float32(0.0)
    # Get the sum nearest neighbours term, accounting for boundaries
    if x == 0 and y == 0:
        cs_nn = cs[x + 1, y] / tau_sL + cs[x, y + 1] / tau_sT
        tau_diff = float32(1.0) / tau_sL + float32(1.0) / tau_sT
    elif x == 0 and y == (cs.shape[1] - 1):
        cs_nn = cs[x + 1, y] / tau_sL + cs[x, y - 1] / tau_sT
        tau_diff = float32(1.0) / tau_sL + float32(1.0) / tau_sT
    elif x == (cs.shape[0] - 1) and y == 0:
        cs_nn = cs[x - 1, y] / tau_sL + cs[x, y + 1] / tau_sT
        tau_diff = float32(1.0) / tau_sL + float32(1.0) / tau_sT
    elif x == (cs.shape[0] - 1) and y == (cs.shape[1] - 1):
        cs_nn = cs[x - 1, y] / tau_sL + cs[x, y - 1] / tau_sT
        tau_diff = float32(1.0) / tau_sL + float32(1.0) / tau_sT
    elif x == 0:
        cs_nn = cs[x + 1, y] / tau_sL + (cs[x, y - 1] + cs[x, y + 1]) / tau_sT
        tau_diff = float32(1.0) / tau_sL + float32(2.0) / tau_sT
    elif x == (cs.shape[0] - 1):
        cs_nn = cs[x - 1, y] / tau_sL + (cs[x, y - 1] + cs[x, y + 1]) / tau_sT
        tau_diff = float32(1.0) / tau_sL + float32(2.0) / tau_sT
    elif y == 0:
        cs_nn = (cs[x - 1, y] + cs[x + 1, y]) / tau_sL + cs[x, y + 1] / tau_sT
        tau_diff = float32(2.0) / tau_sL + float32(1.0) / tau_sT
    elif y == (cs.shape[1] - 1):
        cs_nn = (cs[x - 1, y] + cs[x + 1, y]) / tau_sL + cs[x, y - 1] / tau_sT
        tau_diff = float32(2.0) / tau_sL + float32(1.0) / tau_sT
    else:
        cs_nn = (cs[x - 1, y] + cs[x + 1, y]) / tau_sL + (
            cs[x, y + 1] + cs[x, y - 1]
        ) / tau_sT
        tau_diff = float32(2.0) / tau_sL + float32(2.0) / tau_sT

    cp_tmp = cp[x, y]  # hold a temporary to update cs
    cp[x, y] = (cs[x, y] + tau_p * (kr * cjsr[x, y] - ICa)) / (
        float32(1.0) + tau_p * Jmax * (RyR[x, y, 1] + RyR[x, y, 2]) / vp[x, y]
    )
    # store this in a temporary so that threads don't overwrite each other in nearest neighbour calculation
    cs_tmp[x, y] = (
        cp_tmp * (vp[x, y] / (vs * tau_ps))
        + INaCa
        + ci[x, y] / tau_si
        - ITCs[x, y]
        + cs_nn
    ) / (float32(1.0) / tau_ps + float32(1.0) / tau_si + tau_diff)
