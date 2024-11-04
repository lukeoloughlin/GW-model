import math

import numpy as np
import numpy.typing as npt
from numba import float32
from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32

from RyR import update_RyR_rates, update_RyR_diffusion
from LCC import update_LCC_probs, sample_icdf
from currents import (
    update_diffusive_fluxes,
    update_ITC,
    update_ICa,
    update_INaCa,
    luminal_buffer,
    get_ICa,
    get_INaCa,
)


@cuda.jit(device=True, inline=True)
def get_boundary_values(arr, height, width, idx):
    """Convert idx to appropriate 2d index on boundary of arr and return the corresponding value of arr
    For now I will assume that the values are arranged according to top, bottom, left, right
    """
    if idx < width:
        return arr[0, idx]
    elif idx < 2 * width:
        return arr[height - 1, idx - width]
    elif idx < 2 * width + height - 2:
        return arr[idx - 2 * width, 0]
    else:
        return arr[idx - (2 * width + height - 2), width - 1]


@cuda.jit
def currents_and_RyR(
    RyR,
    RyR_rates,
    # RyR_sorted,
    ci,
    cs,
    cjsr,
    cnsr,
    cp,
    # CaTi,
    # CaTs,
    Delta_ci,
    Delta_cnsr,
    # ITCi,
    # ITCs,
    dW,
    rng_states,
    sqrtdt,
    Ku,
    Kb,
    tau_u,
    tau_b,
    tau_c,
    BCSQN,
    rho_inf,
    K,
    # kon,
    # koff,
    # BT,
    tau_iT,
    tau_iL,
    tau_nsrT,
    tau_nsrL,
):
    x, y = cuda.grid(2)
    Nx, Ny = cs.shape
    tid = y * Nx + x

    if x < Nx and y < Ny:
        # update RyR rates
        update_RyR_rates(
            RyR_rates,
            RyR,
            cp,
            cjsr,
            Ku,
            Kb,
            tau_u,
            tau_b,
            tau_c,
            BCSQN,
            rho_inf,
            K,
            x,
            y,
        )

        # update ITCs
        # update_ITC(ITCi, ITCs, ci, cs, CaTi, CaTs, kon, koff, BT, x, y)

        # Update ci and cnsr diffusions
        update_diffusive_fluxes(
            Delta_ci, Delta_cnsr, ci, cnsr, tau_iT, tau_iL, tau_nsrT, tau_nsrL, x, y
        )

        # update dW
        for i in range(4):
            dW[x, y, i] = sqrtdt * xoroshiro128p_normal_float32(rng_states, tid)


@cuda.jit
def update_boundary_currents_and_LCC(
    LCC,
    LCC_probs,
    cp,
    cs,
    ICa,
    INaCa,
    rng_states,
    alpha,
    beta,
    r1,
    r2,
    s1_,
    cp_bar,
    cp_tilde,
    k1_,
    k2,
    k2_,
    k3,
    k3_,
    k5_,
    k6_,
    Pr,
    Ps,
    R,
    PCa,
    z,
    gamma,
    Cao,
    vNaCa,
    eta,
    Nai3,
    Nao3,
    ksat,
    KmNao3,
    KmCao,
    KmCai,
    Kda,
    t1,
    dt,
):

    idx = cuda.grid(1)
    Nx, Ny = cs.shape

    if idx < ICa.shape[0]:
        # Get cs ans cp values from the corresponding boundary element
        cs_ = get_boundary_values(cs, Nx, Ny, idx)
        cp_ = get_boundary_values(cp, Nx, Ny, idx)

        # Update LCC dist
        update_LCC_probs(
            LCC_probs,
            LCC,
            cp_,
            dt,
            alpha,
            beta,
            r1,
            r2,
            s1_,
            cp_bar,
            cp_tilde,
            k1_,
            k2,
            k2_,
            k3,
            k3_,
            k5_,
            k6_,
            Pr,
            Ps,
            R,
            idx,
        )

        # Update ICa
        update_ICa(ICa, LCC, cp_, PCa, z, gamma, Cao, idx)

        # Update INaCa
        update_INaCa(
            INaCa,
            cs_,
            vNaCa,
            z,
            eta,
            Nai3,
            Cao,
            Nao3,
            ksat,
            KmNao3,
            KmCao,
            KmCai,
            Kda,
            t1,
            idx,
        )

        for j in range(4):
            LCC[idx, j] = sample_icdf(LCC_probs, rng_states, idx, j)


@cuda.jit
def update_RyR_and_euler_step(
    ci,
    cs,
    cp,
    cnsr,
    cjsr,
    CaTi,
    CaTs,
    RyR,
    LCC,
    cs_tmp,
    RyR_sorted,
    ICa,
    INaCa,
    Delta_ci,
    Delta_cnsr,
    RyR_rates,
    dW,
    dt,
    kon,
    koff,
    BT,
    gleak,
    Kjsr2,
    vup,
    Ki,
    Knsr,
    tau_tr,
    rho_inf,
    K,
    BCSQN,
    nM,
    nD,
    KC,
):
    x, y = cuda.grid(2)
    Nx, Ny = cs.shape

    if x < Nx and y < Ny:
        ryr_open = RyR[x, y, 1] + RyR[x, y, 2]  # Hold this value before updating

        update_RyR_diffusion(
            RyR,
            RyR_sorted,
            RyR_rates,
            dW,
            dt,
            x,
            y,
        )

        ITCi = kon * ci[x, y] * (BT - CaTi[x, y]) - koff * CaTi[x, y]
        ITCs = kon * cs[x, y] * (BT - CaTs[x, y]) - koff * CaTs[x, y]

        cjsr2 = cjsr[x, y] * cjsr[x, y]
        Ileak = gleak * cjsr2 / (cjsr2 + Kjsr2) * (cnsr[x, y] - ci[x, y])

        ci_Ki_pow = math.pow(ci[x, y] / Ki, float32(1.787))
        cnsr_Knsr_pow = math.pow(ci[x, y] / Knsr, float32(1.787))

        Iup = (
            vup
            * (ci_Ki_pow - cnsr_Knsr_pow)
            / (float32(1.0) + ci_Ki_pow + cnsr_Knsr_pow)
        )

        Itr = (cnsr[x, y] - cjsr[x, y]) / tau_tr

        calmodulin_buf = ...
        SR_buf = ...
        myosin_Ca_buf = ...
        myosin_Mg_buf = ...
        beta_i = float32(1.0) / (
            float32(1.0) + calmodulin_buf + SR_buf + myosin_Ca_buf + myosin_Mg_buf
        )
        beta_jsr = luminal_buffer(cjsr[x, y], rho_inf, K, BCSQN, nM, nD, KC)

        INaCa = get_INaCa(ICa, x, y, Nx, Ny)
        ICa = get_ICa(INaCa, x, y, Nx, Ny)

        ### TODO: Finish writing this and figure out if i need to syncthreads
