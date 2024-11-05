import math

import numpy as np
import numpy.typing as npt
from numba import float32
from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32

from RyR import update_RyR_rates, update_RyR_diffusion
from LCC import update_LCC_probs, sample_icdf
from utils import square, get_boundary_val
from currents import (
    ITCa,
    Ileak,
    Iup,
    Ir,
    update_ICa,
    update_INaCa,
    luminal_buffer,
    update_diffusive_fluxes,
    cp_cs_iter,
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
    sum_cs_nn,
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
    tau_sL,
    tau_sT,
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

        # Update ci and cnsr diffusions
        update_diffusive_fluxes(
            Delta_ci,
            Delta_cnsr,
            sum_cs_nn,
            ci,
            cnsr,
            cs,
            tau_iT,
            tau_iL,
            tau_nsrT,
            tau_nsrL,
            tau_sL,
            tau_sT,
            x,
            y,
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

        # Safe to update LCC here since ICa has already been calculated and stored for later use
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
    RyR_sorted,
    ICa,
    INaCa,
    Delta_ci,
    Delta_cnsr,
    sum_cs_nn,
    RyR_rates,
    dW,
    dt,
    kon,
    koff,
    BT,
    gleak,
    Kjsr2,
    vup,
    vp,
    Jmax,
    Ki,
    Knsr,
    tau_tr,
    tau_si,
    rho_inf,
    K,
    BCSQN,
    nM,
    nD,
    KC,
    KCAM,
    BCAM,
    KSR,
    BSR,
    KMCa,
    BMCa,
    KMMg,
    BMMg,
    vs,
    vi,
    vnsr,
    vjsr,
    tau_p,
    tau_sT,
    tau_sL,
):
    x, y = cuda.grid(2)
    Nx, Ny = cs.shape

    if x < Nx and y < Ny:
        ryr_open = RyR[x, y, 1] + RyR[x, y, 2]
        ci_ = ci[x, y]
        cnsr_ = cnsr[x, y]
        cjsr_ = cjsr[x, y]
        cs_ = cs[x, y]
        cp_ = cp[x, y]
        CaTi_ = CaTi[x, y]
        CaTs_ = CaTs[x, y]
        vp_ = vp[x, y]
        sum_cs_nn_ = sum_cs_nn[x, y]

        ITCi = ITCa(ci_, CaTi_, kon, koff, BT)
        ITCs = ITCa(cs_, CaTs_, kon, koff, BT)
        Ileak_ = Ileak(cjsr_, cnsr_, ci_, gleak, Kjsr2)
        Iup_ = Iup(ci_, cnsr_, Ki, Knsr, vup)
        Ir_ = Ir(cp_, cjsr_, ryr_open, Jmax, vp)
        Ici = Delta_ci[x, y]
        Icnsr = Delta_cnsr[x, y]

        Itr = (cnsr_ - cjsr_) / tau_tr
        Idsi = (cs_ - ci_) / tau_si

        INaCa = get_boundary_val(ICa, x, y, Nx, Ny)
        ICa = get_boundary_val(INaCa, x, y, Nx, Ny)

        calmodulin_buf = KCAM * BCAM / square(KCAM + ci_)
        SR_buf = KSR * BSR / square(KSR + ci_)
        myosin_Ca_buf = KMCa * BMCa / square(KMCa + ci_)
        myosin_Mg_buf = KMMg * BMMg / square(KMMg + ci_)

        beta_i = float32(1.0) / (
            float32(1.0) + calmodulin_buf + SR_buf + myosin_Ca_buf + myosin_Mg_buf
        )
        beta_jsr = luminal_buffer(cjsr_, rho_inf, K, BCSQN, nM, nD, KC)

        # Euler-Maruyama step for RyRs
        update_RyR_diffusion(
            RyR,
            RyR_sorted,
            RyR_rates,
            dW,
            dt,
            x,
            y,
        )

        # Fixed point iteration for rapid equilibrium approximation of cp and cs
        cp_cs_iter(
            cp,
            cs,
            cp_,
            cs_,
            cjsr_,
            ci_,
            ryr_open,
            ICa,
            INaCa,
            ITCs,
            sum_cs_nn_,
            vp_,
            vs,
            tau_p,
            tau_si,
            tau_sT,
            tau_sL,
            Jmax,
            x,
            y,
            Nx,
            Ny,
        )

        ci[x, y] += dt * beta_i * ((vs / vi) * Idsi - Iup_ + Ileak_ - ITCi + Ici)
        cnsr[x, y] += dt * ((vi / vnsr) * (Iup_ - Ileak_) - (vjsr / vnsr) * Itr + Icnsr)
        cjsr[x, y] += dt * beta_jsr * (Itr - (vp_ / vjsr) * Ir_)
        CaTi[x, y] += dt * ITCi
        CaTs[x, y] += dt * ITCs
