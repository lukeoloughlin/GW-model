import math

import numpy as np
import numpy.typing as npt
from numba import float32
from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32

from params import RestrepoParams
from RyR import update_RyR_rates, update_RyR_diffusion
from LCC import update_LCC_probs, sample_icdf
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
from utils import square, boundary_from_flattened, flattened_from_boundary


@cuda.jit
def currents_and_RyR(
    RyR: npt.NDArray,
    RyR_rates: npt.NDArray,
    ci: npt.NDArray,
    cs: npt.NDArray,
    cjsr: npt.NDArray,
    cnsr: npt.NDArray,
    cp: npt.NDArray,
    Delta_ci: npt.NDArray,
    Delta_cnsr: npt.NDArray,
    sum_cs_nn: npt.NDArray,
    dW: npt.NDArray,
    rng_states: npt.NDArray,
    sqrtdt: float,
    params: RestrepoParams,
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
            params,
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
            params,
            x,
            y,
        )

        # update dW
        for i in range(4):
            dW[x, y, i] = sqrtdt * xoroshiro128p_normal_float32(rng_states, tid)


@cuda.jit
def update_boundary_currents_and_LCC(
    LCC: npt.NDArray,
    LCC_probs: npt.NDArray,
    cp: npt.NDArray,
    cs: npt.NDArray,
    ICa: npt.NDArray,
    INaCa: npt.NDArray,
    rng_states: npt.NDArray,
    Nai3: float,
    alpha: float,
    beta: float,
    k3: float,
    k3_: float,
    k5_: float,
    k6_: float,
    Pr: float,
    Ps: float,
    R: float,
    z: float,
    dt: float,
    params: RestrepoParams,
):

    idx = cuda.grid(1)
    Nx, Ny = cs.shape

    if idx < ICa.shape[0]:
        # Get cs ans cp values from the corresponding boundary element
        cs_ = flattened_from_boundary(cs, Nx, Ny, idx)
        cp_ = flattened_from_boundary(cp, Nx, Ny, idx)

        # Update LCC dist
        update_LCC_probs(
            LCC_probs,
            LCC,
            cp_,
            dt,
            alpha,
            beta,
            k3,
            k3_,
            k5_,
            k6_,
            Pr,
            Ps,
            R,
            params,
            idx,
        )

        # Update ICa
        update_ICa(ICa, LCC, cp_, params.PCa, z, params.gamma, params.Cao, idx)

        # Update INaCa
        update_INaCa(
            INaCa,
            cs_,
            z,
            Nai3,
            params,
            idx,
        )

        # Safe to update LCC here since ICa has already been calculated and stored for later use
        for j in range(4):
            LCC[idx, j] = sample_icdf(LCC_probs, rng_states, idx, j)


@cuda.jit
def update_RyR_and_euler_step(
    ci: npt.NDArray,
    cs: npt.NDArray,
    cp: npt.NDArray,
    cnsr: npt.NDArray,
    cjsr: npt.NDArray,
    CaTi: npt.NDArray,
    CaTs: npt.NDArray,
    RyR: npt.NDArray,
    RyR_sorted: npt.NDArray,
    ICa: npt.NDArray,
    INaCa: npt.NDArray,
    Delta_ci: npt.NDArray,
    Delta_cnsr: npt.NDArray,
    sum_cs_nn: npt.NDArray,
    RyR_rates: npt.NDArray,
    dW: npt.NDArray,
    # vp: npt.NDArray -- Need to add when variable proximal subspace is used
    dt: float,
    params: RestrepoParams,
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
        # vp_ = vp[x, y]
        sum_cs_nn_ = sum_cs_nn[x, y]

        ITCi = ITCa(ci_, CaTi_, params.kon, params.koff, params.BT)
        ITCs = ITCa(cs_, CaTs_, params.kon, params.koff, params.BT)
        Ileak_ = Ileak(cjsr_, cnsr_, ci_, params.gleak, square(params.Kjsr))
        Iup_ = Iup(ci_, cnsr_, params.Ki, params.Knsr, params.vup)
        Ir_ = Ir(cp_, cjsr_, ryr_open, params.Jmax, params.vp)
        Ici = Delta_ci[x, y]
        Icnsr = Delta_cnsr[x, y]

        Itr = (cnsr_ - cjsr_) / params.tau_tr
        Idsi = (cs_ - ci_) / params.tau_si

        INaCa_ = boundary_from_flattened(INaCa, x, y, Nx, Ny)
        ICa_ = boundary_from_flattened(ICa, x, y, Nx, Ny)

        calmodulin_buf = params.KCAM * params.BCAM / square(params.KCAM + ci_)
        SR_buf = params.KSR * params.BSR / square(params.KSR + ci_)
        myosin_Ca_buf = params.KMCa * params.BMCa / square(params.KMCa + ci_)
        myosin_Mg_buf = params.KMMg * params.BMMg / square(params.KMMg + ci_)

        beta_i = float32(1.0) / (
            float32(1.0) + calmodulin_buf + SR_buf + myosin_Ca_buf + myosin_Mg_buf
        )
        beta_jsr = luminal_buffer(
            cjsr_,
            params.rho_inf,
            params.K,
            params.BCSQN,
            params.nM,
            params.nD,
            params.KC,
        )

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
            ICa_,
            INaCa_,
            ITCs,
            sum_cs_nn_,
            params.vp,
            params.vs,
            params.tau_ps,
            params.tau_si,
            params.tau_sT,
            params.tau_sL,
            params.Jmax,
            x,
            y,
            Nx,
            Ny,
        )

        ci[x, y] += (
            dt * beta_i * ((params.vs / params.vi) * Idsi - Iup_ + Ileak_ - ITCi + Ici)
        )
        cnsr[x, y] += dt * (
            (params.vi / params.vnsr) * (Iup_ - Ileak_)
            - (params.vjsr / params.vnsr) * Itr
            + Icnsr
        )
        cjsr[x, y] += dt * beta_jsr * (Itr - (params.vp / params.vjsr) * Ir_)
        CaTi[x, y] += dt * ITCi
        CaTs[x, y] += dt * ITCs
