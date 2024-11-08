from typing import Any

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
    consts: Any,
    params: RestrepoParams,
):
    x, y = cuda.grid(2)
    Nx, Ny = cs.shape
    tid = y * Nx + x

    sqrtdt = consts.sqrtdt[0]

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
    consts: Any,
    params: RestrepoParams,
):

    idx = cuda.grid(1)
    Nx, Ny = cs.shape

    dt = consts.dt[0]

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
        update_ICa(ICa, LCC, cp_, params.PCa[0], z, params.gamma[0], params.Cao[0], idx)

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
    consts: Any,
    params: RestrepoParams,
):
    x, y = cuda.grid(2)
    Nx, Ny = cs.shape

    dt = consts.dt[0]

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

        ITCi = ITCa(ci_, CaTi_, params.kon[0], params.koff[0], params.BT[0])
        ITCs = ITCa(cs_, CaTs_, params.kon[0], params.koff[0], params.BT[0])
        Ileak_ = Ileak(cjsr_, cnsr_, ci_, params.gleak[0], square(params.Kjsr[0]))
        Iup_ = Iup(ci_, cnsr_, params.Ki[0], params.Knsr[0], params.vup[0])
        Ir_ = Ir(cp_, cjsr_, ryr_open, params.Jmax[0], params.vp[0])
        Ici = Delta_ci[x, y]
        Icnsr = Delta_cnsr[x, y]

        Itr = (cnsr_ - cjsr_) / params.tau_tr[0]
        Idsi = (cs_ - ci_) / params.tau_si[0]

        INaCa_ = boundary_from_flattened(INaCa, x, y, Nx, Ny)
        ICa_ = boundary_from_flattened(ICa, x, y, Nx, Ny)

        calmodulin_buf = params.KCAM[0] * params.BCAM[0] / square(params.KCAM[0] + ci_)
        SR_buf = params.KSR[0] * params.BSR[0] / square(params.KSR[0] + ci_)
        myosin_Ca_buf = params.KMCa[0] * params.BMCa[0] / square(params.KMCa[0] + ci_)
        myosin_Mg_buf = params.KMMg[0] * params.BMMg[0] / square(params.KMMg[0] + ci_)

        beta_i = float32(1.0) / (
            float32(1.0) + calmodulin_buf + SR_buf + myosin_Ca_buf + myosin_Mg_buf
        )
        beta_jsr = luminal_buffer(
            cjsr_,
            params.rho_inf[0],
            params.K[0],
            params.BCSQN[0],
            params.nM[0],
            params.nD[0],
            params.KC[0],
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
            params.vp[0],
            params.vs[0],
            params.tau_ps[0],
            params.tau_si[0],
            params.tau_sT[0],
            params.tau_sL[0],
            params.Jmax[0],
            x,
            y,
            Nx,
            Ny,
        )

        ci[x, y] += (
            dt
            * beta_i
            * ((params.vs[0] / params.vi[0]) * Idsi - Iup_ + Ileak_ - ITCi + Ici)
        )
        cnsr[x, y] += dt * (
            (params.vi[0] / params.vnsr[0]) * (Iup_ - Ileak_)
            - (params.vjsr[0] / params.vnsr[0]) * Itr
            + Icnsr
        )
        cjsr[x, y] += dt * beta_jsr * (Itr - (params.vp[0] / params.vjsr[0]) * Ir_)
        CaTi[x, y] += dt * ITCi
        CaTs[x, y] += dt * ITCs
