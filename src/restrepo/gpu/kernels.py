from typing import Any

import numpy as np
import numpy.typing as npt
from numba import float32
from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32

from params import RestrepoParams
from .RyR import update_RyR_rates, update_RyR_diffusion
from .LCC import update_LCC_probs, sample_LCC_icdf
from .currents import (
    ITCa,
    Ileak,
    Iup,
    Ir,
    update_ICa,
    update_INaCa,
    luminal_buffer,
    update_diffusive_fluxes,
)
from .utils import (
    square,
    boundary_from_flattened,
    flattened_from_boundary,
)

f32 = np.float32
RNG_states = Any
RestrepoConsts = Any


@cuda.jit
def currents_and_RyR(
    RyR: npt.NDArray[f32],
    ci: npt.NDArray[f32],
    cs: npt.NDArray[f32],
    cjsr: npt.NDArray[f32],
    cnsr: npt.NDArray[f32],
    cp: npt.NDArray[f32],
    RyR_rates: npt.NDArray[f32],
    Delta_ci: npt.NDArray[f32],
    Delta_cnsr: npt.NDArray[f32],
    Delta_cs: npt.NDArray[f32],
    dW: npt.NDArray[f32],
    rng_states: RNG_states,
    params: RestrepoParams,
    consts: RestrepoConsts,
) -> None:
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
            Delta_cs,
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
    LCC: npt.NDArray[f32],
    LCC_probs: npt.NDArray[f32],
    cp: npt.NDArray[f32],
    cs: npt.NDArray[f32],
    ICa: npt.NDArray[f32],
    INaCa: npt.NDArray[f32],
    rng_states: RNG_states,
    alpha: f32,
    beta: f32,
    k3: f32,
    k3_: f32,
    k5_: f32,
    k6_: f32,
    Pr: f32,
    Ps: f32,
    R: f32,
    z: f32,
    V: f32,
    params: RestrepoParams,
    consts: RestrepoConsts,
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
            V,
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
            consts.Nai3[0],
            params,
            idx,
        )

        # Safe to update LCC here since ICa has already been calculated and stored for later use
        for j in range(4):
            LCC[idx, j] = sample_LCC_icdf(LCC_probs, rng_states, idx, j)


@cuda.jit
def update_RyR_and_euler_step(
    ci: npt.NDArray[f32],
    cs: npt.NDArray[f32],
    cp: npt.NDArray[f32],
    cnsr: npt.NDArray[f32],
    cjsr: npt.NDArray[f32],
    CaTi: npt.NDArray[f32],
    CaTs: npt.NDArray[f32],
    RyR: npt.NDArray[f32],
    RyR_sorted: npt.NDArray[f32],
    ICa: npt.NDArray[f32],
    INaCa: npt.NDArray[f32],
    Delta_ci: npt.NDArray[f32],
    Delta_cnsr: npt.NDArray[f32],
    Delta_cs: npt.NDArray[f32],
    RyR_rates: npt.NDArray[f32],
    dW: npt.NDArray[f32],
    vp: npt.NDArray[f32],
    params: RestrepoParams,
    consts: RestrepoConsts,
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
        vp_ = vp[x, y]

        ITCi = ITCa(ci_, CaTi_, params.kon[0], params.koff[0], params.BT[0])
        ITCs = ITCa(cs_, CaTs_, params.kon[0], params.koff[0], params.BT[0])
        Ileak_ = Ileak(cnsr_, ci_, params.gleak[0], square(params.Kjsr[0]))
        Iup_ = Iup(ci_, cnsr_, params.Ki[0], params.Knsr[0], params.vup[0])
        Ir_ = Ir(cp_, cjsr_, ryr_open, params.Jmax[0], vp_)
        Ici = Delta_ci[x, y]
        Icnsr = Delta_cnsr[x, y]
        Ics = Delta_cs[x, y]

        Itr = (cnsr_ - cjsr_) / params.tau_tr[0]
        Idsi = (cs_ - ci_) / params.tau_si[0]

        INaCa_ = boundary_from_flattened(INaCa, x, y, Nx, Ny)
        ICa_ = boundary_from_flattened(ICa, x, y, Nx, Ny)

        calmodulin_buf_i = (
            params.KCAM[0] * params.BCAM[0] / square(params.KCAM[0] + ci_)
        )
        SR_buf = params.KSR[0] * params.BSR[0] / square(params.KSR[0] + ci_)
        myosin_Ca_buf = params.KMCa[0] * params.BMCa[0] / square(params.KMCa[0] + ci_)
        myosin_Mg_buf = params.KMMg[0] * params.BMMg[0] / square(params.KMMg[0] + ci_)

        calmodulin_buf_s = (
            params.KCAM[0] * params.BCAM[0] / square(params.KCAM[0] + cs_)
        )
        SLH_buf = params.KSLH[0] * params.BSLH[0] / square(params.KSLH[0] + cs_)

        beta_i = float32(1.0) / (
            float32(1.0) + calmodulin_buf_i + SR_buf + myosin_Ca_buf + myosin_Mg_buf
        )
        beta_s = float32(1.0) / (float32(1.0) + calmodulin_buf_s + SLH_buf)
        beta_jsr = luminal_buffer(
            cjsr_,
            params.rho_inf[0],
            params.K[0],
            params.BCSQN[0],
            params.nM[0],
            params.nD[0],
            params.KC[0],
            params.h[0],
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

        kr = (params.Jmax[0] / vp_) * ryr_open
        cp[x, y] = (cs_ + params.tau_ps[0] * (kr * cjsr_ - ICa_)) / (
            float32(1.0) + params.tau_ps[0] * kr
        )
        ci[x, y] += (
            dt
            * beta_i
            * ((params.vs[0] / params.vi[0]) * Idsi - Iup_ + Ileak_ - ITCi + Ici)
        )
        cs[x, y] += (
            dt
            * beta_s
            * (
                (cp_ - cs_) * vp_ / (params.tau_ps[0] * params.vs[0])
                + INaCa_
                - Idsi
                - ITCs
                + Ics
            )
        )
        cnsr[x, y] += dt * (
            (params.vi[0] / params.vnsr[0]) * (Iup_ - Ileak_)
            - (params.vjsr[0] / params.vnsr[0]) * Itr
            + Icnsr
        )
        cjsr[x, y] += dt * beta_jsr * (Itr - (vp_ / params.vjsr[0]) * Ir_)
        CaTi[x, y] += dt * ITCi
        CaTs[x, y] += dt * ITCs
