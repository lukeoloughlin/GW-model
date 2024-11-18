from typing import Any
import math

import numpy as np
import numpy.typing as npt
from numba import float32
from numba import cuda
from numba.cuda.random import xoroshiro128p_normal_float32

from params import RestrepoParams
from .RyR import update_RyR_rates_3d, update_RyR_diffusion_3d
from .LCC import update_LCC_probs_3d, sample_LCC_icdf_3d
from .currents import (
    ITCa,
    Ileak,
    Iup,
    Ir,
    update_ICa_3d,
    update_INaCa_3d,
    luminal_buffer,
    update_diffusive_fluxes_3d,
)
from .utils import square

f32 = np.float32
RNG_states = Any
RestrepoConsts = Any


@cuda.jit
def update_currents_and_lcc_3d(
    RyR: npt.NDArray[f32],
    LCC: npt.NDArray[f32],
    ci: npt.NDArray[f32],
    cs: npt.NDArray[f32],
    cjsr: npt.NDArray[f32],
    cnsr: npt.NDArray[f32],
    cp: npt.NDArray[f32],
    RyR_rates: npt.NDArray[f32],
    LCC_probs: npt.NDArray[f32],
    Delta_ci: npt.NDArray[f32],
    Delta_cnsr: npt.NDArray[f32],
    Delta_cs: npt.NDArray[f32],
    ICa: npt.NDArray[f32],
    INaCa: npt.NDArray[f32],
    junctional: npt.NDArray[np.bool_],
    dW: npt.NDArray[f32],
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
    V: f32,
    params: RestrepoParams,
    consts: RestrepoConsts,
) -> None:
    # strategy -- launch kernel with blockspergrid.x = 2 * ceil(Nx // threadsperblock.x), then use the first half to do the regular updates, and the second
    # half to do the junctional CRU work

    Nx, Ny, Nz = cs.shape
    blocks_per_task = cuda.gridDim.x // 2
    if cuda.blockIdx.x < blocks_per_task:
        x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
        task = 0
    else:
        x = cuda.threadIdx.x + (cuda.blockIdx.x - blocks_per_task) * cuda.blockDim.x
        y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
        task = 1

    if x < Nx and y < Ny and z < Nz:
        if task == 0:
            # update RyR rates
            update_RyR_rates_3d(
                RyR_rates,
                RyR,
                cp[x, y, z],
                cjsr[x, y, z],
                params,
                x,
                y,
                z,
            )

            # Update ci and cnsr diffusions
            update_diffusive_fluxes_3d(
                Delta_ci,
                Delta_cnsr,
                Delta_cs,
                ci,
                cnsr,
                cs,
                junctional,
                params,
                x,
                y,
                z,
            )
        elif task == 1:
            VF_RT = float32(V * 96.5 / (8.314 * 308.0))
            dt = consts.dt[0]
            tid = z * Ny * Nx + y * Nx + x

            # Only need to apply these updates to junctional CRUs
            if junctional[x, y, z]:
                update_LCC_probs_3d(
                    LCC_probs,
                    LCC,
                    cp[x, y, z],
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
                    x,
                    y,
                    z,
                )

                update_ICa_3d(
                    ICa,
                    LCC,
                    cp[x, y, z],
                    params.PCa[0],
                    VF_RT,
                    params.gamma[0],
                    params.Cao[0],
                    x,
                    y,
                    z,
                )
                update_INaCa_3d(INaCa, cs[x, y, z], z, consts.Nai3[0], params, x, y, z)

                for j in range(4):
                    LCC[x, y, z, j] = sample_LCC_icdf_3d(
                        LCC_probs, rng_states, x, y, z, j, tid
                    )

            # May as well get the idle threads to do something
            for i in range(4):
                dW[x, y, z, i] = consts.sqrtdt[0] * xoroshiro128p_normal_float32(
                    rng_states, tid
                )


@cuda.jit
def update_RyR_and_conc_3d(
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
    junctional: npt.NDArray[np.bool_],
    params: RestrepoParams,
    consts: RestrepoConsts,
):
    x, y, z = cuda.grid(3)
    Nx, Ny, Nz = cs.shape
    dt = consts.dt[0]

    if x < Nx and y < Ny and z < Nz:
        ryr_open = RyR[x, y, z, 1] + RyR[x, y, z, 2]
        ci_ = ci[x, y, z]
        cnsr_ = cnsr[x, y, z]
        cjsr_ = cjsr[x, y, z]
        cs_ = cs[x, y, z]
        cp_ = cp[x, y, z]
        CaTi_ = CaTi[x, y, z]
        CaTs_ = CaTs[x, y, z]
        vp_ = vp[x, y, z]
        is_junctional = junctional[x, y, z]

        ITCi = ITCa(ci_, CaTi_, params.kon[0], params.koff[0], params.BT[0])
        ITCs = ITCa(cs_, CaTs_, params.kon[0], params.koff[0], params.BT[0])
        Ileak_ = Ileak(cnsr_, ci_, params.gleak[0], square(params.Knsr[0]))
        Iup_ = Iup(ci_, cnsr_, params.Ki[0], params.Knsr[0], params.vup[0])
        Ir_ = Ir(cp_, cjsr_, ryr_open, params.Jmax[0], vp_)
        Ici = Delta_ci[x, y, z]
        Icnsr = Delta_cnsr[x, y, z]
        Ics = Delta_cs[x, y, z]

        Itr = (cnsr_ - cjsr_) / params.tau_tr[0]
        Idsi = (cs_ - ci_) / params.tau_si[0]
        Idps_scaled = (
            (cp_ - cs_) * vp_ / (params.tau_ps[0] * params.vs[0])
        )  # Idps * (vp/vs)

        INaCa_ = float32(0.0)
        ICa_ = float32(0.0)
        if is_junctional:
            INaCa_ = INaCa[x, y, z]
            ICa_ = ICa[x, y, z]

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
        update_RyR_diffusion_3d(RyR, RyR_sorted, RyR_rates, dW, dt, x, y, z)

        kr = (params.Jmax[0] / vp_) * ryr_open
        cp[x, y, z] = (cs_ + params.tau_ps[0] * (kr * cjsr_ - ICa_)) / (
            float32(1.0) + params.tau_ps[0] * kr
        )
        ci[x, y, z] += (
            dt
            * beta_i
            * ((params.vs[0] / params.vi[0]) * Idsi - Iup_ + Ileak_ - ITCi + Ici)
        )
        cs[x, y, z] += dt * beta_s * (Idps_scaled + INaCa_ - Idsi - ITCs + Ics)
        cnsr[x, y, z] += dt * (
            (params.vi[0] / params.vnsr[0]) * (Iup_ - Ileak_)
            - (params.vjsr[0] / params.vnsr[0]) * Itr
            + Icnsr
        )
        cjsr[x, y, z] += dt * beta_jsr * (Itr - (vp_ / params.vjsr[0]) * Ir_)
        CaTi[x, y, z] += dt * ITCi
        CaTs[x, y, z] += dt * ITCs
