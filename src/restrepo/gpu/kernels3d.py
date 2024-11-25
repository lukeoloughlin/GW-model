from typing import Any
import math

import numpy as np
import numpy.typing as npt
from numba import float32
from numba import cuda
from numba.cuda.random import (
    xoroshiro128p_normal_float32,
    xoroshiro128p_uniform_float32,
)

from params import RestrepoParams
from .RyR import update_RyR_rates_3d, update_RyR_diffusion_3d, update_RyR_tau_leap_3d
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
from .utils import square, ryr_normal_inplace

f32 = np.float32
i32 = np.int32
RNG_states = Any
RestrepoConsts = Any


@cuda.jit
def update_currents_and_lcc_3d(
    RyR: npt.NDArray[f32],
    LCC: npt.NDArray[i32],
    ci: npt.NDArray[f32],
    cs: npt.NDArray[f32],
    cjsr: npt.NDArray[f32],
    cnsr: npt.NDArray[f32],
    cp: npt.NDArray[f32],
    RyR_rates: npt.NDArray[f32],
    RyR_open: npt.NDArray[f32],
    LCC_probs: npt.NDArray[f32],
    Delta_ci: npt.NDArray[f32],
    Delta_cnsr: npt.NDArray[f32],
    Delta_cs: npt.NDArray[f32],
    ICa: npt.NDArray[f32],
    INaCa: npt.NDArray[f32],
    junctional: npt.NDArray[np.bool_],
    dW: npt.NDArray[f32],
    lcc_random_nums: npt.NDArray[f32],
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
    # strategy -- launch kernel with blockspergrid.x = 3 * ceil(Nx // threadsperblock.x), then use the first third to do the RyR rate updates, the second
    # third to calculate the diffusion terms, and the last third to do the junctional CRU work

    Nx, Ny, Nz = cs.shape
    blocks_per_task = cuda.gridDim.x // 3
    if cuda.blockIdx.x < blocks_per_task:
        x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
        task = 0
    elif cuda.blockIdx.x < 2 * blocks_per_task:
        x = cuda.threadIdx.x + (cuda.blockIdx.x - blocks_per_task) * cuda.blockDim.x
        y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
        task = 1
    else:
        x = cuda.threadIdx.x + (cuda.blockIdx.x - 2 * blocks_per_task) * cuda.blockDim.x
        y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
        task = 2

    if x < Nx and y < Ny and z < Nz:
        cp_tmp = cp[x, y, z]
        cs_tmp = cs[x, y, z]
        cjsr_tmp = cjsr[x, y, z]
        if task == 0:
            RyR_open[x, y, z] = RyR[x, y, z, 1] + RyR[x, y, z, 2]
            # update RyR rates
            update_RyR_rates_3d(
                RyR_rates,
                RyR,
                cp_tmp,
                cjsr_tmp,
                params,
                x,
                y,
                z,
            )
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
            # Update ci and cnsr diffusions
            VF_RT = float32(V * 96.5 / (8.314 * 308.0))
            update_LCC_probs_3d(
                LCC_probs,
                LCC,
                cp_tmp,
                consts.dt[0],
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
                cp_tmp,
                params.PCa[0],
                VF_RT,
                params.gamma[0],
                params.Cao[0],
                junctional[x, y, z],
                x,
                y,
                z,
            )
            update_INaCa_3d(
                INaCa,
                cs_tmp,
                VF_RT,
                consts.Nai3[0],
                params,
                junctional[x, y, z],
                x,
                y,
                z,
            )
        elif task == 2:
            # May as well get the idle threads to do something
            tid = z * Ny * Nx + y * Nx + x
            # dW[x, y, z, i] = consts.sqrtdt[0] * xoroshiro128p_normal_float32(
            #    rng_states, tid
            # )
            ryr_normal_inplace(dW, rng_states, tid, x, y, z, consts.sqrtdt[0])
            if junctional[x, y, z]:
                for i in range(4):
                    lcc_random_nums[x, y, z, i] = xoroshiro128p_uniform_float32(
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
    LCC: npt.NDArray[i32],
    RyR_open: npt.NDArray[f32],
    RyR_sorted: npt.NDArray[f32],
    LCC_probs: npt.NDArray[f32],
    ICa: npt.NDArray[f32],
    INaCa: npt.NDArray[f32],
    Delta_ci: npt.NDArray[f32],
    Delta_cnsr: npt.NDArray[f32],
    Delta_cs: npt.NDArray[f32],
    RyR_rates: npt.NDArray[f32],
    dW: npt.NDArray[f32],
    lcc_random_nums: npt.NDArray[f32],
    vp: npt.NDArray[f32],
    junctional: npt.NDArray[np.bool_],
    Po_shift: f32,
    params: RestrepoParams,
    consts: RestrepoConsts,
):
    # x, y, z = cuda.grid(3)
    Nx, Ny, Nz = cs.shape
    blocks_per_task = cuda.gridDim.x // 3
    if cuda.blockIdx.x < blocks_per_task:
        x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
        task = 0
    elif cuda.blockIdx.x < 2 * blocks_per_task:
        x = cuda.threadIdx.x + (cuda.blockIdx.x - blocks_per_task) * cuda.blockDim.x
        y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
        task = 1
    else:
        x = cuda.threadIdx.x + (cuda.blockIdx.x - 2 * blocks_per_task) * cuda.blockDim.x
        y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
        task = 2

    if x < Nx and y < Ny and z < Nz:
        dt = consts.dt[0]
        if task == 0:
            ryr_open = RyR_open[x, y, z]
            ci_tmp = ci[x, y, z]
            cnsr_tmp = cnsr[x, y, z]
            cjsr_tmp = cjsr[x, y, z]
            cs_tmp = cs[x, y, z]
            cp_tmp = cp[x, y, z]
            CaTi_tmp = CaTi[x, y, z]
            CaTs_tmp = CaTs[x, y, z]
            vp_tmp = vp[x, y, z]
            is_junctional = junctional[x, y, z]
            Po_shifted = max(f32(0.0), ryr_open - Po_shift)

            ITCi = ITCa(ci_tmp, CaTi_tmp, params.kon[0], params.koff[0], params.BT[0])
            ITCs = ITCa(cs_tmp, CaTs_tmp, params.kon[0], params.koff[0], params.BT[0])
            Ileak_ = Ileak(cnsr_tmp, ci_tmp, params.gleak[0], square(params.Kjsr[0]))
            Iup_ = Iup(
                ci_tmp,
                cnsr_tmp,
                params.Ki[0],
                params.Knsr[0],
                params.vup[0],
                params.H[0],
            )
            # Ir_ = Ir(cp_tmp, cjsr_tmp, ryr_open, params.Jmax[0], vp_tmp)
            Ir_scaled = (
                params.Jmax[0] * Po_shifted * (cjsr_tmp - cp_tmp) / params.vjsr[0]
            )
            Ici = Delta_ci[x, y, z]
            Icnsr = Delta_cnsr[x, y, z]
            Ics = Delta_cs[x, y, z]

            Itr = (cnsr_tmp - cjsr_tmp) / params.tau_tr[0]
            Idsi = (cs_tmp - ci_tmp) / params.tau_si[0]
            Idps_scaled = (
                (cp_tmp - cs_tmp) * vp_tmp / (params.tau_ps[0] * params.vs[0])
            )  # Idps * (vp/vs)

            INaCa_ = INaCa[x, y, z] if is_junctional else float32(0.0)
            ICa_ = ICa[x, y, z] if is_junctional else float32(0.0)

            calmodulin_buf_i = (
                params.KCAM[0] * params.BCAM[0] / square(params.KCAM[0] + ci_tmp)
            )
            SR_buf = params.KSR[0] * params.BSR[0] / square(params.KSR[0] + ci_tmp)
            myosin_Ca_buf = (
                params.KMCa[0] * params.BMCa[0] / square(params.KMCa[0] + ci_tmp)
            )
            myosin_Mg_buf = (
                params.KMMg[0] * params.BMMg[0] / square(params.KMMg[0] + ci_tmp)
            )

            calmodulin_buf_s = (
                params.KCAM[0] * params.BCAM[0] / square(params.KCAM[0] + cs_tmp)
            )
            SLH_buf = params.KSLH[0] * params.BSLH[0] / square(params.KSLH[0] + cs_tmp)

            beta_i = float32(1.0) / (
                float32(1.0) + calmodulin_buf_i + SR_buf + myosin_Ca_buf + myosin_Mg_buf
            )
            beta_s = float32(1.0) / (float32(1.0) + calmodulin_buf_s + SLH_buf)
            beta_jsr = luminal_buffer(
                cjsr_tmp,
                params.rho_inf[0],
                params.K[0],
                params.BCSQN[0],
                params.nM[0],
                params.nD[0],
                params.KC[0],
                params.h[0],
            )

            kr = params.Jmax[0] / vp_tmp
            cp[x, y, z] = (
                cs_tmp + params.tau_ps[0] * (kr * Po_shifted * cjsr_tmp - ICa_)
            ) / (float32(1.0) + params.tau_ps[0] * kr * Po_shifted)
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
            cjsr[x, y, z] += dt * beta_jsr * (Itr - Ir_scaled)
            CaTi[x, y, z] += dt * ITCi
            CaTs[x, y, z] += dt * ITCs
        elif task == 1:
            is_junctional = junctional[x, y, z]
            for j in range(4):
                u = lcc_random_nums[x, y, z, j]
                LCC[x, y, z, j] = sample_LCC_icdf_3d(
                    LCC_probs, LCC[x, y, z, j], u, x, y, z, j, is_junctional
                )
        elif task == 2:
            # Euler-Maruyama step for RyRs
            update_RyR_diffusion_3d(RyR, RyR_sorted, RyR_rates, dW, dt, x, y, z)


@cuda.jit
def update_currents_and_lcc_3d_tau_leap(
    RyR: npt.NDArray[i32],
    LCC: npt.NDArray[i32],
    ci: npt.NDArray[f32],
    cs: npt.NDArray[f32],
    cjsr: npt.NDArray[f32],
    cnsr: npt.NDArray[f32],
    cp: npt.NDArray[f32],
    RyR_rates: npt.NDArray[f32],
    RyR_open: npt.NDArray[f32],
    LCC_probs: npt.NDArray[f32],
    Delta_ci: npt.NDArray[f32],
    Delta_cnsr: npt.NDArray[f32],
    Delta_cs: npt.NDArray[f32],
    ICa: npt.NDArray[f32],
    INaCa: npt.NDArray[f32],
    junctional: npt.NDArray[np.bool_],
    lcc_random_nums: npt.NDArray[f32],
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
    # strategy -- launch kernel with blockspergrid.x = 3 * ceil(Nx // threadsperblock.x), then use the first third to do the RyR rate updates, the second
    # third to calculate the diffusion terms, and the last third to do the junctional CRU work

    Nx, Ny, Nz = cs.shape
    blocks_per_task = cuda.gridDim.x // 3
    if cuda.blockIdx.x < blocks_per_task:
        x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
        task = 0
    elif cuda.blockIdx.x < 2 * blocks_per_task:
        x = cuda.threadIdx.x + (cuda.blockIdx.x - blocks_per_task) * cuda.blockDim.x
        y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
        task = 1
    else:
        x = cuda.threadIdx.x + (cuda.blockIdx.x - 2 * blocks_per_task) * cuda.blockDim.x
        y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
        task = 2

    if x < Nx and y < Ny and z < Nz:
        cp_tmp = cp[x, y, z]
        cs_tmp = cs[x, y, z]
        cjsr_tmp = cjsr[x, y, z]
        if task == 0:
            RyR_open[x, y, z] = float32(RyR[x, y, z, 1] + RyR[x, y, z, 2]) * float32(
                0.01
            )
            # update RyR rates
            update_RyR_rates_3d(
                RyR_rates,
                RyR,
                cp_tmp,
                cjsr_tmp,
                params,
                x,
                y,
                z,
            )
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
            # Update ci and cnsr diffusions
            VF_RT = float32(V * 96.5 / (8.314 * 308.0))
            update_LCC_probs_3d(
                LCC_probs,
                LCC,
                cp_tmp,
                consts.dt[0],
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
                cp_tmp,
                params.PCa[0],
                VF_RT,
                params.gamma[0],
                params.Cao[0],
                junctional[x, y, z],
                x,
                y,
                z,
            )
            update_INaCa_3d(
                INaCa,
                cs_tmp,
                VF_RT,
                consts.Nai3[0],
                params,
                junctional[x, y, z],
                x,
                y,
                z,
            )
        elif task == 2:
            # May as well get the idle threads to do something
            tid = z * Ny * Nx + y * Nx + x
            if junctional[x, y, z]:
                for i in range(4):
                    lcc_random_nums[x, y, z, i] = xoroshiro128p_uniform_float32(
                        rng_states, tid
                    )


@cuda.jit
def update_RyR_and_conc_3d_tau_leap(
    ci: npt.NDArray[f32],
    cs: npt.NDArray[f32],
    cp: npt.NDArray[f32],
    cnsr: npt.NDArray[f32],
    cjsr: npt.NDArray[f32],
    CaTi: npt.NDArray[f32],
    CaTs: npt.NDArray[f32],
    RyR: npt.NDArray[f32],
    LCC: npt.NDArray[i32],
    RyR_open: npt.NDArray[f32],
    LCC_probs: npt.NDArray[f32],
    ICa: npt.NDArray[f32],
    INaCa: npt.NDArray[f32],
    Delta_ci: npt.NDArray[f32],
    Delta_cnsr: npt.NDArray[f32],
    Delta_cs: npt.NDArray[f32],
    RyR_rates: npt.NDArray[f32],
    lcc_random_nums: npt.NDArray[f32],
    rng_states: RNG_states,
    vp: npt.NDArray[f32],
    junctional: npt.NDArray[np.bool_],
    params: RestrepoParams,
    consts: RestrepoConsts,
):
    # x, y, z = cuda.grid(3)
    Nx, Ny, Nz = cs.shape
    blocks_per_task = cuda.gridDim.x // 3
    if cuda.blockIdx.x < blocks_per_task:
        x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
        y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
        task = 0
    elif cuda.blockIdx.x < 2 * blocks_per_task:
        x = cuda.threadIdx.x + (cuda.blockIdx.x - blocks_per_task) * cuda.blockDim.x
        y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
        task = 1
    else:
        x = cuda.threadIdx.x + (cuda.blockIdx.x - 2 * blocks_per_task) * cuda.blockDim.x
        y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
        z = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z
        task = 2

    if x < Nx and y < Ny and z < Nz:
        dt = consts.dt[0]
        if task == 0:
            ryr_open = RyR_open[x, y, z]
            ci_tmp = ci[x, y, z]
            cnsr_tmp = cnsr[x, y, z]
            cjsr_tmp = cjsr[x, y, z]
            cs_tmp = cs[x, y, z]
            cp_tmp = cp[x, y, z]
            CaTi_tmp = CaTi[x, y, z]
            CaTs_tmp = CaTs[x, y, z]
            vp_tmp = vp[x, y, z]
            is_junctional = junctional[x, y, z]

            ITCi = ITCa(ci_tmp, CaTi_tmp, params.kon[0], params.koff[0], params.BT[0])
            ITCs = ITCa(cs_tmp, CaTs_tmp, params.kon[0], params.koff[0], params.BT[0])
            Ileak_ = Ileak(cnsr_tmp, ci_tmp, params.gleak[0], square(params.Kjsr[0]))
            Iup_ = Iup(
                ci_tmp,
                cnsr_tmp,
                params.Ki[0],
                params.Knsr[0],
                params.vup[0],
                params.H[0],
            )
            # Ir_ = Ir(cp_tmp, cjsr_tmp, ryr_open, params.Jmax[0], vp_tmp)
            Ir_scaled = params.Jmax[0] * ryr_open * (cjsr_tmp - cp_tmp) / params.vjsr[0]
            Ici = Delta_ci[x, y, z]
            Icnsr = Delta_cnsr[x, y, z]
            Ics = Delta_cs[x, y, z]

            Itr = (cnsr_tmp - cjsr_tmp) / params.tau_tr[0]
            Idsi = (cs_tmp - ci_tmp) / params.tau_si[0]
            Idps_scaled = (
                (cp_tmp - cs_tmp) * vp_tmp / (params.tau_ps[0] * params.vs[0])
            )  # Idps * (vp/vs)

            INaCa_ = INaCa[x, y, z] if is_junctional else float32(0.0)
            ICa_ = ICa[x, y, z] if is_junctional else float32(0.0)

            calmodulin_buf_i = (
                params.KCAM[0] * params.BCAM[0] / square(params.KCAM[0] + ci_tmp)
            )
            SR_buf = params.KSR[0] * params.BSR[0] / square(params.KSR[0] + ci_tmp)
            myosin_Ca_buf = (
                params.KMCa[0] * params.BMCa[0] / square(params.KMCa[0] + ci_tmp)
            )
            myosin_Mg_buf = (
                params.KMMg[0] * params.BMMg[0] / square(params.KMMg[0] + ci_tmp)
            )

            calmodulin_buf_s = (
                params.KCAM[0] * params.BCAM[0] / square(params.KCAM[0] + cs_tmp)
            )
            SLH_buf = params.KSLH[0] * params.BSLH[0] / square(params.KSLH[0] + cs_tmp)

            beta_i = float32(1.0) / (
                float32(1.0) + calmodulin_buf_i + SR_buf + myosin_Ca_buf + myosin_Mg_buf
            )
            beta_s = float32(1.0) / (float32(1.0) + calmodulin_buf_s + SLH_buf)
            beta_jsr = luminal_buffer(
                cjsr_tmp,
                params.rho_inf[0],
                params.K[0],
                params.BCSQN[0],
                params.nM[0],
                params.nD[0],
                params.KC[0],
                params.h[0],
            )

            kr = params.Jmax[0] / vp_tmp
            cp[x, y, z] = (
                cs_tmp + params.tau_ps[0] * (kr * ryr_open * cjsr_tmp - ICa_)
            ) / (float32(1.0) + params.tau_ps[0] * kr * ryr_open)
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
            cjsr[x, y, z] += dt * beta_jsr * (Itr - Ir_scaled)
            CaTi[x, y, z] += dt * ITCi
            CaTs[x, y, z] += dt * ITCs
        elif task == 1:
            is_junctional = junctional[x, y, z]
            for j in range(4):
                u = lcc_random_nums[x, y, z, j]
                LCC[x, y, z, j] = sample_LCC_icdf_3d(
                    LCC_probs, LCC[x, y, z, j], u, x, y, z, j, is_junctional
                )
        elif task == 2:
            # Euler-Maruyama step for RyRs
            tid = z * Nx * Ny + y * Nx + x
            update_RyR_tau_leap_3d(RyR, RyR_rates, dt, rng_states, x, y, z, tid)
