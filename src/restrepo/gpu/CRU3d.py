from typing import Any
import math

import numpy as np
import numpy.typing as npt
from numba import cuda, njit
from numba.cuda.random import create_xoroshiro128p_states

from params import RestrepoParams
from .utils import constants_struct_array, truncated_normal
from .kernels3d import *
from .LCC import calculate_V_dep_LCC_params

f32 = np.float32
i32 = np.int32
RNG_states = Any
floating = f32 | float


@njit
def _ICa(
    V: npt.NDArray,
    cp: npt.NDArray,
    LCC: npt.NDArray,
    PCa: float,
    gamma: float,
    Cao: float,
) -> npt.NDArray:
    """Jitted calculation of ICa to bypass slow np.where call"""
    out = np.zeros_like(cp)
    Nt, Nx, Ny, Nz = cp.shape
    for i in range(Nt):
        VF_RT = V[i] * 96.5 / (8.314 * 308.0)
        exp2VF_RT = np.exp(2 * VF_RT)
        for j in range(Nx):
            for k in range(Ny):
                for l in range(Nz):
                    nopen = np.sum(LCC[i, j, k, l, :] == 7)
                    cp_mM = 1e-3 * cp[i, j, k, l]
                    if np.abs(VF_RT) > 0.01:
                        out[i, j, k, l] = (
                            4.0
                            * nopen
                            * PCa
                            * VF_RT
                            * 96.5
                            * gamma
                            * (cp_mM * exp2VF_RT - Cao)
                            / (exp2VF_RT - 1)
                        )
                    else:
                        out[i, j, k, l] = (
                            2.0
                            * nopen
                            * PCa
                            * 96.5
                            * gamma
                            * (cp_mM * exp2VF_RT - Cao)
                            / (1.0 + VF_RT)
                        )
    return out


@njit
def _sample_RyR_int(RyR_props: npt.NDArray[f32]) -> npt.NDArray[i32]:
    out = np.zeros(RyR_props.shape, dtype=np.int32)
    Nx, Ny, Nz, _ = RyR_props.shape
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                N = 100
                out[i, j, k, 0] = np.random.binomial(N, RyR_props[i, j, k, 0])
                N = N - out[i, j, k, 0]
                out[i, j, k, 1] = np.random.binomial(N, RyR_props[i, j, k, 1])
                N = N - out[i, j, k, 1]
                out[i, j, k, 2] = np.random.binomial(N, RyR_props[i, j, k, 2])
                N = N - out[i, j, k, 2]
                out[i, j, k, 3] = N
    return out


class CRUSolution:

    def __init__(
        self,
        params: RestrepoParams,
        nstep: int,
        Nx: int,
        Ny: int,
        Nz: int,
        Nai: floating,
    ):
        self.params = params
        self.t = np.zeros(nstep + 1)
        self.V = np.zeros(nstep + 1)
        self.ci = np.zeros((nstep + 1, Nx, Ny, Nz), dtype=np.float32)
        self.cs = np.zeros((nstep + 1, Nx, Ny, Nz), dtype=np.float32)
        self.cp = np.zeros((nstep + 1, Nx, Ny, Nz), dtype=np.float32)
        self.cnsr = np.zeros((nstep + 1, Nx, Ny, Nz), dtype=np.float32)
        self.cjsr = np.zeros((nstep + 1, Nx, Ny, Nz), dtype=np.float32)
        self.CaTi = np.zeros((nstep + 1, Nx, Ny, Nz), dtype=np.float32)
        self.CaTs = np.zeros((nstep + 1, Nx, Ny, Nz), dtype=np.float32)
        self.RyR = np.zeros((nstep + 1, Nx, Ny, Nz, 4), dtype=np.float32)
        self.LCC = np.zeros((nstep + 1, Nx, Ny, Nz, 4), dtype=np.int32)
        self.junctional = np.zeros((Nx, Ny, Nz), dtype=np.bool_)

        self.Nai = Nai

        self.__Itr: npt.NDArray | None = None
        self.__Ileak: npt.NDArray | None = None
        self.__Iup: npt.NDArray | None = None
        self.__ICa: npt.NDArray | None = None
        self.__INCX: npt.NDArray | None = None

    def Itr(self, cache=False):
        if self.__Itr is not None:
            return self.__Itr
        Itr_ = (self.cnsr - self.cjsr) / self.params.tau_tr
        if cache:
            self.__Itr = Itr_
        return Itr_

    def Ileak(self, cache=False):
        if self.__Ileak is not None:
            return self.__Ileak
        cnsr2 = self.cnsr**2
        Ileak_ = (
            self.params.gleak
            * cnsr2
            / (cnsr2 + self.params.Kjsr**2)
            * (self.cnsr - self.ci)
        )
        if cache:
            self.__Ileak = Ileak_
        return Ileak_

    def Iup(self, cache=False):
        if self.__Iup is not None:
            return self.__Iup
        ci_Ki_H = (self.ci / self.params.Ki) ** self.params.H
        cnsr_Knsr_H = (self.cnsr / self.params.Knsr) ** self.params.H
        Iup_ = self.params.vup * (ci_Ki_H - cnsr_Knsr_H) / (1 + ci_Ki_H + cnsr_Knsr_H)
        if cache:
            self.__Iup = Iup_
        return Iup_

    def ICa(self, cache=False):
        if self.__ICa is not None:
            return self.__ICa
        ICa_ = _ICa(
            self.V,
            self.cp,
            self.LCC,
            self.params.PCa,
            self.params.gamma,
            self.params.Cao,
        )
        if cache:
            self.__ICa = ICa_
        return ICa_

    def INCX(self, cache=False):
        if self.__INCX is not None:
            return self.__INCX

        VF_RT = (self.V * 96.5 / (8.314 * 308.0)).reshape(-1, 1, 1, 1)
        Nai3 = self.Nai**3
        Nao3 = self.params.Nao**3
        KmNao3 = self.params.KmNao**3
        KmNai3 = self.params.KmNai**3
        Ka = 1.0 / (1.0 + (self.params.Kda / self.cs) ** 3)

        cs_mM = self.cs * 1e-3  # convert cs to mM
        t1 = self.params.KmCai * Nao3 * (1.0 + Nai3 / KmNai3)
        t2 = KmNao3 * cs_mM * (1.0 + (cs_mM / self.params.KmCai))
        t3 = self.params.KmCao * Nai3 + Nai3 * self.params.Cao + Nao3 * cs_mM

        exp_etaz = np.exp(self.params.eta * VF_RT)
        exp_etam1z = np.exp((self.params.eta - 1.0) * VF_RT)

        INCX_ = (
            self.junctional[None, ...]
            * self.params.vNaCa
            * Ka
            * (exp_etaz * Nai3 * self.params.Cao - exp_etam1z * Nao3 * cs_mM)
            / ((t1 + t2 + t3) * (1 + self.params.ksat * exp_etam1z))
        )
        if cache:
            self.__INCX = INCX_
        return INCX_

    def line_scan(self, y: int, z: int):
        centre = self.cp[..., y, z]
        left = self.cp[..., y - 1, z]
        right = self.cp[..., y + 1, z]
        up = self.cp[..., y, z + 1]
        down = self.cp[..., y, z - 1]
        left_up = self.cp[..., y - 1, z + 1]
        left_down = self.cp[..., y - 1, z - 1]
        right_up = self.cp[..., y + 1, z + 1]
        right_down = self.cp[..., y + 1, z - 1]

        return (
            centre
            + left
            + right
            + up
            + down
            + left_up
            + left_down
            + right_up
            + right_down
        ) / 9.0


class CRU3D:

    def __init__(
        self,
        params: RestrepoParams,
        RyR_init: npt.NDArray[np.floating],
        LCC_init: npt.NDArray[np.integer],
        ci_init: npt.NDArray[np.floating],
        cnsr_init: npt.NDArray[np.floating],
        cjsr_init: npt.NDArray[np.floating],
        cs_init: npt.NDArray[np.floating],
        cp_init: npt.NDArray[np.floating],
        CaTi_init: npt.NDArray[np.floating],
        CaTs_init: npt.NDArray[np.floating],
        vp: npt.NDArray[np.floating] | None = None,
        junctional: npt.NDArray[np.bool_] | None = None,
        rng_seed: int = 0,
    ):

        self.params: RestrepoParams = params

        self.RyR = RyR_init.astype(np.float32)
        self.LCC = LCC_init.astype(np.int32)
        self.ci = ci_init.astype(np.float32)
        self.cnsr = cnsr_init.astype(np.float32)
        self.cjsr = cjsr_init.astype(np.float32)
        self.cs = cs_init.astype(np.float32)
        self.cp = cp_init.astype(np.float32)
        self.CaTi = CaTi_init.astype(np.float32)
        self.CaTs = CaTs_init.astype(np.float32)
        if vp is None:
            self.vp = params.vp * (
                1
                + truncated_normal(
                    0.3, -0.8, 0.8, ci_init.shape[0], ci_init.shape[1], ci_init.shape[2]
                )
            ).astype(np.float32)
        else:
            self.vp = vp.astype(np.float32)
        if junctional is None:
            self.junctional = np.zeros((*ci_init.shape,), dtype=bool)
            self.junctional[0, ...] = True
            self.junctional[-1, ...] = True
            self.junctional[:, 0, :] = True
            self.junctional[:, -1, :] = True
            self.junctional[..., 0] = True
            self.junctional[..., -1] = True
        else:
            self.junctional = junctional

        # Device array to preallocate
        self.d_params: npt.ArrayLike | None = None
        self.d_vp: npt.NDArray[f32] | None = None
        self.d_consts: npt.ArrayLike | None = None
        self.d_RyR: npt.NDArray[f32] | None = None
        self.d_RyR_int: npt.NDArray[i32] | None = None
        self.d_LCC: npt.NDArray[i32] | None = None
        self.d_ci: npt.NDArray[f32] | None = None
        self.d_cnsr: npt.NDArray[f32] | None = None
        self.d_cjsr: npt.NDArray[f32] | None = None
        self.d_cs: npt.NDArray[f32] | None = None
        self.d_cp: npt.NDArray[f32] | None = None
        self.d_CaTi: npt.NDArray[f32] | None = None
        self.d_CaTs: npt.NDArray[f32] | None = None

        # Additional device memory to allocate
        self.dW: npt.NDArray[f32] | None = None
        self.d_lcc_rns: npt.NDArray[f32] | None = None
        self.RyR_open: npt.NDArray[f32] | None = None
        self.RyR_rates: npt.NDArray[f32] | None = None
        self.RyR_sorted: npt.NDArray[f32] | None = None
        self.LCC_probs: npt.NDArray[f32] | None = None
        self.ICa: npt.NDArray[f32] | None = None
        self.INaCa: npt.NDArray[f32] | None = None
        self.Delta_ci: npt.NDArray[f32] | None = None
        self.Delta_cnsr: npt.NDArray[f32] | None = None
        self.Delta_cs: npt.NDArray[f32] | None = None
        self.d_junctional: npt.NDArray[np.bool_] | None = None

        self.rng_states = create_xoroshiro128p_states(
            self.RyR.shape[0] * self.RyR.shape[1] * self.RyR.shape[2], seed=rng_seed
        )

        self._t: f32 = f32(0.0)
        self._V: f32 = f32(0.0)
        self._memory_initialised: bool = False
        self._deallocated: bool = False

        # create streams
        self.stream_ci = cuda.stream()
        self.stream_cs = cuda.stream()
        self.stream_cp = cuda.stream()
        self.stream_cnsr = cuda.stream()
        self.stream_cjsr = cuda.stream()
        self.stream_CaTi = cuda.stream()
        self.stream_CaTs = cuda.stream()
        self.stream_RyR = cuda.stream()
        self.stream_LCC = cuda.stream()
        self.stream_kern = cuda.stream()

    @staticmethod
    def V_fn(
        t: floating,
        Vmin: floating,
        Vmax: floating,
        T: floating | None,
        delay_pulse_by: float = 0.0,
    ) -> f32:
        if T is None:
            return Vmin
        else:
            if t < delay_pulse_by:
                return Vmin
            else:
                xT = 1000 * T * (2 / 3) / (1000 * (2 / 3) + T)
                t_mod_T = (t - delay_pulse_by) % T
                return np.float32(
                    Vmin + (Vmax - Vmin) * np.sqrt(1.0 - (t_mod_T / xT) ** 2)
                    if t_mod_T < xT
                    else Vmin
                )

    def _init_results(self, nstep: int, Nai: floating, tau_leap: bool) -> CRUSolution:
        # Initialise result arrays
        sol = CRUSolution(self.params, nstep, *self.cs.shape, Nai)  # type: ignore

        sol.t[0] = self._t
        sol.V[0] = self._V
        sol.ci[0, ...] = self.ci
        sol.cs[0, ...] = self.cs
        sol.cp[0, ...] = self.cp
        sol.cnsr[0, ...] = self.cnsr
        sol.cjsr[0, ...] = self.cjsr
        sol.CaTi[0, ...] = self.CaTi
        sol.CaTs[0, ...] = self.CaTs
        sol.LCC[0, ...] = self.LCC
        if tau_leap:
            sol.RyR[0, ...] = 0.01 * self.RyR.astype(np.float32)
        else:
            sol.RyR[0, ...] = self.RyR
        sol.junctional = self.junctional
        return sol

    def forward(
        self,
        dt: np.floating,
        T: floating | None = 400.0,  # Pacing period
        Vmin: floating = -80.0,
        Vmax: floating = 15.0,
        Nai: floating | None = None,
        nstep: int = 1,
        collect_every: int = 1,
        tpb: tuple[int, int, int] = (16, 2, 2),
        reset_t: bool = False,
        profile: bool = False,
        delay_pulse_by: float = 0.0,
        Po_shift: float = 0.0,
    ) -> CRUSolution:
        self._check_memory(tau_leap=False)

        sqrtdt = np.sqrt(dt, dtype=np.float32)
        if T is None:
            assert Nai is not None
        else:
            # T should be in seconds here
            Nai = 78.0 / (1.0 + 10.0 * np.sqrt(T / 1000))

        Nai3 = np.float32(Nai**3)
        Vmin = np.float32(Vmin)
        Vmax = np.float32(Vmax)
        Po_shift = np.float32(Po_shift)

        self.d_consts = cuda.to_device(
            constants_struct_array(np.float32(dt), sqrtdt, Nai3)
        )

        bpg = (
            math.ceil(self.ci.shape[0] / tpb[0]),
            math.ceil(self.ci.shape[1] / tpb[1]),
            math.ceil(self.ci.shape[2] / tpb[2]),
        )

        if reset_t:
            self._t = np.float32(0.0)

        ncollect = nstep // collect_every
        self._V = self.V_fn(self._t, Vmin, Vmax, T, delay_pulse_by)
        sol = self._init_results(ncollect, Nai, tau_leap=False)

        if profile:
            cuda.profile_start()

        for i in range(1, nstep + 1):
            if i % collect_every == 0:
                idx = i // collect_every
                with cuda.pinned(
                    sol.ci[idx, ...],  # type: ignore
                    sol.cs[idx, ...],  # type: ignore
                    sol.cp[idx, ...],  # type: ignore
                    sol.cnsr[idx, ...],  # type: ignore
                    sol.cjsr[idx, ...],  # type: ignore
                    sol.CaTi[idx, ...],  # type: ignore
                    sol.CaTs[idx, ...],  # type: ignore
                    sol.RyR[idx, ...],  # type: ignore
                    sol.LCC[idx, ...],  # type: ignore
                ):
                    self._call_kernels_diffusion(
                        Vmin, Vmax, T, bpg, tpb, delay_pulse_by, Po_shift
                    )
                    self._copy_state(idx, sol, tau_leap=False)
            else:
                self._call_kernels_diffusion(
                    Vmin, Vmax, T, bpg, tpb, delay_pulse_by, Po_shift
                )
            self._t += dt

        if profile:
            cuda.profile_stop()

        self.d_ci.copy_to_host(self.ci, stream=self.stream_ci)  # type: ignore
        self.d_cnsr.copy_to_host(self.cnsr, stream=self.stream_cnsr)  # type: ignore
        self.d_cjsr.copy_to_host(self.cjsr, stream=self.stream_cjsr)  # type: ignore
        self.d_cs.copy_to_host(self.cs, stream=self.stream_cs)  # type: ignore
        self.d_cp.copy_to_host(self.cp, stream=self.stream_cp)  # type: ignore
        self.d_CaTi.copy_to_host(self.CaTi, stream=self.stream_CaTi)  # type: ignore
        self.d_CaTs.copy_to_host(self.CaTs, stream=self.stream_CaTs)  # type: ignore

        self.d_RyR.copy_to_host(self.RyR, stream=self.stream_RyR)  # type: ignore
        self.d_LCC.copy_to_host(self.LCC, stream=self.stream_LCC)  # type: ignore

        cuda.synchronize()
        return sol

    def forward_tau_leap(
        self,
        dt: np.floating,
        T: floating | None = 400.0,  # Pacing period
        Vmin: floating = -80.0,
        Vmax: floating = 15.0,
        Nai: floating | None = None,
        nstep: int = 1,
        collect_every: int = 1,
        tpb: tuple[int, int, int] = (16, 2, 2),
        reset_t: bool = False,
        profile: bool = False,
        delay_pulse_by: float = 0.0,
    ) -> CRUSolution:
        self._check_memory(tau_leap=True)

        if T is None:
            assert Nai is not None
        else:
            # T should be in seconds here
            Nai = 78.0 / (1.0 + 10.0 * np.sqrt(T / 1000))

        Nai3 = np.float32(Nai**3)
        Vmin = np.float32(Vmin)
        Vmax = np.float32(Vmax)

        self.d_consts = cuda.to_device(
            constants_struct_array(np.float32(dt), f32(0.0), Nai3)
        )

        bpg = (
            math.ceil(self.ci.shape[0] / tpb[0]),
            math.ceil(self.ci.shape[1] / tpb[1]),
            math.ceil(self.ci.shape[2] / tpb[2]),
        )

        if reset_t:
            self._t = np.float32(0.0)

        ncollect = nstep // collect_every
        self._V = self.V_fn(self._t, Vmin, Vmax, T, delay_pulse_by)
        sol = self._init_results(ncollect, Nai, tau_leap=True)

        if profile:
            cuda.profile_start()

        for i in range(1, nstep + 1):
            if i % collect_every == 0:
                idx = i // collect_every
                with cuda.pinned(
                    sol.ci[idx, ...],  # type: ignore
                    sol.cs[idx, ...],  # type: ignore
                    sol.cp[idx, ...],  # type: ignore
                    sol.cnsr[idx, ...],  # type: ignore
                    sol.cjsr[idx, ...],  # type: ignore
                    sol.CaTi[idx, ...],  # type: ignore
                    sol.CaTs[idx, ...],  # type: ignore
                    sol.RyR[idx, ...],  # type: ignore
                    sol.LCC[idx, ...],  # type: ignore
                ):
                    self._call_kernels_tau_leap(Vmin, Vmax, T, bpg, tpb, delay_pulse_by)
                    self._copy_state(idx, sol, tau_leap=True)
            else:
                self._call_kernels_tau_leap(Vmin, Vmax, T, bpg, tpb, delay_pulse_by)
            self._t += dt

        if profile:
            cuda.profile_stop()

        self.d_ci.copy_to_host(self.ci, stream=self.stream_ci)  # type: ignore
        self.d_cnsr.copy_to_host(self.cnsr, stream=self.stream_cnsr)  # type: ignore
        self.d_cjsr.copy_to_host(self.cjsr, stream=self.stream_cjsr)  # type: ignore
        self.d_cs.copy_to_host(self.cs, stream=self.stream_cs)  # type: ignore
        self.d_cp.copy_to_host(self.cp, stream=self.stream_cp)  # type: ignore
        self.d_CaTi.copy_to_host(self.CaTi, stream=self.stream_CaTi)  # type: ignore
        self.d_CaTs.copy_to_host(self.CaTs, stream=self.stream_CaTs)  # type: ignore

        self.d_RyR_int.copy_to_host(self.RyR, stream=self.stream_RyR)  # type: ignore
        self.d_LCC.copy_to_host(self.LCC, stream=self.stream_LCC)  # type: ignore

        cuda.synchronize()
        return sol

    def _call_kernels_diffusion(
        self,
        Vmin: f32,
        Vmax: f32,
        T: f32 | None,
        bpg: tuple[int, int, int],
        tpb: tuple[int, int, int],
        delay_pulse_by: float,
        Po_shift: f32,
    ) -> None:
        self._V = self.V_fn(self._t, Vmin, Vmax, T, delay_pulse_by)
        alpha, beta, k3, k5_, k6_, Pr, Ps, R = calculate_V_dep_LCC_params(
            self._V, self.params, single_precision=True
        )
        bpg_kern1 = (3 * bpg[0], bpg[1], bpg[2])  # Execute 3 jobs in different blocks
        bpg_kern2 = (3 * bpg[0], bpg[1], bpg[2])  # Execute 3 jobs in different blocks
        update_currents_and_lcc_3d[bpg_kern1, tpb, self.stream_kern](
            self.d_RyR,
            self.d_LCC,
            self.d_ci,
            self.d_cs,
            self.d_cjsr,
            self.d_cnsr,
            self.d_cp,
            self.RyR_rates,
            self.RyR_open,
            self.LCC_probs,
            self.Delta_ci,
            self.Delta_cnsr,
            self.Delta_cs,
            self.ICa,
            self.INaCa,
            self.d_junctional,
            self.dW,
            self.d_lcc_rns,
            self.rng_states,
            alpha,
            beta,
            k3,
            k3,
            k5_,
            k6_,
            Pr,
            Ps,
            R,
            self._V,
            self.d_params,
            self.d_consts,
        )
        update_RyR_and_conc_3d[bpg_kern2, tpb](
            self.d_ci,
            self.d_cs,
            self.d_cp,
            self.d_cnsr,
            self.d_cjsr,
            self.d_CaTi,
            self.d_CaTs,
            self.d_RyR,
            self.d_LCC,
            self.RyR_open,
            self.RyR_sorted,
            self.LCC_probs,
            self.ICa,
            self.INaCa,
            self.Delta_ci,
            self.Delta_cnsr,
            self.Delta_cs,
            self.RyR_rates,
            self.dW,
            self.d_lcc_rns,
            self.d_vp,
            self.d_junctional,
            Po_shift,
            self.d_params,
            self.d_consts,
        )

    def _call_kernels_tau_leap(
        self,
        Vmin: f32,
        Vmax: f32,
        T: f32 | None,
        bpg: tuple[int, int, int],
        tpb: tuple[int, int, int],
        delay_pulse_by: float,
    ) -> None:
        if T is not None:
            self._V = self.V_fn(self._t, Vmin, Vmax, T, delay_pulse_by)
        else:
            self._V = Vmin
        alpha, beta, k3, k5_, k6_, Pr, Ps, R = calculate_V_dep_LCC_params(
            self._V, self.params, single_precision=True
        )
        bpg_kern1 = (3 * bpg[0], bpg[1], bpg[2])  # Execute 3 jobs in different blocks
        bpg_kern2 = (3 * bpg[0], bpg[1], bpg[2])  # Execute 3 jobs in different blocks
        update_currents_and_lcc_3d_tau_leap[bpg_kern1, tpb, self.stream_kern](
            self.d_RyR_int,
            self.d_LCC,
            self.d_ci,
            self.d_cs,
            self.d_cjsr,
            self.d_cnsr,
            self.d_cp,
            self.RyR_rates,
            self.RyR_open,
            self.LCC_probs,
            self.Delta_ci,
            self.Delta_cnsr,
            self.Delta_cs,
            self.ICa,
            self.INaCa,
            self.d_junctional,
            self.d_lcc_rns,
            self.rng_states,
            alpha,
            beta,
            k3,
            k3,
            k5_,
            k6_,
            Pr,
            Ps,
            R,
            self._V,
            self.d_params,
            self.d_consts,
        )
        update_RyR_and_conc_3d_tau_leap[bpg_kern2, tpb](
            self.d_ci,
            self.d_cs,
            self.d_cp,
            self.d_cnsr,
            self.d_cjsr,
            self.d_CaTi,
            self.d_CaTs,
            self.d_RyR_int,
            self.d_LCC,
            self.RyR_open,
            self.LCC_probs,
            self.ICa,
            self.INaCa,
            self.Delta_ci,
            self.Delta_cnsr,
            self.Delta_cs,
            self.RyR_rates,
            self.d_lcc_rns,
            self.rng_states,
            self.d_vp,
            self.d_junctional,
            self.d_params,
            self.d_consts,
        )

    def _copy_state(self, i: int, sol: CRUSolution, tau_leap: bool) -> None:
        sol.t[i] = self._t  # type: ignore
        sol.V[i] = self._V  # type: ignore
        sol.ci[i, ...] = self.d_ci.copy_to_host(stream=self.stream_ci)  # type: ignore
        sol.cs[i, ...] = self.d_cs.copy_to_host(stream=self.stream_cs)  # type: ignore
        sol.cp[i, ...] = self.d_cp.copy_to_host(stream=self.stream_cp)  # type: ignore
        sol.cnsr[i, ...] = self.d_cnsr.copy_to_host(stream=self.stream_cnsr)  # type: ignore
        sol.cjsr[i, ...] = self.d_cjsr.copy_to_host(stream=self.stream_cjsr)  # type: ignore
        sol.CaTi[i, ...] = self.d_CaTi.copy_to_host(stream=self.stream_CaTi)  # type: ignore
        sol.CaTs[i, ...] = self.d_CaTs.copy_to_host(stream=self.stream_CaTs)  # type: ignore
        sol.LCC[i, ...] = self.d_LCC.copy_to_host(stream=self.stream_LCC)  # type: ignore
        if tau_leap:
            sol.RyR[i, ...] = 0.01 * self.d_RyR_int.copy_to_host(stream=self.stream_RyR).astype(np.float32)  # type: ignore
        else:
            sol.RyR[i, ...] = self.d_RyR.copy_to_host(stream=self.stream_RyR)  # type: ignore

    def _check_memory(self, tau_leap: bool) -> None:
        if not self._memory_initialised:
            self._init_memory()

        if tau_leap and self.RyR.dtype == np.float32:
            self._switch_to_tau_leap()
        elif not tau_leap and self.RyR.dtype == np.int32:
            self._switch_to_diffusion()

    def _init_memory(self) -> None:
        # Convert namedtuple to numpy record array and allocate on device to avoid copying
        # Do this shit to make the array aligned in memory
        offsets = 4 * np.arange(len(self.params))
        dtype = np.dtype(
            (
                np.record,
                dict(
                    names=self.params._fields,
                    formats=[np.dtype("float32")] * len(self.params),
                    offsets=offsets,
                    itemsize=offsets[-1] + 4,
                ),
            ),
            align=True,
        )
        params_array = np.rec.array(tuple(self.params), dtype=dtype, aligned=True)[None]
        self.d_params = cuda.to_device(params_array)
        self.d_vp = cuda.to_device(self.vp)
        self.d_junctional = cuda.to_device(self.junctional)

        self.d_RyR = cuda.to_device(self.RyR)
        self.d_LCC = cuda.to_device(self.LCC)
        self.d_ci = cuda.to_device(self.ci)
        self.d_cnsr = cuda.to_device(self.cnsr)
        self.d_cjsr = cuda.to_device(self.cjsr)
        self.d_cs = cuda.to_device(self.cs)
        self.d_cp = cuda.to_device(self.cp)
        self.d_CaTi = cuda.to_device(self.CaTi)
        self.d_CaTs = cuda.to_device(self.CaTs)

        self.dW = cuda.device_array((*self.RyR.shape,), dtype=np.float32)
        self.RyR_open = cuda.device_array((*self.ci.shape,), dtype=np.float32)
        self.RyR_rates = cuda.device_array((*self.ci.shape, 8), dtype=np.float32)
        self.RyR_sorted = cuda.device_array_like(self.RyR)
        self.LCC_probs = cuda.device_array(
            (*self.LCC.shape, 7), dtype=np.float32  # type: ignore
        )
        self.d_lcc_rns = cuda.device_array(self.LCC.shape, dtype=np.float32)

        self.ICa = cuda.device_array_like(self.d_cp)  # type: ignore
        self.INaCa = cuda.device_array_like(self.ICa)
        self.Delta_ci = cuda.device_array_like(self.ci)
        self.Delta_cnsr = cuda.device_array_like(self.cnsr)
        self.Delta_cs = cuda.device_array_like(self.cs)

        self._memory_initialised = True

    def _switch_to_tau_leap(self) -> None:
        """Handle converting RyR to int"""
        self.RyR = _sample_RyR_int(self.RyR)
        # Deallocate these in tau leap mode
        self.d_RyR = None
        self.dW = None
        self.RyR_sorted = None

        # initialise this in tau leap mode
        self.d_RyR_int = cuda.to_device(self.RyR)

    def _switch_to_diffusion(self) -> None:
        """Handle converting RyR to int"""
        self.RyR = self.RyR.astype(np.float32)
        self.RyR *= 0.01

        # Deallocate this in diffusion mode
        self.d_RyR_int = None

        # Deallocate these in tau leap mode
        self.d_RyR = cuda.to_device(self.RyR)
        self.dW = cuda.device_array_like(self.RyR)
        self.RyR_sorted = cuda.device_array_like(self.RyR)
