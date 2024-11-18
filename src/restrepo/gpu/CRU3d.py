from typing import Any
import math

import numpy as np
import numpy.typing as npt
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from params import RestrepoParams
from .utils import constants_struct_array, truncated_normal
from .kernels3d import update_RyR_and_conc_3d, update_currents_and_lcc_3d
from .LCC import calculate_V_dep_LCC_params

f32 = np.float32
i32 = np.int32
RNG_states = Any
floating = np.floating | float


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
                1 + truncated_normal(0.3, -0.8, 0.8, ci_init.shape[0], ci_init.shape[1])
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
        self.d_LCC: npt.NDArray[i32] | None = None
        self.d_ci: npt.NDArray[f32] | None = None
        self.d_cnsr: npt.NDArray[f32] | None = None
        self.d_cjsr: npt.NDArray[f32] | None = None
        self.d_cs: npt.NDArray[f32] | None = None
        self.d_cp: npt.NDArray[f32] | None = None
        self.d_CaTi: npt.NDArray[f32] | None = None
        self.d_CaTs: npt.NDArray[f32] | None = None

        # Additional device memory to allocate
        self.rng_states: RNG_states | None = None
        self.dW: npt.NDArray[f32] | None = None
        self.RyR_rates: npt.NDArray[f32] | None = None
        self.RyR_sorted: npt.NDArray[f32] | None = None
        self.LCC_probs: npt.NDArray[f32] | None = None
        self.ICa: npt.NDArray[f32] | None = None
        self.INaCa: npt.NDArray[f32] | None = None
        self.Delta_ci: npt.NDArray[f32] | None = None
        self.Delta_cnsr: npt.NDArray[f32] | None = None
        self.Delta_cs: npt.NDArray[f32] | None = None

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

        # self.stream_kern = cuda.stream()

        # Variables holding simulation output
        self.t: npt.NDArray[np.floating] | None = None
        self.V: npt.NDArray[np.floating] | None = None
        self.ci_result: npt.NDArray[f32] | None = None
        self.cs_result: npt.NDArray[f32] | None = None
        self.cp_result: npt.NDArray[f32] | None = None
        self.cnsr_result: npt.NDArray[f32] | None = None
        self.cjsr_result: npt.NDArray[f32] | None = None
        self.CaTi_result: npt.NDArray[f32] | None = None
        self.CaTs_result: npt.NDArray[f32] | None = None
        self.RyR_result: npt.NDArray[f32] | None = None
        self.LCC_result: npt.NDArray[f32] | None = None

    def _init_memory(self, seed: int) -> None:
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
        self.RyR_rates = cuda.device_array((*self.ci.shape, 8), dtype=np.float32)
        self.RyR_sorted = cuda.device_array_like(self.RyR)
        self.LCC_probs = cuda.device_array(
            (*self.LCC.shape, 7), dtype=np.float32  # type: ignore
        )

        self.ICa = cuda.device_array_like(self.d_cp)  # type: ignore
        self.INaCa = cuda.device_array_like(self.ICa)
        self.Delta_ci = cuda.device_array_like(self.ci)
        self.Delta_cnsr = cuda.device_array_like(self.cnsr)
        self.Delta_cs = cuda.device_array_like(self.cs)

        self.rng_states = create_xoroshiro128p_states(
            self.RyR.shape[0] * self.RyR.shape[1] * self.RyR.shape[2], seed=seed
        )
        # self.rng_states_LCC = create_xoroshiro128p_states(
        #    self.d_LCC.shape[0], seed=seed2  # type: ignore
        # )
        cuda.synchronize()
        self._memory_initialised = True

    def _calc_V(
        self, Vmin: floating, Vmax: floating, T: floating, xT: floating
    ) -> floating:
        t_mod_T = self._t % T
        return (
            np.float32(Vmin + (Vmax - Vmin) * np.sqrt(1.0 - (t_mod_T / xT) ** 2))
            if t_mod_T < xT
            else np.float32(Vmin)
        )

    def _init_results(self, nstep: int):
        # Initialise result arrays
        self.t = np.zeros(nstep + 1)
        self.V = np.zeros(nstep + 1)
        self.ci_result = np.zeros((nstep + 1, *self.ci.shape), dtype=self.ci.dtype)
        self.cs_result = np.zeros((nstep + 1, *self.cs.shape), dtype=self.cs.dtype)
        self.cp_result = np.zeros((nstep + 1, *self.cp.shape), dtype=self.cp.dtype)
        self.cnsr_result = np.zeros(
            (nstep + 1, *self.cnsr.shape), dtype=self.cnsr.dtype
        )
        self.cjsr_result = np.zeros(
            (nstep + 1, *self.cjsr.shape), dtype=self.cjsr.dtype
        )
        self.CaTi_result = np.zeros(
            (nstep + 1, *self.CaTi.shape), dtype=self.CaTi.dtype
        )
        self.CaTs_result = np.zeros(
            (nstep + 1, *self.CaTs.shape), dtype=self.CaTs.dtype
        )
        self.RyR_result = np.zeros((nstep + 1, *self.RyR.shape), dtype=self.RyR.dtype)
        self.LCC_result = np.zeros((nstep + 1, *self.LCC.shape), dtype=self.LCC.dtype)

        self.t[0] = self._t
        self.ci_result[0, ...] = self.ci
        self.cs_result[0, ...] = self.cs
        self.cp_result[0, ...] = self.cp
        self.cnsr_result[0, ...] = self.cnsr
        self.cjsr_result[0, ...] = self.cjsr
        self.CaTi_result[0, ...] = self.CaTi
        self.CaTs_result[0, ...] = self.CaTs
        self.RyR_result[0, ...] = self.RyR
        self.LCC_result[0, ...] = self.LCC

    def _copy_state(self, i: int) -> None:
        self.t[i] = self._t  # type: ignore
        self.V[i] = self._V  # type: ignore
        self.ci_result[i, ...] = self.d_ci.copy_to_host(stream=self.stream_ci)  # type: ignore
        self.cs_result[i, ...] = self.d_cs.copy_to_host(stream=self.stream_cs)  # type: ignore
        self.cp_result[i, ...] = self.d_cp.copy_to_host(stream=self.stream_cp)  # type: ignore
        self.cnsr_result[i, ...] = self.d_cnsr.copy_to_host(stream=self.stream_cnsr)  # type: ignore
        self.cjsr_result[i, ...] = self.d_cjsr.copy_to_host(stream=self.stream_cjsr)  # type: ignore
        self.CaTi_result[i, ...] = self.d_CaTi.copy_to_host(stream=self.stream_CaTi)  # type: ignore
        self.CaTs_result[i, ...] = self.d_CaTs.copy_to_host(stream=self.stream_CaTs)  # type: ignore
        self.RyR_result[i, ...] = self.d_RyR.copy_to_host(stream=self.stream_RyR)  # type: ignore
        self.LCC_result[i, ...] = self.d_LCC.copy_to_host(stream=self.stream_LCC)  # type: ignore

    def _call_kernels(
        self,
        Vmin: floating,
        Vmax: floating,
        T: floating,
        xT: floating,
        bpg: tuple[int, int, int],
        tpb: tuple[int, int, int],
    ) -> None:
        self._V = self._calc_V(Vmin, Vmax, T, xT)  # type: ignore
        alpha, beta, k3, k5_, k6_, Pr, Ps, R = calculate_V_dep_LCC_params(
            self._V, self.params, single_precision=True
        )
        bpg_kern1 = (2 * bpg[0], bpg[1], bpg[2])  # Execute 2 jobs in different blocks
        update_currents_and_lcc_3d[bpg_kern1, tpb](
            self.d_RyR,
            self.d_LCC,
            self.d_ci,
            self.d_cs,
            self.d_cjsr,
            self.d_cnsr,
            self.d_cp,
            self.RyR_rates,
            self.LCC_probs,
            self.Delta_ci,
            self.Delta_cnsr,
            self.Delta_cs,
            self.ICa,
            self.INaCa,
            self.d_junctional,
            self.dW,
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
        update_RyR_and_conc_3d[bpg, tpb](
            self.d_ci,
            self.d_cs,
            self.d_cp,
            self.d_cnsr,
            self.d_cjsr,
            self.d_CaTi,
            self.d_CaTs,
            self.d_RyR,
            self.RyR_sorted,
            self.ICa,
            self.INaCa,
            self.Delta_ci,
            self.Delta_cnsr,
            self.Delta_cs,
            self.RyR_rates,
            self.dW,
            self.d_vp,
            self.d_junctional,
            self.d_params,
            self.d_consts,
        )

    def forward(
        self,
        dt: np.floating,
        T: floating = 400.0,  # Pacing period
        Vmin: floating = -80.0,
        Vmax: floating = 15.0,
        nstep: int = 1,
        collect_every: int = 1,
        seed: int = 0,
        tpb: tuple[int, int, int] = (16, 2, 2),
        reset_t: bool = False,
        profile: bool = False,
    ) -> None:
        if not self._memory_initialised:
            self._init_memory(seed)

        sqrtdt = np.sqrt(dt, dtype=np.float32)
        # T should be in seconds here
        Nai3 = np.float32(78.0 / (1.0 + 10.0 * np.sqrt(T / 1000))) ** 3
        self.d_consts = cuda.to_device(
            constants_struct_array(np.float32(dt), sqrtdt, Nai3)
        )

        bpg = (
            math.ceil(self.ci.shape[0] / tpb[0]),
            math.ceil(self.ci.shape[1] / tpb[1]),
            math.ceil(self.ci.shape[2] / tpb[2]),
        )

        # Not mentioned in the paper, but T has to be in seconds here.
        xT = np.float32(1000 * T * (2 / 3) / (1000 * (2 / 3) + T))
        if reset_t:
            self._t = np.float32(0.0)

        ncollect = nstep // collect_every
        self._init_results(ncollect)

        if profile:
            cuda.profile_start()

        for i in range(1, nstep + 1):
            if i % collect_every == 0:
                idx = i // collect_every
                with cuda.pinned(
                    self.ci_result[idx, ...],  # type: ignore
                    self.cs_result[idx, ...],  # type: ignore
                    self.cp_result[idx, ...],  # type: ignore
                    self.cnsr_result[idx, ...],  # type: ignore
                    self.cjsr_result[idx, ...],  # type: ignore
                    self.CaTi_result[idx, ...],  # type: ignore
                    self.CaTs_result[idx, ...],  # type: ignore
                    self.RyR_result[idx, ...],  # type: ignore
                    self.LCC_result[idx, ...],  # type: ignore
                ):
                    self._call_kernels(
                        Vmin,
                        Vmax,
                        T,
                        xT,
                        bpg,
                        tpb,
                    )
                    self._copy_state(idx)
            else:
                self._call_kernels(Vmin, Vmax, T, xT, bpg, tpb)
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
