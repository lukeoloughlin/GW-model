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


class CRUSolution:

    def __init__(self, params: RestrepoParams, nstep: int, Nx: int, Ny: int, Nz: int):
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

    @property
    def Itr(self):
        return (self.cnsr - self.cjsr) / self.params.tau_tr

    @property
    def Ileak(self):
        cnsr2 = self.cnsr**2
        return (
            self.params.gleak
            * cnsr2
            / (cnsr2 + self.params.Knsr**2)
            * (self.cnsr - self.ci)
        )

    @property
    def Iup(self):
        ci_Ki_H = (self.ci / self.params.Ki) ** self.params.H
        cnsr_Knsr_H = (self.cnsr / self.params.Knsr) ** self.params.H
        return self.params.vup * (ci_Ki_H - cnsr_Knsr_H) / (1 + ci_Ki_H + cnsr_Knsr_H)

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
        self.RyR_open = cuda.device_array((*self.ci.shape,), dtype=np.float32)
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

    @staticmethod
    def V_fn(
        t: floating, Vmin: floating, Vmax: floating, T: floating, xT: floating
    ) -> floating:
        t_mod_T = t % T
        return (
            np.float32(Vmin + (Vmax - Vmin) * np.sqrt(1.0 - (t_mod_T / xT) ** 2))
            if t_mod_T < xT
            else np.float32(Vmin)
        )

    def _init_results(self, nstep: int) -> CRUSolution:
        # Initialise result arrays
        sol = CRUSolution(self.params, nstep, *self.cs.shape)

        sol.t[0] = self._t
        sol.V[0] = self._V
        sol.ci[0, ...] = self.ci
        sol.cs[0, ...] = self.cs
        sol.cp[0, ...] = self.cp
        sol.cnsr[0, ...] = self.cnsr
        sol.cjsr[0, ...] = self.cjsr
        sol.CaTi[0, ...] = self.CaTi
        sol.CaTs[0, ...] = self.CaTs
        sol.RyR[0, ...] = self.RyR
        sol.LCC[0, ...] = self.LCC
        return sol

    def _copy_state(self, i: int, sol: CRUSolution) -> None:
        sol.t[i] = self._t  # type: ignore
        sol.V[i] = self._V  # type: ignore
        sol.ci[i, ...] = self.d_ci.copy_to_host(stream=self.stream_ci)  # type: ignore
        sol.cs[i, ...] = self.d_cs.copy_to_host(stream=self.stream_cs)  # type: ignore
        sol.cp[i, ...] = self.d_cp.copy_to_host(stream=self.stream_cp)  # type: ignore
        sol.cnsr[i, ...] = self.d_cnsr.copy_to_host(stream=self.stream_cnsr)  # type: ignore
        sol.cjsr[i, ...] = self.d_cjsr.copy_to_host(stream=self.stream_cjsr)  # type: ignore
        sol.CaTi[i, ...] = self.d_CaTi.copy_to_host(stream=self.stream_CaTi)  # type: ignore
        sol.CaTs[i, ...] = self.d_CaTs.copy_to_host(stream=self.stream_CaTs)  # type: ignore
        sol.RyR[i, ...] = self.d_RyR.copy_to_host(stream=self.stream_RyR)  # type: ignore
        sol.LCC[i, ...] = self.d_LCC.copy_to_host(stream=self.stream_LCC)  # type: ignore

    def _call_kernels(
        self,
        Vmin: floating,
        Vmax: floating,
        T: floating,
        xT: floating,
        bpg: tuple[int, int, int],
        tpb: tuple[int, int, int],
    ) -> None:
        self._V = self.V_fn(self._t, Vmin, Vmax, T, xT)  # type: ignore
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
            self.d_vp,
            self.d_junctional,
            self.rng_states,
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
    ) -> CRUSolution:
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
        self._V = self.V_fn(self._t, Vmin, Vmax, T, xT)  # type: ignore
        sol = self._init_results(ncollect)

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
                    self._call_kernels(
                        Vmin,
                        Vmax,
                        T,
                        xT,
                        bpg,
                        tpb,
                    )
                    self._copy_state(idx, sol)
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
        return sol
