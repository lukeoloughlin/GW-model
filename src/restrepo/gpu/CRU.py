import math

import numpy as np
import numpy.typing as npt
from numba import float32
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from params import RestrepoParams
from .utils import constants_struct_array
from .kernels import (
    currents_and_RyR,
    update_boundary_currents_and_LCC,
    update_RyR_and_euler_step,
)


def _calculate_V_dep_LCC_params(V: float, params: RestrepoParams):
    po_inf = 1.0 / (1.0 + np.exp(-V / 8))
    Pr = 1.0 / (1.0 + np.exp(-(V + 40.0) / 4.0))
    Ps = 1.0 / (1.0 + np.exp(-(V + 40.0) / 11.32))
    R = 10.0 + 4954.0 * np.exp(V / 15.6)
    tauBa = (R - params.TBa) * Pr + params.TBa

    alpha = po_inf / params.tau_po
    beta = (1.0 - po_inf) / params.tau_po
    k3 = np.exp(-(V + 40.0) / 3.0) / (3.0 * (1.0 + np.exp(-(V + 40.0) / 3.0)))
    k5_ = (1.0 - Ps) / tauBa
    k6_ = Ps / tauBa

    return alpha, beta, k3, k5_, k6_, Pr, Ps, R


def RyR_stationary(
    cp: npt.NDArray,
    params: RestrepoParams,
):
    log_cp_K = np.log(cp) - np.log(params.K)
    hill_fn = params.rho_inf / (1.0 + np.exp(23.0 * log_cp_K))
    Mhat = (np.sqrt(1.0 + 8.0 * hill_fn * params.BCSQN) - 1.0) / (
        4.0 * hill_fn * params.BCSQN
    )

    k12 = params.Ku * cp**2  # k12
    k21 = 1 / params.tau_c
    k23 = Mhat / params.tau_b  # k23
    k34 = 1 / params.tau_c
    k43 = params.Kb * cp**2  # k43
    k32 = k12 / (params.tau_u * k43)  # k32 = k41 * k12 / k43

    pi2_un = k12 / k21
    pi3_un = (k23 / k32) * pi2_un
    pi4_un = (k34 / k43) * pi3_un

    norm = 1.0 + pi2_un + pi3_un + pi4_un

    RyR = np.zeros((*cp.shape, 4))
    RyR[..., 0] = 1.0 / norm
    RyR[..., 1] = pi2_un / norm
    RyR[..., 2] = pi3_un / norm
    RyR[..., 3] = pi4_un / norm

    return RyR


def LCC_stationary(
    cp: npt.NDArray,
    V: float,
    params: RestrepoParams,
):
    alpha, beta, k3, k5_, k6_, Pr, Ps, R = _calculate_V_dep_LCC_params(V, params)

    TCa = (78.0329 + 0.1 * (1 + cp / params.cp_bar) ** 4) / (
        1.0 + (cp / params.cp_bar) ** 4
    )

    k1 = 0.03 / (1.0 + (params.cp_tilde / cp) ** 3)
    tauCa = (R - TCa) * Pr + TCa
    k5 = (1.0 - Ps) / tauCa
    k6 = Ps / (tauCa * (1.0 + (params.cp_bar / params.cp_tilde) ** 3))

    piC1_un = alpha / beta
    piI2Ca_un = k6 / k5
    piI2Ba_un = k6_ / k5_
    piI1Ca_un = (k1 / params.k2) * piC1_un
    piI1Ba_un = (params.k1_ / params.k2_) * piC1_un
    piO_un = (params.r1 / params.r2) * piC1_un

    pi_un = np.zeros((*cp.shape, 7))
    pi_un[..., 0] = piC1_un
    pi_un[..., 1] = 1.0
    pi_un[..., 2] = piI1Ca_un
    pi_un[..., 3] = piI2Ca_un
    pi_un[..., 4] = piI1Ba_un
    pi_un[..., 5] = piI2Ba_un
    pi_un[..., 6] = piO_un
    out = np.zeros((*cp.shape, 4), dtype=np.int32)

    ind = [1, 2, 3, 4, 5, 6, 7]
    for i in range(cp.shape[0]):
        out[0, i, :] = np.random.choice(
            ind, p=(pi_un[0, i, :] / pi_un[0, i, :].sum()), size=(4,)
        )
        out[-1, i, :] = np.random.choice(
            ind, p=(pi_un[-1, i, :] / pi_un[-1, i, :].sum()), size=(4,)
        )

    for i in range(1, cp.shape[1] - 1):
        out[i, 0, :] = np.random.choice(
            ind, p=(pi_un[i, 0, :] / pi_un[i, 0, :].sum()), size=(4,)
        )
        out[i, -1, :] = np.random.choice(
            ind, p=(pi_un[i, -1, :] / pi_un[i, -1, :].sum()), size=(4,)
        )

    return out


class CRUs:

    def __init__(
        self,
        params: RestrepoParams,
        RyR_init: npt.NDArray,
        LCC_init: npt.NDArray,
        ci_init: npt.NDArray,
        cnsr_init: npt.NDArray,
        cjsr_init: npt.NDArray,
        cs_init: npt.NDArray,
        cp_init: npt.NDArray,
        CaTi_init: npt.NDArray,
        CaTs_init: npt.NDArray,
    ):

        self.params = params

        self.RyR = RyR_init.astype(np.float32)
        self.LCC = LCC_init.astype(np.int32)
        self.ci = ci_init.astype(np.float32)
        self.cnsr = cnsr_init.astype(np.float32)
        self.cjsr = cjsr_init.astype(np.float32)
        self.cs = cs_init.astype(np.float32)
        self.cp = cp_init.astype(np.float32)
        self.CaTi = CaTi_init.astype(np.float32)
        self.CaTs = CaTs_init.astype(np.float32)

        self.d_params = None
        self.d_consts = None
        self.d_RyR = None
        self.d_LCC = None
        self.d_ci = None
        self.d_cnsr = None
        self.d_cjsr = None
        self.d_cs = None
        self.d_cp = None
        self.d_CaTi = None
        self.d_CaTs = None

        self.rng_states = None
        self.dW = None
        self.RyR_rates = None
        self.RyR_sorted = None
        self.LCC_probs = None
        self.ICa = None
        self.INaCa = None
        self.Delta_ci = None
        self.Delta_cnsr = None
        self.sum_cs_nn = None

        self._t: float = np.float32(0.0)
        self._memory_initialised = False
        self._deallocated = False

    @staticmethod
    def _mangle_LCC(LCC):
        """Get the boundary elements of the LCC array"""
        Nx, Ny, _ = LCC.shape
        out = np.zeros((2 * Nx + 2 * (Ny - 2), 4), dtype=np.int32)
        out[:Ny] = LCC[0, ...]  # Top row
        out[Ny : 2 * Ny] = LCC[-1, ...]  # bottom row
        out[2 * Ny : (2 * Ny + Nx - 2)] = LCC[1:-1, 0, :]  # left column without corners
        out[(2 * Ny + Nx - 2) :] = LCC[1:-1, -1, :]  # right column without corners
        return out

    @staticmethod
    def _demangle_LCC(LCC, mLCC):
        Nx, Ny, _ = LCC.shape
        LCC[0, ...] = mLCC[:Ny]
        LCC[-1, ...] = mLCC[Ny : 2 * Ny]
        LCC[1:-1, 0, :] = mLCC[2 * Ny : (2 * Ny + Nx - 2)]
        LCC[1:-1, -1, :] = mLCC[(2 * Ny + Nx - 2) :]

    def init_memory(self, seed1, seed2):
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

        self.d_RyR = cuda.to_device(self.RyR)
        self.d_LCC = cuda.to_device(self._mangle_LCC(self.LCC))
        self.d_ci = cuda.to_device(self.ci)
        self.d_cnsr = cuda.to_device(self.cnsr)
        self.d_cjsr = cuda.to_device(self.cjsr)
        self.d_cs = cuda.to_device(self.cs)
        self.d_cp = cuda.to_device(self.cp)
        self.d_CaTi = cuda.to_device(self.CaTi)
        self.d_CaTs = cuda.to_device(self.CaTs)

        self.dW = cuda.device_array(
            (self.RyR.shape[0], self.RyR.shape[1], 4), dtype=np.float32
        )
        self.RyR_rates = cuda.device_array(
            (self.RyR.shape[0], self.RyR.shape[1], 8), dtype=np.float32
        )
        self.RyR_sorted = cuda.device_array_like(self.RyR)
        self.LCC_probs = cuda.device_array(
            (self.d_LCC.shape[0], 4, 7), dtype=np.float32
        )

        self.ICa = cuda.device_array((self.d_LCC.shape[0],), dtype=np.float32)
        self.INaCa = cuda.device_array_like(self.ICa)
        self.Delta_ci = cuda.device_array_like(self.ci)
        self.Delta_cnsr = cuda.device_array_like(self.cnsr)
        self.sum_cs_nn = cuda.device_array_like(self.cs)

        self.rng_states_dW = create_xoroshiro128p_states(
            self.RyR.shape[0] * self.RyR.shape[1], seed=seed1
        )
        self.rng_states_LCC = create_xoroshiro128p_states(
            self.d_LCC.shape[0], seed=seed2
        )
        cuda.synchronize()
        self._memory_initialised = True

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

        self.stream_kern1 = cuda.stream()
        self.stream_kern2 = cuda.stream()

    def _calc_V(self, Vmin: float, Vmax: float, T: float, xT: float) -> float:
        t_mod_T = self._t % T
        return (
            np.float32(Vmin + (Vmax - Vmin) * np.sqrt(1.0 - (t_mod_T / xT) ** 2))
            if t_mod_T < xT
            else np.float32(Vmin)
        )

    def _init_results(self, nstep: int):
        # Initialise result arrays
        self.t = np.zeros(nstep + 1)
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

    def _copy_state(self, i: int):
        self.t[i] = self._t
        self.ci_result[i, ...] = self.d_ci.copy_to_host(stream=self.stream_ci)
        self.cs_result[i, ...] = self.d_cs.copy_to_host(stream=self.stream_cs)
        self.cp_result[i, ...] = self.d_cp.copy_to_host(stream=self.stream_cp)
        self.cnsr_result[i, ...] = self.d_cnsr.copy_to_host(stream=self.stream_cnsr)
        self.cjsr_result[i, ...] = self.d_cjsr.copy_to_host(stream=self.stream_cjsr)
        self.CaTi_result[i, ...] = self.d_CaTi.copy_to_host(stream=self.stream_CaTi)
        self.CaTs_result[i, ...] = self.d_CaTs.copy_to_host(stream=self.stream_CaTs)
        self.RyR_result[i, ...] = self.d_RyR.copy_to_host(stream=self.stream_RyR)
        self._demangle_LCC(
            self.LCC_result[i, ...],
            self.d_LCC.copy_to_host(stream=self.stream_LCC),
        )

    def _call_kernels(
        self,
        Vmin: float,
        Vmax: float,
        T: float,
        xT: float,
        bpg_2d,
        tpb_2d,
        bpg_1d,
        tpb_1d,
        sync_LCC_stream: bool,
    ):
        V = self._calc_V(Vmin, Vmax, T, xT)
        z = np.float32(V * 96.5 / (8.314 * 308.0))
        alpha, beta, k3, k5_, k6_, Pr, Ps, R = _calculate_V_dep_LCC_params(
            V, self.params
        )
        currents_and_RyR[bpg_2d, tpb_2d, self.stream_kern1](
            self.d_RyR,
            self.RyR_rates,
            self.d_ci,
            self.d_cs,
            self.d_cjsr,
            self.d_cnsr,
            self.d_cp,
            self.Delta_ci,
            self.Delta_cnsr,
            self.sum_cs_nn,
            self.dW,
            self.rng_states_dW,
            self.d_consts,
            self.d_params,
        )
        if sync_LCC_stream:
            # Must make sure copy has finished before updating LCC
            self.stream_LCC.synchronize()
        update_boundary_currents_and_LCC[bpg_1d, tpb_1d, self.stream_kern2](
            self.d_LCC,
            self.LCC_probs,
            self.d_cp,
            self.d_cs,
            self.ICa,
            self.INaCa,
            self.rng_states_LCC,
            alpha,
            beta,
            k3,
            k3,
            k5_,
            k6_,
            Pr,
            Ps,
            R,
            z,
            self.d_consts,
            self.d_params,
        )
        # execute on default stream which waits for the previous two to finish
        update_RyR_and_euler_step[bpg_2d, tpb_2d](
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
            self.sum_cs_nn,
            self.RyR_rates,
            self.dW,
            self.d_consts,
            self.d_params,
        )

    def forward(
        self,
        dt: float,
        T: float = 400.0,  # Pacing period
        Vmin: float = -80.0,
        Vmax: float = 15.0,
        nstep: int = 1,
        collect_every: int = 1,
        seed1: int = 0,
        seed2: int = 12345,
        _tpb2d: int = 8,
        _tpb1d: int = 16,
        reset_t: bool = False,
        profile: bool = False,
    ):
        if not self._memory_initialised:
            self.init_memory(seed1, seed2)

        if self._deallocated:
            self._reallocate()

        sqrtdt = np.sqrt(dt, dtype=np.float32)
        # T should be in seconds here
        Nai3 = np.float32(78.0 / (1.0 + 10.0 * np.sqrt(T / 1000))) ** 3
        self.d_consts = cuda.to_device(
            constants_struct_array(np.float32(dt), sqrtdt, Nai3)
        )

        bpg_2d = (
            math.ceil(self.ci.shape[0] / _tpb2d),
            math.ceil(self.ci.shape[1] / _tpb2d),
        )
        tpb_2d = (_tpb2d, _tpb2d)

        bpg_1d = math.ceil(self.d_LCC.shape[0] / _tpb1d)
        tpb_1d = _tpb1d

        # Not mentioned in the paper, but T has to be in seconds here.
        xT = np.float32(1000 * T * (2 / 3) / (1000 * (2 / 3) + T))
        if reset_t:
            self._t = np.float32(0.0)

        ncollect = nstep // collect_every
        self._init_results(ncollect)

        if profile:
            cuda.profile_start()

        for i in range(1, nstep + 1):
            sync_LCC_stream = ((i - 1) % collect_every) == 0
            if i % collect_every == 0:
                idx = i // collect_every
                with cuda.pinned(
                    self.ci_result[idx, ...],
                    self.cs_result[idx, ...],
                    self.cp_result[idx, ...],
                    self.cnsr_result[idx, ...],
                    self.cjsr_result[idx, ...],
                    self.CaTi_result[idx, ...],
                    self.CaTs_result[idx, ...],
                    self.RyR_result[idx, ...],
                    self.LCC_result[idx, ...],
                ):
                    self._call_kernels(
                        Vmin,
                        Vmax,
                        T,
                        xT,
                        bpg_2d,
                        tpb_2d,
                        bpg_1d,
                        tpb_1d,
                        sync_LCC_stream,
                    )
                    self._copy_state(idx)
            else:
                self._call_kernels(
                    Vmin, Vmax, T, xT, bpg_2d, tpb_2d, bpg_1d, tpb_1d, sync_LCC_stream
                )
            self._t += dt

        if profile:
            cuda.profile_stop()

        self.d_ci.copy_to_host(self.ci, stream=self.stream_ci)
        self.d_cnsr.copy_to_host(self.cnsr, stream=self.stream_cnsr)
        self.d_cjsr.copy_to_host(self.cjsr, stream=self.stream_cjsr)
        self.d_cs.copy_to_host(self.cs, stream=self.stream_cs)
        self.d_cp.copy_to_host(self.cp, stream=self.stream_cp)
        self.d_CaTi.copy_to_host(self.CaTi, stream=self.stream_CaTi)
        self.d_CaTs.copy_to_host(self.CaTs, stream=self.stream_CaTs)

        self.d_RyR.copy_to_host(self.RyR, stream=self.stream_RyR)
        mangled_lcc = self.d_LCC.copy_to_host(stream=self.stream_LCC)
        self._demangle_LCC(self.LCC, mangled_lcc)

        cuda.synchronize()

    def free_gpu(self):
        if self._memory_initialised:
            self.d_RyR = self.d_RyR.copy_to_host()
            self.d_LCC = self.d_LCC.copy_to_host()
            self.d_cp = self.d_cp.copy_to_host()
            self.rng_states = self.rng_states.copy_to_host()

            self.dW = self.dW.copy_to_host()
            self.RyR_rates = self.RyR_rates.copy_to_host()
            self.RyR_sorted = self.RyR_sorted.copy_to_host()
            self.LCC_probs = self.LCC_probs.copy_to_host()

            self._deallocated = True

    def _reallocate(self):
        if self._deallocated:
            self.d_RyR = cuda.to_device(self.d_RyR)
            self.d_LCC = cuda.to_device(self.d_LCC)
            self.d_cp = cuda.to_device(self.d_cp)
            self.rng_states = cuda.to_device(self.rng_states)

            self.dW = cuda.to_device(self.dW)
            self.RyR_rates = cuda.to_device(self.RyR_rates)
            self.RyR_sorted = cuda.to_device(self.RyR_sorted)
            self.LCC_probs = cuda.to_device(self.LCC_probs)

            self._deallocated = False
