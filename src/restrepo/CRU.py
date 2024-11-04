import math

import numpy as np
import numpy.typing as npt
from numba import float32
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states

from RyR import RyR_kernel
from LCC import LCC_kernel


DEFAULT_LCC_PARAMS = {
    "k0p": 3.0,
    "cp_bar": 1.5,
    "cp_tilde": 0.5,
    "tau_po": 1.0,
    "r1": 0.3,
    "r2": 6.0,
    "s1_": 0.00195,
    "k1_": 0.00413,
    "k2": 0.0001,
    "k2_": 0.00224,
    "TBa": 450.0,
}

DEFAULT_RYR_PARMS = {
    "Ku": 3.8e-4,
    "Kb": 5e-5,
    "tau_u": 125.0,
    "tau_b": 5.0,
    "tau_c": 1.0,
    "BCSQN": 400.0,
    "rho_inf": 5000.0,
    "K": 850.0,
}


def _calculate_V_dep_LCC_params(V: float, tau_po: float, TBa: float):
    po_inf = 1.0 / (1.0 + np.exp(-V / 8))
    Pr = 1.0 / (1.0 + np.exp(-(V + 40.0) / 4.0))
    Ps = 1.0 / (1.0 + np.exp(-(V + 40.0) / 11.32))
    R = 10.0 + 4954.0 * np.exp(V / 15.6)
    tauBa = (R - TBa) * Pr + TBa

    alpha = po_inf / tau_po
    beta = (1.0 - po_inf) / tau_po
    k3 = np.exp(-(V + 40.0) / 3.0) / (3.0 * (1.0 + np.exp(-(V + 40.0) / 3.0)))
    k5_ = (1.0 - Ps) / tauBa
    k6_ = Ps / tauBa

    return alpha, beta, k3, k5_, k6_, Pr, Ps, R


def RyR_stationary(
    cp: npt.NDArray,
    K=DEFAULT_RYR_PARMS["K"],
    rho_inf=DEFAULT_RYR_PARMS["rho_inf"],
    BCSQN=DEFAULT_RYR_PARMS["BCSQN"],
    Ku=DEFAULT_RYR_PARMS["Ku"],
    Kb=DEFAULT_RYR_PARMS["Kb"],
    tau_b=DEFAULT_RYR_PARMS["tau_b"],
    tau_u=DEFAULT_RYR_PARMS["tau_u"],
    tau_c=DEFAULT_RYR_PARMS["tau_c"],
):
    log_cp_K = np.log(cp) - np.log(K)
    hill_fn = rho_inf / (1.0 + np.exp(23.0 * log_cp_K))
    Mhat = (np.sqrt(1.0 + 8.0 * hill_fn * BCSQN) - 1.0) / (4.0 * hill_fn * BCSQN)

    k12 = Ku * cp**2  # k12
    k21 = 1 / tau_c
    k23 = Mhat / tau_b  # k23
    k34 = 1 / tau_c
    k43 = Kb * cp**2  # k43
    k32 = k12 / (tau_u * k43)  # k32 = k41 * k12 / k43

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
    tau_po: float = DEFAULT_LCC_PARAMS["tau_po"],
    TBa: float = DEFAULT_LCC_PARAMS["TBa"],
    cp_bar: float = DEFAULT_LCC_PARAMS["cp_bar"],
    cp_tilde: float = DEFAULT_LCC_PARAMS["cp_tilde"],
    k2: float = DEFAULT_LCC_PARAMS["k2"],
    k1_: float = DEFAULT_LCC_PARAMS["k1_"],
    k2_: float = DEFAULT_LCC_PARAMS["k2_"],
    r1: float = DEFAULT_LCC_PARAMS["r1"],
    r2: float = DEFAULT_LCC_PARAMS["r2"],
):
    alpha, beta, k3, k5_, k6_, Pr, Ps, R = _calculate_V_dep_LCC_params(V, tau_po, TBa)

    TCa = (78.0329 + 0.1 * (1 + cp / cp_bar) ** 4) / (1.0 + (cp / cp_bar) ** 4)

    k1 = 0.03 / (1.0 + (cp_tilde / cp) ** 3)
    tauCa = (R - TCa) * Pr + TCa
    k5 = (1.0 - Ps) / tauCa
    k6 = Ps / (tauCa * (1.0 + (cp_bar / cp_tilde) ** 3))

    piC1_un = alpha / beta
    piI2Ca_un = k6 / k5
    piI2Ba_un = k6_ / k5_
    piI1Ca_un = (k1 / k2) * piC1_un
    piI1Ba_un = (k1_ / k2_) * piC1_un
    piO_un = (r1 / r2) * piC1_un

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

    def __init__(self, RyR_init, LCC_init, cp_init):

        self.k0p = np.float32(DEFAULT_LCC_PARAMS["k0p"])
        self.cp_bar = np.float32(DEFAULT_LCC_PARAMS["cp_bar"])
        self.cp_tilde = np.float32(DEFAULT_LCC_PARAMS["cp_tilde"])
        self.tau_po = np.float32(DEFAULT_LCC_PARAMS["tau_po"])
        self.r1 = np.float32(DEFAULT_LCC_PARAMS["r1"])
        self.r2 = np.float32(DEFAULT_LCC_PARAMS["r2"])
        self.s1_ = np.float32(DEFAULT_LCC_PARAMS["s1_"])
        self.k1_ = np.float32(DEFAULT_LCC_PARAMS["k1_"])
        self.k2 = np.float32(DEFAULT_LCC_PARAMS["k2"])
        self.k2_ = np.float32(DEFAULT_LCC_PARAMS["k2_"])
        self.TBa = np.float32(DEFAULT_LCC_PARAMS["TBa"])

        self.Ku = np.float32(DEFAULT_RYR_PARMS["Ku"])
        self.Kb = np.float32(DEFAULT_RYR_PARMS["Kb"])
        self.tau_u = np.float32(DEFAULT_RYR_PARMS["tau_u"])
        self.tau_b = np.float32(DEFAULT_RYR_PARMS["tau_b"])
        self.tau_c = np.float32(DEFAULT_RYR_PARMS["tau_c"])
        self.BCSQN = np.float32(DEFAULT_RYR_PARMS["BCSQN"])
        self.rho_inf = np.float32(DEFAULT_RYR_PARMS["rho_inf"])
        self.K = np.float32(DEFAULT_RYR_PARMS["K"])

        self._1_tau_u = 1 / self.tau_u
        self._1_tau_b = 1 / self.tau_b
        self._1_tau_c = 1 / self.tau_c

        self.RyR = RyR_init.astype(np.float32)
        self.LCC = LCC_init.astype(np.int32)
        self.cp = cp_init.astype(np.float32)

        self.d_RyR = None
        self.d_LCC = None
        self.d_cp = None
        self.rng_states = None
        self.dW = None
        self.RyR_rates = None
        self.RyR_sorted = None
        self.LCC_probs = None

        self._memory_initialised = False
        self._deallocated = False

    @staticmethod
    def _mangle_LCC(LCC):
        """Get the boundary elements of the LCC array"""
        out = np.zeros((2 * LCC.shape[0] + 2 * (LCC.shape[1] - 2), 4), dtype=np.int32)
        out[: LCC.shape[0]] = LCC[0, ...]  # Top row
        out[LCC.shape[0] : 2 * LCC.shape[0]] = LCC[-1, ...]  # bottom row
        out[2 * LCC.shape[0] : (2 * LCC.shape[0] + LCC.shape[1] - 2)] = LCC[
            0, 1:-1, :
        ]  # left column without corners
        out[(2 * LCC.shape[0] + LCC.shape[1] - 2) :] = LCC[
            -1, 1:-1, :
        ]  # right column without corners
        return out

    @staticmethod
    def _demangle_LCC(LCC, mLCC):
        LCC[0, ...] = mLCC[: LCC.shape[0]]
        LCC[-1, ...] = mLCC[LCC.shape[0] : 2 * LCC.shape[0]]
        LCC[-1, 1:-1, :] = mLCC[
            2 * LCC.shape[0] : (2 * LCC.shape[0] + LCC.shape[1] - 2)
        ]
        LCC[-1, 1:-1, :] = mLCC[(2 * LCC.shape[0] + LCC.shape[1] - 2) :]

    def init_memory(self, seed):
        self.d_RyR = cuda.to_device(self.RyR)
        self.d_LCC = cuda.to_device(self._mangle_LCC(self.LCC))
        self.d_cp = cuda.to_device(self.cp)

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

        self.rng_states = create_xoroshiro128p_states(
            self.RyR.shape[0] * self.RyR.shape[1], seed=seed
        )

        self._memory_initialised = True

    def forward(self, V: float, dt: float, nstep=1, seed=0, tpb_RyR=16):
        if not self._memory_initialised:
            self.init_memory(seed)

        if self._deallocated:
            self._reallocate()

        alpha, beta, k3, k5_, k6_, Pr, Ps, R = _calculate_V_dep_LCC_params(
            V, self.tau_po, self.TBa
        )

        sqrtdt = np.sqrt(dt, dtype=np.float32)

        bpg_RyR = (
            math.ceil(self.RyR.shape[0] / tpb_RyR),
            math.ceil(self.RyR.shape[1] / tpb_RyR),
        )
        for _ in range(nstep):
            RyR_kernel[bpg_RyR, (tpb_RyR, tpb_RyR)](
                self.d_RyR,
                self.RyR_rates,
                self.RyR_sorted,
                self.d_cp,
                self.dW,
                np.float32(0.1),
                np.float32(dt),
                sqrtdt,
                self.Ku,
                self.Kb,
                self._1_tau_u,
                self._1_tau_b,
                self._1_tau_c,
                self.BCSQN,
                self.rho_inf,
                self.K,
                self.rng_states,
            )

            LCC_kernel.forall(self.d_LCC.shape[0])(
                self.d_LCC,
                self.LCC_probs,
                self.d_cp,
                self.rng_states,
                np.float32(dt),
                alpha,
                beta,
                self.r1,
                self.r2,
                self.s1_,
                self.cp_bar,
                self.cp_tilde,
                self.k1_,
                self.k2,
                self.k2_,
                k3,
                k3,  # k3_ = k3
                k5_,
                k6_,
                Pr,
                Ps,
                R,
            )
        cuda.synchronize()

        self.d_RyR.copy_to_host(self.RyR)
        mangled_lcc = self.d_LCC.copy_to_host()
        self._demangle_LCC(self.LCC, mangled_lcc)

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
