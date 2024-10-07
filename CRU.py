from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import numba


class Parameters(NamedTuple):
    F: float = 96.5
    R: float = 8.314
    T: float = 310.0
    VSS: float = 0.2303e-6
    VJSR: float = 22.26e-6
    Cao: float = 2.0

    k12: float = 877.5
    k21: float = 250.0
    k23: float = 2.358e8
    k32: float = 9.6
    k34: float = 1.415e6
    k43: float = 13.65
    k45: float = 0.07
    k54: float = 93.385
    k56: float = 1.887e7
    k65: float = 30.0
    k25: float = 2.358e6
    k52: float = 0.001235
    rRyR: float = 3.92

    f: float = 0.85
    g: float = 2.0
    f1: float = 0.005
    g1: float = 7.0
    a: float = 2.0
    b: float = 1.9356
    gamma0: float = 0.44
    omega: float = 0.02158
    PCaL: float = 9.13e-13
    kfClCh: float = 13.3156
    kbClCh: float = 2.0

    rxfer: float = 200.0
    rtr: float = 0.333
    riss: float = 20.0
    BSRT: float = 0.047
    KBSR: float = 0.00087
    BSLT: float = 1.124
    KBSL: float = 0.0087
    CSQNT: float = 13.5
    KCSQN: float = 0.63
    CMDNT: float = 0.05
    KCMDN: float = 0.00238


@numba.njit
def sample_weights(weights: npt.NDArray, total_weight: float) -> int:
    N = weights.shape[0]
    u = np.random.rand()
    cumweight = 0

    i = 0
    for j in range(N):
        cumweight += weights[j]
        if u * total_weight < cumweight:
            return i
        else:
            i += 1

    return -1


@numba.njit
def SSA_1_step(
    CaSS: npt.NDArray,
    LCC: npt.NDArray,
    LCC_i: npt.NDArray,
    RyR: npt.NDArray,
    ClCh: npt.NDArray,
    LCC_rates: npt.NDArray,
    LCC_i_rates: npt.NDArray,
    RyR_rates: npt.NDArray,
    ClCh_rates: npt.NDArray,
    subunit_rates: npt.NDArray,
    LCC_alpha: float,
    LCC_beta: float,
    kf_LCC_i: float,
    kb_LCC_i: float,
    params: Parameters,
) -> float:
    # Update rates:
    for i in range(4):
        # LCC_rates, 3 rates per state, in the order of (forward rate, backward rate, up/down rate)
        gamma = params.gamma0 * CaSS[i]
        if LCC[i] == 1:
            LCC_rates[i, 0] = 4 * LCC_alpha
            LCC_rates[i, 1] = 0.0
            LCC_rates[i, 2] = gamma
        elif LCC[i] == 2:
            LCC_rates[i, 0] = 3 * LCC_alpha
            LCC_rates[i, 1] = LCC_beta
            LCC_rates[i, 2] = params.a * gamma
        elif LCC[i] == 3:
            LCC_rates[i, 0] = 2 * LCC_alpha
            LCC_rates[i, 1] = 2 * LCC_beta
            LCC_rates[i, 2] = (params.a**2) * gamma
        elif LCC[i] == 4:
            LCC_rates[i, 0] = LCC_alpha
            LCC_rates[i, 1] = 3 * LCC_beta
            LCC_rates[i, 2] = (params.a**3) * gamma
        elif LCC[i] == 5:
            LCC_rates[i, 0] = params.f
            LCC_rates[i, 1] = 4 * LCC_beta
            LCC_rates[i, 2] = (params.a**4) * gamma
        elif LCC[i] == 6:
            LCC_rates[i, 0] = 0.0
            LCC_rates[i, 1] = params.g
            LCC_rates[i, 2] = 0.0
        if LCC[i] == 7:
            LCC_rates[i, 0] = 4 * params.a * LCC_alpha
            LCC_rates[i, 1] = 0.0
            LCC_rates[i, 2] = params.omega
        elif LCC[i] == 8:
            LCC_rates[i, 0] = 3 * params.a * LCC_alpha
            LCC_rates[i, 1] = LCC_beta / params.b
            LCC_rates[i, 2] = params.omega / params.b
        elif LCC[i] == 9:
            LCC_rates[i, 0] = 2 * params.a * LCC_alpha
            LCC_rates[i, 1] = 2 * LCC_beta / params.b
            LCC_rates[i, 2] = params.omega / (params.b**2)
        elif LCC[i] == 10:
            LCC_rates[i, 0] = params.a * LCC_alpha
            LCC_rates[i, 1] = 3 * LCC_beta / params.b
            LCC_rates[i, 2] = params.omega / (params.b**3)
        elif LCC[i] == 11:
            LCC_rates[i, 0] = params.f1
            LCC_rates[i, 1] = 4 * LCC_beta / params.b
            LCC_rates[i, 2] = params.omega / (params.b**4)
        else:
            LCC_rates[i, 0] = 0.0
            LCC_rates[i, 1] = params.g1
            LCC_rates[i, 2] = 0.0

        # LCC_i rates
        LCC_i_rates[i] = kf_LCC_i if LCC_i[i] == 0 else kb_LCC_i

        # RyR rates
        CaSS2 = CaSS[i] ** 2
        eq56 = params.k65 / (params.k56 * CaSS2 + params.k65)
        tau34 = 1.0 / (params.k34 * CaSS2 + params.k43)

        RyR_rates[i, 0] = RyR[i, 0] * params.k12 * CaSS2  # 1 -> 2
        RyR_rates[i, 1] = RyR[i, 1] * params.k23 * CaSS2  # 2 -> 3
        RyR_rates[i, 2] = RyR[i, 1] * params.k25 * CaSS2  # 2 -> 5
        RyR_rates[i, 3] = (
            0 if CaSS[i] > 3.685e-2 else RyR[i, 2] * params.k34 * CaSS2
        )  # 3 -> 4
        RyR_rates[i, 4] = (
            (RyR[i, 2] + RyR[i, 3]) * params.k45 * params.k34 * CaSS2 * tau34
            if CaSS[i] > 3.685e-2
            else RyR[i, 3] * params.k45
        )
        # 4 -> 5
        RyR_rates[i, 5] = (
            0 if CaSS[i] > 1.15e-4 else RyR[i, 4] * params.k56 * CaSS2
        )  # 5 -> 6
        RyR_rates[i, 6] = RyR[i, 1] * params.k21  # 2 -> 1
        RyR_rates[i, 7] = (
            (RyR[i, 2] + RyR[i, 3]) * params.k32 * params.k43 * tau34
            if CaSS[i] > 3.685e-2
            else RyR[i, 2] * params.k32
        )  # 3 -> 2
        RyR_rates[i, 8] = 0 if CaSS[i] > 3.685e-2 else RyR[i, 3] * params.k43  # 4 -> 3
        RyR_rates[i, 9] = (
            (RyR[i, 4] + RyR[i, 5]) * params.k52 * eq56
            if CaSS[i] > 1.15e-4
            else RyR[i, 4] * params.k52
        )  # 5 -> 2
        RyR_rates[i, 10] = (
            (RyR[i, 4] + RyR[i, 5]) * params.k54 * CaSS2 * eq56
            if CaSS[i] > 1.15e-4
            else RyR[i, 4] * params.k54 * CaSS2
        )  # 5 -> 4
        RyR_rates[i, 11] = 0 if CaSS[i] > 1.15e-4 else RyR[i, 5] * params.k65  # 6 -> 5

        # ClCh rates
        ClCh_rates[i] = params.kfClCh * CaSS2 if ClCh[i] == 0 else params.kbClCh

        # Total rate of subunit
        subunit_rates[i] = (
            LCC_rates[i, :].sum()
            + LCC_i_rates[i]
            + RyR_rates[i, :].sum()
            + ClCh_rates[i]
        )

    # get total rate and sample a jump time
    total_rate = subunit_rates.sum()
    dt = -np.log(np.random.rand()) / total_rate

    # sample the subunit to be updated
    u = np.random.rand()
    if u < subunit_rates[0] / total_rate:
        su_idx = 0
    elif u < (subunit_rates[0] + subunit_rates[1]) / total_rate:
        su_idx = 1
    elif u < (subunit_rates[0] + subunit_rates[1] + subunit_rates[2]) / total_rate:
        su_idx = 2
    else:
        su_idx = 3

    # sample which ion channel to update
    su_total_rate = subunit_rates[su_idx]
    LCC_su_rate = LCC_rates[su_idx, :].sum()
    LCC_i_su_rate = LCC_i_rates[su_idx]
    RyR_su_rate = RyR_rates[su_idx, :].sum()
    # ClCh_su_rate = ClCh_rates[su_idx]
    u = np.random.rand()
    if u < LCC_su_rate / su_total_rate:
        transition = sample_weights(LCC_rates[su_idx, :], LCC_su_rate)
        LCC_now = LCC[su_idx]
        if LCC_now == 1:
            if transition == 0:
                LCC[su_idx] = 2
            else:
                LCC[su_idx] = 7
        elif LCC_now == 2:
            if transition == 0:
                LCC[su_idx] = 3
            elif transition == 1:
                LCC[su_idx] = 1
            else:
                LCC[su_idx] = 8
        elif LCC_now == 3:
            if transition == 0:
                LCC[su_idx] = 4
            elif transition == 1:
                LCC[su_idx] = 2
            else:
                LCC[su_idx] = 9
        elif LCC_now == 4:
            if transition == 0:
                LCC[su_idx] = 5
            elif transition == 1:
                LCC[su_idx] = 3
            else:
                LCC[su_idx] = 10
        elif LCC_now == 5:
            if transition == 0:
                LCC[su_idx] = 6
            elif transition == 1:
                LCC[su_idx] = 4
            else:
                LCC[su_idx] = 11
        elif LCC_now == 6:
            LCC[su_idx] = 5
        if LCC_now == 7:
            if transition == 0:
                LCC[su_idx] = 8
            else:
                LCC[su_idx] = 1
        elif LCC_now == 8:
            if transition == 0:
                LCC[su_idx] = 9
            elif transition == 1:
                LCC[su_idx] = 7
            else:
                LCC[su_idx] = 2
        elif LCC_now == 9:
            if transition == 0:
                LCC[su_idx] = 10
            elif transition == 1:
                LCC[su_idx] = 8
            else:
                LCC[su_idx] = 3
        elif LCC_now == 10:
            if transition == 0:
                LCC[su_idx] = 11
            elif transition == 1:
                LCC[su_idx] = 9
            else:
                LCC[su_idx] = 4
        elif LCC_now == 11:
            if transition == 0:
                LCC[su_idx] = 12
            elif transition == 1:
                LCC[su_idx] = 10
            else:
                LCC[su_idx] = 5
        else:
            LCC[su_idx] = 11

    elif u < (LCC_i_su_rate + LCC_i_su_rate) / su_total_rate:
        LCC_i[su_idx] = 0 if LCC_i[su_idx] == 1 else 1

    elif u < (LCC_i_su_rate + LCC_i_su_rate + RyR_su_rate) / su_total_rate:
        transition = sample_weights(RyR_rates[su_idx, :], RyR_su_rate)
        if transition == 0:  # state 1 -> state 2
            RyR[su_idx, 0] -= 1
            RyR[su_idx, 1] += 1
        elif transition == 1:  # state 2 -> state 3
            RyR[su_idx, 1] -= 1
            RyR[su_idx, 2] += 1
        elif transition == 2:  # state 2 -> state 5
            RyR[su_idx, 1] -= 1
            RyR[su_idx, 4] += 1
        elif transition == 3:  # state 3 -> state 4
            RyR[su_idx, 2] -= 1
            RyR[su_idx, 3] += 1
        elif transition == 4:  # state 4 -> state 5
            RyR[su_idx, 3] -= 1
            RyR[su_idx, 4] += 1
        elif transition == 5:  # state 5 -> state 6
            RyR[su_idx, 4] -= 1
            RyR[su_idx, 5] += 1
        elif transition == 6:  # state 2 -> state 1
            RyR[su_idx, 1] -= 1
            RyR[su_idx, 0] += 1
        elif transition == 7:  # state 3 -> state 2
            RyR[su_idx, 2] -= 1
            RyR[su_idx, 1] += 1
        elif transition == 8:  # state 4 -> state 3
            RyR[su_idx, 3] -= 1
            RyR[su_idx, 2] += 1
        elif transition == 9:  # state 5 -> state 2
            RyR[su_idx, 4] -= 1
            RyR[su_idx, 1] += 1
        elif transition == 10:  # state 5 -> state 4
            RyR[su_idx, 4] -= 1
            RyR[su_idx, 3] += 1
        else:  # state 6 -> state 5
            RyR[su_idx, 5] -= 1
            RyR[su_idx, 4] += 1

    else:
        ClCh[su_idx] = 0 if ClCh[su_idx] == 1 else 0

    return dt


@numba.njit
def update_fluxes(
    CaSS: npt.NDArray,
    CaJSR: float,
    LCC: npt.NDArray,
    LCC_i: npt.NDArray,
    RyR: npt.NDArray,
    Cai: float,
    V: float,
    betaSS: npt.NDArray,
    JLCC: npt.NDArray,
    Jrel: npt.NDArray,
    Jxfer: npt.NDArray,
    Jiss: npt.NDArray,
    params: Parameters,
) -> None:
    VF_RT = V * params.F / (params.R * params.T)
    JLCC_exp_term = np.exp(2 * VF_RT)
    for i in range(4):
        # update JLCC = -ILCC / (2*F*VSS)
        if VF_RT == 0:
            JLCC[i] = (
                -(params.PCaL / params.VSS)
                * ((LCC[i] == 6 + LCC[i] == 12) * LCC_i[i])
                * (CaSS[i] - 0.341 * params.Cao)
            )
        else:
            JLCC[i] = (
                -(params.PCaL / params.VSS)
                * (2 * VF_RT)
                * (CaSS[i] * JLCC_exp_term - 0.341 * params.Cao)
                / (JLCC_exp_term - 1)
                * ((LCC[i] == 6 + LCC[i] == 12) * LCC_i[i])
            )

        # update Jrel
        Jrel[i] = params.rRyR * (RyR[i, 2] + RyR[i, 3]) * (CaJSR - CaSS[i])

        # update Jxfer
        Jxfer[i] = params.rxfer * (CaSS[i] - Cai)

        # update Jiss
        Jiss[i] = params.riss * (CaSS[(i + 1) % 4] + CaSS[i - 1] - 2 * CaSS[i])

        # update betaSS
        betaSS[i] = 1.0 / (
            1.0
            + params.BSRT * params.KBSR / (CaSS[i] + params.KBSL) ** 2
            + params.BSLT * params.KBSL / (CaSS[i] + params.KBSL) ** 2
        )


@numba.njit
def CRU_fwd(
    CaSS_init: npt.NDArray,
    CaJSR_init: float,
    LCC_init: npt.NDArray,
    LCC_i_init: npt.NDArray,
    RyR_init: npt.NDArray,
    ClCh_init: npt.NDArray,
    Cai: float,
    CaNSR: float,
    V: float,
    params: Parameters,
    num_SSA_steps: int,
) -> Any:
    CaSS = np.copy(CaSS_init)
    CaJSR = CaJSR_init
    LCC = np.copy(LCC_init)
    LCC_i = np.copy(LCC_i_init)
    RyR = np.copy(RyR_init)
    ClCh = np.copy(ClCh_init)

    betaSS = np.zeros_like(CaSS)
    JLCC = np.zeros_like(CaSS)
    Jrel = np.zeros_like(CaSS)
    Jxfer = np.zeros_like(CaSS)
    Jiss = np.zeros_like(CaSS)

    LCC_rates = np.zeros((4, 3))
    LCC_i_rates = np.zeros(4)
    RyR_rates = np.zeros((4, 12))
    ClCh_rates = np.zeros(4)
    subunit_rates = np.zeros(4)

    times = np.zeros(num_SSA_steps + 1)
    CaSS_out = np.zeros((num_SSA_steps + 1, 4))
    LCC_out = np.zeros((num_SSA_steps + 1, 4))
    LCC_i_out = np.zeros((num_SSA_steps + 1, 4))
    RyR_out = np.zeros((num_SSA_steps + 1, 4, 6))
    ClCh_out = np.zeros((num_SSA_steps + 1, 4))
    CaJSR_out = np.zeros(num_SSA_steps + 1)

    CaSS_out[0, :] = CaSS_init
    LCC_out[0, :] = LCC_init
    LCC_i_out[0, :] = LCC_i_init
    RyR_out[0, :] = RyR_init
    ClCh_out[0, :] = ClCh_init
    CaJSR_out[0] = CaJSR_init

    betaSS_out = np.zeros((num_SSA_steps + 1, 4))
    JLCC_out = np.zeros((num_SSA_steps + 1, 4))
    Jrel_out = np.zeros((num_SSA_steps + 1, 4))
    Jxfer_out = np.zeros((num_SSA_steps + 1, 4))
    Jiss_out = np.zeros((num_SSA_steps + 1, 4))

    LCC_alpha = 2.0 * np.exp(0.012 * (V - 35))
    LCC_beta = 0.0882 * np.exp(-0.05 * (V - 35))
    yinf_LCC_i = 0.4 / (1 + np.exp((V + 12.5) / 5)) + 0.6
    tau_LCC_i = 340 / (1 + np.exp((V + 30) / 12)) + 60
    kf_LCC_i = yinf_LCC_i / tau_LCC_i
    kb_LCC_i = (1 - yinf_LCC_i) / tau_LCC_i

    t = 0.0
    for i in range(num_SSA_steps + 1):

        update_fluxes(
            CaSS,
            CaJSR,
            LCC,
            LCC_i,
            RyR,
            Cai,
            V,
            betaSS,
            JLCC,
            Jrel,
            Jxfer,
            Jiss,
            params,
        )
        betaJSR = 1.0 / (
            1.0 + params.CSQNT * params.KCSQN / (CaJSR + params.KCSQN) ** 2
        )
        Jtr = params.rtr * (CaNSR - CaJSR)

        dt = SSA_1_step(
            CaSS,
            LCC,
            LCC_i,
            RyR,
            ClCh,
            LCC_rates,
            LCC_i_rates,
            RyR_rates,
            ClCh_rates,
            subunit_rates,
            LCC_alpha,
            LCC_beta,
            kf_LCC_i,
            kb_LCC_i,
            params,
        )

        CaJSR += dt * betaJSR * (Jtr - (params.VSS / params.VJSR) * Jrel.sum())
        CaSS += dt * betaSS * (JLCC + Jrel - Jxfer + Jiss)

        t += dt

        if i < num_SSA_steps:
            times[i + 1] = t
            CaSS_out[i + 1, :] = CaSS
            LCC_out[i + 1, :] = LCC
            LCC_i_out[i + 1, :] = LCC_i
            RyR_out[i + 1, :] = RyR
            ClCh_out[i + 1, :] = ClCh
            CaJSR_out[i + 1] = CaJSR

        betaSS_out[i, :] = betaSS
        JLCC_out[i, :] = JLCC
        Jrel_out[i, :] = Jrel
        Jxfer_out[i, :] = Jxfer
        Jiss_out[i, :] = Jiss

    return (
        times,
        CaSS_out,
        CaJSR_out,
        LCC_out,
        LCC_i_out,
        RyR_out,
        ClCh_out,
        betaSS_out,
        JLCC_out,
        Jrel_out,
        Jxfer_out,
        Jiss_out,
    )


class CRU:

    def __init__(self, sim: Any, params: Parameters | None = None):
        if params is None:
            self.params = Parameters()
        else:
            self.params = params
        self.sim = sim

        self.times = None
        self.CaSS = None
        self.CaJSR = None
        self.LCC = None
        self.LCC_i = None
        self.RyR = None
        self.ClCh = None
        self.betaSS = None
        self.JLCC = None
        self.Jrel = None
        self.Jxfer = None
        self.Jiss = None

    def CRU_fwd(self, time_idx: int, CRU_idx: int, num_SSA_steps: int) -> None:
        RyR_init = np.copy(self.sim.RyR[time_idx, CRU_idx])
        for j in range(4):
            if self.sim.CaSS[time_idx, CRU_idx, j] > 1.15e-4:
                p5 = self.params.k65 / (
                    self.params.k65
                    + self.params.k56 * self.sim.CaSS[time_idx, CRU_idx, j] ** 2
                )
                n56 = RyR_init[j, 4] + RyR_init[j, 5]
                RyR_init[j, 4] = np.random.binomial(n56, p5)
                RyR_init[j, 5] = n56 - RyR_init[j, 4]

            if self.sim.CaSS[time_idx, CRU_idx, j] > 3.685e-2:
                p3 = self.params.k43 / (
                    self.params.k43
                    + self.params.k34 * self.sim.CaSS[time_idx, CRU_idx, j] ** 2
                )
                n34 = RyR_init[j, 2] + RyR_init[j, 3]
                RyR_init[j, 2] = np.random.binomial(n34, p3)
                RyR_init[j, 3] = n34 - RyR_init[j, 2]

        out = CRU_fwd(
            CaSS_init=self.sim.CaSS[time_idx, CRU_idx],
            CaJSR_init=self.sim.CaJSR[time_idx, CRU_idx],
            LCC_init=self.sim.LCC[time_idx, CRU_idx],
            LCC_i_init=self.sim.LCC_inactivation[time_idx, CRU_idx],
            RyR_init=RyR_init,
            ClCh_init=self.sim.ClCh[time_idx, CRU_idx],
            Cai=self.sim.Cai[time_idx],
            CaNSR=self.sim.CaNSR[time_idx],
            V=self.sim.V[time_idx],
            params=self.params,
            num_SSA_steps=num_SSA_steps,
        )

        self.times = out[0]
        self.CaSS = out[1]
        self.CaJSR = out[2]
        self.LCC = out[3]
        self.LCC_i = out[4]
        self.RyR = out[5]
        self.ClCh = out[6]
        self.betaSS = out[7]
        self.JLCC = out[8]
        self.Jrel = out[9]
        self.Jxfer = out[10]
        self.Jiss = out[11]


def JLCC(sim: Any, params: Any) -> npt.NDArray:
    F = 96.5
    R = 8.314
    VF_RT = sim.V[:, np.newaxis, np.newaxis] * F / (R * params.T)
    JLCC_exp_term = np.exp(2 * VF_RT)
    return 1e6 * (
        -(params.PCaL / params.VSS)
        * (2 * VF_RT)
        * (sim.CaSS * JLCC_exp_term - 0.341 * params.Cao)
        / (JLCC_exp_term - 1)
        * (np.logical_or(sim.LCC == 6, sim.LCC == 12) * sim.LCC_inactivation)
    )


def Jrel(sim: Any, params: Any) -> npt.NDArray:
    return (
        params.rRyR
        * (sim.RyR[..., 2] + sim.RyR[..., 3])
        * (sim.CaJSR[..., np.newaxis] - sim.CaSS)
    )


def Jxfer(sim: Any, params: Any) -> npt.NDArray:
    return params.rxfer * (sim.CaSS - sim.Cai[:, np.newaxis, np.newaxis])


def Jiss(sim: Any, params: Any) -> npt.NDArray:
    out = np.zeros_like(sim.CaSS)
    for i in range(4):
        out[..., i] = params.riss * (
            sim.CaSS[..., (i + 1) % 4] + sim.CaSS[..., i - 1] - 2 * sim.CaSS[..., i]
        )
    return out


def betaSS(sim: Any, params: Any) -> npt.NDArray:
    return 1.0 / (
        1.0
        + params.BSRT * params.KBSR / (sim.CaSS + params.KBSL) ** 2
        + params.BSLT * params.KBSL / (sim.CaSS + params.KBSL) ** 2
    )


# @numba.njit
# def iss_correlations(CaSS: npt.NDArray) -> npt.NDArray:
#    mean_CaSS = CaSS.mean(axis=(1, 2))
#    out = np.zeros_like(mean_CaSS)
#    for i in range(out.shape[0]):
#        out[i] = np.mean(CaSS[i, 0] * CaSS[..., 1])
