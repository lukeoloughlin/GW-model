import numpy as np
import numpy.typing as npt
import numba

from params import RestrepoParams


@numba.njit
def calculate_rho(cjsr: float, params: RestrepoParams) -> float:
    """Calculate rho(cjsr)"""
    cjsr_K_h = (cjsr / params.K) ** params.h
    return params.rho_inf / (1.0 + cjsr_K_h)


@numba.njit
def calculate_Mhat(rho: float, BCSQN: float) -> float:
    """Calculate Mhat from rho"""
    return (np.sqrt(1.0 + 8.0 * rho * BCSQN) - 1.0) / (4.0 * rho * BCSQN)


@numba.njit
def update_LCC_probs_cpu(
    LCC_probs: npt.NDArray,
    LCC: npt.NDArray,
    cp: float,
    V: float,
    dt: float,
    params: RestrepoParams,
):

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

    cptilde_cp3 = (params.cp_tilde / cp) ** 3
    TCa = (78.0329 + 0.1 * (1.0 + cp / params.cp_bar) ** 4) / (
        1.0 + (cp / params.cp_bar) ** 4
    )
    tauCa = (R - TCa) * Pr + TCa

    s1 = 0.02 / (1.0 + cptilde_cp3)
    k1 = 0.03 / (1.0 + cptilde_cp3)
    k5 = (1.0 - Ps) / tauCa
    k6 = Ps / (tauCa * (1.0 + cptilde_cp3))

    s2 = s1 * params.k2 * params.r1 / (k1 * params.r2)
    s2_ = params.s1_ * params.k2_ * params.r1 / (params.k1_ * params.r2)
    k4 = k3 * (alpha / beta) * (k1 / params.k2) * (k5 / k6)
    k4_ = k3 * (alpha / beta) * (params.k1_ / params.k2_) * (k5_ / k6_)

    for j in range(4):
        if LCC[j] == 1:
            LCC_probs[j, 0] = 1.0 - dt * (params.r1 + beta + k1 + params.k1_)
            LCC_probs[j, 1] = dt * beta
            LCC_probs[j, 2] = dt * k1
            LCC_probs[j, 3] = 0.0
            LCC_probs[j, 4] = dt * params.k1_
            LCC_probs[j, 5] = 0.0
            LCC_probs[j, 6] = dt * params.r1
        elif LCC[j] == 2:
            LCC_probs[j, 0] = dt * alpha
            LCC_probs[j, 1] = 1.0 - dt * (k6 + k6_ + alpha)
            LCC_probs[j, 2] = 0.0
            LCC_probs[j, 3] = dt * k6
            LCC_probs[j, 4] = 0.0
            LCC_probs[j, 5] = dt * k6_
            LCC_probs[j, 6] = 0.0
        elif LCC[j] == 3:
            LCC_probs[j, 0] = dt * params.k2
            LCC_probs[j, 1] = 0.0
            LCC_probs[j, 2] = 1.0 - dt * (params.k2 + k3 + s2)
            LCC_probs[j, 3] = dt * k3
            LCC_probs[j, 4] = 0.0
            LCC_probs[j, 5] = 0.0
            LCC_probs[j, 6] = dt * s2
        elif LCC[j] == 4:
            LCC_probs[j, 0] = 0.0
            LCC_probs[j, 1] = dt * k5
            LCC_probs[j, 2] = dt * k4
            LCC_probs[j, 3] = (1.0) - dt * (k4 + k5)
            LCC_probs[j, 4] = 0.0
            LCC_probs[j, 5] = 0.0
            LCC_probs[j, 6] = 0.0
        elif LCC[j] == 5:
            LCC_probs[j, 0] = dt * params.k2_
            LCC_probs[j, 1] = 0.0
            LCC_probs[j, 2] = 0.0
            LCC_probs[j, 3] = 0.0
            LCC_probs[j, 4] = 1.0 - dt * (params.k2_ + k3 + s2_)
            LCC_probs[j, 5] = dt * k3
            LCC_probs[j, 6] = dt * s2_
        elif LCC[j] == 6:
            LCC_probs[j, 0] = 0.0
            LCC_probs[j, 1] = dt * k5_
            LCC_probs[j, 2] = 0.0
            LCC_probs[j, 3] = 0.0
            LCC_probs[j, 4] = dt * k4_
            LCC_probs[j, 5] = 1.0 - dt * (k4_ + k5_)
            LCC_probs[j, 6] = 0.0
        else:
            LCC_probs[j, 0] = dt * params.r2
            LCC_probs[j, 1] = 0.0
            LCC_probs[j, 2] = dt * s1
            LCC_probs[j, 3] = 0.0
            LCC_probs[j, 4] = dt * params.s1_
            LCC_probs[j, 5] = 0.0
            LCC_probs[j, 6] = 1.0 - dt * (params.r1 + s1 + params.s1_)


@numba.njit
def sample_icdf_cpu(LCC: npt.NDArray, LCC_probs: npt.NDArray) -> None:
    for i in range(4):
        u = np.random.rand()
        cdf = 0.0
        for j in range(7):
            cdf += LCC_probs[i, j]
            if u < cdf:
                LCC[i] = j + 1  # state starts at 1 so increment


@numba.njit
def update_RyR_rates_cpu(
    RyR_rates: npt.NDArray,
    RyR: npt.NDArray,
    cp: float,
    cjsr: float,
    params: RestrepoParams,
):
    """Device func to update RyR rates at position x, y"""
    Mhat = calculate_Mhat(calculate_rho(cjsr, params), params.BCSQN)

    k12 = params.Ku * cp**2  # k12
    k23 = Mhat * cp / params.tau_b  # k23

    k43 = params.Kb * cp**2  # k43
    k32 = k12 / (k43 * params.tau_u)  # k32 = k41 * k12 / k43

    RyR_rates[0] = k12 * RyR[0]  # 1 -> 2
    RyR_rates[1] = RyR[1] / params.tau_c  # 2 -> 1; k21 = _1_tau_c
    RyR_rates[2] = k23 * RyR[1]  # 2 -> 3
    RyR_rates[3] = k32 * RyR[2]  # 3 -> 2
    RyR_rates[4] = RyR[2] / params.tau_c  # 3 -> 4; k34 = _1_tau_c
    RyR_rates[5] = k43 * RyR[3]  # 4 -> 3
    RyR_rates[6] = RyR[3] / params.tau_u  # 4 -> 1; k41 = _1_tau_u
    RyR_rates[7] = k23 * RyR[0]  # 1-> 4; k14 = k23


@numba.njit
def RyR_orth_proj_simplex_cpu(RyR: npt.NDArray, RyR_sorted: npt.NDArray) -> None:
    """Device func to perform orthogonal projection of RyR values onto simplex after Euler Maruyama update"""
    # Copy the RyR values into preallocated array and use bubble sort

    RyR_sorted[:] = np.sort(RyR)

    lambda_ = 0.0
    sum_ = 1.0
    for i in range(4):
        if sum_ - (4.0 - i) * RyR_sorted[i] < 1.0:
            lambda_ = (sum_ - 1.0) / (4.0 - i)
            break
        else:
            sum_ -= RyR_sorted[i]

    RyR[0] = max(RyR[0] - lambda_, 0.0)
    RyR[1] = max(RyR[1] - lambda_, 0.0)
    RyR[2] = max(RyR[2] - lambda_, 0.0)
    RyR[3] = max(RyR[3] - lambda_, 0.0)


@numba.njit
def update_RyR_diffusion_cpu(
    RyR: npt.NDArray, RyR_sorted: npt.NDArray, RyR_rates: npt.NDArray, dt: float
) -> None:
    """Euler Maruyama step for RyR model with reflecting boundary conditions."""
    drift1 = (
        RyR_rates[1] + RyR_rates[6] - (RyR_rates[0] + RyR_rates[7])
    )  # q21 + q41 - (q12 + q14)
    drift2 = (
        RyR_rates[0] + RyR_rates[3] - (RyR_rates[1] + RyR_rates[2])
    )  # q12 + q32 - (q21 + q23)
    drift3 = (
        RyR_rates[2] + RyR_rates[5] - (RyR_rates[3] + RyR_rates[4])
    )  # q23 + q43 - (q32 + q34)

    sigma12 = 0.1 * np.sqrt(RyR_rates[0] + RyR_rates[1])  # q12 + q21
    sigma23 = 0.1 * np.sqrt(RyR_rates[2] + RyR_rates[3])  # q23 + q32
    sigma34 = 0.1 * np.sqrt(RyR_rates[4] + RyR_rates[5])  # q34 + q43
    sigma14 = 0.1 * np.sqrt(RyR_rates[6] + RyR_rates[7])  # q41 + q14

    sqrtdt = np.sqrt(dt)
    dW0 = sqrtdt * np.random.normal()
    dW1 = sqrtdt * np.random.normal()
    dW2 = sqrtdt * np.random.normal()
    dW3 = sqrtdt * np.random.normal()

    RyR[0] += dt * drift1 + sigma12 * dW0 + sigma14 * dW3
    RyR[1] += dt * drift2 - sigma12 * dW0 + sigma23 * dW1
    RyR[2] += dt * drift3 - sigma23 * dW1 + sigma34 * dW2
    RyR[3] = 1.0 - (RyR[0] + RyR[1] + RyR[2])

    RyR_orth_proj_simplex_cpu(RyR, RyR_sorted)


@numba.njit
def ITCa(c: float, CaT: float, params: RestrepoParams) -> float:
    return params.kon * c * (params.BT - CaT) - params.koff * CaT


@numba.njit
def Ileak(cjsr: float, cnsr: float, ci: float, params: RestrepoParams) -> float:
    cjsr2 = cjsr**2
    Kjsr2 = params.Kjsr**2
    return params.gleak * cjsr2 / (cjsr2 + Kjsr2) * (cnsr - ci)


@numba.njit
def Iup(ci: float, cnsr: float, params: RestrepoParams) -> float:
    ci_term = (ci / params.Ki) ** params.H
    cnsr_term = (cnsr / params.Knsr) ** params.H

    return params.vup * (ci_term - cnsr_term) / (1.0 + ci_term + cnsr_term)


@numba.njit
def Ir(cp: float, cjsr: float, RyR_open: float, params: RestrepoParams) -> float:
    return params.Jmax * RyR_open * (cjsr - cp) / params.vp


@numba.njit
def luminal_buffer(cjsr: float, params: RestrepoParams) -> float:
    """Calculate the luminal buffering term"""

    rho = calculate_rho(cjsr, params)
    Mhat = calculate_Mhat(rho, params.BCSQN)

    ncjsr = Mhat * params.nM + (1.0 - Mhat) * params.nD

    dlog_rho = 23.0 * (1.0 - rho / params.rho_inf) / cjsr
    dMhat = (
        -(0.25 / params.BCSQN)
        * dlog_rho
        * (
            1.0
            + (1.0 + 4.0 * rho * params.BCSQN) / np.sqrt(1.0 + 8.0 * rho * params.BCSQN)
        )
        / rho
    )

    dn = dMhat * (params.nM - params.nD)

    return 1.0 / (
        1.0
        + (params.KC * params.BCSQN * ncjsr + dn * (cjsr * params.KC + cjsr**2))
        / ((params.KC + cjsr) ** 2)
    )


@numba.njit
def ICa(LCC: npt.NDArray, cp: float, z: float, params: RestrepoParams) -> None:
    """Update array of ICa values. This is only calclated along the boundaries in a flattened 1d array"""
    F = 96.5
    exp2z = np.exp(2.0 * z)
    NLCC = 0.0
    for i in range(4):
        if LCC[i] == 7:
            NLCC += 1.0
    return (
        NLCC
        * 4.0
        * params.PCa
        * z
        * F
        * params.gamma
        * (cp * 1e-3 * exp2z - params.Cao)
        / (exp2z - 1.0)
    )


@numba.njit
def INaCa(cs: float, z: float, Nai3: float, params: RestrepoParams) -> float:
    Nao3 = params.Nao**3
    KmNao3 = params.KmNao**3
    KmNai3 = params.KmNai**3
    Ka = 1.0 / (1.0 + (params.Kda / cs) ** 3)
    t1 = params.KmCai * Nao3 * (1.0 + Nai3 / KmNai3)
    t2 = KmNao3 * cs * (1.0 + (cs / params.KmCai))
    t3 = params.KmCao * Nai3 + Nai3 * params.Cao + Nao3 * cs
    exp_etaz = np.exp(params.eta * z)
    exp_etam1z = np.exp((params.eta - 1.0) * z)

    return (
        Ka
        * params.vNaCa
        * (exp_etaz * Nai3 * params.Cao - exp_etam1z * Nao3 * cs)
        / ((t1 + t2 + t3) * (1.0 + params.ksat * exp_etam1z))
    )


@numba.njit
def beta_i(ci: float, params: RestrepoParams) -> float:
    calmodulin_buf = params.KCAM * params.BCAM / (params.KCAM + ci) ** 2
    SR_buf = params.KSR * params.BSR / (params.KSR + ci) ** 2
    myosin_Ca_buf = params.KMCa * params.BMCa / (params.KMCa + ci) ** 2
    myosin_Mg_buf = params.KMMg * params.BMMg / (params.KMMg + ci) ** 2

    return 1.0 / (1.0 + calmodulin_buf + SR_buf + myosin_Ca_buf + myosin_Mg_buf)


@numba.njit
def beta_s(cs: float, params: RestrepoParams) -> float:
    calmodulin_buf = params.KCAM * params.BCAM / (params.KCAM + cs) ** 2
    SLH_buf = params.KSLH * params.BSLH / (params.KSLH + cs) ** 2
    return 1.0 / (1.0 + calmodulin_buf + SLH_buf)


@numba.njit
def CRU_step_non_boundary(
    ci: float,
    cs: float,
    cp: float,
    cnsr: float,
    cjsr: float,
    CaTi: float,
    CaTs: float,
    RyR: npt.NDArray,
    RyR_sorted: npt.NDArray,
    RyR_rates: npt.NDArray,
    ci_neighbours: float,
    cs_neighbours: float,
    cnsr_neighbours: float,
    dt: float,
    params: RestrepoParams,
):
    update_RyR_rates_cpu(RyR_rates, RyR, cp, cjsr, params)

    Idsi = (cs - ci) / params.tau_si
    Iup_ = Iup(ci, cnsr, params)
    Ileak_ = Ileak(cjsr, cnsr, ci, params)
    ITCi = ITCa(ci, CaTi, params)
    ICi = (ci_neighbours - ci) * (2 / params.tau_iL + 2 / params.tau_iT)

    Idps = (cp - cs) / params.tau_ps
    # INCX = INaCa(cs, z, Nai**3, params)
    ITCs = ITCa(cs, CaTs, params)
    ICs = (cs_neighbours - cs) * (2 / params.tau_sL + 2 / params.tau_sT)

    Ir_ = Ir(cp, cjsr, RyR[1] + RyR[2], params)

    Itr = (cnsr - cjsr) / params.tau_tr
    ICnsr = (cnsr_neighbours - cnsr) * (2 / params.tau_nsrL + 2 / params.tau_nsrT)

    beta_i_ = beta_i(ci, params)
    beta_s_ = beta_s(cs, params)
    beta_jsr = luminal_buffer(cjsr, params)

    kr = (params.Jmax / params.vp) * (RyR[1] + RyR[2])
    cp_out = (cs + params.tau_ps * (kr * cjsr)) / (1.0 + params.tau_ps * kr)

    ci += dt * beta_i_ * (Idsi * (params.vs / params.vi) - Iup_ + Ileak_ - ITCi + ICi)
    cs += dt * beta_s_ * (Idps * (params.vp / params.vs) - Idsi - ITCs + ICs)
    cnsr += dt * (
        (Iup_ - Ileak_) * (params.vi / params.vnsr)
        - Itr * (params.vjsr / params.vnsr)
        + ICnsr
    )
    cjsr += dt * beta_jsr * (Itr - Ir_ * (params.vp / params.vjsr))
    CaTi += dt * ITCi
    CaTs += dt * ITCs

    update_RyR_diffusion_cpu(RyR, RyR_sorted, RyR_rates, dt)

    return ci, cs, cp_out, cnsr, cjsr, CaTi, CaTs


@numba.njit
def CRU_fwd_non_boundary(
    ci: float,
    cs: float,
    cp: float,
    cnsr: float,
    cjsr: float,
    CaTi: float,
    CaTs: float,
    RyR: npt.NDArray,
    RyR_sorted: npt.NDArray,
    RyR_rates: npt.NDArray,
    ci_neighbours: float,
    cs_neighbours: float,
    cnsr_neighbours: float,
    dt: float,
    params: RestrepoParams,
    nstep: int,
):
    for _ in range(nstep):
        ci, cs, cp, cnsr, cjsr, CaTi, CaTs = CRU_step_non_boundary(
            ci,
            cs,
            cp,
            cnsr,
            cjsr,
            CaTi,
            CaTs,
            RyR,
            RyR_sorted,
            RyR_rates,
            ci_neighbours,
            cs_neighbours,
            cnsr_neighbours,
            dt,
            params,
        )
    return ci, cs, cp, cnsr, cjsr, CaTi, CaTs


class RestrepoCPU:

    def __init__(
        self,
        ci: float = 0.1,
        cs: float = 0.1,
        cp: float = 0.1,
        cnsr: float = 750.0,
        cjsr: float = 750.0,
        CaTi: float = 20.0,
        CaTs: float = 20.0,
        RyR: npt.NDArray = np.array([0.3, 0.0, 0.0, 0.7]),
        LCC: npt.NDArray = np.array([2, 2, 2, 2], dtype=int),
        params: RestrepoParams = RestrepoParams(),
        boundary=False,
    ):
        # Initial values stored privately
        self._ci = ci
        self._cs = cs
        self._cp = cp
        self._cnsr = cnsr
        self._cjsr = cjsr
        self._CaTi = CaTi
        self._CaTs = CaTs
        self._RyR = RyR
        self._LCC = LCC

        self.t: npt.NDArray | None = None
        self.ci: npt.NDArray | None = None
        self.cs: npt.NDArray | None = None
        self.cp: npt.NDArray | None = None
        self.cnsr: npt.NDArray | None = None
        self.cjsr: npt.NDArray | None = None
        self.CaTi: npt.NDArray | None = None
        self.CaTs: npt.NDArray | None = None
        self.RyR: npt.NDArray | None = None
        self.LCC: npt.NDArray | None = None

        self.params = params

        self._RyR_sorted = np.zeros_like(RyR)
        self._RyR_rates = np.zeros(8)
        self._LCC_probs = np.zeros_like(LCC)

        self.boundary = boundary

    def forward(self, dt: float, nstep: int, collect_every: int):
        if self.boundary:
            pass
        else:
            self._forward_non_boundary(dt, nstep, collect_every)

    def _forward_non_boundary(self, dt: float, nstep: int, collect_every: int):
        ncollect = nstep // collect_every + 1
        self.t = np.zeros(ncollect)
        self.ci = np.zeros(ncollect)
        self.cs = np.zeros(ncollect)
        self.cp = np.zeros(ncollect)
        self.cnsr = np.zeros(ncollect)
        self.cjsr = np.zeros(ncollect)
        self.CaTi = np.zeros(ncollect)
        self.CaTs = np.zeros(ncollect)
        self.RyR = np.zeros((ncollect, 4))

        self.ci[0] = ci = self._ci
        self.cs[0] = cs = self._cs
        self.cp[0] = cp = self._cp
        self.cnsr[0] = cnsr = self._cnsr
        self.cjsr[0] = cjsr = self._cjsr
        self.CaTi[0] = CaTi = self._CaTi
        self.CaTs[0] = CaTs = self._CaTs
        self.RyR[0, :] = np.copy(self._RyR)
        ryr = np.copy(self._RyR)

        for i in range(ncollect - 1):
            ci, cs, cp, cnsr, cjsr, CaTi, CaTs = CRU_fwd_non_boundary(
                ci,
                cs,
                cp,
                cnsr,
                cjsr,
                CaTi,
                CaTs,
                ryr,
                self._RyR_sorted,
                self._RyR_rates,
                0.0,
                0.0,
                self._cnsr,
                dt,
                self.params,
                collect_every,
            )

            self.t[i + 1] = self.t[i] + dt * collect_every
            self.ci[i + 1] = ci
            self.cs[i + 1] = cs
            self.cp[i + 1] = cp
            self.cnsr[i + 1] = cnsr
            self.cjsr[i + 1] = cjsr
            self.CaTi[i + 1] = CaTi
            self.CaTs[i + 1] = CaTs
            self.RyR[i + 1, :] = ryr
