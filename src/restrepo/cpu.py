import numpy as np
import numpy.typing as npt

from params import RestrepoParams


def calculate_rho(cjsr: float, K: float, rho_inf: float) -> float:
    """Calculate rho(cjsr)"""
    log_cjsr_K = np.log(cjsr) - np.log(K)
    return rho_inf / (1.0 + np.exp(23.0 * log_cjsr_K))


def calculate_Mhat(rho: float, BCSQN: float) -> float:
    """Calculate Mhat from rho"""
    return (np.sqrt(1.0 + 8.0 * rho * BCSQN) - 1.0) / (4.0 * rho * BCSQN)


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


def sample_icdf_cpu(LCC, LCC_probs):
    for i in range(4):
        u = np.random.rand()
        cdf = 0.0
        for j in range(7):
            cdf += LCC_probs[i, j]
            if u < cdf:
                LCC[i] = j + 1  # state starts at 1 so increment


def update_RyR_rates_cpu(
    RyR_rates: npt.NDArray,
    RyR: npt.NDArray,
    cp: float,
    cjsr: float,
    params: RestrepoParams,
):
    """Device func to update RyR rates at position x, y"""
    Mhat = calculate_Mhat(calculate_rho(cjsr, params.K, params.rho_inf), params.BCSQN)

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


def RyR_orth_proj_simplex_cpu(RyR, RyR_sorted):
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


def update_RyR_diffusion_cpu(RyR, RyR_sorted, RyR_rates, dW, dt):
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

    RyR[0] += dt * drift1 + sigma12 * dW[0] + sigma14 * dW[3]
    RyR[1] += dt * drift2 - sigma12 * dW[0] + sigma23 * dW[1]
    RyR[2] += dt * drift3 - sigma23 * dW[1] + sigma34 * dW[2]
    RyR[3] = 1.0 - (RyR[0] + RyR[1] + RyR[2])

    RyR_orth_proj_simplex_cpu(RyR, RyR_sorted)


def ITCa(c: float, CaT: float, kon: float, koff: float, BT: float) -> float:
    return kon * c * (BT - CaT) - koff * CaT


def Ileak(cjsr: float, cnsr: float, ci: float, gleak: float, Kjsr2: float) -> float:
    cjsr2 = cjsr**2
    return gleak * cjsr2 / (cjsr2 + Kjsr2) * (cnsr - ci)


def Iup(ci: float, cnsr: float, Ki: float, Knsr: float, vup: float) -> float:
    ci_term = (ci / Ki) ** 1.787
    cnsr_term = (cnsr / Knsr) ** 1.787

    return vup * (ci_term - cnsr_term) / (1.0 + ci_term + cnsr_term)


def Ir(cp: float, cjsr: float, RyR_open: float, Jmax: float, vp: float) -> float:
    return Jmax * RyR_open * (cjsr - cp) / vp


def luminal_buffer(
    cjsr: float, rho_inf: float, K: float, BCSQN: float, nM: float, nD: float, KC: float
):
    """Calculate the luminal buffering term"""

    rho = calculate_rho(cjsr, K, rho_inf)
    Mhat = calculate_Mhat(rho, BCSQN)

    ncjsr = Mhat * nM + (1.0 - Mhat) * nD

    dlog_rho = 23.0 * (1.0 - rho / rho_inf) / cjsr
    dMhat = (
        -(0.25 / BCSQN)
        * dlog_rho
        * (1.0 + (1.0 + 4.0 * rho * BCSQN) / np.sqrt(1.0 + 8.0 * rho * BCSQN))
        / rho
    )

    dn = dMhat * (nM - nD)

    return 1.0 / (
        1.0 + (KC * BCSQN * ncjsr + dn * (cjsr * KC + cjsr**2)) / ((KC + cjsr) ** 2)
    )


def ICa(LCC: npt.NDArray, cp: float, z: float, params: RestrepoParams):
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


def INaCa(
    cs: float,
    z: float,
    Nai3: float,
    params: RestrepoParams,
):
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
