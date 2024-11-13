import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


import numpy as np
import numpy.typing as npt
import numba

from params import RestrepoParams
import utils


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
    clamp: bool,
):
    ryr_open = RyR[1] + RyR[2]
    utils.update_RyR_rates_cpu(RyR_rates, RyR, cp, cjsr, params)

    Idsi = (cs - ci) / params.tau_si
    Iup_ = utils.Iup(ci, cnsr, params)
    Ileak_ = utils.Ileak(cjsr, cnsr, ci, params)
    ITCi = utils.ITCa(ci, CaTi, params)
    ICi = (ci_neighbours - ci) * (2 / params.tau_iL + 2 / params.tau_iT)

    Idps = (cp - cs) / params.tau_ps
    ITCs = utils.ITCa(cs, CaTs, params)
    ICs = (cs_neighbours - cs) * (2 / params.tau_sL + 2 / params.tau_sT)

    Ir_ = utils.Ir(cp, cjsr, ryr_open, params)

    Itr = (cnsr - cjsr) / params.tau_tr
    ICnsr = (cnsr_neighbours - cnsr) * (2 / params.tau_nsrL + 2 / params.tau_nsrT)

    beta_i_ = utils.beta_i(ci, params)
    beta_s_ = utils.beta_s(cs, params)
    beta_jsr = utils.luminal_buffer(cjsr, params)

    kr = (params.Jmax / params.vp) * ryr_open
    cp_out = (cs + params.tau_ps * (kr * cjsr)) / (1.0 + params.tau_ps * kr)

    if clamp:
        ci += (
            dt * beta_i_ * (Idsi * (params.vs / params.vi) - Iup_ + Ileak_ - ITCi + ICi)
        )
        cs += dt * beta_s_ * (Idps * (params.vp / params.vs) - Idsi - ITCs + ICs)
        cnsr += dt * (
            (Iup_ - Ileak_) * (params.vi / params.vnsr)
            - Itr * (params.vjsr / params.vnsr)
            + ICnsr
        )
    else:
        ci += dt * beta_i_ * (Idsi * (params.vs / params.vi) - Iup_ + Ileak_ - ITCi)
        cs += dt * beta_s_ * (Idps * (params.vp / params.vs) - Idsi - ITCs)
        cnsr += dt * (
            (Iup_ - Ileak_) * (params.vi / params.vnsr)
            - Itr * (params.vjsr / params.vnsr)
        )
    cjsr += dt * beta_jsr * (Itr - Ir_ * (params.vp / params.vjsr))
    CaTi += dt * ITCi
    CaTs += dt * ITCs

    utils.update_RyR_diffusion_cpu(RyR, RyR_sorted, RyR_rates, dt)

    return ci, cs, cp_out, cnsr, cjsr, CaTi, CaTs


@numba.njit
def CRU_step_boundary(
    ci: float,
    cs: float,
    cp: float,
    cnsr: float,
    cjsr: float,
    CaTi: float,
    CaTs: float,
    LCC: npt.NDArray,
    LCC_probs: npt.NDArray,
    RyR: npt.NDArray,
    RyR_sorted: npt.NDArray,
    RyR_rates: npt.NDArray,
    ci_neighbours: float,
    cs_neighbours: float,
    cnsr_neighbours: float,
    V: float,
    Nai: float,
    dt: float,
    params: RestrepoParams,
    clamp: bool = True,
):
    z = V * 96.5 / (8.314 * 308.0)
    ryr_open = RyR[1] + RyR[2]
    lcc_open = float((LCC == 7).sum())
    utils.update_RyR_rates_cpu(RyR_rates, RyR, cp, cjsr, params)
    utils.update_LCC_probs_cpu(LCC_probs, LCC, cp, V, dt, params)

    Idsi = (cs - ci) / params.tau_si
    Iup_ = utils.Iup(ci, cnsr, params)
    Ileak_ = utils.Ileak(cjsr, cnsr, ci, params)
    ITCi = utils.ITCa(ci, CaTi, params)
    ICi = (ci_neighbours - ci) * (2 / params.tau_iL + 2 / params.tau_iT)

    Idps = (cp - cs) / params.tau_ps
    INCX = utils.INaCa(cs, z, Nai**3, params)
    ITCs = utils.ITCa(cs, CaTs, params)
    ICs = (cs_neighbours - cs) * (2 / params.tau_sL + 2 / params.tau_sT)

    Ir_ = utils.Ir(cp, cjsr, ryr_open, params)
    ICa_ = utils.ICa(lcc_open, cp, z, params)

    Itr = (cnsr - cjsr) / params.tau_tr
    ICnsr = (cnsr_neighbours - cnsr) * (2 / params.tau_nsrL + 2 / params.tau_nsrT)

    beta_i_ = utils.beta_i(ci, params)
    beta_s_ = utils.beta_s(cs, params)
    beta_jsr = utils.luminal_buffer(cjsr, params)

    kr = (params.Jmax / params.vp) * ryr_open
    cp_out = (cs + params.tau_ps * (kr * cjsr - ICa_)) / (1.0 + params.tau_ps * kr)

    if clamp:
        ci += (
            dt * beta_i_ * (Idsi * (params.vs / params.vi) - Iup_ + Ileak_ - ITCi + ICi)
        )
        cs += dt * beta_s_ * (Idps * (params.vp / params.vs) + INCX - Idsi - ITCs + ICs)
        cnsr += dt * (
            (Iup_ - Ileak_) * (params.vi / params.vnsr)
            - Itr * (params.vjsr / params.vnsr)
            + ICnsr
        )
    else:
        ci += dt * beta_i_ * (Idsi * (params.vs / params.vi) - Iup_ + Ileak_ - ITCi)
        cs += dt * beta_s_ * (Idps * (params.vp / params.vs) + INCX - Idsi - ITCs)
        cnsr += dt * (
            (Iup_ - Ileak_) * (params.vi / params.vnsr)
            - Itr * (params.vjsr / params.vnsr)
        )

    cjsr += dt * beta_jsr * (Itr - Ir_ * (params.vp / params.vjsr))
    CaTi += dt * ITCi
    CaTs += dt * ITCs

    utils.update_RyR_diffusion_cpu(RyR, RyR_sorted, RyR_rates, dt)
    utils.sample_lcc_cpu(LCC, LCC_probs)

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
    ci_neighbours: float,
    cs_neighbours: float,
    cnsr_neighbours: float,
    dt: float,
    params: RestrepoParams,
    nstep: int,
    collect_every: int,
    clamp: bool,
):
    ncollect = nstep // collect_every + 1
    t = np.zeros(ncollect)
    ci_out = np.zeros(ncollect)
    ci_out[0] = ci
    cs_out = np.zeros(ncollect)
    cs_out[0] = cs
    cp_out = np.zeros(ncollect)
    cp_out[0] = cp
    cnsr_out = np.zeros(ncollect)
    cnsr_out[0] = cnsr
    cjsr_out = np.zeros(ncollect)
    cjsr_out[0] = cjsr
    CaTi_out = np.zeros(ncollect)
    CaTi_out[0] = CaTi
    CaTs_out = np.zeros(ncollect)
    CaTs_out[0] = CaTs
    RyR_out = np.zeros((ncollect, 4))
    RyR_out[0, :] = np.copy(RyR)

    RyR_rates = np.zeros(8)
    RyR_sorted = np.zeros_like(RyR)
    for i in range(ncollect):
        for _ in range(collect_every):
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
                clamp,
            )
        t[i + 1] = t[i] + dt * collect_every
        ci_out[i + 1] = ci
        cs_out[i + 1] = cs
        cp_out[i + 1] = cp
        cnsr_out[i + 1] = cnsr
        cjsr_out[i + 1] = cjsr
        CaTi_out[i + 1] = CaTi
        CaTs_out[i + 1] = CaTs
        RyR_out[i + 1, :] = np.copy(RyR)
    return (t, ci_out, cs_out, cp_out, cnsr_out, cjsr_out, CaTi_out, CaTs_out, RyR_out)


@numba.njit
def CRU_fwd_boundary(
    ci: float,
    cs: float,
    cp: float,
    cnsr: float,
    cjsr: float,
    CaTi: float,
    CaTs: float,
    LCC: npt.NDArray,
    RyR: npt.NDArray,
    ci_neighbours: float,
    cs_neighbours: float,
    cnsr_neighbours: float,
    V: float,
    Nai: float,
    dt: float,
    params: RestrepoParams,
    nstep: int,
    collect_every: int,
    clamp: bool,
):
    ncollect = nstep // collect_every + 1
    t = np.zeros(ncollect)
    ci_out = np.zeros(ncollect)
    ci_out[0] = ci
    cs_out = np.zeros(ncollect)
    cs_out[0] = cs
    cp_out = np.zeros(ncollect)
    cp_out[0] = cp
    cnsr_out = np.zeros(ncollect)
    cnsr_out[0] = cnsr
    cjsr_out = np.zeros(ncollect)
    cjsr_out[0] = cjsr
    CaTi_out = np.zeros(ncollect)
    CaTi_out[0] = CaTi
    CaTs_out = np.zeros(ncollect)
    CaTs_out[0] = CaTs
    RyR_out = np.zeros((ncollect, 4))
    RyR_out[0, :] = np.copy(RyR)
    LCC_out = np.zeros((ncollect, 4))
    LCC_out[0, :] = np.copy(LCC)

    RyR_rates = np.zeros(8)
    RyR_sorted = np.zeros_like(RyR)
    LCC_probs = np.zeros((4, 7))
    for i in range(ncollect):
        for _ in range(collect_every):
            ci, cs, cp, cnsr, cjsr, CaTi, CaTs = CRU_step_boundary(
                ci,
                cs,
                cp,
                cnsr,
                cjsr,
                CaTi,
                CaTs,
                LCC,
                LCC_probs,
                RyR,
                RyR_sorted,
                RyR_rates,
                ci_neighbours,
                cs_neighbours,
                cnsr_neighbours,
                V,
                Nai,
                dt,
                params,
                clamp,
            )
        t[i + 1] = t[i] + dt * collect_every
        ci_out[i + 1] = ci
        cs_out[i + 1] = cs
        cp_out[i + 1] = cp
        cnsr_out[i + 1] = cnsr
        cjsr_out[i + 1] = cjsr
        CaTi_out[i + 1] = CaTi
        CaTs_out[i + 1] = CaTs
        RyR_out[i + 1, :] = np.copy(RyR)
        LCC_out[i + 1, :] = np.copy(LCC)
    return (
        t,
        ci_out,
        cs_out,
        cp_out,
        cnsr_out,
        cjsr_out,
        CaTi_out,
        CaTs_out,
        RyR_out,
        LCC_out,
    )


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
        RyR: npt.NDArray = np.array([1.0, 0.0, 0.0, 0.0]),
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
        self.boundary = boundary

    def forward(
        self,
        dt: float,
        nstep: int,
        collect_every: int = 1,
        clamp: bool = True,
        V: float | None = None,
        Nai: float | None = None,
        ci_clamp: float | None = None,
        cs_clamp: float | None = None,
        cnsr_clamp: float | None = None,
    ):
        ci_clamp = self._ci if ci_clamp is None else ci_clamp
        cs_clamp = self._cs if cs_clamp is None else cs_clamp
        cnsr_clamp = self._cnsr if cnsr_clamp is None else cnsr_clamp
        if self.boundary:
            assert V is not None, "Must pass V for boundary CRUs"
            assert Nai is not None, "Must pass Nai for boundary CRUs"
            self._forward_boundary(
                V,
                Nai,
                dt,
                nstep,
                collect_every,
                clamp,
                ci_clamp,
                cs_clamp,
                cnsr_clamp,
            )
        else:
            self._forward_non_boundary(
                dt, nstep, collect_every, clamp, ci_clamp, cs_clamp, cnsr_clamp
            )

    def _forward_non_boundary(
        self,
        dt: float,
        nstep: int,
        collect_every: int,
        clamp: bool,
        ci_clamp: float,
        cs_clamp: float,
        cnsr_clamp: float,
    ):
        ryr = np.copy(self._RyR)
        t, ci, cs, cp, cnsr, cjsr, CaTi, CaTs, RyR = CRU_fwd_non_boundary(
            self._ci,
            self._cs,
            self._cp,
            self._cnsr,
            self._cjsr,
            self._CaTi,
            self._CaTs,
            ryr,
            ci_clamp,
            cs_clamp,
            cnsr_clamp,
            dt,
            self.params,
            nstep,
            collect_every,
            clamp,
        )

        self.t = t
        self.ci = ci
        self.cs = cs
        self.cp = cp
        self.cnsr = cnsr
        self.cjsr = cjsr
        self.RyR = RyR
        self.CaTi = CaTi
        self.CaTs = CaTs

    def _forward_boundary(
        self,
        V: float,
        Nai: float,
        dt: float,
        nstep: int,
        collect_every: int,
        clamp: bool,
        ci_clamp: float,
        cs_clamp: float,
        cnsr_clamp: float,
    ):
        ryr = np.copy(self._RyR)
        lcc = np.copy(self._LCC)
        t, ci, cs, cp, cnsr, cjsr, CaTi, CaTs, RyR, LCC = CRU_fwd_boundary(
            self._ci,
            self._cs,
            self._cp,
            self._cnsr,
            self._cjsr,
            self._CaTi,
            self._CaTs,
            lcc,
            ryr,
            ci_clamp,
            cs_clamp,
            cnsr_clamp,
            V,
            Nai,
            dt,
            self.params,
            nstep,
            collect_every,
            clamp,
        )

        self.t = t
        self.ci = ci
        self.cs = cs
        self.cp = cp
        self.cnsr = cnsr
        self.cjsr = cjsr
        self.CaTi = CaTi
        self.CaTs = CaTs
        self.RyR = RyR
        self.LCC = LCC

    def reset_initial_conditions(self):
        if self.ci is not None:
            self._ci = self.ci[-1]
        if self.cs is not None:
            self._cs = self.cs[-1]
        if self.cp is not None:
            self._cp = self.cp[-1]
        if self.cnsr is not None:
            self._cnsr = self.cnsr[-1]
        if self.cjsr is not None:
            self._cjsr = self.cjsr[-1]
        if self.CaTi is not None:
            self._CaTi = self.CaTi[-1]
        if self.CaTs is not None:
            self._CaTs = self.CaTs[-1]
        if self.RyR is not None:
            self._RyR[:] = self.RyR[-1, :]
        if self.LCC is not None:
            self._LCC[:] = self.LCC[-1, :]
