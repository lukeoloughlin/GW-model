# pylint: disable=invalid-name,c-extension-no-member,broad-exception-caught,line-too-long
from typing import Callable, List
from functools import reduce
import pickle as pkl

import numpy as np
import numpy.typing as npt

import build.GreensteinWinslow as gw_cxx  # type: ignore
from .utils import assert_positive, assert_gt_numpy, assert_lt_numpy  # type: ignore
from .parameters import GWParameters  # type: ignore


F = 96.5
R = 8.314
F_R = F / R

GWParametersCXX = type[gw_cxx.GWParameters]
GWVairablesCXX = type[gw_cxx.GWVariables]

__IMPLEMENTED_PRNGS = [
    "mt19937",
    "mt19937_64",
    "xoshiro256+",
    "xoshiro256++",
    "xoshiro256**",
    "xoroshiro128+",
    "xoroshiro128++",
    "xoroshiro128**",
]


def __catch_invalid_state(
    x: npt.NDArray[np.integer], lb: int, ub: int, name: str
) -> None:
    assert ub > lb
    try:
        assert_gt_numpy(x, lb, name, strict=False)
        assert_lt_numpy(x, ub, name, strict=False)
    except ValueError as e:
        raise ValueError(
            f"{name} must consist of an integers in the range (lb, ..., ub)."
        ) from e


def ENa(
    Nai: npt.NDArray[np.floating], parameters: GWParameters
) -> npt.NDArray[np.floating]:
    """Calculates the Nernst potential for sodium ions.

    Args:
        Nai (npt.NDArray[np.floating]): Intracellular sodium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: Nernst potential of sodium.
    """
    assert_gt_numpy(Nai, 0, "Nai", strict=True)
    F_RT = F_R / parameters.T
    return np.log(parameters.Nao / Nai) / F_RT


def EK(
    Ki: npt.NDArray[np.floating], parameters: GWParameters
) -> npt.NDArray[np.floating]:
    """Calculate the Nernst potential for potassium ions.

    Args:
        Ki (npt.NDArray[np.floating]): Intracellular potassium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: Nernst potential of potassium.
    """
    assert_gt_numpy(Ki, 0, "Ki", strict=True)
    F_RT = F_R / parameters.T
    return np.log(parameters.Ko / Ki) / F_RT


def EKs(
    Nai: npt.NDArray[np.floating],
    Ki: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate Nernst potential for slow rectifier current IKs, which is permeable to potassium and sodium.

    Args:
        Nai (npt.NDArray[np.floating]): Intracellular sodium concentration [mM]. Value(s) must be > 0.
        Ki (npt.NDArray[np.floating]): Intracellular potassium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: Nernst potential of slow rectifier current IKs.
    """
    assert_gt_numpy(Nai, 0, "Nai", strict=True)
    assert_gt_numpy(Ki, 0, "Ki", strict=True)
    F_RT = F_R / parameters.T
    return (
        np.log((parameters.Ko + 0.01833 * parameters.Nao) / (Ki + 0.01833 * Nai)) / F_RT
    )


def ECa(
    Cai: npt.NDArray[np.floating], parameters: GWParameters
) -> npt.NDArray[np.floating]:
    """Calculate the Nernst potential for calcium ions.

    Args:
        Cai (npt.NDArray[np.floating]): Intracellular calcium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: Nernst potential of calcium.
    """
    assert_gt_numpy(Cai, 0, "Cai", strict=True)
    F_RT = F_R / parameters.T
    return 0.5 * np.log(parameters.Cao / Cai) / F_RT


def INa(
    V: npt.NDArray[np.floating],
    m: npt.NDArray[np.floating],
    h: npt.NDArray[np.floating],
    j: npt.NDArray[np.floating],
    Nai: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate the fast inward sodium current using the Beeler-Reuter representation of this term.

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV).
        m (npt.NDArray[np.floating]): Activation gating variable. Value(s) must be >= 0 and <= 1.
        h (npt.NDArray[np.floating]): Fast inactivation gating variable. Value(s) must be >= 0 and <= 1.
        j (npt.NDArray[np.floating]): Slow inactivation gating variable. Value(s) must be >= 0 and <= 1.
        Nai (npt.NDArray[np.floating]): Intracellular sodium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: Beeler Reuter fast inward sodium current [pA][pF]^{-1}.
    """
    assert_gt_numpy(m, 0, "m", strict=False)
    assert_lt_numpy(m, 1, "m", strict=False)
    assert_gt_numpy(h, 0, "h", strict=False)
    assert_lt_numpy(h, 1, "h", strict=False)
    assert_gt_numpy(j, 0, "j", strict=False)
    assert_lt_numpy(j, 1, "j", strict=False)
    return parameters.GNa * (m**3) * h * j * (V - ENa(Nai, parameters))


def INab(
    V: npt.NDArray[np.floating], Nai: npt.NDArray[np.floating], parameters: GWParameters
) -> npt.NDArray[np.floating]:
    """Background sodium current

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV).
        Nai (npt.NDArray[np.floating]): Intracellular sodium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: Backround sodium current [pA][pF]^{-1}.
    """
    return parameters.GNab * (V - ENa(Nai, parameters))


def IKr(
    V: npt.NDArray[np.floating],
    XKr: npt.NDArray[np.floating],
    Ki: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate the rapidly-activating delayed rectifier potassium current IKr.

    Uses the 5-state HERG Markov model to represent the ion channels, where state 4 is open and all others closed.

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV).
        XKr (npt.NDArray[np.floating]): Proportion of ion channels in the open state. Value(s) must be >= 0 and <= 1.
        Ki (npt.NDArray[np.floating]): Intracellular potassium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: Rapidly-activating delayed rectifier potassium current IKr [pA][pF]^{-1}.
    """
    assert_gt_numpy(XKr, 0, "XKr", strict=False)
    assert_lt_numpy(XKr, 1, "XKr", strict=False)
    RKr = 1 / (1 + 1.4945 * np.exp(0.0446 * V))
    return (
        0.5
        * parameters.GKr
        * np.sqrt(parameters.Ko)
        * RKr
        * XKr
        * (V - EK(Ki, parameters))
    )


def IKs(
    V: npt.NDArray[np.floating],
    xKs: npt.NDArray[np.floating],
    Ki: npt.NDArray[np.floating],
    Nai: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate the slow-activating delayed rectifier potassium current IKs.

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV).
        xKs (npt.NDArray[np.floating]): Activation gating variable for IKs. Value(s) must be >= 0 and <= 1.
        Ki (npt.NDArray[np.floating]): Intracellular potassium concentration [mM]. Value(s) must be > 0.
        Nai (npt.NDArray[np.floating]): Intracellular sodium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: Slow-activating delayed rectifier potassium current IKs [pA][pF]^{-1}.
    """
    assert_gt_numpy(xKs, 0, "xKs", strict=False)
    assert_lt_numpy(xKs, 1, "xKs", strict=False)
    return parameters.GKs * (xKs**2) * (V - EKs(Nai, Ki, parameters))


def IKv43(
    V: npt.NDArray[np.floating],
    XKv43: npt.NDArray[np.floating],
    Ki: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate the Kv4.3 componentent of the transient outward current Ito1.

    Uses the 10 state Kv4.3 Markov model to represent the ion channel state, where state 5 is open and all others are closed.

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV).
        XKv43 (npt.NDArray[np.floating]): Proportion of ion channels in the open state. Value(s) must be >= 0 and <= 1.
        Ki (npt.NDArray[np.floating]): Intracellular potassium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: Transient outward current through Kv4.3 channels IKv43 [pA][pF]^{-1}.
    """
    assert_gt_numpy(XKv43, 0, "XKv43", strict=False)
    assert_lt_numpy(XKv43, 1, "XKv43", strict=False)
    return parameters.GKv43 * XKv43 * (V - EK(Ki, parameters))


def IKv14(
    V: npt.NDArray[np.floating],
    XKv14: npt.NDArray[np.floating],
    Ki: npt.NDArray[np.floating],
    Nai: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate the Kv1.4 componentent of the transient outward current Ito1.

        Uses the 10 state Kv1.4 Markov model to represent the ion channel state, where state 5 is open and all others are closed.

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV).
        XKv14 (npt.NDArray[np.floating]): Proportion of ion channels in the open state. Value(s) must be >= 0 and <= 1.
        Ki (npt.NDArray[np.floating]): Intracellular potassium concentration [mM]. Value(s) must be > 0.
        Nai (npt.NDArray[np.floating]): Intracellular sodium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: Transient outward current through Kv1.4 channels IKv14 [pA][pF]^{-1}.
    """
    assert_gt_numpy(XKv14, 0, "XKv14", strict=False)
    assert_lt_numpy(XKv14, 1, "XKv14", strict=False)
    VF_RT = V * F_R / parameters.T
    expVF_RT = np.exp(VF_RT)
    multiplier = (
        (parameters.PKv14 / parameters.Csc) * F * VF_RT * XKv14 / (1 - expVF_RT)
    )
    return (
        multiplier
        * ((Ki - parameters.Ko * expVF_RT) + 0.02 * (Nai - parameters.Nao))
        * 1e9
    )


def IK1(
    V: npt.NDArray[np.floating], Ki: npt.NDArray[np.floating], parameters: GWParameters
) -> npt.NDArray[np.floating]:
    """Calculate the time independent potassium current IK1.

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV).
        Ki (npt.NDArray[np.floating]): Intracellular potassium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.
    Returns:
        npt.NDArray[np.floating]: Time independent potassium current IK1 [pA][pF]^{-1}.
    """
    EK_ = EK(Ki, parameters)
    K1inf = 1 / (2 + np.exp(1.5 * (V - EK_) * F_R / parameters.T))
    return (
        parameters.GK1
        * K1inf
        * (parameters.Ko / (parameters.Ko + parameters.KmK1))
        * (V - EK_)
    )


def IKp(
    V: npt.NDArray[np.floating], Ki: npt.NDArray[np.floating], parameters: GWParameters
) -> npt.NDArray[np.floating]:
    """Calculate the plateau potassium current IKp.

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV).
        Ki (npt.NDArray[np.floating]): Intracellular potassium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.
    Returns:
        npt.NDArray[np.floating]: Plateau potassium current IK1 [pA][pF]^{-1}.
    """
    Kp = 1 / (1 + np.exp((7.488 - V) / 5.98))
    return parameters.GKp * Kp * (V - EK(Ki, parameters))


def INaK(
    V: npt.NDArray[np.floating], Nai: npt.NDArray[np.floating], parameters: GWParameters
) -> npt.NDArray[np.floating]:
    """Calculate the current generated by the sodium-potassium pump.

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV).
        Nai (npt.NDArray[np.floating]): Intracellular sodium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: Sodium-potassium pump current INaK [pA][pF]^{-1}.
    """
    assert_gt_numpy(Nai, 0, "Nai", strict=True)
    sigma = (np.exp(parameters.Nao / 67.3) - 1) / 7
    fNaK = 1 / (
        1
        + 0.1245 * np.exp(-0.1 * V * F_R / parameters.T)
        + 0.0365 * sigma * np.exp(-V * F_R / parameters.T)
    )
    multiplier = parameters.INaKmax * parameters.Ko / (parameters.Ko + parameters.KmKo)
    return multiplier * fNaK / (1 + (parameters.KmNai / Nai) ** 1.5)


def INaCa(
    V: npt.NDArray[np.floating],
    Nai: npt.NDArray[np.floating],
    Cai: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate the current generated by the sodium-calcum exchanger (NCX).

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV).
        Nai (npt.NDArray[np.floating]): Intracellular sodium concentration [mM]. Value(s) must be > 0.
        Cai (npt.NDArray[np.floating]): Intracellular calcium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: NCX current INaCa [pA][pF]^{-1}.
    """
    assert_gt_numpy(Nai, 0, "Nai", strict=True)
    assert_gt_numpy(Cai, 0, "Cai", strict=True)
    exp_term1 = np.exp(parameters.eta * V * F_R / parameters.T)
    exp_term2 = np.exp((parameters.eta - 1) * V * F_R / parameters.T)
    multiplier = (
        5000
        * parameters.kNaCa
        / (
            (parameters.KmNa**3 + parameters.Nao**3)
            * (parameters.KmCa + parameters.Cao)
        )
    )
    return (
        multiplier
        * (
            exp_term1 * (Nai**3) * parameters.Cao
            - exp_term2 * (parameters.Nao**3) * Cai
        )
        / (1 + parameters.ksat * exp_term2)
    )


def IpCa(
    Cai: npt.NDArray[np.floating], parameters: GWParameters
) -> npt.NDArray[np.floating]:
    """Calculate the current generated by background calcium pumps.

    Args:
        Cai (npt.NDArray[np.floating]): Intracellular calcium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: Calcium pump current IpCa
    """
    assert_gt_numpy(Cai, 0, "Cai", strict=True)
    return parameters.IpCamax * (Cai / (parameters.KmpCa + Cai))


def ICab(
    V: npt.NDArray[np.floating], Cai: npt.NDArray[np.floating], parameters: GWParameters
) -> npt.NDArray[np.floating]:
    """Background calcium current.

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV).
        Cai (npt.NDArray[np.floating]): Intracellular calcium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: Backround calcium current [pA][pF]^{-1}.
    """
    return parameters.GCab * (V - ECa(Cai, parameters))


def ICaL(
    V: npt.NDArray[np.floating],
    LCC: npt.NDArray[np.integer],
    LCC_inactivation: npt.NDArray[np.integer],
    CaSS: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate the L-type calcium channel (LCC) current for the Greenstein and Winslow model with nCRU calcium release units for any nCaRU > 0.

    The LCC states in are represented by integers 1-12, where 6 and 12 are open states, and all others are closed.
    The LCCs may also become voltage inactivated, represented by the values of LCC_inactivation (0 = inactive, 1 = active).

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV).
        LCC (npt.NDArray[np.integer]): NxNCaRUx4 array of LCC states.
        LCC_inactivation (npt.NDArray[np.integer]): NxNCaRUx4 array of LCC inactivation states.
        CaSS (npt.NDArray): NxNCaRUx4 array of subspace calcium concentrations [mM]. Values must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: Total current through the L-type calcium channels ICaL [pA][pF]^{-1}.
    """
    assert_gt_numpy(CaSS, 0, "CaSS", strict=True)
    __catch_invalid_state(LCC, 1, 12, "LCC")
    __catch_invalid_state(LCC_inactivation, 0, 1, "LCC_inactivation")
    open_LCC = ((LCC == 6) | (LCC == 12)) & (LCC_inactivation == 1)
    VF_RT = V[..., np.newaxis, np.newaxis] * F_R / parameters.T
    exp2VF_RT = np.exp(2 * VF_RT)
    multiplier = (
        (parameters.NCaRU / parameters.NCaRU_sim)
        * (parameters.PCaL / parameters.CSA)
        * (4 * F * VF_RT)
        / (exp2VF_RT - 1)
    )
    ICaL_individual_channels = (
        multiplier * open_LCC * (CaSS * exp2VF_RT - 0.341 * parameters.Cao)
    )
    return ICaL_individual_channels.sum(axis=(1, 2)) * 1e9


def Ito2(
    V: npt.NDArray[np.floating], ClCh: npt.NDArray, parameters: GWParameters
) -> npt.NDArray[np.floating]:
    """Calculate the calcium activated chloride (ClCh) transient outward current Ito2 for the Greenstein and Winslow model with nCRU calcium release units for any nCaRU > 0.

    The ClCh ion channel states are represented by a binary value (0 = closed, 1 = open).

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV).
        ClCh (npt.NDArray): NxNCaRUx4 array of ClCh states.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: Total current through the calcium activated chloride channels Ito2 [pA][pF]^{-1}.
    """
    __catch_invalid_state(ClCh, 0, 1, "ClCh")
    VF_RT = V[..., np.newaxis, np.newaxis] * F_R / parameters.T
    expmVF_RT = np.exp(-VF_RT)
    multiplier = (
        (parameters.NCaRU / parameters.NCaRU_sim)
        * (parameters.Pto2 / parameters.CSA)
        * (F * VF_RT)
        / (expmVF_RT - 1)
    )
    Ito2_individual_channels = (
        multiplier * ClCh * (parameters.Clcyto * expmVF_RT - parameters.Clo)
    )
    return Ito2_individual_channels.sum(axis=(1, 2)) * 1e9


def dCaLTRPN(
    CaLTRPN: npt.NDArray[np.floating],
    Cai: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate the instentaneous flux of calcium bound to low-affinity troponin sites (LTRPN).

    Args:
        CaLTRPN (npt.NDArray[np.floating]): Concentration of calcium bound to LTRPN [mM]. Value(s) must be > 0.
        Cai (npt.NDArray[np.floating]): Intracellular calcium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: Instentaneous flux of calcium bound to LTRPN, dCaLTRPN/dt [mM][ms]^{-1}.
    """
    assert_gt_numpy(Cai, 0, "Cai", strict=True)
    assert_gt_numpy(CaLTRPN, 0, "CaLTRPN", strict=True)
    return (
        parameters.kLTRPNp * Cai * (parameters.LTRPNtot - CaLTRPN)
        - parameters.kLTRPNm * CaLTRPN
    )


def dCaHTRPN(
    CaHTRPN: npt.NDArray[np.floating],
    Cai: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate the instentaneous flux of calcium bound to high-affinity troponin sites (HTRPN).

    Args:
        CaHTRPN (npt.NDArray[np.floating]): Concentration of calcium bound to HTRPN [mM]. Value(s) must be > 0.
        Cai (npt.NDArray[np.floating]): Intracellular calcium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: Instentaneous flux of calcium bound to HTRPN, dCaHTRPN/dt [mM][ms]^{-1}.
    """
    assert_gt_numpy(Cai, 0, "Cai", strict=True)
    assert_gt_numpy(CaHTRPN, 0, "CaHTRPN", strict=True)
    return (
        parameters.kHTRPNp * Cai * (parameters.HTRPNtot - CaHTRPN)
        - parameters.kHTRPNm * CaHTRPN
    )


def Jup(
    Cai: npt.NDArray[np.floating],
    CaNSR: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate the SERCA2a mediated uptake from the intracellular space to the network sarcoplasmic reticulum (NSR).

    Args:
        Cai (npt.NDArray[np.floating]): Intracellular calcium concentration [mM]. Value(s) must be > 0.
        CaNSR (npt.NDArray[np.floating]): NSR calcium concentration [mM]. Value(s) must be > 0.
        parameters (GWParameters): Greenstein and Winslow model parameters.

    Returns:
        npt.NDArray[np.floating]: SERCA2a mediated calcium uptake flux Jup [mM][ms]^{-1}.
    """
    assert_gt_numpy(Cai, 0, "Cai", strict=True)
    assert_gt_numpy(CaNSR, 0, "CaNSR", strict=True)
    f = (Cai / parameters.Kmf) ** parameters.Hf
    r = (CaNSR / parameters.Kmr) ** parameters.Hr
    return (parameters.Vmaxf * f - parameters.Vmaxr * r) / (1 + f + r)


class GWSolution:
    """Container class for the results of simulating the Greenstein and Winslow class."""

    def __init__(self, gw_cxx_output: gw_cxx.GWVariables, gw_parameters: GWParameters):
        """
        Args:
            gw_cxx_output (gw.GWVariables): C++ struct holding snapshots of the model state across time.
            gw_parameters (GWParameters): Model parameters used to simulate the model realisation. Required for calculating the currents.
        """
        self.__vars: GWVairablesCXX = gw_cxx_output
        self.__params: GWParameters = gw_parameters
        self.__INa: npt.NDArray | None = None
        self.__INab: npt.NDArray | None = None
        self.__INaCa: npt.NDArray | None = None
        self.__INaK: npt.NDArray | None = None
        self.__IKr: npt.NDArray | None = None
        self.__IKs: npt.NDArray | None = None
        self.__IKv14: npt.NDArray | None = None
        self.__IKv43: npt.NDArray | None = None
        self.__Ito2: npt.NDArray | None = None
        self.__IK1: npt.NDArray | None = None
        self.__IKp: npt.NDArray | None = None
        self.__ICaL: npt.NDArray | None = None
        self.__ICab: npt.NDArray | None = None
        self.__IpCa: npt.NDArray | None = None
        self.__Jup: npt.NDArray | None = None
        self.__Jtr: npt.NDArray | None = None
        self.__Jxfer: npt.NDArray | None = None

    @classmethod
    def from_dict(cls, state_dict: dict):
        """Construct from dict

        Args:
            state_dict (dict): Dictionary of model variables
        """
        _vars = state_dict["vars"]
        _params = state_dict["params"]

        NCaRU = _params["NCaRU_sim"]
        t = _vars["t"]
        cxx_vars = gw_cxx.GWVariables(NCaRU, t.shape[0], t[-1])
        cxx_vars.t = t
        cxx_vars.V = _vars["V"]
        cxx_vars.Nai = _vars["Nai"]
        cxx_vars.Ki = _vars["Ki"]
        cxx_vars.Cai = _vars["Cai"]
        cxx_vars.CaNSR = _vars["CaNSR"]
        cxx_vars.CaLTRPN = _vars["CaLTRPN"]
        cxx_vars.CaHTRPN = _vars["CaHTRPN"]
        cxx_vars.m = _vars["m"]
        cxx_vars.h = _vars["h"]
        cxx_vars.j = _vars["j"]
        cxx_vars.xKs = _vars["xKs"]
        cxx_vars.XKr = _vars["XKr"]
        cxx_vars.XKv14 = _vars["XKv14"]
        cxx_vars.XKv43 = _vars["XKv43"]
        cxx_vars.CaSS = _vars["CaSS"]
        cxx_vars.CaJSR = _vars["CaJSR"]
        cxx_vars.LCC = _vars["LCC"]
        cxx_vars.LCC_inactivation = _vars["LCC_inactivation"]
        cxx_vars.RyR = _vars["RyR"]
        cxx_vars.ClCh = _vars["ClCh"]

        params = GWParameters.from_dict(_params)
        return cls(cxx_vars, params)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        params_dict = self.__params.to_dict()
        vars_dict = {
            "t": self.t,
            "V": self.V,
            "Nai": self.Nai,
            "Ki": self.Ki,
            "Cai": self.Cai,
            "CaNSR": self.CaNSR,
            "CaLTRPN": self.CaLTRPN,
            "CaHTRPN": self.CaHTRPN,
            "m": self.m,
            "h": self.h,
            "j": self.j,
            "xKs": self.xKs,
            "XKr": self.XKr,
            "XKv14": self.XKv14,
            "XKv43": self.XKv43,
            "CaSS": self.CaSS,
            "CaJSR": self.CaJSR,
            "LCC": self.LCC,
            "LCC_inactivation": self.LCC_inactivation,
            "RyR": self.RyR,
            "ClCh": self.ClCh,
        }
        return {"params": params_dict, "vars": vars_dict}

    def save(self, fname: str) -> None:
        state_dict = self.to_dict()
        with open(fname, "wb") as f:
            pkl.dump(state_dict, f)

    @classmethod
    def load(cls, fname: str):
        with open(fname, "rb") as f:
            state_dict = pkl.load(f)
        return cls.from_dict(state_dict)

    @property
    def parameters(self) -> GWParameters:
        """GWParameters: Model parameters."""
        return self.__params

    @property
    def t(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of recording times."""
        return self.__vars.t

    @property
    def V(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of action potential recordings [mV]."""
        return self.__vars.V

    @property
    def m(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of INa activation gating variable recordings."""
        return self.__vars.m

    @property
    def h(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of INa fast inactivation gating variable recordings."""
        return self.__vars.h

    @property
    def j(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of INa slow inactivation gating variable recordings."""
        return self.__vars.j

    @property
    def Nai(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of intracellular sodium recordings [mM]."""
        return self.__vars.Nai

    @property
    def Ki(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of intracellular potassium recordings [mM]."""
        return self.__vars.Ki

    @property
    def Cai(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of intracellular calcium recordings [mM]."""
        return self.__vars.Cai

    @property
    def CaNSR(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of NSR calcium recordings [mM]."""
        return self.__vars.CaNSR

    @property
    def CaLTRPN(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of LTRPN calcium recordings [mM]."""
        return self.__vars.CaLTRPN

    @property
    def CaHTRPN(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of HTRPN calcium recordings [mM]."""
        return self.__vars.CaHTRPN

    @property
    def xKs(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of IKs activation gating variable recordings."""
        return self.__vars.xKs

    @property
    def XKr(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 2D array of IKr ion channel Markov model recordings. Shape is num_step x 5."""
        return self.__vars.XKr

    @property
    def XKv14(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 2D array of IKv1.4 ion channel Markov model recordings. Shape is num_step x 10."""
        return self.__vars.XKv14

    @property
    def XKv43(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 2D array of IKv4.3 ion channel Markov model recordings. Shape is num_step x 10."""
        return self.__vars.XKv43

    @property
    def CaSS(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 3D array of dyadic subspace calcium concentration recordings [mM]. Shape is num_step x NCaRU_sim x 4."""
        return self.__vars.CaSS

    @property
    def CaJSR(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 2D array of JSR calcium concentration recordings [mM]. Shape is num_step x NCaRU_sim."""
        return self.__vars.CaJSR

    @property
    def LCC(self) -> npt.NDArray[np.integer]:
        """npt.NDArray[np.floating]: 3D array of LCC state (not including voltage inactivation) recordings. Shape is num_step x NCaRU_sim x 4."""
        return self.__vars.LCC

    @property
    def LCC_inactivation(self) -> npt.NDArray[np.integer]:
        """npt.NDArray[np.floating]: 3D array of LCC voltage inactivation state recordings. Shape is num_step x NCaRU_sim x 4."""
        return self.__vars.LCC_inactivation

    @property
    def RyR(self) -> npt.NDArray[np.integer]:
        """npt.NDArray[np.floating]: 4D array of the state of the 5 RyRs in each CaRU subunit. Shape is num_step x NCaRU_sim x 4 x 6.

        Note that unlike the other ion channel states, RyR[time, i, j, k] is the number of RyRs in state k in subunit j of CaRU i.
        """
        return self.__vars.RyR

    @property
    def ClCh(self) -> npt.NDArray[np.integer]:
        """npt.NDArray[np.floating]: 3D array of ClCh state recordings. Shape is num_step x NCaRU_sim x 4."""
        return self.__vars.ClCh

    @property
    def INa(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of INa recordings [pA][pF]^{-1}."""
        if self.__INa is None:
            self.__INa = INa(self.V, self.m, self.h, self.j, self.Nai, self.parameters)
        return self.__INa

    @property
    def INab(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of INab recordings [pA][pF]^{-1}."""
        if self.__INab is None:
            self.__INab = INab(self.V, self.Nai, self.parameters)
        return self.__INab

    @property
    def INaCa(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of INaCa recordings [pA][pF]^{-1}."""
        if self.__INaCa is None:
            self.__INaCa = INaCa(self.V, self.Nai, self.Cai, self.parameters)
        return self.__INaCa

    @property
    def INaK(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of INaK recordings [pA][pF]^{-1}."""
        if self.__INaK is None:
            self.__INaK = INaK(self.V, self.Nai, self.parameters)
        return self.__INaK

    @property
    def IKr(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of IKr recordings [pA][pF]^{-1}."""
        if self.__IKr is None:
            self.__IKr = IKr(self.V, self.XKr[..., 3], self.Ki, self.parameters)
        return self.__IKr

    @property
    def IKs(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of IKs recordings [pA][pF]^{-1}."""
        if self.__IKs is None:
            self.__IKs = IKs(self.V, self.xKs, self.Ki, self.Nai, self.parameters)
        return self.__IKs

    @property
    def IKv14(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of IKv14 recordings [pA][pF]^{-1}."""
        if self.__IKv14 is None:
            self.__IKv14 = IKv14(
                self.V, self.XKv14[..., 4], self.Ki, self.Nai, self.parameters
            )
        return self.__IKv14

    @property
    def IKv43(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of IKv43 recordings [pA][pF]^{-1}."""
        if self.__IKv43 is None:
            self.__IKv43 = IKv43(self.V, self.XKv43[..., 4], self.Ki, self.parameters)
        return self.__IKv43

    @property
    def Ito1(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of Ito1 recordings [pA][pF]^{-1}."""
        return self.IKv14 + self.IKv43

    @property
    def Ito2(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of Ito2 recordings [pA][pF]^{-1}."""
        if self.__Ito2 is None:
            self.__Ito2 = Ito2(self.V, self.ClCh, self.parameters)
        return self.__Ito2

    @property
    def IK1(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of IK1 recordings [pA][pF]^{-1}."""
        if self.__IK1 is None:
            self.__IK1 = IK1(self.V, self.Ki, self.parameters)
        return self.__IK1

    @property
    def IKp(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of IKp recordings [pA][pF]^{-1}."""
        if self.__IKp is None:
            self.__IKp = IKp(self.V, self.Ki, self.parameters)
        return self.__IKp

    @property
    def ICaL(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of ICaL recordings [pA][pF]^{-1}."""
        if self.__ICaL is None:
            self.__ICaL = ICaL(
                self.V, self.LCC, self.LCC_inactivation, self.CaSS, self.parameters
            )
        return self.__ICaL

    @property
    def ICab(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of ICab recordings [pA][pF]^{-1}."""
        if self.__ICab is None:
            self.__ICab = ICab(self.V, self.Cai, self.parameters)
        return self.__ICab

    @property
    def IpCa(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of IpCa recordings [pA][pF]^{-1}."""
        if self.__IpCa is None:
            self.__IpCa = IpCa(self.Cai, self.parameters)
        return self.__IpCa

    @property
    def Jup(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of Jup recordings [mM][ms]^{-1}."""
        if self.__Jup is None:
            self.__Jup = Jup(self.Cai, self.CaNSR, self.parameters)
        return self.__Jup

    @property
    def Jtr(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of Jtr recordings [mM][ms]^{-1}."""
        if self.__Jtr is None:
            self.__Jtr = self.parameters.rtr * np.sum(self.CaNSR - self.CaJSR, axis=1)
        return self.__Jtr

    @property
    def Jxfer(self) -> npt.NDArray[np.floating]:
        """npt.NDArray[np.floating]: 1D array of Jxfer recordings [mM][ms]^{-1}."""
        if self.__Jxfer is None:
            self.__Jxfer = self.parameters.rxfer * np.sum(
                self.CaSS - self.Cai, axis=(1, 2)
            )
        return self.__Jxfer

    @property
    def int_QTXt(self) -> npt.NDArray[np.floating]:
        """Integral of f_{open}(Q^T(s)RyR_s), where f_{open} is the linear functional giving the number of open RyRs"""
        return self.__vars.int_QTXt


class GWModel:
    """Simulates the Greenstein and Winslow model with specified parameters and stimulus."""

    @classmethod
    def PRNG_options(cls) -> List[str]:
        """Get the available arguments for PRNG in the simulate method.

        Returns:
            List[str]: Implemented PRNGs for simulating the model.
        """
        return __IMPLEMENTED_PRNGS

    def __init__(
        self,
        init_state: None | dict = None,
        parameters: None | GWParameters = None,
        stimulus_fn: None | Callable[[float], float] = None,
    ):
        """
        Args:
            parameters (None | GWParameters, optional): Model parameters. Defaults to None, in which case the default parameters are used.
            stimulus_fn (None | Callable[[float], float], optional): Time dependent external stimulus in pA/pF. Defaults to the zero function.
        """
        self.parameters: GWParameters = (
            parameters if parameters is not None else GWParameters()
        )
        """GWParameters: Model parameters."""
        self.__stim: Callable[[float], float] = (
            (lambda t: 0) if stimulus_fn is None else stimulus_fn
        )

        if init_state is not None:
            self.__global_state = _unpack_globals(init_state)
            self.__cru_state = _unpack_crus(init_state)
        else:
            self.__global_state = gw_cxx.GWGlobalState()
            self.__cru_state = gw_cxx.GWCRUState(self.parameters.NCaRU_sim)

    def simulate(
        self,
        step_size: float,
        num_steps: int,
        record_every: int = 1,
        PRNG: str = "xoshiro256++",
    ) -> GWSolution:
        """Simulate the model with a step size of step_size over num_steps steps.

        Args:
            step_size (float): Size of the integrator time step (ms). Value must be > 0.
            num_steps (int): Number of steps that the integrator should take. Value must be > 0.
            record_every (int, optional): Number of steps between recording snapshots of the state. Value must be > 0. Defaults to 1.
            PRNG (str, optional): PRNG to use within the algorithm. Call GWModel.PRNG_options() to get a list of options. Default is 'xoshiro256++'.

        Returns:
            GWSolution: Snapshots of state through simulation.
        """
        assert_positive(step_size, "step_size")
        assert_positive(num_steps, "num_steps")
        assert_positive(record_every, "record_every")
        try:
            cxx_sol = gw_cxx.run(
                self.parameters.cxx_struct,
                self.parameters.NCaRU_sim,
                step_size,
                num_steps,
                self.__stim,
                record_every,
                init_crus=self.__cru_state,
                init_globals=self.__global_state,
                PRNG=PRNG,
            )
        except Exception as e:
            if isinstance(e, ValueError):
                raise ValueError(
                    f"{PRNG} is an invalid argument for PRNG.\nAvailable options are: "
                    + reduce(lambda x, y: x + ", " + y, self.PRNG_options())
                ) from e
            else:
                raise e

        return GWSolution(cxx_sol, self.parameters)

    def stimulus_fn(self, t: float) -> float:
        """Evaluate the stimulus function Istim.

        Args:
            t (float): time (ms).

        Returns:
            float: Stimulus at time t.
        """
        return self.__stim(t)

    def set_stimulus_fn(self, stimulus_fn: Callable[[float], float]) -> None:
        """Set the stimulus function.

        Args:
            stimulus_fn (Callable[[float], float]): Stimulus function.
        """
        self.__stim = stimulus_fn


def _unpack_globals(state_dict: dict) -> gw_cxx.GWGlobalState:
    state = gw_cxx.GWGlobalState()
    state.V = state_dict["V"]
    state.Nai = state_dict["Nai"]
    state.Ki = state_dict["Ki"]
    state.Cai = state_dict["Cai"]
    state.CaNSR = state_dict["CaNSR"]
    state.CaLTRPN = state_dict["CaLTRPN"]
    state.CaHTRPN = state_dict["CaHTRPN"]

    state.m = state_dict["m"]
    state.h = state_dict["h"]
    state.j = state_dict["j"]
    state.xKs = state_dict["xKs"]

    state.XKr = state_dict["XKr"]
    state.XKv14 = state_dict["XKv14"]
    state.XKv43 = state_dict["XKv43"]

    return state


def _unpack_crus(state_dict: dict) -> gw_cxx.GWCRUState:
    NCaRU = state_dict["CaSS"].shape[0]
    state = gw_cxx.GWCRUState(NCaRU)
    state.CaSS = state_dict["CaSS"]
    state.CaJSR = state_dict["CaJSR"]
    state.LCC = state_dict["LCC"]
    state.LCC_inactivation = state_dict["LCC_inactivation"]
    state.RyR = state_dict["RyR"]
    state.ClCh = state_dict["ClCh"]
    return state
