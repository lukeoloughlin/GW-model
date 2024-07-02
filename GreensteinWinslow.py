from typing import Any, Callable, List
from functools import reduce

import numpy as np
import numpy.typing as npt

import build.GreensteinWinslow as gw  # type: ignore
from GWParameters import GWParameters, assert_positive
#from from_cpp_struct import from_cpp_struct

F = 96.5
R = 8.314
F_R = F / R

GWParametersCXX = type[gw.Parameters]
GWVairablesCXX = type[gw.GWVariables]

IMPLEMENTED_PRNGS = [
    "mt19937",
    "mt19937_64",
    "xoshiro256+",
    "xoshiro256++",
    "xoshiro256**",
    "xoroshiro128+",
    "xoroshiro128++",
    "xoroshiro128**",
]



#@from_cpp_struct(gw.Parameters, additional_attributes=(("NCaRU_sim", 1250),))
#class GWParameters(GWParametersBaseClass):
#    """Holds the model parameters for the Greenstein and Winslow model."""




def ENa(
    Nai: npt.NDArray[np.floating], parameters: GWParameters
) -> npt.NDArray[np.floating]:
    """Calculates the Nernst potential for sodium ions

    Args:
        Nai (npt.NDArray[np.floating]): Intracellular sodium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: Nernst potential of sodium
    """
    F_RT = F_R / parameters.T
    return np.log(parameters.Nao / Nai) / F_RT


def EK(
    Ki: npt.NDArray[np.floating], parameters: GWParameters
) -> npt.NDArray[np.floating]:
    """Calculate the Nernst potential for potassium ions

    Args:
        Ki (npt.NDArray[np.floating]): Intracellular potassium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: Nernst potential of potassium
    """
    F_RT = F_R / parameters.T
    return np.log(parameters.Ko / Ki) / F_RT


def EKs(
    Nai: npt.NDArray[np.floating],
    Ki: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate Nernst potential for slow rectifier current IKs, which is permeable to potassium and sodium

    Args:
        Nai (npt.NDArray[np.floating]): Intracellular sodium concentration (mM)
        Ki (npt.NDArray[np.floating]): Intracellular potassium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: Nernst potential of slow rectifier current IKs
    """
    F_RT = F_R / parameters.T
    return (
        np.log((parameters.Ko + 0.01833 * parameters.Nao) / (Ki + 0.01833 * Nai)) / F_RT
    )


def ECa(
    Cai: npt.NDArray[np.floating], parameters: GWParameters
) -> npt.NDArray[np.floating]:
    """Calculate the Nernst potential for calcium ions

    Args:
        Cai (npt.NDArray[np.floating]): Intracellular calcium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: Nernst potential of calcium
    """
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
    """Calculate the fast inward sodium current using the Beeler-Reuter representation of this term

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV)
        m (npt.NDArray[np.floating]): Activation gating variable
        h (npt.NDArray[np.floating]): Fast inactivation gating variable
        j (npt.NDArray[np.floating]): Slow inactivation gating variable
        Nai (npt.NDArray[np.floating]): Intracellular sodium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: Beeler Reuter fast inward sodium current (pA/pF)
    """
    return parameters.GNa * (m**3) * h * j * (V - ENa(Nai, parameters))


def INab(
    V: npt.NDArray[np.floating], Nai: npt.NDArray[np.floating], parameters: GWParameters
) -> npt.NDArray[np.floating]:
    """Background sodium current

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV)
        Nai (npt.NDArray[np.floating]): Intracellular sodium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: Backround sodium current (pA/pF)
    """
    return parameters.GNab * (V - ENa(Nai, parameters))


def IKr(
    V: npt.NDArray[np.floating],
    XKr: npt.NDArray[np.floating],
    Ki: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate the rapidly-activating delayed rectifier potassium current IKr

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV)
        XKr (npt.NDArray[np.floating]): Proportion of HERG ion channels in the open state (state 4)
        Ki (npt.NDArray[np.floating]): Intracellular potassium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: Rapidly-activating delayed rectifier potassium current IKr (pA/pF)
    """
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
    XKs: npt.NDArray[np.floating],
    Ki: npt.NDArray[np.floating],
    Nai: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate the slow-activating delayed rectifier potassium current IKs

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV)
        XKs (npt.NDArray[np.floating]): Activation gating variable for IKs
        Ki (npt.NDArray[np.floating]): Intracellular potassium concentration (mM)
        Nai (npt.NDArray[np.floating]): Intracellular sodium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: Slow-activating delayed rectifier potassium current IKs (pA/pF)
    """
    return parameters.GKs * (XKs**2) * (V - EKs(Nai, Ki, parameters))


def IKv43(
    V: npt.NDArray[np.floating],
    XKv43: npt.NDArray[np.floating],
    Ki: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate the Kv4.3 componentent of the transient outward current Ito1

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV)
        XKv43 (npt.NDArray[np.floating]): Proportion of Kv4.3 ion channels in the open state (state 5)
        Ki (npt.NDArray[np.floating]): Intracellular potassium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: Transient outward current through Kv4.3 channels IKv43 (pA/pF)
    """
    return parameters.GKv43 * XKv43 * (V - EK(Ki, parameters))


def IKv14(
    V: npt.NDArray[np.floating],
    XKv14: npt.NDArray[np.floating],
    Ki: npt.NDArray[np.floating],
    Nai: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate the Kv1.4 componentent of the transient outward current Ito1

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV)
        XKv14 (npt.NDArray[np.floating]): Proportion of Kv4.3 ion channels in the open state (state 5)
        Ki (npt.NDArray[np.floating]): Intracellular potassium concentration (mM)
        Nai (npt.NDArray[np.floating]): Intracellular sodium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: Transient outward current through Kv1.4 channels IKv14 (pA/pF)
    """
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
    """Calculate the time independent potassium current IK1

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV)
        Ki (npt.NDArray[np.floating]): Intracellular potassium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters
    Returns:
        npt.NDArray[np.floating]: Time independent potassium current IK1 (pA/pF)
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
    """Calculate the plateau potassium current IKp

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV)
        Ki (npt.NDArray[np.floating]): Intracellular potassium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters
    Returns:
        npt.NDArray[np.floating]: Plateau potassium current IK1 (pA/pF)
    """
    Kp = 1 / (1 + np.exp((7.488 - V) / 5.98))
    return parameters.GKp * Kp * (V - EK(Ki, parameters))


def INaK(
    V: npt.NDArray[np.floating], Nai: npt.NDArray[np.floating], parameters: GWParameters
) -> npt.NDArray[np.floating]:
    """Calculate the current generated by the sodium-potassium pump

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV)
        Nai (npt.NDArray[np.floating]): Intracellular sodium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: Sodium-potassium pump current INaK (pA/pF)
    """
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
    """Calculate the current generated by the sodium-calcum exchanger (NCX)

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV)
        Nai (npt.NDArray[np.floating]): Intracellular sodium concentration (mM)
        Cai (npt.NDArray[np.floating]): Intracellular calcium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: NCX current INaCa (pA/pF)
    """
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
    """Calculate the current generated by background calcium pumps

    Args:
        Cai (npt.NDArray[np.floating]): Intracellular calcium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: Calcium pump current IpCa
    """
    return parameters.IpCamax * (Cai / (parameters.KmpCa + Cai))


def ICab(
    V: npt.NDArray[np.floating], Cai: npt.NDArray[np.floating], parameters: GWParameters
) -> npt.NDArray[np.floating]:
    """Background calcium current

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV)
        Cai (npt.NDArray[np.floating]): Intracellular calcium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: Backround calcium current (pA/pF)
    """
    return parameters.GCab * (V - ECa(Cai, parameters))


def ICaL(
    V: npt.NDArray[np.floating],
    LCC: npt.NDArray[np.integer],
    LCC_inactivation: npt.NDArray[np.integer],
    CaSS: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate the L-type calcium channel current for the Greenstein and Winslow model with nCRU calcium release units for any nCRU > 0

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV)
        LCC (npt.NDArray[np.integer]): NxnCRUx4 array of L-type calcium channel activation states, represented by an integer 1-12
        LCC_inactivation (npt.NDArray[np.integer]): NxnCRUx4 array of L-type calcium channel inactivation states, active/inactive are given by 0/1 respectively
        CaSS (npt.NDArray): NxnCRUx4 array of subspace calcium concentrations (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: Total current through the L-type calcium channels ICaL (pA/pF)
    """
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
    """Calculate the calcium activated chloride transient outward current Ito2 for the Greenstein and Winslow model with nCRU calcium release units for any nCRU > 0

    Args:
        V (npt.NDArray[np.floating]): Action potential across cell membrane (mV)
        ClCh (npt.NDArray): NxnCRUx4 array of calcium activated chloride channel activation states, where active/inactive are given by 0/1 respectively
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: Total current through the calcium activated chloride channels Ito2 (pA/pF)
    """
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
    """Calculate the instentaneous flux of calcium bound to low-affinity troponin sites (LTRPN)

    Args:
        CaLTRPN (npt.NDArray[np.floating]): Concentration of calcium bound to LTRPN (mM)
        Cai (npt.NDArray[np.floating]): Intracellular calcium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: Instentaneous flux of calcium bound to LTRPN, dCaLTRPN/dt (mM/ms)
    """
    return (
        parameters.kLTRPNp * Cai * (parameters.LTRPNtot - CaLTRPN)
        - parameters.kLTRPNm * CaLTRPN
    )


def dCaHTRPN(
    CaHTRPN: npt.NDArray[np.floating],
    Cai: npt.NDArray[np.floating],
    parameters: GWParameters,
) -> npt.NDArray[np.floating]:
    """Calculate the instentaneous flux of calcium bound to high-affinity troponin sites (HTRPN)

    Args:
        CaHTRPN (npt.NDArray[np.floating]): Concentration of calcium bound to HTRPN (mM)
        Cai (npt.NDArray[np.floating]): Intracellular calcium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: Instentaneous flux of calcium bound to HTRPN, dCaHTRPN/dt (mM/ms)
    """
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
        Cai (npt.NDArray[np.floating]): Intracellular calcium concentration (mM)
        CaNSR (npt.NDArray[np.floating]): NSR calcium concentration (mM)
        parameters (GWParameters): Greenstein and Winslow model parameters

    Returns:
        npt.NDArray[np.floating]: SERCA2a mediated calcium uptake flux Jup (mM/ms)
    """
    f = (Cai / parameters.Kmf) ** parameters.Hf
    r = (CaNSR / parameters.Kmr) ** parameters.Hr
    return (parameters.Vmaxf * f - parameters.Vmaxr * r) / (1 + f + r)


class GWSolution:
    """Container class for the results of simulating the Greenstein and Winslow class.
    Essentially just wraps a C++ struct that holds the recorded states in numpy arrays,
    and provides convenitent access to the model currents and fluxes.
    """

    def __init__(self, gw_cpp_output: gw.GWVariables, gw_parameters: GWParameters):
        """Create the solution container from the associated C++ structs.

        Args:
            gw_cpp_output (gw.GWVariables): C++ struct holding snapshots of the model state across time
            gw_parameters (GWParameters): Model parameters used to simulate the model realisation. Required for calculating the currents.
        """
        self.__vars: GWVairablesCXX = gw_cpp_output
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


    @property
    def parameters(self) -> GWParameters:
        """
        Returns:
            GWParameters: Model parameters
        """
        return self.__params

    @property
    def t(self) -> npt.NDArray[np.floating]:
        """Times of snapshot recordings.

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        return self.__vars.t

    @property
    def V(self) -> npt.NDArray[np.floating]:
        """Recordings of the action potential (mV).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        return self.__vars.V

    @property
    def m(self) -> npt.NDArray[np.floating]:
        """Recordings of activation gate for fast inward sodium current.

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        return self.__vars.m

    @property
    def h(self) -> npt.NDArray[np.floating]:
        """Recordings of fast inactivation gate for fast inward sodium current.

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        return self.__vars.h

    @property
    def j(self) -> npt.NDArray[np.floating]:
        """Recordings of slow inactivation gate for fast inward sodium current.

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        return self.__vars.j

    @property
    def Nai(self) -> npt.NDArray[np.floating]:
        """Recordings of the intracellular sodium concentration (mM).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        return self.__vars.Nai

    @property
    def Ki(self) -> npt.NDArray[np.floating]:
        """Recordings of the intracellular potassium concentration (mM).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        return self.__vars.Ki

    @property
    def Cai(self) -> npt.NDArray[np.floating]:
        """Recordings of the intracellular calcium concentration (mM).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        return self.__vars.Cai

    @property
    def CaNSR(self) -> npt.NDArray[np.floating]:
        """Recordings of the network sarcoplasmic reticulum calcium concentration (mM).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        return self.__vars.CaNSR

    @property
    def CaLTRPN(self) -> npt.NDArray[np.floating]:
        """Recordings of calcium concentration at low affinity troponin sites (mM).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        return self.__vars.CaLTRPN

    @property
    def CaHTRPN(self) -> npt.NDArray[np.floating]:
        """Recordings of calcium concentration at high affinity troponin sites (mM).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        return self.__vars.CaHTRPN

    @property
    def xKs(self) -> npt.NDArray[np.floating]:
        """Recordings of the activation gating variable for the slow-activating delayed rectifier
        current IKs.

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        return self.__vars.xKs

    @property
    def XKr(self) -> npt.NDArray[np.floating]:
        """Recordings of the 5 state representation of HERG ion channels for the current IKr.

        Returns:
            npt.NDArray[np.floating]: 2D array of size num_steps x 5
        """
        return self.__vars.XKr

    @property
    def XKv14(self) -> npt.NDArray[np.floating]:
        """Recordings of the 10 state representation of Kv1.4 ion channels for the current IKv1.4.

        Returns:
            npt.NDArray[np.floating]: 2D array of size num_steps x 10
        """
        return self.__vars.XKv14

    @property
    def XKv43(self) -> npt.NDArray[np.floating]:
        """Recordings of the 10 state representation of Kv4.3 ion channels for the current IKv4.3.

        Returns:
            npt.NDArray[np.floating]: 2D array of size num_steps x 10
        """
        return self.__vars.XKv43

    @property
    def CaSS(self) -> npt.NDArray[np.floating]:
        """Recordings of dyadic subspace calcium concetrations (mM) for all CRU subunits.
        Returns:
            npt.NDArray[np.floating]: 3D array of size num_steps x num_CRU x 4
        """
        return self.__vars.CaSS

    @property
    def CaJSR(self) -> npt.NDArray[np.floating]:
        """Recordings of junctional sarcoplasmic reticulum calcium concetrations (mM) for all CRUs.

        Returns:
            npt.NDArray[np.floating]: 2D array of size num_steps x num_CRU
        """
        return self.__vars.CaJSR

    @property
    def LCC(self) -> npt.NDArray[np.integer]:
        """Recordings of L-type calcium channel state, represented by an integer 1-12,
        for all CRU subunits.

        Returns:
            npt.NDArray[npt.integer]: 3D array of size num_steps x num_CRU x 4
        """
        return self.__vars.LCC

    @property
    def LCC_inactivation(self) -> npt.NDArray[np.integer]:
        """Recordings of L-type calcium channel voltage dependent inactivation state
        (0 = closed, 1 = open) for all CRU subunits.

        Returns:
            npt.NDArray[npt.integer]: 3D array of size num_steps x num_CRU x 4
        """
        return self.__vars.LCC_inactivation

    @property
    def RyR(self) -> npt.NDArray[np.integer]:
        """Recordings of all 5 RyR states for all CRU subunits.
        The 6-states of the model are represented by the last dimension of the array,
        which counts the number (<= 5) of RyRs in the coresponding state.

        Returns:
            npt.NDArray[npt.integer]: 4D array of size num_steps x num_CRU x 4 x 6
        """
        return self.__vars.RyR

    @property
    def ClCh(self) -> npt.NDArray[np.integer]:
        """Recordings of calcium activated chloride channel activation states
        (0 = closed, 1 = open) for all CRU subunits.

        Returns:
            npt.NDArray[npt.integer]: 3D array of size num_steps x num_CRU x 4
        """
        return self.__vars.ClCh

    @property
    def INa(self) -> npt.NDArray[np.floating]:
        """Fast inward sodium current (pA/pF).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        if self.__INa is None:
            self.__INa = INa(self.V, self.m, self.h, self.j, self.Nai, self.parameters)
        return self.__INa

    @property
    def INab(self) -> npt.NDArray[np.floating]:
        """Background sodium current (pA/pF).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        if self.__INab is None:
            self.__INab = INab(self.V, self.Nai, self.parameters)
        return self.__INab

    @property
    def INaCa(self) -> npt.NDArray[np.floating]:
        """Sodium-calcium exchanger current (pA/pF).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        if self.__INaCa is None:
            self.__INaCa = INaCa(self.V, self.Nai, self.Cai, self.parameters)
        return self.__INaCa

    @property
    def INaK(self) -> npt.NDArray[np.floating]:
        """Sodium-potassium pump currrent (pA/pF).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        if self.__INaK is None:
            self.__INaK = INaK(self.V, self.Nai, self.parameters)
        return self.__INaK

    @property
    def IKr(self) -> npt.NDArray[np.floating]:
        """Delayed rapidly-activating rectifier current (pA/pA).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        if self.__IKr is None:
            self.__IKr = IKr(self.V, self.XKr[..., 3], self.Ki, self.parameters)
        return self.__IKr

    @property
    def IKs(self) -> npt.NDArray[np.floating]:
        """Delayed slowly-activating rectifier current (pA/pF).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        if self.__IKs is None:
            self.__IKs = IKs(self.V, self.xKs, self.Ki, self.Nai, self.parameters)
        return self.__IKs

    @property
    def IKv14(self) -> npt.NDArray[np.floating]:
        """Kv1.4 component of transient outward current (pA/pF).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        if self.__IKv14 is None:
            self.__IKv14 = IKv14(
                self.V, self.XKv14[..., 4], self.Ki, self.Nai, self.parameters
            )
        return self.__IKv14

    @property
    def IKv43(self) -> npt.NDArray[np.floating]:
        """Kv4.3 component of transient outward current (pA/pF).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        if self.__IKv43 is None:
            self.__IKv43 = IKv43(self.V, self.XKv43[..., 4], self.Ki, self.parameters)
        return self.__IKv43

    @property
    def Ito1(self) -> npt.NDArray[np.floating]:
        """Transient outward current Ito1 = IKv1.4 + IKv4.3 (pA/pF).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        return self.IKv14 + self.IKv43

    @property
    def Ito2(self) -> npt.NDArray[np.floating]:
        """Calcium mediated chloride transient outward current (pA/pF).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        if self.__Ito2 is None:
            self.__Ito2 = Ito2(self.V, self.ClCh, self.parameters)
        return self.__Ito2

    @property
    def IK1(self) -> npt.NDArray[np.floating]:
        """Time independent potassium current (pA/pF).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        if self.__IK1 is None:
            self.__IK1 = IK1(self.V, self.Ki, self.parameters)
        return self.__IK1

    @property
    def IKp(self) -> npt.NDArray[np.floating]:
        """Plateau potassium current (pA/pF).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        if self.__IKp is None:
            self.__IKp = IKp(self.V, self.Ki, self.parameters)
        return self.__IKp

    @property
    def ICaL(self) -> npt.NDArray[np.floating]:
        """L-type calcium channel current (pA/pF).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        if self.__ICaL is None:
            self.__ICaL = ICaL(
                self.V, self.LCC, self.LCC_inactivation, self.CaSS, self.parameters
            )
        return self.__ICaL

    @property
    def ICab(self) -> npt.NDArray[np.floating]:
        """Background calcium current (pA/pF).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        if self.__ICab is None:
            self.__ICab = ICab(self.V, self.Cai, self.parameters)
        return self.__ICab

    @property
    def IpCa(self) -> npt.NDArray[np.floating]:
        """Sarcolemmal calcium pump current (pA/pF).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        if self.__IpCa is None:
            self.__IpCa = IpCa(self.Cai, self.parameters)
        return self.__IpCa

    @property
    def Jup(self) -> npt.NDArray[np.floating]:
        """SERCA2a pump flux (mM/ms).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        if self.__Jup is None:
            self.__Jup = Jup(self.Cai, self.CaNSR, self.parameters)
        return self.__Jup
    
    @property
    def Jtr(self) -> npt.NDArray[np.floating]:
        """SERCA2a pump flux (mM/ms).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        if self.__Jtr is None:
            pass # TODO: implement
            #self.__Jup = Jup(self.Cai, self.CaNSR, self.parameters)
        return self.__Jtr
    
    @property
    def Jxfer(self) -> npt.NDArray[np.floating]:
        """SERCA2a pump flux (mM/ms).

        Returns:
            npt.NDArray[np.floating]: 1D array of size num_steps
        """
        if self.__Jxfer is None:
            pass # TODO: implement
            #self.__Jup = Jup(self.Cai, self.CaNSR, self.parameters)
        return self.__Jxfer



class GWModel:
    """Simulates the Greenstein and Winslow model given a GWParameters object and an
    external stimulus function
    """

    @classmethod
    def PRNG_options(cls) -> List[str]:
        """Get the available arguments for PRNG in the simulate method

        Returns:
            List[str]: Implemented PRNGs for simulating the model
        """
        return __IMPLEMENTED_PRNGS

    def __init__(
        self,
        parameters: None | GWParameters = None,
        stimulus_fn: None | Callable[[float], float] = None,
    ):
        """Create the GWModel object

        Args:
            parameters (None | GWParameters, optional): Model parameters. Defaults to the default parameters given in Greenstein and Winslow, 2002.
            stimulus_fn (None | Callable[[float], float], optional): Time dependent external stimulus in pA/pF. Defaults to the zero function.
        """
        self.parameters: GWParameters = (
            parameters if parameters is not None else GWParameters()
        )
        self.__stim: Callable[[float], float] = (
            (lambda t: 0) if stimulus_fn is None else stimulus_fn
        )

    def simulate(
        self,
        step_size: float,
        num_steps: int,
        record_every: int = 1,
        PRNG: str = "xoshiro256++",
    ) -> GWSolution:
        """Simulate the model with a step size of step_size over num_steps steps.
           Note that the current implementation does not allow for the intial condition to be changed

        Args:
            step_size (float): Size of the integrator time step (ms)
            num_steps (int): Number of steps that the integrator should take.
            record_every (int, optional): Number of steps between recording snapshots of the state. Defaults to 1.
            PRNG (str, optional): PRNG to use within the algorithm. Default is 'xoshiro256++'.

        Returns:
            GWSolution: Snapshots of state through simulation.
        """
        assert_positive(step_size, "step_size")
        assert_positive(num_steps, "num_steps")
        assert_positive(record_every, "record_every")
        try:
            cpp_sol = gw.run(
                self.parameters.cpp_struct,
                self.parameters.NCaRU_sim,
                step_size,
                num_steps,
                self.__stim,
                record_every,
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

        return GWSolution(cpp_sol, self.parameters)

    def stimulus_fn(self, t: float) -> float:
        """Evaluate the stimulus function Istim

        Args:
            t (float): time (ms)

        Returns:
            float: Stimulus at time t
        """
        return self.__stim(t)

    def set_stimulus_fn(self, stimulus_fn: Callable[[float], float]) -> None:
        """Set the stimulus function

        Args:
            stimulus_fn (Callable[[float], float]): stimulus function
        """
        self.__stim = stimulus_fn
