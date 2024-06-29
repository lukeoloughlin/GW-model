from typing import Callable

import numpy as np
import numpy.typing as npt

import build.GreensteinWinslow as gw

F = 96.5;
R = 8.314;

class GWParameters:

    def __init__(self, nCRU: int, parameters: gw.Parameters = gw.Parameters()):
        self.__parameters = parameters
        self.__F_RT = F / (R * parameters.T)
        self.__nCRU = nCRU

    @property
    def parameters(self):
        return self.__parameters

    def INa(self, V: npt.ArrayLike, m: npt.ArrayLike, h: npt.ArrayLike, j: npt.ArrayLike, Nai: npt.ArrayLike) -> npt.ArrayLike:
        ENa = np.log(self.parameters.Nao / Nai) / self.__F_RT
        return self.parameters.GNa * (m ** 3) * h * j * (V - ENa)
    
    def INab(self, V: npt.ArrayLike, Nai: npt.ArrayLike) -> npt.ArrayLike: 
        ENa = np.log(self.parameters.Nao / Nai) / self.__F_RT
        return self.parameters.GNab * (V - ENa)
    
    def IKr(self, V: npt.ArrayLike, XKr: npt.ArrayLike, Ki: npt.ArrayLike) -> npt.ArrayLike:
        RKr = 1 / (1 + 1.4945 * np.exp(0.0446*V))
        EK = np.log(self.parameters.Ko / Ki) / self.__F_RT
        return 0.5 * self.parameters.GKr * np.sqrt(self.parameters.Ko) * RKr * XKr * (V - EK)
    
    def IKs(self, V: npt.ArrayLike, XKs: npt.ArrayLike, Ki: npt.ArrayLike, Nai: npt.ArrayLike) -> npt.ArrayLike:
        EKs = np.log((self.parameters.Ko + 0.01833 * self.parameters.Nao) / (Ki + 0.01833*Nai)) / self.__F_RT;
        return self.parameters.GKs * (XKs ** 2) * (V - EKs)
    
    def IKv43(self, V: npt.ArrayLike, XKv43: npt.ArrayLike, Ki: npt.ArrayLike) -> npt.ArrayLike:
        EK = np.log(self.parameters.Ko / Ki) / self.__F_RT
        return self.parameters.GKv43 * XKv43 * (V - EK)
    
    def IKv14(self, V: npt.ArrayLike, XKv14: npt.ArrayLike, Ki: npt.ArrayLike, Nai: npt.ArrayLike) -> npt.ArrayLike:
        VF_RT = V * F / (R * self.parameters.T)
        expVF_RT = np.exp(VF_RT)
        multiplier = self.parameters.PKv14_Csc * F * VF_RT * XKv14 / (1 - expVF_RT)
        return multiplier * ((Ki - self.parameters.Ko * expVF_RT) + 0.02 * (Nai - self.parameters.Nao)) * 1e9
    
    def IK1(self, V: npt.ArrayLike, Ki: npt.ArrayLike) -> npt.ArrayLike:
        EK = np.log(self.parameters.Ko / Ki) / self.__F_RT
        K1inf = 1 / (2 + np.exp(1.5 * (V - EK) * self.__F_RT ))
        return self.parameters.GK1 * K1inf * (self.parameters.Ko / (self.parameters.Ko + self.parameters.KmK1)) * (V - EK)
    
    def IKp(self, V: npt.ArrayLike, Ki: npt.ArrayLike) -> npt.ArrayLike:
        EK = np.log(self.parameters.Ko / Ki) / self.__F_RT
        Kp = 1 / (1 + np.exp((7.488 - V) / 5.98))
        return self.parameters.GKp * Kp * (V - EK)
    
    def INaK(self, V: npt.ArrayLike, Nai: npt.ArrayLike) -> npt.ArrayLike:
        sigma = (np.exp(self.parameters.Nao / 67.3) - 1) / 7
        fNaK = 1 / (1 + 0.1245 * np.exp(-0.1 * V * self.__F_RT) + 0.0365 * sigma * np.exp(-V * self.__F_RT))
        multiplier = self.parameters.INaKmax * self.parameters.Ko / (self.parameters.Ko + self.parameters.KmKo)
        return multiplier * fNaK / (1 + (self.parameters.KmNai / Nai) ** 1.5)
    
    def INaCa(self, V: npt.ArrayLike, Nai: npt.ArrayLike, Cai: npt.ArrayLike) -> npt.ArrayLike:
        exp_term1 = np.exp(self.parameters.eta * V * self.__F_RT)
        exp_term2 = np.exp((self.parameters.eta - 1) * V * self.__F_RT)
        multiplier = 5000 * self.parameters.kNaCa / ((self.parameters.KmNa ** 3 + self.parameters.Nao ** 3) * (self.parameters.KmCa + self.parameters.Cao))
        return multiplier * (exp_term1 * (Nai ** 3) * self.parameters.Cao - exp_term2 * (self.parameters.Nao ** 3) * Cai) / (1 + self.parameters.ksat * exp_term2)
    
    
    def IpCa(self, Cai: npt.ArrayLike) -> npt.ArrayLike: 
        return self.parameters.IpCamax * (Cai / (self.parameters.KmpCa + Cai))
    
    def ICab(self, V: npt.ArrayLike, Cai: npt.ArrayLike) -> npt.ArrayLike: 
        ECa = 0.5 * np.log(self.parameters.Cao / Cai) / self.__F_RT
        return self.parameters.GCab * (V - ECa)

    def ICaL(self, V: npt.ArrayLike, LCC: npt.ArrayLike, LCC_activation: npt.ArrayLike, CaSS: npt.ArrayLike) -> npt.ArrayLike:
        open_LCC = np.logical_and(np.logical_or(LCC == 6, LCC == 12), LCC_activation == 1)
        exp2VF_RT = np.exp(2* V * self.__F_RT)
        multiplier = (self.parameters.NCaRU / self.__nCRU) * self.parameters.PCaL * 4 * V * F * self.__F_RT / (exp2VF_RT - 1)
        ICaL_individual_channels = multiplier.reshape(-1,1,1) * open_LCC * (CaSS * exp2VF_RT.reshape(-1,1,1) - 0.341 * self.parameters.Cao)
        return ICaL_individual_channels.sum(axis=(1,2)) * 1e6
    

    def Ito2(self, V: npt.ArrayLike, ClCh: npt.ArrayLike) -> npt.ArrayLike:
        VF_RT = (V * self.__F_RT).reshape(-1,1)
        expmVF_RT = np.exp(-VF_RT)
        const_term = 1e9 * self.parameters.Pto2 * F * (self.parameters.NCaRU / self.__nCRU) / self.parameters.CSA;
        return (ClCh * const_term * VF_RT * (self.parameters.Clcyto * expmVF_RT - self.parameters.Clo) / (expmVF_RT - 1)).sum(axis=-1)
    
    def dCaLTRPN(self, CaLTRPN: npt.ArrayLike, Cai: npt.ArrayLike) -> npt.ArrayLike:
        return self.parameters.kLTRPNp * Cai * (self.parameters.LTRPNtot - CaLTRPN) - self.parameters.kLTRPNm * CaLTRPN
    
    def dCaHTRPN(self, CaHTRPN: npt.ArrayLike, Cai: npt.ArrayLike) -> npt.ArrayLike:
        return self.parameters.kHTRPNp * Cai * (self.parameters.HTRPNtot - CaHTRPN) - self.parameters.kHTRPNm * CaHTRPN

    
    def Jup(self, Cai: npt.ArrayLike, CaNSR: npt.ArrayLike) -> npt.ArrayLike:
        f = (Cai / self.parameters.Kmf) ** self.parameters.Hf
        r = (CaNSR / self.parameters.Kmr) ** self.parameters.Hr
        return (self.parameters.Vmaxf * f - self.parameters.Vmaxr * r) / (1 + f + r)


        

class GWSolution:

    def __init__(self, gw_cpp_output: gw.GWVariables, gw_parameters: GWParameters):
        self.__vars = gw_cpp_output
        self.__params = gw_parameters
        self.__INa = None
        self.__INab = None;
        self.__INaCa = None;
        self.__INaK = None;
        self.__IKr = None;
        self.__IKs = None;
        self.__IKv14 = None;
        self.__IKv43 = None;
        self.__Ito2 = None;
        self.__IK1 = None;
        self.__IKp = None;
        self.__ICaL = None;
        self.__ICab = None;
        self.__IpCa = None;
        self.__Jup = None;

    @property
    def t(self) -> np.ndarray:
        return self.__vars.t

    @property
    def V(self) -> np.ndarray:
        return self.__vars.V
    
    @property
    def m(self) -> np.ndarray:
        return self.__vars.m
    
    @property
    def h(self) -> np.ndarray:
        return self.__vars.h
    
    @property
    def j(self) -> np.ndarray:
        return self.__vars.j
    
    @property
    def Nai(self) -> np.ndarray:
        return self.__vars.Nai
    
    @property
    def Ki(self) -> np.ndarray:
        return self.__vars.Ki
    
    @property
    def Cai(self) -> np.ndarray:
        return self.__vars.Cai
    
    @property
    def CaNSR(self) -> np.ndarray:
        return self.__vars.CaNSR
    
    @property
    def CaLTRPN(self) -> np.ndarray:
        return self.__vars.CaLTRPN
    
    @property
    def CaHTRPN(self) -> np.ndarray:
        return self.__vars.CaHTRPN
    
    @property
    def xKs(self) -> np.ndarray:
        return self.__vars.xKs
    
    @property
    def XKr(self) -> np.ndarray:
        return self.__vars.XKr
    
    @property
    def XKv14(self) -> np.ndarray:
        return self.__vars.XKv14
    
    @property
    def XKv43(self) -> np.ndarray:
        return self.__vars.XKv43
    
    @property
    def CaSS(self) -> np.ndarray:
        return self.__vars.CaSS
    
    @property
    def CaJSR(self) -> np.ndarray:
        return self.__vars.CaJSR
    
    @property
    def LCC(self) -> np.ndarray:
        return self.__vars.LCC
    
    @property
    def LCC_activation(self) -> np.ndarray:
        return self.__vars.LCC_activation
    
    @property
    def RyR(self) -> np.ndarray:
        return self.__vars.RyR
    
    @property
    def ClCh(self) -> np.ndarray:
        return self.__vars.ClCh
        
    @property
    def INa(self) -> np.ndarray:
        if self.__INa is None:
            self.__INa = self.__params.INa(self.V, self.m, self.h, self.j, self.Nai)
        return self.__INa
    
    @property
    def INab(self) -> np.ndarray:
        if self.__INab is None:
            self.__INab = self.__params.INab(self.V, self.Nai)
        return self.__INab
        
    
    @property
    def INaCa(self) -> np.ndarray:
        if self.__INaCa is None:
            self.__INaCa = self.__params.INaCa(self.V, self.Nai, self.Cai)
        return self.__INaCa
    
    @property
    def INaK(self) -> np.ndarray:
        if self.__INaK is None:
            self.__INaK = self.__params.INaK(self.V, self.Nai)
        return self.__INaK
    
    @property
    def IKr(self) -> np.ndarray:
        if self.__IKr is None:
            self.__IKr = self.__params.IKr(self.V, self.XKr[..., 3], self.Ki)
        return self.__IKr
    
    @property
    def IKs(self) -> np.ndarray:
        if self.__IKs is None:
            self.__IKs = self.__params.IKs(self.V, self.xKs, self.Ki, self.Nai)
        return self.__IKs
    
    @property
    def IKv14(self) -> np.ndarray:
        if self.__IKv14 is None:
            self.__IKv14 = self.__params.IKv14(self.V, self.XKv14[..., 4], self.Ki, self.Nai)
        return self.__IKv14
    
    @property
    def IKv43(self) -> np.ndarray:
        if self.__IKv43 is None:
            self.__IKv43 = self.__params.IKv43(self.V, self.XKv43[..., 4], self.Ki)
        return self.__IKv43
    
    @property
    def Ito1(self) -> np.ndarray:
        return self.IKv14 + self.IKv43
    
    @property
    def Ito2(self) -> np.ndarray:
        if self.__Ito2 is None:
            self.__Ito2 = self.__params.Ito2(self.V, self.ClCh)
        return self.__Ito2
    
    @property
    def IK1(self) -> np.ndarray:
        if self.__IK1 is None:
            self.__IK1 = self.__params.IK1(self.V, self.Ki)
        return self.__IK1
        
    
    @property
    def IKp(self) -> np.ndarray:
        if self.__IKp is None:
            self.__IKp = self.__params.IKp(self.V, self.Ki)
        return self.__IKp
    
    @property
    def ICaL(self) -> np.ndarray:
        if self.__ICaL is None:
            self.__ICaL = self.__params.ICaL(self.V, self.LCC, self.LCC_activation, self.CaSS)
        return self.__ICaL
    
    @property
    def ICab(self) -> np.ndarray:        
        if self.__ICab is None:
            self.__ICab = self.__params.ICab(self.V, self.Cai)
        return self.__ICab
        
    
    @property
    def IpCa(self) -> np.ndarray:
        if self.__IpCa is None:
            self.__IpCa = self.__params.IpCa(self.Cai)
        return self.__IpCa
    
    @property
    def Jup(self) -> np.ndarray:
        if self.__Jup is None:
            self.__Jup = self.__params.Jup(self.Cai, self.CaNSR)
        return self.__Jup



class GWModel:

    def __init__(self, num_CRU: int, parameters: None | GWParameters = None, stimulus_fn: None | Callable = None):
        self.__nCRU = num_CRU
        self.parameters = parameters if parameters is not None else GWParameters(num_CRU)
        self.__stim = (lambda t: 0) if stimulus_fn is None else stimulus_fn

    # Main method of this class
    def simulate(self, step_size: float, num_steps: int, record_every: int = 1) -> GWSolution :
        cpp_sol = gw.run(self.parameters.parameters, self.__nCRU, step_size, num_steps, self.__stim, record_every)
        return GWSolution(cpp_sol, self.parameters)


    def stimulus_fn(self, t: float) -> float:
        return self.__stim(t)
    
    def set_stimulus_fn(self, stimulus_fn : Callable) -> None:
        self.__stim = stimulus_fn

    def set_num_CRUs(self, num_CRUs: int) -> None:
        self.__nCRU = num_CRUs

    @property
    def num_CRUs(self) -> int:
        return self.__nCRU 

