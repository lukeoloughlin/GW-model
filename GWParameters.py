from typing import Any, List
from functools import reduce

import build.GreensteinWinslow as gw  # type: ignore


# NOTE: I couldn't think of a good way to automatically make a python interface to the C++ struct such that domain constraints could be checked and documentation could be added.
#       So, I used the below helper methods to generate the getters and setters as strings, printed them out, and altered the relevant fields manually (changed NCaRU to int and made some rate
#        parameters nonnegative)

_GW_NAMES = [
    "T",
    "CSA",
    "Vcyto",
    "VNSR",
    "VJSR",
    "VSS",
    "NCaRU",
    "Ko",
    "Nao",
    "Cao",
    "Clo",
    "Clcyto",
    "f",
    "g",
    "f1",
    "g1",
    "a",
    "b",
    "gamma0",
    "omega",
    "PCaL",
    "kfClCh",
    "kbClCh",
    "Pto2",
    "k12",
    "k21",
    "k23",
    "k32",
    "k34",
    "k43",
    "k45",
    "k54",
    "k56",
    "k65",
    "k25",
    "k52",
    "rRyR",
    "rxfer",
    "rtr",
    "riss",
    "BSRT",
    "KBSR",
    "BSLT",
    "KBSL",
    "CSQNT",
    "KCSQN",
    "CMDNT",
    "KCMDN",
    "GNa",
    "GKr",
    "Kf",
    "Kb",
    "GKs",
    "GKv43",
    "alphaa0Kv43",
    "aaKv43",
    "betaa0Kv43",
    "baKv43",
    "alphai0Kv43",
    "aiKv43",
    "betai0Kv43",
    "biKv43",
    "f1Kv43",
    "f2Kv43",
    "f3Kv43",
    "f4Kv43",
    "b1Kv43",
    "b2Kv43",
    "b3Kv43",
    "b4Kv43",
    "PKv14",
    "alphaa0Kv14",
    "aaKv14",
    "betaa0Kv14",
    "baKv14",
    "alphai0Kv14",
    "aiKv14",
    "betai0Kv14",
    "biKv14",
    "f1Kv14",
    "f2Kv14",
    "f3Kv14",
    "f4Kv14",
    "b1Kv14",
    "b2Kv14",
    "b3Kv14",
    "b4Kv14",
    "Csc",
    "GK1",
    "KmK1",
    "GKp",
    "kNaCa",
    "KmNa",
    "KmCa",
    "ksat",
    "eta",
    "INaKmax",
    "KmNai",
    "KmKo",
    "IpCamax",
    "KmpCa",
    "GCab",
    "GNab",
    "kHTRPNp",
    "kHTRPNm",
    "kLTRPNp",
    "kLTRPNm",
    "HTRPNtot",
    "LTRPNtot",
    "Vmaxf",
    "Vmaxr",
    "Kmf",
    "Kmr",
    "Hf",
    "Hr",
]


# Run this with _GW_NAMES and print result or write to file to get boiler plate for the below class
def __create_boilerplate(names: List[str]) -> str:
    out_str = ""
    for name in names:
        out_str += f"""
    @property
    def {name}(self) -> float:
        return self.__cpp_struct.{name}

    @{name}.setter
    def {name}(self, value: float) -> None:
        assert_positive(value, '{name}')
        self.__cpp_struct.{name} = value
    """
    return out_str

def add_docstring(docstring: str) -> Any:
    """Decorator to add a docstring to a class

    Args:
        docstring (str): Docstring to apply to class
    """
    def apply_docstring(cls: Any) -> Any:
        cls.__doc__ = docstring
        return cls
    return apply_docstring


def assert_positive(x: Any, name: str) -> None:
    """Assert that x > 0

    Args:
        x (Any): Value to check
        name (str): Name of value for throwing error

    Raises:
        ValueError: If x <= 0
    """
    if x <= 0:
        raise ValueError(f"{name} must be positive")


def assert_nonnegative(x: Any, name: str) -> None:
    """Assert that x >= 0

    Args:
        x (Any): Value to check
        name (str): Name of value for throwing error

    Raises:
        ValueError: If x < 0
    """
    if x < 0:
        raise ValueError(f"{name} must be nonnegative")


def assert_type(x: Any, tp: Any, name: str) -> None:
    """Assert that x is of type tp

    Args:
        x (Any): Value to check
        tp (Any): Type to check
        name (str): Name of value for throwing error

    Raises:
        TypeError: If not isinstance(x, tp)
    """
    if not isinstance(x, tp):
        raise TypeError(f"{name} must be of type {tp}. Got type {type(x)}")

__gw_docstring =  "A container for the parameters of the Greenstein and Winslow model. The list of parameters are:\n"
__gw_docstring += reduce(lambda ds, par: ds.append("\n" + par), _GW_NAMES, initial=__gw_docstring)



# TODO: add descriptions to the list of parameters above, then use a decorator to make the docstring automatically.
@add_docstring(__gw_docstring)
class GWParameters:

    def __init__(self, cpp_struct: None | gw.GWParameters = None, **kwargs):
        self.__cpp_struct: gw.GWParameters = (
            cpp_struct if cpp_struct is not None else gw.GWParameters()
        )
        if "NCaRU_sim" in kwargs:
            assert_type(kwargs["NCaRU_sim"], int, "NCaRU_sim")
            assert_positive(kwargs["NCaRU_sim"], "NCaRU_sim")
            self.NCaRU_sim: int = kwargs["NCaRU_sim"]
        else:
            self.NCaRU_sim: int = 1250

        for name, value in kwargs.items():
            if name in dir(self.__cpp_struct):
                setattr(self, name, value)
    
    @property
    def cpp_struct(self) -> gw.GWParameters:
        return self.__cpp_struct

    @property
    def T(self) -> float:
        return self.__cpp_struct.T

    @T.setter
    def T(self, value: float) -> None:
        assert_positive(value, "T")
        self.__cpp_struct.T = value

    @property
    def CSA(self) -> float:
        return self.__cpp_struct.CSA

    @CSA.setter
    def CSA(self, value: float) -> None:
        assert_positive(value, "CSA")
        self.__cpp_struct.CSA = value

    @property
    def Vcyto(self) -> float:
        return self.__cpp_struct.Vcyto

    @Vcyto.setter
    def Vcyto(self, value: float) -> None:
        assert_positive(value, "Vcyto")
        self.__cpp_struct.Vcyto = value

    @property
    def VNSR(self) -> float:
        return self.__cpp_struct.VNSR

    @VNSR.setter
    def VNSR(self, value: float) -> None:
        assert_positive(value, "VNSR")
        self.__cpp_struct.VNSR = value

    @property
    def VJSR(self) -> float:
        return self.__cpp_struct.VJSR

    @VJSR.setter
    def VJSR(self, value: float) -> None:
        assert_positive(value, "VJSR")
        self.__cpp_struct.VJSR = value

    @property
    def VSS(self) -> float:
        return self.__cpp_struct.VSS

    @VSS.setter
    def VSS(self, value: float) -> None:
        assert_positive(value, "VSS")
        self.__cpp_struct.VSS = value

    @property
    def NCaRU(self) -> int:
        return self.__cpp_struct.NCaRU

    @NCaRU.setter
    def NCaRU(self, value: int) -> None:
        assert_type(value, int, "NCaRU")
        assert_positive(value, "NCaRU")
        self.__cpp_struct.NCaRU = value

    @property
    def Ko(self) -> float:
        return self.__cpp_struct.Ko

    @Ko.setter
    def Ko(self, value: float) -> None:
        assert_positive(value, "Ko")
        self.__cpp_struct.Ko = value

    @property
    def Nao(self) -> float:
        return self.__cpp_struct.Nao

    @Nao.setter
    def Nao(self, value: float) -> None:
        assert_positive(value, "Nao")
        self.__cpp_struct.Nao = value

    @property
    def Cao(self) -> float:
        return self.__cpp_struct.Cao

    @Cao.setter
    def Cao(self, value: float) -> None:
        assert_positive(value, "Cao")
        self.__cpp_struct.Cao = value

    @property
    def Clo(self) -> float:
        return self.__cpp_struct.Clo

    @Clo.setter
    def Clo(self, value: float) -> None:
        assert_positive(value, "Clo")
        self.__cpp_struct.Clo = value

    @property
    def Clcyto(self) -> float:
        return self.__cpp_struct.Clcyto

    @Clcyto.setter
    def Clcyto(self, value: float) -> None:
        assert_positive(value, "Clcyto")
        self.__cpp_struct.Clcyto = value

    @property
    def f(self) -> float:
        return self.__cpp_struct.f

    @f.setter
    def f(self, value: float) -> None:
        assert_nonnegative(value, "f")
        self.__cpp_struct.f = value

    @property
    def g(self) -> float:
        return self.__cpp_struct.g

    @g.setter
    def g(self, value: float) -> None:
        assert_nonnegative(value, "g")
        self.__cpp_struct.g = value

    @property
    def f1(self) -> float:
        return self.__cpp_struct.f1

    @f1.setter
    def f1(self, value: float) -> None:
        assert_nonnegative(value, "f1")
        self.__cpp_struct.f1 = value

    @property
    def g1(self) -> float:
        return self.__cpp_struct.g1

    @g1.setter
    def g1(self, value: float) -> None:
        assert_nonnegative(value, "g1")
        self.__cpp_struct.g1 = value

    @property
    def a(self) -> float:
        return self.__cpp_struct.a

    @a.setter
    def a(self, value: float) -> None:
        assert_nonnegative(value, "a")
        self.__cpp_struct.a = value

    @property
    def b(self) -> float:
        return self.__cpp_struct.b

    @b.setter
    def b(self, value: float) -> None:
        assert_positive(value, "b")
        self.__cpp_struct.b = value

    @property
    def gamma0(self) -> float:
        return self.__cpp_struct.gamma0

    @gamma0.setter
    def gamma0(self, value: float) -> None:
        assert_nonnegative(value, "gamma0")
        self.__cpp_struct.gamma0 = value

    @property
    def omega(self) -> float:
        return self.__cpp_struct.omega

    @omega.setter
    def omega(self, value: float) -> None:
        assert_nonnegative(value, "omega")
        self.__cpp_struct.omega = value

    @property
    def PCaL(self) -> float:
        return self.__cpp_struct.PCaL

    @PCaL.setter
    def PCaL(self, value: float) -> None:
        assert_positive(value, "PCaL")
        self.__cpp_struct.PCaL = value

    @property
    def kfClCh(self) -> float:
        return self.__cpp_struct.kfClCh

    @kfClCh.setter
    def kfClCh(self, value: float) -> None:
        assert_nonnegative(value, "kfClCh")
        self.__cpp_struct.kfClCh = value

    @property
    def kbClCh(self) -> float:
        return self.__cpp_struct.kbClC

    @kbClCh.setter
    def kbClCh(self, value: float) -> None:
        assert_nonnegative(value, "kbClCh")
        self.__cpp_struct.kbClC = value

    @property
    def Pto2(self) -> float:
        return self.__cpp_struct.Pto

    @Pto2.setter
    def Pto2(self, value: float) -> None:
        assert_positive(value, "Pto")
        self.__cpp_struct.Pto = value

    @property
    def k12(self) -> float:
        return self.__cpp_struct.k12

    @k12.setter
    def k12(self, value: float) -> None:
        assert_nonnegative(value, "k12")
        self.__cpp_struct.k12 = value

    @property
    def k21(self) -> float:
        return self.__cpp_struct.k21

    @k21.setter
    def k21(self, value: float) -> None:
        assert_nonnegative(value, "k21")
        self.__cpp_struct.k21 = value

    @property
    def k23(self) -> float:
        return self.__cpp_struct.k23

    @k23.setter
    def k23(self, value: float) -> None:
        assert_nonnegative(value, "k23")
        self.__cpp_struct.k23 = value

    @property
    def k32(self) -> float:
        return self.__cpp_struct.k32

    @k32.setter
    def k32(self, value: float) -> None:
        assert_nonnegative(value, "k32")
        self.__cpp_struct.k32 = value

    @property
    def k34(self) -> float:
        return self.__cpp_struct.k34

    @k34.setter
    def k34(self, value: float) -> None:
        assert_nonnegative(value, "k34")
        self.__cpp_struct.k34 = value

    @property
    def k43(self) -> float:
        return self.__cpp_struct.k43

    @k43.setter
    def k43(self, value: float) -> None:
        assert_nonnegative(value, "k43")
        self.__cpp_struct.k43 = value

    @property
    def k45(self) -> float:
        return self.__cpp_struct.k45

    @k45.setter
    def k45(self, value: float) -> None:
        assert_nonnegative(value, "k45")
        self.__cpp_struct.k45 = value

    @property
    def k54(self) -> float:
        return self.__cpp_struct.k54

    @k54.setter
    def k54(self, value: float) -> None:
        assert_nonnegative(value, "k54")
        self.__cpp_struct.k54 = value

    @property
    def k56(self) -> float:
        return self.__cpp_struct.k56

    @k56.setter
    def k56(self, value: float) -> None:
        assert_nonnegative(value, "k56")
        self.__cpp_struct.k56 = value

    @property
    def k65(self) -> float:
        return self.__cpp_struct.k65

    @k65.setter
    def k65(self, value: float) -> None:
        assert_nonnegative(value, "k65")
        self.__cpp_struct.k65 = value

    @property
    def k25(self) -> float:
        return self.__cpp_struct.k25

    @k25.setter
    def k25(self, value: float) -> None:
        assert_nonnegative(value, "k25")
        self.__cpp_struct.k25 = value

    @property
    def k52(self) -> float:
        return self.__cpp_struct.k52

    @k52.setter
    def k52(self, value: float) -> None:
        assert_nonnegative(value, "k52")
        self.__cpp_struct.k52 = value

    @property
    def rRyR(self) -> float:
        return self.__cpp_struct.rRyR

    @rRyR.setter
    def rRyR(self, value: float) -> None:
        assert_positive(value, "rRyR")
        self.__cpp_struct.rRyR = value

    @property
    def rxfer(self) -> float:
        return self.__cpp_struct.rxfer

    @rxfer.setter
    def rxfer(self, value: float) -> None:
        assert_positive(value, "rxfer")
        self.__cpp_struct.rxfer = value

    @property
    def rtr(self) -> float:
        return self.__cpp_struct.rtr

    @rtr.setter
    def rtr(self, value: float) -> None:
        assert_positive(value, "rtr")
        self.__cpp_struct.rtr = value

    @property
    def riss(self) -> float:
        return self.__cpp_struct.riss

    @riss.setter
    def riss(self, value: float) -> None:
        assert_positive(value, "riss")
        self.__cpp_struct.riss = value

    @property
    def BSRT(self) -> float:
        return self.__cpp_struct.BSRT

    @BSRT.setter
    def BSRT(self, value: float) -> None:
        assert_positive(value, "BSRT")
        self.__cpp_struct.BSRT = value

    @property
    def KBSR(self) -> float:
        return self.__cpp_struct.KBSR

    @KBSR.setter
    def KBSR(self, value: float) -> None:
        assert_positive(value, "KBSR")
        self.__cpp_struct.KBSR = value

    @property
    def BSLT(self) -> float:
        return self.__cpp_struct.BSLT

    @BSLT.setter
    def BSLT(self, value: float) -> None:
        assert_positive(value, "BSLT")
        self.__cpp_struct.BSLT = value

    @property
    def KBSL(self) -> float:
        return self.__cpp_struct.KBSL

    @KBSL.setter
    def KBSL(self, value: float) -> None:
        assert_positive(value, "KBSL")
        self.__cpp_struct.KBSL = value

    @property
    def CSQNT(self) -> float:
        return self.__cpp_struct.CSQNT

    @CSQNT.setter
    def CSQNT(self, value: float) -> None:
        assert_positive(value, "CSQNT")
        self.__cpp_struct.CSQNT = value

    @property
    def KCSQN(self) -> float:
        return self.__cpp_struct.KCSQN

    @KCSQN.setter
    def KCSQN(self, value: float) -> None:
        assert_positive(value, "KCSQN")
        self.__cpp_struct.KCSQN = value

    @property
    def CMDNT(self) -> float:
        return self.__cpp_struct.CMDNT

    @CMDNT.setter
    def CMDNT(self, value: float) -> None:
        assert_positive(value, "CMDNT")
        self.__cpp_struct.CMDNT = value

    @property
    def KCMDN(self) -> float:
        return self.__cpp_struct.KCMDN

    @KCMDN.setter
    def KCMDN(self, value: float) -> None:
        assert_positive(value, "KCMDN")
        self.__cpp_struct.KCMDN = value

    @property
    def GNa(self) -> float:
        return self.__cpp_struct.GNa

    @GNa.setter
    def GNa(self, value: float) -> None:
        assert_positive(value, "GNa")
        self.__cpp_struct.GNa = value

    @property
    def GKr(self) -> float:
        return self.__cpp_struct.GKr

    @GKr.setter
    def GKr(self, value: float) -> None:
        assert_positive(value, "GKr")
        self.__cpp_struct.GKr = value

    @property
    def Kf(self) -> float:
        return self.__cpp_struct.Kf

    @Kf.setter
    def Kf(self, value: float) -> None:
        assert_positive(value, "Kf")
        self.__cpp_struct.Kf = value

    @property
    def Kb(self) -> float:
        return self.__cpp_struct.Kb

    @Kb.setter
    def Kb(self, value: float) -> None:
        assert_positive(value, "Kb")
        self.__cpp_struct.Kb = value

    @property
    def GKs(self) -> float:
        return self.__cpp_struct.GKs

    @GKs.setter
    def GKs(self, value: float) -> None:
        assert_positive(value, "GKs")
        self.__cpp_struct.GKs = value

    @property
    def GKv43(self) -> float:
        return self.__cpp_struct.GKv43

    @GKv43.setter
    def GKv43(self, value: float) -> None:
        assert_positive(value, "GKv43")
        self.__cpp_struct.GKv43 = value

    @property
    def alphaa0Kv43(self) -> float:
        return self.__cpp_struct.alphaa0Kv43

    @alphaa0Kv43.setter
    def alphaa0Kv43(self, value: float) -> None:
        assert_nonnegative(value, "alphaa0Kv43")
        self.__cpp_struct.alphaa0Kv43 = value

    @property
    def aaKv43(self) -> float:
        return self.__cpp_struct.aaKv43

    @aaKv43.setter
    def aaKv43(self, value: float) -> None:
        assert_nonnegative(value, "aaKv43")
        self.__cpp_struct.aaKv43 = value

    @property
    def betaa0Kv43(self) -> float:
        return self.__cpp_struct.betaa0Kv43

    @betaa0Kv43.setter
    def betaa0Kv43(self, value: float) -> None:
        assert_nonnegative(value, "betaa0Kv43")
        self.__cpp_struct.betaa0Kv43 = value

    @property
    def baKv43(self) -> float:
        return self.__cpp_struct.baKv43

    @baKv43.setter
    def baKv43(self, value: float) -> None:
        assert_nonnegative(value, "baKv43")
        self.__cpp_struct.baKv43 = value

    @property
    def alphai0Kv43(self) -> float:
        return self.__cpp_struct.alphai0Kv43

    @alphai0Kv43.setter
    def alphai0Kv43(self, value: float) -> None:
        assert_nonnegative(value, "alphai0Kv43")
        self.__cpp_struct.alphai0Kv43 = value

    @property
    def aiKv43(self) -> float:
        return self.__cpp_struct.aiKv43

    @aiKv43.setter
    def aiKv43(self, value: float) -> None:
        assert_nonnegative(value, "aiKv43")
        self.__cpp_struct.aiKv43 = value

    @property
    def betai0Kv43(self) -> float:
        return self.__cpp_struct.betai0Kv43

    @betai0Kv43.setter
    def betai0Kv43(self, value: float) -> None:
        assert_nonnegative(value, "betai0Kv43")
        self.__cpp_struct.betai0Kv43 = value

    @property
    def biKv43(self) -> float:
        return self.__cpp_struct.biKv43

    @biKv43.setter
    def biKv43(self, value: float) -> None:
        assert_nonnegative(value, "biKv43")
        self.__cpp_struct.biKv43 = value

    @property
    def f1Kv43(self) -> float:
        return self.__cpp_struct.f1Kv43

    @f1Kv43.setter
    def f1Kv43(self, value: float) -> None:
        assert_nonnegative(value, "f1Kv43")
        self.__cpp_struct.f1Kv43 = value

    @property
    def f2Kv43(self) -> float:
        return self.__cpp_struct.f2Kv43

    @f2Kv43.setter
    def f2Kv43(self, value: float) -> None:
        assert_nonnegative(value, "f2Kv43")
        self.__cpp_struct.f2Kv43 = value

    @property
    def f3Kv43(self) -> float:
        return self.__cpp_struct.f3Kv43

    @f3Kv43.setter
    def f3Kv43(self, value: float) -> None:
        assert_nonnegative(value, "f3Kv43")
        self.__cpp_struct.f3Kv43 = value

    @property
    def f4Kv43(self) -> float:
        return self.__cpp_struct.f4Kv43

    @f4Kv43.setter
    def f4Kv43(self, value: float) -> None:
        assert_nonnegative(value, "f4Kv43")
        self.__cpp_struct.f4Kv43 = value

    @property
    def b1Kv43(self) -> float:
        return self.__cpp_struct.b1Kv43

    @b1Kv43.setter
    def b1Kv43(self, value: float) -> None:
        assert_nonnegative(value, "b1Kv43")
        self.__cpp_struct.b1Kv43 = value

    @property
    def b2Kv43(self) -> float:
        return self.__cpp_struct.b2Kv43

    @b2Kv43.setter
    def b2Kv43(self, value: float) -> None:
        assert_nonnegative(value, "b2Kv43")
        self.__cpp_struct.b2Kv43 = value

    @property
    def b3Kv43(self) -> float:
        return self.__cpp_struct.b3Kv43

    @b3Kv43.setter
    def b3Kv43(self, value: float) -> None:
        assert_nonnegative(value, "b3Kv43")
        self.__cpp_struct.b3Kv43 = value

    @property
    def b4Kv43(self) -> float:
        return self.__cpp_struct.b4Kv43

    @b4Kv43.setter
    def b4Kv43(self, value: float) -> None:
        assert_nonnegative(value, "b4Kv43")
        self.__cpp_struct.b4Kv43 = value

    @property
    def PKv14(self) -> float:
        return self.__cpp_struct.PKv14

    @PKv14.setter
    def PKv14(self, value: float) -> None:
        assert_positive(value, "PKv14")
        self.__cpp_struct.PKv14 = value

    @property
    def alphaa0Kv14(self) -> float:
        return self.__cpp_struct.alphaa0Kv14

    @alphaa0Kv14.setter
    def alphaa0Kv14(self, value: float) -> None:
        assert_nonnegative(value, "alphaa0Kv14")
        self.__cpp_struct.alphaa0Kv14 = value

    @property
    def aaKv14(self) -> float:
        return self.__cpp_struct.aaKv14

    @aaKv14.setter
    def aaKv14(self, value: float) -> None:
        assert_nonnegative(value, "aaKv14")
        self.__cpp_struct.aaKv14 = value

    @property
    def betaa0Kv14(self) -> float:
        return self.__cpp_struct.betaa0Kv14

    @betaa0Kv14.setter
    def betaa0Kv14(self, value: float) -> None:
        assert_nonnegative(value, "betaa0Kv14")
        self.__cpp_struct.betaa0Kv14 = value

    @property
    def baKv14(self) -> float:
        return self.__cpp_struct.baKv14

    @baKv14.setter
    def baKv14(self, value: float) -> None:
        assert_nonnegative(value, "baKv14")
        self.__cpp_struct.baKv14 = value

    @property
    def alphai0Kv14(self) -> float:
        return self.__cpp_struct.alphai0Kv14

    @alphai0Kv14.setter
    def alphai0Kv14(self, value: float) -> None:
        assert_nonnegative(value, "alphai0Kv14")
        self.__cpp_struct.alphai0Kv14 = value

    @property
    def aiKv14(self) -> float:
        return self.__cpp_struct.aiKv14

    @aiKv14.setter
    def aiKv14(self, value: float) -> None:
        assert_nonnegative(value, "aiKv14")
        self.__cpp_struct.aiKv14 = value

    @property
    def betai0Kv14(self) -> float:
        return self.__cpp_struct.betai0Kv14

    @betai0Kv14.setter
    def betai0Kv14(self, value: float) -> None:
        assert_nonnegative(value, "betai0Kv14")
        self.__cpp_struct.betai0Kv14 = value

    @property
    def biKv14(self) -> float:
        return self.__cpp_struct.biKv14

    @biKv14.setter
    def biKv14(self, value: float) -> None:
        assert_nonnegative(value, "biKv14")
        self.__cpp_struct.biKv14 = value

    @property
    def f1Kv14(self) -> float:
        return self.__cpp_struct.f1Kv14

    @f1Kv14.setter
    def f1Kv14(self, value: float) -> None:
        assert_nonnegative(value, "f1Kv14")
        self.__cpp_struct.f1Kv14 = value

    @property
    def f2Kv14(self) -> float:
        return self.__cpp_struct.f2Kv14

    @f2Kv14.setter
    def f2Kv14(self, value: float) -> None:
        assert_nonnegative(value, "f2Kv14")
        self.__cpp_struct.f2Kv14 = value

    @property
    def f3Kv14(self) -> float:
        return self.__cpp_struct.f3Kv14

    @f3Kv14.setter
    def f3Kv14(self, value: float) -> None:
        assert_nonnegative(value, "f3Kv14")
        self.__cpp_struct.f3Kv14 = value

    @property
    def f4Kv14(self) -> float:
        return self.__cpp_struct.f4Kv14

    @f4Kv14.setter
    def f4Kv14(self, value: float) -> None:
        assert_nonnegative(value, "f4Kv14")
        self.__cpp_struct.f4Kv14 = value

    @property
    def b1Kv14(self) -> float:
        return self.__cpp_struct.b1Kv14

    @b1Kv14.setter
    def b1Kv14(self, value: float) -> None:
        assert_nonnegative(value, "b1Kv14")
        self.__cpp_struct.b1Kv14 = value

    @property
    def b2Kv14(self) -> float:
        return self.__cpp_struct.b2Kv14

    @b2Kv14.setter
    def b2Kv14(self, value: float) -> None:
        assert_nonnegative(value, "b2Kv14")
        self.__cpp_struct.b2Kv14 = value

    @property
    def b3Kv14(self) -> float:
        return self.__cpp_struct.b3Kv14

    @b3Kv14.setter
    def b3Kv14(self, value: float) -> None:
        assert_nonnegative(value, "b3Kv14")
        self.__cpp_struct.b3Kv14 = value

    @property
    def b4Kv14(self) -> float:
        return self.__cpp_struct.b4Kv14

    @b4Kv14.setter
    def b4Kv14(self, value: float) -> None:
        assert_nonnegative(value, "b4Kv14")
        self.__cpp_struct.b4Kv14 = value

    @property
    def Csc(self) -> float:
        return self.__cpp_struct.Csc

    @Csc.setter
    def Csc(self, value: float) -> None:
        assert_positive(value, "Csc")
        self.__cpp_struct.Csc = value

    @property
    def GK1(self) -> float:
        return self.__cpp_struct.GK1

    @GK1.setter
    def GK1(self, value: float) -> None:
        assert_positive(value, "GK1")
        self.__cpp_struct.GK1 = value

    @property
    def KmK1(self) -> float:
        return self.__cpp_struct.KmK1

    @KmK1.setter
    def KmK1(self, value: float) -> None:
        assert_positive(value, "KmK1")
        self.__cpp_struct.KmK1 = value

    @property
    def GKp(self) -> float:
        return self.__cpp_struct.GKp

    @GKp.setter
    def GKp(self, value: float) -> None:
        assert_positive(value, "GKp")
        self.__cpp_struct.GKp = value

    @property
    def kNaCa(self) -> float:
        return self.__cpp_struct.kNaCa

    @kNaCa.setter
    def kNaCa(self, value: float) -> None:
        assert_positive(value, "kNaCa")
        self.__cpp_struct.kNaCa = value

    @property
    def KmNa(self) -> float:
        return self.__cpp_struct.KmNa

    @KmNa.setter
    def KmNa(self, value: float) -> None:
        assert_positive(value, "KmNa")
        self.__cpp_struct.KmNa = value

    @property
    def KmCa(self) -> float:
        return self.__cpp_struct.KmCa

    @KmCa.setter
    def KmCa(self, value: float) -> None:
        assert_positive(value, "KmCa")
        self.__cpp_struct.KmCa = value

    @property
    def ksat(self) -> float:
        return self.__cpp_struct.ksat

    @ksat.setter
    def ksat(self, value: float) -> None:
        assert_positive(value, "ksat")
        self.__cpp_struct.ksat = value

    @property
    def eta(self) -> float:
        return self.__cpp_struct.eta

    @eta.setter
    def eta(self, value: float) -> None:
        assert_positive(value, "eta")
        self.__cpp_struct.eta = value

    @property
    def INaKmax(self) -> float:
        return self.__cpp_struct.INaKmax

    @INaKmax.setter
    def INaKmax(self, value: float) -> None:
        assert_positive(value, "INaKmax")
        self.__cpp_struct.INaKmax = value

    @property
    def KmNai(self) -> float:
        return self.__cpp_struct.KmNai

    @KmNai.setter
    def KmNai(self, value: float) -> None:
        assert_positive(value, "KmNai")
        self.__cpp_struct.KmNai = value

    @property
    def KmKo(self) -> float:
        return self.__cpp_struct.KmKo

    @KmKo.setter
    def KmKo(self, value: float) -> None:
        assert_positive(value, "KmKo")
        self.__cpp_struct.KmKo = value

    @property
    def IpCamax(self) -> float:
        return self.__cpp_struct.IpCamax

    @IpCamax.setter
    def IpCamax(self, value: float) -> None:
        assert_positive(value, "IpCamax")
        self.__cpp_struct.IpCamax = value

    @property
    def KmpCa(self) -> float:
        return self.__cpp_struct.KmpCa

    @KmpCa.setter
    def KmpCa(self, value: float) -> None:
        assert_positive(value, "KmpCa")
        self.__cpp_struct.KmpCa = value

    @property
    def GCab(self) -> float:
        return self.__cpp_struct.GCab

    @GCab.setter
    def GCab(self, value: float) -> None:
        assert_positive(value, "GCab")
        self.__cpp_struct.GCab = value

    @property
    def GNab(self) -> float:
        return self.__cpp_struct.GNab

    @GNab.setter
    def GNab(self, value: float) -> None:
        assert_positive(value, "GNab")
        self.__cpp_struct.GNab = value

    @property
    def kHTRPNp(self) -> float:
        return self.__cpp_struct.kHTRPNp

    @kHTRPNp.setter
    def kHTRPNp(self, value: float) -> None:
        assert_positive(value, "kHTRPNp")
        self.__cpp_struct.kHTRPNp = value

    @property
    def kHTRPNm(self) -> float:
        return self.__cpp_struct.kHTRPNm

    @kHTRPNm.setter
    def kHTRPNm(self, value: float) -> None:
        assert_positive(value, "kHTRPNm")
        self.__cpp_struct.kHTRPNm = value

    @property
    def kLTRPNp(self) -> float:
        return self.__cpp_struct.kLTRPNp

    @kLTRPNp.setter
    def kLTRPNp(self, value: float) -> None:
        assert_positive(value, "kLTRPNp")
        self.__cpp_struct.kLTRPNp = value

    @property
    def kLTRPNm(self) -> float:
        return self.__cpp_struct.kLTRPNm

    @kLTRPNm.setter
    def kLTRPNm(self, value: float) -> None:
        assert_positive(value, "kLTRPNm")
        self.__cpp_struct.kLTRPNm = value

    @property
    def HTRPNtot(self) -> float:
        return self.__cpp_struct.HTRPNtot

    @HTRPNtot.setter
    def HTRPNtot(self, value: float) -> None:
        assert_positive(value, "HTRPNtot")
        self.__cpp_struct.HTRPNtot = value

    @property
    def LTRPNtot(self) -> float:
        return self.__cpp_struct.LTRPNtot

    @LTRPNtot.setter
    def LTRPNtot(self, value: float) -> None:
        assert_positive(value, "LTRPNtot")
        self.__cpp_struct.LTRPNtot = value

    @property
    def Vmaxf(self) -> float:
        return self.__cpp_struct.Vmaxf

    @Vmaxf.setter
    def Vmaxf(self, value: float) -> None:
        assert_positive(value, "Vmaxf")
        self.__cpp_struct.Vmaxf = value

    @property
    def Vmaxr(self) -> float:
        return self.__cpp_struct.Vmaxr

    @Vmaxr.setter
    def Vmaxr(self, value: float) -> None:
        assert_positive(value, "Vmaxr")
        self.__cpp_struct.Vmaxr = value

    @property
    def Kmf(self) -> float:
        return self.__cpp_struct.Kmf

    @Kmf.setter
    def Kmf(self, value: float) -> None:
        assert_positive(value, "Kmf")
        self.__cpp_struct.Kmf = value

    @property
    def Kmr(self) -> float:
        return self.__cpp_struct.Kmr

    @Kmr.setter
    def Kmr(self, value: float) -> None:
        assert_positive(value, "Kmr")
        self.__cpp_struct.Kmr = value

    @property
    def Hf(self) -> float:
        return self.__cpp_struct.Hf

    @Hf.setter
    def Hf(self, value: float) -> None:
        assert_positive(value, "Hf")
        self.__cpp_struct.Hf = value

    @property
    def Hr(self) -> float:
        return self.__cpp_struct.Hr

    @Hr.setter
    def Hr(self, value: float) -> None:
        assert_positive(value, "Hr")
        self.__cpp_struct.Hr = value
