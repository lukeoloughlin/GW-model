from .utils import assert_positive, assert_nonnegative, assert_type  # type: ignore
import build.GreensteinWinslow as gw  # type: ignore


# NOTE: I couldn't think of a good way to automatically make a python interface to the C++ struct such that domain constraints could be checked and documentation could be added.
#       So, I used the below helper methods to generate the getters and setters as strings, printed them out, and altered the relevant fields manually (changed NCaRU to int and made some rate
#        parameters nonnegative)

_GW_NAMES = [
    ("T", "p", "Temperature [K]", 310.0),
    ("CSA", "p", "Cell surface area capacitance [pF]", 153.4),
    ("Vcyto", "p", "Cytosolic volume [pL]", 25.84),
    ("VNSR", "p", "NSR volume [pL]", 1.113),
    ("VJSR", "p", "JSR volume [pL]", 22.26e-6),
    ("VSS", "p", "Subspace volume [pL]", 0.2303e-6),
    ("NCaRU", "p", "True number of calcium release units", 12500),
    ("Ko", "p", "Extracellular potassium concentration [mM]", 4.0),
    ("Nao", "p", "Extracellular sodium concentration [mM]", 138.0),
    ("Cao", "p", "Extracellular calcium concentration [mM]", 2.0),
    ("Clo", "p", "Extracellular chloride concentration [mM]", 150.0),
    ("Clcyto", "p", "Intracellular chloride concentration [mM]", 20.0),
    ("f", "nn", "LCC transition rate into open state [ms]^{-1}", 0.85),
    ("g", "nn", "LCC transition rate out of open state [ms]^{-1}", 2.0),
    (
        "f1",
        "nn",
        "LCC transition rate into open state in mode calcium [ms]^{-1}",
        0.005,
    ),
    (
        "g1",
        "nn",
        "LCC transition rate out of open state in mode calcium [ms]^{-1}",
        7.0,
    ),
    (
        "a",
        "nn",
        "LCC state dependent transition to mode calcium rate parameter [ms]^{-1}",
        2.0,
    ),
    (
        "b",
        "p",
        "LCC state dependent transition to mode voltage rate parameter [ms]^{-1}",
        1.9356,
    ),
    (
        "gamma0",
        "nn",
        "LCC mode voltage to mode calcium transition rate parameter [mM]^{-1}[ms]^{-1}",
        0.44,
    ),
    (
        "omega",
        "nn",
        "LCC mode calcium to mode voltage transition rate parameter [ms]^{-1}",
        0.02158,
    ),
    (
        "PCaL",
        "p",
        "L-type calcium channel permeability to calcium ions [cm]^3[s]^{-1}",
        9.13e-13,
    ),
    ("kfClCh", "nn", "ClCh transition into open state [mM]^{-1}[ms]^{-1}", 13.3156),
    ("kbClCh", "nn", "ClCh transition into closed state [ms]^{-1}", 2.0),
    (
        "Pto2",
        "p",
        "Calcium dependent chloride channel permiability to chloride [cm]^3[s]^{-1}",
        2.65e-15,
    ),
    (
        "k12",
        "nn",
        "RyR state 1 -> state 2 transition rate parameter [mM]^{-2}[ms]^{-1}",
        877.5,
    ),
    (
        "k21",
        "nn",
        "RyR state 2 -> state 1 transition rate [ms]^{-1}",
        250.0,
    ),
    (
        "k23",
        "nn",
        "RyR state 2 -> state 3 transition rate parameter [mM]^{-2}[ms]^{-1}",
        2.358e8,
    ),
    (
        "k32",
        "nn",
        "RyR state 3 -> state 2 transition rate [ms]^{-1}",
        9.6,
    ),
    (
        "k34",
        "nn",
        "RyR state 3 -> state 4 transition rate parameter [mM]^{-2}[ms]^{-1}",
        1.415e6,
    ),
    (
        "k43",
        "nn",
        "RyR state 4 -> state 3 transition rate [ms]^{-1}",
        13.65,
    ),
    (
        "k45",
        "nn",
        "RyR state 4 -> state 5 transition rate [ms]^{-1}",
        0.07,
    ),
    (
        "k54",
        "nn",
        "RyR state 5 -> state 4 transition rate parameter [mM]^{-2}[ms]^{-1}",
        93.385,
    ),
    (
        "k56",
        "nn",
        "RyR state 5 -> state 6 transition rate parameter [mM]^{-2}[ms]^{-1}",
        1.887e7,
    ),
    (
        "k65",
        "nn",
        "RyR state 6 -> state 5 transition rate [ms]^{-1}",
        30.0,
    ),
    (
        "k25",
        "nn",
        "RyR state 2 -> state 5 transition rate parameter [mM]^{-2}[ms]^{-1}",
        2.358e6,
    ),
    (
        "k52",
        "nn",
        "RyR state 5 -> state 2 transition rate [ms]^{-1}",
        0.001235,
    ),
    ("rRyR", "p", "Rate of calcium flux through an open RyR [ms]^{-1}", 3.92),
    (
        "rxfer",
        "p",
        "Rate of calcium flux between subspace and cytosol [ms]^{-1}",
        200.0,
    ),
    ("rtr", "p", "Rate of calcium flux between NSR and JSR [ms]^{-1}", 0.333),
    ("riss", "p", "Intersubspace caclium flux rate [ms]^{-1}", 20.0),
    ("BSRT", "p", "Total subspace SR membrane site concentration [mM]", 0.047),
    ("KBSR", "p", "Calcium half-saturation constant for BSR [mM]", 0.00087),
    ("BSLT", "p", "Total subspace sarcolemma site concentration [mM]", 1.124),
    ("KBSL", "p", "Calcium half-saturation constant for BSL [mM]", 0.0087),
    ("CSQNT", "p", "Total JSR calsequestrin concentration [mM]", 13.5),
    ("KCSQN", "p", "Calcium half-saturation constant for calsequestrin [mM]", 0.63),
    ("CMDNT", "p", "Total cytosolic calmodulin concentration [mM]", 0.05),
    ("KCMDN", "p", "Calcium half-saturation constant for calmodulin [mM]", 0.00238),
    ("GNa", "p", "Peak INa conductance [mS][μF]^{-1}", 12.8),
    ("GKr", "p", "Peak IKr conductance [mS][μF]^{-1}", 0.024),
    ("Kf", "nn", "IKr Markov model state C2 -> state C3 transition [ms]^{-1}", 0.0266),
    ("Kb", "nn", "IKr Markov model state C3 -> state C2 transition [ms]^{-1}", 0.1348),
    ("GKs", "p", "Peak IKs conductance [mS][μF]^{-1}", 0.00271),
    ("GKv43", "p", "Peak IKv4.3 conductance [mS][μF]^{-1}", 0.1389),
    (
        "alphaa0Kv43",
        "Kv4.3 Markov model upwards transition rate scaling parameter [ms]^{-1}",
        "nn",
        0.5437,
    ),
    (
        "aaKv43",
        "Kv4.3 Markov model upwards transition rate exponent parameter [mV]^{-1}",
        "nn",
        0.02898,
    ),
    (
        "betaa0Kv43",
        "Kv4.3 Markov model downwards transition rate scaling parameter [ms]^{-1}",
        "nn",
        0.08019,
    ),
    (
        "baKv43",
        "Kv4.3 Markov model downwards transition rate exponent parameter [mV]^{-1}",
        "nn",
        0.04684,
    ),
    (
        "alphai0Kv43",
        "Kv4.3 Markov model active mode transition rate scaling parameter [ms]^{-1}",
        "nn",
        0.04984,
    ),
    (
        "aiKv43",
        "Kv4.3 Markov model active mode transition rate exponent parameter [mV]^{-1}",
        "nn",
        3.37302e-4,
    ),
    (
        "betai0Kv43",
        "Kv4.3 Markov model inactive mode transition rate scaling parameter [ms]^{-1}",
        "nn",
        8.1948e-4,
    ),
    (
        "biKv43",
        "nn",
        "Kv4.3 Markov model inactive mode transition rate exponent parameter [mV]^{-1}",
        5.374e-8,
    ),
    (
        "f1Kv43",
        "p",
        "Kv4.3 Markov model state C1 -> state CI1 scaling parameter",
        1.8936,
    ),
    (
        "f2Kv43",
        "p",
        "Kv4.3 Markov model state C2 -> state CI2 scaling parameter",
        14.225,
    ),
    (
        "f3Kv43",
        "p",
        "Kv4.3 Markov model state C3 -> state CI3 scaling parameter",
        158.574,
    ),
    (
        "f4Kv43",
        "p",
        "Kv4.3 Markov model state C4 -> state CI4 scaling parameter",
        142.937,
    ),
    (
        "b1Kv43",
        "p",
        "Kv4.3 Markov model state CI1 -> state C1 inverse scaling parameter",
        6.7735,
    ),
    (
        "b2Kv43",
        "p",
        "Kv4.3 Markov model state CI2 -> state C2 inverse scaling parameter",
        15.621,
    ),
    (
        "b3Kv43",
        "p",
        "Kv4.3 Markov model state CI3 -> state C3 inverse scaling parameter",
        28.753,
    ),
    (
        "b4Kv43",
        "p",
        "Kv4.3 Markov model state C4 -> state CI4 scaling parameter",
        524.576,
    ),
    ("PKv14", "p", "Kv1.4 channel permeability to potassium [cm][s]^{-1}", 1.989e-7),
    (
        "alphaa0Kv14",
        "nn",
        "Kv1.4 Markov model upwards transition rate scaling parameter [ms]^{-1}",
        1.8931,
    ),
    (
        "aaKv14",
        "nn",
        "Kv1.4 Markov model upwards transition rate exponent parameter [mV]^{-1}",
        0.006950,
    ),
    (
        "betaa0Kv14",
        "nn",
        "Kv1.4 Markov model downwards transition rate scaling parameter [ms]^{-1}",
        0.01179,
    ),
    (
        "baKv14",
        "nn",
        "Kv1.4 Markov model downwards transition rate exponent parameter [mV]^{-1}",
        0.08527,
    ),
    (
        "alphai0Kv14",
        "nn",
        "Kv1.4 Markov model active mode transition rate scaling parameter [ms]^{-1}",
        0.002963,
    ),
    (
        "aiKv14",
        "nn",
        "Kv1.4 Markov model active mode transition rate exponent parameter [mV]^{-1}",
        0.0,
    ),
    (
        "betai0Kv14",
        "nn",
        "Kv1.4 Markov model inactive mode transition rate scaling parameter [ms]^{-1}",
        1.0571e-4,
    ),
    (
        "biKv14",
        "nn",
        "Kv1.4 Markov model inactive mode transition rate exponent parameter [mV]^{-1}",
        0.0,
    ),
    (
        "f1Kv14",
        "p",
        "Kv1.4 Markov model state C1 -> state CI1 scaling parameter",
        0.2001,
    ),
    (
        "f2Kv14",
        "p",
        "Kv1.4 Markov model state C2 -> state CI2 scaling parameter",
        0.3203,
    ),
    (
        "f3Kv14",
        "p",
        "Kv1.4 Markov model state C3 -> state CI3 scaling parameter",
        13.509,
    ),
    (
        "f4Kv14",
        "p",
        "Kv1.4 Markov model state C4 -> state CI4 scaling parameter",
        1151.765,
    ),
    (
        "b1Kv14",
        "p",
        "Kv1.4 Markov model state CI1 -> state C1 inverse scaling parameter",
        2.230,
    ),
    (
        "b2Kv14",
        "p",
        "Kv1.4 Markov model state CI2 -> state C2 inverse scaling parameter",
        12.0,
    ),
    (
        "b3Kv14",
        "p",
        "Kv1.4 Markov model state CI3 -> state C3 inverse scaling parameter",
        5.370,
    ),
    (
        "b4Kv14",
        "p",
        "Kv1.4 Markov model state C4 -> state CI4 scaling parameter",
        5.240,
    ),
    ("Csc", "p", "Specific membrane capacity [pF][cm]^{-2}", 1.0e6),
    ("GK1", "p", "Peak IK1 conductance [mS][μF]^{-1}", 3.0),
    ("KmK1", "p", "Potassium half-saturation for IK1 [mM]", 13.0),
    ("GKp", "p", "Peak IKp conductance [mS][μF]^{-1}", 0.002659),
    ("kNaCa", "p", "Scaling factor for sodium-calcium exchange [pA][pF]^{-1}", 0.27),
    (
        "KmNa",
        "p",
        "Sodium half-saturation constant for sodium-calcium exchange [mM]",
        87.5,
    ),
    (
        "KmCa",
        "p",
        "Calcium half-saturation constant for sodium-calcium exchange [mM]",
        1.38,
    ),
    (
        "ksat",
        "p",
        "Sodium-calcium exchange saturation factor at negative potentials",
        0.2,
    ),
    ("eta", "p", "Voltage dependence parameter for sodium-calcium exchange", 0.35),
    ("INaKmax", "p", "Maximum sodium-potassium pump current [pA][pF]^{-1}", 0.901),
    (
        "KmNai",
        "p",
        "Sodium half-saturation constant for sodium-potassium pump [mM]",
        10.0,
    ),
    (
        "KmKo",
        "p",
        "Potassium half-saturation constant for sodium-potassium pump [mM]",
        1.5,
    ),
    ("IpCamax", "p", "Maximum sarcolemmal calcium pump current [pA][pF]^{-1}", 0.03),
    (
        "KmpCa",
        "p",
        "Half-saturation constant for sarcolemmal calcium pump [mM]",
        0.0005,
    ),
    ("GCab", "p", "Peak ICab conductance [mS][μF]^{-1}", 0.0002536),
    ("GNab", "p", "Peak INab conductance [mS][μF]^{-1}", 0.00264),
    (
        "kHTRPNp",
        "p",
        "Calcium on rate for troponin high-affinity sites [mM][ms]^{-1}",
        20.0,
    ),
    (
        "kHTRPNm",
        "p",
        "Calcium off rate for troponin high-affinity sites [ms]^{-1}",
        6.60e-5,
    ),
    (
        "kLTRPNp",
        "p",
        "Calcium on rate for troponin low-affinity sites [mM][ms]^{-1}",
        40.0,
    ),
    (
        "kLTRPNm",
        "p",
        "Calcium off rate for troponin low-affinity sites [ms]^{-1}",
        0.04,
    ),
    ("HTRPNtot", "p", "Total troponin high-affinity site concentration [mM]", 0.140),
    ("LTRPNtot", "p", "Total troponin low-affinity site concentration [mM]", 0.070),
    ("Vmaxf", "p", "Calcium ATPase forward rate parameter [mM][ms]^{-1}", 0.0002096),
    ("Vmaxr", "p", "Calcium ATPase reverse rate parameter [mM][ms]^{-1}", 0.0002096),
    ("Kmf", "p", "Forward half-saturation constant for calcium ATPase [mM]", 0.000260),
    ("Kmr", "p", "Backward half-saturation constant for calcium ATPase [mM]", 1.8),
    ("Hf", "p", "Forward cooperativity constant for calcium ATPase", 0.75),
    ("Hr", "p", "Reverse cooperativity constant for calcium ATPase", 0.75),
]


# Run this with _GW_NAMES and print result or write to file to get boiler plate for the below class


class GWParameters:
    """Greenstein and Winslow model parameters."""

    def __init__(self, cxx_struct: None | gw.Parameters = None, **kwargs):
        """Construct the GWParameters object. Uses default parameters unless otherwise specified.

        Args:
            cxx_struct (None | gw.Parameters, optional): C++ Parameters struct for the GW model. Defaults to None.
            **kwargs: Keyword arguments for any of the model parameters.
        """
        self.__cxx_struct: gw.Parameters = (
            cxx_struct if cxx_struct is not None else gw.Parameters()
        )
        self.__NCaRU_sim: int = 1250

        if "NCaRU_sim" in kwargs:
            self.NCaRU_sim = kwargs["NCaRU_sim"]

        for name, value in kwargs.items():
            if name in dir(self.__cxx_struct):
                setattr(self, name, value)
            else:
                raise ValueError(
                    f"{name} not a valid argument to GWParameters constructor. Call help(GWParameters) to see a list of acceptable parameter names"
                )

    @property
    def cxx_struct(self) -> gw.Parameters:
        """The C++ compatible struct.

        Returns:
            gw.GWParameters: Underlying C++ struct.
        """
        return self.__cxx_struct

    @property
    def NCaRU_sim(self) -> int:
        """int: Number of calcium release units to simulate. Value must be > 0. Defaults to 1250."""
        return self.__NCaRU_sim

    @NCaRU_sim.setter
    def NCaRU_sim(self, value: int) -> None:
        assert_type(value, int, "NCaRU_sim")
        assert_positive(value, "NCaRU_sim")
        self.__NCaRU_sim = value

    @property
    def T(self) -> float:
        """float: Temperature [K]. Value must be > 0. Defaults to 310.0."""
        return self.__cxx_struct.T

    @T.setter
    def T(self, value: float) -> None:
        assert_positive(value, "T")
        self.__cxx_struct.T = value

    @property
    def CSA(self) -> float:
        """float: Cell surface area capacitance [pF]. Value must be > 0. Defaults to 153.4."""
        return self.__cxx_struct.CSA

    @CSA.setter
    def CSA(self, value: float) -> None:
        assert_positive(value, "CSA")
        self.__cxx_struct.CSA = value

    @property
    def Vcyto(self) -> float:
        """float: Cytosolic volume [pL]. Value must be > 0. Defaults to 25.84."""
        return self.__cxx_struct.Vcyto

    @Vcyto.setter
    def Vcyto(self, value: float) -> None:
        assert_positive(value, "Vcyto")
        self.__cxx_struct.Vcyto = value

    @property
    def VNSR(self) -> float:
        """float: NSR volume [pL]. Value must be > 0. Defaults to 1.113."""
        return self.__cxx_struct.VNSR

    @VNSR.setter
    def VNSR(self, value: float) -> None:
        assert_positive(value, "VNSR")
        self.__cxx_struct.VNSR = value

    @property
    def VJSR(self) -> float:
        """float: JSR volume [pL]. Value must be > 0. Defaults to 2.226e-05."""
        return self.__cxx_struct.VJSR

    @VJSR.setter
    def VJSR(self, value: float) -> None:
        assert_positive(value, "VJSR")
        self.__cxx_struct.VJSR = value

    @property
    def VSS(self) -> float:
        """float: Subspace volume [pL]. Value must be > 0. Defaults to 2.303e-07."""
        return self.__cxx_struct.VSS

    @VSS.setter
    def VSS(self, value: float) -> None:
        assert_positive(value, "VSS")
        self.__cxx_struct.VSS = value

    @property
    def NCaRU(self) -> int:
        """int: True number of calcium release units. Value must be > 0. Defaults to 12500."""
        return self.__cxx_struct.NCaRU

    @NCaRU.setter
    def NCaRU(self, value: int) -> None:
        assert_type(value, int, "NCaRU")
        assert_positive(value, "NCaRU")
        self.__cxx_struct.NCaRU = value

    @property
    def Ko(self) -> float:
        """float: Extracellular potassium concentration [mM]. Value must be > 0. Defaults to 4.0."""
        return self.__cxx_struct.Ko

    @Ko.setter
    def Ko(self, value: float) -> None:
        assert_positive(value, "Ko")
        self.__cxx_struct.Ko = value

    @property
    def Nao(self) -> float:
        """float: Extracellular sodium concentration [mM]. Value must be > 0. Defaults to 138.0."""
        return self.__cxx_struct.Nao

    @Nao.setter
    def Nao(self, value: float) -> None:
        assert_positive(value, "Nao")
        self.__cxx_struct.Nao = value

    @property
    def Cao(self) -> float:
        """float: Extracellular calcium concentration [mM]. Value must be > 0. Defaults to 2.0."""
        return self.__cxx_struct.Cao

    @Cao.setter
    def Cao(self, value: float) -> None:
        assert_positive(value, "Cao")
        self.__cxx_struct.Cao = value

    @property
    def Clo(self) -> float:
        """float: Extracellular chloride concentration [mM]. Value must be > 0. Defaults to 150.0."""
        return self.__cxx_struct.Clo

    @Clo.setter
    def Clo(self, value: float) -> None:
        assert_positive(value, "Clo")
        self.__cxx_struct.Clo = value

    @property
    def Clcyto(self) -> float:
        """float: Intracellular chloride concentration [mM]. Value must be > 0. Defaults to 20.0."""
        return self.__cxx_struct.Clcyto

    @Clcyto.setter
    def Clcyto(self, value: float) -> None:
        assert_positive(value, "Clcyto")
        self.__cxx_struct.Clcyto = value

    @property
    def f(self) -> float:
        """float: LCC transition rate into open state [ms]^{-1}. Value must be >= 0. Defaults to 0.85."""
        return self.__cxx_struct.f

    @f.setter
    def f(self, value: float) -> None:
        assert_nonnegative(value, "f")
        self.__cxx_struct.f = value

    @property
    def g(self) -> float:
        """float: LCC transition rate out of open state [ms]^{-1}. Value must be >= 0. Defaults to 2.0."""
        return self.__cxx_struct.g

    @g.setter
    def g(self, value: float) -> None:
        assert_nonnegative(value, "g")
        self.__cxx_struct.g = value

    @property
    def f1(self) -> float:
        """float: LCC transition rate into open state in mode calcium [ms]^{-1}. Value must be >= 0. Defaults to 0.005."""
        return self.__cxx_struct.f1

    @f1.setter
    def f1(self, value: float) -> None:
        assert_nonnegative(value, "f1")
        self.__cxx_struct.f1 = value

    @property
    def g1(self) -> float:
        """float: LCC transition rate out of open state in mode calcium [ms]^{-1}. Value must be >= 0. Defaults to 7.0."""
        return self.__cxx_struct.g1

    @g1.setter
    def g1(self, value: float) -> None:
        assert_nonnegative(value, "g1")
        self.__cxx_struct.g1 = value

    @property
    def a(self) -> float:
        """float: LCC state dependent transition to mode calcium rate parameter [ms]^{-1}. Value must be >= 0. Defaults to 2.0."""
        return self.__cxx_struct.a

    @a.setter
    def a(self, value: float) -> None:
        assert_nonnegative(value, "a")
        self.__cxx_struct.a = value

    @property
    def b(self) -> float:
        """float: LCC state dependent transition to mode voltage rate parameter [ms]^{-1}. Value must be > 0. Defaults to 1.9356."""
        return self.__cxx_struct.b

    @b.setter
    def b(self, value: float) -> None:
        assert_positive(value, "b")
        self.__cxx_struct.b = value

    @property
    def gamma0(self) -> float:
        """float: LCC mode voltage to mode calcium transition rate parameter [mM]^{-1}[ms]^{-1}. Value must be >= 0. Defaults to 0.44."""
        return self.__cxx_struct.gamma0

    @gamma0.setter
    def gamma0(self, value: float) -> None:
        assert_nonnegative(value, "gamma0")
        self.__cxx_struct.gamma0 = value

    @property
    def omega(self) -> float:
        """float: LCC mode calcium to mode voltage transition rate parameter [ms]^{-1}. Value must be >= 0. Defaults to 0.02158."""
        return self.__cxx_struct.omega

    @omega.setter
    def omega(self, value: float) -> None:
        assert_nonnegative(value, "omega")
        self.__cxx_struct.omega = value

    @property
    def PCaL(self) -> float:
        """float: L-type calcium channel permeability to calcium ions [cm]^3[s]^{-1}. Value must be > 0. Defaults to 9.13e-13."""
        return self.__cxx_struct.PCaL

    @PCaL.setter
    def PCaL(self, value: float) -> None:
        assert_positive(value, "PCaL")
        self.__cxx_struct.PCaL = value

    @property
    def kfClCh(self) -> float:
        """float: ClCh transition into open state [mM]^{-1}[ms]^{-1}. Value must be >= 0. Defaults to 13.3156."""
        return self.__cxx_struct.kfClCh

    @kfClCh.setter
    def kfClCh(self, value: float) -> None:
        assert_nonnegative(value, "kfClCh")
        self.__cxx_struct.kfClCh = value

    @property
    def kbClCh(self) -> float:
        """float: ClCh transition into closed state [ms]^{-1}. Value must be >= 0. Defaults to 2.0."""
        return self.__cxx_struct.kbClCh

    @kbClCh.setter
    def kbClCh(self, value: float) -> None:
        assert_nonnegative(value, "kbClCh")
        self.__cxx_struct.kbClCh = value

    @property
    def Pto2(self) -> float:
        """float: Calcium dependent chloride channel permiability to chloride [cm]^3[s]^{-1}. Value must be > 0. Defaults to 2.65e-15."""
        return self.__cxx_struct.Pto2

    @Pto2.setter
    def Pto2(self, value: float) -> None:
        assert_positive(value, "Pto2")
        self.__cxx_struct.Pto2 = value

    @property
    def k12(self) -> float:
        """float: RyR state 1 -> state 2 transition rate parameter [mM]^{-2}[ms]^{-1}. Value must be >= 0. Defaults to 877.5."""
        return self.__cxx_struct.k12

    @k12.setter
    def k12(self, value: float) -> None:
        assert_nonnegative(value, "k12")
        self.__cxx_struct.k12 = value

    @property
    def k21(self) -> float:
        """float: RyR state 2 -> state 1 transition rate [ms]^{-1}. Value must be >= 0. Defaults to 250.0."""
        return self.__cxx_struct.k21

    @k21.setter
    def k21(self, value: float) -> None:
        assert_nonnegative(value, "k21")
        self.__cxx_struct.k21 = value

    @property
    def k23(self) -> float:
        """float: RyR state 2 -> state 3 transition rate parameter [mM]^{-2}[ms]^{-1}. Value must be >= 0. Defaults to 235800000.0."""
        return self.__cxx_struct.k23

    @k23.setter
    def k23(self, value: float) -> None:
        assert_nonnegative(value, "k23")
        self.__cxx_struct.k23 = value

    @property
    def k32(self) -> float:
        """float: RyR state 3 -> state 2 transition rate [ms]^{-1}. Value must be >= 0. Defaults to 9.6."""
        return self.__cxx_struct.k32

    @k32.setter
    def k32(self, value: float) -> None:
        assert_nonnegative(value, "k32")
        self.__cxx_struct.k32 = value

    @property
    def k34(self) -> float:
        """float: RyR state 3 -> state 4 transition rate parameter [mM]^{-2}[ms]^{-1}. Value must be >= 0. Defaults to 1415000.0."""
        return self.__cxx_struct.k34

    @k34.setter
    def k34(self, value: float) -> None:
        assert_nonnegative(value, "k34")
        self.__cxx_struct.k34 = value

    @property
    def k43(self) -> float:
        """float: RyR state 4 -> state 3 transition rate [ms]^{-1}. Value must be >= 0. Defaults to 13.65."""
        return self.__cxx_struct.k43

    @k43.setter
    def k43(self, value: float) -> None:
        assert_nonnegative(value, "k43")
        self.__cxx_struct.k43 = value

    @property
    def k45(self) -> float:
        """float: RyR state 4 -> state 5 transition rate [ms]^{-1}. Value must be >= 0. Defaults to 0.07."""
        return self.__cxx_struct.k45

    @k45.setter
    def k45(self, value: float) -> None:
        assert_nonnegative(value, "k45")
        self.__cxx_struct.k45 = value

    @property
    def k54(self) -> float:
        """float: RyR state 5 -> state 4 transition rate parameter [mM]^{-2}[ms]^{-1}. Value must be >= 0. Defaults to 93.385."""
        return self.__cxx_struct.k54

    @k54.setter
    def k54(self, value: float) -> None:
        assert_nonnegative(value, "k54")
        self.__cxx_struct.k54 = value

    @property
    def k56(self) -> float:
        """float: RyR state 5 -> state 6 transition rate parameter [mM]^{-2}[ms]^{-1}. Value must be >= 0. Defaults to 18870000.0."""
        return self.__cxx_struct.k56

    @k56.setter
    def k56(self, value: float) -> None:
        assert_nonnegative(value, "k56")
        self.__cxx_struct.k56 = value

    @property
    def k65(self) -> float:
        """float: RyR state 6 -> state 5 transition rate [ms]^{-1}. Value must be >= 0. Defaults to 30.0."""
        return self.__cxx_struct.k65

    @k65.setter
    def k65(self, value: float) -> None:
        assert_nonnegative(value, "k65")
        self.__cxx_struct.k65 = value

    @property
    def k25(self) -> float:
        """float: RyR state 2 -> state 5 transition rate parameter [mM]^{-2}[ms]^{-1}. Value must be >= 0. Defaults to 2358000.0."""
        return self.__cxx_struct.k25

    @k25.setter
    def k25(self, value: float) -> None:
        assert_nonnegative(value, "k25")
        self.__cxx_struct.k25 = value

    @property
    def k52(self) -> float:
        """float: RyR state 5 -> state 2 transition rate [ms]^{-1}. Value must be >= 0. Defaults to 0.001235."""
        return self.__cxx_struct.k52

    @k52.setter
    def k52(self, value: float) -> None:
        assert_nonnegative(value, "k52")
        self.__cxx_struct.k52 = value

    @property
    def rRyR(self) -> float:
        """float: Rate of calcium flux through an open RyR [ms]^{-1}. Value must be > 0. Defaults to 3.92."""
        return self.__cxx_struct.rRyR

    @rRyR.setter
    def rRyR(self, value: float) -> None:
        assert_positive(value, "rRyR")
        self.__cxx_struct.rRyR = value

    @property
    def rxfer(self) -> float:
        """float: Rate of calcium flux between subspace and cytosol [ms]^{-1}. Value must be > 0. Defaults to 200.0."""
        return self.__cxx_struct.rxfer

    @rxfer.setter
    def rxfer(self, value: float) -> None:
        assert_positive(value, "rxfer")
        self.__cxx_struct.rxfer = value

    @property
    def rtr(self) -> float:
        """float: Rate of calcium flux between NSR and JSR [ms]^{-1}. Value must be > 0. Defaults to 0.333."""
        return self.__cxx_struct.rtr

    @rtr.setter
    def rtr(self, value: float) -> None:
        assert_positive(value, "rtr")
        self.__cxx_struct.rtr = value

    @property
    def riss(self) -> float:
        """float: Intersubspace caclium flux rate [ms]^{-1}. Value must be > 0. Defaults to 20.0."""
        return self.__cxx_struct.riss

    @riss.setter
    def riss(self, value: float) -> None:
        assert_positive(value, "riss")
        self.__cxx_struct.riss = value

    @property
    def BSRT(self) -> float:
        """float: Total subspace SR membrane site concentration [mM]. Value must be > 0. Defaults to 0.047."""
        return self.__cxx_struct.BSRT

    @BSRT.setter
    def BSRT(self, value: float) -> None:
        assert_positive(value, "BSRT")
        self.__cxx_struct.BSRT = value

    @property
    def KBSR(self) -> float:
        """float: Calcium half-saturation constant for BSR [mM]. Value must be > 0. Defaults to 0.00087."""
        return self.__cxx_struct.KBSR

    @KBSR.setter
    def KBSR(self, value: float) -> None:
        assert_positive(value, "KBSR")
        self.__cxx_struct.KBSR = value

    @property
    def BSLT(self) -> float:
        """float: Total subspace sarcolemma site concentration [mM]. Value must be > 0. Defaults to 1.124."""
        return self.__cxx_struct.BSLT

    @BSLT.setter
    def BSLT(self, value: float) -> None:
        assert_positive(value, "BSLT")
        self.__cxx_struct.BSLT = value

    @property
    def KBSL(self) -> float:
        """float: Calcium half-saturation constant for BSL [mM]. Value must be > 0. Defaults to 0.0087."""
        return self.__cxx_struct.KBSL

    @KBSL.setter
    def KBSL(self, value: float) -> None:
        assert_positive(value, "KBSL")
        self.__cxx_struct.KBSL = value

    @property
    def CSQNT(self) -> float:
        """float: Total JSR calsequestrin concentration [mM]. Value must be > 0. Defaults to 13.5."""
        return self.__cxx_struct.CSQNT

    @CSQNT.setter
    def CSQNT(self, value: float) -> None:
        assert_positive(value, "CSQNT")
        self.__cxx_struct.CSQNT = value

    @property
    def KCSQN(self) -> float:
        """float: Calcium half-saturation constant for calsequestrin [mM]. Value must be > 0. Defaults to 0.63."""
        return self.__cxx_struct.KCSQN

    @KCSQN.setter
    def KCSQN(self, value: float) -> None:
        assert_positive(value, "KCSQN")
        self.__cxx_struct.KCSQN = value

    @property
    def CMDNT(self) -> float:
        """float: Total cytosolic calmodulin concentration [mM]. Value must be > 0. Defaults to 0.05."""
        return self.__cxx_struct.CMDNT

    @CMDNT.setter
    def CMDNT(self, value: float) -> None:
        assert_positive(value, "CMDNT")
        self.__cxx_struct.CMDNT = value

    @property
    def KCMDN(self) -> float:
        """float: Calcium half-saturation constant for calmodulin [mM]. Value must be > 0. Defaults to 0.00238."""
        return self.__cxx_struct.KCMDN

    @KCMDN.setter
    def KCMDN(self, value: float) -> None:
        assert_positive(value, "KCMDN")
        self.__cxx_struct.KCMDN = value

    @property
    def GNa(self) -> float:
        """float: Peak INa conductance [mS][μF]^{-1}. Value must be > 0. Defaults to 12.8."""
        return self.__cxx_struct.GNa

    @GNa.setter
    def GNa(self, value: float) -> None:
        assert_positive(value, "GNa")
        self.__cxx_struct.GNa = value

    @property
    def GKr(self) -> float:
        """float: Peak IKr conductance [mS][μF]^{-1}. Value must be > 0. Defaults to 0.024."""
        return self.__cxx_struct.GKr

    @GKr.setter
    def GKr(self, value: float) -> None:
        assert_positive(value, "GKr")
        self.__cxx_struct.GKr = value

    @property
    def Kf(self) -> float:
        """float: IKr Markov model state C2 -> state C3 transition [ms]^{-1}. Value must be >= 0. Defaults to 0.0266."""
        return self.__cxx_struct.Kf

    @Kf.setter
    def Kf(self, value: float) -> None:
        assert_nonnegative(value, "Kf")
        self.__cxx_struct.Kf = value

    @property
    def Kb(self) -> float:
        """float: IKr Markov model state C3 -> state C2 transition [ms]^{-1}. Value must be >= 0. Defaults to 0.1348."""
        return self.__cxx_struct.Kb

    @Kb.setter
    def Kb(self, value: float) -> None:
        assert_nonnegative(value, "Kb")
        self.__cxx_struct.Kb = value

    @property
    def GKs(self) -> float:
        """float: Peak IKs conductance [mS][μF]^{-1}. Value must be > 0. Defaults to 0.00271."""
        return self.__cxx_struct.GKs

    @GKs.setter
    def GKs(self, value: float) -> None:
        assert_positive(value, "GKs")
        self.__cxx_struct.GKs = value

    @property
    def GKv43(self) -> float:
        """float: Peak IKv4.3 conductance [mS][μF]^{-1}. Value must be > 0. Defaults to 0.1389."""
        return self.__cxx_struct.GKv43

    @GKv43.setter
    def GKv43(self, value: float) -> None:
        assert_positive(value, "GKv43")
        self.__cxx_struct.GKv43 = value

    @property
    def alphaa0Kv43(self) -> float:
        """float: nn. Value must be >= 0. Defaults to 0.5437."""
        return self.__cxx_struct.alphaa0Kv43

    @alphaa0Kv43.setter
    def alphaa0Kv43(self, value: float) -> None:
        assert_nonnegative(value, "alphaa0Kv43")
        self.__cxx_struct.alphaa0Kv43 = value

    @property
    def aaKv43(self) -> float:
        """float: nn. Value must be >= 0. Defaults to 0.02898."""
        return self.__cxx_struct.aaKv43

    @aaKv43.setter
    def aaKv43(self, value: float) -> None:
        assert_nonnegative(value, "aaKv43")
        self.__cxx_struct.aaKv43 = value

    @property
    def betaa0Kv43(self) -> float:
        """float: nn. Value must be >= 0. Defaults to 0.08019."""
        return self.__cxx_struct.betaa0Kv43

    @betaa0Kv43.setter
    def betaa0Kv43(self, value: float) -> None:
        assert_nonnegative(value, "betaa0Kv43")
        self.__cxx_struct.betaa0Kv43 = value

    @property
    def baKv43(self) -> float:
        """float: nn. Value must be >= 0. Defaults to 0.04684."""
        return self.__cxx_struct.baKv43

    @baKv43.setter
    def baKv43(self, value: float) -> None:
        assert_nonnegative(value, "baKv43")
        self.__cxx_struct.baKv43 = value

    @property
    def alphai0Kv43(self) -> float:
        """float: nn. Value must be >= 0. Defaults to 0.04984."""
        return self.__cxx_struct.alphai0Kv43

    @alphai0Kv43.setter
    def alphai0Kv43(self, value: float) -> None:
        assert_nonnegative(value, "alphai0Kv43")
        self.__cxx_struct.alphai0Kv43 = value

    @property
    def aiKv43(self) -> float:
        """float: nn. Value must be >= 0. Defaults to 0.000337302."""
        return self.__cxx_struct.aiKv43

    @aiKv43.setter
    def aiKv43(self, value: float) -> None:
        assert_nonnegative(value, "aiKv43")
        self.__cxx_struct.aiKv43 = value

    @property
    def betai0Kv43(self) -> float:
        """float: nn. Value must be >= 0. Defaults to 0.00081948."""
        return self.__cxx_struct.betai0Kv43

    @betai0Kv43.setter
    def betai0Kv43(self, value: float) -> None:
        assert_nonnegative(value, "betai0Kv43")
        self.__cxx_struct.betai0Kv43 = value

    @property
    def biKv43(self) -> float:
        """float: Kv4.3 Markov model inactive mode transition rate exponent parameter [mV]^{-1}. Value must be >= 0. Defaults to 5.374e-08."""
        return self.__cxx_struct.biKv43

    @biKv43.setter
    def biKv43(self, value: float) -> None:
        assert_nonnegative(value, "biKv43")
        self.__cxx_struct.biKv43 = value

    @property
    def f1Kv43(self) -> float:
        """float: Kv4.3 Markov model state C1 -> state CI1 scaling parameter. Value must be > 0. Defaults to 1.8936."""
        return self.__cxx_struct.f1Kv43

    @f1Kv43.setter
    def f1Kv43(self, value: float) -> None:
        assert_positive(value, "f1Kv43")
        self.__cxx_struct.f1Kv43 = value

    @property
    def f2Kv43(self) -> float:
        """float: Kv4.3 Markov model state C2 -> state CI2 scaling parameter. Value must be > 0. Defaults to 14.225."""
        return self.__cxx_struct.f2Kv43

    @f2Kv43.setter
    def f2Kv43(self, value: float) -> None:
        assert_positive(value, "f2Kv43")
        self.__cxx_struct.f2Kv43 = value

    @property
    def f3Kv43(self) -> float:
        """float: Kv4.3 Markov model state C3 -> state CI3 scaling parameter. Value must be > 0. Defaults to 158.574."""
        return self.__cxx_struct.f3Kv43

    @f3Kv43.setter
    def f3Kv43(self, value: float) -> None:
        assert_positive(value, "f3Kv43")
        self.__cxx_struct.f3Kv43 = value

    @property
    def f4Kv43(self) -> float:
        """float: Kv4.3 Markov model state C4 -> state CI4 scaling parameter. Value must be > 0. Defaults to 142.937."""
        return self.__cxx_struct.f4Kv43

    @f4Kv43.setter
    def f4Kv43(self, value: float) -> None:
        assert_positive(value, "f4Kv43")
        self.__cxx_struct.f4Kv43 = value

    @property
    def b1Kv43(self) -> float:
        """float: Kv4.3 Markov model state CI1 -> state C1 inverse scaling parameter. Value must be > 0. Defaults to 6.7735."""
        return self.__cxx_struct.b1Kv43

    @b1Kv43.setter
    def b1Kv43(self, value: float) -> None:
        assert_positive(value, "b1Kv43")
        self.__cxx_struct.b1Kv43 = value

    @property
    def b2Kv43(self) -> float:
        """float: Kv4.3 Markov model state CI2 -> state C2 inverse scaling parameter. Value must be > 0. Defaults to 15.621."""
        return self.__cxx_struct.b2Kv43

    @b2Kv43.setter
    def b2Kv43(self, value: float) -> None:
        assert_positive(value, "b2Kv43")
        self.__cxx_struct.b2Kv43 = value

    @property
    def b3Kv43(self) -> float:
        """float: Kv4.3 Markov model state CI3 -> state C3 inverse scaling parameter. Value must be > 0. Defaults to 28.753."""
        return self.__cxx_struct.b3Kv43

    @b3Kv43.setter
    def b3Kv43(self, value: float) -> None:
        assert_positive(value, "b3Kv43")
        self.__cxx_struct.b3Kv43 = value

    @property
    def b4Kv43(self) -> float:
        """float: Kv4.3 Markov model state C4 -> state CI4 scaling parameter. Value must be > 0. Defaults to 524.576."""
        return self.__cxx_struct.b4Kv43

    @b4Kv43.setter
    def b4Kv43(self, value: float) -> None:
        assert_positive(value, "b4Kv43")
        self.__cxx_struct.b4Kv43 = value

    @property
    def PKv14(self) -> float:
        """float: Kv1.4 channel permeability to potassium [cm][s]^{-1}. Value must be > 0. Defaults to 1.989e-07."""
        return self.__cxx_struct.PKv14

    @PKv14.setter
    def PKv14(self, value: float) -> None:
        assert_positive(value, "PKv14")
        self.__cxx_struct.PKv14 = value

    @property
    def alphaa0Kv14(self) -> float:
        """float: Kv1.4 Markov model upwards transition rate scaling parameter [ms]^{-1}. Value must be >= 0. Defaults to 1.8931."""
        return self.__cxx_struct.alphaa0Kv14

    @alphaa0Kv14.setter
    def alphaa0Kv14(self, value: float) -> None:
        assert_nonnegative(value, "alphaa0Kv14")
        self.__cxx_struct.alphaa0Kv14 = value

    @property
    def aaKv14(self) -> float:
        """float: Kv1.4 Markov model upwards transition rate exponent parameter [mV]^{-1}. Value must be >= 0. Defaults to 0.00695."""
        return self.__cxx_struct.aaKv14

    @aaKv14.setter
    def aaKv14(self, value: float) -> None:
        assert_nonnegative(value, "aaKv14")
        self.__cxx_struct.aaKv14 = value

    @property
    def betaa0Kv14(self) -> float:
        """float: Kv1.4 Markov model downwards transition rate scaling parameter [ms]^{-1}. Value must be >= 0. Defaults to 0.01179."""
        return self.__cxx_struct.betaa0Kv14

    @betaa0Kv14.setter
    def betaa0Kv14(self, value: float) -> None:
        assert_nonnegative(value, "betaa0Kv14")
        self.__cxx_struct.betaa0Kv14 = value

    @property
    def baKv14(self) -> float:
        """float: Kv1.4 Markov model downwards transition rate exponent parameter [mV]^{-1}. Value must be >= 0. Defaults to 0.08527."""
        return self.__cxx_struct.baKv14

    @baKv14.setter
    def baKv14(self, value: float) -> None:
        assert_nonnegative(value, "baKv14")
        self.__cxx_struct.baKv14 = value

    @property
    def alphai0Kv14(self) -> float:
        """float: Kv1.4 Markov model active mode transition rate scaling parameter [ms]^{-1}. Value must be >= 0. Defaults to 0.002963."""
        return self.__cxx_struct.alphai0Kv14

    @alphai0Kv14.setter
    def alphai0Kv14(self, value: float) -> None:
        assert_nonnegative(value, "alphai0Kv14")
        self.__cxx_struct.alphai0Kv14 = value

    @property
    def aiKv14(self) -> float:
        """float: Kv1.4 Markov model active mode transition rate exponent parameter [mV]^{-1}. Value must be >= 0. Defaults to 0.0."""
        return self.__cxx_struct.aiKv14

    @aiKv14.setter
    def aiKv14(self, value: float) -> None:
        assert_nonnegative(value, "aiKv14")
        self.__cxx_struct.aiKv14 = value

    @property
    def betai0Kv14(self) -> float:
        """float: Kv1.4 Markov model inactive mode transition rate scaling parameter [ms]^{-1}. Value must be >= 0. Defaults to 0.00010571."""
        return self.__cxx_struct.betai0Kv14

    @betai0Kv14.setter
    def betai0Kv14(self, value: float) -> None:
        assert_nonnegative(value, "betai0Kv14")
        self.__cxx_struct.betai0Kv14 = value

    @property
    def biKv14(self) -> float:
        """float: Kv1.4 Markov model inactive mode transition rate exponent parameter [mV]^{-1}. Value must be >= 0. Defaults to 0.0."""
        return self.__cxx_struct.biKv14

    @biKv14.setter
    def biKv14(self, value: float) -> None:
        assert_nonnegative(value, "biKv14")
        self.__cxx_struct.biKv14 = value

    @property
    def f1Kv14(self) -> float:
        """float: Kv1.4 Markov model state C1 -> state CI1 scaling parameter. Value must be > 0. Defaults to 0.2001."""
        return self.__cxx_struct.f1Kv14

    @f1Kv14.setter
    def f1Kv14(self, value: float) -> None:
        assert_positive(value, "f1Kv14")
        self.__cxx_struct.f1Kv14 = value

    @property
    def f2Kv14(self) -> float:
        """float: Kv1.4 Markov model state C2 -> state CI2 scaling parameter. Value must be > 0. Defaults to 0.3203."""
        return self.__cxx_struct.f2Kv14

    @f2Kv14.setter
    def f2Kv14(self, value: float) -> None:
        assert_positive(value, "f2Kv14")
        self.__cxx_struct.f2Kv14 = value

    @property
    def f3Kv14(self) -> float:
        """float: Kv1.4 Markov model state C3 -> state CI3 scaling parameter. Value must be > 0. Defaults to 13.509."""
        return self.__cxx_struct.f3Kv14

    @f3Kv14.setter
    def f3Kv14(self, value: float) -> None:
        assert_positive(value, "f3Kv14")
        self.__cxx_struct.f3Kv14 = value

    @property
    def f4Kv14(self) -> float:
        """float: Kv1.4 Markov model state C4 -> state CI4 scaling parameter. Value must be > 0. Defaults to 1151.765."""
        return self.__cxx_struct.f4Kv14

    @f4Kv14.setter
    def f4Kv14(self, value: float) -> None:
        assert_positive(value, "f4Kv14")
        self.__cxx_struct.f4Kv14 = value

    @property
    def b1Kv14(self) -> float:
        """float: Kv1.4 Markov model state CI1 -> state C1 inverse scaling parameter. Value must be > 0. Defaults to 2.23."""
        return self.__cxx_struct.b1Kv14

    @b1Kv14.setter
    def b1Kv14(self, value: float) -> None:
        assert_positive(value, "b1Kv14")
        self.__cxx_struct.b1Kv14 = value

    @property
    def b2Kv14(self) -> float:
        """float: Kv1.4 Markov model state CI2 -> state C2 inverse scaling parameter. Value must be > 0. Defaults to 12.0."""
        return self.__cxx_struct.b2Kv14

    @b2Kv14.setter
    def b2Kv14(self, value: float) -> None:
        assert_positive(value, "b2Kv14")
        self.__cxx_struct.b2Kv14 = value

    @property
    def b3Kv14(self) -> float:
        """float: Kv1.4 Markov model state CI3 -> state C3 inverse scaling parameter. Value must be > 0. Defaults to 5.37."""
        return self.__cxx_struct.b3Kv14

    @b3Kv14.setter
    def b3Kv14(self, value: float) -> None:
        assert_positive(value, "b3Kv14")
        self.__cxx_struct.b3Kv14 = value

    @property
    def b4Kv14(self) -> float:
        """float: Kv1.4 Markov model state C4 -> state CI4 scaling parameter. Value must be > 0. Defaults to 5.24."""
        return self.__cxx_struct.b4Kv14

    @b4Kv14.setter
    def b4Kv14(self, value: float) -> None:
        assert_positive(value, "b4Kv14")
        self.__cxx_struct.b4Kv14 = value

    @property
    def Csc(self) -> float:
        """float: Specific membrane capacity [pF][cm]^{-2}. Value must be > 0. Defaults to 1000000.0."""
        return self.__cxx_struct.Csc

    @Csc.setter
    def Csc(self, value: float) -> None:
        assert_positive(value, "Csc")
        self.__cxx_struct.Csc = value

    @property
    def GK1(self) -> float:
        """float: Peak IK1 conductance [mS][μF]^{-1}. Value must be > 0. Defaults to 3.0."""
        return self.__cxx_struct.GK1

    @GK1.setter
    def GK1(self, value: float) -> None:
        assert_positive(value, "GK1")
        self.__cxx_struct.GK1 = value

    @property
    def KmK1(self) -> float:
        """float: Potassium half-saturation for IK1 [mM]. Value must be > 0. Defaults to 13.0."""
        return self.__cxx_struct.KmK1

    @KmK1.setter
    def KmK1(self, value: float) -> None:
        assert_positive(value, "KmK1")
        self.__cxx_struct.KmK1 = value

    @property
    def GKp(self) -> float:
        """float: Peak IKp conductance [mS][μF]^{-1}. Value must be > 0. Defaults to 0.002659."""
        return self.__cxx_struct.GKp

    @GKp.setter
    def GKp(self, value: float) -> None:
        assert_positive(value, "GKp")
        self.__cxx_struct.GKp = value

    @property
    def kNaCa(self) -> float:
        """float: Scaling factor for sodium-calcium exchange [pA][pF]^{-1}. Value must be > 0. Defaults to 0.27."""
        return self.__cxx_struct.kNaCa

    @kNaCa.setter
    def kNaCa(self, value: float) -> None:
        assert_positive(value, "kNaCa")
        self.__cxx_struct.kNaCa = value

    @property
    def KmNa(self) -> float:
        """float: Sodium half-saturation constant for sodium-calcium exchange [mM]. Value must be > 0. Defaults to 87.5."""
        return self.__cxx_struct.KmNa

    @KmNa.setter
    def KmNa(self, value: float) -> None:
        assert_positive(value, "KmNa")
        self.__cxx_struct.KmNa = value

    @property
    def KmCa(self) -> float:
        """float: Calcium half-saturation constant for sodium-calcium exchange [mM]. Value must be > 0. Defaults to 1.38."""
        return self.__cxx_struct.KmCa

    @KmCa.setter
    def KmCa(self, value: float) -> None:
        assert_positive(value, "KmCa")
        self.__cxx_struct.KmCa = value

    @property
    def ksat(self) -> float:
        """float: Sodium-calcium exchange saturation factor at negative potentials. Value must be > 0. Defaults to 0.2."""
        return self.__cxx_struct.ksat

    @ksat.setter
    def ksat(self, value: float) -> None:
        assert_positive(value, "ksat")
        self.__cxx_struct.ksat = value

    @property
    def eta(self) -> float:
        """float: Voltage dependence parameter for sodium-calcium exchange. Value must be > 0. Defaults to 0.35."""
        return self.__cxx_struct.eta

    @eta.setter
    def eta(self, value: float) -> None:
        assert_positive(value, "eta")
        self.__cxx_struct.eta = value

    @property
    def INaKmax(self) -> float:
        """float: Maximum sodium-potassium pump current [pA][pF]^{-1}. Value must be > 0. Defaults to 0.901."""
        return self.__cxx_struct.INaKmax

    @INaKmax.setter
    def INaKmax(self, value: float) -> None:
        assert_positive(value, "INaKmax")
        self.__cxx_struct.INaKmax = value

    @property
    def KmNai(self) -> float:
        """float: Sodium half-saturation constant for sodium-potassium pump [mM]. Value must be > 0. Defaults to 10.0."""
        return self.__cxx_struct.KmNai

    @KmNai.setter
    def KmNai(self, value: float) -> None:
        assert_positive(value, "KmNai")
        self.__cxx_struct.KmNai = value

    @property
    def KmKo(self) -> float:
        """float: Potassium half-saturation constant for sodium-potassium pump [mM]. Value must be > 0. Defaults to 1.5."""
        return self.__cxx_struct.KmKo

    @KmKo.setter
    def KmKo(self, value: float) -> None:
        assert_positive(value, "KmKo")
        self.__cxx_struct.KmKo = value

    @property
    def IpCamax(self) -> float:
        """float: Maximum sarcolemmal calcium pump current [pA][pF]^{-1}. Value must be > 0. Defaults to 0.03."""
        return self.__cxx_struct.IpCamax

    @IpCamax.setter
    def IpCamax(self, value: float) -> None:
        assert_positive(value, "IpCamax")
        self.__cxx_struct.IpCamax = value

    @property
    def KmpCa(self) -> float:
        """float: Half-saturation constant for sarcolemmal calcium pump [mM]. Value must be > 0. Defaults to 0.0005."""
        return self.__cxx_struct.KmpCa

    @KmpCa.setter
    def KmpCa(self, value: float) -> None:
        assert_positive(value, "KmpCa")
        self.__cxx_struct.KmpCa = value

    @property
    def GCab(self) -> float:
        """float: Peak ICab conductance [mS][μF]^{-1}. Value must be > 0. Defaults to 0.0002536."""
        return self.__cxx_struct.GCab

    @GCab.setter
    def GCab(self, value: float) -> None:
        assert_positive(value, "GCab")
        self.__cxx_struct.GCab = value

    @property
    def GNab(self) -> float:
        """float: Peak INab conductance [mS][μF]^{-1}. Value must be > 0. Defaults to 0.00264."""
        return self.__cxx_struct.GNab

    @GNab.setter
    def GNab(self, value: float) -> None:
        assert_positive(value, "GNab")
        self.__cxx_struct.GNab = value

    @property
    def kHTRPNp(self) -> float:
        """float: Calcium on rate for troponin high-affinity sites [mM][ms]^{-1}. Value must be > 0. Defaults to 20.0."""
        return self.__cxx_struct.kHTRPNp

    @kHTRPNp.setter
    def kHTRPNp(self, value: float) -> None:
        assert_positive(value, "kHTRPNp")
        self.__cxx_struct.kHTRPNp = value

    @property
    def kHTRPNm(self) -> float:
        """float: Calcium off rate for troponin high-affinity sites [ms]^{-1}. Value must be > 0. Defaults to 6.6e-05."""
        return self.__cxx_struct.kHTRPNm

    @kHTRPNm.setter
    def kHTRPNm(self, value: float) -> None:
        assert_positive(value, "kHTRPNm")
        self.__cxx_struct.kHTRPNm = value

    @property
    def kLTRPNp(self) -> float:
        """float: Calcium on rate for troponin low-affinity sites [mM][ms]^{-1}. Value must be > 0. Defaults to 40.0."""
        return self.__cxx_struct.kLTRPNp

    @kLTRPNp.setter
    def kLTRPNp(self, value: float) -> None:
        assert_positive(value, "kLTRPNp")
        self.__cxx_struct.kLTRPNp = value

    @property
    def kLTRPNm(self) -> float:
        """float: Calcium off rate for troponin low-affinity sites [ms]^{-1}. Value must be > 0. Defaults to 0.04."""
        return self.__cxx_struct.kLTRPNm

    @kLTRPNm.setter
    def kLTRPNm(self, value: float) -> None:
        assert_positive(value, "kLTRPNm")
        self.__cxx_struct.kLTRPNm = value

    @property
    def HTRPNtot(self) -> float:
        """float: Total troponin high-affinity site concentration [mM]. Value must be > 0. Defaults to 0.14."""
        return self.__cxx_struct.HTRPNtot

    @HTRPNtot.setter
    def HTRPNtot(self, value: float) -> None:
        assert_positive(value, "HTRPNtot")
        self.__cxx_struct.HTRPNtot = value

    @property
    def LTRPNtot(self) -> float:
        """float: Total troponin low-affinity site concentration [mM]. Value must be > 0. Defaults to 0.07."""
        return self.__cxx_struct.LTRPNtot

    @LTRPNtot.setter
    def LTRPNtot(self, value: float) -> None:
        assert_positive(value, "LTRPNtot")
        self.__cxx_struct.LTRPNtot = value

    @property
    def Vmaxf(self) -> float:
        """float: Calcium ATPase forward rate parameter [mM][ms]^{-1}. Value must be > 0. Defaults to 0.0002096."""
        return self.__cxx_struct.Vmaxf

    @Vmaxf.setter
    def Vmaxf(self, value: float) -> None:
        assert_positive(value, "Vmaxf")
        self.__cxx_struct.Vmaxf = value

    @property
    def Vmaxr(self) -> float:
        """float: Calcium ATPase reverse rate parameter [mM][ms]^{-1}. Value must be > 0. Defaults to 0.0002096."""
        return self.__cxx_struct.Vmaxr

    @Vmaxr.setter
    def Vmaxr(self, value: float) -> None:
        assert_positive(value, "Vmaxr")
        self.__cxx_struct.Vmaxr = value

    @property
    def Kmf(self) -> float:
        """float: Forward half-saturation constant for calcium ATPase [mM]. Value must be > 0. Defaults to 0.00026."""
        return self.__cxx_struct.Kmf

    @Kmf.setter
    def Kmf(self, value: float) -> None:
        assert_positive(value, "Kmf")
        self.__cxx_struct.Kmf = value

    @property
    def Kmr(self) -> float:
        """float: Backward half-saturation constant for calcium ATPase [mM]. Value must be > 0. Defaults to 1.8."""
        return self.__cxx_struct.Kmr

    @Kmr.setter
    def Kmr(self, value: float) -> None:
        assert_positive(value, "Kmr")
        self.__cxx_struct.Kmr = value

    @property
    def Hf(self) -> float:
        """float: Forward cooperativity constant for calcium ATPase. Value must be > 0. Defaults to 0.75."""
        return self.__cxx_struct.Hf

    @Hf.setter
    def Hf(self, value: float) -> None:
        assert_positive(value, "Hf")
        self.__cxx_struct.Hf = value

    @property
    def Hr(self) -> float:
        """float: Reverse cooperativity constant for calcium ATPase. Value must be > 0. Defaults to 0.75."""
        return self.__cxx_struct.Hr

    @Hr.setter
    def Hr(self, value: float) -> None:
        assert_positive(value, "Hr")
        self.__cxx_struct.Hr = value
