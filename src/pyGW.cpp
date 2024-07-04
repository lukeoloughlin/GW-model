#include <random>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/functional.h>

#include "includes/common.hpp"
#include "includes/GW_utils.hpp"
#include "includes/GW.hpp"
#include "includes/xoshiro.hpp"

#include "pyGW.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace XoshiroCpp;

PyGWSimulation run_PRNG_arg(const GW::Parameters<double>& params, const int nCRU, double step_size, int num_steps, 
                            const std::function<double(double)>& Is, int record_every, std::string& rng){
    if (rng == "mt19937")
        return run<std::mt19937>(params, nCRU, step_size, num_steps, Is, record_every);
    else if (rng == "mt19937_64")
        return run<std::mt19937_64>(params, nCRU, step_size, num_steps, Is, record_every);
    else if (rng == "xoshiro256+")
        return run<Xoshiro256Plus>(params, nCRU, step_size, num_steps, Is, record_every);
    else if (rng == "xoshiro256++")
        return run<Xoshiro256PlusPlus>(params, nCRU, step_size, num_steps, Is, record_every);
    else if (rng == "xoshiro256**")
        return run<Xoshiro256StarStar>(params, nCRU, step_size, num_steps, Is, record_every);
    else if (rng == "xoroshiro128+")
        return run<Xoroshiro128Plus>(params, nCRU, step_size, num_steps, Is, record_every);
    else if (rng == "xoroshiro128++")
        return run<Xoroshiro128PlusPlus>(params, nCRU, step_size, num_steps, Is, record_every);
    else if (rng == "xoroshiro128**")
        return run<Xoroshiro128StarStar>(params, nCRU, step_size, num_steps, Is, record_every);
    else
        throw std::invalid_argument(rng);
}


PYBIND11_MODULE(GreensteinWinslow, m) {
    /**************** 
     Some basic functions
    *****************/

    // Add a doc string to module
    m.doc() = "Python interface to Greenstein and Winslow model simulation.";

 

    py::class_<GW::Parameters<double>>(m, "Parameters")
        .def_readwrite("T", &GW::Parameters<double>::T, "Temperature [K]")
        .def_readwrite("CSA", &GW::Parameters<double>::CSA, "Cell surface area capacitance [pF]")
        .def_readwrite("Vcyto", &GW::Parameters<double>::Vcyto, "Cytosolic volume [pL]")
        .def_readwrite("VNSR", &GW::Parameters<double>::VNSR, "NSR volume [pL]")
        .def_readwrite("VJSR", &GW::Parameters<double>::VJSR, "JSR volume [pL]")
        .def_readwrite("VSS", &GW::Parameters<double>::VSS, "Subspace volume [pL]")
        .def_readwrite("NCaRU", &GW::Parameters<double>::NCaRU, "True number of calcium release units")
        .def_readwrite("Ko", &GW::Parameters<double>::Ko, "Extracellular potassium concentration [mM]")
        .def_readwrite("Nao", &GW::Parameters<double>::Nao, "Extracellular sodium concentration [mM]")
        .def_readwrite("Cao", &GW::Parameters<double>::Cao, "Extracellular calcium concentration [mM]")
        .def_readwrite("Clo", &GW::Parameters<double>::Clo, "Extracellular chloride concentration [mM]")
        .def_readwrite("Clcyto", &GW::Parameters<double>::Clcyto, "Intracellular chloride concentration [mM]")
        .def_readwrite("f", &GW::Parameters<double>::f, "LCC transition rate into open state [ms]^{-1}")
        .def_readwrite("g", &GW::Parameters<double>::g, "LCC transition rate out of open state [ms]^{-1}")
        .def_readwrite("f1", &GW::Parameters<double>::f1, "LCC transition rate into open state in mode calcium [ms]^{-1}")
        .def_readwrite("g1", &GW::Parameters<double>::g1, "LCC transition rate out of open state in mode calcium [ms]^{-1}")
        .def_readwrite("a", &GW::Parameters<double>::a, "LCC state dependent transition to mode calcium rate parameter [ms]^{-1}")
        .def_readwrite("b", &GW::Parameters<double>::b, "LCC state dependent transition to mode voltage rate parameter [ms]^{-1}")
        .def_readwrite("gamma0", &GW::Parameters<double>::gamma0, "LCC mode voltage to mode calcium transition rate parameter [mM]^{-1}[ms]^{-1}")
        .def_readwrite("omega", &GW::Parameters<double>::omega, "LCC mode calcium to mode voltage transition rate parameter [ms]^{-1}")
        .def_readwrite("PCaL", &GW::Parameters<double>::PCaL, "L-type calcium channel permeability to calcium ions [cm]^3[s]^{-1}")
        .def_readwrite("kfClCh", &GW::Parameters<double>::kfClCh, "ClCh transition into open state [mM]^{-1}[ms]^{-1}")
        .def_readwrite("kbClCh", &GW::Parameters<double>::kbClCh, "ClCh transition into closed state [ms]^{-1}")
        .def_readwrite("Pto2", &GW::Parameters<double>::Pto2, "Calcium dependent chloride channel permiability to chloride [cm]^3[s]^{-1}")
        .def_readwrite("k12", &GW::Parameters<double>::k12, "RyR state 1 -> state 2 transition rate parameter [mM]^{-2}[ms]^{-1}")
        .def_readwrite("k21", &GW::Parameters<double>::k21, "RyR state 2 -> state 1 transition rate [ms]^{-1}")
        .def_readwrite("k23", &GW::Parameters<double>::k23, "RyR state 2 -> state 3 transition rate parameter [mM]^{-2}[ms]^{-1}")
        .def_readwrite("k32", &GW::Parameters<double>::k32, "RyR state 3 -> state 2 transition rate [ms]^{-1}")
        .def_readwrite("k34", &GW::Parameters<double>::k34, "RyR state 3 -> state 4 transition rate parameter [mM]^{-2}[ms]^{-1}")
        .def_readwrite("k43", &GW::Parameters<double>::k43, "RyR state 4 -> state 3 transition rate [ms]^{-1}")
        .def_readwrite("k45", &GW::Parameters<double>::k45, "RyR state 4 -> state 5 transition rate [ms]^{-1}")
        .def_readwrite("k54", &GW::Parameters<double>::k54, "RyR state 5 -> state 4 transition rate parameter [mM]^{-2}[ms]^{-1}")
        .def_readwrite("k56", &GW::Parameters<double>::k56, "RyR state 5 -> state 6 transition rate parameter [mM]^{-2}[ms]^{-1}")
        .def_readwrite("k65", &GW::Parameters<double>::k65, "RyR state 6 -> state 5 transition rate [ms]^{-1}")
        .def_readwrite("k25", &GW::Parameters<double>::k25, "RyR state 2 -> state 5 transition rate parameter [mM]^{-2}[ms]^{-1}")
        .def_readwrite("k52", &GW::Parameters<double>::k52, "RyR state 5 -> state 2 transition rate [ms]^{-1}")
        .def_readwrite("rRyR", &GW::Parameters<double>::rRyR, "Rate of calcium flux through an open RyR [ms]^{-1}")
        .def_readwrite("rxfer", &GW::Parameters<double>::rxfer, "Rate of calcium flux between subspace and cytosol [ms]^{-1}")
        .def_readwrite("rtr", &GW::Parameters<double>::rtr, "Rate of calcium flux between NSR and JSR [ms]^{-1}")
        .def_readwrite("riss", &GW::Parameters<double>::riss, "Intersubspace caclium flux rate [ms]^{-1}")
        .def_readwrite("BSRT", &GW::Parameters<double>::BSRT, "Total subspace SR membrane site concentration [mM]")
        .def_readwrite("KBSR", &GW::Parameters<double>::KBSR, "Calcium half-saturation constant for BSR [mM]")
        .def_readwrite("BSLT", &GW::Parameters<double>::BSLT, "Total subspace sarcolemma site concentration [mM]")
        .def_readwrite("KBSL", &GW::Parameters<double>::KBSL, "Calcium half-saturation constant for BSL [mM]")
        .def_readwrite("CSQNT", &GW::Parameters<double>::CSQNT, "Total JSR calsequestrin concentration [mM]")
        .def_readwrite("KCSQN", &GW::Parameters<double>::KCSQN, "Calcium half-saturation constant for calsequestrin [mM]")
        .def_readwrite("CMDNT", &GW::Parameters<double>::CMDNT, "Total cytosolic calmodulin concentration [mM]")
        .def_readwrite("KCMDN", &GW::Parameters<double>::KCMDN, "Calcium half-saturation constant for calmodulin [mM]")
        .def_readwrite("GNa", &GW::Parameters<double>::GNa, "Peak INa conductance [mS][μF]^{-1}")
        .def_readwrite("GKr", &GW::Parameters<double>::GKr, "Peak IKr conductance [mS][μF]^{-1}")
        .def_readwrite("Kf", &GW::Parameters<double>::Kf, "IKr Markov model state C2 -> state C3 transition [ms]^{-1}")
        .def_readwrite("Kb", &GW::Parameters<double>::Kb, "IKr Markov model state C3 -> state C2 transition [ms]^{-1}")
        .def_readwrite("GKs", &GW::Parameters<double>::GKs, "Peak IKs conductance [mS][μF]^{-1}")
        .def_readwrite("GKv43", &GW::Parameters<double>::GKv43, "Peak IKv4.3 conductance [mS][μF]^{-1}")
        .def_readwrite("alphaa0Kv43", &GW::Parameters<double>::alphaa0Kv43, "Kv4.3 Markov model upwards transition rate scaling parameter [ms]^{-1}")
        .def_readwrite("aaKv43", &GW::Parameters<double>::aaKv43, "Kv4.3 Markov model upwards transition rate exponent parameter [mV]^{-1}")
        .def_readwrite("betaa0Kv43", &GW::Parameters<double>::betaa0Kv43, "Kv4.3 Markov model downwards transition rate scaling parameter [ms]^{-1}")
        .def_readwrite("baKv43", &GW::Parameters<double>::baKv43, "Kv4.3 Markov model downwards transition rate exponent parameter [mV]^{-1}")
        .def_readwrite("alphai0Kv43", &GW::Parameters<double>::alphai0Kv43, "Kv4.3 Markov model active mode transition rate scaling parameter [ms]^{-1}")
        .def_readwrite("aiKv43", &GW::Parameters<double>::aiKv43, "Kv4.3 Markov model active mode transition rate exponent parameter [mV]^{-1}")
        .def_readwrite("betai0Kv43", &GW::Parameters<double>::betai0Kv43, "Kv4.3 Markov model inactive mode transition rate scaling parameter [ms]^{-1}")
        .def_readwrite("biKv43", &GW::Parameters<double>::biKv43, "Kv4.3 Markov model inactive mode transition rate exponent parameter [mV]^{-1}")
        .def_readwrite("f1Kv43", &GW::Parameters<double>::f1Kv43, "Kv4.3 Markov model state C1 -> state CI1 scaling parameter")
        .def_readwrite("f2Kv43", &GW::Parameters<double>::f2Kv43, "Kv4.3 Markov model state C2 -> state CI2 scaling parameter")
        .def_readwrite("f3Kv43", &GW::Parameters<double>::f3Kv43, "Kv4.3 Markov model state C3 -> state CI3 scaling parameter")
        .def_readwrite("f4Kv43", &GW::Parameters<double>::f4Kv43, "Kv4.3 Markov model state C4 -> state CI4 scaling parameter")
        .def_readwrite("b1Kv43", &GW::Parameters<double>::b1Kv43, "Kv4.3 Markov model state CI1 -> state C1 inverse scaling parameter")
        .def_readwrite("b2Kv43", &GW::Parameters<double>::b2Kv43, "Kv4.3 Markov model state CI2 -> state C2 inverse scaling parameter")
        .def_readwrite("b3Kv43", &GW::Parameters<double>::b3Kv43, "Kv4.3 Markov model state CI3 -> state C3 inverse scaling parameter")
        .def_readwrite("b4Kv43", &GW::Parameters<double>::b4Kv43, "Kv4.3 Markov model state C4 -> state CI4 scaling parameter")
        .def_readwrite("PKv14", &GW::Parameters<double>::PKv14, "Kv1.4 channel permeability to potassium [cm][s]^{-1}")
        .def_readwrite("alphaa0Kv14", &GW::Parameters<double>::alphaa0Kv14, "Kv1.4 Markov model upwards transition rate scaling parameter [ms]^{-1}")
        .def_readwrite("aaKv14", &GW::Parameters<double>::aaKv14, "Kv1.4 Markov model upwards transition rate exponent parameter [mV]^{-1}")
        .def_readwrite("betaa0Kv14", &GW::Parameters<double>::betaa0Kv14, "Kv1.4 Markov model downwards transition rate scaling parameter [ms]^{-1}")
        .def_readwrite("baKv14", &GW::Parameters<double>::baKv14, "Kv1.4 Markov model downwards transition rate exponent parameter [mV]^{-1}")
        .def_readwrite("alphai0Kv14", &GW::Parameters<double>::alphai0Kv14, "Kv1.4 Markov model active mode transition rate scaling parameter [ms]^{-1}")
        .def_readwrite("aiKv14", &GW::Parameters<double>::aiKv14, "Kv1.4 Markov model active mode transition rate exponent parameter [mV]^{-1}")
        .def_readwrite("betai0Kv14", &GW::Parameters<double>::betai0Kv14, "Kv1.4 Markov model inactive mode transition rate scaling parameter [ms]^{-1}")
        .def_readwrite("biKv14", &GW::Parameters<double>::biKv14, "Kv1.4 Markov model inactive mode transition rate exponent parameter [mV]^{-1}")
        .def_readwrite("f1Kv14", &GW::Parameters<double>::f1Kv14, "Kv1.4 Markov model state C1 -> state CI1 scaling parameter")
        .def_readwrite("f2Kv14", &GW::Parameters<double>::f2Kv14, "Kv1.4 Markov model state C2 -> state CI2 scaling parameter")
        .def_readwrite("f3Kv14", &GW::Parameters<double>::f3Kv14, "Kv1.4 Markov model state C3 -> state CI3 scaling parameter")
        .def_readwrite("f4Kv14", &GW::Parameters<double>::f4Kv14, "Kv1.4 Markov model state C4 -> state CI4 scaling parameter")
        .def_readwrite("b1Kv14", &GW::Parameters<double>::b1Kv14, "Kv1.4 Markov model state CI1 -> state C1 inverse scaling parameter")
        .def_readwrite("b2Kv14", &GW::Parameters<double>::b2Kv14, "Kv1.4 Markov model state CI2 -> state C2 inverse scaling parameter")
        .def_readwrite("b3Kv14", &GW::Parameters<double>::b3Kv14, "Kv1.4 Markov model state CI3 -> state C3 inverse scaling parameter")
        .def_readwrite("b4Kv14", &GW::Parameters<double>::b4Kv14, "Kv1.4 Markov model state C4 -> state CI4 scaling parameter")
        .def_readwrite("Csc", &GW::Parameters<double>::Csc, "Specific membrane capacity [pF][cm]^{-2}")
        .def_readwrite("GK1", &GW::Parameters<double>::GK1, "Peak IK1 conductance [mS][μF]^{-1}")
        .def_readwrite("KmK1", &GW::Parameters<double>::KmK1, "Potassium half-saturation for IK1 [mM]")
        .def_readwrite("GKp", &GW::Parameters<double>::GKp, "Peak IKp conductance [mS][μF]^{-1}")
        .def_readwrite("kNaCa", &GW::Parameters<double>::kNaCa, "Scaling factor for sodium-calcium exchange [pA][pF]^{-1}")
        .def_readwrite("KmNa", &GW::Parameters<double>::KmNa, "Sodium half-saturation constant for sodium-calcium exchange [mM]")
        .def_readwrite("KmCa", &GW::Parameters<double>::KmCa, "Calcium half-saturation constant for sodium-calcium exchange [mM]")
        .def_readwrite("ksat", &GW::Parameters<double>::ksat, "Sodium-calcium exchange saturation factor at negative potentials")
        .def_readwrite("eta", &GW::Parameters<double>::eta, "Voltage dependence parameter for sodium-calcium exchange")
        .def_readwrite("INaKmax", &GW::Parameters<double>::INaKmax, "Maximum sodium-potassium pump current [pA][pF]^{-1}")
        .def_readwrite("KmNai", &GW::Parameters<double>::KmNai, "Sodium half-saturation constant for sodium-potassium pump [mM]")
        .def_readwrite("KmKo", &GW::Parameters<double>::KmKo, "Potassium half-saturation constant for sodium-potassium pump [mM]")
        .def_readwrite("IpCamax", &GW::Parameters<double>::IpCamax, "Maximum sarcolemmal calcium pump current [pA][pF]^{-1}")
        .def_readwrite("KmpCa", &GW::Parameters<double>::KmpCa, "Half-saturation constant for sarcolemmal calcium pump [mM]")
        .def_readwrite("GCab", &GW::Parameters<double>::GCab, "Peak ICab conductance [mS][μF]^{-1}")
        .def_readwrite("GNab", &GW::Parameters<double>::GNab, "Peak INab conductance [mS][μF]^{-1}")
        .def_readwrite("kHTRPNp", &GW::Parameters<double>::kHTRPNp, "Calcium on rate for troponin high-affinity sites [mM][ms]^{-1}")
        .def_readwrite("kHTRPNm", &GW::Parameters<double>::kHTRPNm, "Calcium off rate for troponin high-affinity sites [ms]^{-1}")
        .def_readwrite("kLTRPNp", &GW::Parameters<double>::kLTRPNp, "Calcium on rate for troponin low-affinity sites [mM][ms]^{-1}")
        .def_readwrite("kLTRPNm", &GW::Parameters<double>::kLTRPNm, "Calcium off rate for troponin low-affinity sites [ms]^{-1}")
        .def_readwrite("HTRPNtot", &GW::Parameters<double>::HTRPNtot, "Total troponin high-affinity site concentration [mM]")
        .def_readwrite("LTRPNtot", &GW::Parameters<double>::LTRPNtot, "Total troponin low-affinity site concentration [mM]")
        .def_readwrite("Vmaxf", &GW::Parameters<double>::Vmaxf, "Calcium ATPase forward rate parameter [mM][ms]^{-1}")
        .def_readwrite("Vmaxr", &GW::Parameters<double>::Vmaxr, "Calcium ATPase reverse rate parameter [mM][ms]^{-1}")
        .def_readwrite("Kmf", &GW::Parameters<double>::Kmf, "Forward half-saturation constant for calcium ATPase [mM]")
        .def_readwrite("Kmr", &GW::Parameters<double>::Kmr, "Backward half-saturation constant for calcium ATPase [mM]")
        .def_readwrite("Hf", &GW::Parameters<double>::Hf, "Forward cooperativity constant for calcium ATPase")
        .def_readwrite("Hr", &GW::Parameters<double>::Hr, "Reverse cooperativity constant for calcium ATPase")
        .def("__repr__", [](const GW::Parameters<double> &x) { return "Greenstein and Winslow Model parameters object"; });


    
    py::class_<PyGWSimulation>(m, "GWVariables")
        .def(py::init<int,int,double>())
        .def_readwrite("t", &PyGWSimulation::t)
        .def_readwrite("V", &PyGWSimulation::V)
        .def_readwrite("m", &PyGWSimulation::m)
        .def_readwrite("h", &PyGWSimulation::h)
        .def_readwrite("j", &PyGWSimulation::j)
        .def_readwrite("Nai", &PyGWSimulation::Nai)
        .def_readwrite("Ki", &PyGWSimulation::Ki)
        .def_readwrite("Cai", &PyGWSimulation::Cai)
        .def_readwrite("CaNSR", &PyGWSimulation::CaNSR)
        .def_readwrite("CaLTRPN", &PyGWSimulation::CaLTRPN)
        .def_readwrite("CaHTRPN", &PyGWSimulation::CaHTRPN)
        .def_readwrite("xKs", &PyGWSimulation::xKs)
        .def_readwrite("XKr", &PyGWSimulation::XKr)
        .def_readwrite("XKv14", &PyGWSimulation::XKv14)
        .def_readwrite("XKv43", &PyGWSimulation::XKv43)
        .def_readwrite("CaJSR", &PyGWSimulation::CaJSR)
        .def_readwrite("CaSS", &PyGWSimulation::CaSS)
        .def_readwrite("LCC", &PyGWSimulation::LCC)
        .def_readwrite("LCC_inactivation", &PyGWSimulation::LCC_inactivation)
        .def_readwrite("RyR", &PyGWSimulation::RyR)
        .def_readwrite("ClCh", &PyGWSimulation::ClCh)
        .def("__repr__", [](const PyGWSimulation &x) {return "Greenstein and Winslow model solution over the interval [0, " + std::to_string(x.tspan) + "] with " + std::to_string(x.nCRU) + " CRUs"; });
   

    m.def("run", &run_PRNG_arg, "Simulate the model", "parameters"_a, "nCRU"_a, "step_size"_a, "num_steps"_a, 
                                                      "Istim"_a, "record_every"_a, "PRNG"_a = "mt19937_64", 
                                                       py::call_guard<py::gil_scoped_release>());
}