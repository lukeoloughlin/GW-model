#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/functional.h>

#include "includes/common.hpp"
#include "includes/GW_utils.hpp"
#include "includes/GW.hpp"

#include "pyGW.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

template <typename T>
using PyParameters = GW::Parameters<T>;



PYBIND11_MODULE(GreensteinWinslow, m) {
    /**************** 
     Some basic functions
    *****************/

    // Add a doc string to module
    m.doc() = "Pybind11 test.";

    py::class_<PyParameters<double>>(m, "Parameters")
        .def(py::init<>())
        .def("__repr__", [](const PyParameters<double> &params) { return "Greenstein and Winslow Model Parameters"; })
        .def_readwrite("T", &PyParameters<double>::T, "Temperature")
        .def_readwrite("CSA", &PyParameters<double>::CSA, "Membrane Capacitance")
        .def_readwrite("Vcyto", &PyParameters<double>::Vcyto, "Volume of cytosol")
        .def_readwrite("VNSR", &PyParameters<double>::VNSR, "Volume of cytosol")
        .def_readwrite("VJSR", &PyParameters<double>::VJSR, "Volume of cytosol")
        .def_readwrite("VSS", &PyParameters<double>::VSS, "Volume of cytosol")
        .def_readwrite("NCaRU", &PyParameters<double>::NCaRU, "Volume of cytosol")
        .def_readwrite("Ko", &PyParameters<double>::Ko, "Volume of cytosol")
        .def_readwrite("Nao", &PyParameters<double>::Nao, "Volume of cytosol")
        .def_readwrite("Cao", &PyParameters<double>::Cao, "Volume of cytosol")
        .def_readwrite("Clo", &PyParameters<double>::Clo, "Volume of cytosol")
        .def_readwrite("Clcyto", &PyParameters<double>::Clcyto, "Volume of cytosol")
        .def_readwrite("f", &PyParameters<double>::f, "Volume of cytosol")
        .def_readwrite("g", &PyParameters<double>::g, "Volume of cytosol")
        .def_readwrite("f1", &PyParameters<double>::f1, "Volume of cytosol")
        .def_readwrite("g1", &PyParameters<double>::g1, "Volume of cytosol")
        .def_readwrite("a", &PyParameters<double>::a, "Volume of cytosol")
        .def_readwrite("b", &PyParameters<double>::b, "Volume of cytosol")
        .def_readwrite("gamma0", &PyParameters<double>::gamma0, "Volume of cytosol")
        .def_readwrite("omega", &PyParameters<double>::omega, "Volume of cytosol")
        .def_readwrite("PCaL", &PyParameters<double>::PCaL, "Volume of cytosol")
        .def_readwrite("kfClCh", &PyParameters<double>::kfClCh, "Volume of cytosol")
        .def_readwrite("kbClCh", &PyParameters<double>::kbClCh, "Volume of cytosol")
        .def_readwrite("Pto2", &PyParameters<double>::Pto2, "Volume of cytosol")
        .def_readwrite("k12", &PyParameters<double>::k12, "Volume of cytosol")
        .def_readwrite("k21", &PyParameters<double>::k21, "Volume of cytosol")
        .def_readwrite("k23", &PyParameters<double>::k23, "Volume of cytosol")
        .def_readwrite("k32", &PyParameters<double>::k32, "Volume of cytosol")
        .def_readwrite("k34", &PyParameters<double>::k34, "Volume of cytosol")
        .def_readwrite("k43", &PyParameters<double>::k43, "Volume of cytosol")
        .def_readwrite("k45", &PyParameters<double>::k45, "Volume of cytosol")
        .def_readwrite("k54", &PyParameters<double>::k54, "Volume of cytosol")
        .def_readwrite("k56", &PyParameters<double>::k56, "Volume of cytosol")
        .def_readwrite("k65", &PyParameters<double>::k65, "Volume of cytosol")
        .def_readwrite("k25", &PyParameters<double>::k25, "Volume of cytosol")
        .def_readwrite("k52", &PyParameters<double>::k52, "Volume of cytosol")
        .def_readwrite("rRyR", &PyParameters<double>::rRyR, "Volume of cytosol")
        .def_readwrite("rxfer", &PyParameters<double>::rxfer, "Volume of cytosol")
        .def_readwrite("rtr", &PyParameters<double>::rtr, "Volume of cytosol")
        .def_readwrite("riss", &PyParameters<double>::riss, "Volume of cytosol")
        .def_readwrite("BSRT", &PyParameters<double>::BSRT, "Volume of cytosol")
        .def_readwrite("KBSR", &PyParameters<double>::KBSR, "Volume of cytosol")
        .def_readwrite("BSLT", &PyParameters<double>::BSLT, "Volume of cytosol")
        .def_readwrite("KBSL", &PyParameters<double>::KBSL, "Volume of cytosol")
        .def_readwrite("CSQNT", &PyParameters<double>::CSQNT, "Volume of cytosol")
        .def_readwrite("KCSQN", &PyParameters<double>::KCSQN, "Volume of cytosol")
        .def_readwrite("CMDNT", &PyParameters<double>::CMDNT, "Volume of cytosol")
        .def_readwrite("KCMDN", &PyParameters<double>::KCMDN, "Volume of cytosol")
        .def_readwrite("GNa", &PyParameters<double>::GNa, "Conductance of sodium channels")
        .def_readwrite("GKr", &PyParameters<double>::GKr, "Conductance of sodium channels")
        .def_readwrite("Kf", &PyParameters<double>::Kf, "Conductance of sodium channels")
        .def_readwrite("Kb", &PyParameters<double>::Kb, "Conductance of sodium channels")
        .def_readwrite("GKs", &PyParameters<double>::GKs, "Conductance of sodium channels")
        .def_readwrite("GKv43", &PyParameters<double>::GKv43, "Conductance of sodium channels")
        .def_readwrite("alphaa0Kv43", &PyParameters<double>::alphaa0Kv43, "Conductance of sodium channels")
        .def_readwrite("aaKv43", &PyParameters<double>::aaKv43, "Conductance of sodium channels")
        .def_readwrite("betaa0Kv43", &PyParameters<double>::betaa0Kv43, "Conductance of sodium channels")
        .def_readwrite("baKv43", &PyParameters<double>::baKv43, "Conductance of sodium channels")
        .def_readwrite("alphai0Kv43", &PyParameters<double>::alphai0Kv43, "Conductance of sodium channels")
        .def_readwrite("aiKv43", &PyParameters<double>::aiKv43, "Conductance of sodium channels")
        .def_readwrite("betai0Kv43", &PyParameters<double>::betai0Kv43, "Conductance of sodium channels")
        .def_readwrite("biKv43", &PyParameters<double>::biKv43, "Conductance of sodium channels")
        .def_readwrite("f1Kv43", &PyParameters<double>::f1Kv43, "Conductance of sodium channels")
        .def_readwrite("f2Kv43", &PyParameters<double>::f2Kv43, "Conductance of sodium channels")
        .def_readwrite("f3Kv43", &PyParameters<double>::f3Kv43, "Conductance of sodium channels")
        .def_readwrite("f4Kv43", &PyParameters<double>::f4Kv43, "Conductance of sodium channels")
        .def_readwrite("b1Kv43", &PyParameters<double>::b1Kv43, "Conductance of sodium channels")
        .def_readwrite("b2Kv43", &PyParameters<double>::b2Kv43, "Conductance of sodium channels")
        .def_readwrite("b3Kv43", &PyParameters<double>::b3Kv43, "Conductance of sodium channels")
        .def_readwrite("b4Kv43", &PyParameters<double>::b4Kv43, "Conductance of sodium channels")
        .def_readwrite("PKv14", &PyParameters<double>::PKv14, "Conductance of sodium channels")
        .def_readwrite("alphaa0Kv14", &PyParameters<double>::alphaa0Kv14, "Conductance of sodium channels")
        .def_readwrite("aaKv14", &PyParameters<double>::aaKv14, "Conductance of sodium channels")
        .def_readwrite("betaa0Kv14", &PyParameters<double>::betaa0Kv14, "Conductance of sodium channels")
        .def_readwrite("baKv14", &PyParameters<double>::baKv14, "Conductance of sodium channels")
        .def_readwrite("alphai0Kv14", &PyParameters<double>::alphai0Kv14, "Conductance of sodium channels")
        .def_readwrite("aiKv14", &PyParameters<double>::aiKv14, "Conductance of sodium channels")
        .def_readwrite("betai0Kv14", &PyParameters<double>::betai0Kv14, "Conductance of sodium channels")
        .def_readwrite("biKv14", &PyParameters<double>::biKv14, "Conductance of sodium channels")
        .def_readwrite("f1Kv14", &PyParameters<double>::f1Kv14, "Conductance of sodium channels")
        .def_readwrite("f2Kv14", &PyParameters<double>::f2Kv14, "Conductance of sodium channels")
        .def_readwrite("f3Kv14", &PyParameters<double>::f3Kv14, "Conductance of sodium channels")
        .def_readwrite("f4Kv14", &PyParameters<double>::f4Kv14, "Conductance of sodium channels")
        .def_readwrite("b1Kv14", &PyParameters<double>::b1Kv14, "Conductance of sodium channels")
        .def_readwrite("b2Kv14", &PyParameters<double>::b2Kv14, "Conductance of sodium channels")
        .def_readwrite("b3Kv14", &PyParameters<double>::b3Kv14, "Conductance of sodium channels")
        .def_readwrite("b4Kv14", &PyParameters<double>::b4Kv14, "Conductance of sodium channels")
        .def_readwrite("Csc", &PyParameters<double>::Csc, "Conductance of sodium channels")
        .def_readwrite("GK1", &PyParameters<double>::GK1, "Conductance of sodium channels")
        .def_readwrite("KmK1", &PyParameters<double>::KmK1, "Conductance of sodium channels")
        .def_readwrite("GKp", &PyParameters<double>::GKp, "Conductance of sodium channels")
        .def_readwrite("kNaCa", &PyParameters<double>::kNaCa, "Conductance of sodium channels")
        .def_readwrite("KmNa", &PyParameters<double>::KmNa, "Conductance of sodium channels")
        .def_readwrite("KmCa", &PyParameters<double>::KmCa, "Conductance of sodium channels")
        .def_readwrite("ksat", &PyParameters<double>::ksat, "Conductance of sodium channels")
        .def_readwrite("eta", &PyParameters<double>::eta, "Conductance of sodium channels")
        .def_readwrite("INaKmax", &PyParameters<double>::INaKmax, "Conductance of sodium channels")
        .def_readwrite("KmNai", &PyParameters<double>::KmNai, "Conductance of sodium channels")
        .def_readwrite("KmKo", &PyParameters<double>::KmKo, "Conductance of sodium channels")
        .def_readwrite("IpCamax", &PyParameters<double>::IpCamax, "Conductance of sodium channels")
        .def_readwrite("KmpCa", &PyParameters<double>::KmpCa, "Conductance of sodium channels")
        .def_readwrite("GCab", &PyParameters<double>::GCab, "Conductance of sodium channels")
        .def_readwrite("GNab", &PyParameters<double>::GNab, "Conductance of sodium channels")
        .def_readwrite("kHTRPNp", &PyParameters<double>::kHTRPNp, "Conductance of sodium channels")
        .def_readwrite("kHTRPNm", &PyParameters<double>::kHTRPNm, "Conductance of sodium channels")
        .def_readwrite("kLTRPNp", &PyParameters<double>::kLTRPNp, "Conductance of sodium channels")
        .def_readwrite("kLTRPNm", &PyParameters<double>::kLTRPNm, "Conductance of sodium channels")
        .def_readwrite("HTRPNtot", &PyParameters<double>::HTRPNtot, "Conductance of sodium channels")
        .def_readwrite("LTRPNtot", &PyParameters<double>::LTRPNtot, "Conductance of sodium channels")
        .def_readwrite("Vmaxf", &PyParameters<double>::Vmaxf, "Conductance of sodium channels")
        .def_readwrite("Vmaxr", &PyParameters<double>::Vmaxr, "Conductance of sodium channels")
        .def_readwrite("Kmf", &PyParameters<double>::Kmf, "Conductance of sodium channels")
        .def_readwrite("Kmr", &PyParameters<double>::Kmr, "Conductance of sodium channels")
        .def_readwrite("Hf", &PyParameters<double>::Hf, "Conductance of sodium channels")
        .def_readwrite("Hr", &PyParameters<double>::Hr, "Conductance of sodium channels");

    
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
        .def_readwrite("LCC_activation", &PyGWSimulation::LCC_activation)
        .def_readwrite("RyR", &PyGWSimulation::RyR)
        .def_readwrite("ClCh", &PyGWSimulation::ClCh)
        .def("__repr__", [](const PyGWSimulation &x) {return "Greenstein and Winslow model solution over the interval [0, " + std::to_string(x.tspan) + "] with " + std::to_string(x.nCRU) + " CRUs"; });
   

    m.def("run", &run, "Simulate the model", "parameters"_a, "nCRU"_a, "step_size"_a, "num_steps"_a, "Istim"_a, "record_every"_a, py::call_guard<py::gil_scoped_release>());
}