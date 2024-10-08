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

#define PYBIND11_DETAILED_ERROR_MESSAGES

namespace py = pybind11;
using namespace pybind11::literals;

template <typename T>
using PyParameters = GW::Parameters<T>;

using namespace XoshiroCpp;

typedef Eigen::RowVectorXd Array1d;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Array2d;
typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Array2i;
typedef Eigen::Tensor<int,3,Eigen::RowMajor> Array3i;

struct PyGlobalState {
    double V = -91.382;
    double Nai = 10.0;
    double Ki = 131.84;
    double Cai = 1.45273e-4;
    double CaNSR = 0.908882;
    double CaLTRPN = 8.9282e-3;
    double CaHTRPN = 0.137617;
    double m = 5.33837e-4;
    double h = 0.996345;
    double j = 0.997315;
    double xKs = 2.04171e-4;
    Eigen::Matrix<double,1,5,Eigen::RowMajor> XKr;
    Eigen::Matrix<double,1,10,Eigen::RowMajor> XKv14;
    Eigen::Matrix<double,1,10,Eigen::RowMajor> XKv43;

    PyGlobalState() {
        XKr(0) = 0.999503;
        XKr(1) = 4.13720e-4;
        XKr(2) = 7.27568e-5; 
        XKr(3) = 8.73984e-6; 
        XKr(4) = 1.36159e-6;

        XKv14(0) = 0.722328;
        XKv14(1) = 0.101971; 
        XKv14(2) = 0.00539932; 
        XKv14(3) = 1.27081e-4; 
        XKv14(4) = 1.82742e-6; 
        XKv14(5) = 0.152769; 
        XKv14(6) = 0.00962328; 
        XKv14(7) = 0.00439043; 
        XKv14(8) = 0.00195348; 
        XKv14(9) = 0.00143629;
        
        XKv43(0) = 0.953060; 
        XKv43(1) = 0.0253906; 
        XKv43(2) = 2.53848e-4; 
        XKv43(3) = 1.12796e-6; 
        XKv43(4) = 1.87950e-9; 
        XKv43(5) = 0.0151370; 
        XKv43(6) = 0.00517622; 
        XKv43(7) = 8.96600e-4; 
        XKv43(8) = 8.17569e-5; 
        XKv43(9) = 2.24032e-6;
    }
};

struct PyCRUState {
    Array2d CaSS;
    Array1d CaJSR;
    Array2i LCC;
    Array2i LCC_inactivation;
    Array3i RyR;
    Array2i ClCh;

    PyCRUState(int nCRU) : CaSS(nCRU,4), CaJSR(nCRU), LCC(nCRU,4), LCC_inactivation(nCRU,4), RyR(nCRU,4,6), ClCh(nCRU,4) {}
};

GW::GlobalState<double> globals_from_python(const PyGlobalState &py_state){
    GW::GlobalState<double> state;
    state.V = py_state.V;
    state.Nai = py_state.Nai;
    state.Ki = py_state.Ki;
    state.Cai = py_state.Cai;
    state.CaNSR = py_state.CaNSR;
    state.CaLTRPN = py_state.CaLTRPN;
    state.CaHTRPN = py_state.CaHTRPN;
    state.m = py_state.m;
    state.h = py_state.h;
    state.j = py_state.j;
    state.xKs = py_state.xKs;

    for (int i = 0; i < 5; ++i){
        state.Kr[i] = py_state.XKr(i);
    }

    for (int i = 0; i < 10; ++i){
        state.Kv14[i] = py_state.XKv14(i);
        state.Kv43[i] = py_state.XKv43(i);
    }
    return state;
}

// Need this workaround because Python doesn't like Eigen::TensorMap
GW::CRUState<double> crus_from_python(const PyCRUState &py_state, const int ncru){
    GW::CRUState<double> state(ncru);
    state.CaSS = py_state.CaSS;
    state.CaJSR = py_state.CaJSR;
    state.LCC = py_state.LCC;
    state.LCC_inactivation = py_state.LCC_inactivation;
    state.ClCh = py_state.ClCh;

    for (int i = 0; i < ncru; ++i){
        for (int j = 0; j < 4; ++j){
            for (int k = 0; k < 6; ++k){
                state.RyR.array(i,j,k) = py_state.RyR(i,j,k);
            }
        }
    }

    return state;
}

PyGWSimulation run_PRNG_arg(const GW::Parameters<double>& params, const int nCRU, double step_size, int num_steps, 
                            const std::function<double(double)>& Is, int record_every, PyCRUState &init_crus, 
                            PyGlobalState &init_globals, std::string& rng){
    auto globals = globals_from_python(init_globals);
    auto crus = crus_from_python(init_crus, nCRU);
    if (rng == "mt19937")
        return run<std::mt19937>(params, nCRU, step_size, num_steps, Is, record_every, globals, crus);
    else if (rng == "mt19937_64")
        return run<std::mt19937_64>(params, nCRU, step_size, num_steps, Is, record_every, globals, crus);
    else if (rng == "xoshiro256+")
        return run<Xoshiro256Plus>(params, nCRU, step_size, num_steps, Is, record_every, globals, crus);
    else if (rng == "xoshiro256++")
        return run<Xoshiro256PlusPlus>(params, nCRU, step_size, num_steps, Is, record_every, globals, crus);
    else if (rng == "xoshiro256**")
        return run<Xoshiro256StarStar>(params, nCRU, step_size, num_steps, Is, record_every, globals, crus);
    else if (rng == "xoroshiro128+")
        return run<Xoroshiro128Plus>(params, nCRU, step_size, num_steps, Is, record_every, globals, crus);
    else if (rng == "xoroshiro128++")
        return run<Xoroshiro128PlusPlus>(params, nCRU, step_size, num_steps, Is, record_every, globals, crus);
    else if (rng == "xoroshiro128**")
        return run<Xoroshiro128StarStar>(params, nCRU, step_size, num_steps, Is, record_every, globals, crus);
    else
        throw std::invalid_argument(rng);
}


PYBIND11_MODULE(GreensteinWinslow, m) {
    /**************** 
     Some basic functions
    *****************/

    // Add a doc string to module
    m.doc() = "Pybind11 test.";

    py::class_<PyParameters<double>>(m, "GWParameters")
        .def(py::init<>())
        .def("__repr__", [](const PyParameters<double> &params) { return "Greenstein and Winslow Model Parameters"; })
        .def_readwrite("T", &PyParameters<double>::T, "Temperature")
        .def_readwrite("CSA", &PyParameters<double>::CSA, "Membrane Capacitance")
        .def_readwrite("Vcyto", &PyParameters<double>::Vcyto, "Volume of cytosol")
        .def_readwrite("VNSR", &PyParameters<double>::VNSR, "Volume of NSR")
        .def_readwrite("VJSR", &PyParameters<double>::VJSR, "Volume of JSR")
        .def_readwrite("VSS", &PyParameters<double>::VSS, "Volume dyadic subspace")
        .def_readwrite("NCaRU", &PyParameters<double>::NCaRU, "Number of calcium release units")
        .def_readwrite("Ko", &PyParameters<double>::Ko, "Extracellular potassium concentration")
        .def_readwrite("Nao", &PyParameters<double>::Nao, "Extracellular sodium concentration")
        .def_readwrite("Cao", &PyParameters<double>::Cao, "Extracellular calcium concentration")
        .def_readwrite("Clo", &PyParameters<double>::Clo, "Extracellular chloride concentration")
        .def_readwrite("Clcyto", &PyParameters<double>::Clcyto, "Cytoplasm/intracellular chloride concentration")
        .def_readwrite("f", &PyParameters<double>::f, "State 5 -> state 6 LCC transition rate")
        .def_readwrite("g", &PyParameters<double>::g, "State 6 -> state 5 LCC transition rate")
        .def_readwrite("f1", &PyParameters<double>::f1, "State 11 -> state 12 LCC transition rate")
        .def_readwrite("g1", &PyParameters<double>::g1, "State 12 -> state 11 LCC transition rate")
        .def_readwrite("a", &PyParameters<double>::a, "LCC state dependent rate of calcium induced inactivation parameter")
        .def_readwrite("b", &PyParameters<double>::b, "LCC state dependent rate of activation from calcium inactivated state parameter")
        .def_readwrite("gamma0", &PyParameters<double>::gamma0, "LCC calcium induced inactivation rate parameter")
        .def_readwrite("omega", &PyParameters<double>::omega, "LCC reactivation from calcium induced inactivation rate parameter")
        .def_readwrite("PCaL", &PyParameters<double>::PCaL, "Permeability of the L-type calcium channels")
        .def_readwrite("kfClCh", &PyParameters<double>::kfClCh, "Activation rate of calcium dependent chloride channels")
        .def_readwrite("kbClCh", &PyParameters<double>::kbClCh, "Inactivation rate of calcium dependent chloride channels")
        .def_readwrite("Pto2", &PyParameters<double>::Pto2, "Permeability of calcium dependent chloride channels")
        .def_readwrite("k12", &PyParameters<double>::k12, "RyR state 1 -> state 2 transition rate")
        .def_readwrite("k21", &PyParameters<double>::k21, "RyR state 2 -> state 1 transition rate")
        .def_readwrite("k23", &PyParameters<double>::k23, "RyR state 2 -> state 3 transition rate")
        .def_readwrite("k32", &PyParameters<double>::k32, "RyR state 3 -> state 2 transition rate")
        .def_readwrite("k34", &PyParameters<double>::k34, "RyR state 3 -> state 4 transition rate")
        .def_readwrite("k43", &PyParameters<double>::k43, "RyR state 4 -> state 3 transition rate")
        .def_readwrite("k45", &PyParameters<double>::k45, "RyR state 4 -> state 5 transition rate")
        .def_readwrite("k54", &PyParameters<double>::k54, "RyR state 5 -> state 4 transition rate")
        .def_readwrite("k56", &PyParameters<double>::k56, "RyR state 5 -> state 6 transition rate")
        .def_readwrite("k65", &PyParameters<double>::k65, "RyR state 6 -> state 5 transition rate")
        .def_readwrite("k25", &PyParameters<double>::k25, "RyR state 2 -> state 5 transition rate")
        .def_readwrite("k52", &PyParameters<double>::k52, "RyR state 5 -> state 2 transition rate")
        .def_readwrite("rRyR", &PyParameters<double>::rRyR, "Rate of calcium release from RyRs")
        .def_readwrite("rxfer", &PyParameters<double>::rxfer, "Subspace calcium rate of transfer to cytosol")
        .def_readwrite("rtr", &PyParameters<double>::rtr, "Transfer rate of calcium from the NSR to the JSR")
        .def_readwrite("riss", &PyParameters<double>::riss, "Intersubspace calcium transfer rate")
        .def_readwrite("BSRT", &PyParameters<double>::BSRT, "Total concentration of BSR")
        .def_readwrite("KBSR", &PyParameters<double>::KBSR, "BSR half saturation")
        .def_readwrite("BSLT", &PyParameters<double>::BSLT, "Total concentration of BSL")
        .def_readwrite("KBSL", &PyParameters<double>::KBSL, "BSL half concentration")
        .def_readwrite("CSQNT", &PyParameters<double>::CSQNT, "Total concentration of CSQN")
        .def_readwrite("KCSQN", &PyParameters<double>::KCSQN, "CSQN half saturation")
        .def_readwrite("CMDNT", &PyParameters<double>::CMDNT, "Total CMDN concentration")
        .def_readwrite("KCMDN", &PyParameters<double>::KCMDN, "CMDN half saturation")
        .def_readwrite("GNa", &PyParameters<double>::GNa, "Fast sodium channel conductance")
        .def_readwrite("GKr", &PyParameters<double>::GKr, "Delayed rapidly activating rectifier current conductance")
        .def_readwrite("Kf", &PyParameters<double>::Kf, "Forward rate parameter for HERG ion-channel model")
        .def_readwrite("Kb", &PyParameters<double>::Kb, "Backwards rate parameter for HERG ion-channel model")
        .def_readwrite("GKs", &PyParameters<double>::GKs, "Delayed slowly-activating rectifier current conductance")
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
        .def_readwrite("LCC_inactivation", &PyGWSimulation::LCC_inactivation)
        .def_readwrite("RyR", &PyGWSimulation::RyR)
        .def_readwrite("ClCh", &PyGWSimulation::ClCh)
        .def("__repr__", [](const PyGWSimulation &x) {return "Greenstein and Winslow model solution over the interval [0, " + std::to_string(x.tspan) + "] with " + std::to_string(x.nCRU) + " CRUs"; });
   
    // Needed for setting initial conditions
    py::class_<PyGlobalState>(m, "GWGlobalState")
        .def(py::init<>())
        .def_readwrite("V", &PyGlobalState::V)
        .def_readwrite("Nai", &PyGlobalState::Nai)
        .def_readwrite("Ki", &PyGlobalState::Ki)
        .def_readwrite("Cai", &PyGlobalState::Cai)
        .def_readwrite("CaNSR", &PyGlobalState::CaNSR)
        .def_readwrite("CaLTRPN", &PyGlobalState::CaLTRPN)
        .def_readwrite("CaHTRPN", &PyGlobalState::CaHTRPN)
        .def_readwrite("m", &PyGlobalState::m)
        .def_readwrite("h", &PyGlobalState::h)
        .def_readwrite("j", &PyGlobalState::j)
        .def_readwrite("xKs", &PyGlobalState::xKs)
        .def_readwrite("XKr", &PyGlobalState::XKr)
        .def_readwrite("XKv14", &PyGlobalState::XKv14)
        .def_readwrite("XKv43", &PyGlobalState::XKv43);
    
    py::class_<PyCRUState>(m, "GWCRUState")
        .def(py::init<int>())
        .def_readwrite("CaSS", &PyCRUState::CaSS)
        .def_readwrite("CaJSR", &PyCRUState::CaJSR)
        .def_readwrite("LCC", &PyCRUState::LCC)
        .def_readwrite("LCC_inactivation", &PyCRUState::LCC_inactivation)
        .def_readwrite("RyR", &PyCRUState::RyR)
        .def_readwrite("ClCh", &PyCRUState::ClCh);


    m.def("run", &run_PRNG_arg, "Simulate the model", "parameters"_a, "nCRU"_a, "step_size"_a, "num_steps"_a, 
                                                      "Istim"_a, "record_every"_a, "init_crus"_a, "init_globals"_a = PyGlobalState(), 
                                                      "PRNG"_a = "mt19937_64",
                                                       py::call_guard<py::gil_scoped_release>());
}