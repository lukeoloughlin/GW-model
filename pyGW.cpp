#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include "includes/common.hpp"
#include "includes/GW_utils.hpp"
#include "includes/GW.hpp"

#include "pyGW.hpp"

namespace py = pybind11;
using namespace pybind11::literals;



PYBIND11_MODULE(GreensteinWinslow, m) {
    /**************** 
     Some basic functions
    *****************/

    // Add a doc string to module
    m.doc() = "Pybind11 test. Only includes the Beeler Reuter term for INa.";

    // Add a function to m. py::arg() names the arguments if that wasnt clear
    m.def("INa", &common::INa<double>, "Beeler Reuter Sodium current.",
            py::arg("V"), py::arg("m"), py::arg("h"), py::arg("j"), py::arg("GNa"), py::arg("ENa"));
    
    // Same as above but uses the literal _a from pybind11::literals which does the same thing as py::arg(). 
    // Also we are setting defaults
    m.def("nernst", &common::Nernst<double>, "Calculate the Nernst potential using the Nernst equation.",
          "Conc_in"_a, "Conc_ex"_a, "RT_F"_a=GAS_CONST*310/FARADAY, "valence"_a=1.0);
    
    m.def("nernst_v", py::vectorize(common::Nernst<double>), "Vectorized Nernst");

    m.def("urand", &urand<double>, "Uniform random number generator");

    /**************** 
     Some basic structs/classes
    *****************/
   // A basic example of exporting a struct/class. 
   // Here py::init<> is a templated function whose template parameters take the constructor parameters as template arguments. E.g. py::init<const int>
   // The second line adds a method, in this case __repr__ (prints the class in Python), and we assign it a lambda since we didn't define a print method in the struct.
   // The last line shows how to add a read/write field. In this case, we expose the parameter GNa.
   py::class_<GW::Parameters<double>>(m, "Parameters")
        .def(py::init<>())
        .def("__repr__", [](const GW::Parameters<double> &params) { return "<Greenstein and Winslow Model Parameters>"; })
        .def_readwrite("T", &GW::Parameters<double>::T, "Temperature")
        .def_readwrite("CSA", &GW::Parameters<double>::CSA, "Membrane Capacitance")
        .def_readwrite("Vcyto", &GW::Parameters<double>::Vcyto, "Volume of cytosol")
        .def_readwrite("VNSR", &GW::Parameters<double>::VNSR, "Volume of cytosol")
        .def_readwrite("VJSR", &GW::Parameters<double>::VJSR, "Volume of cytosol")
        .def_readwrite("VSS", &GW::Parameters<double>::VSS, "Volume of cytosol")
        .def_readwrite("NCaRU", &GW::Parameters<double>::NCaRU, "Volume of cytosol")
        .def_readwrite("Ko", &GW::Parameters<double>::Ko, "Volume of cytosol")
        .def_readwrite("Nao", &GW::Parameters<double>::Nao, "Volume of cytosol")
        .def_readwrite("Cao", &GW::Parameters<double>::Cao, "Volume of cytosol")
        .def_readwrite("Clo", &GW::Parameters<double>::Clo, "Volume of cytosol")
        .def_readwrite("Clcyto", &GW::Parameters<double>::Clcyto, "Volume of cytosol")
        .def_readwrite("f", &GW::Parameters<double>::f, "Volume of cytosol")
        .def_readwrite("g", &GW::Parameters<double>::g, "Volume of cytosol")
        .def_readwrite("f1", &GW::Parameters<double>::f1, "Volume of cytosol")
        .def_readwrite("g1", &GW::Parameters<double>::g1, "Volume of cytosol")
        .def_readwrite("a", &GW::Parameters<double>::a, "Volume of cytosol")
        .def_readwrite("b", &GW::Parameters<double>::b, "Volume of cytosol")
        .def_readwrite("gamma0", &GW::Parameters<double>::gamma0, "Volume of cytosol")
        .def_readwrite("omega", &GW::Parameters<double>::omega, "Volume of cytosol")
        .def_readwrite("PCaL", &GW::Parameters<double>::PCaL, "Volume of cytosol")
        .def_readwrite("kfClCh", &GW::Parameters<double>::kfClCh, "Volume of cytosol")
        .def_readwrite("kbClCh", &GW::Parameters<double>::kbClCh, "Volume of cytosol")
        .def_readwrite("Pto2", &GW::Parameters<double>::Pto2, "Volume of cytosol")
        .def_readwrite("k12", &GW::Parameters<double>::k12, "Volume of cytosol")
        .def_readwrite("k21", &GW::Parameters<double>::k21, "Volume of cytosol")
        .def_readwrite("k23", &GW::Parameters<double>::k23, "Volume of cytosol")
        .def_readwrite("k32", &GW::Parameters<double>::k32, "Volume of cytosol")
        .def_readwrite("k34", &GW::Parameters<double>::k34, "Volume of cytosol")
        .def_readwrite("k43", &GW::Parameters<double>::k43, "Volume of cytosol")
        .def_readwrite("k45", &GW::Parameters<double>::k45, "Volume of cytosol")
        .def_readwrite("k54", &GW::Parameters<double>::k54, "Volume of cytosol")
        .def_readwrite("k56", &GW::Parameters<double>::k56, "Volume of cytosol")
        .def_readwrite("k65", &GW::Parameters<double>::k65, "Volume of cytosol")
        .def_readwrite("k25", &GW::Parameters<double>::k25, "Volume of cytosol")
        .def_readwrite("k52", &GW::Parameters<double>::k52, "Volume of cytosol")
        .def_readwrite("rRyR", &GW::Parameters<double>::rRyR, "Volume of cytosol")
        .def_readwrite("rxfer", &GW::Parameters<double>::rxfer, "Volume of cytosol")
        .def_readwrite("rtr", &GW::Parameters<double>::rtr, "Volume of cytosol")
        .def_readwrite("riss", &GW::Parameters<double>::riss, "Volume of cytosol")
        .def_readwrite("BSRT", &GW::Parameters<double>::BSRT, "Volume of cytosol")
        .def_readwrite("KBSR", &GW::Parameters<double>::KBSR, "Volume of cytosol")
        .def_readwrite("BSLT", &GW::Parameters<double>::BSLT, "Volume of cytosol")
        .def_readwrite("KBSL", &GW::Parameters<double>::KBSL, "Volume of cytosol")
        .def_readwrite("CSQNT", &GW::Parameters<double>::CSQNT, "Volume of cytosol")
        .def_readwrite("KCSQN", &GW::Parameters<double>::KCSQN, "Volume of cytosol")
        .def_readwrite("CMDNT", &GW::Parameters<double>::CMDNT, "Volume of cytosol")
        .def_readwrite("KCMDN", &GW::Parameters<double>::KCMDN, "Volume of cytosol")
        .def_readwrite("GNa", &GW::Parameters<double>::GNa, "Conductance of sodium channels")
        .def_readwrite("GKr", &GW::Parameters<double>::GKr, "Conductance of sodium channels")
        .def_readwrite("Kf", &GW::Parameters<double>::Kf, "Conductance of sodium channels")
        .def_readwrite("Kb", &GW::Parameters<double>::Kb, "Conductance of sodium channels")
        .def_readwrite("GKs", &GW::Parameters<double>::GKs, "Conductance of sodium channels")
        .def_readwrite("GKv43", &GW::Parameters<double>::GKv43, "Conductance of sodium channels")
        .def_readwrite("alphaa0Kv43", &GW::Parameters<double>::alphaa0Kv43, "Conductance of sodium channels")
        .def_readwrite("aaKv43", &GW::Parameters<double>::aaKv43, "Conductance of sodium channels")
        .def_readwrite("betaa0Kv43", &GW::Parameters<double>::betaa0Kv43, "Conductance of sodium channels")
        .def_readwrite("baKv43", &GW::Parameters<double>::baKv43, "Conductance of sodium channels")
        .def_readwrite("alphai0Kv43", &GW::Parameters<double>::alphai0Kv43, "Conductance of sodium channels")
        .def_readwrite("aiKv43", &GW::Parameters<double>::aiKv43, "Conductance of sodium channels")
        .def_readwrite("betai0Kv43", &GW::Parameters<double>::betai0Kv43, "Conductance of sodium channels")
        .def_readwrite("biKv43", &GW::Parameters<double>::biKv43, "Conductance of sodium channels")
        .def_readwrite("f1Kv43", &GW::Parameters<double>::f1Kv43, "Conductance of sodium channels")
        .def_readwrite("f2Kv43", &GW::Parameters<double>::f2Kv43, "Conductance of sodium channels")
        .def_readwrite("f3Kv43", &GW::Parameters<double>::f3Kv43, "Conductance of sodium channels")
        .def_readwrite("f4Kv43", &GW::Parameters<double>::f4Kv43, "Conductance of sodium channels")
        .def_readwrite("b1Kv43", &GW::Parameters<double>::b1Kv43, "Conductance of sodium channels")
        .def_readwrite("b2Kv43", &GW::Parameters<double>::b2Kv43, "Conductance of sodium channels")
        .def_readwrite("b3Kv43", &GW::Parameters<double>::b3Kv43, "Conductance of sodium channels")
        .def_readwrite("b4Kv43", &GW::Parameters<double>::b4Kv43, "Conductance of sodium channels")
        .def_readwrite("PKv14", &GW::Parameters<double>::PKv14, "Conductance of sodium channels")
        .def_readwrite("alphaa0Kv14", &GW::Parameters<double>::alphaa0Kv14, "Conductance of sodium channels")
        .def_readwrite("aaKv14", &GW::Parameters<double>::aaKv14, "Conductance of sodium channels")
        .def_readwrite("betaa0Kv14", &GW::Parameters<double>::betaa0Kv14, "Conductance of sodium channels")
        .def_readwrite("baKv14", &GW::Parameters<double>::baKv14, "Conductance of sodium channels")
        .def_readwrite("alphai0Kv14", &GW::Parameters<double>::alphai0Kv14, "Conductance of sodium channels")
        .def_readwrite("aiKv14", &GW::Parameters<double>::aiKv14, "Conductance of sodium channels")
        .def_readwrite("betai0Kv14", &GW::Parameters<double>::betai0Kv14, "Conductance of sodium channels")
        .def_readwrite("biKv14", &GW::Parameters<double>::biKv14, "Conductance of sodium channels")
        .def_readwrite("f1Kv14", &GW::Parameters<double>::f1Kv14, "Conductance of sodium channels")
        .def_readwrite("f2Kv14", &GW::Parameters<double>::f2Kv14, "Conductance of sodium channels")
        .def_readwrite("f3Kv14", &GW::Parameters<double>::f3Kv14, "Conductance of sodium channels")
        .def_readwrite("f4Kv14", &GW::Parameters<double>::f4Kv14, "Conductance of sodium channels")
        .def_readwrite("b1Kv14", &GW::Parameters<double>::b1Kv14, "Conductance of sodium channels")
        .def_readwrite("b2Kv14", &GW::Parameters<double>::b2Kv14, "Conductance of sodium channels")
        .def_readwrite("b3Kv14", &GW::Parameters<double>::b3Kv14, "Conductance of sodium channels")
        .def_readwrite("b4Kv14", &GW::Parameters<double>::b4Kv14, "Conductance of sodium channels")
        .def_readwrite("Csc", &GW::Parameters<double>::Csc, "Conductance of sodium channels")
        .def_readwrite("GK1", &GW::Parameters<double>::GK1, "Conductance of sodium channels")
        .def_readwrite("KmK1", &GW::Parameters<double>::KmK1, "Conductance of sodium channels")
        .def_readwrite("GKp", &GW::Parameters<double>::GKp, "Conductance of sodium channels")
        .def_readwrite("kNaCa", &GW::Parameters<double>::kNaCa, "Conductance of sodium channels")
        .def_readwrite("KmNa", &GW::Parameters<double>::KmNa, "Conductance of sodium channels")
        .def_readwrite("KmCa", &GW::Parameters<double>::KmCa, "Conductance of sodium channels")
        .def_readwrite("ksat", &GW::Parameters<double>::ksat, "Conductance of sodium channels")
        .def_readwrite("eta", &GW::Parameters<double>::eta, "Conductance of sodium channels")
        .def_readwrite("INaKmax", &GW::Parameters<double>::INaKmax, "Conductance of sodium channels")
        .def_readwrite("KmNai", &GW::Parameters<double>::KmNai, "Conductance of sodium channels")
        .def_readwrite("KmKo", &GW::Parameters<double>::KmKo, "Conductance of sodium channels")
        .def_readwrite("IpCamax", &GW::Parameters<double>::IpCamax, "Conductance of sodium channels")
        .def_readwrite("KmpCa", &GW::Parameters<double>::KmpCa, "Conductance of sodium channels")
        .def_readwrite("GCab", &GW::Parameters<double>::GCab, "Conductance of sodium channels")
        .def_readwrite("GNab", &GW::Parameters<double>::GNab, "Conductance of sodium channels")
        .def_readwrite("kHTRPNp", &GW::Parameters<double>::kHTRPNp, "Conductance of sodium channels")
        .def_readwrite("kHTRPNm", &GW::Parameters<double>::kHTRPNm, "Conductance of sodium channels")
        .def_readwrite("kLTRPNp", &GW::Parameters<double>::kLTRPNp, "Conductance of sodium channels")
        .def_readwrite("kLTRPNm", &GW::Parameters<double>::kLTRPNm, "Conductance of sodium channels")
        .def_readwrite("HTRPNtot", &GW::Parameters<double>::HTRPNtot, "Conductance of sodium channels")
        .def_readwrite("LTRPNtot", &GW::Parameters<double>::LTRPNtot, "Conductance of sodium channels")
        .def_readwrite("Vmaxf", &GW::Parameters<double>::Vmaxf, "Conductance of sodium channels")
        .def_readwrite("Vmaxr", &GW::Parameters<double>::Vmaxr, "Conductance of sodium channels")
        .def_readwrite("Kmf", &GW::Parameters<double>::Kmf, "Conductance of sodium channels")
        .def_readwrite("Kmr", &GW::Parameters<double>::Kmr, "Conductance of sodium channels")
        .def_readwrite("Hf", &GW::Parameters<double>::Hf, "Conductance of sodium channels")
        .def_readwrite("Hr", &GW::Parameters<double>::Hr, "Conductance of sodium channels");

    // Note that for overloaded methods, the exposed versions must be cast to function pointers otherwise there will be a compilation error. See docs.
    // For C++14 and above, there is an alternative to function pointers, that is to use py::overload_cast<T>(&Object::method), where T is (are) the types of
    // the input arguments and Object is the class name, method is the overloaded method. Addiditionally, we should use
    // py::overload_cast<T>(&Object::method, py::const_) if the difference between the overloads is constness.


    // Need to figure out how to deal with the below. Seems like I'll have to write an overload that takes an std::function instead of an rvalue lambda because
    // otherwise compiler shits the bed 
    py::class_<PyGWOutput>(m, "GWModel")
        .def(py::init<int,int,double>())
        .def_readwrite("V", &PyGWOutput::V)
        .def_readwrite("m", &PyGWOutput::m)
        .def_readwrite("h", &PyGWOutput::h)
        .def_readwrite("j", &PyGWOutput::j)
        .def_readwrite("Nai", &PyGWOutput::Nai)
        .def_readwrite("Ki", &PyGWOutput::Ki)
        .def_readwrite("Cai", &PyGWOutput::Cai)
        .def_readwrite("CaNSR", &PyGWOutput::CaNSR)
        .def_readwrite("CaLTRPN", &PyGWOutput::CaLTRPN)
        .def_readwrite("CaHTRPN", &PyGWOutput::CaHTRPN)
        .def_readwrite("xKs", &PyGWOutput::xKs)
        .def_readwrite("XKr", &PyGWOutput::XKr)
        .def_readwrite("XKv14", &PyGWOutput::XKv14)
        .def_readwrite("XKv43", &PyGWOutput::XKv43)
        .def_readwrite("CaJSR", &PyGWOutput::CaJSR)
        .def_readwrite("CaSS", &PyGWOutput::CaSS)
        .def_readwrite("LCC", &PyGWOutput::LCC)
        .def_readwrite("LCC_activation", &PyGWOutput::LCC_activation)
        .def_readwrite("RyR", &PyGWOutput::RyR)
        .def_readwrite("ClCh", &PyGWOutput::ClCh)
        .def("__repr__", [](const PyGWOutput &x) {return "Greenstein and Winslow model solution over the interval\n" 
                                                        + '[' + '0' + ',' + ' ' + std::to_string(x.t) + "] with "
                                                        + std::to_string(x.nCRU) + " CRUs."; });
        //.def("simulate_and_write", &GW::GW_model<double>::euler_write<std::function<double(double)>>, "Simulate and write to file", "step_size"_a, "num_steps"_a, "Istim"_a, "output_file"_a, "record_every"_a);

}