#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "includes/common.hpp"
#include "includes/GW_utils.hpp"
#include "includes/GW.hpp"

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
        .def_readwrite("GNa", &GW::Parameters<double>::GNa, "Conductance of sodium channels");

    // Note that for overloaded methods, the exposed versions must be cast to function pointers otherwise there will be a compilation error. See docs.
    // For C++14 and above, there is an alternative to function pointers, that is to use py::overload_cast<T>(&Object::method), where T is (are) the types of
    // the input arguments and Object is the class name, method is the overloaded method. Addiditionally, we should use
    // py::overload_cast<T>(&Object::method, py::const_) if the difference between the overloads is constness.


    // Need to figure out how to deal with the below. Seems like I'll have to write an overload that takes an std::function instead of an rvalue lambda because
    // otherwise compiler shits the bed 
    py::class_<GW::GW_model<double>>(m, "GWModel")
        .def(py::init<int>())
        .def("__repr__", [](const GW::GW_model<double> &model) {return "Greenstein and Winslow model simulator with " + std::to_string(model.get_nCRU()) + " CRUs."; });
        //.def("simulate_and_write", &GW::GW_model<double>::euler_write<std::function<double(double)>>, "Simulate and write to file", "step_size"_a, "num_steps"_a, "Istim"_a, "output_file"_a, "record_every"_a);

}