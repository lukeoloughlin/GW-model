#include <pybind11/pybind11.h>
#include "includes/GW/GW.hpp"
#include "includes/lattice/GW_lattice.hpp"

namespace py = pybind11;



PYBIND11_MODULE(Models, m){
    m.doc() = "Some models of cardiac AP and calcium release";

    init_GW(m);
    init_GWLattice(m);

}