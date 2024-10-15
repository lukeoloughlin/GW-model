
#ifndef GW_LATTICE_UTILS_H
#define GW_LATTICE_UTILS_H

//#include "GW.hpp"
//#include "ndarray.hpp"
#include "GW_utils.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>
#include <random>

template<typename T>
using Array1 = Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>;
template<typename T>
using Array2 = Eigen::Array<T,Eigen::Dynamic,4,Eigen::RowMajor>;
template<typename T>
using Array3 = Eigen::TensorMap<Eigen::Tensor<T,3,Eigen::RowMajor>>;

/*
Bug in pybind11 interface of Eigen tensors that causes hangs in python after calling a function more than once. Using a TensorMap works, so
we use this container class as a work around
*/


namespace GW {
    
    template <typename FloatType>//, typename PRNG>
    struct CRULatticeState {
        Array2<FloatType> CaSS;
        Array2<FloatType> CaJSR; // changing this to Array2 because CRUs are no longer distinct structures
        Array2<int> LCC;
        Array2<int> LCC_inactivation;
        Array3Container<int> RyR;
        Array2<int> ClCh;
        Array2<FloatType> RyR_open_int;
        Array2<FloatType> RyR_open_martingale;
        Array2<FloatType> RyR_open_martingale_normalised;
        //Array1<FloatType> sigma_RyR;
        
        //Array2<FloatType> LCC_open_int;
        //Array2<FloatType> LCC_open_martingale;
        //Array2<FloatType> LCC_open_martingale_normalised;
        //Array1<FloatType> sigma_LCC;

        CRULatticeState(const int nCRU_x, const int nCRU_y);
        CRULatticeState& operator=(CRULatticeState& x) = default;
    };

}


#include "GW_lattice_utils.tpp"

#endif