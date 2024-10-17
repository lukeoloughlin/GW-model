#pragma once

#include <functional>

#include <eigen3/Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <pybind11/pybind11.h>

#include "includes/GW.hpp"
#include "includes/GW_lattice.hpp"


namespace py = pybind11;

typedef Eigen::RowVectorXd Array1d;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Array2d;
typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Array2i;
typedef Eigen::Tensor<double,3,Eigen::RowMajor> Array3d;
typedef Eigen::Tensor<int,3,Eigen::RowMajor> Array3i;
typedef Eigen::Tensor<int,4,Eigen::RowMajor> Array4i;


struct PyGWSimulation {
    int nCRU;
    double tspan;

    Array1d t;
    Array1d V;
    Array1d m;
    Array1d h;
    Array1d j;
    Array1d Nai;
    Array1d Ki;
    Array1d Cai;
    Array1d CaNSR;
    Array1d CaLTRPN;
    Array1d CaHTRPN;
    Array1d xKs;
    Array2d XKr;
    Array2d XKv14;
    Array2d XKv43;
    Array2d CaJSR;
    Array3d CaSS;
    Array3i LCC;
    Array3i LCC_inactivation;
    Array4i RyR;
    Array3i ClCh;
    
    Array2d RyR_open_int;
    Array2d RyR_open_martingale;
    Array2d RyR_open_martingale_normalised;
    Array1d sigma_RyR;
    
    Array2d LCC_open_int;
    Array2d LCC_open_martingale;
    Array2d LCC_open_martingale_normalised;
    Array1d sigma_LCC;

    PyGWSimulation(int nCRU_, int num_step, double t_) : nCRU(nCRU_), tspan(t_), t(num_step), V(num_step), m(num_step), h(num_step), j(num_step), 
                                                     Nai(num_step), Ki(num_step), Cai(num_step), CaNSR(num_step), CaLTRPN(num_step), 
                                                     CaHTRPN(num_step), xKs(num_step), XKr(num_step,5), XKv14(num_step,10), XKv43(num_step,10), 
                                                     CaJSR(num_step,nCRU_), CaSS(num_step,nCRU_,4), LCC(num_step,nCRU_,4), 
                                                     LCC_inactivation(num_step,nCRU_,4), RyR(num_step,nCRU_,4,6), ClCh(num_step,nCRU_,4),
                                                     RyR_open_int(num_step,2), RyR_open_martingale(num_step,2), RyR_open_martingale_normalised(num_step,2),
                                                     sigma_RyR(num_step), LCC_open_int(num_step,2), LCC_open_martingale(num_step,2), 
                                                     LCC_open_martingale_normalised(num_step,2), sigma_LCC(num_step) { }
};

struct PyGWLatticeSimulation {
    int nCRU_x;
    int nCRU_y;
    double tspan;

    Array1d t;
    Array1d V;
    Array1d m;
    Array1d h;
    Array1d j;
    Array1d Nai;
    Array1d Ki;
    Array1d Cai;
    Array1d CaNSR;
    Array1d CaLTRPN;
    Array1d CaHTRPN;
    Array1d xKs;
    Array2d XKr;
    Array2d XKv14;
    Array2d XKv43;
    Array3d CaJSR;
    Array3d CaSS;
    Array3i LCC;
    Array3i LCC_inactivation;
    Array4i RyR;
    Array3i ClCh;
    
    PyGWLatticeSimulation(int nCRU_x_, int nCRU_y_, int num_step, double t_) : nCRU_x(nCRU_x_), nCRU_y(nCRU_y_), tspan(t_), t(num_step), V(num_step), 
                                                                        m(num_step), h(num_step), j(num_step), Nai(num_step), Ki(num_step), 
                                                                        Cai(num_step), CaNSR(num_step), CaLTRPN(num_step), CaHTRPN(num_step), 
                                                                        xKs(num_step), XKr(num_step,5), XKv14(num_step,10), XKv43(num_step,10), 
                                                                        CaJSR(num_step,nCRU_x, nCRU_y), CaSS(num_step,nCRU_x,nCRU_y), LCC(num_step,nCRU_x,nCRU_y), 
                                                                        LCC_inactivation(num_step,nCRU_x,nCRU_y), RyR(num_step,nCRU_x,nCRU_y,6), 
                                                                        ClCh(num_step,nCRU_x,nCRU_y) { }
};

void record_state(PyGWSimulation& out, const GW::GW_model<double>& model, const int idx, const int nCRU, const double t){
    out.t(idx) = t;
    out.V(idx) = model.globals.V;
    out.m(idx) = model.globals.m;
    out.h(idx) = model.globals.h;
    out.j(idx) = model.globals.j;
    out.Nai(idx) = model.globals.Nai;
    out.Ki(idx) = model.globals.Ki;
    out.Cai(idx) = model.globals.Cai;
    out.CaNSR(idx) = model.globals.CaNSR;
    out.CaLTRPN(idx) = model.globals.CaLTRPN;
    out.CaHTRPN(idx) = model.globals.CaHTRPN;
    out.xKs(idx) = model.globals.xKs;
    
    out.XKr(idx,0) = model.globals.Kr[0];
    out.XKr(idx,1) = model.globals.Kr[1];
    out.XKr(idx,2) = model.globals.Kr[2];
    out.XKr(idx,3) = model.globals.Kr[3];
    out.XKr(idx,4) = model.globals.Kr[4];

    for (int j = 0; j < 10; ++j){
        out.XKv14(idx,j) = model.globals.Kv14[j];
        out.XKv43(idx,j) = model.globals.Kv43[j];
    }

    for (int j = 0; j < nCRU; ++j){
        out.CaJSR(idx,j) = model.CRUs.CaJSR(j);
        for (int k = 0; k < 4; ++k){
            out.CaSS(idx,j,k) = model.CRUs.CaSS(j,k);
            out.LCC(idx,j,k) = model.CRUs.LCC(j,k);
            out.LCC_inactivation(idx,j,k) = model.CRUs.LCC_inactivation(j,k);
            out.RyR(idx,j,k,0) = model.CRUs.RyR.array(j,k,0);
            out.RyR(idx,j,k,1) = model.CRUs.RyR.array(j,k,1);
            out.RyR(idx,j,k,2) = model.CRUs.RyR.array(j,k,2);
            out.RyR(idx,j,k,3) = model.CRUs.RyR.array(j,k,3);
            out.RyR(idx,j,k,4) = model.CRUs.RyR.array(j,k,4);
            out.RyR(idx,j,k,5) = model.CRUs.RyR.array(j,k,5);
            out.ClCh(idx,j,k) = model.CRUs.ClCh(j,k);
        }
    }

    out.RyR_open_int(idx,0) = model.CRUs.RyR_open_int.sum() / model.get_nCRU();
    out.RyR_open_int(idx,1) = model.CRUs.RyR_open_int(0,0) + model.CRUs.RyR_open_int(0,1) + 
                              model.CRUs.RyR_open_int(0,2) + model.CRUs.RyR_open_int(0,3);
    
    out.RyR_open_martingale(idx,0) = model.CRUs.RyR_open_martingale.sum() / model.get_nCRU();
    out.RyR_open_martingale(idx,1) = model.CRUs.RyR_open_martingale(0,0) + model.CRUs.RyR_open_martingale(0,1) +
                                     model.CRUs.RyR_open_martingale(0,2) + model.CRUs.RyR_open_martingale(0,3);
    
    out.RyR_open_martingale_normalised(idx,0) = model.CRUs.RyR_open_martingale_normalised.sum() / model.get_nCRU();
    out.RyR_open_martingale_normalised(idx,1) = model.CRUs.RyR_open_martingale_normalised(0,0) + model.CRUs.RyR_open_martingale_normalised(0,1) 
                                                + model.CRUs.RyR_open_martingale_normalised(0,2) + model.CRUs.RyR_open_martingale_normalised(0,3);
    
    out.sigma_RyR(idx) = model.CRUs.sigma_RyR.mean();
    
    out.LCC_open_int(idx,0) = model.CRUs.LCC_open_int.sum() / model.get_nCRU();
    out.LCC_open_int(idx,1) = model.CRUs.LCC_open_int(0,0) + model.CRUs.LCC_open_int(0,1) + 
                              model.CRUs.LCC_open_int(0,2) + model.CRUs.LCC_open_int(0,3);
    
    out.LCC_open_martingale(idx,0) = model.CRUs.LCC_open_martingale.sum() / model.get_nCRU();
    out.LCC_open_martingale(idx,1) = model.CRUs.LCC_open_martingale(0,0) + model.CRUs.LCC_open_martingale(0,1) +
                                     model.CRUs.LCC_open_martingale(0,2) + model.CRUs.LCC_open_martingale(0,3);
    
    out.LCC_open_martingale_normalised(idx,0) = model.CRUs.LCC_open_martingale_normalised.sum() / model.get_nCRU();
    out.LCC_open_martingale_normalised(idx,1) = model.CRUs.LCC_open_martingale_normalised(0,0) + model.CRUs.LCC_open_martingale_normalised(0,1) 
                                                + model.CRUs.LCC_open_martingale_normalised(0,2) + model.CRUs.LCC_open_martingale_normalised(0,3);
    
    out.sigma_LCC(idx) = model.CRUs.sigma_LCC.mean();
}

void record_state(PyGWLatticeSimulation& out, const GW_lattice::GW_lattice<double>& model, const int idx, const int nCRU_x, const int nCRU_y, const double t){
    out.t(idx) = t;
    out.V(idx) = model.globals.V;
    out.m(idx) = model.globals.m;
    out.h(idx) = model.globals.h;
    out.j(idx) = model.globals.j;
    out.Nai(idx) = model.globals.Nai;
    out.Ki(idx) = model.globals.Ki;
    out.Cai(idx) = model.globals.Cai;
    out.CaNSR(idx) = model.globals.CaNSR;
    out.CaLTRPN(idx) = model.globals.CaLTRPN;
    out.CaHTRPN(idx) = model.globals.CaHTRPN;
    out.xKs(idx) = model.globals.xKs;
    
    out.XKr(idx,0) = model.globals.Kr[0];
    out.XKr(idx,1) = model.globals.Kr[1];
    out.XKr(idx,2) = model.globals.Kr[2];
    out.XKr(idx,3) = model.globals.Kr[3];
    out.XKr(idx,4) = model.globals.Kr[4];

    for (int j = 0; j < 10; ++j){
        out.XKv14(idx,j) = model.globals.Kv14[j];
        out.XKv43(idx,j) = model.globals.Kv43[j];
    }

    for (int j = 0; j < nCRU_x; ++j){
        for (int k = 0; k < nCRU_y; ++k){
            out.CaJSR(idx,j,k) = model.CRU_lattice.CaJSR(j,k);
            out.CaSS(idx,j,k) = model.CRU_lattice.CaSS(j,k);
            out.LCC(idx,j,k) = model.CRU_lattice.LCC(j,k);
            out.LCC_inactivation(idx,j,k) = model.CRU_lattice.LCC_inactivation(j,k);
            out.RyR(idx,j,k,0) = model.CRU_lattice.RyR.array(j,k,0);
            out.RyR(idx,j,k,1) = model.CRU_lattice.RyR.array(j,k,1);
            out.RyR(idx,j,k,2) = model.CRU_lattice.RyR.array(j,k,2);
            out.RyR(idx,j,k,3) = model.CRU_lattice.RyR.array(j,k,3);
            out.RyR(idx,j,k,4) = model.CRU_lattice.RyR.array(j,k,4);
            out.RyR(idx,j,k,5) = model.CRU_lattice.RyR.array(j,k,5);
            out.ClCh(idx,j,k) = model.CRU_lattice.ClCh(j,k);
        }
    }
}


template <typename PRNG>
PyGWSimulation run(const GW::Parameters<double>& params, const int nCRU, double step_size, int num_steps, 
                   const std::function<double(double)>& Is, int record_every, GW::GlobalState<double>& init_globals, 
                   GW::CRUState<double>& init_crus){
    GW::GW_model<double> model(params, nCRU);
    model.set_initial_value(init_globals, init_crus);

    PyGWSimulation out(nCRU, num_steps / record_every, num_steps*step_size);
    double t = 0.0;
    int counter = 0;

    for (int i = 0; i < num_steps; ++i){
        model.Istim = Is(t);
        model.euler_step<PRNG>(step_size);
        t += step_size;
        if (i % record_every == 0){
            record_state(out, model, counter, nCRU, t);
            ++counter;
        }
    }
    return out;
}

template <typename PRNG>
PyGWLatticeSimulation run(const GW_lattice::Parameters<double>& params, const int nCRU_x, const int nCRU_y, double step_size, int num_steps, 
                   const std::function<double(double)>& Is, const int record_every, GW::GlobalState<double>& init_globals, 
                   GW_lattice::CRULatticeState<double>& init_crus){
    GW_lattice::GW_lattice<double> model(params, nCRU_x, nCRU_y);
    model.set_initial_value(init_globals, init_crus);

    PyGWLatticeSimulation out(nCRU_x, nCRU_y, num_steps / record_every, num_steps*step_size);
    double t = 0.0;
    int counter = 0;

    for (int i = 0; i < num_steps; ++i){
        model.Istim = Is(t);
        model.euler_step<PRNG>(step_size);
        t += step_size;
        if (i % record_every == 0){
            record_state(out, model, counter, nCRU_x, nCRU_y, t);
            ++counter;
        }
    }
    return out;
}