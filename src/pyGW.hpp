#pragma once

#include <functional>

#include <eigen3/Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <pybind11/pybind11.h>

#include "includes/GW.hpp"


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

    PyGWSimulation(int nCRU_, int num_step, double t_) : nCRU(nCRU_), tspan(t_), t(num_step), V(num_step), m(num_step), h(num_step), j(num_step), 
                                                     Nai(num_step), Ki(num_step), Cai(num_step), CaNSR(num_step), CaLTRPN(num_step), 
                                                     CaHTRPN(num_step), xKs(num_step), XKr(num_step,5), XKv14(num_step,10), XKv43(num_step,10), 
                                                     CaJSR(num_step,nCRU_), CaSS(num_step,nCRU_,4), LCC(num_step,nCRU_,4), 
                                                     LCC_inactivation(num_step,nCRU_,4), RyR(num_step,nCRU_,4,6), ClCh(num_step,nCRU_,4) { }
};

struct PyGWMartingaleSimulation {
    int nCRU;
    double tspan;

    Array1d t;
    Array1d V;
    Array1d intQTXt;
    Array1d dM;
    Array1d dM_normalised;
    Array1d sigma2_t;
    Array1d RyR_open;
    //Array1d CaSS_mean;
    Array1d dCaSS_mean;

    PyGWMartingaleSimulation(int nCRU_, int num_step, double t_) : nCRU(nCRU_), tspan(t_), t(num_step), V(num_step), intQTXt(num_step), 
                                                                   dM(num_step), dM_normalised(num_step), sigma2_t(num_step), RyR_open(num_step), dCaSS_mean(num_step) { }
};


template <typename PRNG>
void record_state(PyGWSimulation& out, const GW::GW_model<double, PRNG>& model, const int idx, const int nCRU, const double t){
        

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

    //out.int_QTXt(idx) = model.int_QTXt;
}


template <typename PRNG>
PyGWSimulation run(const GW::Parameters<double>& params, const int nCRU, double step_size, int num_steps, 
                   const std::function<double(double)>& Is, int record_every, GW::GlobalState<double>& init_globals, 
                   GW::CRUState<double>& init_crus){
    GW::GW_model<double, PRNG> model(params, nCRU);
    model.set_initial_value(init_globals, init_crus);

    PyGWSimulation out(nCRU, num_steps / record_every, num_steps*step_size);
    double t = 0.0;
    int counter = 0;

    for (int i = 0; i < num_steps; ++i){
        model.Istim = Is(t);
        model.euler_step(step_size);
        t += step_size;
        if (i % record_every == 0){
            record_state(out, model, counter, nCRU, t);
            ++counter;
        }
    }
    return out;
}

template <typename PRNG>
PyGWMartingaleSimulation run_martingale(const GW::Parameters<double>& params, const int nCRU, double step_size, int num_steps, 
                   const std::function<double(double)>& Is, int record_every, GW::GlobalState<double>& init_globals, 
                   GW::CRUState<double>& init_crus){
    GW::GW_model<double, PRNG> model(params, nCRU);
    model.set_initial_value(init_globals, init_crus);

    PyGWMartingaleSimulation out(nCRU, num_steps / record_every, num_steps*step_size);
    double t = 0.0;
    int counter = 0;

    for (int i = 0; i < num_steps; ++i){
        model.Istim = Is(t);
        model.euler_step_martingale(step_size);
        t += step_size;
        if (i % record_every == 0){
            out.t[counter] = t;
            out.V[counter] = model.globals.V;
            out.intQTXt[counter] = model.int_QTXt;
            out.dM[counter] = model.dM;
            out.dM_normalised[counter] = model.dM_normalised;
            out.sigma2_t[counter] = model.sigma2_t;
            out.RyR_open[counter] = model.mean_RyR_open;
            out.dCaSS_mean[counter] = model.dCaSS_mean;
            ++counter;
        }
    }
    return out;
}