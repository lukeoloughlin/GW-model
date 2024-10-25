#pragma once


#include <omp.h>
#include <functional>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <pybind11/pybind11.h>

#include "../common.hpp"
#include "GW_utils.hpp"
#include "SSA.hpp"

template<typename T>
using Array1 = Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>;
template<typename T>
using Array2 = Eigen::Array<T,Eigen::Dynamic,4,Eigen::RowMajor>;

using QKrMap = Eigen::Map<Eigen::Array<double,5,5>,Eigen::RowMajor>;
using QKvMap = Eigen::Map<Eigen::Array<double,10,10>,Eigen::RowMajor>;

using npArray1d = Eigen::RowVectorXd;
using npArray2d = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using npArray2i = Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using npArray3d = Eigen::Tensor<double,3,Eigen::RowMajor>;
using npArray3i = Eigen::Tensor<int,3,Eigen::RowMajor>;
using npArray4i = Eigen::Tensor<int,4,Eigen::RowMajor>;

namespace py = pybind11;
using namespace pybind11::literals;

// To intialise the GW model in the python module
void init_GW(py::module& m);


namespace GW {    
    class GW_model;

    /* This makes it easier to specify the intial state in python. It is necessary because of the annoying issue with Eigen::Tensor 
        screwing things up on the python end. Need to use Eigen::TensorMap on the c++ end, which doesn't work with python, so I need 
        to copy the np array (Eigen::Tensor) into the Eigen::TensorMap. */
    struct PyInitGWState {
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
        
        npArray2d CaSS;
        npArray1d CaJSR;
        npArray2i LCC;
        npArray2i LCC_inactivation;
        npArray3i RyR;
        npArray2i ClCh;

        PyInitGWState(int nCRU) : CaSS(nCRU,4), CaJSR(nCRU), LCC(nCRU,4), LCC_inactivation(nCRU,4), RyR(nCRU,4,6), ClCh(nCRU,4) {
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
    
    class PyGWSimulation {
    public:
        int nCRU;
        double tspan;

        npArray1d t;
        npArray1d V;
        npArray1d m;
        npArray1d h;
        npArray1d j;
        npArray1d Nai;
        npArray1d Ki;
        npArray1d Cai;
        npArray1d CaNSR;
        npArray1d CaLTRPN;
        npArray1d CaHTRPN;
        npArray1d xKs;
        npArray2d XKr;
        npArray2d XKv14;
        npArray2d XKv43;
        npArray2d CaJSR;
        npArray3d CaSS;
        npArray3i LCC;
        npArray3i LCC_inactivation;
        npArray4i RyR;
        npArray3i ClCh;
        
        npArray1d RyR_open_int;
        npArray1d RyR_open_martingale;
        npArray1d RyR_open_martingale_normalised;
        npArray1d sigma_RyR;
        
        npArray1d LCC_open_int;
        npArray1d LCC_open_martingale;
        npArray1d LCC_open_martingale_normalised;
        npArray1d sigma_LCC;

        PyGWSimulation(int nCRU_, int num_step, double t_) : nCRU(nCRU_), tspan(t_), t(num_step), V(num_step), m(num_step), h(num_step), j(num_step), 
                                                     Nai(num_step), Ki(num_step), Cai(num_step), CaNSR(num_step), CaLTRPN(num_step), 
                                                     CaHTRPN(num_step), xKs(num_step), XKr(num_step,5), XKv14(num_step,10), XKv43(num_step,10), 
                                                     CaJSR(num_step,nCRU_), CaSS(num_step,nCRU_,4), LCC(num_step,nCRU_,4), 
                                                     LCC_inactivation(num_step,nCRU_,4), RyR(num_step,nCRU_,4,6), ClCh(num_step,nCRU_,4),
                                                     RyR_open_int(num_step), RyR_open_martingale(num_step), RyR_open_martingale_normalised(num_step),
                                                     sigma_RyR(num_step), LCC_open_int(num_step), LCC_open_martingale(num_step), 
                                                     LCC_open_martingale_normalised(num_step), sigma_LCC(num_step) { }

        void record_state(const GW_model& model, const int idx, const int nCRU, const double t_);
    };


    //template <typename T>    
    class GW_model {
    public:
        Parameters parameters;
        GlobalState globals;
        CRUState CRUs;
        double Istim = 0;

    private:
        int nCRU;
        Constants consts;
        Array2<double> JLCC;
        Array2<double> Jxfer;
        Array1<double> Jtr;

        double QKr_storage[5*5] = {0};
        double QKv14_storage[10*10] = {0};
        double QKv43_storage[10*10] = {0};
        QKrMap QKr;
        QKvMap QKv14;
        QKvMap QKv43;

        Currents currents;
        GlobalState dGlobals;
    
        inline void initialise_Jxfer(){ Jxfer = parameters.rxfer * (CRUs.CaSS - globals.Cai); }
        inline void initialise_Jtr(){ Jtr = parameters.rtr * (globals.CaNSR - CRUs.CaJSR); }
        inline void initialise_QKr(){ QKr(1,2) = parameters.Kf; QKr(2,1) = parameters.Kb; }
        void initialise_JLCC();
        
        void update_QKr();
        void update_QKv();

        void update_V_and_concentration_derivatives(const double dt);
        void update_gate_derivatives(const double dt);
        void update_Kr_derivatives(const double dt);
        void update_Kv_derivatives(const double dt); // Updates both Kv14 and Kv43        

        template <typename PRNG>
        void SSA(const double dt);

        /* Record the values of the CRUStateThread temp back to the CRUState state for CRU i */
        void update_CRUstate_from_temp(const CRUStateThread& temp, const int i);

        void update_increment_and_sigma(const double dt);
        void update_mean_RyR_open();
        void update_martingale_quantities();
        
        
    public:
        GW_model(int nCRU_simulated) : parameters(), globals(), CRUs(nCRU_simulated), nCRU(nCRU_simulated), consts(parameters, nCRU), JLCC(nCRU_simulated,4), 
                                       Jxfer(nCRU_simulated,4), Jtr(nCRU_simulated), QKr(QKr_storage), QKv14(QKv14_storage), QKv43(QKv43_storage), currents(), 
                                       dGlobals(0.0) { 
            consts.VF_RT = globals.V * consts.F_RT;
            consts.JLCC_exp = exp(2*consts.VF_RT);
            initialise_JLCC();
            initialise_Jxfer();
            initialise_Jtr();
            initialise_QKr();
        }

        GW_model(const Parameters& params, int nCRU_simulated) : parameters(params), globals(), CRUs(nCRU_simulated), nCRU(nCRU_simulated), consts(parameters, nCRU), JLCC(nCRU_simulated,4), 
                                       Jxfer(nCRU_simulated,4), Jtr(nCRU_simulated), QKr(QKr_storage), QKv14(QKv14_storage), QKv43(QKv43_storage), currents(), 
                                       dGlobals(0.0) { 
            consts.VF_RT = globals.V * consts.F_RT;
            consts.JLCC_exp = exp(2*consts.VF_RT);
            initialise_JLCC();
            initialise_Jxfer();
            initialise_Jtr();
            initialise_QKr();
        }

        void init_from_python(const PyInitGWState& init_state);
        int get_nCRU() const { return nCRU; }

        template <typename PRNG>
        void euler_step(const double dt);

        template <typename PRNG>
        PyGWSimulation run_sim(const double dt, const int num_steps, const std::function<double(double)>& Is, const int record_every); 

};
    

    template <typename PRNG>
    void GW_model::SSA(const double dt){
        consts.alphaLCC = alphaLCC(globals.V);
        consts.betaLCC = betaLCC(globals.V);
        consts.yinfLCC = yinfLCC(globals.V);
        consts.tauLCC = tauLCC(globals.V);
        consts.JLCC_exp = square(1.0 / consts.expmVF_RT);
        consts.JLCC_multiplier = consts.JLCC_const * globals.V * consts.F_RT / (consts.JLCC_exp - 1);

        #pragma omp parallel
        {
            CRUStateThread temp;
            
            #pragma omp for schedule( static )
            for (int i = 0; i < nCRU; i++){
                temp.copy_from_CRUState(CRUs, JLCC, i, parameters);
                SSA_single_CRU<PRNG>(temp, globals.Cai, globals.CaNSR, dt, parameters, consts);
                update_CRUstate_from_temp(temp, i);
            }
        }
    }
    
    template <typename PRNG>
    void GW_model::euler_step(const double dt){
        consts.VF_RT = globals.V*consts.F_RT;
        consts.expmVF_RT = exp(-consts.VF_RT);

        update_QKr();
        update_QKv(); // Updates Kv14 and Kv43
        
        update_V_and_concentration_derivatives(dt);
        update_gate_derivatives(dt);
        update_Kr_derivatives(dt);
        update_Kv_derivatives(dt); // Updates both Kv14 and Kv43

        SSA<PRNG>(dt);

        globals.V += dGlobals.V;
        globals.Nai += dGlobals.Nai;
        globals.Ki += dGlobals.Ki;
        globals.Cai += dGlobals.Cai;
        globals.CaNSR += dGlobals.CaNSR;
        globals.CaLTRPN += dGlobals.CaLTRPN;
        globals.CaHTRPN += dGlobals.CaHTRPN;
        globals.m += dGlobals.m;
        globals.h += dGlobals.h;
        globals.j += dGlobals.j;
        globals.xKs += dGlobals.xKs;
        globals.Kr[0] += dGlobals.Kr[0];
        globals.Kr[1] += dGlobals.Kr[1];
        globals.Kr[2] += dGlobals.Kr[2];
        globals.Kr[3] += dGlobals.Kr[3];
        globals.Kr[4] += dGlobals.Kr[4];
        for (int j = 0; j < 10; j++){
            globals.Kv14[j] += dGlobals.Kv14[j];
            globals.Kv43[j] += dGlobals.Kv43[j];
        }
    }

    template <typename PRNG>
    PyGWSimulation GW_model::run_sim(const double dt, const int num_steps, const std::function<double(double)>& Is, const int record_every){
        PyGWSimulation out(nCRU, num_steps / record_every, num_steps*dt);
        double t = 0.0;
        int counter = 0;

        for (int i = 0; i < num_steps; ++i){
            Istim = Is(t);
            euler_step<PRNG>(dt);
            t += dt;
            if (i % record_every == 0){
                out.record_state(*this, counter, nCRU, t);
                ++counter;
            }
        }
        return out;
    }


}




