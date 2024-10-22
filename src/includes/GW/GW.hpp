#pragma once


#include <omp.h>
#include "common.hpp"
#include "GW_utils.hpp"
#include "SSA.hpp"
#include <Eigen/Core>

template<typename T>
using Array1 = Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>;
template<typename T>
using Array2 = Eigen::Array<T,Eigen::Dynamic,4,Eigen::RowMajor>;

using QKrMap = Eigen::Map<Eigen::Array<double,5,5>,Eigen::RowMajor>;
using QKvMap = Eigen::Map<Eigen::Array<double,10,10>,Eigen::RowMajor>;


namespace GW {


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
        inline void initialise_JLCC();
        
        inline void update_QKr();
        inline void update_QKv();

        inline void update_V_and_concentration_derivatives(const double dt);
        inline void update_gate_derivatives(const double dt);
        inline void update_Kr_derivatives(const double dt);
        inline void update_Kv_derivatives(const double dt); // Updates both Kv14 and Kv43        

        template <typename PRNG>
        void SSA(const double dt);

        /* Record the values of the CRUStateThread temp back to the CRUState state for CRU i */
        inline void update_CRUstate_from_temp(const CRUStateThread& temp, const int i);

        inline void update_increment_and_sigma(const double dt);
        inline void update_mean_RyR_open();
        inline void update_martingale_quantities();
        
        
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

        void set_initial_value(GlobalState& global_vals, CRUState& cru_vals);

        int get_nCRU() const { return nCRU; }

        template <typename PRNG>
        void euler_step(const double dt);

        //template <typename PRNG>
        //void euler(const FloatType dt, const int nstep, const std::function<FloatType(FloatType)>& Is);
    

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


}




