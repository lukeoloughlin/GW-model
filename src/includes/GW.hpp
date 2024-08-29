#ifndef GW_H
#define GW_H


#include <omp.h>
#include "common.hpp"
#include "GW_utils.hpp"
#include "SSA.hpp"
#include <Eigen/Core>

template<typename T>
using Array1 = Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>;
template<typename T>
using Array2 = Eigen::Array<T,Eigen::Dynamic,4,Eigen::RowMajor>;

template<typename T>
using QKrMap = Eigen::Map<Eigen::Array<T,5,5>,Eigen::RowMajor>;
template<typename T>
using QKvMap = Eigen::Map<Eigen::Array<T,10,10>,Eigen::RowMajor>;


namespace GW {

    template <typename FloatType>
    struct Currents {
        FloatType INa = 0;
        FloatType INab = 0;
        FloatType INaCa = 0;
        FloatType INaK = 0;
        FloatType IKr = 0;
        FloatType IKs = 0;
        FloatType Ito1 = 0;
        FloatType Ito2 = 0;
        FloatType IK1 = 0;
        FloatType IKp = 0;
        FloatType ICaL = 0;
        FloatType ICab = 0;
        FloatType IpCa = 0;
        FloatType Jup = 0;
        FloatType Jtr_tot = 0;
        FloatType Jxfer_tot = 0;
    };


    template <typename FloatType, typename PRNG>    
    class GW_model {
    public:
        Parameters<FloatType> parameters;
        GlobalState<FloatType> globals;
        //CRUState<FloatType, PRNG> CRUs;
        CRUState<FloatType> CRUs;
        
        FloatType Istim = 0;
    private:

        int nCRU;
        Constants<FloatType> consts;
        Array2<FloatType> JLCC;
        Array2<FloatType> Jxfer;
        Array1<FloatType> Jtr;

        FloatType QKr_storage[5*5] = {0};
        FloatType QKv14_storage[10*10] = {0};
        FloatType QKv43_storage[10*10] = {0};
        QKrMap<FloatType> QKr;
        QKvMap<FloatType> QKv14;
        QKvMap<FloatType> QKv43;

        Currents<FloatType> currents;
        GlobalState<FloatType> dGlobals;

    
        inline void initialise_Jxfer(){ Jxfer = parameters.rxfer * (CRUs.CaSS - globals.Cai); }
        inline void initialise_Jtr(){ Jtr = parameters.rtr * (globals.CaNSR - CRUs.CaJSR); }
        inline void initialise_QKr(){ QKr(1,2) = parameters.Kf; QKr(2,1) = parameters.Kb; }
        inline void initialise_JLCC();
        
        inline void update_QKr();
        inline void update_QKv();

        inline void update_V_and_concentration_derivatives(const FloatType dt);
        inline void update_gate_derivatives(const FloatType dt);
        inline void update_Kr_derivatives(const FloatType dt);
        inline void update_Kv_derivatives(const FloatType dt); // Updates both Kv14 and Kv43        

        void SSA(const FloatType dt);

        /* Record the values of the CRUStateThread temp back to the CRUState state for CRU i */
        inline void update_CRUstate_from_temp(const CRUStateThread<FloatType> &temp, const int i);
        
        
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

        GW_model(const Parameters<FloatType>& params, int nCRU_simulated) : parameters(params), globals(), CRUs(nCRU_simulated), nCRU(nCRU_simulated), consts(parameters, nCRU), JLCC(nCRU_simulated,4), 
                                       Jxfer(nCRU_simulated,4), Jtr(nCRU_simulated), QKr(QKr_storage), QKv14(QKv14_storage), QKv43(QKv43_storage), currents(), 
                                       dGlobals(0.0) { 
            consts.VF_RT = globals.V * consts.F_RT;
            consts.JLCC_exp = exp(2*consts.VF_RT);
            initialise_JLCC();
            initialise_Jxfer();
            initialise_Jtr();
            initialise_QKr();
        }

        void set_initial_value(GlobalState<FloatType>& global_vals, CRUState<FloatType>& cru_vals);

        int get_nCRU() const { return nCRU; }

        void euler_step(const FloatType dt);

        void euler(const FloatType dt, const int nstep, const std::function<FloatType(FloatType)>& Is);


};

}
    
#include "GW.tpp"
#endif




