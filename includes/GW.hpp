#ifndef GW_H
#define GW_H

#include <cmath>
#include <fstream>
#include <omp.h>
#include "ndarray.hpp"
#include "common.hpp"
#include "GW_utils.hpp"
#include "SSA.hpp"

//const double F = 96.5;
//const double T = 310.;
//const double R = 8.314;
//const double FRT = F / (R*T);


// This holds the model parameters with defaults specified.
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

    template <typename FloatType>    
    class GW_model {
    public:
        Parameters<FloatType> parameters;
        GlobalState<FloatType> globals;
        CRUState<FloatType> CRUs;
        
        FloatType Istim = 0;
    private:

        int nCRU;
        Constants<FloatType> consts;
        NDArray<FloatType,2> JLCC;
        NDArray<FloatType,2> Jxfer;
        NDArray<FloatType,1> Jtr;

        FloatType QKr_storage[5*5] = {0};
        FloatType QKv14_storage[10*10] = {0};
        FloatType QKv43_storage[10*10] = {0};
        NDArrayMap<FloatType,2> QKr;
        NDArrayMap<FloatType,2> QKv14;
        NDArrayMap<FloatType,2> QKv43;
        static constexpr int Kr_dims[2] = { 5, 5 };
        static constexpr int Kv_dims[2] = { 10, 10 };

        Currents<FloatType> currents;
        GlobalState<FloatType> dGlobals;

        void initialise_QKr();
        void initialise_JLCC();
        void initialise_Jxfer();
        void initialise_Jtr();
        
        void update_QKr();
        void update_QKv();

        void update_V_and_concentration_derivatives(const FloatType dt);
        void update_gate_derivatives(const FloatType dt);
        void update_Kr_derivatives(const FloatType dt);
        void update_Kv_derivatives(const FloatType dt); // Updates both Kv14 and Kv43        

        void SSA(const FloatType dt);

        /* Record the values of the CRUStateThread temp back to the CRUState state for CRU i */
        inline void update_CRUstate_from_temp(const CRUStateThread<FloatType> &temp, const int i);
        
        void write_header(std::ofstream &file);
        void write_state(std::ofstream &file, const FloatType t);
        
    public:

        GW_model(int nCRU_simulated) : parameters(), globals(), CRUs(nCRU_simulated), nCRU(nCRU_simulated), consts(parameters, nCRU), 
                                       JLCC(NDArray<FloatType,2>(nCRU_simulated,4)), Jxfer(NDArray<FloatType,2>(nCRU_simulated,4)), 
                                       Jtr(NDArray<FloatType,1>(nCRU_simulated)), QKr(NDArrayMap<FloatType,2>(QKr_storage,5,5)), 
                                       QKv14(NDArrayMap<FloatType,2>(QKv14_storage,10,10)), QKv43(NDArrayMap<FloatType,2>(QKv43_storage,10,10)),  
                                       currents(), dGlobals(0.0)
        { 
            consts.VF_RT = globals.V * consts.F_RT;
            consts.JLCC_exp = exp(2*consts.VF_RT);
            initialise_JLCC();
            initialise_Jxfer();
            initialise_Jtr();
            
            initialise_QKr();
            QKv14.set_to_zeros();
            QKv43.set_to_zeros();
        }

        int get_nCRU() const { return nCRU; }

        void euler_step(const FloatType dt);

        template <typename LambdaType>
        void euler(const FloatType dt, const int nstep, const LambdaType&& Is);

        template <typename LambdaType>
        void euler_write(const FloatType dt, const int nstep, const LambdaType&& Is, std::ofstream &file, const int record_every);

};

}
    
#include "GW.tpp"
#endif




