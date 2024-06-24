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
    class GW_model {
    private:

        int nCRU;
        Constants<FloatType> consts;

        //FloatType VFRT;
        //FloatType expmVFRT;

        FloatType QKr_storage[5*5] = {0};
        FloatType QKv14_storage[10*10] = {0};
        FloatType QKv43_storage[10*10] = {0};
        NDArrayMap<FloatType,2> QKr;
        NDArrayMap<FloatType,2> QKv14;
        NDArrayMap<FloatType,2> QKv43;
        static constexpr int Kr_dims[2] = { 5, 5 };
        static constexpr int Kv_dims[2] = { 10, 10 };

        GlobalState<FloatType> dGlobals;

        NDArray<FloatType,2> JLCC;
        NDArray<FloatType,2> Jxfer;
        NDArray<FloatType,1> Jtr;

        FloatType Istim = 0;

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

        Parameters<FloatType> parameters;
        GlobalState<FloatType> globals;
        CRUState<FloatType> CRUs;

        GW_model(int nCRU_simulated) : nCRU(nCRU_simulated), parameters(), globals(), CRUs(nCRU_simulated), consts(), 
                                       dGlobals(0.0), JLCC(NDArray<FloatType,2>(nCRU_simulated,4)), Jxfer(NDArray<FloatType,2>(nCRU_simulated,4)), 
                                       Jtr(NDArray<FloatType,1>(nCRU_simulated))  
        { 
            
            //initialise_CaSS(CaSS);
            //initialise_CaJSR(CaJSR);
            //initialise_LCC(LCC);
            //initialise_LCC_a(LCC_activation);
            //initialise_RyR(RyR);
            //initialise_ClCh(ClCh);

            set_from_params(consts, parameters, nCRU);
            consts.VF_RT = globals.V * consts.F_RT;
            consts.JLCC_exp = exp(2*consts.VF_RT);
            
            initialise_JLCC();
            initialise_Jxfer();
            initialise_Jtr();
            


            //Kr[0] = 0.999503;
            //Kr[1] = 4.13720e-4;
            //Kr[2] = 7.27568e-5;
            //Kr[3] = 8.73984e-6;
            //Kr[4] = 1.36159e-6;
            
            //Kv43[0] = 0.953060;
            //Kv43[1] = 0.0253906;
            //Kv43[2] = 2.53848e-4;
            //Kv43[3] = 1.12796e-6;
            //Kv43[4] = 1.87950e-9;
            //Kv43[5] = 0.0151370;
            //Kv43[6] = 0.00517622;
            //Kv43[7] = 8.96600e-4;
            //Kv43[8] = 8.17569e-5;
            //Kv43[9] = 2.24032e-6;

            //Kv14[0] = 0.722328;
            //Kv14[1] = 0.101971;
            //Kv14[2] = 0.00539932;
            //Kv14[3] = 1.27081e-4;
            //Kv14[4] = 1.82742e-6;
            //Kv14[5] = 0.152769;
            //Kv14[6] = 0.00962328;
            //Kv14[7] = 0.00439043;
            //Kv14[8] = 0.00195348;
            //Kv14[9] = 0.00143629;
            

            QKr = NDArrayMap<double,2>(QKr_storage, Kr_dims, 5*5);
            QKv14 = NDArrayMap<double,2>(QKv14_storage, Kv_dims, 10*10);
            QKv43 = NDArrayMap<double,2>(QKv43_storage, Kv_dims, 10*10);
            initialise_QKr();
            QKv14.set_to_zeros();
            QKv43.set_to_zeros();
            //QKr(1,2) = parameters.Kf;
            //QKr(2,1) = parameters.Kb;
        }

        void euler_step(const FloatType dt);

        template <typename LambdaType>
        void euler(const FloatType dt, const int nstep, const LambdaType&& Is);

        template <typename LambdaType>
        void euler_write(const FloatType dt, const int nstep, const LambdaType&& Is, std::ofstream &file, const int record_every);

};

}
    
#include "GW.tpp"
#endif

//void initialise_QKv14(NDArrayMap<double,2> &Q, const Parameters<double> &params);


//void update_Kr_derivative(double* const deriv, const double* const state, const NDArrayMap<double,2> &Q, const double dt);
//void update_Kv_derivative(double* const deriv, const double* const state, const NDArrayMap<double,2> &Q, const double dt);
//void update_QKr(NDArrayMap<double,2> &Q, const double V, const Parameters<double> &params);
//void update_QKv(NDArrayMap<double,2> &Q, const double V, const double alphaa0, const double aa, const double alphai0, const double ai, 
//                const double betaa0, const double ba, const double betai0, const double bi, const double f1, const double f2,
 //               const double f3, const double f4, const double b1, const double b2, const double b3, const double b4);





//double urand(){
//    static thread_local std::random_device rd;
//    static thread_local std::mt19937_64 gen(rd());
//    static std::uniform_real_distribution<double> dist(0.0, 1.0);
//    return dist(gen);
//}



