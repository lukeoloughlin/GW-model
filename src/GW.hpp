#ifndef GW_H
#define GW_H

#include <iostream>
#include <array>
#include <cmath>
#include <ctime>
#include <random>
#include "ndarray.hpp"
#include "common.hpp"
#include "GW_currents.hpp"

//const double F = 96.5;
//const double T = 310.;
//const double R = 8.314;
//const double FRT = F / (R*T);


// This holds the model parameters with defaults specified.
namespace GW {
    template <typename FloatType>
    struct Parameters {
        FloatType T = 310.0;
        FloatType CSA = 153.4;
        FloatType Vcyto = 25.84;
        FloatType VNSR = 1.113;
        FloatType VJSR = 22.26e-6;
        FloatType VSS = 0.2303e-6;
        int NCaRU = 12500;
        FloatType Ko = 4.0;
        FloatType Nao = 138.0;
        FloatType Cao = 2.0;
        FloatType Clo = 150.0;
        FloatType Clcyto = 20.0;
        FloatType f = 0.85;
        FloatType g = 2.0;
        FloatType f1 = 0.005;
        FloatType g1 = 7.0;
        FloatType a = 2.0;
        FloatType b = 1.9356;
        FloatType gamma0 = 0.44;
        FloatType omega = 0.02158;
        FloatType PCaL = 9.13e-13;
        FloatType KdClCh = 0.1502;
        FloatType kfClCh = 13.3156;
        FloatType kbClCh = 2.0;
        FloatType Pto2 = 2.65e-15;

        FloatType k12 = 877.5;
        FloatType k21 = 250.0;
        FloatType k23 = 2.358e8;
        FloatType k32 = 9.6;
        FloatType k34 = 1.415e6;
        FloatType k43 = 13.65;
        FloatType k45 = 0.07;
        FloatType k54 = 93.385;
        FloatType k56 = 1.887e7;
        FloatType k65 = 30.0;
        FloatType k25 = 2.358e6;
        FloatType k52 = 0.001235;
        FloatType rRyR = 3.92;

        FloatType rxfer = 200.0;
        FloatType rtr = 0.333;
        FloatType riss = 20.0;
        FloatType BSRT = 0.047;
        FloatType KBSR = 0.00087;
        FloatType BSLT = 1.124;
        FloatType KBSL = 0.0087;
        FloatType CSQNT = 13.5;
        FloatType KCSQN = 0.63;
        FloatType CMDNT = 0.05;
        FloatType KCMDN = 0.00238;
        FloatType GNa = 12.8;
        FloatType GKr = 0.024;
        FloatType Kf = 0.0266;
        FloatType Kb = 0.1348;
        FloatType GKs = 0.00271;
        FloatType GKv43 = 0.1389;
        FloatType alphaa0Kv43 = 0.5437;
        FloatType aaKv43 = 0.02898;
        FloatType betaa0Kv43 = 0.08019;
        FloatType baKv43 = 0.04684;
        FloatType alphai0Kv43 = 0.04984;
        FloatType aiKv43 = 3.37302e-4;
        FloatType betai0Kv43 = 8.1948e-4;
        FloatType biKv43 = 5.374e-8;
        FloatType f1Kv43 = 1.8936;
        FloatType f2Kv43 = 14.225;
        FloatType f3Kv43 = 158.574;
        FloatType f4Kv43 = 142.937;
        FloatType b1Kv43 = 6.7735;
        FloatType b2Kv43 = 15.621;
        FloatType b3Kv43 = 28.753;
        FloatType b4Kv43 = 524.576;
        FloatType PKv14 = 1.989e-7;
        FloatType alphaa0Kv14 = 1.8931;
        FloatType aaKv14 = 0.006950;
        FloatType betaa0Kv14 = 0.01179;
        FloatType baKv14 = 0.08527;
        FloatType alphai0Kv14 = 0.002963;
        FloatType aiKv14 = 0.0;
        FloatType betai0Kv14 = 1.0571e-4;
        FloatType biKv14 = 0.0;
        FloatType f1Kv14 = 0.2001;
        FloatType f2Kv14 = 0.3203;
        FloatType f3Kv14 = 13.509;
        FloatType f4Kv14 = 1151.765;
        FloatType b1Kv14 = 2.230;
        FloatType b2Kv14 = 12.0;
        FloatType b3Kv14 = 5.370;
        FloatType b4Kv14 = 5.240;
        FloatType Csc = 1.0e6;
        FloatType GK1 = 3.0;
        FloatType KmK1 = 13.0;
        FloatType GKp = 0.002659;
        FloatType kNaCa = 0.27;
        FloatType KmNa = 87.5;
        FloatType KmCa = 1.38;
        FloatType ksat = 0.2;
        FloatType eta = 0.35;
        FloatType INaKmax = 0.901;
        FloatType KmNai = 10.0;
        FloatType KmKo = 1.5;
        FloatType IpCamax = 0.03;
        FloatType KmpCa = 0.0005;
        FloatType GCab = 0.0002536;
        FloatType GNab = 0.00264;
        FloatType kHTRPNp = 20.0;
        FloatType kHTRPNm = 6.60e-5;
        FloatType kLTRPNp = 40.0;
        FloatType kLTRPNm = 0.04;
        FloatType HTRPNtot = 0.140;
        FloatType LTRPNtot = 0.070;
        FloatType Vmaxf = 0.0002096;
        FloatType Vmaxr = 0.0002096;
        FloatType Kmf = 0.000260;
        FloatType Kmr = 1.8;
        FloatType Hf = 0.75;
        FloatType Hr = 0.75;
    };

    template <typename FloatType>
    struct Constants {
        FloatType RT_F;
        FloatType CRU_factor;
        FloatType CSA_FVcyto;
        FloatType VSS_Vcyto;
        FloatType Vcyto_VNSR;
        FloatType VJSR_VNSR;
        FloatType FRT;
        FloatType riss;
        FloatType rxfer;
        // LCC rates
        FloatType gamma0;
        FloatType omega;
        FloatType a;
        FloatType a2; // a^2
        FloatType a3; // a^3
        FloatType a4; // a^4
        FloatType bi; // 1/b
        FloatType bi2; // 1/b^2
        FloatType bi3; // 1/b^3
        FloatType bi4; // 1/b^4
        FloatType f;
        FloatType g;
        FloatType f1;
        FloatType g1;
        // RyR rates
        FloatType k12;
        FloatType k21;
        FloatType k23;
        FloatType k32;
        FloatType k34;
        FloatType k43;
        FloatType k45;
        FloatType k54;
        FloatType k56;
        FloatType k65;
        FloatType k25;
        FloatType k52;
        // ClCh rates
        FloatType kfClCh;
        FloatType kbClCh;
        // CaSS constants
        FloatType KBSR;
        FloatType BSR_const;
        FloatType KBSL;
        FloatType BSL_const;
        // CaJSR constants
        FloatType VSS_VJSR; // VSS / VJSR
        FloatType KCSQN;
        FloatType CSQN_const;
        // JLCC constants
        FloatType JLCC_const;
        FloatType Cao_scaled; // 0.341*Cao
        // Jrel constants
        FloatType rRyR;
        // Jtr constants
        FloatType rtr;

        // INaCa constants
        FloatType Nao3; // Nao^3
        FloatType INaCa_const; // 5000*kNaCa / ((KmNa^3 + Nao^3) * (KmCa + Cao))
        // INaK consts
        FloatType sigma; // (exp(Nao / 67.3) - 1.0) / 7.0
        FloatType INaK_const; // (exp(Nao / 67.3) - 1) / 7
        // Ikr consts
        FloatType sqrtKo;
        // Ito1 consts
        FloatType PKv14_Csc; // PKv14 / Csc
        // Ito2 consts
        FloatType Ito2_const; // 1e9 * Pto2 * F * (NCaRU / size) / CSA
        // IK1 consts
        FloatType IK1_const; // Ko / (Ko + KmK1)
        // ICaL consts
        FloatType ICaL_const; // -1000. * (2F * VSS) * (NCaRU / size) / CSA
        // CMDN consts
        FloatType CMDN_const;


        Constants(const Parameters<FloatType> &params, const int nCRU_simulated);
        
    };

    template <typename FloatType>
    struct GlobalState {
        FloatType V = -91.382;

        FloatType Nai = 10.0;
        FloatType Ki = 131.84;
        FloatType Cai = 1.45273e-4;
        FloatType CaNSR = 0.908882;
        FloatType CaLTRPN = 8.98282e-3;
        FloatType CaHTRPN = 0.137617;
        
        FloatType m = 5.33837e-4;
        FloatType h = 0.996345;
        FloatType j = 0.997315;
        FloatType xKs = 2.04171e-4;
        
        FloatType Kr[5] = { 0.999503, 4.13720e-4, 7.27568e-5, 8.73984e-6, 1.36159e-6 };
        FloatType Kv43[10] = { 0.953060, 0.0253906, 2.53848e-4, 1.12796e-6, 1.87950e-9, 0.0151370, 0.00517622, 8.96600e-4, 8.17569e-5, 2.24032e-6 };
        FloatType Kv14[10] = { 0.722328, 0.101971, 0.00539932, 1.27081e-4, 1.82742e-6, 0.152769, 0.00962328, 0.00439043, 0.00195348, 0.00143629 }; 

        GlobalState(FloatType value);
    };
    
    template <typename FloatType>
    struct CRUState {
        NDArray<FloatType,2> CaSS;
        NDArray<FloatType,1> CaJSR;
        NDArray<int,2> LCC;
        NDArray<int,2> LCC_activation;
        NDArray<int,3> RyR;
        NDArray<int,2> ClCh;

        CRUState(const int nCRU);
    };


    template <typename FloatType>    
    class GW_model {
    private:

        int nCRU;
        Constants<FloatType> consts;

        FloatType VFRT;
        FloatType expmVFRT;

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
        
        void write_state(ofstream &file, const FloatType t);
        
    public:

        Parameters<FloatType> parameters;
        GlobalState<FloatType> globals;
        CRUState<FloatType> CRUs;

        GW_model(int nCRU_simulated) : nCRU(nCRU_simulated), parameters(), globals(), CRUs(nCRU_simulated), consts(parameters, nCRU_simulated), 
                                       dGlobals(0.0), JLCC(NDArray<FloatType,2>(nCRU_simulated,4)), Jxfer(NDArray<FloatType,2>(nCRU_simulated,4)), 
                                       Jtr(NDArray<FloatType,1>(nCRU_simulated))  
        { 
            
            //initialise_CaSS(CaSS);
            //initialise_CaJSR(CaJSR);
            //initialise_LCC(LCC);
            //initialise_LCC_a(LCC_activation);
            //initialise_RyR(RyR);
            //initialise_ClCh(ClCh);

            VFRT = globals.V * FARADAY / (GAS_CONST * parameters.T);
            
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
        void euler(const FloatType dt, const int nstep, const std::function<FloatType(FloatType)> Is);
        void euler_write(const FloatType dt, const int nstep, const std::function<FloatType(FloatType)> Is, ofstream &file, const int record_every);

};
    
#include "GW.tpp"

//void initialise_QKv14(NDArrayMap<double,2> &Q, const Parameters<double> &params);


//void update_Kr_derivative(double* const deriv, const double* const state, const NDArrayMap<double,2> &Q, const double dt);
//void update_Kv_derivative(double* const deriv, const double* const state, const NDArrayMap<double,2> &Q, const double dt);
//void update_QKr(NDArrayMap<double,2> &Q, const double V, const Parameters<double> &params);
//void update_QKv(NDArrayMap<double,2> &Q, const double V, const double alphaa0, const double aa, const double alphai0, const double ai, 
//                const double betaa0, const double ba, const double betai0, const double bi, const double f1, const double f2,
 //               const double f3, const double f4, const double b1, const double b2, const double b3, const double b4);


}


//double urand(){
//    static thread_local std::random_device rd;
//    static thread_local std::mt19937_64 gen(rd());
//    static std::uniform_real_distribution<double> dist(0.0, 1.0);
//    return dist(gen);
//}




#endif