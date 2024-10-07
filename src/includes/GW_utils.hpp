#ifndef GW_UTILS_H
#define GW_UTILS_H

//#include "GW.hpp"
//#include "ndarray.hpp"
#include "common.hpp"
#include <cstring>
#include <vector>
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

template<typename T>
class Array3Container {
private:
    std::vector<T> storage;
public:
    Eigen::TensorMap<Eigen::Tensor<T,3,Eigen::RowMajor>> array;
    Array3Container(int n1, int n2, int n3) : storage(n1*n2*n3), array(storage.data(),n1,n2,n3) { }

};


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

        Parameters() = default;
        Parameters(const Parameters<FloatType>& other);
    };

    template <typename FloatType>
    struct Constants {
        FloatType RT_F;
        FloatType F_RT;
        FloatType CRU_factor;
        FloatType CSA_FVcyto;
        FloatType VSS_Vcyto;
        FloatType Vcyto_VNSR;
        FloatType VJSR_VNSR;
        // LCC rates
        FloatType a;
        FloatType gamma0;
        FloatType gamma0a;
        FloatType gamma0a2; // gamma0*a^2
        FloatType gamma0a3; // gamma0*a^3
        FloatType gamma0a4; // gamma0*a^4
        FloatType binv; // 1/b
        FloatType omega;
        FloatType omega_b; // omega / b
        FloatType omega_b2; // omega / b^2
        FloatType omega_b3; // omega / b^3
        FloatType omega_b4; // omega / b^4
        // CaSS constants
        FloatType BSR_const;
        FloatType BSL_const;
        // CaJSR constants
        FloatType VSS_VJSR; // VSS / VJSR
        FloatType CSQN_const;
        // JLCC constants
        FloatType JLCC_const;
        FloatType Cao_scaled; // 0.341*Cao
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

        // All are constant within the SSA
        FloatType VF_RT; // (V*F) / (R*T) 
        FloatType expmVF_RT; // exp(-VF_RT)
        FloatType ENa;
        FloatType EK;
        FloatType ECa;
        FloatType beta_cyto;
        FloatType JLCC_multiplier; // 
        FloatType JLCC_exp; // exp(2 * VF_RT)
        FloatType alphaLCC; // LCC upwards transition rate
        FloatType betaLCC; // LCC downwards transition rate
        FloatType yinfLCC; // LCC voltage inactivation steady state
        FloatType tauLCC; // LCC voltage inactivation time constant

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

        GlobalState() = default;
        GlobalState(FloatType value);
        GlobalState& operator=(GlobalState& x) = default;
    };

    
    template <typename FloatType>//, typename PRNG>
    struct CRUState {
        Array2<FloatType> CaSS;
        Array1<FloatType> CaJSR;
        Array2<int> LCC;
        Array2<int> LCC_inactivation;
        Array3Container<int> RyR;
        Array2<int> ClCh;
        Array2<FloatType> RyR_open_int;
        Array2<FloatType> RyR_open_martingale;
        Array2<FloatType> RyR_open_martingale_normalised;
        Array1<FloatType> sigma_RyR;
        
        Array2<FloatType> LCC_open_int;
        Array2<FloatType> LCC_open_martingale;
        Array2<FloatType> LCC_open_martingale_normalised;
        Array1<FloatType> sigma_LCC;

        CRUState(const int nCRU);
        CRUState& operator=(CRUState& x) = default;
    };

    inline void initialise_LCC(Array2<int> &LCC);
    inline void initialise_LCC_i(Array2<int> &LCC_i);
    inline void initialise_RyR(Array3<int> &RyR);
    inline void initialise_ClCh(Array2<int> &ClCh);

    template <typename FloatType>
    inline FloatType RKr(const FloatType V){ return 1.0 / (1.0 + 1.4945*exp(0.0446*V)); }

    template <typename FloatType>
    inline FloatType IKr(const FloatType V, const FloatType XKr, const FloatType EK, const FloatType GKr, const FloatType sqrtKo){
        return GKr * sqrtKo * RKr(V) * XKr * (V - EK) * 0.5;
    }


    template <typename FloatType>
    inline FloatType EKs(const FloatType Ki, const FloatType Ko, const FloatType Nai, const FloatType Nao, const FloatType RT_F){
        return log((Ko+0.01833*Nao) / (Ki + 0.01833*Nai)) * RT_F;
    }


    template <typename FloatType>
    inline FloatType IKs(const FloatType V, const FloatType XKs, const FloatType Ki, const FloatType Nai, const FloatType Nao, const FloatType Ko, const FloatType GKs, const FloatType RT_F){
        return GKs * square(XKs) * (V - EKs(Ki, Ko, Nai, Nao, RT_F));
    }


    template <typename FloatType>
    inline FloatType IKv43(const FloatType V, const FloatType XKv43, const FloatType EK, const FloatType GKv43){
        return GKv43 * XKv43 * (V - EK);
    }

    template <typename FloatType>
    inline FloatType IKv14(const FloatType VFRT, const FloatType exp_term, const FloatType XKv14, const FloatType Ki, const FloatType Nai, const FloatType PKv14_Csc, const FloatType Nao, const FloatType Ko){
        FloatType m = (PKv14_Csc) * (FARADAY*VFRT) * XKv14 / (1 - exp_term);
        return m * ((Ki - Ko*exp_term) + 0.02*(Nai - Nao*exp_term)) * 1.0e9; // 1e9 required to convert to mV / ms
    }


    template <typename FloatType>
    inline FloatType K1inf(const FloatType V, const FloatType EK, const FloatType F_RT){ return 1.0 / (2.0 + exp(1.5*(V-EK)*F_RT)); }

    template <typename FloatType>
    inline FloatType IK1(const FloatType V, const FloatType EK, const FloatType GK1, const FloatType IK1_const, const FloatType FR_T){ 
        return GK1 * K1inf(V, EK, FR_T) * IK1_const * (V - EK); 
    }


    template <typename FloatType>
    inline FloatType Kp(const FloatType V){ return 1. / (1. + exp((7.488 - V) / 5.98)); }

    template <typename FloatType>
    inline FloatType IKp(const FloatType V, const FloatType EK, const FloatType GKp){ return GKp * Kp(V) * (V-EK); }


    template <typename FloatType>
    inline FloatType ICaL(const Array2<FloatType> &JLCC, const FloatType ICaL_const){
        return JLCC.sum() * ICaL_const;
    }


    template <typename FloatType>
    inline FloatType Ito2(const Array2<int> &ClCh, const FloatType VFRT, const FloatType expmVFRT, const FloatType Cl_cyto, const FloatType Clo, const FloatType Ito2_const){
        return FloatType(ClCh.sum()) * Ito2_const * VFRT * (Cl_cyto * expmVFRT - Clo) / (expmVFRT - 1.0);
    }

    template <typename FloatType>
    inline FloatType dTRPNCa(const FloatType TRPNCa, const FloatType Cai, const FloatType TRPNtot, const FloatType kTRPNp, const FloatType kTRPNm){
        return kTRPNp*Cai*(TRPNtot - TRPNCa) - kTRPNm*TRPNCa;
    }


    template <typename FloatType>
    inline FloatType Jup(const FloatType Cai, const FloatType CaNSR, const FloatType Vmaxf, const FloatType Vmaxr, const FloatType Kmf, const FloatType Kmr, const FloatType Hf, const FloatType Hr){
        const FloatType f = pow(Cai/Kmf, Hf);
        const FloatType r = pow(CaNSR/Kmr, Hr);
        return (Vmaxf*f - Vmaxr*r) / (1.0 + f + r);
    }


    template <typename FloatType>
    inline FloatType beta_cyto(const FloatType Cai, const FloatType CMDNconst, const FloatType KCMDN){
        const FloatType x = KCMDN + Cai;
        return 1.0 / (1.0 + CMDNconst / square(x));
    }


    template <typename FloatType, typename Array>
    inline FloatType flux_average(const Array &flux_container, const FloatType CRU_factor){ return flux_container.sum() * CRU_factor; }


    template <typename FloatType>
    inline FloatType XKsinf(const FloatType V){ return 1.0 / (1.0 + exp(-(V - 24.7) / 13.6)); }

    template <typename FloatType>
    inline FloatType tauXKs_inv(const FloatType V){ return 0.0000719*(V-10.0)/(1.0 - exp(-0.148*(V-10.0))) + 0.000131*(V-10.0)/(exp(0.0687*(V-10.0)) - 1.0); }


    template <typename FloatType>
    inline FloatType alphaLCC(const FloatType V) { return 2.0 * exp(0.012 * (V - 35.0)); }

    template <typename FloatType>
    inline FloatType betaLCC(const FloatType V) { return 0.0882 * exp(-0.05 * (V - 35.0)); }

    template <typename FloatType>
    inline FloatType yinfLCC(const FloatType V) { return 0.4 / (1.0 + exp((V + 12.5) / 5.0)) + 0.6; }

    template <typename FloatType>
    inline FloatType tauLCC(const FloatType V)  { return 340.0 / (1.0 + exp((V + 30.0) / 12.0)) + 60.0; }


    template <typename FloatType>
    inline void update_LCC_inactivation_rates(FloatType* const LCC_inactivation_rates, FloatType* const subunit_rates, const int* const LCC_inactivation, const FloatType yinf, 
                                            const FloatType tau, const int idx){
        LCC_inactivation_rates[idx] = (LCC_inactivation[idx] == 0) ? yinf / tau : (1.0 - yinf) / tau;
        subunit_rates[idx] += LCC_inactivation_rates[idx];
    }


    template <typename FloatType>
    inline void update_ClCh_rates(FloatType* const ClCh_rates, FloatType* const subunit_rates, const int* const ClCh, const FloatType* const CaSS, const FloatType kfClCh, 
                                  const FloatType kbClCh, const int idx){
        ClCh_rates[idx] = (ClCh[idx] == 0) ? kfClCh * CaSS[idx] : kbClCh;
        subunit_rates[idx] += ClCh_rates[idx];
    }


    template <typename FloatType>
    inline void update_LCC_rates(FloatType* const LCC_rates, FloatType* const subunit_rates, const int* const LCC, const FloatType* const CaSS, const int idx, const Parameters<FloatType> &params, const Constants<FloatType> &consts);

    
    template <typename FloatType>
    void update_RyR_rates(FloatType* const RyR_rates, FloatType* const subunit_rates, const int* const RyR, const FloatType* const CaSS, const int idx, const Parameters<FloatType> &params);
  
}

#include "GW_utils.tpp"

#endif



