
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


namespace GW_lattice {
    
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
    
    template <typename FloatType>
    struct Parameters {
        FloatType T = 310.0;
        FloatType CSA = 153.4;
        FloatType Vcyto = 25.84;
        FloatType VNSR = 1.113;
        FloatType VJSR = 22.26e-6 / 4; // Dividing by 4 since I split the JSR compartment into 4 subpaces
        FloatType VSS = 0.2303e-6;
        int NCaRU = 50000; // 12,500 * 4 since I only consider subspaces now
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
        FloatType riss = 20.0*0.5; // Halving this since each subunit interacts with double the number of neighbouring subunits now (except on the boundary)
        FloatType rijsr = 100.0; // New parameter, JSR diffusion rate. Setting to 10*riss for simplicity
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
        FloatType JLCC_multiplier; // 
        FloatType JLCC_exp; // exp(2 * VF_RT)
        FloatType alphaLCC; // LCC upwards transition rate
        FloatType betaLCC; // LCC downwards transition rate
        FloatType yinfLCC; // LCC voltage inactivation steady state
        FloatType tauLCC; // LCC voltage inactivation time constant

        Constants(const Parameters<FloatType> &params, const int nCRU_simulated);
    };
    

}


#include "GW_lattice_utils.tpp"

#endif