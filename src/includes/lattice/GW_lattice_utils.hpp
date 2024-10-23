#pragma once

//#include "GW_utils.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>
#include <random>

template<typename T>
using Array1 = Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>;
template<typename T>
using Array2L = Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
template<typename T>
using Array3 = Eigen::TensorMap<Eigen::Tensor<T,3,Eigen::RowMajor>>;

/*
Bug in pybind11 interface of Eigen tensors that causes hangs in python after calling a function more than once. Using a TensorMap works, so
we use this container class as a work around
*/


namespace GW_lattice {
    
    //template <typename T>
    struct CRULatticeState {
        Array2L<double> CaSS;
        Array2L<double> CaJSR; // changing this to Array2L because CRUs are no longer distinct structures
        Array2L<double> Cai;
        Array2L<double> CaNSR;
        Array2L<double> CaLTRPN;
        Array2L<double> CaHTRPN;
        Array2L<int> LCC;
        Array2L<int> LCC_inactivation;
        Array3Container<int> RyR;
        Array2L<int> ClCh;

        CRULatticeState(const int nCRU_x, const int nCRU_y);
        CRULatticeState& operator=(CRULatticeState& x) = default;
    };
    
    //template <typename T>
    struct Parameters {
        double T = 310.0;
        double CSA = 153.4;
        double Vcyto = 25.84;
        double VNSR = 1.113;
        double VJSR = 22.26e-6 / 4; // Dividing by 4 since I split the JSR compartment into 4 subpaces
        double VSS = 0.2303e-6;
        int NCaRU = 50000; // 12,500 * 4 since I only consider subspaces now
        double Ko = 4.0;
        double Nao = 138.0;
        double Cao = 2.0;
        double Clo = 150.0;
        double Clcyto = 20.0;
        double f = 0.85;
        double g = 2.0;
        double f1 = 0.005;
        double g1 = 7.0;
        double a = 2.0;
        double b = 1.9356;
        double gamma0 = 0.44;
        double omega = 0.02158;
        double PCaL = 9.13e-13;
        double kfClCh = 13.3156;
        double kbClCh = 2.0;
        double Pto2 = 2.65e-15;

        double k12 = 877.5;
        double k21 = 250.0;
        double k23 = 2.358e8;
        double k32 = 9.6;
        double k34 = 1.415e6;
        double k43 = 13.65;
        double k45 = 0.07;
        double k54 = 93.385;
        double k56 = 1.887e7;
        double k65 = 30.0;
        double k25 = 2.358e6;
        double k52 = 0.001235;
        double rRyR = 3.92;

        double rxfer = 200.0;
        double rtr = 0.333;
        double rcyto = 0.7; // Dcyto / dx^2
        double rnsr = 2.7; // Dnsr / dx^2
        double BSRT = 0.047;
        double KBSR = 0.00087;
        double BSLT = 1.124;
        double KBSL = 0.0087;
        double CSQNT = 13.5;
        double KCSQN = 0.63;
        double CMDNT = 0.05;
        double KCMDN = 0.00238;
        double GNa = 12.8;
        double GKr = 0.024;
        double Kf = 0.0266;
        double Kb = 0.1348;
        double GKs = 0.00271;
        double GKv43 = 0.1389;
        double alphaa0Kv43 = 0.5437;
        double aaKv43 = 0.02898;
        double betaa0Kv43 = 0.08019;
        double baKv43 = 0.04684;
        double alphai0Kv43 = 0.04984;
        double aiKv43 = 3.37302e-4;
        double betai0Kv43 = 8.1948e-4;
        double biKv43 = 5.374e-8;
        double f1Kv43 = 1.8936;
        double f2Kv43 = 14.225;
        double f3Kv43 = 158.574;
        double f4Kv43 = 142.937;
        double b1Kv43 = 6.7735;
        double b2Kv43 = 15.621;
        double b3Kv43 = 28.753;
        double b4Kv43 = 524.576;
        double PKv14 = 1.989e-7;
        double alphaa0Kv14 = 1.8931;
        double aaKv14 = 0.006950;
        double betaa0Kv14 = 0.01179;
        double baKv14 = 0.08527;
        double alphai0Kv14 = 0.002963;
        double aiKv14 = 0.0;
        double betai0Kv14 = 1.0571e-4;
        double biKv14 = 0.0;
        double f1Kv14 = 0.2001;
        double f2Kv14 = 0.3203;
        double f3Kv14 = 13.509;
        double f4Kv14 = 1151.765;
        double b1Kv14 = 2.230;
        double b2Kv14 = 12.0;
        double b3Kv14 = 5.370;
        double b4Kv14 = 5.240;
        double Csc = 1.0e6;
        double GK1 = 3.0;
        double KmK1 = 13.0;
        double GKp = 0.002659;
        double kNaCa = 0.27;
        double KmNa = 87.5;
        double KmCa = 1.38;
        double ksat = 0.2;
        double eta = 0.35;
        double INaKmax = 0.901;
        double KmNai = 10.0;
        double KmKo = 1.5;
        double IpCamax = 0.03;
        double KmpCa = 0.0005;
        double GCab = 0.0002536;
        double GNab = 0.00264;
        double kHTRPNp = 20.0;
        double kHTRPNm = 6.60e-5;
        double kLTRPNp = 40.0;
        double kLTRPNm = 0.04;
        double HTRPNtot = 0.140;
        double LTRPNtot = 0.070;
        double Vmaxf = 0.0002096;
        double Vmaxr = 0.0002096;
        double Kmf = 0.000260;
        double Kmr = 1.8;
        double Hf = 0.75;
        double Hr = 0.75;

        Parameters() = default;
        Parameters(const Parameters& other) = default;
    };

    //template <typename T>
    struct Constants {
        double RT_F;
        double F_RT;

        double Vcyto_elem;
        double VNSR_elem;

        double CRU_factor;
        double CSA_F;
        double VSS_Vcyto;
        double Vcyto_VNSR;
        double VJSR_VNSR;
        // LCC rates
        double a;
        double gamma0;
        double gamma0a;
        double gamma0a2; // gamma0*a^2
        double gamma0a3; // gamma0*a^3
        double gamma0a4; // gamma0*a^4
        double binv; // 1/b
        double omega;
        double omega_b; // omega / b
        double omega_b2; // omega / b^2
        double omega_b3; // omega / b^3
        double omega_b4; // omega / b^4
        double f; 
        double f1; 
        double g; 
        double g1; 
        // CaSS constants
        double BSR_const;
        double BSL_const;
        // CaJSR constants
        double VSS_VJSR; // VSS / VJSR
        double CSQN_const;
        // JLCC constants
        double JLCC_const;
        double Cao_scaled; // 0.341*Cao
        // INaCa constants
        double Nao3; // Nao^3
        double INaCa_const; // 5000*kNaCa / ((KmNa^3 + Nao^3) * (KmCa + Cao))
        // INaK consts
        double sigma; // (exp(Nao / 67.3) - 1.0) / 7.0
        double INaK_const; // (exp(Nao / 67.3) - 1) / 7
        // Ikr consts
        double sqrtKo;
        // Ito1 consts
        double PKv14_Csc; // PKv14 / Csc
        // Ito2 consts
        double Ito2_const; // 1e9 * Pto2 * F * (NCaRU / size) / CSA
        // IK1 consts
        double IK1_const; // Ko / (Ko + KmK1)
        // ICaL consts
        double ICaL_const; // -1000. * (2F * VSS) * (NCaRU / size) / CSA
        // CMDN consts
        double CMDN_const;

        // All are constant within the SSA
        double VF_RT; // (V*F) / (R*T) 
        double expmVF_RT; // exp(-VF_RT)
        double JLCC_multiplier; // 
        double JLCC_exp; // exp(2 * VF_RT)
        double alphaLCC; // LCC upwards transition rate
        double betaLCC; // LCC downwards transition rate
        double yinfLCC; // LCC voltage inactivation steady state
        double tauLCC; // LCC voltage inactivation time constant

        Constants(const Parameters& params, const int nCRU_x, const int nCRU_y);
    };


    struct GlobalState {
        double V = -91.382;

        double Nai = 10.0;
        double Ki = 131.84;
        
        double m = 5.33837e-4;
        double h = 0.996345;
        double j = 0.997315;
        double xKs = 2.04171e-4;
        
        double Kr[5] = { 0.999503, 4.13720e-4, 7.27568e-5, 8.73984e-6, 1.36159e-6 };
        double Kv43[10] = { 0.953060, 0.0253906, 2.53848e-4, 1.12796e-6, 1.87950e-9, 0.0151370, 0.00517622, 8.96600e-4, 8.17569e-5, 2.24032e-6 };
        double Kv14[10] = { 0.722328, 0.101971, 0.00539932, 1.27081e-4, 1.82742e-6, 0.152769, 0.00962328, 0.00439043, 0.00195348, 0.00143629 }; 

        GlobalState() = default;
        GlobalState(double value);
        GlobalState& operator=(GlobalState& x) = default;
    }; 

}


