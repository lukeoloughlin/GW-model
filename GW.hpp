#ifndef _GW_H
#define _GW_H

#include <iostream>
#include <array>
#include <cmath>
#include <ctime>
#include <random>
#include "ndarray.hpp"


using namespace std;

const double F = 96.5;
const double T = 310.;
const double R = 8.314;
const double FRT = F / (R*T);


// This holds the model parameters with defaults specified.
struct GW_parameters {
    double CSA = 153.4;
    double Vcyto = 25.84;
    double VNSR = 1.113;
    double VJSR = 22.26e-6;
    double VSS = 0.2303e-6;
    int NCaRU = 12500;
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
    double KdClCh = 0.1502;
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
    double riss = 20.0;
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
};

// This holds constant values when the simulation is executed, preventing unnecessary recalculations.
struct Constants {
    double CRU_factor;
    double CSA_FVcyto;
    double VSS_Vcyto;
    double Vcyto_VNSR;
    double VJSR_VNSR;
    double FRT;
    double riss;
    double rxfer;
    // LCC rates
    double gamma0;
    double omega;
    double a;
    double a2; // a^2
    double a3; // a^3
    double a4; // a^4
    double bi; // 1/b
    double bi2; // 1/b^2
    double bi3; // 1/b^3
    double bi4; // 1/b^4
    double f;
    double g;
    double f1;
    double g1;
    // RyR rates
    double k12;
    double k21;
    double k23;
    double k32;
    double k34;
    double k43;
    double k45;
    double k54;
    double k56;
    double k65;
    double k25;
    double k52;
    // ClCh rates
    double kfClCh;
    double kbClCh;
    // CaSS constants
    double KBSR;
    double BSR_const;
    double KBSL;
    double BSL_const;
    // CaJSR constants
    double VSS_VJSR; // VSS / VJSR
    double KCSQN;
    double CSQN_const;
    // JLCC constants
    double JLCC_const;
    double Cao_scaled; // 0.341*Cao
    // Jrel constants
    double rRyR;
    // Jtr constants
    double rtr;

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
};
    

Constants consts_from_params(const GW_parameters &params, const int nCRU_simulated)
{
    Constants consts;
    consts.CRU_factor = double(params.NCaRU) / double(nCRU_simulated);
    consts.CSA_FVcyto = params.CSA / (1000.0 * params.Vcyto * F);
    consts.VSS_Vcyto = params.VSS / params.Vcyto;
    consts.Vcyto_VNSR = params.Vcyto / params.VNSR;
    consts.VJSR_VNSR = params.VJSR / params.VNSR;
    consts.FRT = FRT;
    consts.riss = params.riss;
    consts.rxfer = params.rxfer;
    // LCC rates
    consts.gamma0 = params.gamma0;
    consts.omega = params.omega;
    consts.a = params.a;
    consts.a2 = consts.a*params.a;
    consts.a3 = consts.a2*params.a;
    consts.a4 = consts.a3*params.a;
    consts.bi = 1.0 / params.b; // 1/b
    consts.bi2 = consts.bi*consts.bi; // 1/b^2
    consts.bi3 = consts.bi2*consts.bi; // 1/b^3
    consts.bi4 = consts.bi3*consts.bi; // 1/b^4
    std::cout << consts.a << " " << consts.a2 << " " << consts.a3 << " " << consts.a4 << endl;
    std::cout << consts.bi << " " << consts.bi2 << " " << consts.bi3 << " " << consts.bi4 << endl;
    consts.f = params.f;
    consts.g = params.g;
    consts.f1 = params.f1;
    consts.g1 = params.g1;
    // RyR rates
    consts.k12 = params.k12;
    consts.k21 = params.k21;
    consts.k23 = params.k23;
    consts.k32 = params.k32;
    consts.k34 = params.k34;
    consts.k43 = params.k43;
    consts.k45 = params.k45;
    consts.k54 = params.k54;
    consts.k56 = params.k56;
    consts.k65 = params.k65;
    consts.k25 = params.k25;
    consts.k52 = params.k52;
    // ClCh rates
    consts.kfClCh = params.kfClCh;
    consts.kbClCh = params.kbClCh;
    // CaSS constants
    consts.KBSR = params.KBSR;
    consts.BSR_const = params.KBSR * params.BSRT;
    consts.KBSL = params.KBSL;
    consts.BSL_const = params.KBSL * params.BSLT;
    
    consts.VSS_VJSR = params.VSS / params.VJSR;
    consts.KCSQN = params.KCSQN;
    consts.CSQN_const = params.KCSQN * params.CSQNT;
    // JLCC constants
    consts.JLCC_const = 2.0e6 * params.PCaL / params.VSS;
    consts.Cao_scaled = 0.341 * params.Cao;
    // Jrel constants
    consts.rRyR = params.rRyR;
    // Jtr constants
    consts.rtr = params.rtr;
    // INaCa consts
    consts.Nao3 = params.Nao*params.Nao*params.Nao;
    consts.INaCa_const = 5000.0 * params.kNaCa / ((params.KmNa*params.KmNa*params.KmNa + params.Nao*params.Nao*params.Nao) * (params.KmCa + params.Cao));
    // INaK coonsts
    consts.sigma = (exp(params.Nao / 67.3) - 1.0) / 7.0;
    consts.INaK_const = params.INaKmax * params.Ko / (params.Ko + params.KmKo);
    // IKr consts
    consts.sqrtKo = sqrt(params.Ko);
    // Ito1 consts
    consts.PKv14_Csc = params.PKv14 / params.Csc;
    // Ito2 consts
    consts.Ito2_const = 1.0e9 * params.Pto2 * F * (double(params.NCaRU) / double(nCRU_simulated)) / params.CSA;
    // IK1 consts
    consts.IK1_const = params.Ko / (params.Ko + params.KmK1);
    // ICaL consts
    consts.ICaL_const = -1000.0 * (2.0*F * params.VSS) * (double(params.NCaRU) / double(nCRU_simulated)) / params.CSA;
    // CMDN_consts
    consts.CMDN_const = params.KCMDN * params.CMDNT;

    return consts;
}


void update_Kr_derivative(double* const deriv, const double* const state, const NDArrayMap<double,2> &Q, const double dt){
    deriv[0] = dt*(Q(1,0)*state[1] + Q(0,0)*state[0]);
    deriv[1] = dt*(Q(0,1)*state[0] + Q(2,1)*state[2] + Q(1,1)*state[1]);
    deriv[2] = dt*(Q(1,2)*state[1] + Q(3,2)*state[3] + Q(4,2)*state[4] + Q(2,2)*state[2]);
    deriv[3] = dt*(Q(2,3)*state[2] + Q(4,3)*state[4] + Q(3,3)*state[3]);
    deriv[4] = dt*(Q(2,4)*state[2] + Q(3,4)*state[3] + Q(4,4)*state[4]);
}

void update_Kv_derivative(double* const deriv, const double* const state, const NDArrayMap<double,2> &Q, const double dt){
    deriv[0] = dt*(Q(1,0)*state[1] + Q(5,0)*state[5] + Q(0,0)*state[0]);
    deriv[1] = dt*(Q(0,1)*state[0] + Q(2,1)*state[2] + Q(6,1)*state[6] + Q(1,1)*state[1]);
    deriv[2] = dt*(Q(1,2)*state[1] + Q(3,2)*state[3] + Q(7,2)*state[7] + Q(2,2)*state[2]);
    deriv[3] = dt*(Q(2,3)*state[2] + Q(4,3)*state[4] + Q(8,3)*state[8] + Q(3,3)*state[3]);
    deriv[4] = dt*(Q(3,4)*state[3] + Q(9,4)*state[9] + Q(4,4)*state[4]);
    
    deriv[5] = dt*(Q(6,5)*state[6] + Q(0,5)*state[0] + Q(5,5)*state[5]);
    deriv[6] = dt*(Q(5,6)*state[5] + Q(7,6)*state[7] + Q(1,6)*state[1] + Q(6,6)*state[6]);
    deriv[7] = dt*(Q(6,7)*state[6] + Q(8,7)*state[8] + Q(2,7)*state[2] + Q(7,7)*state[7]);
    deriv[8] = dt*(Q(7,8)*state[7] + Q(9,8)*state[9] + Q(3,8)*state[3] + Q(8,8)*state[8]);
    deriv[9] = dt*(Q(8,9)*state[8] + Q(4,9)*state[4] + Q(9,9)*state[9]);
}

void update_QKr(NDArrayMap<double,2> &Q, const double V, const GW_parameters &params){
    Q(0,1) = 0.0069*exp(0.0272*V);
    Q(0,0) = -Q(0,1);

    Q(1,0) = 0.0227*exp(-0.0431*V);
    Q(1,2) = params.Kf;
    Q(1,1) = -(Q(1,0) + Q(1,2));

    Q(2,1) = params.Kb;
    Q(2,3) = 0.0218*exp(0.0262*V);
    Q(2,4) = 1.29e-5*exp(2.71e-6 * V);
    Q(2,2) = -(Q(2,1) + Q(2,3) + Q(2,4));    

    Q(3,2) = 0.0009*exp(-0.0269*V);
    Q(3,4) = 0.0622*exp(0.0120*V);
    Q(3,3) = -(Q(3,2) + Q(3,4));

    Q(4,3) = 0.0059*exp(-0.0443*V);
    Q(4,2) = Q(3,2)*Q(4,3)*Q(2,4)/(Q(2,3)*Q(3,4));
    Q(4,4) = -(Q(4,2) + Q(4,3));
}

void update_QKv(NDArrayMap<double,2> &Q, const double V, const double alphaa0, const double aa, const double alphai0, const double ai, 
                const double betaa0, const double ba, const double betai0, const double bi, const double f1, const double f2,
                const double f3, const double f4, const double b1, const double b2, const double b3, const double b4)
{
    const double alphaa = alphaa0 * exp(aa * V);
    const double alphai = alphai0 * exp(-ai * V);
    const double betaa = betaa0 * exp(-ba * V);
    const double betai = betai0 * exp(bi * V);

    Q(0,1) = 4.0*alphaa;
    Q(0,5) = betai;
    Q(0,0) = -(Q(0,1)+Q(0,5));

    Q(1,0) = betaa;
    Q(1,2) = 3.0*alphaa;
    Q(1,6) = f1*betai;
    Q(1,1) = -(Q(1,0) + Q(1,2) + Q(1,6));
    
    Q(2,1) = 2.0*betaa;
    Q(2,3) = 2.0*alphaa;
    Q(2,7) = f2*betai;
    Q(2,2) = -(Q(2,1) + Q(2,3) + Q(2,7));
    
    Q(3,2) = 3.0*betaa;
    Q(3,4) = alphaa;
    Q(3,8) = f3*betai;
    Q(3,3) = -(Q(3,2) + Q(3,4) + Q(3,8));
    
    Q(4,3) = 4.0*betaa;
    Q(4,9) = f4*betai;
    Q(4,4) = -(Q(4,3) + Q(4,9));
    
    Q(5,6) = 4.0*alphaa*b1;
    Q(5,0) = alphai;
    Q(5,5) = -(Q(5,6) + Q(5,0));

    Q(6,5) = betaa / f1;
    Q(6,7) = 3.0*alphaa * b2/b1;
    Q(6,1) = alphai / b1;
    Q(6,6) = -(Q(6,5) + Q(6,7) + Q(6,1));
    
    Q(7,6) = 2.0*betaa * f1/f2;
    Q(7,8) = 2.0*alphaa * b3/b2;
    Q(7,2) = alphai / b2;
    Q(7,7) = -(Q(7,6) + Q(7,8) + Q(7,2));
    
    Q(8,7) = 3.0*betaa * f2/f3;
    Q(8,9) = alphaa * b4/b3;
    Q(8,3) = alphai/b3;
    Q(8,8) = -(Q(8,7) + Q(8,9) + Q(8,3));
    
    Q(9,8) = 4.0*betaa * f3/f4;
    Q(9,4) = alphai/b4;
    Q(9,9) = -(Q(9,8) + Q(9,4));
}


void initialise_QKv14(NDArrayMap<double,2> &Q, const GW_parameters &params){
    Q.set_to_zeros();
    Q.set(params.Kf,1,2);
    Q.set(params.Kb,2,1);
}


double update_rates(const int* const, const int* const, const int* const, const int* const, double* const, double* const, double* const, double* const, const double* const, double*, const double, const double, const double, const double, const Constants&);
void update_fluxes(const double* const, const double, double*, double*, const Constants&);
void update_CaSS(double* const, int*, const double* const, const double* const, const double* const, const double* const, const double, const Constants&);
void update_state(int* const, int* const, int* const, double* const, int* const, const double* const, const double* const, const double* const, const double* const, const double* const, const int, const double* const, double*, double*, const double, const double, const double, const Constants&);
void sample_LCC(int* const, const double* const, const double, const int* const, const double* const, double*, const int, const double, const double, const Constants&);
void sample_RyR(int* const, double* const, const double* const, const double, double*, const int, const double* const, const double ,  const Constants&);
void SSA_subunit(int* const, int* const, int* const, double* const, int* const, double* const, double* const, double* const, double* const, double* const, double* const, double* const, double* const, double* const, double* const, double, const double, const double, const double, const double, const double, const double, const double, const double, const double, double* const, const Constants&);
void SSA(NDArray<int,2>&, NDArray<int,2>&, NDArray<int,3>&, NDArray<int,2>&, NDArray<double,2>&, NDArray<double,1>&, const double, const double, NDArray<double,2>&, NDArray<double,2>&, NDArray<double,1>&, const double, const double, const double, const int, const Constants&);

double urand(){
    static thread_local std::random_device rd;
    static thread_local std::mt19937_64 gen(rd());
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}

int sample_binomial(const double p, const int N){
    int X = 0;
    double u;
    for (int i = 0; i < N; i++){
        u = urand();
        if (u < p){
            X++;
        }
    }
    return X;
}

inline double square(const double x) { return x*x; }

#endif