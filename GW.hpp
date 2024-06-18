#ifndef _GW_H
#define _GW_H

#include <iostream>
#include <array>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>
#include <ctime>
#include <random>


using namespace std;
using VectorMap = Eigen::Map<Eigen::VectorXd>;

using MatrixX4d = Eigen::Matrix<double,Eigen::Dynamic,4,Eigen::RowMajor>;
using MatrixX4i = Eigen::Matrix<int,Eigen::Dynamic,4,Eigen::RowMajor>;
using MatrixMap = Eigen::Map<MatrixX4d>;
using MatrixMapi = Eigen::Map<MatrixX4i>;

using Array3dMap = Eigen::TensorMap<Eigen::Tensor<double,3>>;
using Array3dMapi = Eigen::TensorMap<Eigen::Tensor<int,3>>;
using Array4dMap = Eigen::TensorMap<Eigen::Tensor<double,4>>;

using GeneratorKr = Eigen::Map<Eigen::Matrix<double,5,5>>;
using GeneratorKv = Eigen::Map<Eigen::Matrix<double, 10, 10>>;

const double F = 96.5;
const double T = 310.;
const double R = 8.314;
const double FRT = F / (R*T);

// This holds the model parameters with defaults specified.
typedef struct {
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
    double CSQNT = 1.124;
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
    double alphai0Kv14 = 0.0;
    double aiKv14 = 1.0571e-4;
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
    double kHTRPNm = 66.0e-6;
    double kLTRPNp = 40.0;
    double kLTRPNm = 0.04;
    double HTRPNtot = 140.0e-3;
    double LTRPNtot = 70.0e-3;
    double Vmaxf = 0.0002096;
    double Vmaxr = 0.0002096;
    double Kmf = 0.000260;
    double Kmr = 1.8;
    double Hf = 0.75;
    double Hr = 0.75;
} GW_parameters;

// This holds constant values when the simulation is executed, preventing unnecessary recalculations.
typedef struct {
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
} Constants;

Constants consts_from_params(const GW_parameters &params)
{
    Constants consts;
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

    return consts;
}


double update_rates(const int* const, const int* const, const int* const, const int* const, double* const, double* const, double* const, double* const, const double* const, double*, const double, const double, const double, const double, const int, const Constants&);
void update_fluxes(const double* const, const double, double*, double*,const int, const Constants&);
void update_CaSS(double* const, int*, const double* const, const double* const, const double* const, const double* const, const double, const int, const Constants&);
void update_state(int* const, int* const, int* const, double* const, int* const, const double* const, const double* const, const double* const, const double* const, const double* const, const int, const int, const double* const, double*, double*, const double, const double, const double, const Constants&);
void sample_LCC(int* const, const double* const, const double, const int* const, const double* const, double*, const int, const int, const double, const double, const Constants&);
void sample_RyR(int* const, double* const, const double* const, const double, double*, const int, const int, const double* const, const double ,  const Constants&);
void SSA_subunit(MatrixMapi&, MatrixMapi&, Array3dMapi&, MatrixMapi&, MatrixMap&, VectorMap&, const double, const double, MatrixMap&, MatrixMap&, VectorMap&, const double, const double, const double, const double, const double, const double, const double, const int, const Constants&);
void SSA(MatrixMapi&, MatrixMapi&, Array3dMapi&, MatrixMapi&, MatrixMap&, VectorMap&, const double, const double, MatrixMap&, MatrixMap&, VectorMap&, const double, const double, const double, const int, const Constants&);

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