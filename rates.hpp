#include "GW.hpp"

inline double alpham(const double V){
    return V == 47.13 ? 3.2 : 0.32 * (V + 47.13) / (1.0 - exp(-0.1*(V + 47.13)));
}

inline double betam(const double V){
    return 0.08 * exp(-V/11);
}



inline double alphah(const double V){
    return V >= -40. ? 0.0 : 0.135 * exp(-(80 + V) / 6.8);
}

inline double betah(const double V){
    return V >= -40. ? 1.0 / (0.13 * (1 + exp(-(V + 10.66) / 11.1))) : 3.56 * exp(0.079*V) + 3.1e5 * exp(0.35*V);
}



inline double alphaj(const double V){
    return V >= -40. ? 0. : (-127140.0 * exp(0.2444*V) -3.474e-5 * exp(-0.04391*V)) * (V + 37.78) / (1 + exp(0.311*(V + 79.23)));
}

inline double betaj(const double V){
    return V >= -40. ? 0.3 * exp(-2.535e-7 * V) / (1 + exp(-0.1*(V + 32.0))) : 0.1212 * exp(-0.01052*V) / (1.0 + exp(-0.1378*(V + 40.14)));
}

inline double XKsinf(const double V){
    return 1.0 / (1.0 + exp(-(V - 24.7) / 13.6));
}

inline double tauXKs(const double V){
    return 1.0 / (0.0000719*(V-10.0)/(1.0 - exp(-0.148*(V-10.0))) + 0.000131*(V-10.0)/(exp(0.0687*(V-10.0)) - 1.0));
}

// Make sure to set Q(2,3) and Q(3,2) ahead of time
/*
void update_QKrHERG(GeneratorKr &Q, const double V){
    Q(1,2) = 0.0069*exp(0.0272*V);
    Q(1,1) = -Q(1,2);

    Q(2,1) = 0.0227*exp(-0.0431*V);
    //Q(2,3) = Kf;
    Q(2,2) = -(Q(2,1) + Q(2,3));

    //Q(3,2) = Kb;
    Q(3,4) = 0.0218*exp(0.0262*V);
    Q(3,5) = 1.29e-5*exp(2.71e-6 * V);
    Q(3,3) = -(Q(3,2) + Q(3,4) + Q(3,5));

    Q(4,3) = 0.0009*exp(-0.0269*V);
    Q(4,5) = 0.0622*exp(0.0120*V);
    Q(4,4) = -(Q(4,3) + Q(4,5));

    Q(5,4) = 0.0059 * exp(-0.0443*V);
    Q(5,3) = Q(4,3)*Q(5,4)*Q(3,5)/(Q(3,4)*Q(4,5));
    Q(5,5) = -(Q(5,3) + Q(5,4));
}
*/

/*
void update_QKrHERGhKCNE2(GeneratorKr &Q, const double V){
    Q(1,2) = 0.0171*exp(0.0330*V);
    Q(1,1) = -Q(1,2);

    Q(2,1) = 0.0397*exp(-0.0431*V);
    //Q(2,3) = Kf * 0.98;
    Q(2,2) = -(Q(2,1) + Q(2,3));

    //Q(3,2) = Kb * 1.1;
    Q(3,4) = 0.0206*exp(0.0262*V);
    Q(3,5) = 8.04e-5*exp(6.98e-7 * V);
    Q(3,3) = -(Q(3,2) + Q(3,4) + Q(3,5));

    Q(4,3) = 0.0013*exp(-0.0269*V);
    Q(4,5) = 0.1067*exp(0.0057*V);
    Q(4,4) = -(Q(4,3) + Q(4,5));

    Q(5,4) = 0.0065 * exp(-0.0454*V);
    Q(5,3) = Q(4,3)*Q(5,4)*Q(3,5)/(Q(3,4)*Q(4,5));
    Q(5,5) = -(Q(5,3) + Q(5,4));
}
*/

void update_QKv(GeneratorKv &Q, const double V, const double alphaa0, const double aa, const double alphai0, const double ai, 
                const double betaa0, const double ba, const double betai0, const double bi, const double f1, const double f2,
                const double f3, const double f4, const double b1, const double b2, const double b3, const double b4)
{
    const double alphaa = alphaa0 * exp(aa * V);
    const double alphai = alphai0 * exp(-ai * V);
    const double betaa = betaa0 * exp(-ba * V);
    const double betai = betai0 * exp(bi * V);

    Q(1,2) = 4*alphaa;
    Q(1,6) = betai;
    Q(1,1) = -(Q(1,2)+Q(1,6));

    Q(2,1) = betaa;
    Q(2,3) = 3*alphaa;
    Q(2,7) = f1 *betai;
    Q(2,2) = -(Q(2,1) + Q(2,3) + Q(2,7));
    
    Q(3,2) = 2*betaa;
    Q(3,4) = 2*alphaa;
    Q(3,8) = f2*betai;
    Q(3,3) = -(Q(3,2) + Q(3,4) + Q(3,8));
    
    Q(4,3) = 3*betaa;
    Q(4,5) = alphaa;
    Q(4,9) = f3*betai;
    Q(4,4) = -(Q(4,3) + Q(4,5) + Q(4,9));
    
    Q(5,4) = 4*betaa;
    Q(5,10) = f4*betai;
    Q(5,5) = -(Q(5,4) + Q(5,10));
    
    Q(6,7) = 4*alphaa * b1;
    Q(6,1) = alphai;
    Q(6,6) = -(Q(6,7)+Q(6,1));

    Q(7,6) = betaa / f1;
    Q(7,8) = 3*alphaa * b2/b1;
    Q(7,2) = alphai / b1;
    Q(7,7) = -(Q(7,6) + Q(7,8) + Q(7,2));
    
    Q(8,7) = 2*betaa * f1/f2;
    Q(8,9) = 2*alphaa * b3/b2;
    Q(8,3) = alphai / b2;
    Q(8,8) = -(Q(8,7) + Q(8,9) + Q(8,3));
    
    Q(9,8) = 3*betaa * f2/f3;
    Q(9,10) = alphaa * b4/b3;
    Q(9,4) = alphai/b3;
    Q(9,9) = -(Q(9,8) + Q(9,10) + Q(9,4));
    
    Q(10,9) = 4*alphaa * f3/f4;
    Q(10,5) = alphai/b4;
    Q(10,10) = -(Q(10,9) + Q(10,5));
}

inline double alphaLCC(const double V) { return 2.0 * exp(0.012 * (V - 35.0)); }
inline double betaLCC(const double V) { return 0.0882 * exp(-0.05 * (V - 35.0)); }
inline double yinfLCC(const double V) { return 0.4 / (1.0 + exp((V + 12.5) / 5.0)) + 0.6; }
inline double tauLCC(const double V)  { return 340.0 / (1.0 + exp((V + 30.0) / 12.0)) + 60.0; }

inline void LCC_activation_rate(double* LCC_a_rates, const int* const LCC_a, const double yinf, const double tau){
    #pragma omp simd
    for (int j = 0; j < 4; j++){
        LCC_a_rates[j] = (LCC_a[j] == 0) ? yinf / tau : (1.0 - yinf) / tau;
    }
}
inline void ClCh_rate(double* ClCh_rates, const int* const ClCh, const double* const CaSS, const double kfClCh, const double kbClCh, const int i){
    #pragma omp simd
    for (int j = 0; j < 4; j++){
        ClCh_rates[j] = (ClCh[j] == 0) ? kfClCh * CaSS[j] : kbClCh;
    }
}

void update_LCC_rates(double* LCC_rates, const int* const LCC, const double* const CaSS, const int i, const int j, const double alpha, const double beta, const Constants &consts){
    switch (LCC[j])
    {
    case 1:
        LCC_rates[3*j] = 4*alpha;
        LCC_rates[3*j+1] = consts.gamma0*CaSS[j];
        LCC_rates[3*j+2] = 0.0;
        break;
    case 2:
        LCC_rates[3*j] = beta;
        LCC_rates[3*j+1] = 3*alpha;
        LCC_rates[3*j+2] = consts.a*consts.gamma0*CaSS[j];
        break;
    case 3:
        LCC_rates[3*j] = 2*beta;
        LCC_rates[3*j+1] = 2*alpha;
        LCC_rates[3*j+2] = consts.a2*consts.gamma0*CaSS[j];
        break;
    case 4:
        LCC_rates[3*j] = 3*beta;
        LCC_rates[3*j+1] = alpha;
        LCC_rates[3*j+2] = consts.a3*consts.gamma0*CaSS[j];
        break;
    case 5:
        LCC_rates[3*j] = 4*beta;
        LCC_rates[3*j+1] = consts.f;
        LCC_rates[3*j+2] = consts.a4*consts.gamma0*CaSS[j];
        break;
    case 6:
        LCC_rates[3*j] = consts.g;
        LCC_rates[3*j+1] = 0.0;
        LCC_rates[3*j+2] = 0.0;
        break;
    case 7:
        LCC_rates[3*j] = consts.omega;
        LCC_rates[3*j+1] = 4*alpha*consts.a;
        LCC_rates[3*j+2] = 0.0;
        break;
    case 8:
        LCC_rates[3*j] = consts.omega*consts.bi;
        LCC_rates[3*j+1] = beta*consts.bi;
        LCC_rates[3*j+2] = 3*alpha*consts.a;
        break;
    case 9:
        LCC_rates[3*j] = consts.omega*consts.bi2;
        LCC_rates[3*j+1] = 2*beta*consts.bi;
        LCC_rates[3*j+2] = 2*alpha*consts.a;
        break;
    case 10:
        LCC_rates[3*j] = consts.omega*consts.bi3;
        LCC_rates[3*j+1] = 3*beta*consts.bi;
        LCC_rates[3*j+2] = alpha*consts.a;
        break;
    case 11:
        LCC_rates[3*j] = consts.omega*consts.bi4;
        LCC_rates[3*j+1] = 4*beta*consts.bi;
        LCC_rates[3*j+2] = consts.f1;
        break;
    case 12:
        LCC_rates[3*j] = consts.g1;
        LCC_rates[3*j+1] = 0.0;
        LCC_rates[3*j+2] = 0.0;
        break;
    default:
        break;
    }    
}
/*
double update_LCC_rates(Array3dMap &LCC_rates, const Array3dMap &LCC, const MatrixMap &CaSS, const int i, const int j, const double alpha, const double beta, const Constants &consts){
    if (LCC(i,j,0) == 1.0) {
        LCC_rates(i,j,1) = 4*alpha;
        LCC_rates(i,j,6) = consts.gamma0*CaSS(i,j);
        return LCC_rates(i,j,1) + LCC_rates(i,j,6);
    } else if (LCC(i,j,1) == 1.0){
        LCC_rates(i,j,0) = beta;
        LCC_rates(i,j,2) = 3*alpha;
        LCC_rates(i,j,7) = consts.a*consts.gamma0*CaSS(i,j);
        return LCC_rates(i,j,0) + LCC_rates(i,j,2) + LCC_rates(i,j,7);
    } else if (LCC(i,j,2) == 1.0){
        LCC_rates(i,j,1) = 2*beta;
        LCC_rates(i,j,3) = 2*alpha;
        LCC_rates(i,j,8) = consts.a2*consts.gamma0*CaSS(i,j);
        return LCC_rates(i,j,1) + LCC_rates(i,j,3) + LCC_rates(i,j,8);
    } else if (LCC(i,j,3) == 1.0){
        LCC_rates(i,j,2) = 3*beta;
        LCC_rates(i,j,4) = alpha;
        LCC_rates(i,j,9) = consts.a3*consts.gamma0*CaSS(i,j);
        return LCC_rates(i,j,2) + LCC_rates(i,j,4) + LCC_rates(i,j,9);
    } else if (LCC(i,j,4) == 1.0) {
        LCC_rates(i,j,3) = 4*beta;
        LCC_rates(i,j,5) = consts.f;
        LCC_rates(i,j,10) = consts.a4*consts.gamma0*CaSS(i,j);
        return LCC_rates(i,j,3) + LCC_rates(i,j,5) + LCC_rates(i,j,10);
    } else if (LCC(i,j,5) == 1.0) {
        LCC_rates(i,j,4) = consts.g;
        return LCC_rates(i,j,4);
    } else if (LCC(i,j,6) == 1.0) {
        LCC_rates(i,j,7) = 4*alpha*consts.a;
        LCC_rates(i,j,0) = consts.omega;
        return LCC_rates(i,j,0) + LCC_rates(i,j,7);
    } else if (LCC(i,j,7) == 1.0) {
        LCC_rates(i,j,8) = 3*alpha*consts.a;
        LCC_rates(i,j,6) = beta*consts.bi;
        LCC_rates(i,j,1) = consts.omega*consts.bi;
        return LCC_rates(i,j,1) + LCC_rates(i,j,6) + LCC_rates(i,j,8);
    } else if (LCC(i,j,8) == 1.0) {
        LCC_rates(i,j,9) = 2*alpha*consts.a;
        LCC_rates(i,j,7) = 2*beta*consts.bi;
        LCC_rates(i,j,2) = consts.omega*consts.bi2;
        return LCC_rates(i,j,2) + LCC_rates(i,j,7) + LCC_rates(i,j,9);
    } else if (LCC(i,j,9) == 1.0) {
        LCC_rates(i,j,10) = alpha*consts.a;
        LCC_rates(i,j,8) = 3*beta*consts.bi;
        LCC_rates(i,j,3) = consts.omega*consts.bi3;
        return LCC_rates(i,j,3) + LCC_rates(i,j,8) + LCC_rates(i,j,10);
    } else if (LCC(i,j,10) == 1.) {
        LCC_rates(i,j,11) = consts.f1;
        LCC_rates(i,j,9) = 4*beta*consts.bi;
        LCC_rates(i,j,4) = consts.omega*consts.bi4;
        return LCC_rates(i,j,4) + LCC_rates(i,j,9) + LCC_rates(i,j,11);
    } else {
        LCC_rates(i,j,10) = consts.g1;
        return LCC_rates(i,j,10);
    }
}
*/

double update_RyR_rates(double* RyR_rates, const int* const RyR, const double* const CaSS, const int i, const int j, const Constants &consts){
    // Rates correspond to  [(1,2), (2,3), (2,5), (3,4), (4,5), (5,6), (2,1), (3,2), (4,3), (5,2), (5,4), (6,5)]
    const double CaSS2 = CaSS[j]*CaSS[j];
    const double eq56 = consts.k65 / (consts.k56*CaSS2 + consts.k65);
    const double tau34 = 1.0 / (consts.k34*CaSS2 + consts.k43);
    double sum_rates = 0.0;

    RyR_rates[0] = RyR[0]*consts.k12*CaSS2; // 1 -> 2
    RyR_rates[1] = RyR[1]*consts.k23*CaSS2; // 2 -> 3
    RyR_rates[2] = RyR[1]*consts.k25*CaSS2; // 2 -> 5
    RyR_rates[3] = CaSS[j] > 3.685e-2 ? 0.0 : RyR[2]*consts.k34*CaSS2; // 3 -> 4
    RyR_rates[4] = CaSS[j] > 3.685e-2 ? (RyR[2]+RyR[3])*consts.k45*consts.k34*CaSS2*tau34 : RyR[3]*consts.k45; // 4 -> 5
    RyR_rates[5] = CaSS[j] > 1.15e-4 ? 0.0 : RyR[4]*consts.k56*CaSS2; // 5 -> 6
    RyR_rates[6] = RyR[1]*consts.k21; // 2 -> 1
    RyR_rates[7] = CaSS[j] > 3.685e-2 ? (RyR[2]+RyR[3])*consts.k32*tau34 : RyR[2]*consts.k32; // 3 -> 2
    RyR_rates[8] = CaSS[j] > 3.685e-2 ? 0.0 : RyR[3]*consts.k43; // 4 -> 3
    RyR_rates[9] = CaSS[j] > 1.15e-4 ? (RyR[4]+RyR[5])*consts.k52*eq56 : RyR[4]*consts.k52; // 5 -> 2
    RyR_rates[10] = CaSS[j] > 1.15e-4 ? (RyR[4]+RyR[5])*consts.k54*CaSS2*eq56 : RyR[4]*consts.k54*CaSS2; // 5 -> 4
    RyR_rates[11] = CaSS[j] > 1.15e-4 ? 0.0 : RyR[5]*consts.k65; // 6 -> 5
    
    for (int k = 0; k < 12; k++){
        sum_rates += RyR_rates[k];
    }
    return sum_rates;
}