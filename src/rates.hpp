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

inline double alphaLCC(const double V) { return 2.0 * exp(0.012 * (V - 35.0)); }
inline double betaLCC(const double V) { return 0.0882 * exp(-0.05 * (V - 35.0)); }
inline double yinfLCC(const double V) { return 0.4 / (1.0 + exp((V + 12.5) / 5.0)) + 0.6; }
inline double tauLCC(const double V)  { return 340.0 / (1.0 + exp((V + 30.0) / 12.0)) + 60.0; }

inline void LCC_activation_rate(double* const LCC_a_rates, const int* const LCC_a, const double yinf, const double tau){
    #pragma omp simd
    for (int j = 0; j < 4; j++){
        LCC_a_rates[j] = (LCC_a[j] == 0) ? yinf / tau : (1.0 - yinf) / tau;
    }
}

inline void ClCh_rate(double* ClCh_rates, const int* const ClCh, const double* const CaSS, const double kfClCh, const double kbClCh){
    #pragma omp simd
    for (int j = 0; j < 4; j++){
        ClCh_rates[j] = (ClCh[j] == 0) ? kfClCh * CaSS[j] : kbClCh;
    }
}

void update_LCC_rates(double* LCC_rates, const int* const LCC, const double* const CaSS, const int j, const double alpha, const double beta, double* const subunit_rates, const Constants &consts){
    switch (LCC[j])
    {
    case 1:
        LCC_rates[3*j] = 4.0*alpha;
        LCC_rates[3*j+1] = consts.gamma0*CaSS[j];
        LCC_rates[3*j+2] = 0.0;
        break;
    case 2:
        LCC_rates[3*j] = beta;
        LCC_rates[3*j+1] = 3.0*alpha;
        LCC_rates[3*j+2] = consts.a*consts.gamma0*CaSS[j];
        break;
    case 3:
        LCC_rates[3*j] = 2.0*beta;
        LCC_rates[3*j+1] = 2.0*alpha;
        LCC_rates[3*j+2] = consts.a2*consts.gamma0*CaSS[j];
        break;
    case 4:
        LCC_rates[3*j] = 3.0*beta;
        LCC_rates[3*j+1] = alpha;
        LCC_rates[3*j+2] = consts.a3*consts.gamma0*CaSS[j];
        break;
    case 5:
        LCC_rates[3*j] = 4.0*beta;
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
        LCC_rates[3*j+1] = 4.0*alpha*consts.a;
        LCC_rates[3*j+2] = 0.0;
        break;
    case 8:
        LCC_rates[3*j] = consts.omega*consts.bi;
        LCC_rates[3*j+1] = beta*consts.bi;
        LCC_rates[3*j+2] = 3.0*alpha*consts.a;
        break;
    case 9:
        LCC_rates[3*j] = consts.omega*consts.bi2;
        LCC_rates[3*j+1] = 2.0*beta*consts.bi;
        LCC_rates[3*j+2] = 2.0*alpha*consts.a;
        break;
    case 10:
        LCC_rates[3*j] = consts.omega*consts.bi3;
        LCC_rates[3*j+1] = 3.0*beta*consts.bi;
        LCC_rates[3*j+2] = alpha*consts.a;
        break;
    case 11:
        LCC_rates[3*j] = consts.omega*consts.bi4;
        LCC_rates[3*j+1] = 4.0*beta*consts.bi;
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
    subunit_rates[j] += (LCC_rates[3*j] + LCC_rates[3*j+1] + LCC_rates[3*j+2]);
}


void update_RyR_rates(double* RyR_rates, const int* const RyR, const double* const CaSS, const int j, double* const subunit_rates, const Constants &consts){
    // Rates correspond to  [(1,2), (2,3), (2,5), (3,4), (4,5), (5,6), (2,1), (3,2), (4,3), (5,2), (5,4), (6,5)]
    const double CaSS2 = CaSS[j]*CaSS[j];
    const double eq56 = consts.k65 / (consts.k56*CaSS2 + consts.k65);
    const double tau34 = 1.0 / (consts.k34*CaSS2 + consts.k43);

    assert((RyR[6*j] + RyR[6*j+1] + RyR[6*j+2] + RyR[6*j+3] + RyR[6*j+4] + RyR[6*j+5]) == 5);

    RyR_rates[12*j] = double(RyR[6*j])*consts.k12*CaSS2; // 1 -> 2
    RyR_rates[12*j+1] = double(RyR[6*j+1])*consts.k23*CaSS2; // 2 -> 3
    RyR_rates[12*j+2] = double(RyR[6*j+1])*consts.k25*CaSS2; // 2 -> 5
    RyR_rates[12*j+3] = CaSS[j] > 3.685e-2 ? 0.0 : double(RyR[6*j+2])*consts.k34*CaSS2; // 3 -> 4
    RyR_rates[12*j+4] = CaSS[j] > 3.685e-2 ? double(RyR[6*j+2]+RyR[6*j+3])*consts.k45*consts.k34*CaSS2*tau34 : double(RyR[6*j+3])*consts.k45; // 4 -> 5
    RyR_rates[12*j+5] = CaSS[j] > 1.15e-4 ? 0.0 : double(RyR[6*j+4])*consts.k56*CaSS2; // 5 -> 6
    RyR_rates[12*j+6] = double(RyR[6*j+1])*consts.k21; // 2 -> 1
    RyR_rates[12*j+7] = CaSS[j] > 3.685e-2 ? double(RyR[6*j+2]+RyR[6*j+3])*consts.k32*consts.k43*tau34 : double(RyR[6*j+2])*consts.k32; // 3 -> 2
    RyR_rates[12*j+8] = CaSS[j] > 3.685e-2 ? 0.0 : double(RyR[6*j+3])*consts.k43; // 4 -> 3
    RyR_rates[12*j+9] = CaSS[j] > 1.15e-4 ? double(RyR[6*j+4]+RyR[6*j+5])*consts.k52*eq56 : double(RyR[6*j+4])*consts.k52; // 5 -> 2
    RyR_rates[12*j+10] = CaSS[j] > 1.15e-4 ? double(RyR[6*j+4]+RyR[6*j+5])*consts.k54*CaSS2*eq56 : double(RyR[6*j+4])*consts.k54*CaSS2; // 5 -> 4
    RyR_rates[12*j+11] = CaSS[j] > 1.15e-4 ? 0.0 : double(RyR[6*j+5])*consts.k65; // 6 -> 5
    
    for (int k = 0; k < 12; k++){
        subunit_rates[j] += RyR_rates[12*j+k];
    }
}