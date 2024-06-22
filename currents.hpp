#ifndef _CURRENTSH
#define _CURRENTSH

#include "GW.hpp"


// The global currents

namespace currents{

inline double Nernst(const double Xi, const double Xo){
    return log(Xo/Xi) / FRT;
}

double INa(double V, double m, double h, double j, double ENa, double GNa){
    return GNa*(m*m*m)*h*j*(V - ENa);
}


inline double RKr(double V){
    return 1.0 / (1.0 + 1.4945*exp(0.0446*V));
}
double IKr(double V, double XKr, double EK, double GKr, double sqrtKo){
    return GKr * sqrtKo * RKr(V) * XKr * (V - EK) * 0.5;
}



inline double EKs(double Ki, double Ko, double Nai, double Nao){
    return log((Ko+0.01833*Nao) / (Ki + 0.01833*Nai)) / FRT;
}
double IKs(double V, double XKs, double Ki, double Nai, double Nao, double Ko, double GKs){
    return GKs * (XKs*XKs) * (V - EKs(Ki, Ko, Nai, Nao));
}



inline double IKv43(double V, double XKv43, double EK, double GKv43){
    return GKv43 * XKv43 * (V - EK);
}
inline double IKv14(double VFRT, double exp_term, double XKv14, double Ki, double Nai, double PKv14_Csc, double Nao, double Ko){
    double m = (PKv14_Csc) * (F*VFRT) * XKv14 / (1 - exp_term); // PKv14_Csc = PKv14 / Csc
    return m * ((Ki - Ko*exp_term) + 0.02*(Nai - Nao*exp_term)) * 1.0e9; // 1e9 required to convert to mV / ms
}
double Ito1(double V, double VFRT, double expmVFRT, double XKv14, double XKv43, double Ki, double Nai, double EK, double PKv14_Csc, double Nao, double Ko, double GKv43){
    return IKv14(VFRT, expmVFRT, XKv14, Ki, Nai, PKv14_Csc, Nao, Ko) + IKv43(V, XKv43, EK, GKv43);
}


inline double K1inf(double V, double EK){
    return 1.0 / (2.0 + exp(1.5*(V-EK)*FRT));
}
double IK1(double V, double EK, double GK1, double IK1_const){
    // K_const = Ko / (Ko + KmK1)
    return GK1 * K1inf(V, EK) * IK1_const * (V - EK);
}

inline double Kp(double V){
    return 1. / (1. + exp((7.488 - V) / 5.98));
}
double IKp(double V, double EK, double GKp){
    return GKp * Kp(V) * (V-EK);
}



double INaCa(double VFRT, double expmVFRT, double Nai, double Cai, double Nao3, double Cao, double eta, double INaCa_const, double ksat){
    // INaCa_const = 5000*kNaCa / ((KmNa^3 + Nao^3) * (KmCa + Cao))
    double exp_term1 = exp(eta*VFRT);
    double exp_term2 = exp_term1 * expmVFRT;
    return INaCa_const * (exp_term1*(Nai*Nai*Nai)*Cao - exp_term2*Nao3*Cai) / (1.0 + ksat*exp_term2);
}



inline double fNaK(double VFRT, double expmVFRT, double sigma){
    // sigma = (exp(Nao/67.3) - 1) / 7
    return 1.0 / (1.0 + 0.1245*exp(-0.1*VFRT) + 0.0365*sigma*expmVFRT);
}
double INaK(double VFRT, double expmVFRT, double Nai, double sigma, double KmNai, double INaK_const){
    // INaK_const = INaKmax * Ko / (Ko + KmKo)
    return INaK_const * fNaK(VFRT, expmVFRT, sigma) / (1.0 + pow(KmNai/Nai, 1.5));
}



double IpCa(double Cai, double IpCamax, double KmpCa){
    return IpCamax * Cai / (KmpCa + Cai);
}



double ICab(double V, double Cai, double Cao, double GCab){
    return GCab * (V - 0.5*Nernst(Cai, Cao));
}



double INab(double V, double ENa, double GNab){
    return GNab * (V - ENa);
}



double ICaL(const NDArray<double,2> &JLCC, double ICaL_const){
    // ICaL_const = -1000. * (2F * VSS) * (NCaRU / size) / CSA
    return JLCC.sum() * ICaL_const;
}



double Ito2(const NDArray<int,2> &ClCh, double VFRT, double expmVFRT, double Cl_cyto, double Clo, double Ito2_const){
    // Ito2_const = 1e9 * Pto2 * F * (NCaRU / size) / CSA
    // expmVFRT = exp(-VFRT)
    return double(ClCh.sum()) * Ito2_const * VFRT * (Cl_cyto * expmVFRT - Clo) / (expmVFRT - 1.0);
}



double dTRPNCa(double TRPNCa, double Cai, double TRPNtot, double kTRPNp, double kTRPNm){
    // Same for HTRPN and LTRPN with different params
    return kTRPNp*Cai*(TRPNtot - TRPNCa) - kTRPNm*TRPNCa;
}

double Jup(double Cai, double CaNSR, double Vmaxf, double Vmaxr, double Kmf, double Kmr, double Hf, double Hr){
    const double f = pow(Cai/Kmf, Hf);
    const double r = pow(CaNSR/Kmr, Hr);
    return (Vmaxf*f - Vmaxr*r) / (1.0 + f + r);
}

double beta_cyto(double Cai, double CMDNconst, double KCMDN){
    // CMDNconst = CMDNT * KCMDN
    const double x = KCMDN + Cai;
    return 1.0 / (1.0 + CMDNconst / (x*x));
}

template <std::size_t N>
double flux_average(const NDArray<double,N> &flux_container, const double CRU_factor){ return flux_container.sum() * CRU_factor; }

}
    

#endif



