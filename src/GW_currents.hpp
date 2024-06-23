#ifndef CURRENTS_H
#define CURRENTS_H

//#include "GW.hpp"
#include "ndarray.hpp"
#include "common.hpp"


// The global currents

namespace GW {

//inline double Nernst(const double Xi, const double Xo){
//    return log(Xo/Xi) / FRT;
//}

//double INa(double V, double m, double h, double j, double ENa, double GNa){
//    return GNa*(m*m*m)*h*j*(V - ENa);
//}


template <typename FloatType>
inline FloatType RKr(const FloatType V){ return 1.0 / (1.0 + 1.4945*exp(0.0446*V)); }

template <typename FloatType>
inline FloatType IKr(const FloatType V, const FloatType XKr, const FloatType EK, const FloatType GKr, const FloatType sqrtKo){
    return GKr * sqrtKo * RKr(V) * XKr * (V - EK) * 0.5;
}


template <typename FloatType>
inline FloatType EKs(const FloatType Ki, const FloatType Ko, const FloatType Nai, const FloatType Nao){
    return log((Ko+0.01833*Nao) / (Ki + 0.01833*Nai)) / FRT;
}


template <typename FloatType>
inline FloatType IKs(const FloatType V, const FloatType XKs, const FloatType Ki, const FloatType Nai, const FloatType Nao, const FloatType Ko, const FloatType GKs){
    return GKs * (XKs*XKs) * (V - EKs(Ki, Ko, Nai, Nao));
}


template <typename FloatType>
inline FloatType IKv43(const FloatType V, const FloatType XKv43, const FloatType EK, const FloatType GKv43){
    return GKv43 * XKv43 * (V - EK);
}


template <typename FloatType>
inline FloatType IKv14(const FloatType VFRT, const FloatType exp_term, const FloatType XKv14, const FloatType Ki, const FloatType Nai, const FloatType PKv14_Csc, const FloatType Nao, const FloatType Ko){
    FloatType m = (PKv14_Csc) * (F*VFRT) * XKv14 / (1 - exp_term); // PKv14_Csc = PKv14 / Csc
    return m * ((Ki - Ko*exp_term) + 0.02*(Nai - Nao*exp_term)) * 1.0e9; // 1e9 required to convert to mV / ms
}
//double Ito1(double V, double VFRT, double expmVFRT, double XKv14, double XKv43, double Ki, double Nai, double EK, double PKv14_Csc, double Nao, double Ko, double GKv43){
//    return IKv14(VFRT, expmVFRT, XKv14, Ki, Nai, PKv14_Csc, Nao, Ko) + IKv43(V, XKv43, EK, GKv43);
//}


template <typename FloatType>
inline FloatType K1inf(const FloatType V, const FloatType EK){ return 1.0 / (2.0 + exp(1.5*(V-EK)*FRT)); }

template <typename FloatType>
inline FloatType IK1(const FloatType V, const FloatType EK, const FloatType GK1, const FloatType IK1_const){ return GK1 * K1inf(V, EK) * IK1_const * (V - EK); }

template <typename FloatType>
inline FloatType Kp(const FloatType V){ return 1. / (1. + exp((7.488 - V) / 5.98)); }

template <typename FloatType>
inline FloatType IKp(const FloatType V, const FloatType EK, const FloatType GKp){ return GKp * Kp(V) * (V-EK); }


//template <typename FloatType>
//inline FloatType INaCa(FloatType VFRT, FloatType expmVFRT, FloatType Nai, FloatType Cai, FloatType Nao3, FloatType Cao, FloatType eta, FloatType INaCa_const, FloatType ksat){
//    FloatType exp_term1 = exp(eta*VFRT);
//    FloatType exp_term2 = exp_term1 * expmVFRT;
//    return INaCa_const * (exp_term1*(Nai*Nai*Nai)*Cao - exp_term2*Nao3*Cai) / (1.0 + ksat*exp_term2);
//}



//template <typename FloatType>
//inline FloatType fNaK(FloatType VFRT, FloatType expmVFRT, FloatType sigma){ return 1.0 / (1.0 + 0.1245*exp(-0.1*VFRT) + 0.0365*sigma*expmVFRT); }
//double INaK(double VFRT, double expmVFRT, double Nai, double sigma, double KmNai, double INaK_const){
    // INaK_const = INaKmax * Ko / (Ko + KmKo)
//    return INaK_const * fNaK(VFRT, expmVFRT, sigma) / (1.0 + pow(KmNai/Nai, 1.5));
//}



//double IpCa(double Cai, double IpCamax, double KmpCa){
//    return IpCamax * Cai / (KmpCa + Cai);
//}



//double ICab(double V, double Cai, double Cao, double GCab){
//    return GCab * (V - 0.5*Nernst(Cai, Cao));
//}



//double INab(double V, double ENa, double GNab){
//    return GNab * (V - ENa);
//}


template <typename FloatType>
inline FloatType ICaL(const NDArray<FloatType,2> &JLCC, const FloatType ICaL_const){
    return JLCC.sum() * ICaL_const;
}


template <typename FloatType, typename IntType>
inline FloatType Ito2(const NDArray<IntType,2> &ClCh, const FloatType VFRT, const FloatType expmVFRT, const FloatType Cl_cyto, const FloatType Clo, const FloatType Ito2_const){
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

template <typename FloatType, std::size_t N>
inline FloatType flux_average(const NDArray<FloatType,N> &flux_container, const FloatType CRU_factor){ return flux_container.sum() * CRU_factor; }

template <typename FloatType>
inline FloatType XKsinf(const FloatType V){ return 1.0 / (1.0 + exp(-(V - 24.7) / 13.6)); }

template <typename FloatType>
inline FloatType tauXKs_inv(const FloatType V){ return 0.0000719*(V-10.0)/(1.0 - exp(-0.148*(V-10.0))) + 0.000131*(V-10.0)/(exp(0.0687*(V-10.0)) - 1.0); }


}

#endif



