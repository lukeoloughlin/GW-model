#ifndef _CURRENTSH
#define _CURRENTSH

#include "GW.hpp"


// The global currents

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




/*
void update_Jiss(MatrixMap &Jiss, const MatrixMap &CaSS, const double riss, const int size){
    #pragma omp simd
    for (int i = 0; i < size; i++){
        Jiss(i,0) = riss * ((CaSS(i,1) + CaSS(i,3)) - (2 * CaSS(i,0)));
        Jiss(i,1) = riss * ((CaSS(i,2) + CaSS(i,0)) - (2 * CaSS(i,1)));
        Jiss(i,2) = riss * ((CaSS(i,3) + CaSS(i,1)) - (2 * CaSS(i,2)));
        Jiss(i,3) = riss * ((CaSS(i,0) + CaSS(i,2)) - (2 * CaSS(i,3)));
    }
}

void update_Jxfer(MatrixMap &Jxfer, const MatrixMap &CaSS, const double Cai, const double rxfer, const int size){
    #pragma omp simd collapse(2)
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < size; j++){
            Jxfer(j,i) = rxfer * (CaSS(j,i) - Cai);
        }
    }
}


void update_JLCC(MatrixMap &JLCC, const Array3dMap &LCC, const MatrixMap &LCC_a, const MatrixMap &CaSS, const double VFRT, const double expVFRT, const double JLCC_const1, const double JLCC_const2, const int size){
    // JLCC_const1 = 1e6 * PCaL / VSS
    // JLCC_const2 = 0.341*Cao
    const double exp_term = expVFRT * expVFRT;
    const double m = -2.0 * VFRT * JLCC_const1 / (exp_term-1.0);
    #pragma omp simd collapse(2)
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < size; j++){
            JLCC(j,i) = m * (LCC_a(j,i) * (LCC(j,i,5) + LCC(j,i,11)) * ((exp_term * CaSS(j,i)) - JLCC_const2));
        }
    }
}

void update_Jrel(MatrixMap &Jrel, const Array3dMap &RyR, const VectorMap &CaJSR, const MatrixMap &CaSS, const double rRyR, const int size){
    #pragma omp simd collapse(2)
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < size; j++){
            Jrel(j,i) = rRyR * (RyR(j,i,3) + RyR(j,i,4)) * (CaJSR(j) - CaSS(j,i));
        }
    }
}

void update_Jtr(VectorMap &Jtr, const VectorMap &CaJSR, const double CaNSR, const double rtr, const int size){
    #pragma omp simd collapse(2)
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < size; j++){
            Jtr(j,i) = rtr * (CaNSR - CaJSR(j,i));
        }
    }
}

void update_betaSS(MatrixMap &betaSS, const MatrixMap &CaSS, const double KBSL, const double KBSR, const double BSR_const, const double BSL_const, const int size){
    // BSR_const = KBSR*BSRT, BSL_const = KBSL*BSLT
    #pragma omp simd collapse(2)
    for (int i = 0; i < 4; i++){
        for (int j = 0; j < size; j++){
            betaSS(j,i) = 1. / (1. + BSR_const / square(KBSR + CaSS(j,i)) + BSL_const / square(KBSL + CaSS(j,i)));
        }
    }
}

void update_betaJSR(VectorMap &betaJSR, const VectorMap &CaJSR, const double KCSQN, const double CSQN_const, const int size){
    // CSQN_const = KCSQN*CSQNT
    #pragma omp simd
    for (int i = 0; i < size; i++){
        betaJSR(i) = 1. / (1. + (CSQN_const / square(KCSQN + CaJSR(i))));
    }
}
*/
    

#endif

/*
int main(int argc, char* argv[]){
    const int nCRU = 1000;
    const double riss = 1.0;
    const double JLCC_const1 = 1.0;
    const double JLCC_const2 = 1.0;
    const double expVFRT = 0.1;
    double VFRT = 0.1;
    double Cai = 1e-3;
    
    double* Jiss_storage = new double[nCRU*4];
    double* CaSS_storage = new double [nCRU*4];
    double* LCC_storage = new double [nCRU*4*12];
    double* JLCC_storage = new double [nCRU*4];
    double* LCC_a_storage = new double [nCRU*4];


    MatrixMap Jiss(Jiss_storage,nCRU,4);
    MatrixMap CaSS(CaSS_storage,nCRU,4);
    Array3dMap LCC(LCC_storage,nCRU,4,12);
    MatrixMap JLCC(JLCC_storage,nCRU,4);
    MatrixMap LCC_a(LCC_a_storage,nCRU,4);
    
    Eigen::internal::set_is_malloc_allowed(false);
    for (int i = 0; i < nCRU; i++){
        for (int j = 0; j < 4; j++){
            Jiss(i,j) = 1.0;
            CaSS(i,j) = 1.0;
            LCC_a(i,j) = 1.0;
            JLCC(i,j) = 0.0;
            for (int k = 0; k < 12; k++){
                LCC(i,j,k) = 1.0;
            }
        }
    }


    clock_t tstart;
    clock_t tfin;

    tstart = clock();
    for (int i = 0; i < 100000; i++){
        update_JLCC(JLCC, LCC, LCC_a, CaSS, VFRT, expVFRT, JLCC_const1, JLCC_const2, nCRU);
        VFRT += 1e-10;
    }
    tfin = clock();
    std::cout << JLCC(1,1) << endl;
    std::cout << "Not unrolled: 100000 iterations finished in " << float(tfin - tstart)/CLOCKS_PER_SEC << " seconds" << endl;

    for (int i = 0; i < nCRU; i++){
        for (int j = 0; j < 4; j++){
            Jiss(i,j) = 1.0;
            CaSS(i,j) = 1.0;
        }
    }

    //tstart = clock();
    //for (int i = 0; i < 1000000; i++){
    //    update_Jxfer_unrolled(Jiss, CaSS, Cai, riss, nCRU);
    //    Cai += 1e-10;
    //}
    //tfin = clock();
    //std::cout << Jiss(1,1) << endl;
    //std::cout << "Unrolled: 1000000 iterations finished in " << float(tfin - tstart)/CLOCKS_PER_SEC << " seconds" << endl;


    Eigen::internal::set_is_malloc_allowed(true);
    delete[] Jiss_storage;
    delete[] CaSS_storage;
    delete[] LCC_storage;
    delete[] JLCC_storage;
    delete[] LCC_a_storage;

    return 0;
}

*/



