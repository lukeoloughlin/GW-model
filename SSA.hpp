#ifndef _SSAH
#define _SSAH

#define EIGEN_RUNTIME_NO_MALLOC
#include "GW.hpp"
#include "rates.hpp"
#include <omp.h>
//#include <unistd.h>

int sample_weights(const double* const weights, const double total_weight, const int size){
    const double u = urand() * total_weight;
    double cum_weight = weights[0];
    int i = 0;
    while (cum_weight < u && i < (size-1)){
        i++;
        cum_weight += weights[i];
    }
    return i;
}


inline void initialise_temp_states(int* const LCC_i, int* const RyR_i, double* const open_RyR, int* const LCC_a_i, int* const ClCh_i, double* const LCC_i_rates, double* const RyR_i_rates, double* const LCC_a_i_rates, double* const ClCh_i_rates, 
                                   double* const subunit_i_rates, double* const CaSS_i, double* const JLCC_i, double* const Jrel_i, double* const Jxfer_i, double* const Jiss_i, const NDArray<int,2> &LCC, const NDArray<int,2> &LCC_a, const NDArray<int,3> &RyR,
                                   const NDArray<int,2> &ClCh, const NDArray<double,2> &CaSS, const NDArray<double,2> &JLCC, const double CaJSR_i, const int i, const Constants &consts){
for (int j = 0; j < 4; j++){
        for (int k = 0; k < 6; k++)
            RyR_i[6*j+k] = RyR(i,j,k);
        for (int k = 0; k < 12; k++)
            RyR_i_rates[12*j+k] = 0.0;

        LCC_i[j] = LCC(i,j);
        LCC_a_i[j] = LCC_a(i,j);
        ClCh_i[j] = ClCh(i,j);
        open_RyR[j] = double(RyR(i,j,2) + RyR(i,j,3));

        LCC_i_rates[3*j] = 0.0;
        LCC_i_rates[3*j+1] = 0.0;
        LCC_i_rates[3*j+2] = 0.0;
        LCC_a_i_rates[j] = 0.0;
        ClCh_i_rates[j] = 0.0;
        subunit_i_rates[j] = 0.0;

        CaSS_i[j] = CaSS(i,j);
        JLCC_i[j] = JLCC(i,j); // This doesn't update at every iteration of loop so have to set it here
        Jrel_i[j] = consts.rRyR * open_RyR[j] * (CaJSR_i - CaSS_i[j]);
        Jxfer_i[j] = 0.0;
        Jiss_i[j] = 0.0;
    }    

}

inline void record_from_temp(const int* const LCC_i, const int* const RyR_i, const int* const LCC_a_i, const int* const ClCh_i, double* const CaSS_i, double* const JLCC_i, 
                            const double* const Jrel_i, const double* const Jxfer_i, const double* const Jiss_i, const double* const out_vals, NDArray<int,2> &LCC, NDArray<int,2> &LCC_a, NDArray<int,3> &RyR,
                            NDArray<int,2> &ClCh, NDArray<double,2> &CaSS, NDArray<double,1> &CaJSR, NDArray<double,2> &JLCC, NDArray<double,2> &Jxfer, NDArray<double,1> &Jtr, const int i){
    for (int j = 0; j < 4; j++){
        LCC(i,j) = LCC_i[j];
        LCC_a(i,j) = LCC_a_i[j];
        for (int k = 0; k < 6; k++){
            RyR(i,j,k) = RyR_i[6*j+k];
        }
        ClCh(i,j) = ClCh_i[j];
        JLCC(i,j) = JLCC_i[j];
        Jxfer(i,j) = Jxfer_i[j];
        CaSS(i,j) = CaSS_i[j];
    }
    CaJSR(i) = out_vals[0];
    Jtr(i) = out_vals[1];

}

void SSA(NDArray<int,2> &LCC, NDArray<int,2> &LCC_a, NDArray<int,3> &RyR, NDArray<int,2> &ClCh, NDArray<double,2> &CaSS, NDArray<double,1> &CaJSR, const double Cai, const double CaNSR, 
         NDArray<double,2> &JLCC, NDArray<double,2> &Jxfer, NDArray<double,1> &Jtr, const double V, const double expmVFRT, const double T, const int nCRU, const Constants &consts){
    const double alpha = alphaLCC(V);
    const double beta = betaLCC(V);
    const double yinf = yinfLCC(V);
    const double tau = tauLCC(V);
    const double JLCC_exp = square(1.0 / expmVFRT);
    const double JLCC_mult = consts.JLCC_const * V * FRT / (JLCC_exp - 1.0);

    #pragma omp parallel
    {
        int LCC_i[4];
        int RyR_i[4*6];
        double open_RyR[4];
        int LCC_a_i[4];
        int ClCh_i[4];

        double LCC_i_rates[3*4]; // At most 3 non-zero rates
        double RyR_i_rates[12*4];
        double LCC_a_i_rates[4];
        double ClCh_i_rates[4];
        double subunit_rates[4];

        double CaSS_i[4];
        double JLCC_i[4];
        double Jrel_i[4];
        double Jxfer_i[4];
        double Jiss_i[4];
        double CaJSR_i;
        double out_vals[2]; // {CaJSR_i, Jtr_i}
        
        #pragma omp for schedule( static )
        for (int i = 0; i < nCRU; i++){
            CaJSR_i = CaJSR(i);
            initialise_temp_states(LCC_i, RyR_i, open_RyR, LCC_a_i, ClCh_i, LCC_i_rates, RyR_i_rates, LCC_a_i_rates, ClCh_i_rates, subunit_rates, CaSS_i, JLCC_i, Jrel_i, Jxfer_i, Jiss_i, LCC, LCC_a, RyR, ClCh, CaSS, JLCC, CaJSR_i, i, consts);

            SSA_subunit(LCC_i, LCC_a_i, RyR_i, open_RyR, ClCh_i, LCC_i_rates, RyR_i_rates, LCC_a_i_rates, ClCh_i_rates, subunit_rates, CaSS_i, JLCC_i, Jrel_i, Jxfer_i, Jiss_i, CaJSR_i, Cai, CaNSR, alpha, beta, yinf, tau, JLCC_mult, JLCC_exp, T, out_vals, consts);

            record_from_temp(LCC_i, RyR_i, LCC_a_i, ClCh_i, CaSS_i, JLCC_i, Jrel_i, Jxfer_i, Jiss_i, out_vals, LCC, LCC_a, RyR, ClCh, CaSS, CaJSR, JLCC, Jxfer, Jtr, i);

        }
    }
}


void SSA_subunit(int* const LCC_i, int* const LCC_a_i, int* const RyR_i, double* const open_RyR, int* const ClCh_i, double* const LCC_i_rates, double* const RyR_i_rates, double* const LCC_a_i_rates, double* const ClCh_i_rates, double* const subunit_rates, 
                 double* const CaSS_i, double* const JLCC_i, double* const Jrel_i, double* const Jxfer_i, double* const Jiss_i, double CaJSR_i, const double Cai, const double CaNSR, const double alpha, const double beta, const double yinf, const double tau, 
                 const double JLCC_mult, const double JLCC_exp, const double T, double* const out_vals, const Constants &consts){
    double Jtr_i;
    double total_rate;
    int subunit_idx;
    
    double t = 0.0;
    double dt = 0.0;
    while (1)
    {
        update_fluxes(CaSS_i, Cai, Jiss_i, Jxfer_i, consts);
        Jtr_i = consts.rtr * (CaNSR - CaJSR_i);
        total_rate = update_rates(LCC_i, LCC_a_i, RyR_i, ClCh_i, LCC_i_rates, LCC_a_i_rates, RyR_i_rates, ClCh_i_rates, CaSS_i, subunit_rates, alpha, beta, yinf, tau, consts);
        
        subunit_idx = sample_weights(subunit_rates, total_rate, 4);
        dt = -log(urand()) / total_rate;
        if (t + dt < T){
            update_CaSS(CaSS_i, RyR_i, JLCC_i, Jrel_i, Jxfer_i, Jiss_i, dt, consts);
            CaJSR_i += (dt * (Jtr_i - consts.VSS_VJSR * (Jrel_i[0] + Jrel_i[1] + Jrel_i[2] + Jrel_i[3])) / (1.0 + consts.CSQN_const / square(consts.KCSQN + CaJSR_i)));
        } 
        else {
            update_CaSS(CaSS_i, RyR_i, JLCC_i, Jrel_i, Jxfer_i, Jiss_i, T-t, consts);
            CaJSR_i += ((T-t) * (Jtr_i - consts.VSS_VJSR * (Jrel_i[0] + Jrel_i[1] + Jrel_i[2] + Jrel_i[3])) / (1.0 + consts.CSQN_const / square(consts.KCSQN + CaJSR_i)));
            break;
        }
        t += dt;

        update_state(LCC_i, LCC_a_i, RyR_i, open_RyR, ClCh_i, LCC_i_rates, LCC_a_i_rates, RyR_i_rates, ClCh_i_rates, subunit_rates, subunit_idx, CaSS_i, JLCC_i, Jrel_i, JLCC_mult, JLCC_exp, CaJSR_i, consts);
    }
    
    out_vals[0] = CaJSR_i;
    out_vals[1] = Jtr_i;
}

void update_fluxes(const double* const CaSS, const double Cai, double* const Jiss, double* const Jxfer, const Constants &consts){
    Jiss[0] = consts.riss * (CaSS[1] + CaSS[3] - 2.0*CaSS[0]);
    Jiss[1] = consts.riss * (CaSS[2] + CaSS[0] - 2.0*CaSS[1]);
    Jiss[2] = consts.riss * (CaSS[3] + CaSS[1] - 2.0*CaSS[2]);
    Jiss[3] = consts.riss * (CaSS[0] + CaSS[2] - 2.0*CaSS[3]);
    for (int j = 0; j < 4; j++){
        Jxfer[j] = consts.rxfer * (CaSS[j] - Cai); 
    }
}

double update_rates(const int* const LCC, const int* const LCC_a, const int* const RyR, const int* const ClCh, double* const LCC_rates, double* const LCC_a_rates, double* const RyR_rates, 
                 double* const ClCh_rates, const double* const CaSS, double* const subunit_rates, const double alpha, const double beta, const double yinf, const double tau, const Constants &consts){
    double total_rate = 0.0;

    LCC_activation_rate(LCC_a_rates, LCC_a, yinf, tau);
    ClCh_rate(ClCh_rates, ClCh, CaSS, consts.kfClCh, consts.kbClCh);
    for (int j = 0; j < 4; j++){
        subunit_rates[j] = 0.0;
        update_LCC_rates(LCC_rates, LCC, CaSS, j, alpha, beta, subunit_rates, consts);
        update_RyR_rates(RyR_rates, RyR, CaSS, j, subunit_rates, consts); 
        subunit_rates[j] += LCC_a_rates[j];
        subunit_rates[j] += ClCh_rates[j];
        total_rate += subunit_rates[j];
    }
    return total_rate;
}

void update_CaSS(double* const CaSS, int* const RyR, const double* const JLCC, const double* const Jrel, const double* const Jxfer, const double* const Jiss, const double dt, const Constants &consts){
    double dCaSS;
    double CaSS_tmp;
    int n56;
    int n34;
    double p;
    for (int j = 0; j < 4; j++){
        dCaSS = dt * (JLCC[j] + Jrel[j] - Jxfer[j] + Jiss[j]) / (1.0 + (consts.BSR_const / square(consts.KBSR + CaSS[j])) + (consts.BSL_const / square(consts.KBSL + CaSS[j])));
        CaSS_tmp = CaSS[j] + dCaSS;
        if (CaSS[j] > 1.15e-4 && CaSS_tmp <= 1.15e-4){
            n56 = RyR[4+6*j] + RyR[5+6*j];
            if (n56 > 0){
                p = consts.k65 / (consts.k65 + consts.k56 * square(CaSS[j]));
                RyR[4+6*j] = sample_binomial(p, n56);
                RyR[5+6*j] = n56 - RyR[4+6*j];
            }
            else {
                RyR[4+6*j] = 0;
                RyR[5+6*j] = 0;
            }
        } 
        else if (CaSS[j] > 0.03685 && CaSS_tmp <= 0.03685){
            n34 = RyR[2+6*j] + RyR[3+6*j];
            if (n34 > 0){
                p = consts.k43 / (consts.k43 + consts.k34 * square(CaSS[j]));
                RyR[2+6*j] = sample_binomial(p, n34);
                RyR[3+6*j] = n34 - RyR[2+6*j];
            }
            else {
                RyR[2+6*j] = 0;
                RyR[3+6*j] = 0;
            }
        }
        CaSS[j] = CaSS_tmp;
    }
}

void update_state(int* const LCC, int* const LCC_a, int* const RyR, double* const open_RyR, int* const ClCh, const double* const LCC_rates, const double* const LCC_a_rates, const double* const RyR_rates, 
                 const double* const ClCh_rates, const double* const subunit_rates, const int subunit_idx, const double* const CaSS, double* JLCC, 
                 double* Jrel, const double JLCC_mult, const double JLCC_exp, const double CaJSR, const Constants &consts){
    double subunit_total = subunit_rates[subunit_idx];
    double LCC_rate_tot = LCC_rates[3*subunit_idx] + LCC_rates[3*subunit_idx+1] + LCC_rates[3*subunit_idx+2];
    double LCC_a_rate = LCC_a_rates[subunit_idx];
    double ClCh_rate = ClCh_rates[subunit_idx];
    double RyR_rate_tot = subunit_total - (LCC_rate_tot + LCC_a_rate + ClCh_rate);

    double u = urand() * subunit_total;

    if (u < LCC_rate_tot)
        sample_LCC(LCC, LCC_rates, LCC_rate_tot, LCC_a, CaSS, JLCC, subunit_idx, JLCC_mult, JLCC_exp, consts);
    else if (u < (LCC_rate_tot + LCC_a_rate)) {
        LCC_a[subunit_idx] = 1 - LCC_a[subunit_idx];
        if (LCC_a[subunit_idx] == 0)
            JLCC[subunit_idx] = 0.0;
        else if ((LCC[subunit_idx] == 6) || (LCC[subunit_idx] == 12))
            JLCC[subunit_idx] = JLCC_mult * (consts.Cao_scaled - JLCC_exp * CaSS[subunit_idx]);
    } 
    else if (u < (LCC_rate_tot + LCC_a_rate + ClCh_rate))
        ClCh[subunit_idx] = 1 - ClCh[subunit_idx];
    else 
        sample_RyR(RyR, open_RyR, RyR_rates, RyR_rate_tot, Jrel, subunit_idx, CaSS, CaJSR, consts);
}

void sample_LCC(int* const LCC, const double* const LCC_rates, const double total_LCC_rate, const int* const LCC_a, const double* const CaSS, double* const JLCC, 
                const int subunit_idx, const double JLCC_mult, const double JLCC_exp, const Constants &consts){

    const int transition = sample_weights(LCC_rates + 3*subunit_idx, total_LCC_rate, 3); // using pointer arithmetic here
    switch (LCC[subunit_idx]){
    case 1:
        if (transition == 0)
            LCC[subunit_idx] = 2;
        else 
            LCC[subunit_idx] = 7;
        break;
    case 2:
        if (transition == 0)
            LCC[subunit_idx] = 1;
        else if (transition == 1)
            LCC[subunit_idx] = 3;
        else 
            LCC[subunit_idx] = 8;
        break;
    case 3:
        if (transition == 0)
            LCC[subunit_idx] = 2;
        else if (transition == 1)
            LCC[subunit_idx] = 4;
        else 
            LCC[subunit_idx] = 9;
        break;
    case 4:
        if (transition == 0)
            LCC[subunit_idx] = 3;
        else if (transition == 1)
            LCC[subunit_idx] = 5;
        else 
            LCC[subunit_idx] = 10;
        break;
    case 5:
        if (transition == 0)
            LCC[subunit_idx] = 4; 
        else if (transition == 1){ 
            LCC[subunit_idx] = 6; 
            JLCC[subunit_idx] = (LCC_a[subunit_idx] == 1) ? JLCC_mult * (consts.Cao_scaled - JLCC_exp * CaSS[subunit_idx]) : 0.0; 
        }
        else 
            LCC[subunit_idx] = 11;
        break;
    case 6:
        LCC[subunit_idx] = 5;
        JLCC[subunit_idx] = 0.0;
        break;
    case 7:
        if (transition == 0)
            LCC[subunit_idx] = 1;
        else 
            LCC[subunit_idx] = 8;
        break;
    case 8:
        if (transition == 0)
            LCC[subunit_idx] = 2;
        else if (transition == 1)
            LCC[subunit_idx] = 7;
        else 
            LCC[subunit_idx] = 9;
        break;
    case 9:
        if (transition == 0)
            LCC[subunit_idx] = 3;
        else if (transition == 1)
            LCC[subunit_idx] = 8;
        else 
            LCC[subunit_idx] = 10;
        break;
    case 10:
        if (transition == 0)
            LCC[subunit_idx] = 4;
        else if (transition == 1)
            LCC[subunit_idx] = 9;
        else 
            LCC[subunit_idx] = 11;
        break;
    case 11:
        if (transition == 0)
            LCC[subunit_idx] = 5; 
        else if (transition == 1)
            LCC[subunit_idx] = 10; 
        else {
            LCC[subunit_idx] = 12; 
            JLCC[subunit_idx] = (LCC_a[subunit_idx] == 1) ? JLCC_mult * (consts.Cao_scaled - JLCC_exp * CaSS[subunit_idx]) : 0.0;
        }
        break;
    case 12:
        LCC[subunit_idx] = 11;
        JLCC[subunit_idx] = 0.0;
        break;    
    default:
        break;
    }
}

void sample_RyR(int* RyR, double* open_RyR, const double* const RyR_rates, const double total_RyR_rate, double* Jrel, const int subunit_idx, const double* const CaSS, const double CaJSR,  const Constants &consts){
    const int transition = sample_weights(RyR_rates + 12*subunit_idx, total_RyR_rate, 12); // using pointer arithmetic here
    switch (transition){
    case 0: // 1 -> 2
        RyR[6*subunit_idx]--;
        RyR[6*subunit_idx+1]++;   
        break;
    case 1: // 2 -> 3
        RyR[6*subunit_idx+1]--;
        RyR[6*subunit_idx+2]++;   
        open_RyR[subunit_idx]++;
        break;
    case 2: // 2 -> 5
        RyR[6*subunit_idx+1]--;
        RyR[6*subunit_idx+4]++;   
        break;
    case 3: // 3 -> 4
        RyR[6*subunit_idx+2]--;
        RyR[6*subunit_idx+3]++;   
        break;
    case 4: // 4 -> 5
        RyR[6*subunit_idx+3]--;
        RyR[6*subunit_idx+4]++;
        open_RyR[subunit_idx]--;
        break;
    case 5: // 5 -> 6
        RyR[6*subunit_idx+4]--;
        RyR[6*subunit_idx+5]++;   
        break;
    case 6: // 2 -> 1
        RyR[6*subunit_idx+1]--;
        RyR[6*subunit_idx]++;   
        break;
    case 7: // 3 -> 2
        RyR[6*subunit_idx+2]--;
        RyR[6*subunit_idx+1]++;
        open_RyR[subunit_idx]--; 
        break;
    case 8: // 4 -> 3
        RyR[6*subunit_idx+3]--;
        RyR[6*subunit_idx+2]++;   
        break;
    case 9: // 5 -> 2
        RyR[6*subunit_idx+4]--;
        RyR[6*subunit_idx+1]++;   
        break;
    case 10: // 5 -> 4
        RyR[6*subunit_idx+4]--;
        RyR[6*subunit_idx+3]++;
        open_RyR[subunit_idx]++;
        break;
    case 11: // 6 -> 5
        RyR[6*subunit_idx+5]--;
        RyR[6*subunit_idx+4]++;   
        break;    
    default:
        break;
    }
    Jrel[subunit_idx] = consts.rRyR * open_RyR[subunit_idx] * (CaJSR - CaSS[subunit_idx]);
}


#endif
