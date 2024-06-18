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


inline void unpack_RyR_temp(Array3dMapi &RyR, const int* const RyR_temp, const int i, const int j){
    for (int k = 0; k < 6; k++){
        RyR(i,j,k) = RyR_temp[k];
    }
}

void SSA(MatrixMapi &LCC, MatrixMapi &LCC_a, Array3dMapi &RyR, MatrixMapi &ClCh, MatrixMap &CaSS, VectorMap &CaJSR, const double Cai, const double CaNSR, 
         MatrixMap &JLCC, MatrixMap &Jxfer, VectorMap &Jtr, const double V, const double expVFRT, const double T, const int nCRU, const Constants &consts)
{
    const double alpha = alphaLCC(V);
    const double beta = betaLCC(V);
    const double yinf = yinfLCC(V);
    const double tau = tauLCC(V);
    const double JLCC_exp = square(expVFRT);
    const double JLCC_mult = consts.JLCC_const * V * consts.FRT / (JLCC_exp - 1.0);

    #pragma omp parallel
    {
        #pragma omp for schedule( static )
        for (int i = 0; i < nCRU; i++){
            SSA_subunit(LCC, LCC_a, RyR, ClCh, CaSS, CaJSR, Cai, CaNSR, JLCC, Jxfer, Jtr, alpha, beta, yinf, tau, JLCC_mult, JLCC_exp, T, i, consts);
        }
    }
}


void SSA_subunit(MatrixMapi &LCC, MatrixMapi &LCC_a, Array3dMapi &RyR, MatrixMapi &ClCh, MatrixMap &CaSS, VectorMap &CaJSR, const double Cai, const double CaNSR, 
                 MatrixMap &JLCC, MatrixMap &Jxfer, VectorMap &Jtr, const double alpha, const double beta, const double yinf, const double tau, const double JLCC_mult, 
                 const double JLCC_exp, const double T, const int i, const Constants &consts)
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
    double Jtr_i = Jtr(i);
    double CaJSR_i = CaJSR(i);

    for (int j = 0; j < 4; j++){
        open_RyR[j] = 0.0;
        for (int k = 0; k < 6; k++){
            RyR_i[6*j+k] = RyR(i,j,k);
            if (k == 2 || k == 3){
                open_RyR[j] += double(RyR(i,j,k));
            }
        }
        LCC_i[j] = LCC(i,j);
        LCC_a_i[j] = LCC_a(i,j);
        ClCh_i[j] = ClCh(i,j);

        JLCC_i[j] = JLCC(i,j);
        Jxfer_i[j] = Jxfer(i,j);
        CaSS_i[j] = CaSS(i,j);
    }

    double total_rate;
    int subunit_idx;
    
    double t = 0.0;
    double dt = 0.0;
    while (1)
    {
        update_fluxes(CaSS_i, Cai, Jiss_i, Jxfer_i, i, consts);
        Jtr_i = consts.rtr * (CaNSR - CaJSR_i);
        total_rate = update_rates(LCC_i, LCC_a_i, RyR_i, ClCh_i, LCC_i_rates, LCC_a_i_rates, RyR_i_rates, ClCh_i_rates, CaSS_i, subunit_rates, alpha, beta, yinf, tau, i, consts);
        
        subunit_idx = sample_weights(subunit_rates, total_rate, 4);
        dt = -log(urand()) / total_rate;
        if (t + dt < T){
            update_CaSS(CaSS_i, RyR_i, JLCC_i, Jrel_i, Jxfer_i, Jiss_i, dt, i, consts);
            CaJSR_i += dt * (Jtr_i - consts.VSS_VJSR * (Jrel_i[0] + Jrel_i[1] + Jrel_i[2] + Jrel_i[3])) / (consts.CSQN_const / square(consts.KCSQN + CaJSR_i));
        } else {
            update_CaSS(CaSS_i, RyR_i, JLCC_i, Jrel_i, Jxfer_i, Jiss_i, T-t, i, consts);
            CaJSR_i += dt * (Jtr_i - consts.VSS_VJSR * (Jrel_i[0] + Jrel_i[1] + Jrel_i[2] + Jrel_i[3])) / (consts.CSQN_const / square(consts.KCSQN + CaJSR_i));
            break;
        }
        t += dt;

        update_state(LCC_i, LCC_a_i, RyR_i, open_RyR, ClCh_i, LCC_i_rates, LCC_a_i_rates, RyR_i_rates, ClCh_i_rates, subunit_rates, subunit_idx, i, CaSS_i, JLCC_i, Jrel_i, JLCC_mult, JLCC_exp, CaJSR_i, consts);
    }
    
    for (int j = 0; j < 4; j++){
        LCC(i,j) = LCC_i[j];
        LCC_a(i,j) = LCC_a_i[j];
        unpack_RyR_temp(RyR, RyR_i + 6*j, i, j);
        ClCh(i,j) = ClCh_i[j];
        JLCC(i,j) = JLCC_i[j];
        Jxfer(i,j) = Jxfer_i[j];
        CaSS(i,j) = CaSS_i[j];
    }
    CaJSR(i) = CaJSR_i;
    Jtr(i) = Jtr_i;
}

void update_fluxes(const double* const CaSS, const double Cai, double* const Jiss, double* const Jxfer, const int i, const Constants &consts){
    Jiss[0] = consts.riss * (CaSS[1] + CaSS[3] - 2.0*CaSS[0]);
    Jiss[1] = consts.riss * (CaSS[2] + CaSS[0] - 2.0*CaSS[1]);
    Jiss[2] = consts.riss * (CaSS[3] + CaSS[1] - 2.0*CaSS[2]);
    Jiss[3] = consts.riss * (CaSS[0] + CaSS[2] - 2.0*CaSS[3]);
    for (int j = 0; j < 4; j++){
        Jxfer[j] = consts.rxfer * (CaSS[j] - Cai); 
    }
}

double update_rates(const int* const LCC, const int* const LCC_a, const int* const RyR, const int* const ClCh, double* const LCC_rates, double* const LCC_a_rates, double* const RyR_rates, 
                 double* const ClCh_rates, const double* const CaSS, double* const subunit_rates, const double alpha, const double beta, const double yinf, const double tau, const int i, const Constants &consts){
    double total_rate = 0.0;

    LCC_activation_rate(LCC_a_rates, LCC_a, yinf, tau);
    ClCh_rate(ClCh_rates, ClCh, CaSS, consts.kfClCh, consts.kbClCh, i);
    for (int j = 0; j < 4; j++){
        subunit_rates[j] = 0.0;
        update_LCC_rates(LCC_rates, LCC, CaSS, i, j, alpha, beta, consts);
        subunit_rates[j] += update_RyR_rates(RyR_rates+12*j, RyR+6*j, CaSS, i, j, consts); // Using pointer arithmetic here
        subunit_rates[j] += (LCC_a_rates[j] + ClCh_rates[j] + LCC_rates[3*j] + LCC_rates[3*j+1] + LCC_rates[3*j+2]);
        total_rate += subunit_rates[j];

    }
    return total_rate;
}

void update_CaSS(double* const CaSS, int* const RyR, const double* const JLCC, const double* const Jrel, const double* const Jxfer, const double* const Jiss, const double dt, const int i, const Constants &consts){
    double dCaSS;
    double CaSS_tmp;
    int n56;
    int n34;
    double p;
    for (int j = 0; j < 4; j++){
        dCaSS = dt * (JLCC[j] + Jrel[j] - Jxfer[j] + Jiss[j]) / (1.0 + consts.BSR_const / square(consts.KBSR + CaSS[j]) + consts.BSL_const / square(consts.KBSL + CaSS[j]));
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
                 const double* const ClCh_rates, const double* const subunit_rates, const int subunit_idx, const int i, const double* const CaSS, double* JLCC, 
                 double* Jrel, const double JLCC_mult, const double JLCC_exp, const double CaJSR, const Constants &consts){
    double subunit_total = subunit_rates[subunit_idx];
    double LCC_rate_tot = LCC_rates[3*subunit_idx] + LCC_rates[3*subunit_idx+1] + LCC_rates[3*subunit_idx+2];
    double LCC_a_rate = LCC_a_rates[subunit_idx];
    double ClCh_rate = ClCh_rates[subunit_idx];
    double RyR_rate_tot = subunit_total - (LCC_rate_tot + LCC_a_rate + ClCh_rate);

    double u = urand() * subunit_total;

    if (u < LCC_rate_tot){
        sample_LCC(LCC, LCC_rates, LCC_rate_tot, LCC_a, CaSS, JLCC, i, subunit_idx, JLCC_mult, JLCC_exp, consts);
    }
    else if (u < LCC_rate_tot + LCC_a_rate) {
        LCC_a[subunit_idx] = 1 - LCC_a[subunit_idx];
        if (LCC_a[subunit_idx] == 1 && (LCC[subunit_idx] == 6 || LCC[subunit_idx] == 12)){
            JLCC[subunit_idx] = JLCC_mult * (consts.Cao_scaled - JLCC_exp * CaSS[subunit_idx]);
        }
    } 
    else if (u < LCC_rate_tot + LCC_a_rate + ClCh_rate){
        ClCh[subunit_idx] = 1 - ClCh[subunit_idx];
    }
    else {
        sample_RyR(RyR, open_RyR, RyR_rates, RyR_rate_tot, Jrel, subunit_idx, i, CaSS, CaJSR, consts);
    }
}

void sample_LCC(int* LCC, const double* const LCC_rates, const double total_LCC_rate, const int* const LCC_a, const double* const CaSS, double* JLCC, const int i, 
                const int subunit_idx, const double JLCC_mult, const double JLCC_exp, const Constants &consts){

    //double u = urand() * total_LCC_rate;
    const int transition = sample_weights(LCC_rates + 3*subunit_idx, total_LCC_rate, 3); // using pointer arithmetic here
    //const double cw1 = LCC_rates[3*subunit_idx];
    //const double cw2 = cw1 + LCC_rates[3*subunit_idx+1];
    switch (LCC[subunit_idx])
    {
    case 1:
        if (transition == 0){ LCC[subunit_idx] = 2;}
        else { LCC[subunit_idx] = 7;}
        break;
    case 2:
        if (transition == 0){ LCC[subunit_idx] = 1; }
        else if (transition == 1){ LCC[subunit_idx] = 3; }
        else { LCC[subunit_idx] = 8; }
        break;
    case 3:
        if (transition == 0){ LCC[subunit_idx] = 2; }
        else if (transition == 1){ LCC[subunit_idx] = 4; }
        else { LCC[subunit_idx] = 9; }
        break;
    case 4:
        if (transition == 0){ LCC[subunit_idx] = 3; }
        else if (transition == 1){ LCC[subunit_idx] = 5; }
        else { LCC[subunit_idx] = 10; }
        break;
    case 5:
        if (transition == 0){ LCC[subunit_idx] = 4; }
        else if (transition == 1){ LCC[subunit_idx] = 6; }
        else { LCC[subunit_idx] = 11; }
        break;
    case 6:
        LCC[subunit_idx] = 5;
        JLCC[subunit_idx] = LCC_a[subunit_idx] == 1 ? JLCC_mult * (consts.Cao_scaled - JLCC_exp * CaSS[subunit_idx]) : 0.0; 
        break;
    case 7:
        if (transition == 0){ LCC[subunit_idx] = 1; }
        else { LCC[subunit_idx] = 8; }
        break;
    case 8:
        if (transition == 0){ LCC[subunit_idx] = 2; }
        else if (transition == 1){ LCC[subunit_idx] = 7; }
        else { LCC[subunit_idx] = 9; }
        break;
    case 9:
        if (transition == 0){ LCC[subunit_idx] = 3; }
        else if (transition == 1){ LCC[subunit_idx] = 8; }
        else { LCC[subunit_idx] = 10; }
        break;
    case 10:
        if (transition == 0){ LCC[subunit_idx] = 4; }
        else if (transition == 1){ LCC[subunit_idx] = 9; }
        else { LCC[subunit_idx] = 11; }
        break;
    case 11:
        if (transition == 0){ LCC[subunit_idx] = 5; }
        else if (transition == 1){ LCC[subunit_idx] = 12; }
        else { LCC[subunit_idx] = 10; }
        break;
    case 12:
        LCC[subunit_idx] = 11;
        JLCC[subunit_idx] = LCC_a[subunit_idx] == 1 ? JLCC_mult * (consts.Cao_scaled - JLCC_exp * CaSS[subunit_idx]) : 0.0; 
        break;    
    default:
        break;
    }
}

void sample_RyR(int* RyR, double* open_RyR, const double* const RyR_rates, const double total_RyR_rate, double* Jrel, const int subunit_idx, const int i, const double* const CaSS, const double CaJSR,  const Constants &consts){
    const int transition = sample_weights(RyR_rates + 12*subunit_idx, total_RyR_rate, 12); // using pointer arithmetic here
    switch (transition)
    {
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
        open_RyR[6*subunit_idx]++;
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




void test_sample_weights(int N)
{
    double weights1[3];
    double weights2[5];
    double weights3[12];
    double weights4[100];

    double results1[3]; 
    double results2[5]; 
    double results3[12]; 
    double results4[100];

    int i1, i2, i3, i4;
    // Test on a uniform distribution
    double mse1 = 0.0, mse2 = 0.0, mse3 = 0.0, mse4 = 0.0;

    for (int j = 0; j < 3; j++) {weights1[j] = 1.0; results1[j] = 0.0;}
    for (int j = 0; j < 5; j++) {weights2[j] = 1.0; results2[j] = 0.0;}
    for (int j = 0; j < 12; j++) {weights3[j] = 1.0; results3[j] = 0.0;}
    for (int j = 0; j < 100; j++) {weights4[j] = 1.0; results4[j] = 0.0;}
    
    for (int i = 0; i < N; i++){
        i1 = sample_weights(weights1, 3.0, 3);
        i2 = sample_weights(weights2, 5.0, 5);
        i3 = sample_weights(weights3, 12.0, 12);
        i4 = sample_weights(weights4, 100.0, 100);

        results1[i1]++;
        results2[i2]++;
        results3[i3]++;
        results4[i4]++;
    }

    cout << results1[0] << ", " << results1[1] << ", " << results1[2] << endl;

    for (int j = 0; j < 3; j++) {mse1 += (results1[j]/N - 1/3) * (results1[j]/N - 1/3) / N;}
    for (int j = 0; j < 5; j++) {mse2 += (results2[j]/N - 0.2) * (results2[j]/N - 0.2) / N;}
    for (int j = 0; j < 12; j++) {mse3 += (results3[j]/N - 1/12) * (results3[j]/N - 1/12) / N;}
    for (int j = 0; j < 100; j++) {mse4 += (results4[j]/N - 0.01) * (results4[j]/N - 0.01) / N;}

    cout << "Mean square error 3 weights: " << mse1 << endl;
    cout << "Mean square error 5 weights: " << mse2 << endl;
    cout << "Mean square error 12 weights: " << mse3 << endl;
    cout << "Mean square error 100 weights: " << mse4 << endl;
}


void test_SSA_subunit(const int N, const GW_parameters &params, const Constants &consts)
{   
    std::vector<int> LCC_storage(4*N);
    MatrixMapi LCC(LCC_storage.data(),N,4);
    
    std::vector<int> LCC_a_storage(4*N);
    MatrixMapi LCC_a(LCC_a_storage.data(),N,4);
    
    std::vector<int> RyR_storage(4*6*N);
    Array3dMapi RyR(RyR_storage.data(),N,4,6);
    
    std::vector<int> ClCh_storage(4*N);
    MatrixMapi ClCh(ClCh_storage.data(),N,4);
    
    std::vector<double> CaSS_storage(4*N);
    MatrixMap CaSS(CaSS_storage.data(),N,4);
    
    std::vector<double> CaJSR_storage(N);
    VectorMap CaJSR(CaJSR_storage.data(),N);


    std::vector<double> JLCC_storage(4*N);
    MatrixMap JLCC(JLCC_storage.data(),N,4);
    
    std::vector<double> Jxfer_storage(4*N);
    MatrixMap Jxfer(Jxfer_storage.data(),N,4);
    
    std::vector<double> Jtr_storage(N);
    VectorMap Jtr(Jtr_storage.data(),N);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            LCC(i,j) = 1;
            LCC_a(i,j) = 1;
            RyR(i,j,0) = 1;
            RyR(i,j,1) = 1;
            RyR(i,j,2) = 1;
            RyR(i,j,3) = 1;
            RyR(i,j,4) = 1;
            RyR(i,j,5) = 0;
            ClCh(i,j) = 0;
            CaSS(i,j) = 1e-3 * urand() + 1e-10;
            JLCC(i,j) = 0.0;
            Jxfer(i,j) = 0.0;
        }
        CaJSR(i) = 0.9 * (0.5 * urand() + 0.5);
        Jtr(i) = 0.0;   
    }





    double Cai = 1e-4;
    double CaNSR = 0.9;
    double V = 120.0 * urand() - 80.0;
    double expVFRT = exp(V*FRT);
    double alpha = alphaLCC(V);
    double beta = betaLCC(V);
    double yinf = yinfLCC(V);
    double tau = tauLCC(V);
    double JLCC_exp = exp(2*V*FRT);
    double JLCC_mult = 2e6 * V*FRT * params.PCaL / (params.VSS*(JLCC_exp - 1.0));
    double T = 1e-2;

    Eigen::internal::set_is_malloc_allowed(false);
    SSA_subunit(LCC, LCC_a, RyR, ClCh, CaSS, CaJSR, Cai, CaNSR, JLCC, Jxfer, Jtr, alpha, beta, yinf, tau, JLCC_mult, JLCC_exp, T, 0, consts);
    Eigen::internal::set_is_malloc_allowed(true);
    
    /*
    cout << "States before: " << endl;
    cout << "LCC: " << LCC(10,0) << ", " << LCC(10,1) << ", " << LCC(10,2) << ", " << LCC(10,3) << endl;
    cout << "LCC_a: " << LCC_a(10,0) << ", " << LCC_a(10,1) << ", " << LCC_a(10,2) << ", " << LCC_a(10,3) << endl;
    cout << "RyR (num state 1): " << RyR(10,0,0) << ", " << RyR(10,1,0) << ", " << RyR(10,2,0) << ", " << RyR(10,3,0) << endl;
    cout << "RyR (num state 2): " << RyR(10,0,1) << ", " << RyR(10,1,1) << ", " << RyR(10,2,1) << ", " << RyR(10,3,1) << endl;
    cout << "RyR (num state 3): " << RyR(10,0,2) << ", " << RyR(10,1,2) << ", " << RyR(10,2,2) << ", " << RyR(10,3,2) << endl;
    cout << "RyR (num state 4): " << RyR(10,0,3) << ", " << RyR(10,1,3) << ", " << RyR(10,2,3) << ", " << RyR(10,3,3) << endl;
    cout << "RyR (num state 5): " << RyR(10,0,4) << ", " << RyR(10,1,4) << ", " << RyR(10,2,4) << ", " << RyR(10,3,4) << endl;
    cout << "RyR (num state 6): " << RyR(10,0,5) << ", " << RyR(10,1,5) << ", " << RyR(10,2,5) << ", " << RyR(10,3,5) << endl;
    cout << "ClCh: " << ClCh(10,0) << ", " << ClCh(10,1) << ", " << ClCh(10,2) << ", " << ClCh(10,3) << endl;
    cout << "CaSS: " << CaSS(10,0) << ", " << CaSS(10,1) << ", " << CaSS(10,2) << ", " << CaSS(10,3) << endl;
    cout << "CaJSR: " << CaJSR(10) << endl;
    cout << "JLCC: " << JLCC(10,0) << ", " << JLCC(10,1) << ", " << JLCC(10,2) << ", " << JLCC(10,3) << endl;
    cout << "Jxfer: " << Jxfer(10,0) << ", " << Jxfer(10,1) << ", " << Jxfer(10,2) << ", " << Jxfer(10,3) << endl;
    cout << "Jtr: " << Jtr(10) << endl << endl;
    */

    Eigen::internal::set_is_malloc_allowed(false);
    double start, stop;
    start = omp_get_wtime();
    SSA(LCC, LCC_a, RyR, ClCh, CaSS, CaJSR, Cai, CaNSR, JLCC, Jxfer, Jtr, V, expVFRT, T, N, consts);
    stop = omp_get_wtime();
    cout << (stop - start) << endl;
    Eigen::internal::set_is_malloc_allowed(true);
    
    /*
    cout << "States after: " << endl;
    cout << "LCC: " << LCC(10,0) << ", " << LCC(10,1) << ", " << LCC(10,2) << ", " << LCC(10,3) << endl;
    cout << "LCC_a: " << LCC_a(10,0) << ", " << LCC_a(10,1) << ", " << LCC_a(10,2) << ", " << LCC_a(10,3) << endl;
    cout << "RyR (num state 1): " << RyR(10,0,0) << ", " << RyR(10,1,0) << ", " << RyR(10,2,0) << ", " << RyR(10,3,0) << endl;
    cout << "RyR (num state 2): " << RyR(10,0,1) << ", " << RyR(10,1,1) << ", " << RyR(10,2,1) << ", " << RyR(10,3,1) << endl;
    cout << "RyR (num state 3): " << RyR(10,0,2) << ", " << RyR(10,1,2) << ", " << RyR(10,2,2) << ", " << RyR(10,3,2) << endl;
    cout << "RyR (num state 4): " << RyR(10,0,3) << ", " << RyR(10,1,3) << ", " << RyR(10,2,3) << ", " << RyR(10,3,3) << endl;
    cout << "RyR (num state 5): " << RyR(10,0,4) << ", " << RyR(10,1,4) << ", " << RyR(10,2,4) << ", " << RyR(10,3,4) << endl;
    cout << "RyR (num state 6): " << RyR(10,0,5) << ", " << RyR(10,1,5) << ", " << RyR(10,2,5) << ", " << RyR(10,3,5) << endl;
    cout << "ClCh: " << ClCh(10,0) << ", " << ClCh(10,1) << ", " << ClCh(10,2) << ", " << ClCh(10,3) << endl;
    cout << "CaSS: " << CaSS(10,0) << ", " << CaSS(10,1) << ", " << CaSS(10,2) << ", " << CaSS(10,3) << endl;
    cout << "CaJSR: " << CaJSR(10) << endl;
    cout << "JLCC: " << JLCC(10,0) << ", " << JLCC(10,1) << ", " << JLCC(10,2) << ", " << JLCC(10,3) << endl;
    cout << "Jxfer: " << Jxfer(10,0) << ", " << Jxfer(10,1) << ", " << Jxfer(10,2) << ", " << Jxfer(10,3) << endl;
    cout << "Jtr: " << Jtr(10) << endl << endl;
    */

}



int main(int argc, char* argv[]){
    int N = 50000;
    const GW_parameters params;
    const Constants GW_consts = consts_from_params(params);
    test_sample_weights(N);
    cout << endl;
    test_SSA_subunit(N, params, GW_consts);
    return 0;
}