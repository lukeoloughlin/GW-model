#include "SSA.hpp"

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
    //double alpha = alphaLCC(V);
    //double beta = betaLCC(V);
    //double yinf = yinfLCC(V);
    //double tau = tauLCC(V);
    //double JLCC_exp = exp(2*V*FRT);
    //double JLCC_mult = 2e6 * V*FRT * params.PCaL / (params.VSS*(JLCC_exp - 1.0));
    double T = 1e-3;

    Eigen::internal::set_is_malloc_allowed(false);
    double start, stop;
    start = omp_get_wtime();
    for (int j=0; j < 1000; j++){
        SSA(LCC, LCC_a, RyR, ClCh, CaSS, CaJSR, Cai, CaNSR, JLCC, Jxfer, Jtr, V, expVFRT, T, N, consts);
    }
    stop = omp_get_wtime();
    cout << (stop - start) << endl;
    Eigen::internal::set_is_malloc_allowed(true);

}



int main(int argc, char* argv[]){
    int N = 1000;
    const GW_parameters params;
    const Constants GW_consts = consts_from_params(params);
    test_sample_weights(N);
    cout << endl;
    test_SSA_subunit(N, params, GW_consts);
    return 0;
}