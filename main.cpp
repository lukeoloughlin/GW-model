#include "SSA.hpp"
#include "currents.hpp"
#include <iostream>
#include <functional>
#include <fstream>

using namespace std;

void initialise_CaSS(NDArray<double,2> &CaSS){ CaSS.set_to_val(1.45370e-4); }
void initialise_CaJSR(NDArray<double,1> &CaJSR){ CaJSR.set_to_val(0.908408); }


void initialise_LCC(NDArray<int,2> &LCC)
{
    const double weights[3] = { 0.958, 0.038, 0.004 };
    int idx;

    for (unsigned int i = 0; i < LCC.shape(0); i++){
        for (unsigned int j = 0; j < 4; j++){
            idx = sample_weights(weights, 1.0, 3);
            if (idx == 0){
                LCC(i,j) = 1;
            } 
            else if (idx == 1){
                LCC(i,j) = 2;
            }
            else {
                LCC(i,j) = 7;
            }
        }
    }
}

void initialise_LCC_a(NDArray<int,2> &LCC_a)
{
    const double weights[2] = { 0.9425, 0.0575 };
    int idx;
    for (unsigned int i = 0; i < LCC_a.shape(0); i++){
        for (unsigned int j = 0; j < 4; j++){
            idx = sample_weights(weights, 1.0, 2);
            if (idx == 0){
                LCC_a(i,j) = 1;
            } 
            else {
                LCC_a(i,j) = 0;
            }
        }
    }
}


void initialise_RyR(NDArray<int,3> &RyR){
    const double weights[3] = { 0.609, 0.5*0.391, 0.5*0.391 };
    int idx;
    for (int i = 0; i < RyR.shape(0); i++){
        for (int j = 0; j < 4; j++){
            for (int k = 0; k < 5; k++){
                idx = sample_weights(weights, 1.0, 3);
                if (idx == 0){
                    RyR(i,j,0)++;
                } 
                else if (idx == 1){
                    RyR(i,j,4)++;
                }
                else {
                    RyR(i,j,5)++;
                }
            }    
        }
    }
}


void initialise_ClCh(NDArray<int,2> &ClCh){
    const double weights[2] = { 0.998, 0.002 };
    int idx;
    for (unsigned int i = 0; i < ClCh.shape(0); i++){
        for (unsigned int j = 0; j < 4; j++){
            idx = sample_weights(weights, 1.0, 2);
            if (idx == 0){
                ClCh(i,j) = 0;
            } 
            else {
                ClCh(i,j) = 1;
            }
        }
    }
}

void initialise_JLCC(NDArray<double,2> &JLCC, const NDArray<int,2> &LCC, const NDArray<int,2> &LCC_a, const NDArray<double,2> &CaSS, const double VFRT, const Constants &consts){
    double exp_term = exp(2*VFRT);
    for (unsigned int i = 0; i < JLCC.shape(0); i++){
        for (unsigned int j = 0; j < 4; j++){
            if ((LCC_a(i,j) == 1) && (LCC(i,j) == 6 || LCC(i,j) == 12))
                JLCC(i,j) = consts.JLCC_const * VFRT * (consts.Cao_scaled - exp_term * CaSS(i,j));
            else
                JLCC(i,j) = 0.0;
        }
    }
}

void initialise_Jxfer(NDArray<double,2> &Jxfer, const NDArray<double,2> &CaSS, const double Cai, const GW_parameters &params){
    Jxfer.set_to_zeros();
    Jxfer += CaSS;
    Jxfer -= Cai;
    Jxfer *= params.rxfer;
}

void initialise_Jtr(NDArray<double,1> &Jtr, const NDArray<double,1> &CaJSR, const double CaNSR, const GW_parameters &params){
    Jtr.set_to_zeros();
    Jtr += CaNSR;
    Jtr -= CaJSR;
    Jtr *= params.rtr;
}

struct Concentrations
{
    double Nai = 10.0;
    double Ki = 131.84;
    double Cai = 1.45273e-4;
    double CaNSR = 0.908882;
    double CaLTRPN = 8.98282e-3;
    double CaHTRPN = 0.137617;
};

struct Gates
{
    double m = 5.33837e-4;
    double h = 0.996345;
    double j = 0.997315;
    double xKs = 2.04171e-4;
};

const int Kr_idx[2] = { 5, 5 };
const int Kv_idx[2] = { 10, 10 };

class GW_model
{
private:

    int nCRU;
    Constants consts;
    double VFRT;
    double expmVFRT;

    double QKr_storage[5*5];
    double QKv14_storage[10*10];
    double QKv43_storage[10*10];
    NDArrayMap<double,2> QKr;
    NDArrayMap<double,2> QKv14;
    NDArrayMap<double,2> QKv43;

    double dV;
    Concentrations dconc;
    Gates dgates; 
    double dKr[5];
    double dKv43[9];
    double dKv14[9];

    NDArray<double,2> JLCC;
    NDArray<double,2> Jxfer;
    NDArray<double,1> Jtr;

    double Istim;


    void update_dconc_and_dV();
    void update_dgates();
    void update_dKr_and_dKV();
    
    void write_state(ofstream &file, const double t);
    
public:

    GW_parameters parameters;
    double V;
    Concentrations concentrations;
    Gates gates;
    double Kr[5];
    double Kv43[10];
    double Kv14[10];
    NDArray<double,2> CaSS;
    NDArray<double,1> CaJSR;
    NDArray<int,2> LCC;
    NDArray<int,2> LCC_activation;
    NDArray<int,3> RyR;
    NDArray<int,2> ClCh;

    GW_model(int nCRU_) : nCRU(nCRU_), parameters(), concentrations(), gates(), CaSS(NDArray<double,2>(nCRU_,4)), CaJSR(NDArray<double,1>(nCRU_)),
                        LCC(NDArray<int,2>(nCRU_,4)), LCC_activation(NDArray<int,2>(nCRU_,4)), RyR(NDArray<int,3>(nCRU_,4,6)), ClCh(NDArray<int,2>(nCRU_,4)),
                        JLCC(NDArray<double,2>(nCRU_,4)), Jxfer(NDArray<double,2>(nCRU_,4)), Jtr(NDArray<double,1>(nCRU_))  
    { 
        V = -91.382;
        VFRT = V*FRT;
        expmVFRT = exp(-VFRT);


        consts = consts_from_params(parameters, nCRU_);
        
        initialise_CaSS(CaSS);
        initialise_CaJSR(CaJSR);
        initialise_LCC(LCC);
        initialise_LCC_a(LCC_activation);
        initialise_RyR(RyR);
        initialise_ClCh(ClCh);
        

        initialise_JLCC(JLCC, LCC, LCC_activation, CaSS, VFRT, consts);
        initialise_Jxfer(Jxfer, CaSS, concentrations.Cai, parameters);
        initialise_Jtr(Jtr, CaJSR, concentrations.CaNSR, parameters);
        


        Kr[0] = 0.999503;
        Kr[1] = 4.13720e-4;
        Kr[2] = 7.27568e-5;
        Kr[3] = 8.73984e-6;
        Kr[4] = 1.36159e-6;
        
        Kv43[0] = 0.953060;
        Kv43[1] = 0.0253906;
        Kv43[2] = 2.53848e-4;
        Kv43[3] = 1.12796e-6;
        Kv43[4] = 1.87950e-9;
        Kv43[5] = 0.0151370;
        Kv43[6] = 0.00517622;
        Kv43[7] = 8.96600e-4;
        Kv43[8] = 8.17569e-5;
        Kv43[9] = 2.24032e-6;

        Kv14[0] = 0.722328;
        Kv14[1] = 0.101971;
        Kv14[2] = 0.00539932;
        Kv14[3] = 1.27081e-4;
        Kv14[4] = 1.82742e-6;
        Kv14[5] = 0.152769;
        Kv14[6] = 0.00962328;
        Kv14[7] = 0.00439043;
        Kv14[8] = 0.00195348;
        Kv14[9] = 0.00143629;
        

        QKr = NDArrayMap<double,2>(QKr_storage, Kr_idx, 5*5);
        QKv14 = NDArrayMap<double,2>(QKv14_storage, Kv_idx, 10*10);
        QKv43 = NDArrayMap<double,2>(QKv43_storage, Kv_idx, 10*10);
        QKr.set_to_zeros();
        QKv14.set_to_zeros();
        QKv43.set_to_zeros();
        QKr(1,2) = parameters.Kf;
        QKr(2,1) = parameters.Kb;

        dconc.Nai = 0.0;
        dconc.Ki = 0.0;
        dconc.Cai = 0.0;
        dconc.CaNSR = 0.0;
        dconc.CaLTRPN = 0.0;
        dconc.CaHTRPN = 0.0;

        dgates.m = 0.0;
        dgates.h = 0.0;
        dgates.j = 0.0;
        dgates.xKs = 0.0;

        Istim = 0.0;

        std::cout << consts.CRU_factor << std::endl;
        std::cout << consts.ICaL_const << std::endl;
        std::cout << consts.JLCC_const << std::endl;
    }

    void euler_step(const double dt);
    void euler(const double dt, const int nstep, const std::function<double(double)> Is);
    void euler_write(const double dt, const int nstep, const std::function<double(double)> Is, ofstream &file, const int record_every);


};

void GW_model::update_dconc_and_dV()
{
    double INa_, INab_, INaCa_, INaK_, IKr_, IKs_, Ito1_, Ito2_, IK1_, IKp_, ICaL_, ICab_, IpCa_; 
    double Jup_, Jtr_av_, Jxfer_av_, dCaLTRPN_, dCaHTRPN_, beta_cyto_;

    double Nai = concentrations.Nai, Ki = concentrations.Ki, Cai = concentrations.Cai, CaNSR = concentrations.CaNSR;
    double CaLTRPN = concentrations.CaLTRPN, CaHTRPN = concentrations.CaHTRPN;

    const double ENa = Nernst(Nai, parameters.Nao);
    const double EK = Nernst(Ki, parameters.Ko);

    INa_ = INa(V, gates.m, gates.h, gates.j, ENa, parameters.GNa);
    INab_ = INab(V, ENa, parameters.GNab);
    INaCa_ = INaCa(VFRT, expmVFRT, Nai, Cai, consts.Nao3, parameters.Cao, parameters.eta, consts.INaCa_const, parameters.ksat);
    INaK_ = INaK(VFRT, expmVFRT, Nai, consts.sigma, parameters.KmNai, consts.INaK_const);

    IKr_ = IKr(V, Kr[3], EK, parameters.GKr, consts.sqrtKo);
    IKs_ = IKs(V, gates.xKs, Ki, Nai, parameters.Nao, parameters.Ko, parameters.GKs);
    Ito1_ = Ito1(V, VFRT, expmVFRT, Kv14[4], Kv43[4], Ki, Nai, EK, consts.PKv14_Csc, parameters.Nao, parameters.Ko, parameters.GKv43);
    Ito2_ = Ito2(ClCh, VFRT, expmVFRT, parameters.Clcyto, parameters.Clo, consts.Ito2_const);
    IK1_ = IK1(V, EK, parameters.GK1, consts.IK1_const);
    IKp_ = IKp(V, EK, parameters.GKp);


    ICaL_ = ICaL(JLCC, consts.ICaL_const);
    ICab_ = ICab(V, Cai, parameters.Cao, parameters.GCab);
    IpCa_ = IpCa(Cai, parameters.IpCamax, parameters.KmpCa);

    Jup_ = Jup(Cai, CaNSR, parameters.Vmaxf, parameters.Vmaxr, parameters.Kmf, parameters.Kmr, parameters.Hf, parameters.Hr);
    Jtr_av_ = flux_average(Jtr, consts.CRU_factor);
    Jxfer_av_ = flux_average(Jxfer, consts.CRU_factor);

    dCaLTRPN_ = dTRPNCa(CaLTRPN, Cai, parameters.LTRPNtot, parameters.kLTRPNp, parameters.kLTRPNm);
    dCaHTRPN_ = dTRPNCa(CaHTRPN, Cai, parameters.HTRPNtot, parameters.kHTRPNp, parameters.kHTRPNm);
    
    beta_cyto_ = beta_cyto(Cai, consts.CMDN_const, parameters.KCMDN);

    dconc.Nai = -consts.CSA_FVcyto * (INa_ + INab_ + 3*INaCa_ + 3*INaK_);
    dconc.Ki = -consts.CSA_FVcyto * (IKr_ + IKs_ + Ito1_ + IK1_ + IKp_ - 2*INaK_);
    dconc.Cai = beta_cyto_ * (-0.5*consts.CSA_FVcyto*(ICab_ + IpCa_ - 2*INaCa_) + consts.VSS_Vcyto*Jxfer_av_ - Jup_ - (dCaLTRPN_ + dCaHTRPN_));
    dconc.CaNSR = consts.Vcyto_VNSR * Jup_ - consts.VJSR_VNSR * Jtr_av_;
    dconc.CaLTRPN = dCaLTRPN_;
    dconc.CaHTRPN = dCaHTRPN_;

    dV = Istim -(INa_ + ICaL_ + IKr_ + IKs_ + Ito1_ + IK1_ + IKp_ + Ito2_ + INaK_ + INaCa_ + IpCa_ + ICab_ + INab_);
}

void GW_model::update_dgates(){
    dgates.m = alpham(V) * (1.0 - gates.m) - betam(V) * gates.m;
    dgates.h = alphah(V) * (1.0 - gates.h) - betah(V) * gates.h;
    dgates.j = alphaj(V) * (1.0 - gates.j) - betaj(V) * gates.j;
    dgates.xKs =  (XKsinf(V) - gates.xKs) / tauXKs(V);
}
    
void GW_model::update_dKr_and_dKV(){
    update_QKr(QKr, V);
    update_QKv(QKv14, V, parameters.alphaa0Kv14, parameters.aaKv14, parameters.alphai0Kv14, parameters.aiKv14, parameters.betaa0Kv14, 
               parameters.baKv14, parameters.betai0Kv14, parameters.biKv14, parameters.f1Kv14, parameters.f2Kv14, parameters.f3Kv14,
               parameters.f4Kv14, parameters.b1Kv14, parameters.b2Kv14, parameters.b3Kv14, parameters.b4Kv14);
    update_QKv(QKv43, V, parameters.alphaa0Kv43, parameters.aaKv43, parameters.alphai0Kv43, parameters.aiKv43, parameters.betaa0Kv43, 
               parameters.baKv43, parameters.betai0Kv43, parameters.biKv43, parameters.f1Kv43, parameters.f2Kv43, parameters.f3Kv43,
               parameters.f4Kv43, parameters.b1Kv43, parameters.b2Kv43, parameters.b3Kv43, parameters.b4Kv43);

    update_Kr_derivative(dKr, Kr, QKr);
    update_Kv_derivative(dKv43, Kv43, QKv43);
    update_Kv_derivative(dKv14, Kv14, QKv14);
}


void GW_model::euler_step(const double dt){
    VFRT = V*FRT;
    expmVFRT = exp(-VFRT);
    update_dconc_and_dV();
    update_dgates();
    update_dKr_and_dKV();

    SSA(LCC, LCC_activation, RyR, ClCh, CaSS, CaJSR, concentrations.Cai, concentrations.CaNSR, JLCC, Jxfer, Jtr, V, expmVFRT, dt, nCRU, consts);

    V += (dt*dV);

    concentrations.Nai += (dt*dconc.Nai);
    concentrations.Ki += (dt*dconc.Ki);
    concentrations.Cai += (dt*dconc.Cai);
    concentrations.CaNSR += (dt*dconc.CaNSR);
    concentrations.CaLTRPN += (dt*dconc.CaLTRPN);
    concentrations.CaHTRPN += (dt*dconc.CaHTRPN);

    gates.m += (dt*dgates.m);
    gates.h += (dt*dgates.h);
    gates.j += (dt*dgates.j);
    gates.xKs += (dt*dgates.xKs);

    assert((dKr[0]+dKr[1]+dKr[2]+dKr[3]+dKr[4]) == 0.0);

    Kr[0] += (dt*dKr[0]);
    Kr[1] += (dt*dKr[1]);
    Kr[2] += (dt*dKr[2]);
    Kr[3] += (dt*dKr[3]);
    Kr[4] += (dt*dKr[4]);

    double sum_Kv14 = 0.0, sum_Kv43 = 0.0;
    for (int j = 0; j < 9; j++){
        Kv14[j] += (dt*dKv14[j]);
        Kv43[j] += (dt*dKv14[j]);
        sum_Kv14 += Kv14[j];
        sum_Kv43 += Kv43[j];
    }
    Kv14[9] = 1.0 - sum_Kv14;
    Kv43[9] = 1.0 - sum_Kv43;
}

void GW_model::euler(const double dt, const int nstep, const std::function<double(double)> Ist)
{
    double t = 0.0;
    for (int i = 0; i < nstep; i++){
        Istim = Ist(t);
        euler_step(dt);
        t += dt;
    }
}

void write_header(ofstream &file, const int nCRU){
    file << "t,V,m,h,j,Nai,Ki,Cai,CaNSR,CaLTRPN,CaHTRPN,xKs,Kr1,Kr2,Kr3,Kr4,Kr5,Kv14_1,Kv14_2,Kv14_3,Kv14_4,Kv14_5,Kv14_6,Kv14_7,Kv14_8,Kv14_9,Kv14_10,Kv43_1,Kv43_2,Kv43_3,Kv43_4,Kv43_5,Kv43_6,Kv43_7,Kv43_8,Kv43_9,Kv43_10,";
    file << "CaJSRs,CaSSs,LCCs,LCCas,RyRs,ClChs" << std::endl;
    //for (int i = 0; i < nCRU; i++){
    //    file << ",CaJSR" << i << ",CaSS1" << i << ",CaSS2" << i << ",CaSS3" << i << ",CaSS4" << i 
    //         << ",LCC1" << i << ",LCC2" << i << ",LCC3" << i << ",LCC4" << i << ",LCCa1" << i << ",LCCa2" << i << ",LCCa3" 
    //         << i << ",LCCa4" << i << ",RyR1" << i << ",RyR2" << i << ",RyR3" << i << ",RyR4"
    //         << i << ",ClCh1" << i << ",ClCh2" << i << ",ClCh3" << i << ",ClCh4" << i; 
    //}i
    //file << '\n';
}

void GW_model::write_state(ofstream &file, const double t){
    int nlcc = 0, nryr = 0;
    for (int i = 0; i < nCRU; i++){
        for (int j = 0; j < 4; j++){
            if ((LCC(i,j) == 6 || LCC(i,j) == 12) && LCC_activation(i,j) == 1)
                nlcc++;
            
            nryr += (RyR(i,j,2) + RyR(i,j,3));
        }
    }
    file << t << ',' << V << ',' << gates.m << ',' << gates.h << ',' << gates.j << ',' << concentrations.Nai << ',' << concentrations.Ki << ','
         << concentrations.Cai << ',' << concentrations.CaNSR << ',' << concentrations.CaLTRPN << ',' << concentrations.CaHTRPN << ','
         << gates.xKs << ',' << Kr[0] << ',' << Kr[1] << ',' << Kr[2] << ',' << Kr[3] << ',' << Kr[4] << ',' << Kv14[0] << ',' 
         << Kv14[1] << ',' << Kv14[2] << ',' << Kv14[3] << ',' << Kv14[4] << ',' << Kv14[5] << ',' << Kv14[6] << ',' << Kv14[7]
         << ',' << Kv14[8] << ',' << Kv14[9] << ',' << Kv43[0] << ',' << Kv43[1] << ',' << Kv43[2] << ',' << Kv43[3] << ',' << Kv43[4]
         << ',' << Kv43[5] << ',' << Kv43[6] << ',' << Kv43[7] << ',' << Kv43[8] << ',' << Kv43[9];
    file << ',' << CaJSR.sum() << ',' << CaSS.sum() << ',' << nlcc << ',' << LCC_activation.sum() << ',' << nryr << ',' << ClCh.sum() << std::endl;
    //for (int i = 0; i < nCRU; i++){
    //    file << ',' << CaJSR(i) << ',' << CaSS(i,0) << ',' << CaSS(i,1) << ',' << CaSS(i,2) << ',' << CaSS(i,3) << ','
    //         << LCC(i,0) << ',' << LCC(i,1) << ',' << LCC(i,2) << ',' << LCC(i,3) << ',' << LCC_activation(i,0) << ','
    //         << LCC_activation(i,1) << ',' << LCC_activation(i,2) << ',' << LCC_activation(i,3) << ',' << RyR(i,0,2)+RyR(i,0,3) << ','
   //          << RyR(i,1,2)+RyR(i,1,3) << ',' << RyR(i,2,2)+RyR(i,2,3) << ',' << RyR(i,3,2)+RyR(i,3,3) << ','
   //          << ClCh(i,0) << ',' << ClCh(i,1) << ',' << ClCh(i,2) << ',' << ClCh(i,3);
    //}
    //file << '\n';
}


void GW_model::euler_write(const double dt, const int nstep, const std::function<double(double)> Ist, ofstream &file, const int record_every)
{
    double t = 0.0;
    write_header(file, nCRU);
    for (int i = 0; i < nstep; i++){
        if (i % record_every == 0){
            write_state(file, t);
            //std::cout << "dV: " << dV << std::endl;
            //std::cout << "dNai: " << dconc.Nai << std::endl;
            //std::cout << "dKi: " << dconc.Ki << std::endl;
            //std::cout << "dCai: " << dconc.Cai << std::endl;
            //std::cout << "dCaNSR: " << dconc.CaNSR << std::endl << std::endl;
        }
        
        Istim = Ist(t);
        euler_step(dt);
        t += dt;

    }
}

double Ist(double t) { return (t < 2.0) ? 35.0 : 0.0; }

int main(int argc, char* argv[])
{
    GW_model model(1000);

    /*
    cout << model.V << endl;
    cout << model.Kr[0] << endl;
    cout << model.Kv14[0] << endl;
    cout << model.Kv43[0] << endl;
    cout << model.CaSS(0,0) << endl;
    cout << model.CaJSR(0) << endl;
    cout << model.LCC(50,2) << endl;
    cout << model.LCC_activation(54,3) << endl;
    cout << model.RyR(32,1,4) << endl;
    cout << model.ClCh(11,0) << endl;
    */

    ofstream file;
    file.open("data.csv", std::ofstream::out | std::ofstream::trunc );
    model.euler_write(1e-3, 50000, &Ist, file, 100);
    file.close();
    cout << model.V << endl;


    return 0;
}