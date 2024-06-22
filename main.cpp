#include "SSA.hpp"
#include "currents.hpp"
#include <iostream>
#include <iomanip>
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
    for (int i = 0; i < ClCh.shape(0); i++){
        for (int j = 0; j < 4; j++){
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
    for (int i = 0; i < JLCC.shape(0); i++){
        for (int j = 0; j < 4; j++){
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
    double dKv43[10];
    double dKv14[10];

    NDArray<double,2> JLCC;
    NDArray<double,2> Jxfer;
    NDArray<double,1> Jtr;

    double Istim;

    double INa;
    double INab;
    double INaCa;
    double INaK;

    double IKr;
    double IKs;
    double Ito1;
    double Ito2;
    double IK1;
    double IKp;

    double ICaL;
    double ICab;
    double IpCa;
    
    double Jup;
    double Jtr_tot;
    double Jxfer_tot;


    void update_dconc_and_dV(const double dt);
    void update_dgates(const double dt);
    void update_dKr_and_dKV(const double dt);
    
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
    }

    void euler_step(const double dt);
    void euler(const double dt, const int nstep, const std::function<double(double)> Is);
    void euler_write(const double dt, const int nstep, const std::function<double(double)> Is, ofstream &file, const int record_every);


};

void GW_model::update_dconc_and_dV(const double dt)
{
    double beta_cyto_;

    double Nai = concentrations.Nai, Ki = concentrations.Ki, Cai = concentrations.Cai, CaNSR = concentrations.CaNSR;
    double CaLTRPN = concentrations.CaLTRPN, CaHTRPN = concentrations.CaHTRPN;

    const double ENa = currents::Nernst(Nai, parameters.Nao);
    const double EK = currents::Nernst(Ki, parameters.Ko);

    INa = currents::INa(V, gates.m, gates.h, gates.j, ENa, parameters.GNa);
    INab = currents::INab(V, ENa, parameters.GNab);
    INaCa = currents::INaCa(VFRT, expmVFRT, Nai, Cai, consts.Nao3, parameters.Cao, parameters.eta, consts.INaCa_const, parameters.ksat);
    INaK = currents::INaK(VFRT, expmVFRT, Nai, consts.sigma, parameters.KmNai, consts.INaK_const);

    IKr = currents::IKr(V, Kr[3], EK, parameters.GKr, consts.sqrtKo);
    IKs = currents::IKs(V, gates.xKs, Ki, Nai, parameters.Nao, parameters.Ko, parameters.GKs);
    Ito1 = currents::Ito1(V, VFRT, expmVFRT, Kv14[4], Kv43[4], Ki, Nai, EK, consts.PKv14_Csc, parameters.Nao, parameters.Ko, parameters.GKv43);
    Ito2 = currents::Ito2(ClCh, VFRT, expmVFRT, parameters.Clcyto, parameters.Clo, consts.Ito2_const);
    IK1 = currents::IK1(V, EK, parameters.GK1, consts.IK1_const);
    IKp = currents::IKp(V, EK, parameters.GKp);

    ICaL = currents::ICaL(JLCC, consts.ICaL_const);
    ICab = currents::ICab(V, Cai, parameters.Cao, parameters.GCab);
    IpCa = currents::IpCa(Cai, parameters.IpCamax, parameters.KmpCa);

    Jup = currents::Jup(Cai, CaNSR, parameters.Vmaxf, parameters.Vmaxr, parameters.Kmf, parameters.Kmr, parameters.Hf, parameters.Hr);
    Jtr_tot = currents::flux_average(Jtr, consts.CRU_factor);
    Jxfer_tot = currents::flux_average(Jxfer, consts.CRU_factor);
    beta_cyto_ = currents::beta_cyto(Cai, consts.CMDN_const, parameters.KCMDN);

    dconc.CaLTRPN = currents::dTRPNCa(CaLTRPN, Cai, parameters.LTRPNtot, parameters.kLTRPNp, parameters.kLTRPNm);
    dconc.CaHTRPN = currents::dTRPNCa(CaHTRPN, Cai, parameters.HTRPNtot, parameters.kHTRPNp, parameters.kHTRPNm);

    dconc.Nai = -dt*consts.CSA_FVcyto * (INa + INab + 3*INaCa + 3*INaK);
    dconc.Ki = -dt*consts.CSA_FVcyto * (IKr + IKs + Ito1 + IK1 + IKp - 2*INaK);
    dconc.Cai = dt*beta_cyto_ * (-0.5*consts.CSA_FVcyto*(ICab + IpCa - 2*INaCa) + consts.VSS_Vcyto*Jxfer_tot - Jup - (dconc.CaLTRPN + dconc.CaHTRPN));
    dconc.CaNSR = dt*(consts.Vcyto_VNSR * Jup - consts.VJSR_VNSR * Jtr_tot);
    dconc.CaLTRPN *= dt;
    dconc.CaHTRPN *= dt;

    dV = dt*(Istim - (INa + ICaL + IKr + IKs + Ito1 + IK1 + IKp + Ito2 + INaK + INaCa + IpCa + ICab + INab));
}

void GW_model::update_dgates(const double dt){
    dgates.m = dt * (alpham(V) * (1.0 - gates.m) - betam(V) * gates.m);
    dgates.h = dt * (alphah(V) * (1.0 - gates.h) - betah(V) * gates.h);
    dgates.j = dt * (alphaj(V) * (1.0 - gates.j) - betaj(V) * gates.j);
    dgates.xKs =  dt * (XKsinf(V) - gates.xKs) / tauXKs(V);
}
    
void GW_model::update_dKr_and_dKV(const double dt){
    update_QKr(QKr, V, parameters);
    update_QKv(QKv14, V, parameters.alphaa0Kv14, parameters.aaKv14, parameters.alphai0Kv14, parameters.aiKv14, parameters.betaa0Kv14, 
               parameters.baKv14, parameters.betai0Kv14, parameters.biKv14, parameters.f1Kv14, parameters.f2Kv14, parameters.f3Kv14,
               parameters.f4Kv14, parameters.b1Kv14, parameters.b2Kv14, parameters.b3Kv14, parameters.b4Kv14);
    update_QKv(QKv43, V, parameters.alphaa0Kv43, parameters.aaKv43, parameters.alphai0Kv43, parameters.aiKv43, parameters.betaa0Kv43, 
               parameters.baKv43, parameters.betai0Kv43, parameters.biKv43, parameters.f1Kv43, parameters.f2Kv43, parameters.f3Kv43,
               parameters.f4Kv43, parameters.b1Kv43, parameters.b2Kv43, parameters.b3Kv43, parameters.b4Kv43);

    update_Kr_derivative(dKr, Kr, QKr, dt);
    update_Kv_derivative(dKv43, Kv43, QKv43, dt);
    update_Kv_derivative(dKv14, Kv14, QKv14, dt);
}


void GW_model::euler_step(const double dt){
    VFRT = V*FRT;
    expmVFRT = exp(-VFRT);
    update_dconc_and_dV(dt);
    update_dgates(dt);
    update_dKr_and_dKV(dt);

    SSA(LCC, LCC_activation, RyR, ClCh, CaSS, CaJSR, concentrations.Cai, concentrations.CaNSR, JLCC, Jxfer, Jtr, V, expmVFRT, dt, nCRU, consts);

    V += dV;

    concentrations.Nai += dconc.Nai;
    concentrations.Ki += dconc.Ki;
    concentrations.Cai += dconc.Cai;
    concentrations.CaNSR += dconc.CaNSR;
    concentrations.CaLTRPN += dconc.CaLTRPN;
    concentrations.CaHTRPN += dconc.CaHTRPN;

    gates.m += dgates.m;
    gates.h += dgates.h;
    gates.j += dgates.j;
    gates.xKs += dgates.xKs;

    Kr[0] += dKr[0];
    Kr[1] += dKr[1];
    Kr[2] += dKr[2];
    Kr[3] += dKr[3];
    Kr[4] += dKr[4];

    for (int j = 0; j < 10; j++){
        Kv14[j] += dKv14[j];
        Kv43[j] += dKv43[j];
    }
}

void GW_model::euler(const double dt, const int nstep, const std::function<double(double)> Ist){
    double t = 0.0;
    for (int i = 0; i < nstep; i++){
        Istim = Ist(t);
        euler_step(dt);
        t += dt;
    }
}

void write_header(ofstream &file, const int nCRU){
    file << "t,V,m,h,j,Nai,Ki,Cai,CaNSR,CaLTRPN,CaHTRPN,xKs,Kr1,Kr2,Kr3,Kr4,Kr5,Kv14_1,Kv14_2,Kv14_3,Kv14_4,Kv14_5,Kv14_6,Kv14_7,Kv14_8,Kv14_9,Kv14_10,Kv43_1,Kv43_2,Kv43_3,Kv43_4,Kv43_5,Kv43_6,Kv43_7,Kv43_8,Kv43_9,Kv43_10,";
    file << "CaJSRs,CaSSs,LCC1,LCC2,LCC3,LCC4,LCC5,LCC6,LCC7,LCC8,LCC9,LCC10,LCC11,LCC12,LCCas,RyRs,ClChs,";
    file << "INa,ICaL,IKr,IKs,Ito1,IK1,IKp,Ito2,INaK,INaCa,IpCa,ICab,INab,JLCC" << std::endl;
}

void GW_model::write_state(ofstream &file, const double t){
    int nlcc1 = 0, nlcc2 = 0, nlcc3 = 0, nlcc4 = 0, nlcc5 = 0, nlcc6 = 0, nlcc7 = 0, nlcc8 = 0, nlcc9 = 0, nlcc10 = 0, nlcc11 = 0, nlcc12 = 0, nryr = 0;
    for (int i = 0; i < nCRU; i++){
        for (int j = 0; j < 4; j++){
            switch (LCC(i,j)){
            case 1:
                nlcc1++;
                break;
            case 2:
                nlcc2++;
                break;
            case 3:
                nlcc3++;
                break;
            case 4:
                nlcc4++;
                break;
            case 5:
                nlcc5++;
                break;
            case 6:
                nlcc6++;
                break;
            case 7:
                nlcc7++;
                break;
            case 8:
                nlcc8++;
                break;
            case 9:
                nlcc9++;
                break;
            case 10:
                nlcc10++;
                break;
            case 11:
                nlcc11++;
                break;
            case 12:
                nlcc12++;
                break;
            default:
                break;
            }
            
            nryr += (RyR(i,j,2) + RyR(i,j,3));
        }
    }
    file << t << ',' << V << ',' << gates.m << ',' << gates.h << ',' << gates.j << ',' << concentrations.Nai << ',' << concentrations.Ki << ','
         << concentrations.Cai << ',' << concentrations.CaNSR << ',' << concentrations.CaLTRPN << ',' << concentrations.CaHTRPN << ','
         << gates.xKs << ',' << Kr[0] << ',' << Kr[1] << ',' << Kr[2] << ',' << Kr[3] << ',' << Kr[4] << ',' << Kv14[0] << ',' 
         << Kv14[1] << ',' << Kv14[2] << ',' << Kv14[3] << ',' << Kv14[4] << ',' << Kv14[5] << ',' << Kv14[6] << ',' << Kv14[7]
         << ',' << Kv14[8] << ',' << Kv14[9] << ',' << Kv43[0] << ',' << Kv43[1] << ',' << Kv43[2] << ',' << Kv43[3] << ',' << Kv43[4]
         << ',' << Kv43[5] << ',' << Kv43[6] << ',' << Kv43[7] << ',' << Kv43[8] << ',' << Kv43[9];
         
    file << ',' << CaJSR.sum() << ',' << CaSS.sum() << ',' << nlcc1 << ',' << nlcc2 << ',' << nlcc3 << ',' << nlcc4 << ',' 
         << nlcc5 << ',' << nlcc6 << ',' << nlcc7 << ',' << nlcc8 << ',' << nlcc9 << ',' << nlcc10 << ',' << nlcc11 << ',' << nlcc12 << ',' 
         << LCC_activation.sum() << ',' << nryr << ',' << ClCh.sum();

    file << ',' << INa << ',' << ICaL << ',' << IKr << ',' << IKs << ',' << Ito1 << ',' << IK1 << ',' << IKp << ',' << Ito2 << ',' << INaK
         <<  ',' << INaCa << ',' << IpCa << ',' << ICab << ',' << INab << std::endl; 
}


void GW_model::euler_write(const double dt, const int nstep, const std::function<double(double)> Ist, ofstream &file, const int record_every){
    double t = 0.0;
    write_header(file, nCRU);
    for (int i = 0; i < nstep; i++){ 
        Istim = Ist(t);
        euler_step(dt);
        t += dt;

         if (i % record_every == 0){
            write_state(file, t);
        }

    }
}

double Ist(double t) { return (t < 2.0) ? 35.0 : 0.0; }

int main(int argc, char* argv[])
{
    GW_model model(2000);

    ofstream file;
    file.open("data.csv", std::ofstream::out | std::ofstream::trunc );
    file << std::setprecision(12);
    model.euler_write(1e-3, 500000, &Ist, file, 2000);
    file.close();
    cout << model.V << endl;


    return 0;
}