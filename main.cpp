#include "SSA.hpp"
#include "currents.hpp"
#include <iostream>

using namespace std;

void initialise_CaSS(std::vector<double> &CaSS)
{
    for (int i = 0; i < CaSS.size(); i++)
    {
        CaSS[i] = 1.45370e-4;
    }
}

void initialise_CaJSR(std::vector<double> &CaJSR)
{
    for (int i = 0; i < CaJSR.size(); i++)
    {
        CaJSR[i] = 0.908408;
    }
}


void initialise_LCC(std::vector<int> &LCC)
{
    const double weights[3] = { 0.958, 0.038, 0.004 };
    int idx;
    for (int i = 0; i < LCC.size(); i++)
    {
        idx = sample_weights(weights, 1.0, 3);
        if (idx == 0){
            LCC[i] = 1;
        } 
        else if (idx == 1){
            LCC[i] = 2;
        }
        else {
            LCC[i] = 7;
        }
    }
}

void initialise_LCC_a(std::vector<int> &LCC_a)
{
    const double weights[2] = { 0.9425, 0.0575 };
    int idx;
    for (int i = 0; i < LCC_a.size(); i++)
    {
        idx = sample_weights(weights, 1.0, 2);
        if (idx == 0){
            LCC_a[i] = 1;
        } 
        else {
            LCC_a[i] = 0;
        }
    }
}


void initialise_RyR(std::vector<int> &RyR)
{
    const double weights[3] = { 0.609, 0.5*0.391, 0.5*0.391 };
    int idx;
    for (int i = 0; i < RyR.size(); i++)
    {
        idx = sample_weights(weights, 1.0, 3);
        if (idx == 0){
            RyR[i] = 1;
        } 
        else if (idx == 1){
            RyR[i] = 5;
        }
        else {
            RyR[i] = 6;
        }
    }
}


void initialise_ClCh(std::vector<int> &ClCh)
{
    const double weights[2] = { 0.998, 0.002 };
    int idx;
    for (int i = 0; i < ClCh.size(); i++)
    {
        idx = sample_weights(weights, 1.0, 2);
        if (idx == 0){
            ClCh[i] = 0;
        } 
        else {
            ClCh[i] = 1;
        }
    }
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

class GW_model
{
private:

    int nCRU;
    Constants consts;
    double VFRT;
    double expmVFRT;
    double QKr[5*5];
    double QKv14[10*10];
    double QKv43[10*10];

    double dV;
    Concentrations dconc;
    Gates dgates; 
    double dKr[5];
    double dKv43[10];
    double dKv14[10];

    std::vector<double> CaSS_storage;
    std::vector<double> CaJSR_storage;
    std::vector<int> LCC_storage;
    std::vector<int> LCC_a_storage;
    std::vector<int> RyR_storage;
    std::vector<int> ClCh_storage;

    std::vector<double> JLCC_storage;
    std::vector<double> Jxfer_storage;
    std::vector<double> Jtr_storage;



    MatrixMap* CaSS_ptr;
    VectorMap* CaJSR_ptr;
    MatrixMapi* LCC_ptr;
    Array3dMapi* RyR_ptr;
    MatrixMapi* ClCh_ptr;    

    void update_global_variables(const double dt);
    void SSA_step(const double dt);
    
public:

    GW_parameters parameters;
    double V;
    Concentrations concentrations;
    Gates gates;
    double Kr[5];
    double Kv43[10];
    double Kv14[10];

    GW_model() : parameters(), concentrations(), gates() { }

    GW_model(int nCRU_)
    {   
        nCRU = nCRU_
        //parameters();
        consts = consts_from_params(parameters);
        //concentrations();
        //gates();
        
        std::vector<double> CaSS_storage_(nCRU*4);
        std::vector<double> CaJSR_storage_(nCRU);
        std::vector<int> LCC_storage_(nCRU*4);
        std::vector<int> LCC_a_storage_(nCRU*4);
        std::vector<int> RyR_storage_(nCRU*4*6);
        std::vector<int> ClCh_storage_(nCRU*4);
        
        initialise_CaSS(CaSS_storage_);
        initialise_CaJSR(CaJSR_storage_);
        initialise_LCC(LCC_storage_);
        initialise_LCC_a(LCC_a_storage_);
        initialise_RyR(RyR_storage_);
        initialise_ClCh(ClCh_storage_);

        //parameters = params;
        V = -91.382;
        //concentrations = conc;
        //gates = g;

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
        
        CaSS_storage = CaSS_storage_;
        CaJSR_storage = CaJSR_storage_;
        LCC_storage = LCC_storage_;
        LCC_a_storage = LCC_a_storage_;
        RyR_storage = RyR_storage_;
        ClCh_storage = ClCh_storage_;
    }

    double CaSS(const int row, const int col){ 
        if (4 * row + col < CaSS_storage.size()){
            return CaSS_storage[4*row + col];
        }
        else {
            throw(std::invalid_argument("Invalid indices."));
        }
    }
    
    double CaJSR(const int idx){ 
        if (idx < CaJSR_storage.size()){
            return CaJSR_storage[idx];
        }
        else {
            throw(std::invalid_argument("Invalid indices."));
        }
    }


    double LCC(const int row, const int col){ 
        if (4 * row + col < LCC_storage.size()){
            return LCC_storage[4*row + col];
        }
        else {
            throw(std::invalid_argument("Invalid indices."));
        }
    }


    double LCC_activation(const int row, const int col){ 
        if (4 * row + col < LCC_a_storage.size()){
            return LCC_a_storage[4*row + col];
        }
        else {
            throw(std::invalid_argument("Invalid indices."));
        }
    }
    
    double RyR(const int row, const int col, const int num){ 
        if (4 * 6 * row + 6 * col + num < RyR_storage.size()){
            return RyR_storage[4*6*row + 6*col + num];
        }
        else {
            throw(std::invalid_argument("Invalid indices."));
        }
    }

    double ClCh(const int row, const int col){ 
        if (4 * row + col < ClCh_storage.size()){
            return ClCh_storage[4*row + col];
        }
        else {
            throw(std::invalid_argument("Invalid indices."));
        }
    }

};

void GW_model::update_global_variables(const double dt)
{
    const double INa_, INab_, INaCa_, INaK_, IKr_, IKs_, Ito1_, Ito2_, IK1_, IKp_, ICaL_, ICab_, IpCa_; 
    const double Jup_, Jtr_av_, Jxfer_av_, dLTRPNCa_, dHTRPNCa_, beta_cyto_;

    double Nai = concentrations.Nai, Ki = concentrations.Ki, Cai = concentrations.Cai, CaNSR = concentrations.CaNSR;
    double CaLTRPN = concentrations.CaLTRPN, CaHTRPN = concentrations.CaHTRPN;

    const double ENa = Nernst(Nai, parameters.Nao);
    const double EK = Nernst(Ki, parameters.Ko);

    INa_ = INa(V, concentrations.m, concentrations.h, concentrations.j, ENa, parameters.GNa);
    INab_ = INab(V, ENa, parameters.GNab);
    INaCa_ = INaCa(VFRT, expmVFRT, Nai, Cai, consts.Nao3, parameters.Cao, parameters.eta, consts.INaCa_const, parameters.ksat);
    INaK_ = INaK(VFRT, expmVFRT, Nai, consts.sigma, parameters.KmNai, consts.INaK_const);

    IKr_ = IKr(V, Kr[3], EK, parameters.GKr, consts.sqrtKo);
    IKs_ = IKs(V, gates.xKs, Ki, Nai, parameters.Nao, parameters.Ko, parameters.GKs);
    Ito1_ = Ito1(V, VFRT, expmVFRT, Kv14[4], Kv43[5], Ki, Nai, EK, consts.PKv14_Csc, parameters.Nao, parameters.Ko, parameters.GKv43);
    Ito2_ = Ito2(ClCh_storage, VFRT, expmVFRT, parameters.Clcyto, parameters.Clo, consts.Ito2_const);
    IK1_ = IK1(V, EK, parameters.GK1, consts.K_const);
    IKp_ = IKp(V, EK, parameters.GKp);

    ICaL_ = ICaL(JLCC_storage, consts.ICaL_const);
    ICab_ = ICab(V, Cai, parameters.Cao, parameters.GCab);
    IpCa_ = IpCa(Cai, parameters.IpCamax, parameters.KmpCa);

    Jup_ = Jup(Cai, CaNSR, parameters.Vmaxf, parameters.Vmaxr, parameters.Kmf, parameters.Kmr, parameters.Hf, parameters.Hr);
    Jtr_av_ = flux_average(Jtr_storage, consts.CRU_factor);
    Jxfer_av_ = flux_average(Jxfer_storage, consts.CRU_factor);

    dLTRPNCa_ = dTRPNCa(CaLTRPN, Cai, parameters.LTRPNtot, parameters.kLTRPNp, parameters.kLTRPNm);
    dHTRPNCa_ = dTRPNCa(CaHTRPN, Cai, parameters.HTRPNtot, parameters.kHTRPNp, parameters.kHTRPNm);
    
    beta_cyto_ = beta_cyto(Cai, consts.CMDNconst, parameters.KCMDN);

    dconc[0] = -constants.dconc_mul * (INa_ + INab_ + 3*INaCa_ + 3*INaK_);
    dconc[1] = -constants.dconc_mul * (IKr_ + IKs_ + Ito1_ + IK1_ + IKp_ - 2*INaK_);

}

int main(int argc, char* argv[])
{
    GW_model model(100);

    //cout << model.parameters->a << endl;
    cout << model.V << endl;
    //cout << model.concentrations->Nai << endl;
    //cout << model.gates->m << endl;
    cout << model.Kr[0] << endl;
    cout << model.Kv14[0] << endl;
    cout << model.Kv43[0] << endl;
    cout << model.CaSS(0,0) << endl;
    cout << model.CaJSR(0) << endl;
    cout << model.LCC(0,0) << endl;
    cout << model.LCC_activation(0,0) << endl;
    cout << model.RyR(0,0,0) << endl;
    cout << model.ClCh(0,0) << endl;

    return 0;
}