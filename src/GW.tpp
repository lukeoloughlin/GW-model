#include "GW.hpp"

namespace GW {

template <typename FloatType>
Constants<FloatType>::Constants(const Parameters<FloatType> &params, const int nCRU_simulated){
    RT_F = GAS_CONST * params.T / FARADAY;
    CRU_factor = FloatType(params.NCaRU) / FloatType(nCRU_simulated);
    CSA_FVcyto = params.CSA / (1000.0 * params.Vcyto * FARADAY);
    VSS_Vcyto = params.VSS / params.Vcyto;
    Vcyto_VNSR = params.Vcyto / params.VNSR;
    VJSR_VNSR = params.VJSR / params.VNSR;
    FRT = FARADAY / (GAS_CONST * params.T);
    riss = params.riss;
    rxfer = params.rxfer;
    // LCC rates
    gamma0 = params.gamma0;
    omega = params.omega;
    a = params.a;
    a2 = consts.a*params.a;
    a3 = consts.a2*params.a;
    a4 = consts.a3*params.a;
    bi = 1.0 / params.b; // 1/b
    bi2 = consts.bi*consts.bi; // 1/b^2
    bi3 = consts.bi2*consts.bi; // 1/b^3
    bi4 = consts.bi3*consts.bi; // 1/b^4
    f = params.f;
    g = params.g;
    f1 = params.f1;
    g1 = params.g1;
    // RyR rates
    k12 = params.k12;
    k21 = params.k21;
    k23 = params.k23;
    k32 = params.k32;
    k34 = params.k34;
    k43 = params.k43;
    k45 = params.k45;
    k54 = params.k54;
    k56 = params.k56;
    k65 = params.k65;
    k25 = params.k25;
    k52 = params.k52;
    // ClCh rates
    kfClCh = params.kfClCh;
    kbClCh = params.kbClCh;
    // CaSS constants
    KBSR = params.KBSR;
    BSR_const = params.KBSR * params.BSRT;
    KBSL = params.KBSL;
    BSL_const = params.KBSL * params.BSLT;
    
    VSS_VJSR = params.VSS / params.VJSR;
    KCSQN = params.KCSQN;
    CSQN_const = params.KCSQN * params.CSQNT;
    // JLCC constants
    JLCC_const = 2.0e6 * params.PCaL / params.VSS;
    Cao_scaled = 0.341 * params.Cao;
    // Jrel constants
    rRyR = params.rRyR;
    // Jtr constants
    rtr = params.rtr;
    // INaCa consts
    Nao3 = params.Nao*params.Nao*params.Nao;
    INaCa_const = 5000.0 * params.kNaCa / ((params.KmNa*params.KmNa*params.KmNa + params.Nao*params.Nao*params.Nao) * (params.KmCa + params.Cao));
    // INaK coonsts
    sigma = (exp(params.Nao / 67.3) - 1.0) / 7.0;
    INaK_const = params.INaKmax * params.Ko / (params.Ko + params.KmKo);
    // IKr consts
    sqrtKo = sqrt(params.Ko);
    // Ito1 consts
    PKv14_Csc = params.PKv14 / params.Csc;
    // Ito2 consts
    Ito2_const = 1.0e9 * params.Pto2 * FARADAY * (double(params.NCaRU) / double(nCRU_simulated)) / params.CSA;
    // IK1 consts
    IK1_const = params.Ko / (params.Ko + params.KmK1);
    // ICaL consts
    ICaL_const = -1000.0 * (2.0*FARADAY * params.VSS) * (double(params.NCaRU) / double(nCRU_simulated)) / params.CSA;
    // CMDN_consts
    CMDN_const = params.KCMDN * params.CMDNT;
}

void initialise_LCC(NDArray<int,2> &LCC){
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

void initialise_LCC_a(NDArray<int,2> &LCC_a){
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

template <typename FloatType>
GlobalState<FloatType>::GlobalState(FloatType val){
    V = val;
    Nai = val;
    Ki = val;
    Cai = val;
    CaNSR = val;
    CaLTRPN = val;
    CaHTRPN = val;
    m = val;
    h = val;
    j = val;
    xKs = val;

    memset(Kr, val, 5*sizeof(FloatType));
    memset(Kv43, val, 10*sizeof(FloatType));
    memset(Kv14, val, 10*sizeof(FloatType));
}

template <typename FloatType>
void GW_model<FloatType>::initialise_JLCC(){
    const FloatType exp_term = exp(2*VFRT);
    for (int i = 0; i < JLCC.shape(0); ++i){
        for (int j = 0; j < 4; j++){
            if ((CRUs.LCC_activation(i,j) == 1) && (CRUs.LCC(i,j) == 6 || CRUs.LCC(i,j) == 12))
                JLCC(i,j) = consts.JLCC_const * VFRT * (consts.Cao_scaled - exp_term * CRUs.CaSS(i,j));
            else
                JLCC(i,j) = 0.0;
        }
    }
}

template <typename FloatType>
void GW_model<FloatType>::initialise_Jxfer(){
    Jxfer.set_to_zeros();
    Jxfer += CRUs.CaSS;
    Jxfer -= globals.Cai;
    Jxfer *= params.rxfer;
}

template <typename FloatType>
void GW_model<FloatType>::initialise_Jtr(){
    Jtr.set_to_zeros();
    Jtr += globals.CaNSR;
    Jtr -= CRUs.CaJSR;
    Jtr *= params.rtr;
}

template <typename FloatType>
void GW_model<FloatType>::initialise_QKr(){
    QKr.set_to_zeros();
    QKr(1,2) = parameters.Kf;
    QKr(2,1) = parameters.Kb;
}

template <typename FloatType>
CRUState<FloatType>::CRUState(const int nCRU) : CaSS(NDArray<FloatType,2>(nCRU,4)), CaJSR(NDArray<FloatType,1>(nCRU)), 
                                                LCC(NDArray<int,2>(nCRU,4)), LCC_activation(NDArray<int,2>(nCRU,,4)),
                                                RyR(NDArray<int,3>(nCRU,4,6)), ClCh(NDArray<int,2>(nCRU,4))
{
    CaSS.set_to_val(1.45370e-4);
    CaJSR.set_to_val(0.908408);
    initialise_LCC(LCC);
    initialise_LCC_a(LCC_activation);
    initialise_RyR(RyR);
    initialise_ClCh(ClCh);
}

template <typename FloatType>
void GW_model<FloatType>::update_QKr(){
    QKr(0,1) = 0.0069*exp(0.0272*globals.V);
    QKr(0,0) = -QKr(0,1);

    QKr(1,0) = 0.0227*exp(-0.0431*globals.V);
    //QKr(1,2) = params.Kf;
    QKr(1,1) = -(QKr(1,0) + QKr(1,2));

    //QKr(2,1) = params.Kb;
    QKr(2,3) = 0.0218*exp(0.0262*globals.V);
    QKr(2,4) = 1.29e-5*exp(2.71e-6*globals.V);
    QKr(2,2) = -(QKr(2,1) + QKr(2,3) + QKr(2,4));    

    QKr(3,2) = 0.0009*exp(-0.0269*globals.V);
    QKr(3,4) = 0.0622*exp(0.0120*globals.V);
    QKr(3,3) = -(QKr(3,2) + QKr(3,4));

    QKr(4,3) = 0.0059*exp(-0.0443*globals.V);
    QKr(4,2) = QKr(3,2)*QKr(4,3)*QKr(2,4)/(QKr(2,3)*QKr(3,4));
    QKr(4,4) = -(QKr(4,2) + QKr(4,3));
}

//void update_QKv(NDArrayMap<double,2> &Q, const double V, const double alphaa0, const double aa, const double alphai0, const double ai, 
//                const double betaa0, const double ba, const double betai0, const double bi, const double f1, const double f2,
//                const double f3, const double f4, const double b1, const double b2, const double b3, const double b4)
template <typename FloatType>
void GW_model<FloatType>::update_QKv(){
    const FloatType alphaa14 = parameters.alphaa0Kv14 * exp(parameters.aaKv14 * globals.V);
    const FloatType alphaa43 = parameters.alphaa0Kv43 * exp(parameters.aaKv43 * globals.V);

    const FloatType alphai14 = parameters.alphai0Kv14 * exp(-parameters.aiKv14 * globals.V);
    const FloatType alphai43 = parameters.alphai0Kv43 * exp(-parameters.aiKv43 * globals.V);

    const FloatType betaa14 = parameters.betaa0Kv14 * exp(-parameters.baKv14 * globals.V);
    const FloatType betaa43 = parameters.betaa0Kv43 * exp(-parameters.baKv43 * globals.V);

    const FloatType betai14 = parameters.betai0Kv14 * exp(parameters.biKv14 * globals.V);
    const FloatType betai43 = parameters.betai0Kv43 * exp(parameters.biKv43 * globals.V);

    QKv14(0,1) = 4*alphaa14; QKv43(0,1) = 4*alphaa43;
    QKv14(0,5) = betai14; QKv43(0,5) = betai43;
    QKv14(0,0) = -(QKv14(0,1)+QKv14(0,5)); QKv43(0,0) = -(QKv43(0,1)+QKv43(0,5));

    QKv14(1,0) = betaa14; QKv43(1,0) = betaa43;
    QKv14(1,2) = 3*alphaa14; QKv43(1,2) = 3*alphaa43;
    QKv14(1,6) = parameters.f1Kv14*betai14; QKv43(1,6) = parameters.f1Kv43*betai43;
    QKv14(1,1) = -(QKv14(1,0) + QKv14(1,2) + QKv14(1,6)); QKv43(1,1) = -(QKv43(1,0) + QKv43(1,2) + QKv43(1,6));
    
    QKv14(2,1) = 2*betaa14; QKv43(2,1) = 2*betaa43;
    QKv14(2,3) = 2*alphaa14; QKv43(2,3) = 2*alphaa43;
    QKv14(2,7) = parameters.f2Kv14*betai14; QKv43(2,7) = parameters.f2Kv43*betai43;
    QKv14(2,2) = -(QKv14(2,1) + QKv14(2,3) + QKv14(2,7)); QKv43(2,2) = -(QKv43(2,1) + QKv43(2,3) + QKv43(2,7));
    
    QKv14(3,2) = 3*betaa14; QKv43(3,2) = 3*betaa43;
    QKv14(3,4) = alphaa14; QKv43(3,4) = alphaa43;
    QKv14(3,8) = parameters.f3Kv14*betai14; QKv43(3,8) = parameters.f3Kv43*betai43;
    QKv14(3,3) = -(QKv14(3,2) + QKv14(3,4) + QKv14(3,8)); QKv43(3,3) = -(QKv43(3,2) + QKv43(3,4) + QKv43(3,8));
    
    QKv14(4,3) = 4*betaa14; QKv43(4,3) = 4*betaa43;
    QKv14(4,9) = parameters.f4Kv14*betai14; QKv43(4,9) = parameters.f4Kv43*betai43;
    QKv14(4,4) = -(QKv14(4,3) + QKv14(4,9)); QKv43(4,4) = -(QKv43(4,3) + QKv43(4,9));
    
    QKv14(5,6) = 4*alphaa14*parameters.b1Kv14; QKv43(5,6) = 4*alphaa43*parameters.b1Kv43;
    QKv14(5,0) = alphai14; QKv43(5,0) = alphai43;
    QKv14(5,5) = -(QKv14(5,6) + QKv14(5,0)); QKv43(5,5) = -(QKv43(5,6) + QKv43(5,0));

    QKv14(6,5) = betaa14 / parameters.f1Kv14; QKv43(6,5) = betaa43 / parameters.f1Kv43;
    QKv14(6,7) = 3*alphaa14 * parameters.b2Kv14/parameters.b1Kv14; QKv43(6,7) = 3*alphaa43 * parameters.b2Kv43/parameters.b1Kv43;
    QKv14(6,1) = alphai14 / parameters.b1Kv14; QKv43(6,1) = alphai43 / parameters.b1Kv43;
    QKv14(6,6) = -(QKv14(6,5) + QKv14(6,7) + QKv14(6,1)); QKv43(6,6) = -(QKv43(6,5) + QKv43(6,7) + QKv43(6,1));
    
    QKv14(7,6) = 2*betaa14 * parameters.f1Kv14/parameters.f2Kv14; QKv43(7,6) = 2*betaa43 * parameters.f1Kv43/parameters.f2Kv43;
    QKv14(7,8) = 2*alphaa14 * parameters.b3Kv14/parameters.b2Kv14; QKv43(7,8) = 2*alphaa43 * parameters.b3Kv43/parameters.b2Kv43;
    QKv14(7,2) = alphai14 / parameters.b2Kv14; QKv43(7,2) = alphai43 / parameters.b2Kv43;
    QKv14(7,7) = -(QKv14(7,6) + QKv14(7,8) + QKv14(7,2)); QKv43(7,7) = -(QKv43(7,6) + QKv43(7,8) + QKv43(7,2));
    
    QKv14(8,7) = 3*betaa14 * parameters.f2Kv14/parameters.f3Kv14; QKv43(8,7) = 3*betaa43 * parameters.f2Kv43/parameters.f3Kv43;
    QKv14(8,9) = alphaa14 * parameters.b4Kv14/parameters.b3Kv14; QKv43(8,9) = alphaa43 * parameters.b4Kv43/parameters.b3Kv43;
    QKv14(8,3) = alphai14/parameters.b3Kv14; QKv43(8,3) = alphai43/parameters.b3Kv43;
    QKv14(8,8) = -(QKv14(8,7) + QKv14(8,9) + QKv14(8,3)); QKv43(8,8) = -(QKv43(8,7) + QKv43(8,9) + QKv43(8,3));
    
    QKv14(9,8) = 4*betaa14 * parameters.f3Kv14/parameters.f4Kv14; QKv43(9,8) = 4*betaa43 * parameters.f3Kv43/parameters.f4Kv43;
    QKv14(9,4) = alphai14/parameters.b4Kv14; QKv43(9,4) = alphai43/parameters.b4Kv43;
    QKv14(9,9) = -(QKv14(9,8) + QKv14(9,4)); QKv43(9,9) = -(QKv43(9,8) + QKv43(9,4));
}


template <typename FloatType>
void GW_model<FloatType>::update_Kr_derivatives(const FloatType dt){
    dGlobals.Kr[0] = dt*(QKr(1,0)*globals.Kr[1] + QKr(0,0)*globals.Kr[0]);
    dGlobals.Kr[1] = dt*(QKr(0,1)*globals.Kr[0] + QKr(2,1)*globals.Kr[2] + QKr(1,1)*globals.Kr[1]);
    dGlobals.Kr[2] = dt*(QKr(1,2)*globals.Kr[1] + QKr(3,2)*globals.Kr[3] + QKr(4,2)*globals.Kr[4] + QKr(2,2)*globals.Kr[2]);
    dGlobals.Kr[3] = dt*(QKr(2,3)*globals.Kr[2] + QKr(4,3)*globals.Kr[4] + QKr(3,3)*globals.Kr[3]);
    dGlobals.Kr[4] = dt*(QKr(2,4)*globals.Kr[2] + QKr(3,4)*globals.Kr[3] + QKr(4,4)*globals.Kr[4]);
}

template <typename FloatType>
void GW_model<FloatType>::update_Kv_derivatives(const FloatType dt){
    dGlobals.Kv14[0] = dt*(QKv14(1,0)*globals.Kv14[1] + QKv14(5,0)*globals.Kv14[5] + QKv14(0,0)*globals.Kv14[0]);
    dGlobals.Kv14[1] = dt*(QKv14(0,1)*globals.Kv14[0] + QKv14(2,1)*globals.Kv14[2] + QKv14(6,1)*globals.Kv14[6] + QKv14(1,1)*globals.Kv14[1]);
    dGlobals.Kv14[2] = dt*(QKv14(1,2)*globals.Kv14[1] + QKv14(3,2)*globals.Kv14[3] + QKv14(7,2)*globals.Kv14[7] + QKv14(2,2)*globals.Kv14[2]);
    dGlobals.Kv14[3] = dt*(QKv14(2,3)*globals.Kv14[2] + QKv14(4,3)*globals.Kv14[4] + QKv14(8,3)*globals.Kv14[8] + QKv14(3,3)*globals.Kv14[3]);
    dGlobals.Kv14[4] = dt*(QKv14(3,4)*globals.Kv14[3] + QKv14(9,4)*globals.Kv14[9] + QKv14(4,4)*globals.Kv14[4]);
    
    dGlobals.Kv14[5] = dt*(QKv14(6,5)*globals.Kv14[6] + QKv14(0,5)*globals.Kv14[0] + QKv14(5,5)*globals.Kv14[5]);
    dGlobals.Kv14[6] = dt*(QKv14(5,6)*globals.Kv14[5] + QKv14(7,6)*globals.Kv14[7] + QKv14(1,6)*globals.Kv14[1] + QKv14(6,6)*globals.Kv14[6]);
    dGlobals.Kv14[7] = dt*(QKv14(6,7)*globals.Kv14[6] + QKv14(8,7)*globals.Kv14[8] + QKv14(2,7)*globals.Kv14[2] + QKv14(7,7)*globals.Kv14[7]);
    dGlobals.Kv14[8] = dt*(QKv14(7,8)*globals.Kv14[7] + QKv14(9,8)*globals.Kv14[9] + QKv14(3,8)*globals.Kv14[3] + QKv14(8,8)*globals.Kv14[8]);
    dGlobals.Kv14[9] = dt*(QKv14(8,9)*globals.Kv14[8] + QKv14(4,9)*globals.Kv14[4] + QKv14(9,9)*globals.Kv14[9]);
    
    dGlobals.Kv43[0] = dt*(QKv43(1,0)*globals.Kv43[1] + QKv43(5,0)*globals.Kv43[5] + QKv43(0,0)*globals.Kv43[0]);
    dGlobals.Kv43[1] = dt*(QKv43(0,1)*globals.Kv43[0] + QKv43(2,1)*globals.Kv43[2] + QKv43(6,1)*globals.Kv43[6] + QKv43(1,1)*globals.Kv43[1]);
    dGlobals.Kv43[2] = dt*(QKv43(1,2)*globals.Kv43[1] + QKv43(3,2)*globals.Kv43[3] + QKv43(7,2)*globals.Kv43[7] + QKv43(2,2)*globals.Kv43[2]);
    dGlobals.Kv43[3] = dt*(QKv43(2,3)*globals.Kv43[2] + QKv43(4,3)*globals.Kv43[4] + QKv43(8,3)*globals.Kv43[8] + QKv43(3,3)*globals.Kv43[3]);
    dGlobals.Kv43[4] = dt*(QKv43(3,4)*globals.Kv43[3] + QKv43(9,4)*globals.Kv43[9] + QKv43(4,4)*globals.Kv43[4]);
    
    dGlobals.Kv43[5] = dt*(QKv43(6,5)*globals.Kv43[6] + QKv43(0,5)*globals.Kv43[0] + QKv43(5,5)*globals.Kv43[5]);
    dGlobals.Kv43[6] = dt*(QKv43(5,6)*globals.Kv43[5] + QKv43(7,6)*globals.Kv43[7] + QKv43(1,6)*globals.Kv43[1] + QKv43(6,6)*globals.Kv43[6]);
    dGlobals.Kv43[7] = dt*(QKv43(6,7)*globals.Kv43[6] + QKv43(8,7)*globals.Kv43[8] + QKv43(2,7)*globals.Kv43[2] + QKv43(7,7)*globals.Kv43[7]);
    dGlobals.Kv43[8] = dt*(QKv43(7,8)*globals.Kv43[7] + QKv43(9,8)*globals.Kv43[9] + QKv43(3,8)*globals.Kv43[3] + QKv43(8,8)*globals.Kv43[8]);
    dGlobals.Kv43[9] = dt*(QKv43(8,9)*globals.Kv43[8] + QKv43(4,9)*globals.Kv43[4] + QKv43(9,9)*globals.Kv43[9]);
}

template <typename FloatType>
void GW_model<FloatType>::update_gate_derivatives(const FloatType dt){
    dGlobals.m = dt * (common::alpham(globals.V) * (1.0 - globals.m) - common::betam(globals.V) * globals.m);
    dGlobals.h = dt * (common::alphah(globals.V) * (1.0 - globals.h) - common::betah(globals.V) * globals.h);
    dGlobals.j = dt * (common::alphaj(globals.V) * (1.0 - globals.j) - common::betaj(globals.V) * globals.j);
    dGlobals.xKs =  dt * (GW::XKsinf(globals.V) - globals.xKs) * GW::tauXKs_inv(globls.V);
}

template <typename FloatType>
void GW_model<FloatType>::update_V_and_concentration_derivatives(const FloatType dt)
{
    FloatType beta_cyto;

    //double Nai = concentrations.Nai, Ki = concentrations.Ki, Cai = concentrations.Cai, CaNSR = concentrations.CaNSR;
    //double CaLTRPN = concentrations.CaLTRPN, CaHTRPN = concentrations.CaHTRPN;
    FloatType IKv14, IKv43;

    const FloatType ENa = common::Nernst(globals.Nai, parameters.Nao, consts.RT_F, 1.0);
    const FloatType EK = common::Nernst(globals.Ki, parameters.Ko, consts.RT_F, 1.0);
    const FloatType ECa = common::Nernst(globals.Ki, parameters.Ko, consts.RT_F, 2.0);

    INa = common::INa(globals.V, globals.m, globals.h, globals.j, ENa, parameters.GNa);
    INab = common::Ib(globals.V, ENa, parameters.GNab);
    INaCa = common::INaCa(VFRT, expmVFRT, globals.Nai, globals.Cai, consts.Nao3, parameters.Cao, parameters.eta, consts.INaCa_const, parameters.ksat);
    INaK = common::INaK(VFRT, expmVFRT, globals.Nai, consts.sigma, parameters.KmNai, consts.INaK_const);

    IKr = GW::IKr(globals.V, globals.Kr[3], EK, parameters.GKr, consts.sqrtKo);
    IKs = GW::IKs(globals.V, gates.xKs, globals.Ki, globals.Nai, parameters.Nao, parameters.Ko, parameters.GKs);
    IKv14 = GW::IKv14(VFRT, expmVFRT, globals.Kv14[4], globals.Ki, globals.Nai, consts.PKv14_Csc, parameters.Nao, parameters.Ko);
    IKv43 = GW::IKv43(globals.V, globals.XKv43[4], EK, parameters.GKv43);
    Ito1 = IKv14 + IKv43;
    Ito2 = GW::Ito2(CRUs.ClCh, VFRT, expmVFRT, parameters.Clcyto, parameters.Clo, consts.Ito2_const);
    IK1 = GW::IK1(globals.V, EK, parameters.GK1, consts.IK1_const);
    IKp = GW::IKp(globals.V, EK, parameters.GKp);

    ICaL = GW::ICaL(JLCC, consts.ICaL_const);
    ICab = common::Ib(globals.V, ECa, parameters.GCab);
    IpCa = common::IpCa(globals.Cai, parameters.IpCamax, parameters.KmpCa);

    Jup = GW:Jup(globals.Cai, globals.CaNSR, parameters.Vmaxf, parameters.Vmaxr, parameters.Kmf, parameters.Kmr, parameters.Hf, parameters.Hr);
    Jtr_tot = GW::flux_average(Jtr, consts.CRU_factor);
    Jxfer_tot = GW::flux_average(Jxfer, consts.CRU_factor);
    beta_cyto = GW::beta_cyto(globals.Cai, consts.CMDN_const, parameters.KCMDN);

    dGlobals.CaLTRPN = GW::dTRPNCa(globals.CaLTRPN, globals.Cai, parameters.LTRPNtot, parameters.kLTRPNp, parameters.kLTRPNm);
    dGlobals.CaHTRPN = GW::dTRPNCa(globals.CaHTRPN, globals.Cai, parameters.HTRPNtot, parameters.kHTRPNp, parameters.kHTRPNm);

    dGlobals.Nai = -dt*consts.CSA_FVcyto * (INa + INab + 3*INaCa + 3*INaK);
    dGlobals.Ki = -dt*consts.CSA_FVcyto * (IKr + IKs + Ito1 + IK1 + IKp - 2*INaK);
    dGlobals.Cai = dt*beta_cyto * (-0.5*consts.CSA_FVcyto*(ICab + IpCa - 2*INaCa) + consts.VSS_Vcyto*Jxfer_tot - Jup - (dconc.CaLTRPN + dconc.CaHTRPN));
    dGlobals.CaNSR = dt*(consts.Vcyto_VNSR * Jup - consts.VJSR_VNSR * Jtr_tot);
    dGlobals.CaLTRPN *= dt;
    dGlobals.CaHTRPN *= dt;

    dGlobals.V = dt*(Istim - (INa + ICaL + IKr + IKs + Ito1 + IK1 + IKp + Ito2 + INaK + INaCa + IpCa + ICab + INab));
}


}