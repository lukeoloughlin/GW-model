#include "GW_utils.hpp"

namespace GW {
    
    void initialise_LCC(NDArray<int,2> &LCC){
        const double weights[3] = { 0.958, 0.038, 0.004 };
        int idx;

        for (unsigned int i = 0; i < LCC.shape(0); i++){
            for (unsigned int j = 0; j < 4; j++){
                idx = sample_weights(weights, 1.0, 3);
                if (idx == 0)
                    LCC(i,j) = 1;
                else if (idx == 1)
                    LCC(i,j) = 2;
                else 
                    LCC(i,j) = 7;
            }
        }
    }

    void initialise_LCC_a(NDArray<int,2> &LCC_a){
        const double weights[2] = { 0.9425, 0.0575 };
        int idx;
        for (unsigned int i = 0; i < LCC_a.shape(0); i++){
            for (unsigned int j = 0; j < 4; j++){
                idx = sample_weights(weights, 1.0, 2);
                LCC_a(i,j) = (idx == 0) ? 1 : 0;
            }
        }
    }


    void initialise_RyR(NDArray<int,3> &RyR){
        const double weights[3] = { 0.609, 0.5*0.391, 0.5*0.391 };
        int idx;
        for (unsigned int i = 0; i < RyR.shape(0); i++){
            for (unsigned int j = 0; j < 4; j++){
                for (unsigned int k = 0; k < 5; k++){
                    idx = sample_weights(weights, 1.0, 3);
                    if (idx == 0)
                        RyR(i,j,0)++;
                    else if (idx == 1)
                        RyR(i,j,4)++;
                    else 
                        RyR(i,j,5)++;
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
                ClCh(i,j) = idx;
            }
        }
    }


    template <typename FloatType>
    Constants<FloatType>::Constants(const Parameters<FloatType> &params, const int nCRU_simulated){
        RT_F = GAS_CONST * params.T / FARADAY;
        F_RT = 1.0 / RT_F;
        CRU_factor = (FloatType)params.NCaRU / (FloatType)nCRU_simulated;
        CSA_FVcyto = params.CSA / (1000 * params.Vcyto * FARADAY);
        VSS_Vcyto = params.VSS / params.Vcyto;
        Vcyto_VNSR = params.Vcyto / params.VNSR;
        VJSR_VNSR = params.VJSR / params.VNSR;
        // LCC rates
        a = params.a;
        gamma0 = params.gamma0;
        gamma0a = gamma0 * params.a;
        gamma0a2 = gamma0a * params.a;
        gamma0a3 = gamma0a2 * params.a;
        gamma0a4 = gamma0a3 * params.a;
        binv = 1.0 / params.b; 
        omega = params.omega;
        omega_b = omega * binv;
        omega_b2 = omega_b * binv;
        omega_b3 = omega_b2 * binv;
        omega_b4 = omega_b3 * binv;
        // CaSS constants
        BSR_const = params.KBSR * params.BSRT;
        BSL_const = params.KBSL * params.BSLT;
        // CaJSR constants
        VSS_VJSR = params.VSS / params.VJSR;
        CSQN_const = params.KCSQN * params.CSQNT;
        // JLCC constants
        JLCC_const = 2.0e6 * params.PCaL / params.VSS;
        Cao_scaled = 0.341 * params.Cao;
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
        Ito2_const = 1.0e9 * params.Pto2 * FARADAY * CRU_factor / params.CSA;
        // IK1 consts
        IK1_const = params.Ko / (params.Ko + params.KmK1);
        // ICaL consts
        ICaL_const = -1000.0 * (2.0*FARADAY * params.VSS) * CRU_factor / params.CSA;
        // CMDN_consts
        CMDN_const = params.KCMDN * params.CMDNT;
    }
    
    template <typename FloatType>
    void update_LCC_rates(FloatType* const LCC_rates, FloatType* const subunit_rates, const int* const LCC, const FloatType* const CaSS, 
                            const int idx, const Parameters<FloatType> &params, const Constants<FloatType> &consts){
        switch (LCC[idx]){
        case 1:
            LCC_rates[3*idx] = 4*consts.alphaLCC;
            LCC_rates[3*idx+1] = consts.gamma0*CaSS[idx];
            LCC_rates[3*idx+2] = 0;
            break;
        case 2:
            LCC_rates[3*idx] = consts.betaLCC;
            LCC_rates[3*idx+1] = 3*consts.alphaLCC;
            LCC_rates[3*idx+2] = consts.gamma0a*CaSS[idx];
            break;
        case 3:
            LCC_rates[3*idx] = 2*consts.betaLCC;
            LCC_rates[3*idx+1] = 2*consts.alphaLCC;
            LCC_rates[3*idx+2] = consts.gamma0a2*CaSS[idx];
            break;
        case 4:
            LCC_rates[3*idx] = 3*consts.betaLCC;
            LCC_rates[3*idx+1] = consts.alphaLCC;
            LCC_rates[3*idx+2] = consts.gamma0a3*CaSS[idx];
            break;
        case 5:
            LCC_rates[3*idx] = 4*consts.betaLCC;
            LCC_rates[3*idx+1] = params.f;
            LCC_rates[3*idx+2] = consts.gamma0a4*CaSS[idx];
            break;
        case 6:
            LCC_rates[3*idx] = params.g;
            LCC_rates[3*idx+1] = 0;
            LCC_rates[3*idx+2] = 0;
            break;
        case 7:
            LCC_rates[3*idx] = consts.omega;
            LCC_rates[3*idx+1] = 4*consts.alphaLCC*consts.a;
            LCC_rates[3*idx+2] = 0;
            break;
        case 8:
            LCC_rates[3*idx] = consts.omega_b;
            LCC_rates[3*idx+1] = consts.betaLCC*consts.binv;
            LCC_rates[3*idx+2] = 3*consts.alphaLCC*consts.a;
            break;
        case 9:
            LCC_rates[3*idx] = consts.omega_b2;
            LCC_rates[3*idx+1] = 2*consts.betaLCC*consts.binv;
            LCC_rates[3*idx+2] = 2*consts.alphaLCC*consts.a;
            break;
        case 10:
            LCC_rates[3*idx] = consts.omega_b3;
            LCC_rates[3*idx+1] = 3*consts.betaLCC*consts.binv;
            LCC_rates[3*idx+2] = consts.alphaLCC*consts.a;
            break;
        case 11:
            LCC_rates[3*idx] = consts.omega_b4;
            LCC_rates[3*idx+1] = 4*consts.betaLCC*consts.binv;
            LCC_rates[3*idx+2] = params.f1;
            break;
        case 12:
            LCC_rates[3*idx] = params.g1;
            LCC_rates[3*idx+1] = 0;
            LCC_rates[3*idx+2] = 0;
            break;
        default:
            break;
        }    
        subunit_rates[idx] += (LCC_rates[3*idx] + LCC_rates[3*idx+1] + LCC_rates[3*idx+2]);
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
        for (int i = 0; i < 5; ++i)
            Kr[i] = val;
        for (int i = 0; i < 10; ++i){
            Kv14[i] = val;
            Kv43[i] = val;            
        }
    }


    template <typename FloatType>
    CRUState<FloatType>::CRUState(const int nCRU) : CaSS(NDArray<FloatType,2>(nCRU,4)), CaJSR(NDArray<FloatType,1>(nCRU)), 
                                                    LCC(NDArray<int,2>(nCRU,4)), LCC_activation(NDArray<int,2>(nCRU,4)),
                                                    RyR(NDArray<int,3>(nCRU,4,6)), ClCh(NDArray<int,2>(nCRU,4))
    {
        CaSS.set_to(1.45370e-4);
        CaJSR.set_to(0.908408);
        initialise_LCC(LCC);
        initialise_LCC_a(LCC_activation);
        initialise_RyR(RyR);
        initialise_ClCh(ClCh);
    }

    // There are 12 separate rates we need to keep track of since there are multiple RyRs per subunit. Using (i,j) to denote the transition from state i to state j,
    // then the rates are stored in bookeeping order in terms of (i,j) for convenience
    template <typename FloatType>
    void update_RyR_rates(FloatType* const RyR_rates, FloatType* const subunit_rates, const int* const RyR, const FloatType* const CaSS, const int idx, const Parameters<FloatType> &params){
        const FloatType CaSS2 = CaSS[idx]*CaSS[idx];
        const FloatType eq56 = params.k65 / (params.k56*CaSS2 + params.k65);
        const FloatType tau34 = 1.0 / (params.k34*CaSS2 + params.k43);

        // state 1 -> state 2
        RyR_rates[12*idx] = RyR[6*idx]*params.k12*CaSS2;
        // state 2 -> state 3
        RyR_rates[12*idx+1] = RyR[6*idx+1]*params.k23*CaSS2;
        // state 2 -> state 5
        RyR_rates[12*idx+2] = RyR[6*idx+1]*params.k25*CaSS2;
        // state 3 -> state 4
        RyR_rates[12*idx+3] = CaSS[idx] > 3.685e-2 ? 0 : RyR[6*idx+2]*params.k34*CaSS2;
        // state 4 -> state 5
        RyR_rates[12*idx+4] = CaSS[idx] > 3.685e-2 ? (RyR[6*idx+2]+RyR[6*idx+3])*params.k45*params.k34*CaSS2*tau34 : RyR[6*idx+3]*params.k45;
        // state 5 -> state 6
        RyR_rates[12*idx+5] = CaSS[idx] > 1.15e-4 ? 0 : RyR[6*idx+4]*params.k56*CaSS2;
        // state 2 -> state 1
        RyR_rates[12*idx+6] = RyR[6*idx+1]*params.k21;
        // state 3 -> state 2
        RyR_rates[12*idx+7] = CaSS[idx] > 3.685e-2 ? (RyR[6*idx+2]+RyR[6*idx+3])*params.k32*params.k43*tau34 : RyR[6*idx+2]*params.k32;
        // state 4 -> state 3
        RyR_rates[12*idx+8] = CaSS[idx] > 3.685e-2 ? 0 : RyR[6*idx+3]*params.k43;
        // state 5 -> state 2
        RyR_rates[12*idx+9] = CaSS[idx] > 1.15e-4 ? (RyR[6*idx+4]+RyR[6*idx+5])*params.k52*eq56 : RyR[6*idx+4]*params.k52; 
        // state 5 -> state 4
        RyR_rates[12*idx+10] = CaSS[idx] > 1.15e-4 ? (RyR[6*idx+4]+RyR[6*idx+5])*params.k54*CaSS2*eq56 : RyR[6*idx+4]*params.k54*CaSS2;
        // state 6 -> state 5
        RyR_rates[12*idx+11] = CaSS[idx] > 1.15e-4 ? 0 : RyR[6*idx+5]*params.k65; 
        
        for (int k = 0; k < 12; k++)
            subunit_rates[idx] += RyR_rates[12*idx+k];
    }
}