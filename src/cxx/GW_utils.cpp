#include <random>

#include "includes/GW/GW_utils.hpp"
#include "includes/common.hpp"

namespace GW {

    //template <typename PRNG>
    void initialise_LCC(Array2<int> &LCC){
        const double weights[3] = { 0.958, 0.038, 0.004 };
        int idx;

        for (int i = 0; i < LCC.rows(); i++){
            for (int j = 0; j < 4; j++){
                idx = sample_weights<double, int, std::mt19937_64>(weights, 1.0, 3);
                if (idx == 0)
                    LCC(i,j) = 1;
                else if (idx == 1)
                    LCC(i,j) = 2;
                else 
                    LCC(i,j) = 7;
            }
        }
    }

    //template <typename PRNG>
    void initialise_LCC_i(Array2<int> &LCC_i){
        const double weights[2] = { 0.9425, 0.0575 };
        int idx;
        for (unsigned int i = 0; i < LCC_i.rows(); i++){
            for (unsigned int j = 0; j < 4; j++){
                idx = sample_weights<double, int, std::mt19937_64>(weights, 1.0, 2);
                LCC_i(i,j) = (idx == 0) ? 1 : 0;
            }
        }
    }

    //template <typename PRNG>
    void initialise_RyR(Array3Container<int> &RyR){
        const double weights[3] = { 0.609, 0.5*0.391, 0.5*0.391 };
        int idx;
        for (int i = 0; i < RyR.array.dimensions()[0]; i++){
            for (int j = 0; j < 4; j++){
                for (int k = 0; k < 5; k++){
                    idx = sample_weights<double, int, std::mt19937_64>(weights, 1.0, 3);
                    if (idx == 0)
                        ++RyR.array(i,j,0);
                    else if (idx == 1)
                        ++RyR.array(i,j,4);
                    else 
                        ++RyR.array(i,j,5);
                }    
            }
        }
    }

    //template <typename PRNG>
    void initialise_ClCh(Array2<int> &ClCh){
        const double weights[2] = { 0.998, 0.002 };
        int idx;
        for (unsigned int i = 0; i < ClCh.rows(); i++){
            for (unsigned int j = 0; j < 4; j++){
                idx = sample_weights<double, int, std::mt19937_64>(weights, 1.0, 2);
                ClCh(i,j) = idx;
            }
        }
    }


    //template <typename T>
    Constants::Constants(const Parameters& params, const int nCRU_simulated){
        RT_F = GAS_CONST * params.T / FARADAY;
        F_RT = 1.0 / RT_F;
        CRU_factor = (double)params.NCaRU / (double)nCRU_simulated;
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
    
    //template <typename T>
    void update_LCC_rates(double* const LCC_rates, double* const subunit_rates, const int* const LCC, const double* const CaSS, const int idx, const Parameters& params, const Constants& consts){
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

    //template <typename T>
    GlobalState::GlobalState(double val){
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


    //template <typename T>
    CRUState::CRUState(const int nCRU) : CaSS(nCRU,4), CaJSR(nCRU), LCC(nCRU,4), LCC_inactivation(nCRU,4), RyR(nCRU,4,6), ClCh(nCRU,4), 
                                        RyR_open_int(nCRU,4), RyR_open_martingale(nCRU,4), RyR_open_martingale_normalised(nCRU,4), sigma_RyR(nCRU),
                                        LCC_open_int(nCRU,4), LCC_open_martingale(nCRU,4), LCC_open_martingale_normalised(nCRU,4), sigma_LCC(nCRU) {
        CaSS.setConstant(1.45370e-4);
        CaJSR.setConstant(0.908408);
        initialise_LCC(LCC);
        initialise_LCC_i(LCC_inactivation);
        initialise_RyR(RyR);
        initialise_ClCh(ClCh);

        RyR_open_int.setZero();
        RyR_open_martingale.setZero();
        RyR_open_martingale_normalised.setZero();
        sigma_RyR.setZero();
        
        LCC_open_int.setZero();
        LCC_open_martingale.setZero();
        LCC_open_martingale_normalised.setZero();
        sigma_LCC.setZero();
    }

    // There are 12 separate rates we need to keep track of since there are multiple RyRs per subunit. Using (i,j) to denote the transition from state i to state j,
    // then the rates are stored in bookeeping order in terms of (i,j) for convenience
    //template <typename T>
    void update_RyR_rates(double* const RyR_rates, double* const subunit_rates, const int* const RyR, const double* const CaSS, const int idx, const Parameters& params){
        const double CaSS2 = CaSS[idx]*CaSS[idx];
        const double eq56 = params.k65 / (params.k56*CaSS2 + params.k65);
        const double tau34 = 1.0 / (params.k34*CaSS2 + params.k43);

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

    //template <typename T>
    Parameters::Parameters(const Parameters& other){
        T = other.T;
        CSA = other.CSA;
        Vcyto = other.Vcyto;
        VNSR = other.VNSR;
        VJSR = other.VJSR;
        VSS = other.VSS;
        NCaRU = other.NCaRU;
        Ko = other.Ko;
        Nao = other.Nao;
        Cao = other.Cao;
        Clo = other.Clo;
        Clcyto = other.Clcyto;
        f = other.f;
        g = other.g;
        f1 = other.f1;
        g1 = other.g1;
        a = other.a;
        b = other.b;
        gamma0 = other.gamma0;
        omega = other.omega;
        PCaL = other.PCaL;
        kfClCh = other.kfClCh;
        kbClCh = other.kbClCh;
        Pto2 = other.Pto2;

        k12 = other.k12;
        k21 = other.k21;
        k23 = other.k23;
        k32 = other.k32;
        k34 = other.k34;
        k43 = other.k43;
        k45 = other.k45;
        k54 = other.k54;
        k56 = other.k56;
        k65 = other.k65;
        k25 = other.k25;
        k52 = other.k52;
        rRyR = other.rRyR;

        rxfer = other.rxfer;
        rtr = other.rtr;
        riss = other.riss;
        BSRT = other.BSRT;
        KBSR = other.KBSR;
        BSLT = other.BSLT;
        KBSL = other.KBSL;
        CSQNT = other.CSQNT;
        KCSQN = other.KCSQN;
        CMDNT = other.CMDNT;
        KCMDN = other.KCMDN;
        GNa = other.GNa;
        GKr = other.GKr;
        Kf = other.Kf;
        Kb = other.Kb;
        GKs = other.GKs;
        GKv43 = other.GKv43;
        alphaa0Kv43 = other.alphaa0Kv43;
        aaKv43 = other.aaKv43;
        betaa0Kv43 = other.betaa0Kv43;
        baKv43 = other.baKv43;
        alphai0Kv43 = other.alphai0Kv43;
        aiKv43 = other.aiKv43;
        betai0Kv43 = other.betai0Kv43;
        biKv43 = other.biKv43;
        f1Kv43 = other.f1Kv43;
        f2Kv43 = other.f2Kv43;
        f3Kv43 = other.f3Kv43;
        f4Kv43 = other.f4Kv43;
        b1Kv43 = other.b1Kv43;
        b2Kv43 = other.b2Kv43;
        b3Kv43 = other.b3Kv43;
        b4Kv43 = other.b4Kv43;
        PKv14 = other.PKv14;
        alphaa0Kv14 = other.alphaa0Kv14;
        aaKv14 = other.aaKv14;
        betaa0Kv14 = other.betaa0Kv14;
        baKv14 = other.baKv14;
        alphai0Kv14 = other.alphai0Kv14;
        aiKv14 = other.aiKv14;
        betai0Kv14 = other.betai0Kv14;
        biKv14 = other.biKv14;
        f1Kv14 = other.f1Kv14;
        f2Kv14 = other.f2Kv14;
        f3Kv14 = other.f3Kv14;
        f4Kv14 = other.f4Kv14;
        b1Kv14 = other.b1Kv14;
        b2Kv14 = other.b2Kv14;
        b3Kv14 = other.b3Kv14;
        b4Kv14 = other.b4Kv14;
        Csc = other.Csc;
        GK1 = other.GK1;
        KmK1 = other.KmK1;
        GKp = other.GKp;
        kNaCa = other.kNaCa;
        KmNa = other.KmNa;
        KmCa = other.KmCa;
        ksat = other.ksat;
        eta = other.eta;
        INaKmax = other.INaKmax;
        KmNai = other.KmNai;
        KmKo = other.KmKo;
        IpCamax = other.IpCamax;
        KmpCa = other.KmpCa;
        GCab = other.GCab;
        GNab = other.GNab;
        kHTRPNp = other.kHTRPNp;
        kHTRPNm = other.kHTRPNm;
        kLTRPNp = other.kLTRPNp;
        kLTRPNm = other.kLTRPNm;
        HTRPNtot = other.HTRPNtot;
        LTRPNtot = other.LTRPNtot;
        Vmaxf = other.Vmaxf;
        Vmaxr = other.Vmaxr;
        Kmf = other.Kmf;
        Kmr = other.Kmr;
        Hf = other.Hf;
        Hr = other.Hr;
    }
}