#include "GW_lattice.hpp"
    
namespace GW_lattice {

    template <typename FloatType>
    CRULatticeState<FloatType>::CRULatticeState(const int nCRU_x, const int nCRU_y) : CaSS(nCRU_x,nCRU_x), CaJSR(nCRU_x,nCRU_y), Cai(nCRU_x, nCRU_y), 
                                                                                      CaNSR(nCRU_x, nCRU_y), CaLTRPN(nCRU_x,nCRU_y), CaHTRPN(nCRU_x,nCRU_y), 
                                                                                      LCC(nCRU_x,nCRU_y), LCC_inactivation(nCRU_x,nCRU_y), 
                                                                                      RyR(nCRU_x,nCRU_y,6), ClCh(nCRU_x, nCRU_y) {
        CaSS.setConstant(1.45370e-4);
        CaJSR.setConstant(0.908408);
        Cai.setConstant(1.45273e-4);
        CaNSR.setConstant(0.908882);
        CaLTRPN.setConstant(8.98282e-3);
        CaHTRPN.setConstant(0.137617);


        const double LCC_weights[3] = { 0.958, 0.038, 0.004 };
        const double LCC_i_weights[2] = { 0.9425, 0.0575 };
        const double RyR_weights[3] = { 0.609, 0.5*0.391, 0.5*0.391 };
        const double ClCh_weights[2] = { 0.998, 0.002 };
        int LCC_idx, LCC_i_idx, RyR_idx, ClCh_idx;

        for (int i = 0; i < nCRU_x; i++){
            for (int j = 0; j < nCRU_y; j++){
                LCC_idx = sample_weights<double, int, std::mt19937_64>(LCC_weights, 1.0, 3);
                if (LCC_idx == 0)
                    LCC(i,j) = 1;
                else if (LCC_idx == 1)
                    LCC(i,j) = 2;
                else 
                    LCC(i,j) = 7;
                
                LCC_i_idx = sample_weights<double, int, std::mt19937>(LCC_i_weights, 1.0, 2);
                LCC_inactivation(i,j) = (LCC_i_idx == 0) ? 1 : 0;
                    
                for (int k = 0; k < 5; k++){
                    RyR_idx = sample_weights<double, int, std::mt19937>(RyR_weights, 1.0, 3);
                    if (RyR_idx == 0)
                        ++RyR.array(i,j,0);
                    else if (RyR_idx == 1)
                        ++RyR.array(i,j,4);
                    else 
                        ++RyR.array(i,j,5);
                }    
                
                ClCh_idx = sample_weights<double, int, std::mt19937_64>(ClCh_weights, 1.0, 2);
                ClCh(i,j) = ClCh_idx;
                                
            }
        }
    }

    
    template <typename FloatType>
    Constants<FloatType>::Constants(const Parameters<FloatType> &params, const int nCRU_x, const int nCRU_y){
        RT_F = GAS_CONST * params.T / FARADAY;
        F_RT = 1.0 / RT_F;

        CRU_factor = (FloatType)params.NCaRU / (FloatType)(nCRU_x*nCRU_y);
        Vcyto_elem = params.Vcyto / (nCRU_x*nCRU_y);
        VNSR_elem = params.VNSR / (nCRU_x*nCRU_y);

        CSA_F = params.CSA / (1000 * FARADAY);
        VSS_Vcyto = CRU_factor * params.VSS / Vcyto_elem;
        Vcyto_VNSR = params.Vcyto / params.VNSR;
        VJSR_VNSR = CRU_factor * params.VJSR / VNSR_elem;
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
        f = params.f;
        f1 = params.f1;
        g = params.g;
        g1 = params.g1;
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
    Parameters<FloatType>::Parameters(const Parameters<FloatType>& other){
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
        rcyto = other.rcyto;
        rnsr = other.rnsr;
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
