#include <random>

#include "includes/lattice/GW_lattice_utils.hpp"
#include "includes/common.hpp"
    
namespace GW_lattice {

    CRULatticeState::CRULatticeState(const int nCRU_x, const int nCRU_y) : CaSS(nCRU_x,nCRU_x), CaJSR(nCRU_x,nCRU_y), Cai(nCRU_x, nCRU_y), 
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
                
                LCC_i_idx = sample_weights<double, int, std::mt19937_64>(LCC_i_weights, 1.0, 2);
                LCC_inactivation(i,j) = (LCC_i_idx == 0) ? 1 : 0;
                    
                for (int k = 0; k < 5; k++){
                    RyR_idx = sample_weights<double, int, std::mt19937_64>(RyR_weights, 1.0, 3);
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

    
    Constants::Constants(const Parameters& params, const int nCRU_x, const int nCRU_y){
        RT_F = GAS_CONST * params.T / FARADAY;
        F_RT = 1.0 / RT_F;

        CRU_factor = (double)params.NCaRU / (double)(nCRU_x*nCRU_y);
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

}
