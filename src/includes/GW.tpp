#include "GW.hpp"

namespace GW {

    template <typename FloatType, typename PRNG>
    void GW_model<FloatType, PRNG>::set_initial_value(GlobalState<FloatType>& global_vals, CRUState<FloatType>& cru_vals){
        // Will this automatically create a copy constructor?
        globals = global_vals; 
        CRUs = cru_vals;
    }

    template <typename FloatType, typename PRNG>
    void GW_model<FloatType, PRNG>::initialise_JLCC(){
        for (int i = 0; i < JLCC.rows(); ++i){
        //for (int i = 0; i < JLCC.shape(0); ++i){
            for (int j = 0; j < 4; j++){
                if ((CRUs.LCC_inactivation(i,j) == 1) && (CRUs.LCC(i,j) == 6 || CRUs.LCC(i,j) == 12))
                    JLCC(i,j) = consts.JLCC_const * consts.VF_RT * (consts.Cao_scaled - consts.JLCC_exp * CRUs.CaSS(i,j));
                else
                    JLCC(i,j) = 0.0;
            }
        }
    }

    template <typename FloatType, typename PRNG>
    void GW_model<FloatType, PRNG>::update_QKr(){
        QKr(0,1) = 0.0069*exp(0.0272*globals.V);
        QKr(0,0) = -QKr(0,1);

        QKr(1,0) = 0.0227*exp(-0.0431*globals.V);
        QKr(1,1) = -(QKr(1,0) + QKr(1,2));

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


    template <typename FloatType, typename PRNG>
    void GW_model<FloatType, PRNG>::update_CRUstate_from_temp(const CRUStateThread<FloatType> &temp, const int idx){
        for (int j = 0; j < 4; j++){
            CRUs.LCC(idx,j) = temp.LCC[j];
            CRUs.LCC_inactivation(idx,j) = temp.LCC_inactivation[j];
            for (int k = 0; k < 6; k++)
                CRUs.RyR.array(idx,j,k) = temp.RyR[6*j+k];
            CRUs.ClCh(idx,j) = temp.ClCh[j];
            CRUs.CaSS(idx,j) = temp.CaSS[j];

            JLCC(idx,j) = temp.JLCC[j];
            Jxfer(idx,j) = temp.Jxfer[j];
        }
        CRUs.CaJSR(idx) = temp.CaJSR;
        Jtr(idx) = temp.Jtr;
    }

    template <typename FloatType, typename PRNG>
    void GW_model<FloatType, PRNG>::update_QKv(){
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


    template <typename FloatType, typename PRNG>
    void GW_model<FloatType, PRNG>::update_Kr_derivatives(const FloatType dt){
        dGlobals.Kr[0] = dt*(QKr(1,0)*globals.Kr[1] + QKr(0,0)*globals.Kr[0]);
        dGlobals.Kr[1] = dt*(QKr(0,1)*globals.Kr[0] + QKr(2,1)*globals.Kr[2] + QKr(1,1)*globals.Kr[1]);
        dGlobals.Kr[2] = dt*(QKr(1,2)*globals.Kr[1] + QKr(3,2)*globals.Kr[3] + QKr(4,2)*globals.Kr[4] + QKr(2,2)*globals.Kr[2]);
        dGlobals.Kr[3] = dt*(QKr(2,3)*globals.Kr[2] + QKr(4,3)*globals.Kr[4] + QKr(3,3)*globals.Kr[3]);
        dGlobals.Kr[4] = dt*(QKr(2,4)*globals.Kr[2] + QKr(3,4)*globals.Kr[3] + QKr(4,4)*globals.Kr[4]);
    }

    template <typename FloatType, typename PRNG>
    void GW_model<FloatType, PRNG>::update_Kv_derivatives(const FloatType dt){
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

    template <typename FloatType, typename PRNG>
    void GW_model<FloatType, PRNG>::update_gate_derivatives(const FloatType dt){
        dGlobals.m = dt * (common::alpha_m(globals.V) * (1.0 - globals.m) - common::beta_m(globals.V) * globals.m);
        dGlobals.h = dt * (common::alpha_h(globals.V) * (1.0 - globals.h) - common::beta_h(globals.V) * globals.h);
        dGlobals.j = dt * (common::alpha_j(globals.V) * (1.0 - globals.j) - common::beta_j(globals.V) * globals.j);
        dGlobals.xKs =  dt * (XKsinf(globals.V) - globals.xKs) * tauXKs_inv(globals.V);
    }

    template <typename FloatType, typename PRNG>
    void GW_model<FloatType, PRNG>::update_V_and_concentration_derivatives(const FloatType dt){
        FloatType IKv14, IKv43;

        consts.ENa = common::Nernst<FloatType>(globals.Nai, parameters.Nao, consts.RT_F, 1.0);
        consts.EK = common::Nernst<FloatType>(globals.Ki, parameters.Ko, consts.RT_F, 1.0);
        consts.ECa = common::Nernst<FloatType>(globals.Ki, parameters.Ko, consts.RT_F, 2.0);

        currents.INa = common::INa<FloatType>(globals.V, globals.m, globals.h, globals.j, parameters.GNa, consts.ENa);
        currents.INab = common::Ib<FloatType>(globals.V, parameters.GNab, consts.ENa);
        currents.INaCa = common::INaCa<FloatType>(consts.VF_RT, consts.expmVF_RT, globals.Nai, globals.Cai, consts.Nao3, parameters.Cao, parameters.eta, parameters.ksat, consts.INaCa_const);
        currents.INaK = common::INaK<FloatType>(consts.VF_RT, consts.expmVF_RT, globals.Nai, consts.sigma, parameters.KmNai, consts.INaK_const);
        
        currents.IKr = GW::IKr<FloatType>(globals.V, globals.Kr[3], consts.EK, parameters.GKr, consts.sqrtKo);
        currents.IKs = GW::IKs<FloatType>(globals.V, globals.xKs, globals.Ki, globals.Nai, parameters.Nao, parameters.Ko, parameters.GKs, consts.RT_F);
        IKv14 = GW::IKv14<FloatType>(consts.VF_RT, consts.expmVF_RT, globals.Kv14[4], globals.Ki, globals.Nai, consts.PKv14_Csc, parameters.Nao, parameters.Ko);
        IKv43 = GW::IKv43<FloatType>(globals.V, globals.Kv43[4], consts.EK, parameters.GKv43);
        currents.Ito1 = IKv14 + IKv43;
        currents.Ito2 = GW::Ito2<FloatType>(CRUs.ClCh, consts.VF_RT, consts.expmVF_RT, parameters.Clcyto, parameters.Clo, consts.Ito2_const);
        currents.IK1 = GW::IK1<FloatType>(globals.V, consts.EK, parameters.GK1, consts.IK1_const, consts.F_RT);
        currents.IKp = GW::IKp<FloatType>(globals.V, consts.EK, parameters.GKp);
        
        currents.ICaL = GW::ICaL<FloatType>(JLCC, consts.ICaL_const);
        currents.ICab = common::Ib<FloatType>(globals.V, parameters.GCab, consts.ECa);
        currents.IpCa = common::IpCa<FloatType>(globals.Cai, parameters.IpCamax, parameters.KmpCa);

        currents.Jup = GW::Jup<FloatType>(globals.Cai, globals.CaNSR, parameters.Vmaxf, parameters.Vmaxr, parameters.Kmf, parameters.Kmr, parameters.Hf, parameters.Hr);
        currents.Jtr_tot = GW::flux_average<FloatType>(Jtr, consts.CRU_factor);
        currents.Jxfer_tot = GW::flux_average<FloatType>(Jxfer, consts.CRU_factor);
        consts.beta_cyto = GW::beta_cyto<FloatType>(globals.Cai, consts.CMDN_const, parameters.KCMDN);

        dGlobals.CaLTRPN = GW::dTRPNCa<FloatType>(globals.CaLTRPN, globals.Cai, parameters.LTRPNtot, parameters.kLTRPNp, parameters.kLTRPNm);
        dGlobals.CaHTRPN = GW::dTRPNCa<FloatType>(globals.CaHTRPN, globals.Cai, parameters.HTRPNtot, parameters.kHTRPNp, parameters.kHTRPNm);

        dGlobals.Nai = -dt*consts.CSA_FVcyto * (currents.INa + currents.INab + 3*currents.INaCa + 3*currents.INaK);
        dGlobals.Ki = -dt*consts.CSA_FVcyto * (currents.IKr + currents.IKs + currents.Ito1 + currents.IK1 + currents.IKp - 2*currents.INaK);
        dGlobals.Cai = dt*consts.beta_cyto * (-0.5*consts.CSA_FVcyto*(currents.ICab + currents.IpCa - 2*currents.INaCa) + consts.VSS_Vcyto*currents.Jxfer_tot - currents.Jup - (dGlobals.CaLTRPN + dGlobals.CaHTRPN));
        dGlobals.CaNSR = dt*(consts.Vcyto_VNSR * currents.Jup - consts.VJSR_VNSR * currents.Jtr_tot);
        dGlobals.CaLTRPN *= dt;
        dGlobals.CaHTRPN *= dt;

        dGlobals.V = dt*(Istim - (currents.INa + currents.ICaL + currents.IKr + currents.IKs + currents.Ito1 + currents.IK1 + currents.IKp + currents.Ito2 + currents.INaK + currents.INaCa + currents.IpCa + currents.ICab + currents.INab));
    }

    template <typename FloatType, typename PRNG>
    void GW_model<FloatType, PRNG>::SSA(const FloatType dt){
        consts.alphaLCC = alphaLCC<FloatType>(globals.V);
        consts.betaLCC = betaLCC<FloatType>(globals.V);
        consts.yinfLCC = yinfLCC<FloatType>(globals.V);
        consts.tauLCC = tauLCC<FloatType>(globals.V);
        consts.JLCC_exp = square(1.0 / consts.expmVF_RT);
        consts.JLCC_multiplier = consts.JLCC_const * globals.V * consts.F_RT / (consts.JLCC_exp - 1);

        #pragma omp parallel
        {
            CRUStateThread<FloatType> temp;
            
            #pragma omp for schedule( static )
            for (int i = 0; i < nCRU; i++){
                temp.copy_from_CRUState(CRUs, JLCC, i, parameters);
                SSA_single_CRU<FloatType, PRNG>(temp, globals.Cai, globals.CaNSR, dt, parameters, consts);
                update_CRUstate_from_temp(temp, i);
            }
        }
    }

    template <typename FloatType, typename PRNG>
    void GW_model<FloatType, PRNG>::euler_step(const FloatType dt){
        consts.VF_RT = globals.V*consts.F_RT;
        consts.expmVF_RT = exp(-consts.VF_RT);

        update_QKr();
        update_QKv(); // Updates Kv14 and Kv43
        
        update_V_and_concentration_derivatives(dt);
        update_gate_derivatives(dt);
        update_Kr_derivatives(dt);
        update_Kv_derivatives(dt); // Updates both Kv14 and Kv43

        update_integral(dt); // Must do this before update to ensure we are taking the lefthand process

        SSA(dt);

        globals.V += dGlobals.V;
        globals.Nai += dGlobals.Nai;
        globals.Ki += dGlobals.Ki;
        globals.Cai += dGlobals.Cai;
        globals.CaNSR += dGlobals.CaNSR;
        globals.CaLTRPN += dGlobals.CaLTRPN;
        globals.CaHTRPN += dGlobals.CaHTRPN;
        globals.m += dGlobals.m;
        globals.h += dGlobals.h;
        globals.j += dGlobals.j;
        globals.xKs += dGlobals.xKs;
        globals.Kr[0] += dGlobals.Kr[0];
        globals.Kr[1] += dGlobals.Kr[1];
        globals.Kr[2] += dGlobals.Kr[2];
        globals.Kr[3] += dGlobals.Kr[3];
        globals.Kr[4] += dGlobals.Kr[4];
        for (int j = 0; j < 10; j++){
            globals.Kv14[j] += dGlobals.Kv14[j];
            globals.Kv43[j] += dGlobals.Kv43[j];
        }
    }

    template <typename FloatType, typename PRNG>
    void GW_model<FloatType, PRNG>::euler(const FloatType dt, const int nstep, const std::function<FloatType(FloatType)>& Ist){
        FloatType t = 0.0;
        for (int i = 0; i < nstep; ++i){
            Istim = Ist(t);
            euler_step(dt);
            t += dt;
        }
    }
    
    template <typename FloatType, typename PRNG>
    void GW_model<FloatType, PRNG>::update_integral(const FloatType dt){
        FloatType increment = 0.0;
        FloatType r1p, r2p, r1m, r2m, CaSS2;
        for (int i = 0; i < nCRU; ++i){
            for (int j = 0; j < 4; ++j){
                CaSS2 = square(CRUs.CaSS(i,j));

                r1p = CRUs.RyR.array(i,j,1) * parameters.k23 * CaSS2;
                if (CRUs.CaSS(i,j) > 0.000115) {
                    r2p = ((CRUs.RyR.array(i,j,4) + CRUs.RyR.array(i,j,5)) 
                            * parameters.k54*parameters.k65*CaSS2 
                            / (parameters.k56*CaSS2 + parameters.k65));
                }
                else {
                    r2p = CRUs.RyR.array(i,j,4) * parameters.k54 * CaSS2;
                } 
                if (CRUs.CaSS(i,j) > 0.03685) {
                    r1m = (CRUs.RyR.array(i,j,2) + CRUs.RyR.array(i,j,3)) * parameters.k32 * parameters.k43 
                         / (parameters.k34 * CaSS2 + parameters.k43);
                    r2m = (CRUs.RyR.array(i,j,2) + CRUs.RyR.array(i,j,3)) * (parameters.k45*parameters.k34*CaSS2)
                         / (parameters.k34 * CaSS2 + parameters.k43);
                }
                else {
                    r1m = CRUs.RyR.array(i,j,2) * parameters.k32;
                    r2m = CRUs.RyR.array(i,j,3) * parameters.k45;
                }

                increment += (r1p + r2p - (r1m + r2m));
            }
        }
        int_QTXt += dt * increment / (4.0*nCRU); // Update approximation of int_{0}^t RyR_open(Q^T(s) X_s) ds
    }

/*
    template <typename FloatType>
    template <typename LambdaType>
    void GW_model<FloatType>::euler_write(const FloatType dt, const int nstep, const LambdaType&& Ist, std::ofstream &file, const int record_every){
        FloatType t = 0.0;
        write_header(file);
        for (int i = 0; i < nstep; ++i){
            Istim = Ist(t);
            euler_step(dt);
            t += dt;
            if (i % record_every == 0)
                write_state(file, t);
        }
    }

    template <typename FloatType>
    void GW_model<FloatType>::write_header(std::ofstream &file){
        file << "t,V,m,h,j,Nai,Ki,Cai,CaNSR,CaLTRPN,CaHTRPN,xKs,Kr1,Kr2,Kr3,Kr4,Kr5,Kv14_1,Kv14_2,Kv14_3,Kv14_4,Kv14_5,Kv14_6,Kv14_7,Kv14_8,Kv14_9,Kv14_10,Kv43_1,Kv43_2,Kv43_3,Kv43_4,Kv43_5,Kv43_6,Kv43_7,Kv43_8,Kv43_9,Kv43_10,";
        file << "CaJSRs,CaSSs,LCC1,LCC2,LCC3,LCC4,LCC5,LCC6,LCC7,LCC8,LCC9,LCC10,LCC11,LCC12,LCCas,RyRs,ClChs,";
        file << "INa,ICaL,IKr,IKs,Ito1,IK1,IKp,Ito2,INaK,INaCa,IpCa,ICab,INab,JLCC" << std::endl;
    }

    template <typename FloatType>
    void GW_model<FloatType>::write_state(std::ofstream &file, const FloatType t){
        int nlcc1 = 0, nlcc2 = 0, nlcc3 = 0, nlcc4 = 0, nlcc5 = 0, nlcc6 = 0, nlcc7 = 0, nlcc8 = 0, nlcc9 = 0, nlcc10 = 0, nlcc11 = 0, nlcc12 = 0, nryr = 0;
        for (int i = 0; i < nCRU; i++){
            for (int j = 0; j < 4; j++){
                switch (CRUs.LCC(i,j)){
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
                
                nryr += (CRUs.RyR.array(i,j,2) + CRUs.RyR.array(i,j,3));
            }
        }
        file << t << ',' << globals.V << ',' << globals.m << ',' << globals.h << ',' << globals.j << ',' << globals.Nai << ',' << globals.Ki << ','
            << globals.Cai << ',' << globals.CaNSR << ',' << globals.CaLTRPN << ',' << globals.CaHTRPN << ',' << globals.xKs << ',' << globals.Kr[0] << ',' 
            << globals.Kr[1] << ',' << globals.Kr[2] << ',' << globals.Kr[3] << ',' << globals.Kr[4] << ',' << globals.Kv14[0] << ',' << globals.Kv14[1] << ',' 
            << globals.Kv14[2] << ',' << globals.Kv14[3] << ',' << globals.Kv14[4] << ',' << globals.Kv14[5] << ',' << globals.Kv14[6] << ',' << globals.Kv14[7] << ',' 
            << globals.Kv14[8] << ',' << globals.Kv14[9] << ',' << globals.Kv43[0] << ',' << globals.Kv43[1] << ',' << globals.Kv43[2] << ',' << globals.Kv43[3] << ',' 
            << globals.Kv43[4] << ',' << globals.Kv43[5] << ',' << globals.Kv43[6] << ',' << globals.Kv43[7] << ',' << globals.Kv43[8] << ',' << globals.Kv43[9];
            
        file << ',' << CRUs.CaJSR.sum() << ',' << CRUs.CaSS.sum() << ',' << nlcc1 << ',' << nlcc2 << ',' << nlcc3 << ',' << nlcc4 << ',' 
            << nlcc5 << ',' << nlcc6 << ',' << nlcc7 << ',' << nlcc8 << ',' << nlcc9 << ',' << nlcc10 << ',' << nlcc11 << ',' << nlcc12 << ',' 
            << CRUs.LCC_inactivation.sum() << ',' << nryr << ',' << CRUs.ClCh.sum();

        file << ',' << currents.INa << ',' << currents.ICaL << ',' << currents.IKr << ',' << currents.IKs << ',' << currents.Ito1 << ',' << currents.IK1 << ',' << currents.IKp << ',' << currents.Ito2 << ',' << currents.INaK
            <<  ',' << currents.INaCa << ',' << currents.IpCa << ',' << currents.ICab << ',' << currents.INab << std::endl; 
    }
*/

}