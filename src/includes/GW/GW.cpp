#pragma once
#include "GW.hpp"

namespace GW {

    //template <typename T>
    void GW_model::set_initial_value(GlobalState& global_vals, CRUState& cru_vals){
        // Will this automatically create a copy constructor?
        globals = global_vals; 
        CRUs = cru_vals;
    }

    //template <typename T>
    void GW_model::initialise_JLCC(){
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

    //template <typename T>
    void GW_model::update_QKr(){
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


    //template <typename T>
    void GW_model::update_CRUstate_from_temp(const CRUStateThread& temp, const int idx){
        for (int j = 0; j < 4; j++){
            CRUs.LCC(idx,j) = temp.LCC[j];
            CRUs.LCC_inactivation(idx,j) = temp.LCC_inactivation[j];
            for (int k = 0; k < 6; k++)
                CRUs.RyR.array(idx,j,k) = temp.RyR[6*j+k];
            CRUs.ClCh(idx,j) = temp.ClCh[j];
            CRUs.CaSS(idx,j) = temp.CaSS[j];

            JLCC(idx,j) = temp.JLCC[j];
            Jxfer(idx,j) = temp.Jxfer[j];

            CRUs.RyR_open_martingale(idx,j) = temp.RyR_open_martingale[j];
            CRUs.RyR_open_martingale_normalised(idx,j) = temp.RyR_open_martingale_normalised[j];
            
            CRUs.LCC_open_martingale(idx,j) = temp.LCC_open_martingale[j];
            CRUs.LCC_open_martingale_normalised(idx,j) = temp.LCC_open_martingale_normalised[j];
        }
        CRUs.CaJSR(idx) = temp.CaJSR;
        Jtr(idx) = temp.Jtr;
        CRUs.sigma_RyR(idx) = temp.sigma_RyR;
        CRUs.sigma_LCC(idx) = temp.sigma_LCC;

        CRUs.RyR_open_int(idx,0) = temp.RyR_open_int[0];
        CRUs.RyR_open_int(idx,1) = temp.RyR_open_int[1];
        CRUs.RyR_open_int(idx,2) = temp.RyR_open_int[2];
        CRUs.RyR_open_int(idx,3) = temp.RyR_open_int[3];
            
        CRUs.LCC_open_int(idx,0) = temp.LCC_open_int[0];
        CRUs.LCC_open_int(idx,1) = temp.LCC_open_int[1];
        CRUs.LCC_open_int(idx,2) = temp.LCC_open_int[2];
        CRUs.LCC_open_int(idx,3) = temp.LCC_open_int[3];
    }

    //template <typename T>
    void GW_model::update_QKv(){
        const double alphaa14 = parameters.alphaa0Kv14 * exp(parameters.aaKv14 * globals.V);
        const double alphaa43 = parameters.alphaa0Kv43 * exp(parameters.aaKv43 * globals.V);

        const double alphai14 = parameters.alphai0Kv14 * exp(-parameters.aiKv14 * globals.V);
        const double alphai43 = parameters.alphai0Kv43 * exp(-parameters.aiKv43 * globals.V);

        const double betaa14 = parameters.betaa0Kv14 * exp(-parameters.baKv14 * globals.V);
        const double betaa43 = parameters.betaa0Kv43 * exp(-parameters.baKv43 * globals.V);

        const double betai14 = parameters.betai0Kv14 * exp(parameters.biKv14 * globals.V);
        const double betai43 = parameters.betai0Kv43 * exp(parameters.biKv43 * globals.V);

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


    //template <typename T>
    void GW_model::update_Kr_derivatives(const double dt){
        dGlobals.Kr[0] = dt*(QKr(1,0)*globals.Kr[1] + QKr(0,0)*globals.Kr[0]);
        dGlobals.Kr[1] = dt*(QKr(0,1)*globals.Kr[0] + QKr(2,1)*globals.Kr[2] + QKr(1,1)*globals.Kr[1]);
        dGlobals.Kr[2] = dt*(QKr(1,2)*globals.Kr[1] + QKr(3,2)*globals.Kr[3] + QKr(4,2)*globals.Kr[4] + QKr(2,2)*globals.Kr[2]);
        dGlobals.Kr[3] = dt*(QKr(2,3)*globals.Kr[2] + QKr(4,3)*globals.Kr[4] + QKr(3,3)*globals.Kr[3]);
        dGlobals.Kr[4] = dt*(QKr(2,4)*globals.Kr[2] + QKr(3,4)*globals.Kr[3] + QKr(4,4)*globals.Kr[4]);
    }

    //template <typename T>
    void GW_model::update_Kv_derivatives(const double dt){
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

    //template <typename T>
    void GW_model::update_gate_derivatives(const double dt){
        dGlobals.m = dt * (common::alpha_m(globals.V) * (1.0 - globals.m) - common::beta_m(globals.V) * globals.m);
        dGlobals.h = dt * (common::alpha_h(globals.V) * (1.0 - globals.h) - common::beta_h(globals.V) * globals.h);
        dGlobals.j = dt * (common::alpha_j(globals.V) * (1.0 - globals.j) - common::beta_j(globals.V) * globals.j);
        dGlobals.xKs =  dt * (XKsinf(globals.V) - globals.xKs) * tauXKs_inv(globals.V);
    }

    //template <typename T>
    void GW_model::update_V_and_concentration_derivatives(const double dt){
        double IKv14, IKv43;

        consts.ENa = common::Nernst<double>(globals.Nai, parameters.Nao, consts.RT_F, 1.0);
        consts.EK = common::Nernst<double>(globals.Ki, parameters.Ko, consts.RT_F, 1.0);
        consts.ECa = common::Nernst<double>(globals.Cai, parameters.Cao, consts.RT_F, 2.0);

        currents.INa = common::INa<double>(globals.V, globals.m, globals.h, globals.j, parameters.GNa, consts.ENa);
        currents.INab = common::Ib<double>(globals.V, parameters.GNab, consts.ENa);
        currents.INaCa = common::INaCa<double>(consts.VF_RT, consts.expmVF_RT, globals.Nai, globals.Cai, consts.Nao3, parameters.Cao, parameters.eta, parameters.ksat, consts.INaCa_const);
        currents.INaK = common::INaK<double>(consts.VF_RT, consts.expmVF_RT, globals.Nai, consts.sigma, parameters.KmNai, consts.INaK_const);
        
        currents.IKr = GW::IKr(globals.V, globals.Kr[3], consts.EK, parameters.GKr, consts.sqrtKo);
        currents.IKs = GW::IKs(globals.V, globals.xKs, globals.Ki, globals.Nai, parameters.Nao, parameters.Ko, parameters.GKs, consts.RT_F);
        IKv14 = GW::IKv14(consts.VF_RT, consts.expmVF_RT, globals.Kv14[4], globals.Ki, globals.Nai, consts.PKv14_Csc, parameters.Nao, parameters.Ko);
        IKv43 = GW::IKv43(globals.V, globals.Kv43[4], consts.EK, parameters.GKv43);
        currents.Ito1 = IKv14 + IKv43;
        currents.Ito2 = GW::Ito2(CRUs.ClCh, consts.VF_RT, consts.expmVF_RT, parameters.Clcyto, parameters.Clo, consts.Ito2_const);
        currents.IK1 = GW::IK1(globals.V, consts.EK, parameters.GK1, consts.IK1_const, consts.F_RT);
        currents.IKp = GW::IKp(globals.V, consts.EK, parameters.GKp);
        
        currents.ICaL = GW::ICaL(JLCC, consts.ICaL_const);
        currents.ICab = common::Ib<double>(globals.V, parameters.GCab, consts.ECa);
        currents.IpCa = common::IpCa<double>(globals.Cai, parameters.IpCamax, parameters.KmpCa);

        currents.Jup = GW::Jup(globals.Cai, globals.CaNSR, parameters.Vmaxf, parameters.Vmaxr, parameters.Kmf, parameters.Kmr, parameters.Hf, parameters.Hr);
        currents.Jtr_tot = Jtr.sum() * consts.CRU_factor;
        currents.Jxfer_tot = Jxfer.sum() * consts.CRU_factor;
        consts.beta_cyto = GW::beta_cyto(globals.Cai, consts.CMDN_const, parameters.KCMDN);

        dGlobals.CaLTRPN = GW::dTRPNCa(globals.CaLTRPN, globals.Cai, parameters.LTRPNtot, parameters.kLTRPNp, parameters.kLTRPNm);
        dGlobals.CaHTRPN = GW::dTRPNCa(globals.CaHTRPN, globals.Cai, parameters.HTRPNtot, parameters.kHTRPNp, parameters.kHTRPNm);

        dGlobals.Nai = -dt*consts.CSA_FVcyto * (currents.INa + currents.INab + 3*currents.INaCa + 3*currents.INaK);
        dGlobals.Ki = -dt*consts.CSA_FVcyto * (currents.IKr + currents.IKs + currents.Ito1 + currents.IK1 + currents.IKp - 2*currents.INaK);
        dGlobals.Cai = dt*consts.beta_cyto * (-0.5*consts.CSA_FVcyto*(currents.ICab + currents.IpCa - 2*currents.INaCa) + consts.VSS_Vcyto*currents.Jxfer_tot - currents.Jup - (dGlobals.CaLTRPN + dGlobals.CaHTRPN));
        dGlobals.CaNSR = dt*(consts.Vcyto_VNSR * currents.Jup - consts.VJSR_VNSR * currents.Jtr_tot);
        dGlobals.CaLTRPN *= dt;
        dGlobals.CaHTRPN *= dt;

        dGlobals.V = dt*(Istim - (currents.INa + currents.ICaL + currents.IKr + currents.IKs + currents.Ito1 + currents.IK1 + currents.IKp + currents.Ito2 + currents.INaK + currents.INaCa + currents.IpCa + currents.ICab + currents.INab));
    }

    //template <typename T>

    //template <typename T>

    //template <typename T>
    //template <typename PRNG>
    //void GW_model<double>::euler(const double dt, const int nstep, const std::function<double(double)>& Ist){
    //    double t = 0.0;
    //    for (int i = 0; i < nstep; ++i){
    //        Istim = Ist(t);
    //        euler_step(dt);
    //        t += dt;
    //    }
    //}
    
}