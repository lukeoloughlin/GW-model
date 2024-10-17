#include "GW_lattice.hpp"


namespace GW_lattice {

    template <typename FloatType>
    void GW_lattice<FloatType>::set_initial_value(GW::GlobalState<FloatType>& global_vals, CRULatticeState<FloatType>& cru_vals){
        // Will this automatically create a copy constructor?
        globals = global_vals; 
        CRU_lattice = cru_vals;
    }

    //template <typename FloatType>
    //void GW_lattice<FloatType>::initialise_JLCC(){
    //    for (int i = 0; i < JLCC.rows(); ++i){
        //for (int i = 0; i < JLCC.shape(0); ++i){
    //        for (int j = 0; j < JLCC.cols(); j++){
    //            if ((CRU_lattice.LCC_inactivation(i,j) == 1) && (CRU_lattice.LCC(i,j) == 6 || CRU_lattice.LCC(i,j) == 12))
    //                JLCC(i,j) = consts.JLCC_const * consts.VF_RT * (consts.Cao_scaled - consts.JLCC_exp * CRU_lattice.CaSS(i,j));
    //            else
    //                JLCC(i,j) = 0.0;
    //        }
    //    }
    //}

    template <typename FloatType>
    void GW_lattice<FloatType>::update_QKr(){
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


    template <typename FloatType>
    void GW_lattice<FloatType>::update_CRUstate_from_temp(const CRULatticeStateThread<FloatType> &temp, const int x, const int y){
        LCC_tmp(x,y) = temp.LCC;
        LCC_inactivation_tmp(x,y) = temp.LCC_inactivation;
        for (int j = 0; j < 6; j++)
            RyR_tmp.array(x,y,j) = temp.RyR[j];            
        ClCh_tmp(x,y) = temp.ClCh;
    }

    template <typename FloatType>
    void GW_lattice<FloatType>::update_QKv(){
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
    void GW_lattice<FloatType>::update_diffusion_fluxes(){
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < nCRU_x; i++){
            for (int j = 0; j < nCRU_y; j++){
                // Diffusion terms
                if ((i > 0) && (i < nCRU_x-1) && (j > 0) && (j < nCRU_y-1)) {
                    // interior
                    Jiss_DS(i,j) = parameters.riss * (CRU_lattice.CaSS(i-1,j) + CRU_lattice.CaSS(i+1,j) + CRU_lattice.CaSS(i,j-1) + CRU_lattice.CaSS(i,j+1) - 4*CRU_lattice.CaSS(i,j));
                    Jiss_JSR(i,j) = parameters.rijsr * (CRU_lattice.CaJSR(i-1,j) + CRU_lattice.CaJSR(i+1,j) + CRU_lattice.CaJSR(i,j-1) + CRU_lattice.CaJSR(i,j+1) - 4*CRU_lattice.CaJSR(i,j));
                }
                else if ((i == 0) && (i < nCRU_x-1) && (j > 0) && (j < nCRU_y-1)) {
                    // left side, no corner
                    Jiss_DS(i,j) = parameters.riss * (CRU_lattice.CaSS(i+1,j) + CRU_lattice.CaSS(i,j-1) + CRU_lattice.CaSS(i,j+1) - 3*CRU_lattice.CaSS(i,j));
                    Jiss_JSR(i,j) = parameters.rijsr * (CRU_lattice.CaJSR(i+1,j) + CRU_lattice.CaJSR(i,j-1) + CRU_lattice.CaJSR(i,j+1) - 3*CRU_lattice.CaJSR(i,j));
                }
                else if ((i > 0) && (i == nCRU_x-1) && (j > 0) && (j < nCRU_y-1)) {
                    // right side, no corner
                    Jiss_DS(i,j) = parameters.riss * (CRU_lattice.CaSS(i-1,j) + CRU_lattice.CaSS(i,j-1) + CRU_lattice.CaSS(i,j+1) - 3*CRU_lattice.CaSS(i,j));
                    Jiss_JSR(i,j) = parameters.rijsr * (CRU_lattice.CaJSR(i-1,j) + CRU_lattice.CaJSR(i,j-1) + CRU_lattice.CaJSR(i,j+1) - 3*CRU_lattice.CaJSR(i,j));
                }
                else if ((i > 0) && (i < nCRU_x-1) && (j == 0) && (j < nCRU_y-1)) {
                    // top, no corner
                    Jiss_DS(i,j) = parameters.riss * (CRU_lattice.CaSS(i-1,j) + CRU_lattice.CaSS(i+1,j) + CRU_lattice.CaSS(i,j+1) - 3*CRU_lattice.CaSS(i,j));
                    Jiss_JSR(i,j) = parameters.rijsr * (CRU_lattice.CaJSR(i-1,j) + CRU_lattice.CaJSR(i+1,j) + CRU_lattice.CaJSR(i,j+1) - 3*CRU_lattice.CaJSR(i,j));
                }
                else if ((i > 0) && (i < nCRU_x-1) && (j > 0) && (j == nCRU_y-1)) {
                    // bottom, no corner
                    Jiss_DS(i,j) = parameters.riss * (CRU_lattice.CaSS(i-1,j) + CRU_lattice.CaSS(i+1,j) + CRU_lattice.CaSS(i,j-1) - 3*CRU_lattice.CaSS(i,j));
                    Jiss_JSR(i,j) = parameters.rijsr * (CRU_lattice.CaJSR(i-1,j) + CRU_lattice.CaJSR(i+1,j) + CRU_lattice.CaJSR(i,j-1) - 3*CRU_lattice.CaJSR(i,j));
                }
                else if ((i == 0) && (j == 0)) {
                    // top left corner
                    Jiss_DS(i,j) = parameters.riss * (CRU_lattice.CaSS(i+1,j) + CRU_lattice.CaSS(i,j+1) - 2*CRU_lattice.CaSS(i,j));
                    Jiss_JSR(i,j) = parameters.rijsr * (CRU_lattice.CaJSR(i+1,j) + CRU_lattice.CaJSR(i,j+1) - 2*CRU_lattice.CaJSR(i,j));
                }
                else if ((i == 0) && (j == nCRU_y-1)) {
                    // top right corner
                    Jiss_DS(i,j) = parameters.riss * (CRU_lattice.CaSS(i+1,j) + CRU_lattice.CaSS(i,j-1) - 2*CRU_lattice.CaSS(i,j));
                    Jiss_JSR(i,j) = parameters.rijsr * (CRU_lattice.CaJSR(i+1,j) + CRU_lattice.CaJSR(i,j-1) - 2*CRU_lattice.CaJSR(i,j));
                }
                else if ((i == nCRU_x-1) && (j == 0)) {
                    // bottom left corner
                    Jiss_DS(i,j) = parameters.riss * (CRU_lattice.CaSS(i-1,j) + CRU_lattice.CaSS(i,j+1) - 2*CRU_lattice.CaSS(i,j));
                    Jiss_JSR(i,j) = parameters.rijsr * (CRU_lattice.CaJSR(i-1,j) + CRU_lattice.CaJSR(i,j+1) - 2*CRU_lattice.CaJSR(i,j));
                }
                else {
                    // bottom right corner
                    Jiss_DS(i,j) = parameters.riss * (CRU_lattice.CaSS(i-1,j) + CRU_lattice.CaSS(i,j-1) - 2*CRU_lattice.CaSS(i,j));
                    Jiss_JSR(i,j) = parameters.rijsr * (CRU_lattice.CaJSR(i-1,j) + CRU_lattice.CaJSR(i,j-1) - 2*CRU_lattice.CaJSR(i,j));
                }

                betaSS(i,j) = 1.0 / (1 + (consts.BSR_const / square(parameters.KBSR + CRU_lattice.CaSS(i,j))) + (consts.BSL_const / square(parameters.KBSL + CRU_lattice.CaSS(i,j))));
                betaJSR(i,j) = 1.0 / (1 + (consts.CSQN_const / square(parameters.KCSQN + CRU_lattice.CaJSR(i,j))));
            }
        }
    }
    
    template <typename FloatType>
    void GW_lattice<FloatType>::update_fluxes(){
        // Update these terms
        consts.VF_RT = globals.V*consts.F_RT;
        consts.expmVF_RT = exp(-consts.VF_RT);
        consts.JLCC_exp = square(1.0 / consts.expmVF_RT);
        consts.JLCC_multiplier = consts.JLCC_const * consts.VF_RT / (consts.JLCC_exp - 1);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < nCRU_x; i++){
            for (int j = 0; j < nCRU_y; j++){
                Jtr(i,j) = parameters.rtr * (globals.CaNSR - CRU_lattice.CaJSR(i,j));
                Jxfer(i,j) = parameters.rxfer * (CRU_lattice.CaSS(i,j) - globals.Cai); 
                if ((CRU_lattice.LCC_inactivation(i,j) == 1) && (CRU_lattice.LCC(i,j) == 6 || CRU_lattice.LCC(i,j) == 12))
                    JLCC(i,j) = consts.JLCC_multiplier * (consts.Cao_scaled - consts.JLCC_exp * CRU_lattice.CaSS(i,j));
                else {
                    JLCC(i,j) = 0.0;
                }

                Jrel(i,j) = (CRU_lattice.RyR.array(i,j,2) + CRU_lattice.RyR.array(i,j,3)) * parameters.rRyR * (CRU_lattice.CaJSR(i,j) - CRU_lattice.CaSS(i,j));

                betaSS(i,j) = 1.0 / (1 + (consts.BSR_const / square(parameters.KBSR + CRU_lattice.CaSS(i,j))) + (consts.BSL_const / square(parameters.KBSL + CRU_lattice.CaSS(i,j))));
                betaJSR(i,j) = 1.0 / (1 + (consts.CSQN_const / square(parameters.KCSQN + CRU_lattice.CaJSR(i,j))));
            }
        }
    }




    template <typename FloatType>
    inline void GW_lattice<FloatType>::euler_diffusion_step(const FloatType dt){
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < nCRU_x; i++){
            for (int j = 0; j < nCRU_y; j++){
                CRU_lattice.CaSS(i,j) += 0.5*dt*betaSS(i,j)*Jiss_DS(i,j);
                CRU_lattice.CaJSR(i,j) += 0.5*dt*betaJSR(i,j)*Jiss_JSR(i,j);
                if (CRU_lattice.CaSS(i,j) < 1e-10)
                    CRU_lattice.CaSS(i,j) = 1e-10; // Clamp this to prevent negative values
                if (CRU_lattice.CaJSR(i,j) < 1e-10)
                    CRU_lattice.CaJSR(i,j) = 1e-10; // Clamp this to prevent negative values
            }
        }
    }

    template <typename FloatType>
    void GW_lattice<FloatType>::euler_reaction_step(const FloatType dt){
        // Get all currents and fluxes
        FloatType ENa = common::Nernst<FloatType>(globals.Nai, parameters.Nao, consts.RT_F, 1.0);
        FloatType EK = common::Nernst<FloatType>(globals.Ki, parameters.Ko, consts.RT_F, 1.0);
        FloatType ECa = common::Nernst<FloatType>(globals.Cai, parameters.Cao, consts.RT_F, 2.0);

        FloatType INa = common::INa<FloatType>(globals.V, globals.m, globals.h, globals.j, parameters.GNa, ENa);
        FloatType INab = common::Ib<FloatType>(globals.V, parameters.GNab, ENa);
        FloatType INaCa = common::INaCa<FloatType>(consts.VF_RT, consts.expmVF_RT, globals.Nai, globals.Cai, consts.Nao3, parameters.Cao, parameters.eta, parameters.ksat, consts.INaCa_const);
        FloatType INaK = common::INaK<FloatType>(consts.VF_RT, consts.expmVF_RT, globals.Nai, consts.sigma, parameters.KmNai, consts.INaK_const);
        
        FloatType IKr = GW::IKr<FloatType>(globals.V, globals.Kr[3], EK, parameters.GKr, consts.sqrtKo);
        FloatType IKs = GW::IKs<FloatType>(globals.V, globals.xKs, globals.Ki, globals.Nai, parameters.Nao, parameters.Ko, parameters.GKs, consts.RT_F);
        FloatType IKv14 = GW::IKv14<FloatType>(consts.VF_RT, consts.expmVF_RT, globals.Kv14[4], globals.Ki, globals.Nai, consts.PKv14_Csc, parameters.Nao, parameters.Ko);
        FloatType IKv43 = GW::IKv43<FloatType>(globals.V, globals.Kv43[4], EK, parameters.GKv43);
        FloatType Ito1 = IKv14 + IKv43;
        FloatType Ito2 = GW::Ito2<FloatType>(CRU_lattice.ClCh, consts.VF_RT, consts.expmVF_RT, parameters.Clcyto, parameters.Clo, consts.Ito2_const);
        FloatType IK1 = GW::IK1<FloatType>(globals.V, EK, parameters.GK1, consts.IK1_const, consts.F_RT);
        FloatType IKp = GW::IKp<FloatType>(globals.V, EK, parameters.GKp);
        
        FloatType ICaL = GW::ICaL<FloatType>(JLCC, consts.ICaL_const);
        FloatType ICab = common::Ib<FloatType>(globals.V, parameters.GCab, ECa);
        FloatType IpCa = common::IpCa<FloatType>(globals.Cai, parameters.IpCamax, parameters.KmpCa);

        FloatType Jup = GW::Jup<FloatType>(globals.Cai, globals.CaNSR, parameters.Vmaxf, parameters.Vmaxr, parameters.Kmf, parameters.Kmr, parameters.Hf, parameters.Hr);
        FloatType Jtr_tot = GW::flux_average<FloatType>(Jtr, consts.CRU_factor);
        FloatType Jxfer_tot = GW::flux_average<FloatType>(Jxfer, consts.CRU_factor);
        FloatType beta_cyto = GW::beta_cyto<FloatType>(globals.Cai, consts.CMDN_const, parameters.KCMDN);

        FloatType dCaLTRPN = GW::dTRPNCa<FloatType>(globals.CaLTRPN, globals.Cai, parameters.LTRPNtot, parameters.kLTRPNp, parameters.kLTRPNm);
        FloatType dCaHTRPN = GW::dTRPNCa<FloatType>(globals.CaHTRPN, globals.Cai, parameters.HTRPNtot, parameters.kHTRPNp, parameters.kHTRPNm);

        // Calculate derivatives of gating terms
        FloatType dm = common::alpha_m(globals.V) * (1.0 - globals.m) - common::beta_m(globals.V) * globals.m;
        FloatType dh = common::alpha_h(globals.V) * (1.0 - globals.h) - common::beta_h(globals.V) * globals.h;
        FloatType dj = common::alpha_j(globals.V) * (1.0 - globals.j) - common::beta_j(globals.V) * globals.j;
        FloatType dxKs =  (GW::XKsinf(globals.V) - globals.xKs) * GW::tauXKs_inv(globals.V);
        
        update_QKr();
        // Calculate derivatives of XKr terms
        FloatType dXKr0 = QKr(1,0)*globals.Kr[1] + QKr(0,0)*globals.Kr[0];
        FloatType dXKr1 = QKr(0,1)*globals.Kr[0] + QKr(2,1)*globals.Kr[2] + QKr(1,1)*globals.Kr[1];
        FloatType dXKr2 = QKr(1,2)*globals.Kr[1] + QKr(3,2)*globals.Kr[3] + QKr(4,2)*globals.Kr[4] + QKr(2,2)*globals.Kr[2];
        FloatType dXKr3 = QKr(2,3)*globals.Kr[2] + QKr(4,3)*globals.Kr[4] + QKr(3,3)*globals.Kr[3];
        FloatType dXKr4 = QKr(2,4)*globals.Kr[2] + QKr(3,4)*globals.Kr[3] + QKr(4,4)*globals.Kr[4];
        
        update_QKv(); 
        // Calculate XKv14 derivatives
        FloatType dXKv14_0 = QKv14(1,0)*globals.Kv14[1] + QKv14(5,0)*globals.Kv14[5] + QKv14(0,0)*globals.Kv14[0];
        FloatType dXKv14_1 = QKv14(0,1)*globals.Kv14[0] + QKv14(2,1)*globals.Kv14[2] + QKv14(6,1)*globals.Kv14[6] + QKv14(1,1)*globals.Kv14[1];
        FloatType dXKv14_2 = QKv14(1,2)*globals.Kv14[1] + QKv14(3,2)*globals.Kv14[3] + QKv14(7,2)*globals.Kv14[7] + QKv14(2,2)*globals.Kv14[2];
        FloatType dXKv14_3 = QKv14(2,3)*globals.Kv14[2] + QKv14(4,3)*globals.Kv14[4] + QKv14(8,3)*globals.Kv14[8] + QKv14(3,3)*globals.Kv14[3];
        FloatType dXKv14_4 = QKv14(3,4)*globals.Kv14[3] + QKv14(9,4)*globals.Kv14[9] + QKv14(4,4)*globals.Kv14[4];
        FloatType dXKv14_5 = QKv14(6,5)*globals.Kv14[6] + QKv14(0,5)*globals.Kv14[0] + QKv14(5,5)*globals.Kv14[5];
        FloatType dXKv14_6 = QKv14(5,6)*globals.Kv14[5] + QKv14(7,6)*globals.Kv14[7] + QKv14(1,6)*globals.Kv14[1] + QKv14(6,6)*globals.Kv14[6];
        FloatType dXKv14_7 = QKv14(6,7)*globals.Kv14[6] + QKv14(8,7)*globals.Kv14[8] + QKv14(2,7)*globals.Kv14[2] + QKv14(7,7)*globals.Kv14[7];
        FloatType dXKv14_8 = QKv14(7,8)*globals.Kv14[7] + QKv14(9,8)*globals.Kv14[9] + QKv14(3,8)*globals.Kv14[3] + QKv14(8,8)*globals.Kv14[8];
        FloatType dXKv14_9 = QKv14(8,9)*globals.Kv14[8] + QKv14(4,9)*globals.Kv14[4] + QKv14(9,9)*globals.Kv14[9];
        
        // Calculate XKv43 derivatives
        FloatType dXKv43_0 = QKv43(1,0)*globals.Kv43[1] + QKv43(5,0)*globals.Kv43[5] + QKv43(0,0)*globals.Kv43[0];
        FloatType dXKv43_1 = QKv43(0,1)*globals.Kv43[0] + QKv43(2,1)*globals.Kv43[2] + QKv43(6,1)*globals.Kv43[6] + QKv43(1,1)*globals.Kv43[1];
        FloatType dXKv43_2 = QKv43(1,2)*globals.Kv43[1] + QKv43(3,2)*globals.Kv43[3] + QKv43(7,2)*globals.Kv43[7] + QKv43(2,2)*globals.Kv43[2];
        FloatType dXKv43_3 = QKv43(2,3)*globals.Kv43[2] + QKv43(4,3)*globals.Kv43[4] + QKv43(8,3)*globals.Kv43[8] + QKv43(3,3)*globals.Kv43[3];
        FloatType dXKv43_4 = QKv43(3,4)*globals.Kv43[3] + QKv43(9,4)*globals.Kv43[9] + QKv43(4,4)*globals.Kv43[4];
        FloatType dXKv43_5 = QKv43(6,5)*globals.Kv43[6] + QKv43(0,5)*globals.Kv43[0] + QKv43(5,5)*globals.Kv43[5];
        FloatType dXKv43_6 = QKv43(5,6)*globals.Kv43[5] + QKv43(7,6)*globals.Kv43[7] + QKv43(1,6)*globals.Kv43[1] + QKv43(6,6)*globals.Kv43[6];
        FloatType dXKv43_7 = QKv43(6,7)*globals.Kv43[6] + QKv43(8,7)*globals.Kv43[8] + QKv43(2,7)*globals.Kv43[2] + QKv43(7,7)*globals.Kv43[7];
        FloatType dXKv43_8 = QKv43(7,8)*globals.Kv43[7] + QKv43(9,8)*globals.Kv43[9] + QKv43(3,8)*globals.Kv43[3] + QKv43(8,8)*globals.Kv43[8];
        FloatType dXKv43_9 = QKv43(8,9)*globals.Kv43[8] + QKv43(4,9)*globals.Kv43[4] + QKv43(9,9)*globals.Kv43[9];

        globals.V += dt*(Istim - (INa + ICaL + IKr + IKs + Ito1 + IK1 + IKp + Ito2 + INaK + INaCa + IpCa + ICab + INab));

        globals.Nai += -dt*consts.CSA_FVcyto * (INa + INab + 3*INaCa + 3*INaK);
        globals.Ki += -dt*consts.CSA_FVcyto * (IKr + IKs + Ito1 + IK1 + IKp - 2*INaK);
        globals.Cai += dt*beta_cyto*(-0.5*consts.CSA_FVcyto*(ICab + IpCa - 2*INaCa) + consts.VSS_Vcyto*Jxfer_tot - Jup - (dCaLTRPN + dCaHTRPN));
        globals.CaNSR += dt*(consts.Vcyto_VNSR * Jup - consts.VJSR_VNSR * Jtr_tot);
        globals.CaLTRPN += dt*dCaLTRPN;
        globals.CaHTRPN += dt*dCaHTRPN;

        globals.m += dt*dm;
        globals.h += dt*dh;
        globals.j += dt*dj;
        globals.xKs += dt*dxKs;

        globals.Kr[0] += dt*dXKr0;
        globals.Kr[1] += dt*dXKr1;
        globals.Kr[2] += dt*dXKr2;
        globals.Kr[3] += dt*dXKr3;
        globals.Kr[4] += dt*dXKr4;

        globals.Kv14[0] += dt*dXKv14_0;
        globals.Kv14[1] += dt*dXKv14_1;
        globals.Kv14[2] += dt*dXKv14_2;
        globals.Kv14[3] += dt*dXKv14_3;
        globals.Kv14[4] += dt*dXKv14_4;
        globals.Kv14[5] += dt*dXKv14_5;
        globals.Kv14[6] += dt*dXKv14_6;
        globals.Kv14[7] += dt*dXKv14_7;
        globals.Kv14[8] += dt*dXKv14_8;
        globals.Kv14[9] += dt*dXKv14_9;
        
        globals.Kv43[0] += dt*dXKv43_0;
        globals.Kv43[1] += dt*dXKv43_1;
        globals.Kv43[2] += dt*dXKv43_2;
        globals.Kv43[3] += dt*dXKv43_3;
        globals.Kv43[4] += dt*dXKv43_4;
        globals.Kv43[5] += dt*dXKv43_5;
        globals.Kv43[6] += dt*dXKv43_6;
        globals.Kv43[7] += dt*dXKv43_7;
        globals.Kv43[8] += dt*dXKv43_8;
        globals.Kv43[9] += dt*dXKv43_9;

        // Update the non-diffusion terms of CaSS and CaJSR
        CRU_lattice.CaSS += dt*betaSS*(JLCC + Jrel - Jxfer);
        CRU_lattice.CaJSR += dt*betaJSR*(Jtr - consts.VSS_VJSR*Jrel);

    }

    template <typename FloatType>
    template <typename PRNG>
    void GW_lattice<FloatType>::SSA(const FloatType dt){
        consts.alphaLCC = GW::alphaLCC<FloatType>(globals.V);
        consts.betaLCC = GW::betaLCC<FloatType>(globals.V);
        consts.yinfLCC = GW::yinfLCC<FloatType>(globals.V);
        consts.tauLCC = GW::tauLCC<FloatType>(globals.V);

        #pragma omp parallel
        {
            CRULatticeStateThread<FloatType> temp;
            
            #pragma omp for collapse(2) schedule( static )
            for (int i = 0; i < nCRU_x; i++){
                for (int j = 0; j < nCRU_y; j++){
                    temp.copy_from_CRULatticeState(CRU_lattice, i, j, parameters);
                    SSA_single_su<FloatType, PRNG>(temp, dt, parameters, consts);
                    update_CRUstate_from_temp(temp, i, j);
                }
            }
        }
    }

    template <typename FloatType>
    template <typename PRNG>
    void GW_lattice<FloatType>::euler_step(const FloatType dt){
        // TODO: I think the implementation of this is incorrect. Check and fix.
        
        SSA<PRNG>(dt); // Do SSA step and hold the Markov chain states in temporary arrays

        // First diffusion half step
        update_diffusion_fluxes(); // Update fluxes to do diffusion half step
        euler_diffusion_step(dt);

        // Reaction step
        update_fluxes();
        euler_reaction_step(dt);

        // Second diffusion half step
        update_diffusion_fluxes();
        euler_diffusion_step(dt);

        // Copy Markov Chain states into the CRU_lattice object
        // Check that this is copying correctly
        CRU_lattice.LCC = LCC_tmp; 
        CRU_lattice.LCC_inactivation = LCC_inactivation_tmp;
        CRU_lattice.RyR.set(RyR_tmp);
        CRU_lattice.ClCh = ClCh_tmp; 
    }

    template <typename FloatType>
    template <typename PRNG>
    void GW_lattice<FloatType>::euler(const FloatType dt, const int nstep, const std::function<FloatType(FloatType)>& Ist){
        FloatType t = 0.0;
        for (int i = 0; i < nstep; ++i){
            Istim = Ist(t);
            euler_step(dt);
            t += dt;
        }
    }



}