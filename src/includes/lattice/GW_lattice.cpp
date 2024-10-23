#include "GW_lattice.hpp"


namespace GW_lattice {

    void GW_lattice::set_initial_value(GlobalState& global_vals, CRULatticeState& cru_vals){
        globals = global_vals; 
        CRU_lattice = cru_vals;
    }


    void GW_lattice::update_QKr(){
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


    void GW_lattice::update_CRUstate_from_temp(const CRULatticeStateThread& temp, const int x, const int y){
        LCC_tmp(x,y) = temp.LCC;
        LCC_inactivation_tmp(x,y) = temp.LCC_inactivation;
        for (int j = 0; j < 6; j++)
            RyR_tmp.array(x,y,j) = temp.RyR[j];            
        ClCh_tmp(x,y) = temp.ClCh;
    }

    void GW_lattice::update_QKv(){
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



    void GW_lattice::update_diffusion_fluxes(){
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < nCRU_x; i++){
            for (int j = 0; j < nCRU_y; j++){
                // Diffusion terms
                if ((i > 0) && (i < nCRU_x-1) && (j > 0) && (j < nCRU_y-1)) {
                    // interior
                    Jcyto(i,j) = parameters.rcyto * (CRU_lattice.Cai(i-1,j) + CRU_lattice.Cai(i+1,j) + CRU_lattice.Cai(i,j-1) + CRU_lattice.Cai(i,j+1) - 4*CRU_lattice.Cai(i,j));
                    JNSR(i,j) = parameters.rnsr * (CRU_lattice.CaNSR(i-1,j) + CRU_lattice.CaNSR(i+1,j) + CRU_lattice.CaNSR(i,j-1) + CRU_lattice.CaNSR(i,j+1) - 4*CRU_lattice.CaNSR(i,j));
                }
                else if ((i == 0) && (i < nCRU_x-1) && (j > 0) && (j < nCRU_y-1)) {
                    // left side, no corner
                    Jcyto(i,j) = parameters.rcyto * (CRU_lattice.Cai(i+1,j) + CRU_lattice.Cai(i,j-1) + CRU_lattice.Cai(i,j+1) - 3*CRU_lattice.Cai(i,j));
                    JNSR(i,j) = parameters.rnsr * (CRU_lattice.CaNSR(i+1,j) + CRU_lattice.CaNSR(i,j-1) + CRU_lattice.CaNSR(i,j+1) - 3*CRU_lattice.CaNSR(i,j));
                }
                else if ((i > 0) && (i == nCRU_x-1) && (j > 0) && (j < nCRU_y-1)) {
                    // right side, no corner
                    Jcyto(i,j) = parameters.rcyto * (CRU_lattice.Cai(i-1,j) + CRU_lattice.Cai(i,j-1) + CRU_lattice.Cai(i,j+1) - 3*CRU_lattice.Cai(i,j));
                    JNSR(i,j) = parameters.rnsr * (CRU_lattice.CaNSR(i-1,j) + CRU_lattice.CaNSR(i,j-1) + CRU_lattice.CaNSR(i,j+1) - 3*CRU_lattice.CaNSR(i,j));
                }
                else if ((i > 0) && (i < nCRU_x-1) && (j == 0) && (j < nCRU_y-1)) {
                    // top, no corner
                    Jcyto(i,j) = parameters.rcyto * (CRU_lattice.Cai(i-1,j) + CRU_lattice.Cai(i+1,j) + CRU_lattice.Cai(i,j+1) - 3*CRU_lattice.Cai(i,j));
                    JNSR(i,j) = parameters.rnsr * (CRU_lattice.CaNSR(i-1,j) + CRU_lattice.CaNSR(i+1,j) + CRU_lattice.CaNSR(i,j+1) - 3*CRU_lattice.CaNSR(i,j));
                }
                else if ((i > 0) && (i < nCRU_x-1) && (j > 0) && (j == nCRU_y-1)) {
                    // bottom, no corner
                    Jcyto(i,j) = parameters.rcyto * (CRU_lattice.Cai(i-1,j) + CRU_lattice.Cai(i+1,j) + CRU_lattice.Cai(i,j-1) - 3*CRU_lattice.Cai(i,j));
                    JNSR(i,j) = parameters.rnsr * (CRU_lattice.CaNSR(i-1,j) + CRU_lattice.CaNSR(i+1,j) + CRU_lattice.CaNSR(i,j-1) - 3*CRU_lattice.CaNSR(i,j));
                }
                else if ((i == 0) && (j == 0)) {
                    // top left corner
                    Jcyto(i,j) = parameters.rcyto * (CRU_lattice.Cai(i+1,j) + CRU_lattice.Cai(i,j+1) - 2*CRU_lattice.Cai(i,j));
                    JNSR(i,j) = parameters.rnsr * (CRU_lattice.CaNSR(i+1,j) + CRU_lattice.CaNSR(i,j+1) - 2*CRU_lattice.CaNSR(i,j));
                }
                else if ((i == 0) && (j == nCRU_y-1)) {
                    // top right corner
                    Jcyto(i,j) = parameters.rcyto * (CRU_lattice.Cai(i+1,j) + CRU_lattice.Cai(i,j-1) - 2*CRU_lattice.Cai(i,j));
                    JNSR(i,j) = parameters.rnsr * (CRU_lattice.CaNSR(i+1,j) + CRU_lattice.CaNSR(i,j-1) - 2*CRU_lattice.CaNSR(i,j));
                }
                else if ((i == nCRU_x-1) && (j == 0)) {
                    // bottom left corner
                    Jcyto(i,j) = parameters.rcyto * (CRU_lattice.Cai(i-1,j) + CRU_lattice.Cai(i,j+1) - 2*CRU_lattice.Cai(i,j));
                    JNSR(i,j) = parameters.rnsr * (CRU_lattice.CaNSR(i-1,j) + CRU_lattice.CaNSR(i,j+1) - 2*CRU_lattice.CaNSR(i,j));
                }
                else {
                    // bottom right corner
                    Jcyto(i,j) = parameters.rcyto * (CRU_lattice.Cai(i-1,j) + CRU_lattice.Cai(i,j-1) - 2*CRU_lattice.Cai(i,j));
                    JNSR(i,j) = parameters.rnsr * (CRU_lattice.CaNSR(i-1,j) + CRU_lattice.CaNSR(i,j-1) - 2*CRU_lattice.CaNSR(i,j));
                }

                beta_cyto(i,j) = 1.0 / (1 + consts.CMDN_const / square(parameters.KCMDN + CRU_lattice.Cai(i,j)));
            }
        }
    }
    
    void GW_lattice::update_fluxes(){
        // Update these terms
        consts.VF_RT = globals.V*consts.F_RT;
        consts.expmVF_RT = exp(-consts.VF_RT);
        consts.JLCC_exp = square(1.0 / consts.expmVF_RT);
        consts.JLCC_multiplier = consts.JLCC_const * globals.V * consts.F_RT / (consts.JLCC_exp - 1);

        //#pragma omp parallel for collapse(2)
        for (int i = 0; i < nCRU_x; ++i){
            for (int j = 0; j < nCRU_y; ++j){
                Jtr(i,j) = parameters.rtr * (CRU_lattice.CaNSR(i,j) - CRU_lattice.CaJSR(i,j));
                Jxfer(i,j) = parameters.rxfer * (CRU_lattice.CaSS(i,j) - CRU_lattice.Cai(i,j)); 
                if ((CRU_lattice.LCC(i,j) == 6 || CRU_lattice.LCC(i,j) == 12))
                    JLCC(i,j) = CRU_lattice.LCC_inactivation(i,j) * consts.JLCC_multiplier * (consts.Cao_scaled - consts.JLCC_exp * CRU_lattice.CaSS(i,j));
                else {
                    JLCC(i,j) = 0.0;
                }

                Jrel(i,j) = (CRU_lattice.RyR.array(i,j,2) + CRU_lattice.RyR.array(i,j,3)) * parameters.rRyR * (CRU_lattice.CaJSR(i,j) - CRU_lattice.CaSS(i,j));
                Jup(i,j) = common::Jup(CRU_lattice.Cai(i,j), CRU_lattice.CaNSR(i,j), parameters.Vmaxf, parameters.Vmaxr, parameters.Kmf, parameters.Kmr, parameters.Hf, parameters.Hr);

                betaSS(i,j) = 1.0 / (1 + (consts.BSR_const / square(parameters.KBSR + CRU_lattice.CaSS(i,j))) + (consts.BSL_const / square(parameters.KBSL + CRU_lattice.CaSS(i,j))));
                betaJSR(i,j) = 1.0 / (1 + (consts.CSQN_const / square(parameters.KCSQN + CRU_lattice.CaJSR(i,j))));
                beta_cyto(i,j) = 1.0 / (1 + consts.CMDN_const / square(parameters.KCMDN + CRU_lattice.Cai(i,j)));
        
                dCaLTRPN(i,j) = common::dTRPNCa(CRU_lattice.CaLTRPN(i,j), CRU_lattice.Cai(i,j), parameters.LTRPNtot, parameters.kLTRPNp, parameters.kLTRPNm);
                dCaHTRPN(i,j) = common::dTRPNCa(CRU_lattice.CaHTRPN(i,j), CRU_lattice.Cai(i,j), parameters.HTRPNtot, parameters.kHTRPNp, parameters.kHTRPNm);
            }
        }
    }




    inline void GW_lattice::euler_diffusion_step(const double dt){
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < nCRU_x; i++){
            for (int j = 0; j < nCRU_y; j++){
                CRU_lattice.Cai(i,j) += 0.5*dt*beta_cyto(i,j)*Jcyto(i,j);
                CRU_lattice.CaNSR(i,j) += 0.5*dt*JNSR(i,j);
                if (CRU_lattice.Cai(i,j) < 1e-10)
                    CRU_lattice.Cai(i,j) = 1e-10; // Clamp this to prevent negative values
                if (CRU_lattice.CaNSR(i,j) < 1e-10)
                    CRU_lattice.CaNSR(i,j) = 1e-10; // Clamp this to prevent negative values
            }
        }
    }

    void GW_lattice::euler_reaction_step(const double dt){
        // Get all currents and fluxes
        Cai_tot = CRU_lattice.Cai.sum() / (nCRU_x * nCRU_y);
        double ENa = common::Nernst<double>(globals.Nai, parameters.Nao, consts.RT_F, 1.0);
        double EK = common::Nernst<double>(globals.Ki, parameters.Ko, consts.RT_F, 1.0);
        double ECa = common::Nernst<double>(Cai_tot, parameters.Cao, consts.RT_F, 2.0);

        double INa = common::INa<double>(globals.V, globals.m, globals.h, globals.j, parameters.GNa, ENa);
        double INab = common::Ib<double>(globals.V, parameters.GNab, ENa);
        double INaCa = common::INaCa<double>(consts.VF_RT, consts.expmVF_RT, globals.Nai, Cai_tot, consts.Nao3, parameters.Cao, parameters.eta, parameters.ksat, consts.INaCa_const);
        double INaK = common::INaK<double>(consts.VF_RT, consts.expmVF_RT, globals.Nai, consts.sigma, parameters.KmNai, consts.INaK_const);
        
        double IKr = common::IKr(globals.V, globals.Kr[3], EK, parameters.GKr, consts.sqrtKo);
        double IKs = common::IKs(globals.V, globals.xKs, globals.Ki, globals.Nai, parameters.Nao, parameters.Ko, parameters.GKs, consts.RT_F);
        double IKv14 = common::IKv14(consts.VF_RT, consts.expmVF_RT, globals.Kv14[4], globals.Ki, globals.Nai, consts.PKv14_Csc, parameters.Nao, parameters.Ko);
        double IKv43 = common::IKv43(globals.V, globals.Kv43[4], EK, parameters.GKv43);
        double Ito1 = IKv14 + IKv43;
        double Ito2 = double(CRU_lattice.ClCh.sum()) * consts.Ito2_const * consts.VF_RT * (parameters.Clcyto * consts.expmVF_RT - parameters.Clo) / (consts.expmVF_RT - 1.0);
        double IK1 = common::IK1(globals.V, EK, parameters.GK1, consts.IK1_const, consts.F_RT);
        double IKp = common::IKp(globals.V, EK, parameters.GKp);
        
        double ICaL = JLCC.sum() * consts.ICaL_const;
        double ICab = common::Ib<double>(globals.V, parameters.GCab, ECa);
        double IpCa = common::IpCa<double>(Cai_tot, parameters.IpCamax, parameters.KmpCa);

        //double Jup = GW::Jup<double>(globals.Cai, globals.CaNSR, parameters.Vmaxf, parameters.Vmaxr, parameters.Kmf, parameters.Kmr, parameters.Hf, parameters.Hr);
        //double Jtr_tot = GW::flux_average<double>(Jtr, consts.CRU_factor);
        //double Jxfer_tot = GW::flux_average<double>(Jxfer, consts.CRU_factor);
        //double beta_cyto = GW::beta_cyto<double>(globals.Cai, consts.CMDN_const, parameters.KCMDN);

        //double dCaLTRPN = GW::dTRPNCa<double>(globals.CaLTRPN, globals.Cai, parameters.LTRPNtot, parameters.kLTRPNp, parameters.kLTRPNm);
        //double dCaHTRPN = GW::dTRPNCa<double>(globals.CaHTRPN, globals.Cai, parameters.HTRPNtot, parameters.kHTRPNp, parameters.kHTRPNm);

        // Calculate derivatives of gating terms
        double dm = common::alpha_m(globals.V) * (1.0 - globals.m) - common::beta_m(globals.V) * globals.m;
        double dh = common::alpha_h(globals.V) * (1.0 - globals.h) - common::beta_h(globals.V) * globals.h;
        double dj = common::alpha_j(globals.V) * (1.0 - globals.j) - common::beta_j(globals.V) * globals.j;
        double dxKs =  (common::XKsinf(globals.V) - globals.xKs) * common::tauXKs_inv(globals.V);
        
        update_QKr();
        // Calculate derivatives of XKr terms
        double dXKr0 = QKr(1,0)*globals.Kr[1] + QKr(0,0)*globals.Kr[0];
        double dXKr1 = QKr(0,1)*globals.Kr[0] + QKr(2,1)*globals.Kr[2] + QKr(1,1)*globals.Kr[1];
        double dXKr2 = QKr(1,2)*globals.Kr[1] + QKr(3,2)*globals.Kr[3] + QKr(4,2)*globals.Kr[4] + QKr(2,2)*globals.Kr[2];
        double dXKr3 = QKr(2,3)*globals.Kr[2] + QKr(4,3)*globals.Kr[4] + QKr(3,3)*globals.Kr[3];
        double dXKr4 = QKr(2,4)*globals.Kr[2] + QKr(3,4)*globals.Kr[3] + QKr(4,4)*globals.Kr[4];
        
        update_QKv(); 
        // Calculate XKv14 derivatives
        double dXKv14_0 = QKv14(1,0)*globals.Kv14[1] + QKv14(5,0)*globals.Kv14[5] + QKv14(0,0)*globals.Kv14[0];
        double dXKv14_1 = QKv14(0,1)*globals.Kv14[0] + QKv14(2,1)*globals.Kv14[2] + QKv14(6,1)*globals.Kv14[6] + QKv14(1,1)*globals.Kv14[1];
        double dXKv14_2 = QKv14(1,2)*globals.Kv14[1] + QKv14(3,2)*globals.Kv14[3] + QKv14(7,2)*globals.Kv14[7] + QKv14(2,2)*globals.Kv14[2];
        double dXKv14_3 = QKv14(2,3)*globals.Kv14[2] + QKv14(4,3)*globals.Kv14[4] + QKv14(8,3)*globals.Kv14[8] + QKv14(3,3)*globals.Kv14[3];
        double dXKv14_4 = QKv14(3,4)*globals.Kv14[3] + QKv14(9,4)*globals.Kv14[9] + QKv14(4,4)*globals.Kv14[4];
        double dXKv14_5 = QKv14(6,5)*globals.Kv14[6] + QKv14(0,5)*globals.Kv14[0] + QKv14(5,5)*globals.Kv14[5];
        double dXKv14_6 = QKv14(5,6)*globals.Kv14[5] + QKv14(7,6)*globals.Kv14[7] + QKv14(1,6)*globals.Kv14[1] + QKv14(6,6)*globals.Kv14[6];
        double dXKv14_7 = QKv14(6,7)*globals.Kv14[6] + QKv14(8,7)*globals.Kv14[8] + QKv14(2,7)*globals.Kv14[2] + QKv14(7,7)*globals.Kv14[7];
        double dXKv14_8 = QKv14(7,8)*globals.Kv14[7] + QKv14(9,8)*globals.Kv14[9] + QKv14(3,8)*globals.Kv14[3] + QKv14(8,8)*globals.Kv14[8];
        double dXKv14_9 = QKv14(8,9)*globals.Kv14[8] + QKv14(4,9)*globals.Kv14[4] + QKv14(9,9)*globals.Kv14[9];
        
        // Calculate XKv43 derivatives
        double dXKv43_0 = QKv43(1,0)*globals.Kv43[1] + QKv43(5,0)*globals.Kv43[5] + QKv43(0,0)*globals.Kv43[0];
        double dXKv43_1 = QKv43(0,1)*globals.Kv43[0] + QKv43(2,1)*globals.Kv43[2] + QKv43(6,1)*globals.Kv43[6] + QKv43(1,1)*globals.Kv43[1];
        double dXKv43_2 = QKv43(1,2)*globals.Kv43[1] + QKv43(3,2)*globals.Kv43[3] + QKv43(7,2)*globals.Kv43[7] + QKv43(2,2)*globals.Kv43[2];
        double dXKv43_3 = QKv43(2,3)*globals.Kv43[2] + QKv43(4,3)*globals.Kv43[4] + QKv43(8,3)*globals.Kv43[8] + QKv43(3,3)*globals.Kv43[3];
        double dXKv43_4 = QKv43(3,4)*globals.Kv43[3] + QKv43(9,4)*globals.Kv43[9] + QKv43(4,4)*globals.Kv43[4];
        double dXKv43_5 = QKv43(6,5)*globals.Kv43[6] + QKv43(0,5)*globals.Kv43[0] + QKv43(5,5)*globals.Kv43[5];
        double dXKv43_6 = QKv43(5,6)*globals.Kv43[5] + QKv43(7,6)*globals.Kv43[7] + QKv43(1,6)*globals.Kv43[1] + QKv43(6,6)*globals.Kv43[6];
        double dXKv43_7 = QKv43(6,7)*globals.Kv43[6] + QKv43(8,7)*globals.Kv43[8] + QKv43(2,7)*globals.Kv43[2] + QKv43(7,7)*globals.Kv43[7];
        double dXKv43_8 = QKv43(7,8)*globals.Kv43[7] + QKv43(9,8)*globals.Kv43[9] + QKv43(3,8)*globals.Kv43[3] + QKv43(8,8)*globals.Kv43[8];
        double dXKv43_9 = QKv43(8,9)*globals.Kv43[8] + QKv43(4,9)*globals.Kv43[4] + QKv43(9,9)*globals.Kv43[9];

        globals.V += dt*(Istim - (INa + ICaL + IKr + IKs + Ito1 + IK1 + IKp + Ito2 + INaK + INaCa + IpCa + ICab + INab));

        globals.Nai += -dt*(consts.CSA_F / parameters.Vcyto) * (INa + INab + 3*INaCa + 3*INaK);
        globals.Ki += -dt*(consts.CSA_F / parameters.Vcyto) * (IKr + IKs + Ito1 + IK1 + IKp - 2*INaK);

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
        // Note: when doing unit calculations properly, CSA / Vcyto is invariant under rescaling of area, so thats why Vcyto and not Vcyto_elem is used
        CRU_lattice.Cai += dt*beta_cyto*(-0.5*(consts.CSA_F / parameters.Vcyto) * (ICab + IpCa - 2*INaCa) + consts.VSS_Vcyto*Jxfer - Jup - (dCaLTRPN + dCaHTRPN));
        CRU_lattice.CaNSR += dt*(consts.Vcyto_VNSR * Jup - consts.VJSR_VNSR * Jtr);
        CRU_lattice.CaLTRPN += dt*dCaLTRPN;
        CRU_lattice.CaHTRPN += dt*dCaHTRPN;

    }


}