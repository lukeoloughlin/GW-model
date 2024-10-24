#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/eigen/tensor.h>
#include <pybind11/functional.h>

#include "includes/GW/GW.hpp"
#include "includes/xoshiro.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

namespace GW {

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
        
        currents.IKr = common::IKr(globals.V, globals.Kr[3], consts.EK, parameters.GKr, consts.sqrtKo);
        currents.IKs = common::IKs(globals.V, globals.xKs, globals.Ki, globals.Nai, parameters.Nao, parameters.Ko, parameters.GKs, consts.RT_F);
        IKv14 = common::IKv14(consts.VF_RT, consts.expmVF_RT, globals.Kv14[4], globals.Ki, globals.Nai, consts.PKv14_Csc, parameters.Nao, parameters.Ko);
        IKv43 = common::IKv43(globals.V, globals.Kv43[4], consts.EK, parameters.GKv43);
        currents.Ito1 = IKv14 + IKv43;
        currents.Ito2 = double(CRUs.ClCh.sum()) * consts.Ito2_const * consts.VF_RT * (parameters.Clcyto * consts.expmVF_RT - parameters.Clo) / (consts.expmVF_RT - 1.0);
        currents.IK1 = common::IK1(globals.V, consts.EK, parameters.GK1, consts.IK1_const, consts.F_RT);
        currents.IKp = common::IKp(globals.V, consts.EK, parameters.GKp);
        
        currents.ICaL = JLCC.sum() * consts.ICaL_const;
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


    void PyGWSimulation::record_state(const GW_model& model, const int idx, const int nCRU, const double t_){
        t(idx) = t_;
        V(idx) = model.globals.V;
        m(idx) = model.globals.m;
        h(idx) = model.globals.h;
        j(idx) = model.globals.j;
        Nai(idx) = model.globals.Nai;
        Ki(idx) = model.globals.Ki;
        Cai(idx) = model.globals.Cai;
        CaNSR(idx) = model.globals.CaNSR;
        CaLTRPN(idx) = model.globals.CaLTRPN;
        CaHTRPN(idx) = model.globals.CaHTRPN;
        xKs(idx) = model.globals.xKs;
        
        XKr(idx,0) = model.globals.Kr[0];
        XKr(idx,1) = model.globals.Kr[1];
        XKr(idx,2) = model.globals.Kr[2];
        XKr(idx,3) = model.globals.Kr[3];
        XKr(idx,4) = model.globals.Kr[4];

        for (int j = 0; j < 10; ++j){
            XKv14(idx,j) = model.globals.Kv14[j];
            XKv43(idx,j) = model.globals.Kv43[j];
        }

        for (int j = 0; j < nCRU; ++j){
            CaJSR(idx,j) = model.CRUs.CaJSR(j);
            for (int k = 0; k < 4; ++k){
                CaSS(idx,j,k) = model.CRUs.CaSS(j,k);
                LCC(idx,j,k) = model.CRUs.LCC(j,k);
                LCC_inactivation(idx,j,k) = model.CRUs.LCC_inactivation(j,k);
                RyR(idx,j,k,0) = model.CRUs.RyR.array(j,k,0);
                RyR(idx,j,k,1) = model.CRUs.RyR.array(j,k,1);
                RyR(idx,j,k,2) = model.CRUs.RyR.array(j,k,2);
                RyR(idx,j,k,3) = model.CRUs.RyR.array(j,k,3);
                RyR(idx,j,k,4) = model.CRUs.RyR.array(j,k,4);
                RyR(idx,j,k,5) = model.CRUs.RyR.array(j,k,5);
                ClCh(idx,j,k) = model.CRUs.ClCh(j,k);
            }
        }

        RyR_open_int(idx) = model.CRUs.RyR_open_int.sum() / model.get_nCRU();
        RyR_open_martingale(idx) = model.CRUs.RyR_open_martingale.sum() / model.get_nCRU();
        RyR_open_martingale_normalised(idx) = model.CRUs.RyR_open_martingale_normalised.sum() / model.get_nCRU();
        sigma_RyR(idx) = model.CRUs.sigma_RyR.mean();
        
        LCC_open_int(idx) = model.CRUs.LCC_open_int.sum() / model.get_nCRU();
        LCC_open_martingale(idx) = model.CRUs.LCC_open_martingale.sum() / model.get_nCRU();
        LCC_open_martingale_normalised(idx) = model.CRUs.LCC_open_martingale_normalised.sum() / model.get_nCRU();
        sigma_LCC(idx) = model.CRUs.sigma_LCC.mean();
    }

    void GW_model::init_from_python(const PyInitGWState& py_state){
        globals.V = py_state.V;
        globals.Nai = py_state.Nai;
        globals.Ki = py_state.Ki;
        globals.Cai = py_state.Cai;
        globals.CaNSR = py_state.CaNSR;
        globals.CaLTRPN = py_state.CaLTRPN;
        globals.CaHTRPN = py_state.CaHTRPN;
        globals.m = py_state.m;
        globals.h = py_state.h;
        globals.j = py_state.j;
        globals.xKs = py_state.xKs;

        for (int i = 0; i < 5; ++i)
            globals.Kr[i] = py_state.XKr(i);

        for (int i = 0; i < 10; ++i){
            globals.Kv14[i] = py_state.XKv14(i);
            globals.Kv43[i] = py_state.XKv43(i);
        }

        // These work fine because 2d array is compatible with numpy array
        CRUs.CaSS = py_state.CaSS; 
        CRUs.CaJSR = py_state.CaJSR;
        CRUs.LCC = py_state.LCC;
        CRUs.LCC_inactivation = py_state.LCC_inactivation;
        CRUs.ClCh = py_state.ClCh;
        // This is where the numpy array and Eigen::TensorMap don't get along so have to copy in loop
        for (int i = 0; i < nCRU; ++i){
            for (int j = 0; j < 4; ++j){
                for (int k = 0; k < 6; ++k)
                    CRUs.RyR.array(i,j,k) = py_state.RyR(i,j,k);
            }
        }
    }

}

void init_GW(py::module& m){
    py::class_<GW::Parameters>(m, "GWParameters")
        .def(py::init<>())
        .def_readwrite("T", &GW::Parameters::T)
        .def_readwrite("CSA", &GW::Parameters::CSA)
        .def_readwrite("Vcyto", &GW::Parameters::Vcyto)
        .def_readwrite("VNSR", &GW::Parameters::VNSR)
        .def_readwrite("VJSR", &GW::Parameters::VJSR)
        .def_readwrite("VSS", &GW::Parameters::VSS)
        .def_readwrite("NCaRU", &GW::Parameters::NCaRU)
        .def_readwrite("Ko", &GW::Parameters::Ko)
        .def_readwrite("Nao", &GW::Parameters::Nao)
        .def_readwrite("Cao", &GW::Parameters::Cao)
        .def_readwrite("Clo", &GW::Parameters::Clo)
        .def_readwrite("Clcyto", &GW::Parameters::Clcyto)
        .def_readwrite("f", &GW::Parameters::f)
        .def_readwrite("g", &GW::Parameters::g)
        .def_readwrite("f1", &GW::Parameters::f1)
        .def_readwrite("g1", &GW::Parameters::g1)
        .def_readwrite("a", &GW::Parameters::a)
        .def_readwrite("b", &GW::Parameters::b)
        .def_readwrite("gamma0", &GW::Parameters::gamma0)
        .def_readwrite("omega", &GW::Parameters::omega)
        .def_readwrite("PCaL", &GW::Parameters::PCaL)
        .def_readwrite("kfClCh", &GW::Parameters::kfClCh)
        .def_readwrite("kbClCh", &GW::Parameters::kbClCh)
        .def_readwrite("Pto2", &GW::Parameters::Pto2)
        .def_readwrite("k12", &GW::Parameters::k12)
        .def_readwrite("k21", &GW::Parameters::k21)
        .def_readwrite("k23", &GW::Parameters::k23)
        .def_readwrite("k32", &GW::Parameters::k32)
        .def_readwrite("k34", &GW::Parameters::k34)
        .def_readwrite("k43", &GW::Parameters::k43)
        .def_readwrite("k45", &GW::Parameters::k45)
        .def_readwrite("k54", &GW::Parameters::k54)
        .def_readwrite("k56", &GW::Parameters::k56)
        .def_readwrite("k65", &GW::Parameters::k65)
        .def_readwrite("k25", &GW::Parameters::k25)
        .def_readwrite("k52", &GW::Parameters::k52)
        .def_readwrite("rRyR", &GW::Parameters::rRyR)
        .def_readwrite("rxfer", &GW::Parameters::rxfer)
        .def_readwrite("rtr", &GW::Parameters::rtr)
        .def_readwrite("riss", &GW::Parameters::riss)
        .def_readwrite("BSRT", &GW::Parameters::BSRT)
        .def_readwrite("KBSR", &GW::Parameters::KBSR)
        .def_readwrite("BSLT", &GW::Parameters::BSLT)
        .def_readwrite("KBSL", &GW::Parameters::KBSL)
        .def_readwrite("CSQNT", &GW::Parameters::CSQNT)
        .def_readwrite("KCSQN", &GW::Parameters::KCSQN)
        .def_readwrite("CMDNT", &GW::Parameters::CMDNT)
        .def_readwrite("KCMDN", &GW::Parameters::KCMDN)
        .def_readwrite("GNa", &GW::Parameters::GNa)
        .def_readwrite("GKr", &GW::Parameters::GKr)
        .def_readwrite("Kf", &GW::Parameters::Kf)
        .def_readwrite("Kb", &GW::Parameters::Kb)
        .def_readwrite("GKs", &GW::Parameters::GKs)
        .def_readwrite("GKv43", &GW::Parameters::GKv43)
        .def_readwrite("alphaa0Kv43", &GW::Parameters::alphaa0Kv43)
        .def_readwrite("aaKv43", &GW::Parameters::aaKv43)
        .def_readwrite("betaa0Kv43", &GW::Parameters::betaa0Kv43)
        .def_readwrite("baKv43", &GW::Parameters::baKv43)
        .def_readwrite("alphai0Kv43", &GW::Parameters::alphai0Kv43)
        .def_readwrite("aiKv43", &GW::Parameters::aiKv43)
        .def_readwrite("betai0Kv43", &GW::Parameters::betai0Kv43)
        .def_readwrite("biKv43", &GW::Parameters::biKv43)
        .def_readwrite("f1Kv43", &GW::Parameters::f1Kv43)
        .def_readwrite("f2Kv43", &GW::Parameters::f2Kv43)
        .def_readwrite("f3Kv43", &GW::Parameters::f3Kv43)
        .def_readwrite("f4Kv43", &GW::Parameters::f4Kv43)
        .def_readwrite("b1Kv43", &GW::Parameters::b1Kv43)
        .def_readwrite("b2Kv43", &GW::Parameters::b2Kv43)
        .def_readwrite("b3Kv43", &GW::Parameters::b3Kv43)
        .def_readwrite("b4Kv43", &GW::Parameters::b4Kv43)
        .def_readwrite("PKv14", &GW::Parameters::PKv14)
        .def_readwrite("alphaa0Kv14", &GW::Parameters::alphaa0Kv14)
        .def_readwrite("aaKv14", &GW::Parameters::aaKv14)
        .def_readwrite("betaa0Kv14", &GW::Parameters::betaa0Kv14)
        .def_readwrite("baKv14", &GW::Parameters::baKv14)
        .def_readwrite("alphai0Kv14", &GW::Parameters::alphai0Kv14)
        .def_readwrite("aiKv14", &GW::Parameters::aiKv14)
        .def_readwrite("betai0Kv14", &GW::Parameters::betai0Kv14)
        .def_readwrite("biKv14", &GW::Parameters::biKv14)
        .def_readwrite("f1Kv14", &GW::Parameters::f1Kv14)
        .def_readwrite("f2Kv14", &GW::Parameters::f2Kv14)
        .def_readwrite("f3Kv14", &GW::Parameters::f3Kv14)
        .def_readwrite("f4Kv14", &GW::Parameters::f4Kv14)
        .def_readwrite("b1Kv14", &GW::Parameters::b1Kv14)
        .def_readwrite("b2Kv14", &GW::Parameters::b2Kv14)
        .def_readwrite("b3Kv14", &GW::Parameters::b3Kv14)
        .def_readwrite("b4Kv14", &GW::Parameters::b4Kv14)
        .def_readwrite("Csc", &GW::Parameters::Csc)
        .def_readwrite("GK1", &GW::Parameters::GK1)
        .def_readwrite("KmK1", &GW::Parameters::KmK1)
        .def_readwrite("GKp", &GW::Parameters::GKp)
        .def_readwrite("kNaCa", &GW::Parameters::kNaCa)
        .def_readwrite("KmNa", &GW::Parameters::KmNa)
        .def_readwrite("KmCa", &GW::Parameters::KmCa)
        .def_readwrite("ksat", &GW::Parameters::ksat)
        .def_readwrite("eta", &GW::Parameters::eta)
        .def_readwrite("INaKmax", &GW::Parameters::INaKmax)
        .def_readwrite("KmNai", &GW::Parameters::KmNai)
        .def_readwrite("KmKo", &GW::Parameters::KmKo)
        .def_readwrite("IpCamax", &GW::Parameters::IpCamax)
        .def_readwrite("KmpCa", &GW::Parameters::KmpCa)
        .def_readwrite("GCab", &GW::Parameters::GCab)
        .def_readwrite("GNab", &GW::Parameters::GNab)
        .def_readwrite("kHTRPNp", &GW::Parameters::kHTRPNp)
        .def_readwrite("kHTRPNm", &GW::Parameters::kHTRPNm)
        .def_readwrite("kLTRPNp", &GW::Parameters::kLTRPNp)
        .def_readwrite("kLTRPNm", &GW::Parameters::kLTRPNm)
        .def_readwrite("HTRPNtot", &GW::Parameters::HTRPNtot)
        .def_readwrite("LTRPNtot", &GW::Parameters::LTRPNtot)
        .def_readwrite("Vmaxf", &GW::Parameters::Vmaxf)
        .def_readwrite("Vmaxr", &GW::Parameters::Vmaxr)
        .def_readwrite("Kmf", &GW::Parameters::Kmf)
        .def_readwrite("Kmr", &GW::Parameters::Kmr)
        .def_readwrite("Hf", &GW::Parameters::Hf)
        .def_readwrite("Hr", &GW::Parameters::Hr);


    py::class_<GW::PyGWSimulation>(m, "GWSimulation")
        .def(py::init<int,int,double>())
        .def_readwrite("t", &GW::PyGWSimulation::t)
        .def_readwrite("V", &GW::PyGWSimulation::V)
        .def_readwrite("m", &GW::PyGWSimulation::m)
        .def_readwrite("h", &GW::PyGWSimulation::h)
        .def_readwrite("j", &GW::PyGWSimulation::j)
        .def_readwrite("Nai", &GW::PyGWSimulation::Nai)
        .def_readwrite("Ki", &GW::PyGWSimulation::Ki)
        .def_readwrite("Cai", &GW::PyGWSimulation::Cai)
        .def_readwrite("CaNSR", &GW::PyGWSimulation::CaNSR)
        .def_readwrite("CaLTRPN", &GW::PyGWSimulation::CaLTRPN)
        .def_readwrite("CaHTRPN", &GW::PyGWSimulation::CaHTRPN)
        .def_readwrite("xKs", &GW::PyGWSimulation::xKs)
        .def_readwrite("XKr", &GW::PyGWSimulation::XKr)
        .def_readwrite("XKv14", &GW::PyGWSimulation::XKv14)
        .def_readwrite("XKv43", &GW::PyGWSimulation::XKv43)
        .def_readwrite("CaJSR", &GW::PyGWSimulation::CaJSR)
        .def_readwrite("CaSS", &GW::PyGWSimulation::CaSS)
        .def_readwrite("LCC", &GW::PyGWSimulation::LCC)
        .def_readwrite("LCC_inactivation", &GW::PyGWSimulation::LCC_inactivation)
        .def_readwrite("RyR", &GW::PyGWSimulation::RyR)
        .def_readwrite("ClCh", &GW::PyGWSimulation::ClCh)
        .def_readwrite("RyR_open_int", &GW::PyGWSimulation::RyR_open_int)
        .def_readwrite("RyR_open_martingale", &GW::PyGWSimulation::RyR_open_martingale)
        .def_readwrite("RyR_open_martingale_normalised", &GW::PyGWSimulation::RyR_open_martingale_normalised)
        .def_readwrite("sigma_RyR", &GW::PyGWSimulation::sigma_RyR)
        .def_readwrite("LCC_open_int", &GW::PyGWSimulation::LCC_open_int)
        .def_readwrite("LCC_open_martingale", &GW::PyGWSimulation::LCC_open_martingale)
        .def_readwrite("LCC_open_martingale_normalised", &GW::PyGWSimulation::LCC_open_martingale_normalised)
        .def_readwrite("sigma_LCC", &GW::PyGWSimulation::sigma_LCC);
    
    py::class_<GW::PyInitGWState>(m, "GWInitialState")
        .def(py::init<int>())
        .def_readwrite("V", &GW::PyInitGWState::V)
        .def_readwrite("Nai", &GW::PyInitGWState::Nai)
        .def_readwrite("Ki", &GW::PyInitGWState::Ki)
        .def_readwrite("Cai", &GW::PyInitGWState::Cai)
        .def_readwrite("CaNSR", &GW::PyInitGWState::CaNSR)
        .def_readwrite("CaLTRPN", &GW::PyInitGWState::CaLTRPN)
        .def_readwrite("CaHTRPN", &GW::PyInitGWState::CaHTRPN)
        .def_readwrite("m", &GW::PyInitGWState::m)
        .def_readwrite("h", &GW::PyInitGWState::h)
        .def_readwrite("j", &GW::PyInitGWState::j)
        .def_readwrite("xKs", &GW::PyInitGWState::xKs)
        .def_readwrite("XKr", &GW::PyInitGWState::XKr)
        .def_readwrite("XKv14", &GW::PyInitGWState::XKv14)
        .def_readwrite("XKv43", &GW::PyInitGWState::XKv43) 
        .def_readwrite("CaSS", &GW::PyInitGWState::CaSS)
        .def_readwrite("CaJSR", &GW::PyInitGWState::CaJSR)
        .def_readwrite("LCC", &GW::PyInitGWState::LCC)
        .def_readwrite("LCC_inactivation", &GW::PyInitGWState::LCC_inactivation)
        .def_readwrite("RyR", &GW::PyInitGWState::RyR)
        .def_readwrite("ClCh", &GW::PyInitGWState::ClCh);

    py::class_<GW::GW_model>(m, "GWModel")
        .def(py::init<int>())
        .def(py::init<GW::Parameters,int>())
        .def("init_state", &GW::GW_model::init_from_python, "inital_state"_a)
        .def("run", &GW::GW_model::run_sim<XoshiroCpp::Xoshiro256PlusPlus>, "dt"_a, "num_steps"_a, "Is"_a, "record_every"_a, py::call_guard<py::gil_scoped_release>());
}