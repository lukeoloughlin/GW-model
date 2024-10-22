#pragma once

#include "GW_utils.hpp"
#include "common.hpp"
//#include "ndarray.hpp"

#include <Eigen/Core>

template<typename T>
using Array2 = Eigen::Array<T,Eigen::Dynamic,4,Eigen::RowMajor>;

namespace GW {

    //template <typename T>
    struct CRUStateThread {
        int LCC[4];
        int LCC_inactivation[4];
        int RyR[4*6]; // Have to count states here so we include all 6 possible states
        double open_RyR[4]; // Keep track of number open RyR
        int ClCh[4];

        double CaSS[4];
        double CaJSR;
        double JLCC[4];
        double Jrel[4];
        double Jxfer[4];
        double Jiss[4];
        double Jtr;

        double LCC_rates[3*4]; // 3 rates to track at most
        double LCC_inactivation_rates[4];
        double RyR_rates[12*4];
        double ClCh_rates[4];
        double subunit_rates[4];

        double RyR_open_increment[4];
        double RyR_open_int_increment[4];
        double RyR_open_int[4];
        double RyR_open_martingale[4];
        double RyR_open_martingale_normalised[4];
        double sigma_RyR;
        
        double LCC_open_increment[4];
        double LCC_open_int_increment[4];
        double LCC_open_int[4];
        double LCC_open_martingale[4];
        double LCC_open_martingale_normalised[4];
        double sigma_LCC;



        void copy_from_CRUState(const CRUState& state, const Array2<double>& JLCC, const int idx, const Parameters& params){
            for (int j = 0; j < 4; j++){
                for (int k = 0; k < 6; k++)
                    RyR[6*j+k] = state.RyR.array(idx,j,k);

                LCC[j] = state.LCC(idx,j);
                LCC_inactivation[j] = state.LCC_inactivation(idx,j);
                ClCh[j] = state.ClCh(idx,j);
                open_RyR[j] = (double)(state.RyR.array(idx,j,2) + state.RyR.array(idx,j,3));

                CaSS[j] = state.CaSS(idx,j);
                this->JLCC[j] = JLCC(idx,j); // This doesn't update at every iteration of loop so have to set it here
                Jrel[j] = params.rRyR * open_RyR[j] * (state.CaJSR(idx) - CaSS[j]);

                RyR_open_increment[j] = 0.0;
                RyR_open_int_increment[j] = 0.0;
                RyR_open_int[j] = state.RyR_open_int(idx,j);
                RyR_open_martingale[j] = state.RyR_open_martingale(idx,j);
                RyR_open_martingale_normalised[j] = state.RyR_open_martingale_normalised(idx,j);
                
                LCC_open_increment[j] = 0.0;
                LCC_open_int_increment[j] = 0.0;
                LCC_open_int[j] = state.LCC_open_int(idx,j);
                LCC_open_martingale[j] = state.LCC_open_martingale(idx,j);
                LCC_open_martingale_normalised[j] = state.LCC_open_martingale_normalised(idx,j);
            }
            memset(LCC_rates, 0, 3*4*sizeof(double));
            memset(LCC_inactivation_rates, 0, 4*sizeof(double));
            memset(RyR_rates, 0, 12*4*sizeof(double));
            memset(ClCh_rates, 0, 4*sizeof(double));
            memset(subunit_rates, 0, 4*sizeof(double));
            memset(Jxfer, 0, 4*sizeof(double));
            memset(Jiss, 0, 4*sizeof(double));
            CaJSR = state.CaJSR(idx);
            Jtr = 0;
        }
    };


    /* Update the flux values (except JLCC and Jrel) of the CRUStateThread object */
    //template <typename T>
    inline void update_fluxes(CRUStateThread& state, const double Cai, const double CaNSR, const Parameters& params){
        state.Jtr = params.rtr * (CaNSR - state.CaJSR);
        state.Jiss[0] = params.riss * (state.CaSS[1] + state.CaSS[3] - 2*state.CaSS[0]);
        state.Jiss[1] = params.riss * (state.CaSS[2] + state.CaSS[0] - 2*state.CaSS[1]);
        state.Jiss[2] = params.riss * (state.CaSS[3] + state.CaSS[1] - 2*state.CaSS[2]);
        state.Jiss[3] = params.riss * (state.CaSS[0] + state.CaSS[2] - 2*state.CaSS[3]);
        for (int j = 0; j < 4; j++)
            state.Jxfer[j] = params.rxfer * (state.CaSS[j] - Cai); 
    }

    /* Update the transition rates of the CRUStateThread object */
    //template <typename T>
    inline void update_rates(CRUStateThread& state, const Parameters& params, const Constants& consts){
        for (int j = 0; j < 4; j++){
            state.subunit_rates[j] = 0;
            update_LCC_rates(state.LCC_rates, state.subunit_rates, state.LCC, state.CaSS, j, params, consts);
            update_LCC_inactivation_rates(state.LCC_inactivation_rates, state.subunit_rates, state.LCC_inactivation, consts.yinfLCC, consts.tauLCC, j);
            update_RyR_rates(state.RyR_rates, state.subunit_rates, state.RyR, state.CaSS, j, params); 
            update_ClCh_rates(state.ClCh_rates, state.subunit_rates, state.ClCh, state.CaSS, params.kfClCh, params.kbClCh, j);
        }
    }
    
    //template <typename T>
    inline void update_integral_increment(CRUStateThread& state, const double dt, const Parameters& params){
        for (int j = 0; j < 4; ++j){
            state.RyR_open_int_increment[j] = dt * (state.RyR_rates[12*j+1] + state.RyR_rates[12*j+10] - (state.RyR_rates[12*j+4] + state.RyR_rates[12*j+7]));
            if ((state.LCC[j] == 5) && (state.LCC_inactivation[j] == 1)) {
                state.LCC_open_int_increment[j] = dt * params.f;
            }
            else if ((state.LCC[j] == 11) && (state.LCC_inactivation[j] == 1)) {
                state.LCC_open_int_increment[j] = dt * params.f1;
            }
            else if ((state.LCC[j] == 6) && (state.LCC_inactivation[j] == 0)) {
                state.LCC_open_int_increment[j] = dt * state.LCC_inactivation_rates[j];
            }
            else if ((state.LCC[j] == 12) && (state.LCC_inactivation[j] == 0)) {
                state.LCC_open_int_increment[j] = dt * state.LCC_inactivation_rates[j];
            }
            else if ((state.LCC[j] == 6) && (state.LCC_inactivation[j] == 1)) {
                state.LCC_open_int_increment[j] = -dt * (state.LCC_inactivation_rates[j] + params.g);
            }
            else if ((state.LCC[j] == 12) && (state.LCC_inactivation[j] == 1)) {
                state.LCC_open_int_increment[j] = -dt * (state.LCC_inactivation_rates[j] + params.g1);
            }
            else {
                state.LCC_open_int_increment[j] = 0.0;
            } 

            state.RyR_open_int[j] += state.RyR_open_int_increment[j];
            state.LCC_open_int[j] += state.LCC_open_int_increment[j];
        }
    }


    /* Do an Euler step on CaSS for a single CRU */
    template <typename PRNG>
    inline void update_CaSS(CRUStateThread& state, const double dt, const Parameters& params, const Constants& consts){
        double betaSS, dCaSS, CaSS_tmp;
        for (int j = 0; j < 4; j++){
            betaSS = 1.0 / (1 + (consts.BSR_const / square(params.KBSR + state.CaSS[j])) + (consts.BSL_const / square(params.KBSL + state.CaSS[j])));
            dCaSS = dt * (state.JLCC[j] + state.Jrel[j] - state.Jxfer[j] + state.Jiss[j]) * betaSS;
            CaSS_tmp = state.CaSS[j] + dCaSS;
            if (state.CaSS[j] > 1.15e-4 && CaSS_tmp <= 1.15e-4)
                sample_RyR56<double, PRNG>(state, j, params);
            else if (state.CaSS[j] > 0.03685 && CaSS_tmp <= 0.03685)
                sample_RyR34<double, PRNG>(state, j, params);
            state.CaSS[j] = CaSS_tmp;
        }
    }

    /* Do an Euler step on CaJSR for a single CRU */
    //template <typename T>
    inline void update_CaJSR(CRUStateThread& state, const double dt, const Parameters& params, const Constants& consts){
        const double betaJSR = 1.0 / (1 + consts.CSQN_const / square(params.KCSQN + state.CaJSR));
        state.CaJSR += (dt * betaJSR * (state.Jtr - consts.VSS_VJSR * (state.Jrel[0] + state.Jrel[1] + state.Jrel[2] + state.Jrel[3])));
    }
    /* Sample values of state 5 and state 6 of the RyRs if CaSS crosses appropriate threshold. Above this threshold the two states are 
    treated as equivalent so we do not track them directly. */
    template <typename PRNG>
    inline void sample_RyR56(CRUStateThread& state, const int idx, const Parameters& params){
        int n56 = state.RyR[4+6*idx] + state.RyR[5+6*idx];
        if (n56 > 0){
            double p = params.k65 / (params.k65 + params.k56 * square(state.CaSS[idx]));
            state.RyR[4+6*idx] = sample_binomial<double, PRNG>(p, n56);
            state.RyR[5+6*idx] = n56 - state.RyR[4+6*idx];
        }
        else {
            state.RyR[4+6*idx] = 0;
            state.RyR[5+6*idx] = 0;
        }
    }
    /* Sample values of state 3 and state 4 of the RyRs if CaSS crosses appropriate threshold. Above this threshold the two states are 
    treated as equivalent so we do not track them directly. */
    template <typename PRNG>
    inline void sample_RyR34(CRUStateThread& state, const int idx, const Parameters& params){
        int n34 = state.RyR[2+6*idx] + state.RyR[3+6*idx];
        if (n34 > 0){
            double p = params.k43 / (params.k43 + params.k34 * square(state.CaSS[idx]));
            state.RyR[2+6*idx] = sample_binomial<double, PRNG>(p, n34);
            state.RyR[3+6*idx] = n34 - state.RyR[2+6*idx];
        }
        else {
            state.RyR[2+6*idx] = 0;
            state.RyR[3+6*idx] = 0;
        }
    }


    //template <typename double, typename PRNG>
    template <typename PRNG>
    inline void update_martingale(CRUStateThread& state, const double dt){
        double sigma2_RyR = 0.0;
        double sigma2_LCC = 0.0;
        for (int j = 0; j < 4; ++j){
            state.RyR_open_martingale[j] += (state.RyR_open_increment[j] - state.RyR_open_int_increment[j]);
            state.LCC_open_martingale[j] += (state.LCC_open_increment[j] - state.LCC_open_int_increment[j]);

            sigma2_RyR += (state.RyR_rates[12*j+1] + state.RyR_rates[12*j+10] + (state.RyR_rates[12*j+4] + state.RyR_rates[12*j+7]));
            sigma2_LCC += abs(state.LCC_open_int_increment[j]) / dt;
        }
        if (sigma2_RyR > 0){
            state.sigma_RyR = sqrt(sigma2_RyR);
            for (int j = 0; j < 4; ++j)
                state.RyR_open_martingale_normalised[j] += (state.RyR_open_increment[j] - state.RyR_open_int_increment[j]) / state.sigma_RyR;
        } 
        else {
            state.sigma_RyR = 0.0;
            for (int j = 0; j < 4; ++j)
                state.RyR_open_martingale_normalised[j] += sqrt(dt) * nrand<double, PRNG>();
        }
        
        if (sigma2_LCC > 0){
            state.sigma_LCC = sqrt(sigma2_LCC);
            for (int j = 0; j < 4; ++j)
                state.LCC_open_martingale_normalised[j] += (state.LCC_open_increment[j] - state.LCC_open_int_increment[j]) / state.sigma_LCC;
        } 
        else {
            state.sigma_LCC = 0.0;
            for (int j = 0; j < 4; ++j)
                state.LCC_open_martingale_normalised[j] += sqrt(dt) * nrand<double, PRNG>();
        }
    }
    

    /* Sample a new LCC state for the subunit given by subunit_idx. Also updates JLCC when appropriate. */
    template <typename PRNG>
    inline void sample_LCC(CRUStateThread& state, const double sum_LCC_rates, const int subunit_idx, const Constants& consts){
        const int transition = sample_weights<double, int, PRNG>(state.LCC_rates + 3*subunit_idx, sum_LCC_rates, 3); // using pointer arithmetic here
        switch (state.LCC[subunit_idx]){
        case 1:
            if (transition == 0)
                state.LCC[subunit_idx] = 2;
            else 
                state.LCC[subunit_idx] = 7;
            break;
        case 2:
            if (transition == 0)
                state.LCC[subunit_idx] = 1;
            else if (transition == 1)
                state.LCC[subunit_idx] = 3;
            else 
                state.LCC[subunit_idx] = 8;
            break;
        case 3:
            if (transition == 0)
                state.LCC[subunit_idx] = 2;
            else if (transition == 1)
                state.LCC[subunit_idx] = 4;
            else 
                state.LCC[subunit_idx] = 9;
            break;
        case 4:
            if (transition == 0)
                state.LCC[subunit_idx] = 3;
            else if (transition == 1)
                state.LCC[subunit_idx] = 5;
            else 
                state.LCC[subunit_idx] = 10;
            break;
        case 5:
            if (transition == 0)
                state.LCC[subunit_idx] = 4; 
            else if (transition == 1){ 
                state.LCC[subunit_idx] = 6; 
                state.JLCC[subunit_idx] = (state.LCC_inactivation[subunit_idx] == 1) ? consts.JLCC_multiplier * (consts.Cao_scaled - consts.JLCC_exp * state.CaSS[subunit_idx]) : 0; 
                state.LCC_open_increment[subunit_idx] = (state.LCC_inactivation[subunit_idx] == 1) ? 1.0 : 0.0;
            }
            else 
                state.LCC[subunit_idx] = 11;
            break;
        case 6:
            state.LCC[subunit_idx] = 5;
            state.JLCC[subunit_idx] = 0;
            state.LCC_open_increment[subunit_idx] = (state.LCC_inactivation[subunit_idx] == 1) ? -1.0 : 0.0;
            break;
        case 7:
            if (transition == 0)
                state.LCC[subunit_idx] = 1;
            else 
                state.LCC[subunit_idx] = 8;
            break;
        case 8:
            if (transition == 0)
                state.LCC[subunit_idx] = 2;
            else if (transition == 1)
                state.LCC[subunit_idx] = 7;
            else 
                state.LCC[subunit_idx] = 9;
            break;
        case 9:
            if (transition == 0)
                state.LCC[subunit_idx] = 3;
            else if (transition == 1)
                state.LCC[subunit_idx] = 8;
            else 
                state.LCC[subunit_idx] = 10;
            break;
        case 10:
            if (transition == 0)
                state.LCC[subunit_idx] = 4;
            else if (transition == 1)
                state.LCC[subunit_idx] = 9;
            else 
                state.LCC[subunit_idx] = 11;
            break;
        case 11:
            if (transition == 0)
                state.LCC[subunit_idx] = 5; 
            else if (transition == 1)
                state.LCC[subunit_idx] = 10; 
            else {
                state.LCC[subunit_idx] = 12; 
                state.JLCC[subunit_idx] = (state.LCC_inactivation[subunit_idx] == 1) ? consts.JLCC_multiplier * (consts.Cao_scaled - consts.JLCC_exp * state.CaSS[subunit_idx]) : 0.0;
                state.LCC_open_increment[subunit_idx] = (state.LCC_inactivation[subunit_idx] == 1) ? 1.0 : 0.0;
            }
            break;
        case 12:
            state.LCC[subunit_idx] = 11;
            state.JLCC[subunit_idx] = 0.0;
            state.LCC_open_increment[subunit_idx] = (state.LCC_inactivation[subunit_idx] == 1) ? -1.0 : 0.0;
            break;    
        default:
            break;
        }
    }

    /* Sample a new RyR state for the subunit given by subunit_idx. Also updates Jrel. */
    template <typename PRNG>
    inline void sample_RyR(CRUStateThread& state, const double sum_RyR_rates, const int subunit_idx, const Parameters& params){
        const int transition = sample_weights<double, int, PRNG>(state.RyR_rates + 12*subunit_idx, sum_RyR_rates, 12); // using pointer arithmetic here
        switch (transition){
        case 0: // 1 -> 2
            --state.RyR[6*subunit_idx];
            ++state.RyR[6*subunit_idx+1];   
            break;
        case 1: // 2 -> 3
            --state.RyR[6*subunit_idx+1];
            ++state.RyR[6*subunit_idx+2];   
            ++state.open_RyR[subunit_idx];
            state.RyR_open_increment[subunit_idx] = 1.0;
            break;
        case 2: // 2 -> 5
            --state.RyR[6*subunit_idx+1];
            ++state.RyR[6*subunit_idx+4];   
            break;
        case 3: // 3 -> 4
            --state.RyR[6*subunit_idx+2];
            ++state.RyR[6*subunit_idx+3];   
            break;
        case 4: // 4 -> 5
            --state.RyR[6*subunit_idx+3];
            ++state.RyR[6*subunit_idx+4];
            --state.open_RyR[subunit_idx];
            state.RyR_open_increment[subunit_idx] = -1.0;
            break;
        case 5: // 5 -> 6
            --state.RyR[6*subunit_idx+4];
            ++state.RyR[6*subunit_idx+5];   
            break;
        case 6: // 2 -> 1
            --state.RyR[6*subunit_idx+1];
            ++state.RyR[6*subunit_idx];   
            break;
        case 7: // 3 -> 2
            --state.RyR[6*subunit_idx+2];
            ++state.RyR[6*subunit_idx+1];
            --state.open_RyR[subunit_idx]; 
            state.RyR_open_increment[subunit_idx] = -1.0;
            break;
        case 8: // 4 -> 3
            --state.RyR[6*subunit_idx+3];
            ++state.RyR[6*subunit_idx+2];   
            break;
        case 9: // 5 -> 2
            --state.RyR[6*subunit_idx+4];
            ++state.RyR[6*subunit_idx+1];   
            break;
        case 10: // 5 -> 4
            --state.RyR[6*subunit_idx+4];
            ++state.RyR[6*subunit_idx+3];
            ++state.open_RyR[subunit_idx];
            state.RyR_open_increment[subunit_idx] = 1.0;
            break;
        case 11: // 6 -> 5
            --state.RyR[6*subunit_idx+5];
            ++state.RyR[6*subunit_idx+4];   
            break;    
        default:
            break;
        }
        state.Jrel[subunit_idx] = params.rRyR * state.open_RyR[subunit_idx] * (state.CaJSR - state.CaSS[subunit_idx]);
    }
    
    /* sample a new state based on the rates of the LCCs, RyRs and ClChs */
    template <typename PRNG>
    inline void sample_new_state(CRUStateThread& state, const int subunit_idx, const Parameters& params, const Constants& consts){
        double subunit_rate = state.subunit_rates[subunit_idx];
        double sum_LCC_rates = state.LCC_rates[3*subunit_idx] + state.LCC_rates[3*subunit_idx+1] + state.LCC_rates[3*subunit_idx+2];
        double LCC_inactivation_rate = state.LCC_inactivation_rates[subunit_idx];
        double ClCh_rate = state.ClCh_rates[subunit_idx];
        double sum_RyR_rates = subunit_rate - (sum_LCC_rates + LCC_inactivation_rate + ClCh_rate);

        double u = urand<double, PRNG>() * subunit_rate;
        if (u < sum_LCC_rates)
            sample_LCC<double, PRNG>(state, sum_LCC_rates, subunit_idx, consts);
        else if (u < (sum_LCC_rates + LCC_inactivation_rate)) {
            state.LCC_inactivation[subunit_idx] = 1 - state.LCC_inactivation[subunit_idx];
            if ((state.LCC[subunit_idx] == 6) || (state.LCC[subunit_idx] == 12)){
                state.JLCC[subunit_idx] = (state.LCC_inactivation[subunit_idx] == 1) ? consts.JLCC_multiplier * (consts.Cao_scaled - consts.JLCC_exp * state.CaSS[subunit_idx]) : 0.0;
                state.LCC_open_increment[subunit_idx] = (state.LCC_inactivation[subunit_idx] == 1) ? 1.0 : -1.0;
            }
            
        } 
        else if (u < (sum_LCC_rates + LCC_inactivation_rate + sum_RyR_rates))
            sample_RyR<double, PRNG>(state, sum_RyR_rates, subunit_idx, params);
        else 
            state.ClCh[subunit_idx] = 1 - state.ClCh[subunit_idx];
    }
    
    /* Perform the SSA on a single CRU (with fixed global variables) over an interval of length dt */
    template <typename PRNG>
    inline void SSA_single_CRU(CRUStateThread& state, const double Cai, const double CaNSR, const double sim_time, const Parameters& params, const Constants& consts){
        int subunit_idx;
        double t = 0, dt = 0, total_rate;

        while (1){
            update_fluxes(state, Cai, CaNSR, params);
            update_rates(state, params, consts);

            total_rate = state.subunit_rates[0] + state.subunit_rates[1] + state.subunit_rates[2] + state.subunit_rates[3];
            dt = -log(urand<double, PRNG>()) / total_rate;

            for (int i = 0; i < 4; ++i) { 
                state.RyR_open_increment[i] = 0; 
                state.LCC_open_increment[i] = 0; 
            }
            if (t + dt < sim_time){
                update_integral_increment<double>(state, dt, params);
                update_CaSS<PRNG>(state, dt, params, consts);
                update_CaJSR(state, dt, params, consts);
            } 
            else {
                update_integral_increment(state, sim_time-t, params);
                update_CaSS<PRNG>(state, sim_time-t, params, consts);
                update_CaJSR(state, sim_time-t, params, consts);
                update_martingale<PRNG>(state, sim_time-t);
                break;
            }
            t += jump_t;
            
            subunit_idx = sample_weights<double, int, PRNG>(state.subunit_rates, total_rate, 4);
            sample_new_state<PRNG>(state, subunit_idx, params, consts);
            update_martingale<PRNG>(state, dt);
        }
    }

}

