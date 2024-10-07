#include "SSA.hpp"


namespace GW {

    template <typename FloatType>
    void CRUStateThread<FloatType>::copy_from_CRUState(const CRUState<FloatType> &state, const Array2<FloatType> &JLCC, const int idx, const Parameters<FloatType> &params){
        for (int j = 0; j < 4; j++){
            for (int k = 0; k < 6; k++)
                RyR[6*j+k] = state.RyR.array(idx,j,k);

            LCC[j] = state.LCC(idx,j);
            LCC_inactivation[j] = state.LCC_inactivation(idx,j);
            ClCh[j] = state.ClCh(idx,j);
            open_RyR[j] = (FloatType)(state.RyR.array(idx,j,2) + state.RyR.array(idx,j,3));

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
        memset(LCC_rates, 0, 3*4*sizeof(FloatType));
        memset(LCC_inactivation_rates, 0, 4*sizeof(FloatType));
        memset(RyR_rates, 0, 12*4*sizeof(FloatType));
        memset(ClCh_rates, 0, 4*sizeof(FloatType));
        memset(subunit_rates, 0, 4*sizeof(FloatType));
        memset(Jxfer, 0, 4*sizeof(FloatType));
        memset(Jiss, 0, 4*sizeof(FloatType));
        CaJSR = state.CaJSR(idx);
        Jtr = 0;
    }


    template <typename FloatType>
    void update_fluxes(CRUStateThread<FloatType> &state, const FloatType Cai, const FloatType CaNSR, const Parameters<FloatType> &params){
        state.Jtr = params.rtr * (CaNSR - state.CaJSR);
        state.Jiss[0] = params.riss * (state.CaSS[1] + state.CaSS[3] - 2*state.CaSS[0]);
        state.Jiss[1] = params.riss * (state.CaSS[2] + state.CaSS[0] - 2*state.CaSS[1]);
        state.Jiss[2] = params.riss * (state.CaSS[3] + state.CaSS[1] - 2*state.CaSS[2]);
        state.Jiss[3] = params.riss * (state.CaSS[0] + state.CaSS[2] - 2*state.CaSS[3]);
        for (int j = 0; j < 4; j++)
            state.Jxfer[j] = params.rxfer * (state.CaSS[j] - Cai); 
    }

    template <typename FloatType>
    void update_rates(CRUStateThread<FloatType> &state, const Parameters<FloatType> &params, const Constants<FloatType> &consts){
        for (int j = 0; j < 4; j++){
            state.subunit_rates[j] = 0;
            update_LCC_rates(state.LCC_rates, state.subunit_rates, state.LCC, state.CaSS, j, params, consts);
            update_LCC_inactivation_rates(state.LCC_inactivation_rates, state.subunit_rates, state.LCC_inactivation, consts.yinfLCC, consts.tauLCC, j);
            update_RyR_rates(state.RyR_rates, state.subunit_rates, state.RyR, state.CaSS, j, params); 
            update_ClCh_rates(state.ClCh_rates, state.subunit_rates, state.ClCh, state.CaSS, params.kfClCh, params.kbClCh, j);
        }
    }

    template <typename FloatType, typename Generator>
    void sample_RyR56(CRUStateThread<FloatType> &state, const int idx, const Parameters<FloatType> &params){
        int n56 = state.RyR[4+6*idx] + state.RyR[5+6*idx];
        if (n56 > 0){
            FloatType p = params.k65 / (params.k65 + params.k56 * square(state.CaSS[idx]));
            state.RyR[4+6*idx] = sample_binomial<FloatType, Generator>(p, n56);
            state.RyR[5+6*idx] = n56 - state.RyR[4+6*idx];
        }
        else {
            state.RyR[4+6*idx] = 0;
            state.RyR[5+6*idx] = 0;
        }
    }
    
    template <typename FloatType, typename Generator>
    void sample_RyR34(CRUStateThread<FloatType> &state, const int idx, const Parameters<FloatType> &params){
        int n34 = state.RyR[2+6*idx] + state.RyR[3+6*idx];
        if (n34 > 0){
            FloatType p = params.k43 / (params.k43 + params.k34 * square(state.CaSS[idx]));
            state.RyR[2+6*idx] = sample_binomial<FloatType, Generator>(p, n34);
            state.RyR[3+6*idx] = n34 - state.RyR[2+6*idx];
        }
        else {
            state.RyR[2+6*idx] = 0;
            state.RyR[3+6*idx] = 0;
        }
    }

    template <typename FloatType, typename Generator>
    void update_CaSS(CRUStateThread<FloatType> &state, const FloatType dt, const Parameters<FloatType> &params, const Constants<FloatType> &consts){
        FloatType betaSS, dCaSS, CaSS_tmp;
        for (int j = 0; j < 4; j++){
            betaSS = 1.0 / (1 + (consts.BSR_const / square(params.KBSR + state.CaSS[j])) + (consts.BSL_const / square(params.KBSL + state.CaSS[j])));
            dCaSS = dt * (state.JLCC[j] + state.Jrel[j] - state.Jxfer[j] + state.Jiss[j]) * betaSS;
            CaSS_tmp = state.CaSS[j] + dCaSS;
            if (state.CaSS[j] > 1.15e-4 && CaSS_tmp <= 1.15e-4)
                sample_RyR56<FloatType, Generator>(state, j, params);
            else if (state.CaSS[j] > 0.03685 && CaSS_tmp <= 0.03685)
                sample_RyR34<FloatType, Generator>(state, j, params);
            state.CaSS[j] = CaSS_tmp;
        }
    }


    template <typename FloatType, typename Generator>
    void sample_LCC(CRUStateThread<FloatType> &state, const FloatType sum_LCC_rates, const int subunit_idx, const Constants<FloatType> &consts){
        const int transition = sample_weights<FloatType, int, Generator>(state.LCC_rates + 3*subunit_idx, sum_LCC_rates, 3); // using pointer arithmetic here
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

    template <typename FloatType, typename Generator>
    inline void sample_RyR(CRUStateThread<FloatType> &state, const FloatType sum_RyR_rates, const int subunit_idx, const Parameters<FloatType> &params){
        const int transition = sample_weights<FloatType, int, Generator>(state.RyR_rates + 12*subunit_idx, sum_RyR_rates, 12); // using pointer arithmetic here
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

    template <typename FloatType, typename Generator>
    void sample_new_state(CRUStateThread<FloatType> &state, const int subunit_idx, const Parameters<FloatType> &params, const Constants<FloatType> &consts){
        FloatType subunit_rate = state.subunit_rates[subunit_idx];
        FloatType sum_LCC_rates = state.LCC_rates[3*subunit_idx] + state.LCC_rates[3*subunit_idx+1] + state.LCC_rates[3*subunit_idx+2];
        FloatType LCC_inactivation_rate = state.LCC_inactivation_rates[subunit_idx];
        FloatType ClCh_rate = state.ClCh_rates[subunit_idx];
        FloatType sum_RyR_rates = subunit_rate - (sum_LCC_rates + LCC_inactivation_rate + ClCh_rate);

        FloatType u = urand<FloatType, Generator>() * subunit_rate;
        if (u < sum_LCC_rates)
            sample_LCC<FloatType, Generator>(state, sum_LCC_rates, subunit_idx, consts);
        else if (u < (sum_LCC_rates + LCC_inactivation_rate)) {
            state.LCC_inactivation[subunit_idx] = 1 - state.LCC_inactivation[subunit_idx];
            if ((state.LCC[subunit_idx] == 6) || (state.LCC[subunit_idx] == 12)){
                state.JLCC[subunit_idx] = (state.LCC_inactivation[subunit_idx] == 1) ? consts.JLCC_multiplier * (consts.Cao_scaled - consts.JLCC_exp * state.CaSS[subunit_idx]) : 0.0;
                state.LCC_open_increment[subunit_idx] = (state.LCC_inactivation[subunit_idx] == 1) ? 1.0 : -1.0;
            }
            
        } 
        else if (u < (sum_LCC_rates + LCC_inactivation_rate + sum_RyR_rates))
            sample_RyR<FloatType, Generator>(state, sum_RyR_rates, subunit_idx, params);
        else 
            state.ClCh[subunit_idx] = 1 - state.ClCh[subunit_idx];
    }

    template <typename FloatType, typename Generator>
    void SSA_single_CRU(CRUStateThread<FloatType> &state, const FloatType Cai, const FloatType CaNSR, const FloatType dt, const Parameters<FloatType> &params, const Constants<FloatType> &consts){
        int subunit_idx;
        FloatType t = 0, jump_t = 0, total_rate;

        while (1){
            update_fluxes(state, Cai, CaNSR, params);
            update_rates(state, params, consts);

            total_rate = state.subunit_rates[0] + state.subunit_rates[1] + state.subunit_rates[2] + state.subunit_rates[3];
            jump_t = -log(urand<FloatType, Generator>()) / total_rate;

            for (int i = 0; i < 4; ++i) { 
                state.RyR_open_increment[i] = 0; 
                state.LCC_open_increment[i] = 0; 
            }
            if (t + jump_t < dt){
                update_integral_increment<FloatType>(state, jump_t, params);
                update_CaSS<FloatType, Generator>(state, jump_t, params, consts);
                update_CaJSR(state, jump_t, params, consts);
            } 
            else {
                update_integral_increment<FloatType>(state, dt-t, params);
                update_CaSS<FloatType, Generator>(state, dt-t, params, consts);
                update_CaJSR(state, dt-t, params, consts);
                update_martingale<FloatType, Generator>(state, dt-t);
                break;
            }
            t += jump_t;
            
            subunit_idx = sample_weights<FloatType, int, Generator>(state.subunit_rates, total_rate, 4);
            sample_new_state<FloatType, Generator>(state, subunit_idx, params, consts);
            update_martingale<FloatType, Generator>(state, jump_t);
        }
    }

    template <typename FloatType>
    void update_integral_increment(CRUStateThread<FloatType> &state, const FloatType dt, const Parameters<FloatType> &params){
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
    
    template <typename FloatType, typename PRNG>
    void update_martingale(CRUStateThread<FloatType> &state, const FloatType dt){
        FloatType sigma2_RyR = 0.0;
        FloatType sigma2_LCC = 0.0;
        for (int j = 0; j < 4; ++j){
            state.RyR_open_martingale[j] += (state.RyR_open_increment[j] - state.RyR_open_int_increment[j]);
            state.LCC_open_martingale[j] += (state.LCC_open_increment[j] - state.LCC_open_int_increment[j]);

            sigma2_RyR += (state.RyR_rates[12*j+1] + state.RyR_rates[12*j+10] + (state.RyR_rates[12*j+4] + state.RyR_rates[12*j+7]));
            sigma2_LCC += abs(state.LCC_open_int_increment[j]) / dt;
            //if (sigma2 > 0){
            //    state.RyR_open_martingale_normalised[j] += (state.RyR_open_increment[j] - state.RyR_open_int_increment[j]) / sqrt(sigma2);
            //} else {
            //    state.RyR_open_martingale_normalised[j] += sqrt(dt) * nrand<FloatType, PRNG>();
            //}
        }
        if (sigma2_RyR > 0){
            state.sigma_RyR = sqrt(sigma2_RyR);
            for (int j = 0; j < 4; ++j)
                state.RyR_open_martingale_normalised[j] += (state.RyR_open_increment[j] - state.RyR_open_int_increment[j]) / state.sigma_RyR;
        } 
        else {
            state.sigma_RyR = 0.0;
            for (int j = 0; j < 4; ++j)
                state.RyR_open_martingale_normalised[j] += sqrt(dt) * nrand<FloatType, PRNG>();
        }
        
        if (sigma2_LCC > 0){
            state.sigma_LCC = sqrt(sigma2_LCC);
            for (int j = 0; j < 4; ++j)
                state.LCC_open_martingale_normalised[j] += (state.LCC_open_increment[j] - state.LCC_open_int_increment[j]) / state.sigma_LCC;
        } 
        else {
            state.sigma_LCC = 0.0;
            for (int j = 0; j < 4; ++j)
                state.LCC_open_martingale_normalised[j] += sqrt(dt) * nrand<FloatType, PRNG>();
        }
    }
}