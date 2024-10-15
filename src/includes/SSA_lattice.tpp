#include "SSA_lattice.hpp"


namespace GW {

    template <typename FloatType>
    void CRULatticeStateThread<FloatType>::copy_from_CRULatticeState(const CRULatticeState<FloatType> &state, const int x, const int y, const Parameters<FloatType> &params){
        for (int j = 0; j < 6; ++j)
            RyR[j] = state.RyR.array(x,y,j);

        LCC = state.LCC(x,y);
        LCC_inactivation = state.LCC_inactivation(x,y);
        ClCh = state.ClCh(x,y);

        CaSS = state.CaSS(x,y);
        CaJSR = state.CaJSR(x,y);

        // Only need to set these at the end
        this->JLCC = 0.0; 
        Jrel = 0.0;
        Jxfer = 0.0;
        Jtr = 0.0;
    }
    
    template <typename FloatType>
    void init_rates(CRULatticeStateThread<FloatType> &state, const Parameters<FloatType> &params, const Constants<FloatType> &consts){
        switch (state.LCC){
        case 1:
            state.LCC_rates[0] = 4*consts.alphaLCC;
            state.LCC_rates[1] = consts.gamma0*state.CaSS;
            state.LCC_rates[2] = 0;
            break;
        case 2:
            state.LCC_rates[0] = consts.betaLCC;
            state.LCC_rates[1] = 3*consts.alphaLCC;
            state.LCC_rates[2] = consts.gamma0a*state.CaSS;
            break;
        case 3:
            state.LCC_rates[0] = 2*consts.betaLCC;
            state.LCC_rates[1] = 2*consts.alphaLCC;
            state.LCC_rates[2] = consts.gamma0a2*state.CaSS;
            break;
        case 4:
            state.LCC_rates[0] = 3*consts.betaLCC;
            state.LCC_rates[1] = consts.alphaLCC;
            state.LCC_rates[2] = consts.gamma0a3*state.CaSS;
            break;
        case 5:
            state.LCC_rates[0] = 4*consts.betaLCC;
            state.LCC_rates[1] = params.f;
            state.LCC_rates[2] = consts.gamma0a4*state.CaSS;
            break;
        case 6:
            state.LCC_rates[0] = params.g;
            state.LCC_rates[1] = 0;
            state.LCC_rates[2] = 0;
            break;
        case 7:
            state.LCC_rates[0] = consts.omega;
            state.LCC_rates[1] = 4*consts.alphaLCC*consts.a;
            state.LCC_rates[2] = 0;
            break;
        case 8:
            state.LCC_rates[0] = consts.omega_b;
            state.LCC_rates[1] = consts.betaLCC*consts.binv;
            state.LCC_rates[2] = 3*consts.alphaLCC*consts.a;
            break;
        case 9:
            state.LCC_rates[0] = consts.omega_b2;
            state.LCC_rates[1] = 2*consts.betaLCC*consts.binv;
            state.LCC_rates[2] = 2*consts.alphaLCC*consts.a;
            break;
        case 10:
            state.LCC_rates[0] = consts.omega_b3;
            state.LCC_rates[1] = 3*consts.betaLCC*consts.binv;
            state.LCC_rates[2] = consts.alphaLCC*consts.a;
            break;
        case 11:
            state.LCC_rates[0] = consts.omega_b4;
            state.LCC_rates[1] = 4*consts.betaLCC*consts.binv;
            state.LCC_rates[2] = params.f1;
            break;
        case 12:
            state.LCC_rates[0] = params.g1;
            state.LCC_rates[1] = 0;
            state.LCC_rates[2] = 0;
            break;
        default:
            break;
        }    
        state.LCC_tot_rate = state.LCC_rates[0] + state.LCC_rates[1] + state.LCC_rates[2];

        const FloatType CaSS2 = square(state.CaSS);
        const FloatType eq56 = params.k65 / (params.k56*CaSS2 + params.k65);
        const FloatType tau34 = 1.0 / (params.k34*CaSS2 + params.k43);

        // state 1 -> state 2
        state.RyR_rates[0] = state.RyR[0]*params.k12*CaSS2;
        // state 2 -> state 3
        state.RyR_rates[1] = state.RyR[1]*params.k23*CaSS2;
        // state 2 -> state 5
        state.RyR_rates[2] = state.RyR[1]*params.k25*CaSS2;
        // state 3 -> state 4
        state.RyR_rates[3] = state.CaSS > 3.685e-2 ? 0 : state.RyR[2]*params.k34*CaSS2;
        // state 4 -> state 5
        state.RyR_rates[4] = state.CaSS > 3.685e-2 ? (state.RyR[2]+state.RyR[3])*params.k45*params.k34*CaSS2*tau34 : state.RyR[3]*params.k45;
        // state 5 -> state 6
        state.RyR_rates[5] = state.CaSS > 1.15e-4 ? 0 : state.RyR[4]*params.k56*CaSS2;
        // state 2 -> state 1
        state.RyR_rates[6] = state.RyR[1]*params.k21;
        // state 3 -> state 2
        state.RyR_rates[7] = state.CaSS > 3.685e-2 ? (state.RyR[2]+state.RyR[3])*params.k32*params.k43*tau34 : state.RyR[2]*params.k32;
        // state 4 -> state 3
        state.RyR_rates[8] = state.CaSS > 3.685e-2 ? 0 : state.RyR[3]*params.k43;
        // state 5 -> state 2
        state.RyR_rates[9] = state.CaSS > 1.15e-4 ? (state.RyR[4]+state.RyR[5])*params.k52*eq56 : state.RyR[4]*params.k52; 
        // state 5 -> state 4
        state.RyR_rates[10] = state.CaSS > 1.15e-4 ? (state.RyR[4]+state.RyR[5])*params.k54*CaSS2*eq56 : state.RyR[4]*params.k54*CaSS2;
        // state 6 -> state 5
        state.RyR_rates[11] = state.CaSS > 1.15e-4 ? 0 : state.RyR[5]*params.k65; 
        
        state.RyR_tot_rate = state.RyR_rates[0] + state.RyR_rates[1] + state.RyR_rates[2] + state.RyR_rates[3] + state.RyR_rates[4] + state.RyR_rates[5] + state.RyR_rates[6] + state.RyR_rates[7] + 
                             state.RyR_rates[8] + state.RyR_rates[9] + state.RyR_rates[10] + state.RyR_rates[11]; 

        
        if (state.LCC_inactivation == 0){
            state.LCC_inactivation = 1;
            state.LCC_inactivation_rate = (1.0 - consts.yinfLCC) / consts.tauLCC;
        }
        else {
            state.LCC_inactivation = 0;
            state.LCC_inactivation_rates = consts.yinfLCC / consts.tauLCC;
        }
            
        if (state.ClCh == 0){
            state.ClCh = 1;
            state.ClCh_rate = params.kbClCh;
        }
        else {
            state.ClCh = 0;
            state.ClCh_rate = params.kfClCh * state.CaSS;
        }
    }


    template <typename FloatType>
    void calculate_fluxes(CRUStateThread<FloatType> &state, const FloatType Cai, const FloatType CaNSR, const Parameters<FloatType> &params){
        state.Jtr = params.rtr * (CaNSR - state.CaJSR);
        state.Jxfer = params.rxfer * (state.CaSS - Cai); 
        state.Jrel = (state.RyR[2] + state.RyR[3]) * params.rRyR * (state.CaJSR - state.CaSS);
        if (state.LCC == 6 || state.LCC == 12)
            state.JLCC = state.LCC_inactivation * consts.JLCC_multiplier * (consts.Cao_scaled - consts.JLCC_exp * state.CaSS); 
        else
            state.JLCC = 0.0;
    }


    template <typename FloatType, typename Generator>
    void sample_RyR56(CRULatticeStateThread<FloatType> &state, const Parameters<FloatType> &params){
        int n56 = state.RyR[4] + state.RyR[5];
        if (n56 > 0){
            FloatType p = params.k65 / (params.k65 + params.k56 * square(state.CaSS));
            state.RyR[4] = sample_binomial<FloatType, Generator>(p, n56);
            state.RyR[5] = n56 - state.RyR[4];
        }
        else {
            state.RyR[4] = 0;
            state.RyR[5] = 0;
        }
    }
    
    template <typename FloatType, typename Generator>
    void sample_RyR34(CRULatticeStateThread<FloatType> &state, const Parameters<FloatType> &params){
        int n34 = state.RyR[2] + state.RyR[3];
        if (n34 > 0){
            FloatType p = params.k43 / (params.k43 + params.k34 * square(state.CaSS));
            state.RyR[2] = sample_binomial<FloatType, Generator>(p, n34);
            state.RyR[3] = n34 - state.RyR[2];
        }
        else {
            state.RyR[2] = 0;
            state.RyR[3] = 0;
        }
    }


    template <typename FloatType, typename Generator>
    void sample_LCC(CRULatticeStateThread<FloatType> &state, const Constants<FloatType> &consts){
        const int transition = sample_weights<FloatType, int, Generator>(state.LCC_rates, state.LCC_tot_rate, 3);
        switch (state.LCC){
        case 1:
            if (transition == 0){
                state.LCC = 2;
                state.LCC_rates[0] = consts.betaLCC;
                state.LCC_rates[1] = 3*consts.alphaLCC;
                state.LCC_rates[2] = consts.gamma0a*state.CaSS;
            }
            else {
                state.LCC = 7;
                state.LCC_rates[0] = consts.omega;
                state.LCC_rates[1] = 4*consts.alphaLCC*consts.a;
                state.LCC_rates[2] = 0;
            }
            break;
        case 2:
            if (transition == 0){
                state.LCC = 1;
                state.LCC_rates[0] = 4*consts.alphaLCC;
                state.LCC_rates[1] = consts.gamma0*state.CaSS;
                state.LCC_rates[2] = 0;
            }
            else if (transition == 1){
                state.LCC = 3;
                state.LCC_rates[0] = 2*consts.betaLCC;
                state.LCC_rates[1] = 2*consts.alphaLCC;
                state.LCC_rates[2] = consts.gamma0a2*state.CaSS;
            }
            else { 
                state.LCC = 8;
                state.LCC_rates[0] = consts.omega_b;
                state.LCC_rates[1] = consts.betaLCC*consts.binv;
                state.LCC_rates[2] = 3*consts.alphaLCC*consts.a;
            }
            break;
        case 3:
            if (transition == 0){
                state.LCC = 2;
                state.LCC_rates[0] = consts.betaLCC;
                state.LCC_rates[1] = 3*consts.alphaLCC;
                state.LCC_rates[2] = consts.gamma0a*state.CaSS;
            }
            else if (transition == 1) {
                state.LCC = 4;
                state.LCC_rates[0] = 3*consts.betaLCC;
                state.LCC_rates[1] = consts.alphaLCC;
                state.LCC_rates[2] = consts.gamma0a3*state.CaSS;
            }
            else { 
                state.LCC = 9;
                state.LCC_rates[0] = consts.omega_b2;
                state.LCC_rates[1] = 2*consts.betaLCC*consts.binv;
                state.LCC_rates[2] = 2*consts.alphaLCC*consts.a;
            }
            break;
        case 4:
            if (transition == 0){
                state.LCC = 3;
                state.LCC_rates[0] = 2*consts.betaLCC;
                state.LCC_rates[1] = 2*consts.alphaLCC;
                state.LCC_rates[2] = consts.gamma0a2*state.CaSS;
            }
            else if (transition == 1){
                state.LCC = 5;
                state.LCC_rates[0] = 4*consts.betaLCC;
                state.LCC_rates[1] = params.f;
                state.LCC_rates[2] = consts.gamma0a4*state.CaSS;
            }
            else {
                state.LCC = 10;
                state.LCC_rates[0] = consts.omega_b3;
                state.LCC_rates[1] = 3*consts.betaLCC*consts.binv;
                state.LCC_rates[2] = consts.alphaLCC*consts.a;
            }
            break;
        case 5:
            if (transition == 0){
                state.LCC = 4; 
                state.LCC_rates[0] = 3*consts.betaLCC;
                state.LCC_rates[1] = consts.alphaLCC;
                state.LCC_rates[2] = consts.gamma0a3*state.CaSS;
            }
            else if (transition == 1) {
                state.LCC = 6; 
                state.LCC_rates[0] = params.g;
                state.LCC_rates[1] = 0;
                state.LCC_rates[2] = 0;
            }
            else { 
                state.LCC = 11;
                state.LCC_rates[0] = consts.omega_b4;
                state.LCC_rates[1] = 4*consts.betaLCC*consts.binv;
                state.LCC_rates[2] = params.f1;
            }
            break;
        case 6:
            state.LCC = 5;
            state.LCC_rates[0] = 4*consts.betaLCC;
            state.LCC_rates[1] = params.f;
            state.LCC_rates[2] = consts.gamma0a4*state.CaSS;
            break;
        case 7:
            if (transition == 0){
                state.LCC = 1;
                state.LCC_rates[0] = 4*consts.alphaLCC;
                state.LCC_rates[1] = consts.gamma0*state.CaSS;
                state.LCC_rates[2] = 0;
            }
            else { 
                state.LCC = 8;
                state.LCC_rates[0] = consts.omega_b;
                state.LCC_rates[1] = consts.betaLCC*consts.binv;
                state.LCC_rates[2] = 3*consts.alphaLCC*consts.a;
            }
            break;
        case 8:
            if (transition == 0) {
                state.LCC = 2;
                state.LCC_rates[0] = consts.betaLCC;
                state.LCC_rates[1] = 3*consts.alphaLCC;
                state.LCC_rates[2] = consts.gamma0a*state.CaSS;
            }
            else if (transition == 1) {
                state.LCC = 7;
                state.LCC_rates[0] = consts.omega;
                state.LCC_rates[1] = 4*consts.alphaLCC*consts.a;
                state.LCC_rates[2] = 0;
            }
            else {
                state.LCC = 9;
                state.LCC_rates[0] = consts.omega_b2;
                state.LCC_rates[1] = 2*consts.betaLCC*consts.binv;
                state.LCC_rates[2] = 2*consts.alphaLCC*consts.a;
            }
            break;
        case 9:
            if (transition == 0){
                state.LCC = 3;
                state.LCC_rates[0] = 2*consts.betaLCC;
                state.LCC_rates[1] = 2*consts.alphaLCC;
                state.LCC_rates[2] = consts.gamma0a2*state.CaSS;
            }
            else if (transition == 1){
                state.LCC = 8;
                state.LCC_rates[0] = consts.omega_b;
                state.LCC_rates[1] = consts.betaLCC*consts.binv;
                state.LCC_rates[2] = 3*consts.alphaLCC*consts.a;
            }
            else { 
                state.LCC = 10;
                state.LCC_rates[0] = consts.omega_b3;
                state.LCC_rates[1] = 3*consts.betaLCC*consts.binv;
                state.LCC_rates[2] = consts.alphaLCC*consts.a;
            }
            break;
        case 10:
            if (transition == 0){
                state.LCC = 4;
                state.LCC_rates[0] = 3*consts.betaLCC;
                state.LCC_rates[1] = consts.alphaLCC;
                state.LCC_rates[2] = consts.gamma0a3*state.CaSS;
            }
            else if (transition == 1){
                state.LCC = 9;
                state.LCC_rates[0] = consts.omega_b2;
                state.LCC_rates[1] = 2*consts.betaLCC*consts.binv;
                state.LCC_rates[2] = 2*consts.alphaLCC*consts.a;
            }
            else { 
                state.LCC = 11;
                state.LCC_rates[0] = consts.omega_b4;
                state.LCC_rates[1] = 4*consts.betaLCC*consts.binv;
                state.LCC_rates[2] = params.f1;
            }
            break;
        case 11:
            if (transition == 0) {
                state.LCC = 5; 
                state.LCC_rates[0] = 4*consts.betaLCC;
                state.LCC_rates[1] = params.f;
                state.LCC_rates[2] = consts.gamma0a4*state.CaSS;
            }
            else if (transition == 1) {
                state.LCC = 10; 
                state.LCC_rates[0] = consts.omega_b3;
                state.LCC_rates[1] = 3*consts.betaLCC*consts.binv;
                state.LCC_rates[2] = consts.alphaLCC*consts.a;
            }
            else { 
                state.LCC = 12; 
                state.LCC_rates[0] = params.g1;
                state.LCC_rates[1] = 0;
                state.LCC_rates[2] = 0;
            }
            break;
        case 12:
            state.LCC = 11;
            state.LCC_rates[0] = consts.omega_b4;
            state.LCC_rates[1] = 4*consts.betaLCC*consts.binv;
            state.LCC_rates[2] = params.f1;
            break;    
        default:
            break;
        }
        state.LCC_tot_rate = state.LCC_rates[0] + state.LCC_rates[1] + state.LCC_rates[2];
    }
    

    template <typename FloatType, typename Generator>
    inline void sample_RyR(CRULatticeStateThread<FloatType> &state, const Parameters<FloatType> &params){
        const int transition = sample_weights<FloatType, int, Generator>(state.RyR_rates, state.RyR_tot_rate, 12);
        FloatType CaSS2 = square(state.CaSS);
        switch (transition){
        case 0: // 1 -> 2
            --state.RyR[0];
            ++state.RyR[1];

            state.RyR_rates[0] -= params.k12*CaSS2; // 1 -> 2 --

            state.RyR_rates[1] += params.k23*CaSS2; // 2 -> 3 ++
            state.RyR_rates[6] += params.k21; // 2 -> 1 ++
            state.RyR_rates[2] += params.k25*CaSS2; // 2 -> 5 ++
            break;
        case 1: // 2 -> 3
            --state.RyR[1];
            ++state.RyR[2];   

            state.RyR_rates[1] -= params.k23*CaSS2; // 2 -> 3 --
            state.RyR_rates[6] -= params.k21; // 2 -> 1 --
            state.RyR_rates[2] -= params.k25*CaSS2; // 2 -> 5 --

            if (state.CaSS > 3.685e-2) {
                state.RyR_rates[3] = 0; // 3 -> 4 = 0
                state.RyR_rates[7] += (params.k32*params.k43 / (params.k34*CaSS2 + params.k43)); // 3+4 -> 2 ++
                state.RyR_rates[4] += params.k45*params.k34*CaSS2 / (params.k34*CaSS2 + params.k43); // 3+4 -> 5 ++
            }
            else {
                state.RyR_rates[3] += params.k34*CaSS2; // 3 -> 4 ++
                state.RyR_rates[7] += params.k32; // 3 -> 2 ++
            }
            break;
        case 2: // 2 -> 5
            --state.RyR[1];
            ++state.RyR[4];   
            
            state.RyR_rates[1] -= params.k23*CaSS2; // 2 -> 3 --
            state.RyR_rates[6] -= params.k21; // 2 -> 1 --
            state.RyR_rates[2] -= params.k25*CaSS2; // 2 -> 5 --
            if (state.CaSS > 1.15e-4) {
                const FloatType eq56 = params.k65 / (params.k56*CaSS2 + params.k65);
                state.RyR_rates[5] = 0; // 5 -> 6 = 0
                state.RyR_rates[9] += params.k52*eq56; // 5+6 -> 2 ++ 
                state.RyR_rates[10] += params.k54*CaSS2*eq56; // 5+6 -> 4 ++
            }
            else {
                state.RyR_rates[5] += params.k56*CaSS2; // 5 -> 6 ++
                state.RyR_rates[9] += params.k52;  // 5 -> 2 ++
                state.RyR_rates[10] += params.k54*CaSS2; // 5 -> 4 ++
            }
            break;
        case 3: // 3 -> 4
            --state.RyR[2];
            ++state.RyR[3];
            if (state.CaSS <= 3.685e-2) {
                state.RyR_rates[3] -= params.k34*CaSS2; // 3 -> 4 --
                state.RyR_rates[7] -= params.k32; // 3 -> 2 --
                state.RyR_rates[4] += params.k45; // 4 -> 5 ++
                state.RyR_rates[8] += params.k43; // 4 -> 3 ++
            }
            break;
        case 4: // 4 -> 5
            --state.RyR[3];
            ++state.RyR[4];
            if (state.CaSS > 3.685e-2) {
                state.RyR_rates[4] -= params.k45*params.k34*CaSS2 / (params.k34*CaSS2 + params.k43); // 3+4 -> 5 --
                state.RyR_rates[7] -= (params.k32*params.k43 / (params.k34*CaSS2 + params.k43)); // 3+4 -> 2 --
                state.RyR_rates[8] = 0; // 4 -> 3 = 0
            }
            else {
                state.RyR_rates[4] -= params.k45; // 4 -> 5 --
                state.RyR_rates[8] -= params.k43; // 4 -> 3 --
            }
            
            if (state.CaSS > 1.15e-4) {
                const FloatType eq56 = params.k65 / (params.k56*CaSS2 + params.k65);
                state.RyR_rates[5] = 0; // 5 -> 6 = 0
                state.RyR_rates[9] += params.k52*eq56; // 5+6 -> 2 ++ 
                state.RyR_rates[10] += params.k54*CaSS2*eq56; // 5+6 -> 4 ++
            }
            else {
                state.RyR_rates[5] += params.k56*CaSS2; // 5 -> 6 ++
                state.RyR_rates[9] += params.k52; // 5 -> 2 ++
                state.RyR_rates[10] += params.k54*CaSS2; // 5 -> 4 ++
            }
            break;
        case 5: // 5 -> 6
            --state.RyR[4];
            ++state.RyR[5];   
            
            if (state.CaSS <= 1.15e-4) { 
                state.RyR_rates[5] -= params.k56*CaSS2; // 5 -> 6 --
                state.RyR_rates[9] -= params.k52; // 5 -> 2 --
                state.RyR_rates[10] -= params.k54*CaSS2; // 5 -> 4 --
                state.RyR_rates[11] += params.k65; // 6 -> 5 ++
            }
            break;
        case 6: // 2 -> 1
            --state.RyR[1];
            ++state.RyR[0];   
            
            state.RyR_rates[1] -= params.k23*CaSS2; // 2 -> 3 --
            state.RyR_rates[6] -= params.k21; // 2 -> 1 --
            state.RyR_rates[2] -= params.k25*CaSS2; // 2 -> 5 --
            
            state.RyR_rates[0] += params.k12*CaSS2; // 1 -> 2 ++
            break;
        case 7: // 3 -> 2
            --state.RyR[2];
            ++state.RyR[1];
            if (state.CaSS > 3.685e-2) {
                state.RyR_rates[3] = 0; // 3 -> 4 = 0
                state.RyR_rates[7] -= (params.k32*params.k43 / (params.k34*CaSS2 + params.k43)); // 3+4 -> 2 --
                state.RyR_rates[4] -= params.k45*params.k34*CaSS2 / (params.k34*CaSS2 + params.k43); // 3+4 -> 5 --
            }
            else {
                state.RyR_rates[3] -= params.k34*CaSS2; // 3 -> 4 --
                state.RyR_rates[7] -= params.k32; // 3 -> 2 --
            }
            
            state.RyR_rates[1] += params.k23*CaSS2; // 2 -> 3 ++
            state.RyR_rates[6] += params.k21; // 2 -> 1 ++
            state.RyR_rates[2] += params.k25*CaSS2; // 2 -> 5 ++

            break;
        case 8: // 4 -> 3
            --state.RyR[3];
            ++state.RyR[2];   
            if (state.CaSS <= 3.685e-2) {
                state.RyR_rates[4] -= params.k45; // 4 -> 5 --
                state.RyR_rates[8] -= params.k43; // 4 -> 3 --
                state.RyR_rates[3] += params.k34*CaSS2; // 3 -> 4 ++
                state.RyR_rates[7] += params.k32; // 3 -> 2 ++
            }
            break;
        case 9: // 5 -> 2
            --state.RyR[4];
            ++state.RyR[1];   
            if (state.CaSS > 1.15e-4) {
                const FloatType eq56 = params.k65 / (params.k56*CaSS2 + params.k65);
                state.RyR_rates[5] = 0; // 5 -> 6 = 0
                state.RyR_rates[9] -= params.k52*eq56; // 5+6 -> 2 -- 
                state.RyR_rates[10] -= params.k54*CaSS2*eq56; // 5+6 -> 4 --
            }
            else {
                state.RyR_rates[5] -= params.k56*CaSS2; // 5 -> 6 --
                state.RyR_rates[9] -= params.k52; // 5 -> 2 --
                state.RyR_rates[10] -= params.k54*CaSS2; // 5 -> 4 --
            }
            state.RyR_rates[1] += params.k23*CaSS2; // 2 -> 3 ++
            state.RyR_rates[6] += params.k21; // 2 -> 1 ++
            state.RyR_rates[2] += params.k25*CaSS2; // 2 -> 5 ++
            break;
        case 10: // 5 -> 4
            --state.RyR[4];
            ++state.RyR[3];
            if (state.CaSS > 1.15e-4) {
                const FloatType eq56 = params.k65 / (params.k56*CaSS2 + params.k65);
                state.RyR_rates[5] = 0; // 5 -> 6 = 0
                state.RyR_rates[9] -= params.k52*eq56; // 5+6 -> 2 -- 
                state.RyR_rates[10] -= params.k54*CaSS2*eq56; // 5+6 -> 4 --
            }
            else {
                state.RyR_rates[5] -= params.k56*CaSS2; // 5 -> 6 --
                state.RyR_rates[9] -= params.k52; // 5 -> 2 --
                state.RyR_rates[10] -= params.k54*CaSS2; // 5 -> 4 --
            }
            if (state.CaSS > 3.685e-2) {
                state.RyR_rates[4] += params.k45*params.k34*CaSS2 / (params.k34*CaSS2 + params.k43); // 3+4 -> 5 ++
                state.RyR_rates[7] += (params.k32*params.k43 / (params.k34*CaSS2 + params.k43)); // 3+4 -> 2 ++
                state.RyR_rates[8] = 0; // 4 -> 3 = 0
            }
            else {
                state.RyR_rates[4] += params.k45; // 4 -> 5 ++
                state.RyR_rates[8] += params.k43; // 4 -> 3 ++
            }
            break;
        case 11: // 6 -> 5
            --state.RyR[5];
            ++state.RyR[4];   
            if (state.CaSS <= 1.15e-4) {
                state.RyR_rates[5] += params.k56*CaSS2; // 5 -> 6 ++
                state.RyR_rates[9] += params.k52;  // 5 -> 2 ++
                state.RyR_rates[10] += params.k54*CaSS2; // 5 -> 4 ++
                state.RyR_rates[11] -= params.k65; // 6 -> 5 --
            }
            break;
        default:
            break;
        }
        state.RyR_tot_rate = state.RyR_rates[0] + state.RyR_rates[1] + state.RyR_rates[2] + state.RyR_rates[3] + state.RyR_rates[4] + state.RyR_rates[5] + state.RyR_rates[6] + state.RyR_rates[7] + 
                             state.RyR_rates[8] + state.RyR_rates[9] + state.RyR_rates[10] + state.RyR_rates[11]; 
    }

    template <typename FloatType, typename Generator>
    void sample_new_state(CRULatticeStateThread<FloatType> &state, const FloatType total_rate, const Parameters<FloatType> &params, const Constants<FloatType> &consts){
        FloatType u = urand<FloatType, Generator>() * total_rate;
        if (u < state.LCC_tot_rate)
            sample_LCC<FloatType, Generator>(state, consts);
        else if (u < (state.LCC_tot_rate + state.LCC_inactivation_rate)) {
            if (state.LCC_inactivation == 0){
                state.LCC_inactivation = 1;
                state.LCC_inactivation_rate = (1.0 - consts.yinfLCC) / consts.tauLCC;
            }
            else {
                state.LCC_inactivation = 0;
                state.LCC_inactivation_rates = consts.yinfLCC / consts.tauLCC;
            }
        } 
        else if (u < (state.LCC_tot_rate + state.LCC_inactivation_rate + state.RyR_tot_rate))
            sample_RyR<FloatType, Generator>(state, params);
        else {
            if (state.ClCh == 0){
                state.ClCh = 1;
                state.ClCh_rate = params.kbClCh;
            }
            else {
                state.ClCh = 0;
                state.ClCh_rate = params.kfClCh * state.CaSS;
            }
        }
    }

    template <typename FloatType, typename Generator>
    void SSA_single_su(CRULatticeStateThread<FloatType> &state, const FloatType Cai, const FloatType CaNSR, const FloatType time_int, const Parameters<FloatType> &params, const Constants<FloatType> &consts){
        int subunit_idx;
        FloatType t = 0, dt = 0, total_rate;
        init_rates(state, params, consts);

        while (1){
            //update_fluxes(state, Cai, CaNSR, params);

            total_rate = state.LCC_tot_rate + state.LCC_inactivation_rate + state.RyR_tot_rate + state.ClCh_rate;
            dt = -log(urand<FloatType, Generator>()) / total_rate;


            if (t + dt > time_int){
                update_fluxes(state, Cai, CaNSR, params);
                partial_euler_step(state, time_int, params, consts);
                update_fluxes(state, Cai, CaNSR, params); // update again for use in diffusion step.
                break;
            } 
            t += dt;
            
            sample_new_state<FloatType, Generator>(state, total_rate, params, consts);
        }
    }
    
    template <typename FloatType>
    void partial_euler_step(CRULatticeStateThread<FloatType> &state, const FloatType dt, const Parameters<FloatType> &params, const Constants<FloatType> &consts){
        FloatType betaSS = 1.0 / (1 + (consts.BSR_const / square(params.KBSR + state.CaSS)) + (consts.BSL_const / square(params.KBSL + state.CaSS)));
        FloatType betaJSR = 1.0 / (1 + (consts.CSQN_const / square(params.KCSQN + state.CaJSR)));
        state.CaSS += (dt * betaSS * (state.Jrel + state.JLCC - state.Jxfer));
        state.CaJSR += (dt * (state.Jtr - consts.VSS_VJSR * state.Jrel));
    }

}