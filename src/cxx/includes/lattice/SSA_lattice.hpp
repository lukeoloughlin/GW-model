#pragma once

#include "GW_lattice_utils.hpp"
#include "../common.hpp"
#include <Eigen/Core>

namespace GW_lattice { 
    
    class CRULatticeStateThread {
    public:
        int LCC;
        int LCC_inactivation;
        int RyR[6]; // Have to count states here so we include all 6 possible states
        int ClCh;

        double CaSS;
        double CaJSR;

        double LCC_rates[3]; // 3 rates to track at most
        double LCC_inactivation_rate;
        double RyR_rates[12];
        double ClCh_rate;

        double LCC_tot_rate;
        double RyR_tot_rate;

        double CaSS_prev; // Needed to know whether to sample RyR state accroding to stationary dist

        inline void copy_from_CRULatticeState(const CRULatticeState& state, const double CaSS_prev_, const int x, const int y, const Parameters& params){
            for (int j = 0; j < 6; ++j)
                RyR[j] = state.RyR.array(x,y,j);

            LCC = state.LCC(x,y);
            LCC_inactivation = state.LCC_inactivation(x,y);
            ClCh = state.ClCh(x,y);

            CaSS = state.CaSS(x,y);
            CaJSR = state.CaJSR(x,y);

            CaSS_prev = CaSS_prev_;
        }

        inline void init_rates(const Parameters& params, const Constants& consts);
        
    };

    void CRULatticeStateThread::init_rates(const Parameters& params, const Constants& consts){
        switch (LCC){
        case 1:
            LCC_rates[0] = 4*consts.alphaLCC;
            LCC_rates[1] = consts.gamma0*CaSS;
            LCC_rates[2] = 0;
            break;
        case 2:
            LCC_rates[0] = consts.betaLCC;
            LCC_rates[1] = 3*consts.alphaLCC;
            LCC_rates[2] = consts.gamma0a*CaSS;
            break;
        case 3:
            LCC_rates[0] = 2*consts.betaLCC;
            LCC_rates[1] = 2*consts.alphaLCC;
            LCC_rates[2] = consts.gamma0a2*CaSS;
            break;
        case 4:
            LCC_rates[0] = 3*consts.betaLCC;
            LCC_rates[1] = consts.alphaLCC;
            LCC_rates[2] = consts.gamma0a3*CaSS;
            break;
        case 5:
            LCC_rates[0] = 4*consts.betaLCC;
            LCC_rates[1] = params.f;
            LCC_rates[2] = consts.gamma0a4*CaSS;
            break;
        case 6:
            LCC_rates[0] = params.g;
            LCC_rates[1] = 0;
            LCC_rates[2] = 0;
            break;
        case 7:
            LCC_rates[0] = consts.omega;
            LCC_rates[1] = 4*consts.alphaLCC*consts.a;
            LCC_rates[2] = 0;
            break;
        case 8:
            LCC_rates[0] = consts.omega_b;
            LCC_rates[1] = consts.betaLCC*consts.binv;
            LCC_rates[2] = 3*consts.alphaLCC*consts.a;
            break;
        case 9:
            LCC_rates[0] = consts.omega_b2;
            LCC_rates[1] = 2*consts.betaLCC*consts.binv;
            LCC_rates[2] = 2*consts.alphaLCC*consts.a;
            break;
        case 10:
            LCC_rates[0] = consts.omega_b3;
            LCC_rates[1] = 3*consts.betaLCC*consts.binv;
            LCC_rates[2] = consts.alphaLCC*consts.a;
            break;
        case 11:
            LCC_rates[0] = consts.omega_b4;
            LCC_rates[1] = 4*consts.betaLCC*consts.binv;
            LCC_rates[2] = params.f1;
            break;
        case 12:
            LCC_rates[0] = params.g1;
            LCC_rates[1] = 0;
            LCC_rates[2] = 0;
            break;
        default:
            break;
        }    
        LCC_tot_rate = LCC_rates[0] + LCC_rates[1] + LCC_rates[2];

        const double CaSS2 = square(CaSS);
        const double eq56 = params.k65 / (params.k56*CaSS2 + params.k65);
        const double tau34 = 1.0 / (params.k34*CaSS2 + params.k43);

        // state 1 -> state 2
        RyR_rates[0] = RyR[0]*params.k12*CaSS2;
        // state 2 -> state 3
        RyR_rates[1] = RyR[1]*params.k23*CaSS2;
        // state 2 -> state 5
        RyR_rates[2] = RyR[1]*params.k25*CaSS2;
        // state 3 -> state 4
        RyR_rates[3] = CaSS > 3.685e-2 ? 0 : RyR[2]*params.k34*CaSS2;
        // state 4 -> state 5
        RyR_rates[4] = CaSS > 3.685e-2 ? (RyR[2]+RyR[3])*params.k45*params.k34*CaSS2*tau34 : RyR[3]*params.k45;
        // state 5 -> state 6
        RyR_rates[5] = CaSS > 1.15e-4 ? 0 : RyR[4]*params.k56*CaSS2;
        // state 2 -> state 1
        RyR_rates[6] = RyR[1]*params.k21;
        // state 3 -> state 2
        RyR_rates[7] = CaSS > 3.685e-2 ? (RyR[2]+RyR[3])*params.k32*params.k43*tau34 : RyR[2]*params.k32;
        // state 4 -> state 3
        RyR_rates[8] = CaSS > 3.685e-2 ? 0 : RyR[3]*params.k43;
        // state 5 -> state 2
        RyR_rates[9] = CaSS > 1.15e-4 ? (RyR[4]+RyR[5])*params.k52*eq56 : RyR[4]*params.k52; 
        // state 5 -> state 4
        RyR_rates[10] = CaSS > 1.15e-4 ? (RyR[4]+RyR[5])*params.k54*CaSS2*eq56 : RyR[4]*params.k54*CaSS2;
        // state 6 -> state 5
        RyR_rates[11] = CaSS > 1.15e-4 ? 0 : RyR[5]*params.k65; 
        
        RyR_tot_rate = RyR_rates[0] + RyR_rates[1] + RyR_rates[2] + RyR_rates[3] + RyR_rates[4] + RyR_rates[5] + RyR_rates[6] + RyR_rates[7] + 
                             RyR_rates[8] + RyR_rates[9] + RyR_rates[10] + RyR_rates[11]; 

        
        if (LCC_inactivation == 0)
            LCC_inactivation_rate = consts.yinfLCC / consts.tauLCC;
        else 
            LCC_inactivation_rate = (1.0 - consts.yinfLCC) / consts.tauLCC;
            
        if (ClCh == 0)
            ClCh_rate = params.kfClCh * CaSS;
        else 
            ClCh_rate = params.kbClCh;
        
    }

    
    template <typename PRNG>
    inline void sample_RyR56(CRULatticeStateThread& state, const Parameters& params){
        int n56 = state.RyR[4] + state.RyR[5];
        if (n56 > 0){
            double p = params.k65 / (params.k65 + params.k56 * square(state.CaSS_prev));
            state.RyR[4] = sample_binomial<double, PRNG>(p, n56);
            state.RyR[5] = n56 - state.RyR[4];
        }
        else {
            state.RyR[4] = 0;
            state.RyR[5] = 0;
        }
    }

    template <typename PRNG>
    inline void sample_RyR34(CRULatticeStateThread& state, const Parameters& params){
        int n34 = state.RyR[2] + state.RyR[3];
        if (n34 > 0){
            double p = params.k43 / (params.k43 + params.k34 * square(state.CaSS_prev));
            state.RyR[2] = sample_binomial<double, PRNG>(p, n34);
            state.RyR[3] = n34 - state.RyR[2];
        }
        else {
            state.RyR[2] = 0;
            state.RyR[3] = 0;
        }
    }

    template <typename PRNG>
    inline void sample_LCC(CRULatticeStateThread& state, const Constants& consts){
        const int transition = sample_weights<double, int, PRNG>(state.LCC_rates, state.LCC_tot_rate, 3);
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
                state.LCC_rates[1] = consts.f;
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
                state.LCC_rates[0] = consts.g;
                state.LCC_rates[1] = 0;
                state.LCC_rates[2] = 0;
            }
            else { 
                state.LCC = 11;
                state.LCC_rates[0] = consts.omega_b4;
                state.LCC_rates[1] = 4*consts.betaLCC*consts.binv;
                state.LCC_rates[2] = consts.f1;
            }
            break;
        case 6:
            state.LCC = 5;
            state.LCC_rates[0] = 4*consts.betaLCC;
            state.LCC_rates[1] = consts.f;
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
                state.LCC_rates[2] = consts.f1;
            }
            break;
        case 11:
            if (transition == 0) {
                state.LCC = 5; 
                state.LCC_rates[0] = 4*consts.betaLCC;
                state.LCC_rates[1] = consts.f;
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
                state.LCC_rates[0] = consts.g1;
                state.LCC_rates[1] = 0;
                state.LCC_rates[2] = 0;
            }
            break;
        case 12:
            state.LCC = 11;
            state.LCC_rates[0] = consts.omega_b4;
            state.LCC_rates[1] = 4*consts.betaLCC*consts.binv;
            state.LCC_rates[2] = consts.f1;
            break;    
        default:
            break;
        }
        state.LCC_tot_rate = state.LCC_rates[0] + state.LCC_rates[1] + state.LCC_rates[2];
    }

    template <typename PRNG>
    inline void sample_RyR(CRULatticeStateThread &state, const Parameters &params){
        const int transition = sample_weights<double, int, PRNG>(state.RyR_rates, state.RyR_tot_rate, 12);
        double CaSS2 = square(state.CaSS);
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
                const double eq56 = params.k65 / (params.k56*CaSS2 + params.k65);
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
                const double eq56 = params.k65 / (params.k56*CaSS2 + params.k65);
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
                const double eq56 = params.k65 / (params.k56*CaSS2 + params.k65);
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
                const double eq56 = params.k65 / (params.k56*CaSS2 + params.k65);
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
 
    template <typename PRNG>
    inline void sample_new_state(CRULatticeStateThread& state, const double total_rate, const Parameters& params, const Constants& consts){
        double u = urand<double, PRNG>() * total_rate;
        if (u < state.LCC_tot_rate)
            sample_LCC<PRNG>(state, consts);
        else if (u < (state.LCC_tot_rate + state.LCC_inactivation_rate)) {
            if (state.LCC_inactivation == 0){
                state.LCC_inactivation = 1;
                state.LCC_inactivation_rate = (1.0 - consts.yinfLCC) / consts.tauLCC;
            }
            else {
                state.LCC_inactivation = 0;
                state.LCC_inactivation_rate = consts.yinfLCC / consts.tauLCC;
            }
        } 
        else if (u < (state.LCC_tot_rate + state.LCC_inactivation_rate + state.RyR_tot_rate))
            sample_RyR<PRNG>(state, params);
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
    
    template <typename PRNG>
    inline void SSA_single_su(CRULatticeStateThread& state, const double time_int, const Parameters& params, const Constants& consts){
        double t = 0, dt = 0, total_rate;

        if ((state.CaSS_prev > 0.115e-3 && state.CaSS <= 0.115e-3))
            sample_RyR56<PRNG>(state, params);

        if ((state.CaSS_prev > 36.85e-3 && state.CaSS <= 36.85e-3))
            sample_RyR34<PRNG>(state, params);
        
        state.init_rates(params, consts);
        while (1){
            total_rate = state.LCC_tot_rate + state.LCC_inactivation_rate + state.RyR_tot_rate + state.ClCh_rate;
            dt = -log(urand<double, PRNG>()) / total_rate;
            if (t + dt > time_int)
                break;
            t += dt;
            sample_new_state<PRNG>(state, total_rate, params, consts);
        }
    }




}

