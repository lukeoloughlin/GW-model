#ifndef SSA_H
#define SSA_H

#include "GW_utils.hpp"
#include "common.hpp"
#include "ndarray.hpp"
//#include <omp.h>
//#include <unistd.h>

/*
double update_rates(const int* const, const int* const, const int* const, const int* const, double* const, double* const, double* const, double* const, const double* const, double*, const double, const double, const double, const double, const Constants&);
void update_fluxes(const double* const, const double, double*, double*, const Constants&);
void update_CaSS(double* const, int*, const double* const, const double* const, const double* const, const double* const, const double, const Constants&);
void update_state(int* const, int* const, int* const, double* const, int* const, const double* const, const double* const, const double* const, const double* const, const double* const, const int, const double* const, double*, double*, const double, const double, const double, const Constants&);
void sample_LCC(int* const, const double* const, const double, const int* const, const double* const, double*, const int, const double, const double, const Constants&);
void sample_RyR(int* const, double* const, const double* const, const double, double*, const int, const double* const, const double ,  const Constants&);
void SSA_subunit(int* const, int* const, int* const, double* const, int* const, double* const, double* const, double* const, double* const, double* const, double* const, double* const, double* const, double* const, double* const, double, const double, const double, const double, const double, const double, const double, const double, const double, const double, double* const, const Constants&);
void SSA(NDArray<int,2>&, NDArray<int,2>&, NDArray<int,3>&, NDArray<int,2>&, NDArray<double,2>&, NDArray<double,1>&, const double, const double, NDArray<double,2>&, NDArray<double,2>&, NDArray<double,1>&, const double, const double, const double, const int, const Constants&);
*/

/*
int sample_weights(const double* const weights, const double total_weight, const int size){
    const double u = urand() * total_weight;
    double cum_weight = weights[0];
    int i = 0;
    while (cum_weight < u && i < (size-1)){
        i++;
        cum_weight += weights[i];
    }
    return i;
}
*/

namespace GW {

    template <typename FloatType>
    struct CRUStateThread {
        int LCC[4];
        int LCC_activation[4];
        int RyR[4*6]; // Have to count states here so we include all 6 possible states
        FloatType open_RyR[4]; // Keep track of number open RyR
        int ClCh[4];

        FloatType CaSS[4];
        FloatType CaJSR;
        FloatType JLCC[4];
        FloatType Jrel[4];
        FloatType Jxfer[4];
        FloatType Jiss[4];
        FloatType Jtr;

        FloatType LCC_rates[3*4]; // 3 rates to track at most
        FloatType LCC_activation_rates[4];
        FloatType RyR_rates[12*4];
        FloatType ClCh_rates[4];
        FloatType subunit_rates[4];

        void copy_from_CRUState(const CRUState<FloatType> &cru_state, const NDArray<FloatType,2> &JLCC, const int idx, const Parameters<FloatType> &params);
    };

    /* Perform the SSA on a single CRU (with fixed global variables) over an interval of length dt */
    template <typename FloatType>
    inline void SSA_single_CRU(CRUStateThread<FloatType> &state, const FloatType Cai, const FloatType CaNSR, const FloatType dt, const Parameters<FloatType> &params, const Constants<FloatType> &consts);

    /* sample a new state based on the rates of the LCCs, RyRs and ClChs */
    template <typename FloatType>
    inline void sample_new_state(CRUStateThread<FloatType> &state, const Parameters<FloatType> &params, const Constants<FloatType> &consts);

    /* Sample a new LCC state for the subunit given by subunit_idx. Also updates JLCC when appropriate. */
    template <typename FloatType>
    inline void sample_LCC(CRUStateThread<FloatType> &state, const FloatType sum_LCC_rates, const int subunit_idx, const Constants<FloatType> &consts);

    /* Sample a new RyR state for the subunit given by subunit_idx. Also updates Jrel. */
    template <typename FloatType>
    inline void sample_RyR(CRUStateThread<FloatType> &state, const FloatType sum_RyR_rates, const int subunit_idx, const Parameters<FloatType> &params);



    /* Do an Euler step on CaSS for a single CRU */
    template <typename FloatType>
    inline void update_CaSS(CRUStateThread<FloatType> &state, const FloatType dt, const Parameters<FloatType> &params, const Constants<FloatType> &consts);

    /* Do an Euler step on CaJSR for a single CRU */
    template <typename FloatType>
    inline void update_CaJSR(CRUStateThread<FloatType> &state, const FloatType dt, const Parameters<FloatType> &params, const Constants<FloatType> &consts){
        const FloatType betaJSR = 1.0 / (1 + consts.CSQN_const / square(params.KCSQN + state.CaJSR));
        state.CaJSR += (dt * betaJSR * (state.Jtr - consts.VSS_VJSR * (state.Jrel[0] + state.Jrel[1] + state.Jrel[2] + state.Jrel[3])));
    }
    /* Sample values of state 5 and state 6 of the RyRs if CaSS crosses appropriate threshold. Above this threshold the two states are 
    treated as equivalent so we do not track them directly. */
    template <typename FloatType>
    inline void sample_RyR56(CRUStateThread<FloatType> &state, const int idx, const Parameters<FloatType> &params);
    /* Sample values of state 3 and state 4 of the RyRs if CaSS crosses appropriate threshold. Above this threshold the two states are 
    treated as equivalent so we do not track them directly. */
    template <typename FloatType>
    inline void sample_RyR34(CRUStateThread<FloatType> &state, const int idx, const Parameters<FloatType> &params);


    /* Update the flux values (except JLCC and Jrel) of the CRUStateThread object */
    template <typename FloatType>
    void update_fluxes(CRUStateThread<FloatType> &state, const FloatType Cai, const FloatType CaNSR, const Parameters<FloatType> &params);

    /* Update the transition rates of the CRUStateThread object */
    template <typename FloatType>
    inline void update_rates(CRUStateThread<FloatType> &state, const Parameters<FloatType> &params, const Constants<FloatType> &consts);

}
#include "SSA.tpp"

#endif
