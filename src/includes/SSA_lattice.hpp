#ifndef SSA_LATTICE_H
#define SSA_LATTICE_H

#include "GW_lattice_utils.hpp"
#include "common.hpp"
//#include "ndarray.hpp"

#include <Eigen/Core>

template<typename T>
using Array2 = Eigen::Array<T,Eigen::Dynamic,4,Eigen::RowMajor>;


namespace GW { 
    
    template <typename FloatType>
    struct CRULatticeStateThread {
        int LCC;
        int LCC_inactivation;
        int RyR[6]; // Have to count states here so we include all 6 possible states
        int ClCh;

        // Cant calculate iss terms until after all CRUs are updated, so no Jiss terms here.
        // May as well track the other fluxes though, because we use operator splitting, so the fluxes can be applied in an euler step first independently of one another
        FloatType CaSS;
        FloatType CaJSR;
        FloatType JLCC;
        FloatType Jrel;
        FloatType Jxfer;
        FloatType Jtr;

        FloatType LCC_rates[3]; // 3 rates to track at most
        FloatType LCC_inactivation_rate;
        FloatType RyR_rates[12];
        FloatType ClCh_rate;

        FloatType LCC_tot_rate;
        FloatType RyR_tot_rate;

        void copy_from_CRULatticeState(const CRULatticeState<FloatType> &cru_state, const int x, const int y, const Parameters<FloatType> &params);
        
    };


    /* Perform the SSA on a single subunit (with fixed global variables) over an interval of length dt */
    template <typename FloatType, typename Generator>
    inline void SSA_single_su(CRULatticeStateThread<FloatType> &state, const FloatType Cai, const FloatType CaNSR, const FloatType time_int, const Parameters<FloatType> &params, const Constants<FloatType> &consts);

    /* sample a new state based on the rates of the LCCs, RyRs and ClChs */
    template <typename FloatType, typename Generator>
    inline void sample_new_state(CRULatticeStateThread<FloatType> &state, const FloatType total_rate, const Parameters<FloatType> &params, const Constants<FloatType> &consts);

    /* Sample a new LCC state for the subunit given by subunit_idx. Also updates JLCC when appropriate. */
    template <typename FloatType, typename Generator>
    inline void sample_LCC(CRULatticeStateThread<FloatType> &state, const Constants<FloatType> &consts);

    /* Sample a new RyR state for the subunit given by subunit_idx. Also updates Jrel. */
    template <typename FloatType, typename Generator>
    inline void sample_RyR(CRULatticeStateThread<FloatType> &state, const Parameters<FloatType> &params);

    template <typename FloatType>
    inline void update_LCC_rates(CRULatticeStateThread<FloatType> &state, const int new_state, const Constants<FloatType> &consts);
    

    /* Sample values of state 5 and state 6 of the RyRs if CaSS crosses appropriate threshold. Above this threshold the two states are 
    treated as equivalent so we do not track them directly. */
    template <typename FloatType, typename Generator>
    inline void sample_RyR56(CRULatticeStateThread<FloatType> &state, const Parameters<FloatType> &params);
    /* Sample values of state 3 and state 4 of the RyRs if CaSS crosses appropriate threshold. Above this threshold the two states are 
    treated as equivalent so we do not track them directly. */
    template <typename FloatType, typename Generator>
    inline void sample_RyR34(CRULatticeStateThread<FloatType> &state, const Parameters<FloatType> &params);


    /*  Calculate the flux values (except JLCC and Jrel) of the CRULatticeStateThread object */
    template <typename FloatType>
    void calculate_fluxes(CRULatticeStateThread<FloatType> &state, const FloatType Cai, const FloatType CaNSR, const Parameters<FloatType> &params);

    /* Update the transition rates of the CRULatticeStateThread object */
    template <typename FloatType>
    inline void init_rates(CRULatticeStateThread<FloatType> &state, const Parameters<FloatType> &params, const Constants<FloatType> &consts);

    template <typename FloatType>
    inline void partial_euler_step(CRULatticeStateThread<FloatType> &state, const FloatType dt, const Parameters<FloatType> &params, const Constants<FloatType> &consts);


}

#include "SSA_lattice.tpp"

#endif