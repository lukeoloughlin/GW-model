#ifndef GW_LATTICE_H
#define GW_LATTICE_H


#include "GW.hpp"
#include <Eigen/Core>

template<typename T>
using Array1 = Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>;
template<typename T>
using Array2 = Eigen::Array<T,Eigen::Dynamic,4,Eigen::RowMajor>;

template<typename T>
using QKrMap = Eigen::Map<Eigen::Array<T,5,5>,Eigen::RowMajor>;
template<typename T>
using QKvMap = Eigen::Map<Eigen::Array<T,10,10>,Eigen::RowMajor>;


namespace GW {
    
    template <typename FloatType>//, typename PRNG>
    struct CRULatticeState {
        Array2<FloatType> CaSS;
        Array2<FloatType> CaJSR; // changing this to Array2 because CRUs are no longer distinct structures
        Array2<int> LCC;
        Array2<int> LCC_inactivation;
        Array3Container<int> RyR;
        Array2<int> ClCh;
        Array2<FloatType> RyR_open_int;
        Array2<FloatType> RyR_open_martingale;
        Array2<FloatType> RyR_open_martingale_normalised;
        //Array1<FloatType> sigma_RyR;
        
        //Array2<FloatType> LCC_open_int;
        //Array2<FloatType> LCC_open_martingale;
        //Array2<FloatType> LCC_open_martingale_normalised;
        //Array1<FloatType> sigma_LCC;

        CRULatticeState(const int nCRU_x, const int nCRU_y);
        CRULatticeState& operator=(CRULatticeState& x) = default;
    };



    // This is defined in GW.hpp
    /*
    template <typename FloatType>
    struct Currents {
        FloatType INa = 0;
        FloatType INab = 0;
        FloatType INaCa = 0;
        FloatType INaK = 0;
        FloatType IKr = 0;
        FloatType IKs = 0;
        FloatType Ito1 = 0;
        FloatType Ito2 = 0;
        FloatType IK1 = 0;
        FloatType IKp = 0;
        FloatType ICaL = 0;
        FloatType ICab = 0;
        FloatType IpCa = 0;
        FloatType Jup = 0;
        FloatType Jtr_tot = 0;
        FloatType Jxfer_tot = 0;
    };
    */


    template <typename FloatType>    
    class GW_lattice {
    public:
        Parameters<FloatType> parameters; // Will possibly need to adjust to hold a diffusion term for JSR
        GlobalState<FloatType> globals;
        CRULatticeState<FloatType> CRU_lattice; // Replace the usual Array of CRUs with a lattice of subunits
        
        FloatType CaSS_mean;
        FloatType dCaSS_mean;
        
        FloatType Istim = 0;
    private:

        int nCRU;
        Constants<FloatType> consts; // Will probably need to adjust to account for different constants showing up
        Array2<FloatType> JLCC;
        Array2<FloatType> Jxfer;
        Array2<FloatType> Jtr; // This is a 2D array now since the JSR concentrations are configured on a lattice
        Array2<FloatType> Jiss_DS; // The Laplacian term for the dyadic space
        Array2<FloatType> Jiss_JSR; // The Laplacian term for the JSRs

        FloatType QKr_storage[5*5] = {0};
        FloatType QKv14_storage[10*10] = {0};
        FloatType QKv43_storage[10*10] = {0};
        QKrMap<FloatType> QKr;
        QKvMap<FloatType> QKv14;
        QKvMap<FloatType> QKv43;

        Currents<FloatType> currents;
        GlobalState<FloatType> dGlobals;
    
        inline void initialise_Jxfer(){ Jxfer = parameters.rxfer * (CRUs.CaSS - globals.Cai); }
        inline void initialise_Jtr(){ Jtr = parameters.rtr * (globals.CaNSR - CRUs.CaJSR); }
        inline void initialise_QKr(){ QKr(1,2) = parameters.Kf; QKr(2,1) = parameters.Kb; }
        inline void initialise_JLCC();
        
        inline void update_QKr();
        inline void update_QKv();

        inline void update_V_and_concentration_derivatives(const FloatType dt);
        inline void update_gate_derivatives(const FloatType dt);
        inline void update_Kr_derivatives(const FloatType dt);
        inline void update_Kv_derivatives(const FloatType dt); // Updates both Kv14 and Kv43        

        template <typename PRNG>
        void SSA(const FloatType dt);

        /* Record the values of the CRUStateThread temp back to the CRUState state for CRU i */
        inline void update_CRUstate_from_temp(const CRUStateThread<FloatType> &temp, const int i);

        inline void update_increment_and_sigma(const FloatType dt);
        inline void update_mean_RyR_open();
        inline void update_martingale_quantities();
        
        
    public:
        GW_model(int nCRU_simulated) : parameters(), globals(), CRUs(nCRU_simulated), nCRU(nCRU_simulated), consts(parameters, nCRU), JLCC(nCRU_simulated,4), 
                                       Jxfer(nCRU_simulated,4), Jtr(nCRU_simulated), QKr(QKr_storage), QKv14(QKv14_storage), QKv43(QKv43_storage), currents(), 
                                       dGlobals(0.0) { 
            consts.VF_RT = globals.V * consts.F_RT;
            consts.JLCC_exp = exp(2*consts.VF_RT);
            initialise_JLCC();
            initialise_Jxfer();
            initialise_Jtr();
            initialise_QKr();
        }

        GW_model(const Parameters<FloatType>& params, int nCRU_simulated) : parameters(params), globals(), CRUs(nCRU_simulated), nCRU(nCRU_simulated), consts(parameters, nCRU), JLCC(nCRU_simulated,4), 
                                       Jxfer(nCRU_simulated,4), Jtr(nCRU_simulated), QKr(QKr_storage), QKv14(QKv14_storage), QKv43(QKv43_storage), currents(), 
                                       dGlobals(0.0) { 
            consts.VF_RT = globals.V * consts.F_RT;
            consts.JLCC_exp = exp(2*consts.VF_RT);
            initialise_JLCC();
            initialise_Jxfer();
            initialise_Jtr();
            initialise_QKr();
        }

        void set_initial_value(GlobalState<FloatType>& global_vals, CRUState<FloatType>& cru_vals);

        int get_nCRU() const { return nCRU; }

        template <typename PRNG>
        void euler_step(const FloatType dt);

        template <typename PRNG>
        void euler(const FloatType dt, const int nstep, const std::function<FloatType(FloatType)>& Is);
    

};

}
    
#include "GW_lattice.tpp"
#endif




