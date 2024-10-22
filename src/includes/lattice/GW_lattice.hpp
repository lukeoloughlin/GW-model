#ifndef GW_LATTICE_H
#define GW_LATTICE_H


#include "GW.hpp"
#include "SSA_lattice.hpp"
#include "GW_lattice_utils.hpp"
#include <Eigen/Core>

template<typename T>
using Array1 = Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>;
template<typename T>
using Array2L = Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

template<typename T>
using QKrMap = Eigen::Map<Eigen::Array<T,5,5>,Eigen::RowMajor>;
template<typename T>
using QKvMap = Eigen::Map<Eigen::Array<T,10,10>,Eigen::RowMajor>;


namespace GW_lattice {



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
        GW::GlobalState<FloatType> globals;
        CRULatticeState<FloatType> CRU_lattice; // Replace the usual Array of CRUs with a lattice of subunits
        
        //FloatType CaSS_mean;
        //FloatType dCaSS_mean;
        
        FloatType Istim = 0;

        // Hold the SSA results while ODE updates are applied
        Array2L<int> LCC_tmp;
        Array2L<int> LCC_inactivation_tmp;
        Array3Container<int> RyR_tmp;
        Array2L<int> ClCh_tmp;
        Array2L<FloatType> CaSS_tmp;

        int nCRU_x;
        int nCRU_y;
        Constants<FloatType> consts; // Will probably need to adjust to account for different constants showing up
        Array2L<FloatType> JLCC;
        Array2L<FloatType> Jxfer;
        Array2L<FloatType> Jrel;
        Array2L<FloatType> Jtr; // This is a 2D array now since the JSR concentrations are configured on a lattice
        Array2L<FloatType> Jcyto; // The Laplacian term for the cyto subspace
        Array2L<FloatType> JNSR; // The Laplacian term for the NSR subspace
        Array2L<FloatType> Jup;
        Array2L<FloatType> betaSS;
        Array2L<FloatType> betaJSR;
        Array2L<FloatType> beta_cyto;

        Array2L<FloatType> dCaLTRPN;
        Array2L<FloatType> dCaHTRPN;

        FloatType Cai_tot;
    private:
        FloatType QKr_storage[5*5] = {0};
        FloatType QKv14_storage[10*10] = {0};
        FloatType QKv43_storage[10*10] = {0};
        QKrMap<FloatType> QKr;
        QKvMap<FloatType> QKv14;
        QKvMap<FloatType> QKv43;

        //GW::Currents<FloatType> currents;
        //GW::GlobalState<FloatType> dGlobals;
    
        //inline void update_Jxfer(){ Jxfer = parameters.rxfer * (CRU_lattice.CaSS - globals.Cai); }
        //inline void initialise_Jtr(){ Jtr = parameters.rtr * (globals.CaNSR - CRU_lattice.CaJSR); }
        inline void initialise_QKr(){ QKr(1,2) = parameters.Kf; QKr(2,1) = parameters.Kb; }
        //inline void initialise_JLCC();
        
        inline void update_QKr();
        inline void update_QKv();

        //inline void update_V_and_concentration_derivatives(const FloatType dt);
        inline void euler_reaction_step(const FloatType dt);

        void update_fluxes(); // Update non-diffusion flux terms
        void update_diffusion_fluxes(); // Update diffusion flux terms
        inline void euler_diffusion_step(const FloatType dt); // Apply diffusion update half step

        template <typename PRNG>
        void SSA(const FloatType dt);

        /* Record the values of the CRUStateThread temp back to the CRUState state for CRU i */
        inline void update_CRUstate_from_temp(const CRULatticeStateThread<FloatType> &temp, const int x, const int y);
        
        
    public:
        GW_lattice(int x, int y) : parameters(), globals(), CRU_lattice(x, y), LCC_tmp(x,y), LCC_inactivation_tmp(x,y), RyR_tmp(x,y,6), ClCh_tmp(x,y), CaSS_tmp(x,y), nCRU_x(x), nCRU_y(y), consts(parameters, x, y), JLCC(x,y), 
                                       Jxfer(x,y), Jrel(x,y), Jtr(x,y), Jcyto(x,y), JNSR(x,y), Jup(x,y), betaSS(x,y), betaJSR(x,y), beta_cyto(x,y), dCaLTRPN(x,y), dCaHTRPN(x,y), 
                                       QKr(QKr_storage), QKv14(QKv14_storage), QKv43(QKv43_storage) { 
            //consts.VF_RT = globals.V * consts.F_RT;
            //consts.JLCC_exp = exp(2*consts.VF_RT);
            initialise_QKr();
            RyR_tmp.set(CRU_lattice.RyR);
            for (int i = 0; i < nCRU_x; ++i){
                for (int j = 0; j < nCRU_y; ++j){
                    LCC_tmp(i,j) = CRU_lattice.LCC(i,j);
                    LCC_inactivation_tmp(i,j) = CRU_lattice.LCC_inactivation(i,j);
                    ClCh_tmp(i,j) = CRU_lattice.ClCh(i,j);
                    CaSS_tmp(i,j) = CRU_lattice.CaSS(i,j);
                }
            }
            Cai_tot = CRU_lattice.Cai.sum() / (nCRU_x * nCRU_y);
            //initialise_JLCC();
            //initialise_Jxfer();
            //initialise_Jtr();
        }

        GW_lattice(const Parameters<FloatType>& params, int x, int y) : parameters(params), globals(), CRU_lattice(x,y), LCC_tmp(x,y), CaSS_tmp(x,y), LCC_inactivation_tmp(x,y), RyR_tmp(x,y,6), ClCh_tmp(x,y), nCRU_x(x), 
                                       nCRU_y(y), consts(parameters, x, y), JLCC(x,y), Jxfer(x,y), Jrel(x,y), Jtr(x,y), Jcyto(x,y), JNSR(x,y), Jup(x,y), betaSS(x,y), betaJSR(x,y), 
                                       dCaLTRPN(x,y), dCaHTRPN(x,y), beta_cyto(x,y), QKr(QKr_storage), QKv14(QKv14_storage), 
                                       QKv43(QKv43_storage) { 
            initialise_QKr();
            RyR_tmp.set(CRU_lattice.RyR);
            for (int i = 0; i < nCRU_x; ++i){
                for (int j = 0; j < nCRU_y; ++j){
                    LCC_tmp(i,j) = CRU_lattice.LCC(i,j);
                    LCC_inactivation_tmp(i,j) = CRU_lattice.LCC_inactivation(i,j);
                    ClCh_tmp(i,j) = CRU_lattice.ClCh(i,j);
                    CaSS_tmp(i,j) = CRU_lattice.CaSS(i,j);
                }
            }
            Cai_tot = CRU_lattice.Cai.sum() / (nCRU_x * nCRU_y);
        }

        void set_initial_value(GW::GlobalState<FloatType>& global_vals, CRULatticeState<FloatType>& cru_vals);

        //int get_nCRU() const { return nCRU; }

        template <typename PRNG>
        void euler_step(const FloatType dt);

        template <typename PRNG>
        void euler(const FloatType dt, const int nstep, const std::function<FloatType(FloatType)>& Is);
    

};

}
    
#include "GW_lattice.tpp"
#endif



