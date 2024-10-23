#pragma once

#include "SSA_lattice.hpp"
#include "GW_lattice_utils.hpp"
#include <Eigen/Core>

template<typename T>
using Array1 = Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>;
template<typename T>
using Array2L = Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

using QKrMap = Eigen::Map<Eigen::Array<double,5,5>,Eigen::RowMajor>;
using QKvMap = Eigen::Map<Eigen::Array<double,10,10>,Eigen::RowMajor>;


namespace GW_lattice {


    //template <typename T>    
    class GW_lattice {
    public:
        Parameters parameters; // Will possibly need to adjust to hold a diffusion term for JSR
        GlobalState globals;
        CRULatticeState CRU_lattice; // Replace the usual Array of CRUs with a lattice of subunits
        
        double Istim = 0;

        // Hold the SSA results while ODE updates are applied
        Array2L<int> LCC_tmp;
        Array2L<int> LCC_inactivation_tmp;
        Array3Container<int> RyR_tmp;
        Array2L<int> ClCh_tmp;
        Array2L<double> CaSS_tmp;

        int nCRU_x;
        int nCRU_y;
        Constants consts; // Will probably need to adjust to account for different constants showing up
        Array2L<double> JLCC;
        Array2L<double> Jxfer;
        Array2L<double> Jrel;
        Array2L<double> Jtr; // This is a 2D array now since the JSR concentrations are configured on a lattice
        Array2L<double> Jcyto; // The Laplacian term for the cyto subspace
        Array2L<double> JNSR; // The Laplacian term for the NSR subspace
        Array2L<double> Jup;
        Array2L<double> betaSS;
        Array2L<double> betaJSR;
        Array2L<double> beta_cyto;

        Array2L<double> dCaLTRPN;
        Array2L<double> dCaHTRPN;

        double Cai_tot;
    private:
        double QKr_storage[5*5] = {0};
        double QKv14_storage[10*10] = {0};
        double QKv43_storage[10*10] = {0};
        QKrMap QKr;
        QKvMap QKv14;
        QKvMap QKv43;

        inline void initialise_QKr(){ QKr(1,2) = parameters.Kf; QKr(2,1) = parameters.Kb; }
        
        inline void update_QKr();
        inline void update_QKv();

        inline void euler_reaction_step(const double dt);

        void update_fluxes(); // Update non-diffusion flux terms
        void update_diffusion_fluxes(); // Update diffusion flux terms
        inline void euler_diffusion_step(const double dt); // Apply diffusion update half step

        template <typename PRNG>
        void SSA(const double dt);

        inline void update_CRUstate_from_temp(const CRULatticeStateThread& temp, const int x, const int y);
        
        
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

        GW_lattice(const Parameters& params, int x, int y) : parameters(params), globals(), CRU_lattice(x,y), LCC_tmp(x,y), CaSS_tmp(x,y), LCC_inactivation_tmp(x,y), RyR_tmp(x,y,6), ClCh_tmp(x,y), nCRU_x(x), 
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

        void set_initial_value(GlobalState& global_vals, CRULatticeState& cru_vals);

        //int get_nCRU() const { return nCRU; }

        template <typename PRNG>
        void euler_step(const double dt);

        //template <typename PRNG>
        //void euler(const double dt, const int nstep, const std::function<double(double)>& Is);
    

    };

    template <typename PRNG>
    void GW_lattice::SSA(const double dt){
        consts.alphaLCC = common::alphaLCC<double>(globals.V);
        consts.betaLCC = common::betaLCC<double>(globals.V);
        consts.yinfLCC = common::yinfLCC<double>(globals.V);
        consts.tauLCC = common::tauLCC<double>(globals.V);

        #pragma omp parallel
        {
            CRULatticeStateThread temp;
            
            #pragma omp for collapse(2) schedule( static )
            for (int i = 0; i < nCRU_x; i++){
                for (int j = 0; j < nCRU_y; j++){
                    temp.copy_from_CRULatticeState(CRU_lattice, CaSS_tmp(i,j), i, j, parameters);
                    SSA_single_su<PRNG>(temp, dt, parameters, consts);
                    //update_CRUstate_from_temp(temp, i, j);
                    LCC_tmp(i,j) = temp.LCC;
                    LCC_inactivation_tmp(i,j) = temp.LCC_inactivation;
                    for (int k = 0; k < 6; ++k)
                        RyR_tmp.array(i,j,k) = temp.RyR[k];            
                    ClCh_tmp(i,j) = temp.ClCh;
                    CaSS_tmp(i,j) = temp.CaSS;
                }
            }
        }
    }

    template <typename PRNG>
    void GW_lattice::euler_step(const double dt){
        SSA<PRNG>(dt); // Do SSA step and hold the Markov chain states in temporary arrays

        // First diffusion half step
        update_diffusion_fluxes(); // Update fluxes to do diffusion half step
        euler_diffusion_step(dt);

        // Reaction step
        update_fluxes();
        euler_reaction_step(dt);

        // Second diffusion half step
        update_diffusion_fluxes();
        euler_diffusion_step(dt);

        // Copy Markov Chain states into the CRU_lattice object
        // Check that this is copying correctly
        for (int i = 0; i < nCRU_x; ++i){
            for (int j = 0; j < nCRU_y; ++j){
                CRU_lattice.LCC(i,j) = LCC_tmp(i,j);
                CRU_lattice.LCC_inactivation(i,j) = LCC_inactivation_tmp(i,j);
                CRU_lattice.ClCh(i,j) = ClCh_tmp(i,j);
            }
        }
        CRU_lattice.RyR.set(RyR_tmp);
    }

}
    
//#include "GW_lattice.tpp"




