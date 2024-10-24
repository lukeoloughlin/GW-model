#pragma once

#include <omp.h>
#include <functional>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <pybind11/pybind11.h>

#include "SSA_lattice.hpp"
#include "GW_lattice_utils.hpp"
#include "../common.hpp"

template<typename T>
using Array1 = Eigen::Array<T,1,Eigen::Dynamic,Eigen::RowMajor>;
template<typename T>
using Array2L = Eigen::Array<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

using QKrMap = Eigen::Map<Eigen::Array<double,5,5>,Eigen::RowMajor>;
using QKvMap = Eigen::Map<Eigen::Array<double,10,10>,Eigen::RowMajor>;

using npArray1d = Eigen::RowVectorXd;
using npArray2d = Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using npArray2i = Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using npArray3d = Eigen::Tensor<double,3,Eigen::RowMajor>;
using npArray3i = Eigen::Tensor<int,3,Eigen::RowMajor>;
using npArray4i = Eigen::Tensor<int,4,Eigen::RowMajor>;

namespace py = pybind11;


void init_GWLattice(py::module& m);

namespace GW_lattice {
    class GW_lattice;

    class PyGWLatticeSimulation {
    public:
        int nCRU_x;
        int nCRU_y;
        double tspan;

        npArray1d t;
        npArray1d V;
        npArray1d m;
        npArray1d h;
        npArray1d j;
        npArray1d Nai;
        npArray1d Ki;
        npArray1d xKs;
        npArray2d XKr;
        npArray2d XKv14;
        npArray2d XKv43;
        npArray3d Cai;
        npArray3d CaNSR;
        npArray3d CaLTRPN;
        npArray3d CaHTRPN;
        npArray3d CaJSR;
        npArray3d CaSS;
        npArray3i LCC;
        npArray3i LCC_inactivation;
        npArray4i RyR;
        npArray3i ClCh;
        
        PyGWLatticeSimulation(int nCRU_x_, int nCRU_y_, int num_step, double t_) : nCRU_x(nCRU_x_), nCRU_y(nCRU_y_), tspan(t_), t(num_step), V(num_step), 
                                                                            m(num_step), h(num_step), j(num_step), Nai(num_step), Ki(num_step), 
                                                                            xKs(num_step), XKr(num_step,5), XKv14(num_step,10), XKv43(num_step,10), 
                                                                            Cai(num_step,nCRU_x_,nCRU_y_), CaNSR(num_step,nCRU_x_,nCRU_y_), CaLTRPN(num_step,nCRU_x_,nCRU_y_),
                                                                            CaHTRPN(num_step,nCRU_x_,nCRU_y_), CaJSR(num_step,nCRU_x, nCRU_y), CaSS(num_step,nCRU_x,nCRU_y), 
                                                                            LCC(num_step,nCRU_x,nCRU_y), LCC_inactivation(num_step,nCRU_x,nCRU_y), 
                                                                            RyR(num_step,nCRU_x,nCRU_y,6), ClCh(num_step,nCRU_x,nCRU_y) { }
        
        void record_state(const GW_lattice& model, const int idx, const int nCRU_x, const int nCRU_y, const double t_);
    };
    
    struct PyInitGWLatticeState {
        double V = -91.382;
        double Nai = 10.0;
        double Ki = 131.84;
        double m = 5.33837e-4;
        double h = 0.996345;
        double j = 0.997315;
        double xKs = 2.04171e-4;
        Eigen::Matrix<double,1,5,Eigen::RowMajor> XKr;
        Eigen::Matrix<double,1,10,Eigen::RowMajor> XKv14;
        Eigen::Matrix<double,1,10,Eigen::RowMajor> XKv43;
        
        npArray2d Cai;
        npArray2d CaNSR;
        npArray2d CaLTRPN;
        npArray2d CaHTRPN;
        npArray2d CaSS;
        npArray1d CaJSR;
        npArray2i LCC;
        npArray2i LCC_inactivation;
        npArray3i RyR;
        npArray2i ClCh;

        PyInitGWLatticeState(int nCRU_x, int nCRU_y) : Cai(nCRU_x,nCRU_y), CaNSR(nCRU_x,nCRU_y), CaLTRPN(nCRU_x,nCRU_y), CaHTRPN(nCRU_x,nCRU_y), 
                                                       CaSS(nCRU_x,nCRU_y), CaJSR(nCRU_x,nCRU_y), LCC(nCRU_x,nCRU_y), LCC_inactivation(nCRU_x,nCRU_y), 
                                                       RyR(nCRU_x,nCRU_y,6), ClCh(nCRU_x,nCRU_y) {
            XKr(0) = 0.999503;
            XKr(1) = 4.13720e-4;
            XKr(2) = 7.27568e-5; 
            XKr(3) = 8.73984e-6; 
            XKr(4) = 1.36159e-6;

            XKv14(0) = 0.722328;
            XKv14(1) = 0.101971; 
            XKv14(2) = 0.00539932; 
            XKv14(3) = 1.27081e-4; 
            XKv14(4) = 1.82742e-6; 
            XKv14(5) = 0.152769; 
            XKv14(6) = 0.00962328; 
            XKv14(7) = 0.00439043; 
            XKv14(8) = 0.00195348; 
            XKv14(9) = 0.00143629;
            
            XKv43(0) = 0.953060; 
            XKv43(1) = 0.0253906; 
            XKv43(2) = 2.53848e-4; 
            XKv43(3) = 1.12796e-6; 
            XKv43(4) = 1.87950e-9; 
            XKv43(5) = 0.0151370; 
            XKv43(6) = 0.00517622; 
            XKv43(7) = 8.96600e-4; 
            XKv43(8) = 8.17569e-5; 
            XKv43(9) = 2.24032e-6;
        }
    };


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
        GW_lattice(int x, int y) : parameters(), 
                                   globals(), 
                                   CRU_lattice(x, y), 
                                   LCC_tmp(x,y), 
                                   LCC_inactivation_tmp(x,y), 
                                   RyR_tmp(x,y,6), 
                                   ClCh_tmp(x,y), 
                                   CaSS_tmp(x,y), 
                                   nCRU_x(x), 
                                   nCRU_y(y), 
                                   consts(parameters, x, y), 
                                   JLCC(x,y), 
                                   Jxfer(x,y), 
                                   Jrel(x,y), 
                                   Jtr(x,y), 
                                   Jcyto(x,y), 
                                   JNSR(x,y), 
                                   Jup(x,y), 
                                   betaSS(x,y), 
                                   betaJSR(x,y), 
                                   beta_cyto(x,y), 
                                   dCaLTRPN(x,y), 
                                   dCaHTRPN(x,y), 
                                   QKr(QKr_storage), 
                                   QKv14(QKv14_storage), 
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

        GW_lattice(const Parameters& params, int x, int y) : parameters(params), 
                                                             globals(), 
                                                             CRU_lattice(x,y), 
                                                             LCC_tmp(x,y), 
                                                             LCC_inactivation_tmp(x,y), 
                                                             RyR_tmp(x,y,6), 
                                                             ClCh_tmp(x,y), 
                                                             CaSS_tmp(x,y), 
                                                             nCRU_x(x), 
                                                             nCRU_y(y), 
                                                             consts(parameters, x, y), 
                                                             JLCC(x,y), 
                                                             Jxfer(x,y), 
                                                             Jrel(x,y), 
                                                             Jtr(x,y), 
                                                             Jcyto(x,y), 
                                                             JNSR(x,y), 
                                                             Jup(x,y), 
                                                             betaSS(x,y), 
                                                             betaJSR(x,y), 
                                                             beta_cyto(x,y), 
                                                             dCaLTRPN(x,y), 
                                                             dCaHTRPN(x,y), 
                                                             QKr(QKr_storage), 
                                                             QKv14(QKv14_storage), 
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

        void init_from_python(PyInitGWLatticeState& py_state);

        template <typename PRNG>
        void euler_step(const double dt);

        template <typename PRNG>
        PyGWLatticeSimulation run_sim(const double dt, const int num_steps, const std::function<double(double)>& Is, const int record_every);
    };

    template <typename PRNG>
    void GW_lattice::SSA(const double dt){
        consts.alphaLCC = common::alphaLCC(globals.V);
        consts.betaLCC = common::betaLCC(globals.V);
        consts.yinfLCC = common::yinfLCC(globals.V);
        consts.tauLCC = common::tauLCC(globals.V);

        #pragma omp parallel
        {
            CRULatticeStateThread temp;
            
            #pragma omp for collapse(2) schedule( static )
            for (int i = 0; i < nCRU_x; i++){
                for (int j = 0; j < nCRU_y; j++){
                    temp.copy_from_CRULatticeState(CRU_lattice, CaSS_tmp(i,j), i, j, parameters);
                    SSA_single_su<PRNG>(temp, dt, parameters, consts);
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
        
    template <typename PRNG>
    PyGWLatticeSimulation GW_lattice::run_sim(const double dt, const int num_steps, const std::function<double(double)>& Is, const int record_every){
        PyGWLatticeSimulation out(nCRU_x, nCRU_y, num_steps / record_every, num_steps*dt);
        double t = 0.0;
        int counter = 0;

        for (int i = 0; i < num_steps; ++i){
            Istim = Is(t);
            euler_step<PRNG>(dt);
            t += dt;
            if (i % record_every == 0){
                out.record_state(*this, counter, nCRU_x, nCRU_y, t);
                ++counter;
            }
        }
        return out;
    }

}
    




