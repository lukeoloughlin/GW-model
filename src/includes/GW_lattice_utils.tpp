#include "GW_lattice.hpp"
    
namespace GW {

    template <typename FloatType>//, typename PRNG>
    CRULatticeState<FloatType>::CRULatticeState(const int nCRU_x, const int nCRU_y) : CaSS(nCRU_x,nCRU_x), CaJSR(nCRU_x,nCRU_y), LCC(nCRU_x,nCRU_y), LCC_inactivation(nCRU_x,nCRU_y), RyR(nCRU_x,nCRU_y,6), ClCh(nCRU_x, nCRU_y, 4) {
        CaSS.setConstant(1.45370e-4);
        CaJSR.setConstant(0.908408);

        const double LCC_weights[3] = { 0.958, 0.038, 0.004 };
        const double LCC_i_weights[2] = { 0.9425, 0.0575 };
        const double RyR_weights[3] = { 0.609, 0.5*0.391, 0.5*0.391 };
        const double ClCh_weights[2] = { 0.998, 0.002 };
        int LCC_idx, LCC_i_idx, RyR_idx, ClCh_idx;

        for (int i = 0; i < nCRU_x; i++){
            for (int j = 0; j < nCRU_y; j++){
                LCC_idx = sample_weights<double, int, std::mt19937_64>(LCC_weights, 1.0, 3);
                if (LCC_idx == 0)
                    LCC(i,j) = 1;
                else if (LCC_idx == 1)
                    LCC(i,j) = 2;
                else 
                    LCC(i,j) = 7;
                
                LCC_i_idx = sample_weights<double, int, std::mt19937>(LCC_i_weights, 1.0, 2);
                LCC_inactivation(i,j) = (LCC_i_idx == 0) ? 1 : 0;
                    
                for (int k = 0; k < 5; k++){
                    RyR_idx = sample_weights<double, int, std::mt19937>(RyR_weights, 1.0, 3);
                    if (RyR_idx == 0)
                        ++RyR.array(i,j,0);
                    else if (RyR_idx == 1)
                        ++RyR.array(i,j,4);
                    else 
                        ++RyR.array(i,j,5);
                }    
                
                ClCh_idx = sample_weights<double, int, std::mt19937_64>(ClCh_weights, 1.0, 2);
                ClCh(i,j) = ClCh_idx;
                                
            }
        }
    }
}
