// Contains RyR and LCC implementations for Restrepo model
#ifndef MARKOV_H
#define MARKOV_H

#include <curand.h>

/* 
NOTE: I will probably use a diffusion approixmation of the RyRs since there are 100 of them. Check approximation first though 
*/
// Update the 8 element array of RyR rates. Mhat is in eq 35 of Restrepo et al.
__device__ void update_RyR_rates(float* RyR_rates, const float* const RyR, const float cp, const float Ku, const float Kb, const float _1_tau_u, 
                                 const float _1_tau_b, const float _1_tau_c, const float BCSQN, const float rho_inf, const float K, 
                                 const int grid_idx);


/* 
NOTE: There are 4 LCCs, so in line with above I will probably use tau leaping if I can't figure out a better way.
*/
// Update the 20 element array for the LCC rates. See Mahajan et al. 2008.
__device__ void update_LCC_rates(float* LCC_rates, const float alpha, const float beta, const float s1, const float s2,
                                 const float k1, const float k2, const float k3, const float k4, const float k5, const float k6,
                                 const float r1, const float r2, const int grid_idx);


__device__ void update_RyR_diffusion(float* RyR, float* RyR_tmp, float* RyR_tmp_sorted, const float* const RyR_rates, const float* const dW, const float eps, const float dt, const int grid_idx);

/* Could this work? Since there are only 4 LCCs, might be able to approximate Kolmogorov equations with an euler step, then sample
using inverse CDF method.*/
__device__ void update_LCC_exact(float* LCC, float* LCC_rates, const int grid_idx);

__device__ void RyR_orth_proj_simplex(float* RyR_tmp, float* RyR_tmp_sorted, const int grid_idx);



#endif