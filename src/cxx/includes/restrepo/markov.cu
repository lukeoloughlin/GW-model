#include <math.h>
#include <curand.h>

#include "markov.cuh"



__device__ void update_RyR_rates(float* RyR_rates, const float* const RyR, const float cp, const float Ku, const float Kb, 
                                 const float _1_tau_u, const float _1_tau_b, const float _1_tau_c, const float BCSQN, 
                                 const float rho_inf, const float K, const int grid_idx){
    const float log_cp_K = logf(cp[grid_idx]) - logf(K);
    const float hill_fn = rho_inf / (1.0f + expf(23*log_cp_K));
    const float Mhat = (sqrtf(1.0f + 8*hill_fn*BCSQN) - 1.0f) / (4*hill_fn*BCSQN);

    const float k12 = Ku * cp * cp; // k12
    //const float k21 = _1_tau_c; // k21
    const float k23 = Mhat * _1_tau_b; // k23

    //const float k34 = _1_tau_c; // k34
    const float k43 = Kb * cp * cp; // k43
    //const float k41 = _1_tau_u; // k41
    const float k32 = _1_tau_u * k12 / k43; // k32 = k41 * k12 / k43

    RyR_rates[grid_idx] = k12 * RyR[grid_idx]; // 1 -> 2
    RyR_rates[grid_idx+1] = _1_tau_c * RyR[grid_idx+1]; // 2 -> 1; k21 = _1_tau_c
    RyR_rates[grid_idx+2] = k23 * RyR[grid_idx+1]; // 2 -> 3
    RyR_rates[grid_idx+3] = k32 * RyR[grid_idx+2]; // 3 -> 2
    RyR_rates[grid_idx+4] = _1_tau_c * RyR[grid_idx+2]; // 3 -> 4; k34 = _1_tau_c
    RyR_rates[grid_idx+5] = k43 * RyR[grid_idx+3]; // 4 -> 3
    RyR_rates[grid_idx+6] = _1_tau_u * RyR[grid_idx+3]; // 4 -> 1; k41 = _1_tau_u
    RyR_rates[grid_idx+7] = k23 * RyR[grid_idx]; // 1-> 4; k14 = k23
}

__device__ void update_RyR_diffusion(float* RyR, float* RyR_tmp, const float* const RyR_rates, const float* const dW, const float eps, const float dt, const int grid_idx){
    const float drift1 = RyR_rates[grid_idx+1] + RyR_rates[grid_idx+6] - (RyR_rates[grid_idx] + RyR_rates[grid_idx+7]); // q21 + q41 - (q12 + q14)
    const float drift2 = RyR_rates[grid_idx] + RyR_rates[grid_idx+3] - (RyR_rates[grid_idx+1] + RyR_rates[grid_idx+2]); // q12 + q32 - (q21 + q23)
    const float drift3 = RyR_rates[grid_idx+2] + RyR_rates[grid_idx+5] - (RyR_rates[grid_idx+3] + RyR_rates[grid_idx+4]); // q23 + q43 - (q32 + q34)

    const float sigma12 = eps * sqrtf(RyR_rates[grid_idx] + RyR_rates[grid_idx+1]);
    const float sigma23 = eps * sqrtf(RyR_rates[grid_idx+2] + RyR_rates[grid_idx+3]);
    const float sigma34 = eps * sqrtf(RyR_rates[grid_idx+4] + RyR_rates[grid_idx+5]);
    const float sigma14 = eps * sqrtf(RyR_rates[grid_idx+6] + RyR_rates[grid_idx+7]);

    RyR[grid_idx] += (dt * drift1 + sigma12 * dW[grid_idx] + sigma14 * dW[grid_idx+3]);
    RyR[grid_idx+1] += (dt * drift2 - sigma12 * dW[grid_idx] + sigma23 * dW[grid_idx+1]);
    RyR[grid_idx+2] += (dt * drift3 - sigma23 * dW[grid_idx+1] + sigma34 * dW[grid_idx+2]);
    RyR[grid_idx+3] = 1.0f - (RyR_tmp[grid_idx] + RyR_tmp[grid_idx+1] + RyR_tmp[grid_idx+2]);

    RyR_orth_proj_simplex(RyR_tmp, RyR_tmp, grid_idx);
}

__device__ void RyR_orth_proj_simplex(float* RyR, float* sorted, const int grid_idx){
    // Copy the values and sort using bubble sort
    sorted[grid_idx] = RyR[grid_idx];
    sorted[grid_idx+1] = RyR[grid_idx+1];
    sorted[grid_idx+2] = RyR[grid_idx+2];
    sorted[grid_idx+3] = RyR[grid_idx+3];

    bool swapped;
    for (int i = 0; i < 3; ++i) {
        swapped = false;
        for (int j = 0; j < 3 - i; ++j) {
            if (sorted[grid_idx+j] > sorted[grid_idx+j+1]) {
                swap(sorted[grid_idx+j], sorted[grid_idx+j+1]);
                swapped = true;
            }
        }
      
        // If no two elements were swapped, then break
        if (!swapped)
            break;
    }


    float lambda;
    if ((sorted[grid_idx] + sorted[grid_idx+1] + sorted[grid_idx+2] + sorted[grid_idx+3]) - 4*sorted[grid_idx] < 1)
        lambda = (sorted[grid_idx] + sorted[grid_idx+1] + sorted[grid_idx+2] + sorted[grid_idx+3] - 1.0f) / 4.0f;
    else if ((sorted[grid_idx+1] + sorted[grid_idx+2] + sorted[grid_idx+3]) - 3*sorted[grid_idx+1] < 1)
        lambda = (sorted[grid_idx+1] + sorted[grid_idx+2] + sorted[grid_idx+3] - 1.0f) / 3.0f;
    else if ((sorted[grid_idx+2] + sorted[grid_idx+3]) - 2*sorted[grid_idx+1] < 1)
        lambda = (sorted[grid_idx+2] + sorted[grid_idx+3] - 1.0f) / 2.0f;
    else 
        lambda = sorted[grid_idx+3] - 1.0f;

    RyR[grid_idx] = RyR[grid_idx] - lambda;
    RyR[grid_idx+1] = RyR[grid_idx+1] - lambda;
    RyR[grid_idx+2] = RyR[grid_idx+2] - lambda;
    RyR[grid_idx+3] = RyR[grid_idx+3] - lambda;

}