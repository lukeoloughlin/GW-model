#ifndef COMMON_H
#define COMMON_H

#include <cmath>
#include <iostream>
#include <random>


constexpr double FARADAY = 96.5;
constexpr double GAS_CONST = 8.314;

template <typename NumType>
inline NumType square(NumType x) { return x*x; }

template <typename NumType>
inline NumType cube(NumType x) { return x*x*x; }


template <typename PRNG>
inline PRNG& seed(){
    static thread_local PRNG gen(std::random_device{}());
    return gen;
}

template <typename FloatType, typename PRNG>
inline FloatType urand(){
    static std::uniform_real_distribution<FloatType> dist(0.0, 1.0);
    return dist(seed<PRNG>());
}

template <typename FloatType, typename PRNG>
inline FloatType nrand(){
    static std::normal_distribution<FloatType> dist(0.0, 1.0);
    return dist(seed<PRNG>());
}

template<typename FloatType, typename PRNG>
inline int sample_binomial(const FloatType p, const int N){
    int X = 0;
    FloatType u;
    for (int i = 0; i < N; i++){
        u = urand<FloatType, PRNG>();
        if (u < p)
            X++;
    }
    return X;
}

template <typename FloatType, typename IntType, typename PRNG>
inline int sample_weights(const FloatType* const weights, const FloatType total_weight, const IntType size){
    const FloatType u = urand<FloatType, PRNG>() * total_weight;
    FloatType cum_weight = weights[0];
    IntType i = 0;
    while (cum_weight < u && i < (size-1)){
        ++i;
        cum_weight += weights[i];
    }
    return i;
}


namespace common {
    /*****
    Calculate the Nernst potential using the standard expression (R*T) / (F*z) * log(Xo/Xi), where
        * Xi > 0 is the intracellular concentration of X
        * Xo > 0 is the extracellular concentration of X
        * z is the valence of X
        * RT_F = (R*T)/F
    *****/
    template <typename FloatType>
    inline FloatType Nernst(const FloatType Xi, const FloatType Xo, const FloatType RT_F, const FloatType valence){
        if (Xi <= 0.0 || Xo <= 0.0){ 
            std::cout << "Invalid arguments to Nernst. Got Xi=" << Xi << ", Xo=" << Xo << ". Exiting program." << std::endl;
            exit(EXIT_FAILURE);
        }
        return log(Xo / Xi) * RT_F / valence;
    }


    /*******
    The fast sodium current INa in the Beeler-Reuter model. Most cardiac AP models leave this unchanged.
        * V is the action potential
        * m is the activation gate
        * h is the fast inactivation gate
        * j is the slow inactivation gate
        * GNa >= 0 is the channel conductance 
        * ENa is the precalculated Nernst potential
    *******/
    template <typename FloatType>
    inline FloatType INa(const FloatType V, const FloatType m, const FloatType h, const FloatType j, const FloatType GNa, const FloatType ENa){ 
        return GNa*cube(m)*h*j*(V-ENa); 
    }

    /****** 
    Background current term of the form Gx(V - Ex) where
    * V is the action potential
    * Gx is the conductance of the channel
    * Ex is the Nernst potential of the channel
    ******/
    template <typename FloatType>
    inline FloatType Ib(const FloatType V, const FloatType Gx, const FloatType Ex){ return Gx*(V-Ex); }

    /****** 
    Common term for the sarcolemmal calcium pump Ip(Ca). 
        * Cai > 0 is the intracellular calcium concentration
        * IpCa_max > 0 is the maximum pump current
        * Km_pCa is the half saturation constant for the Sarcolemmal pump
    ******/
    template <typename FloatType>
    inline FloatType IpCa(const FloatType Cai, const FloatType IpCa_max, const FloatType Km_pCa) { return IpCa_max * (Cai / (Km_pCa + Cai)); }


    /***** 
    The term fNaK in the standard formulation for the Na-K pump.
        * VF_RT = (V*F) / (R*T)
        * sigma = (exp(Nao / 67.3) - 1) / 7 is assumed to be precalculated
        * expmVF_RT (in the second definition) = exp(-VF/RT) which will be precalculated in most models. 
    *****/
    template <typename FloatType>
    inline FloatType fNaK(const FloatType VF_RT, const FloatType sigma){ return 1.0 / (1.0 + 0.1245 * exp(-0.1*VF_RT) + 0.0365 * sigma * exp(-VF_RT)); }
    template <typename FloatType>
    inline FloatType fNaK(const FloatType VF_RT, const FloatType expmVF_RT, FloatType sigma){ 
        return 1.0 / (1.0 + 0.1245 * exp(-0.1*VF_RT) + 0.0365 * sigma * expmVF_RT); 
    }

    /***** 
    The Na-K pump current common to most models.
        * VF_RT = (V*F) / (R*T)
        * expmVF_RT (in the second definition) = exp(-VF/RT) will be precalculated for most models.
        * Nai > 0 is the intracellular calcium current.
        * sigma = (exp(Nao / 67.3) - 1) / 7 is assumed to be precalculated
        * INaK_multiplier = INaK_max * (Ko / (KmKo + Ko)) is assumed to be constant.
        * Km_Nai > 0 is the Na half saturation constant for the Na-K pump
        * expmVF_RT (in the second definition) = exp(-VF/RT) which will be precalculated in most models. 
    *****/
    template <typename FloatType>
    inline FloatType INaK(const FloatType VF_RT, const FloatType Nai, const FloatType sigma, const FloatType Km_Nai, const FloatType INaK_multiplier){
        return INaK_multiplier * fNaK(VF_RT, sigma) / (1.0 + pow(Km_Nai / Nai, 1.5));
    }
    template <typename FloatType>
    inline FloatType INaK(const FloatType VF_RT, const FloatType expmVF_RT, const FloatType Nai, const FloatType sigma, const FloatType Km_Nai, const FloatType INaK_multiplier){
        return INaK_multiplier * fNaK(VF_RT, expmVF_RT, sigma) / (1.0 + pow(Km_Nai / Nai, 1.5));
    }

    /******
     The Na-Ca exchanger current common to most models. 
        * VF_RT = V*F / (R*T)
        * expmVF_RT (in the second definition) = exp(-VF/RT) will be precalculated for most models.
        * Nai > 0 is the intracellular Na concentration
        * Cai > 0 is the intracellular Ca concentration
        * Nao3 = Nao^3, where Nao > 0 is the extracellular sodium concentration 
        * Cao > 0 is the extracellular calcium concentration
        * eta is a unitless parameter controlling voltage dependence of Na-Ca exchange
        * ksat is the Na-Ca saturation factor at negative potentials
        * INaCa_multiplier = INaCa_max / ((Km_Na^3 + Nao^3) * (Km_Ca + Cao)) is precalculated.
    *******/
    template <typename FloatType>
    inline FloatType INaCa(const FloatType VF_RT, const FloatType Nai, const FloatType Cai, const FloatType Nao3, const FloatType Cao, const FloatType eta, const FloatType ksat, const FloatType INaCa_multiplier){
        const FloatType exp_term1 = exp(eta*VF_RT);
        const FloatType exp_term2 = exp((eta-1.0)*VF_RT);
        return INaCa_multiplier * (exp_term1 * cube(Nai) * Cao - exp_term2 * Nao3 * Cai) / (1.0 + ksat * exp_term2);
    }
    template <typename FloatType>
    inline FloatType INaCa(const FloatType VF_RT, const FloatType expmVF_RT, const FloatType Nai, const FloatType Cai, const FloatType Nao3, const FloatType Cao, const FloatType eta, const FloatType ksat, const FloatType INaCa_multiplier){
        const FloatType exp_term1 = exp(eta*VF_RT);
        const FloatType exp_term2 = exp_term1 * expmVF_RT;
        return INaCa_multiplier * (exp_term1 * cube(Nai) * Cao - exp_term2 * Nao3 * Cai) / (1.0 + ksat * exp_term2);
    }


    /****** 
    Forward rate for gating variable m in the Beeler-Reuter model. Most Cardiac AP models leave this unchanged.
        * V is the action potential.  
    ******/
    template <typename FloatType>
    inline FloatType alpha_m(const FloatType V){ return (V == -47.13) ? 3.2 : 0.32 * (V + 47.13) / (1.0 - exp(-0.1 * (V + 47.13))); }

    /****** 
    Backwards rate for gating variable m in the Beeler-Reuter model. Most Cardiac AP models leave this unchanged.
        * V is the action potential.  
    ******/
    template <typename FloatType>
    inline FloatType beta_m(const FloatType V){ return 0.08 * exp(-V / 11.0); }

    /****** 
    Forward rate for gating variable h in the Beeler-Reuter model. Most Cardiac AP models leave this unchanged.
        * V is the action potential.  
    ******/
    template <typename FloatType>
    inline FloatType alpha_h(const FloatType V){ return (V >= -40.0) ? 0.0 : 0.135 * exp(-(V + 80.0) / 6.8); }

    /****** 
    Backwards rate for gating variable h in the Beeler-Reuter model. Most Cardiac AP models leave this unchanged.
        * V is the action potential.  
    ******/
    template <typename FloatType>
    inline FloatType beta_h(const FloatType V){ return (V >= -40.0) ?  1.0 / (0.13 * (1.0 + exp(-(V + 10.66) / 11.1)))
                                                                    : 3.56 * exp(0.079 * V) + 3.1e5 * exp(0.35 * V); }


    /****** 
    Forward rate for gating variable j in the Beeler-Reuter model. Most Cardiac AP models leave this unchanged.
        * V is the action potential.  
    ******/
    template <typename FloatType>
    inline FloatType alpha_j(const FloatType V){
        return (V >= -40.0) ? 0.0 
                            : (-127140.0*exp(0.2444*V) - 3.474e-5 * exp(-0.04391 * V)) * (V + 37.78) / (1.0 + exp(0.311 * (V + 79.23)));
    }


    /****** 
    Backwards rate for gating variable j in the Beeler-Reuter model. Most Cardiac AP models leave this unchanged.
        * V is the action potential.  
    ******/
    template <typename FloatType>
    inline FloatType beta_j(const FloatType V){ return (V >= -40.0) ? 0.3 * exp(-2.535e-7 * V) / (1.0 + exp(-0.1 * (V + 32.0))) 
                                                                    : 0.1212 * exp(-0.01052 * V) / (1.0 +  exp(-0.1378 * (V + 40.14))); }

}


#endif