#pragma once

#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <unsupported/Eigen/CXX11/Tensor>


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

template <typename T, typename PRNG>
inline T urand(){
    static std::uniform_real_distribution<T> dist(0.0, 1.0);
    return dist(seed<PRNG>());
}

template <typename T, typename PRNG>
inline T nrand(){
    static std::normal_distribution<T> dist{0.0, 1.0};
    return dist(seed<PRNG>());
}

template<typename T, typename PRNG>
inline int sample_binomial(const T p, const int N){
    int X = 0;
    T u;
    for (int i = 0; i < N; i++){
        u = urand<T, PRNG>();
        if (u < p)
            X++;
    }
    return X;
}

/*
Sample from an unnormalised probability distribution, where weights are the unnormalised probabilities and total_weight is the normalising constant.
Useful when sampling new state in the Gillespie algorithm, where weights are the transition rates and total_weight is the jump rate.
*/
template <typename T, typename SizeType, typename PRNG>
inline int sample_weights(const T* const weights, const T total_weight, const SizeType size){
    const T u = urand<T, PRNG>() * total_weight;
    T cum_weight = weights[0];
    SizeType i = 0;
    while (cum_weight < u && i < (size-1)){
        ++i;
        cum_weight += weights[i];
    }
    return i;
}

template<typename T>
class Array3Container {
private:
    std::vector<T> storage;
public:
    Eigen::TensorMap<Eigen::Tensor<T,3,Eigen::RowMajor>> array;
    Array3Container(int n1, int n2, int n3) : storage(n1*n2*n3), array(storage.data(),n1,n2,n3) { }

    // Set the values of storage to that of other
    void set(Array3Container &other){
        for (int i; i < storage.size(); ++i) { storage[i] = other.storage[i]; }
    }

};


namespace common {
    /*****
    Calculate the Nernst potential using the standard expression (R*T) / (F*z) * log(Xo/Xi), where
        * Xi > 0 is the intracellular concentration of X
        * Xo > 0 is the extracellular concentration of X
        * z is the valence of X
        * RT_F = (R*T)/F
    *****/
    template <typename T>
    inline T Nernst(const T Xi, const T Xo, const T RT_F, const T valence){
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
    template <typename T>
    inline T INa(const T V, const T m, const T h, const T j, const T GNa, const T ENa){ 
        return GNa*cube(m)*h*j*(V-ENa); 
    }

    /****** 
    Background current term of the form Gx(V - Ex) where
    * V is the action potential
    * Gx is the conductance of the channel
    * Ex is the Nernst potential of the channel
    ******/
    template <typename T>
    inline T Ib(const T V, const T Gx, const T Ex){ return Gx*(V-Ex); }

    /****** 
    Common term for the sarcolemmal calcium pump Ip(Ca). 
        * Cai > 0 is the intracellular calcium concentration
        * IpCa_max > 0 is the maximum pump current
        * Km_pCa is the half saturation constant for the Sarcolemmal pump
    ******/
    template <typename T>
    inline T IpCa(const T Cai, const T IpCa_max, const T Km_pCa) { return IpCa_max * (Cai / (Km_pCa + Cai)); }


    /***** 
    The term fNaK in the standard formulation for the Na-K pump.
        * VF_RT = (V*F) / (R*T)
        * sigma = (exp(Nao / 67.3) - 1) / 7 is assumed to be precalculated
        * expmVF_RT (in the second definition) = exp(-VF/RT) which will be precalculated in most models. 
    *****/
    template <typename T>
    inline T fNaK(const T VF_RT, const T sigma){ return 1.0 / (1.0 + 0.1245 * exp(-0.1*VF_RT) + 0.0365 * sigma * exp(-VF_RT)); }
    template <typename T>
    inline T fNaK(const T VF_RT, const T expmVF_RT, T sigma){ 
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
    template <typename T>
    inline T INaK(const T VF_RT, const T Nai, const T sigma, const T Km_Nai, const T INaK_multiplier){
        return INaK_multiplier * fNaK(VF_RT, sigma) / (1.0 + pow(Km_Nai / Nai, 1.5));
    }
    template <typename T>
    inline T INaK(const T VF_RT, const T expmVF_RT, const T Nai, const T sigma, const T Km_Nai, const T INaK_multiplier){
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
    template <typename T>
    inline T INaCa(const T VF_RT, const T Nai, const T Cai, const T Nao3, const T Cao, const T eta, const T ksat, const T INaCa_multiplier){
        const T exp_term1 = exp(eta*VF_RT);
        const T exp_term2 = exp((eta-1.0)*VF_RT);
        return INaCa_multiplier * (exp_term1 * cube(Nai) * Cao - exp_term2 * Nao3 * Cai) / (1.0 + ksat * exp_term2);
    }
    template <typename T>
    inline T INaCa(const T VF_RT, const T expmVF_RT, const T Nai, const T Cai, const T Nao3, const T Cao, const T eta, const T ksat, const T INaCa_multiplier){
        const T exp_term1 = exp(eta*VF_RT);
        const T exp_term2 = exp_term1 * expmVF_RT;
        return INaCa_multiplier * (exp_term1 * cube(Nai) * Cao - exp_term2 * Nao3 * Cai) / (1.0 + ksat * exp_term2);
    }


    /****** 
    Forward rate for gating variable m in the Beeler-Reuter model. Most Cardiac AP models leave this unchanged.
        * V is the action potential.  
    ******/
    template <typename T>
    inline T alpha_m(const T V){ return (V == -47.13) ? 3.2 : 0.32 * (V + 47.13) / (1.0 - exp(-0.1 * (V + 47.13))); }

    /****** 
    Backwards rate for gating variable m in the Beeler-Reuter model. Most Cardiac AP models leave this unchanged.
        * V is the action potential.  
    ******/
    template <typename T>
    inline T beta_m(const T V){ return 0.08 * exp(-V / 11.0); }

    /****** 
    Forward rate for gating variable h in the Beeler-Reuter model. Most Cardiac AP models leave this unchanged.
        * V is the action potential.  
    ******/
    template <typename T>
    inline T alpha_h(const T V){ return (V >= -40.0) ? 0.0 : 0.135 * exp(-(V + 80.0) / 6.8); }

    /****** 
    Backwards rate for gating variable h in the Beeler-Reuter model. Most Cardiac AP models leave this unchanged.
        * V is the action potential.  
    ******/
    template <typename T>
    inline T beta_h(const T V){ return (V >= -40.0) ?  1.0 / (0.13 * (1.0 + exp(-(V + 10.66) / 11.1)))
                                                                    : 3.56 * exp(0.079 * V) + 3.1e5 * exp(0.35 * V); }


    /****** 
    Forward rate for gating variable j in the Beeler-Reuter model. Most Cardiac AP models leave this unchanged.
        * V is the action potential.  
    ******/
    template <typename T>
    inline T alpha_j(const T V){
        return (V >= -40.0) ? 0.0 
                            : (-127140.0*exp(0.2444*V) - 3.474e-5 * exp(-0.04391 * V)) * (V + 37.78) / (1.0 + exp(0.311 * (V + 79.23)));
    }


    /****** 
    Backwards rate for gating variable j in the Beeler-Reuter model. Most Cardiac AP models leave this unchanged.
        * V is the action potential.  
    ******/
    template <typename T>
    inline T beta_j(const T V){ return (V >= -40.0) ? 0.3 * exp(-2.535e-7 * V) / (1.0 + exp(-0.1 * (V + 32.0))) 
                                                                    : 0.1212 * exp(-0.01052 * V) / (1.0 +  exp(-0.1378 * (V + 40.14))); }
    
    inline double dTRPNCa(const double TRPNCa, const double Cai, const double TRPNtot, const double kTRPNp, const double kTRPNm){
        return kTRPNp*Cai*(TRPNtot - TRPNCa) - kTRPNm*TRPNCa;
    }


    //template <typename T>
    inline double Jup(const double Cai, const double CaNSR, const double Vmaxf, const double Vmaxr, const double Kmf, const double Kmr, const double Hf, const double Hr){
        const double f = pow(Cai/Kmf, Hf);
        const double r = pow(CaNSR/Kmr, Hr);
        return (Vmaxf*f - Vmaxr*r) / (1.0 + f + r);
    }
    
    //template <typename T>
    inline double RKr(const double V){ return 1.0 / (1.0 + 1.4945*exp(0.0446*V)); }

    //template <typename T>
    inline double IKr(const double V, const double XKr, const double EK, const double GKr, const double sqrtKo){
        return GKr * sqrtKo * RKr(V) * XKr * (V - EK) * 0.5;
    }


    //template <typename T>
    inline double EKs(const double Ki, const double Ko, const double Nai, const double Nao, const double RT_F){
        return log((Ko+0.01833*Nao) / (Ki + 0.01833*Nai)) * RT_F;
    }


    //template <typename T>
    inline double IKs(const double V, const double XKs, const double Ki, const double Nai, const double Nao, const double Ko, const double GKs, const double RT_F){
        return GKs * square(XKs) * (V - EKs(Ki, Ko, Nai, Nao, RT_F));
    }


    //template <typename T>
    inline double IKv43(const double V, const double XKv43, const double EK, const double GKv43){
        return GKv43 * XKv43 * (V - EK);
    }

    //template <typename T>
    inline double IKv14(const double VFRT, const double exp_term, const double XKv14, const double Ki, const double Nai, const double PKv14_Csc, const double Nao, const double Ko){
        double m = (PKv14_Csc) * (FARADAY*VFRT) * XKv14 / (1 - exp_term);
        return m * ((Ki - Ko*exp_term) + 0.02*(Nai - Nao*exp_term)) * 1.0e9; // 1e9 required to convert to mV / ms
    }


    //template <typename T>
    inline double K1inf(const double V, const double EK, const double F_RT){ return 1.0 / (2.0 + exp(1.5*(V-EK)*F_RT)); }

    //template <typename T>
    inline double IK1(const double V, const double EK, const double GK1, const double IK1_const, const double FR_T){ 
        return GK1 * K1inf(V, EK, FR_T) * IK1_const * (V - EK); 
    }


    //template <typename T>
    inline double Kp(const double V){ return 1. / (1. + exp((7.488 - V) / 5.98)); }

    //template <typename T>
    inline double IKp(const double V, const double EK, const double GKp){ return GKp * Kp(V) * (V-EK); }
    
    inline double XKsinf(const double V){ return 1.0 / (1.0 + exp(-(V - 24.7) / 13.6)); }

    //template <typename T>
    inline double tauXKs_inv(const double V){ return 0.0000719*(V-10.0)/(1.0 - exp(-0.148*(V-10.0))) + 0.000131*(V-10.0)/(exp(0.0687*(V-10.0)) - 1.0); }


    //template <typename T>
    inline double alphaLCC(const double V) { return 2.0 * exp(0.012 * (V - 35.0)); }

    //template <typename T>
    inline double betaLCC(const double V) { return 0.0882 * exp(-0.05 * (V - 35.0)); }

    //template <typename T>
    inline double yinfLCC(const double V) { return 0.4 / (1.0 + exp((V + 12.5) / 5.0)) + 0.6; }

    //template <typename T>
    inline double tauLCC(const double V)  { return 340.0 / (1.0 + exp((V + 30.0) / 12.0)) + 60.0; }

}
