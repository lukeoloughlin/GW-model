
template<typename FloatType, typename ParamType, typename ConstType>
class Model
{
private:
    FloatType VFRT; // VF / RT
    FloatType expmVFRT; // exp(-VF/RT)
    FloatType dV; // dV/dt
    FloatType* dconc; // Array of d[X]/dt
    FloatType* dgates; // Array of dx/dt where x is a gating variable
    const ConstType* const constants; // Constants derived from params


public:
    FloatType V;
    FloatType* concentration;
    FloatType* gates;
    const ParamType* const parameters; // Points to a struct of parameter values

};