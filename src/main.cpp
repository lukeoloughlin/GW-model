#include "GW.hpp"
#include <iostream>
#include <iomanip>
#include <functional>
#include <fstream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>

using namespace GW;

double Ist(double t) { return (t < 2.0) ? 35.0 : 0.0; }


typedef Eigen::VectorXd Array1d;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> Array2d;
typedef Eigen::Tensor<double,3,Eigen::RowMajor> Array3d;
typedef Eigen::Tensor<int,3,Eigen::RowMajor> Array3i;
typedef Eigen::Tensor<int,4,Eigen::RowMajor> Array4i;

struct PyGWOutput {
    int nCRU;
    double t;

    Array1d V;
    Array1d m;
    Array1d h;
    Array1d j;
    Array1d Nai;
    Array1d Ki;
    Array1d Cai;
    Array1d CaNSR;
    Array1d CaLTRPN;
    Array1d CaHTRPN;
    Array1d xKs;
    Array2d XKr;
    Array2d XKv14;
    Array2d XKv43;
    Array2d CaJSR;
    Array3d CaSS;
    Array3i LCC;
    Array3i LCC_activation;
    Array4i RyR;
    Array3i ClCh;

    PyGWOutput(int nCRU_, int num_step, double t_) : nCRU(nCRU), t(t_), V(num_step), m(num_step), h(num_step), j(num_step), 
                                                     Nai(num_step), Ki(num_step), Cai(num_step), CaNSR(num_step), CaLTRPN(num_step), 
                                                     CaHTRPN(num_step), xKs(num_step), XKr(num_step,5), XKv14(num_step,10), XKv43(num_step,10), 
                                                     CaJSR(num_step,nCRU_), CaSS(num_step,nCRU_,4), LCC(num_step,nCRU_,4), 
                                                     LCC_activation(num_step,nCRU_,4), RyR(num_step,nCRU_,4,6), ClCh(num_step,nCRU_,4) { }
};


class PyGW {
public:
    GW::GW_model<double> model;

    PyGW(int nCRU) : model(nCRU) { }
    PyGW(int nCRU, GW::Parameters<double> params) : model(nCRU) {
        model.parameters = params;
    }

    PyGWOutput run(double step_size, int num_steps, const std::function<double(double)> &Is, int record_every){

        const int nCRU = model.get_nCRU();
        PyGWOutput out(nCRU, num_steps / record_every, num_steps*step_size);
        double t = 0.0;
        int counter = 0;
        for (int i = 0; i < num_steps; ++i){
            model.Istim = Is(t);
            model.euler_step(step_size);
            t += step_size;
            if (i % record_every == 0){
                record_state(out, counter, nCRU);
                ++counter;
            }
        }
        return out;
    }

    void record_state(PyGWOutput& out, int idx, const int nCRU){
        out.V(idx) = model.globals.V;
        out.m(idx) = model.globals.m;
        out.h(idx) = model.globals.h;
        out.j(idx) = model.globals.j;
        out.Nai(idx) = model.globals.Nai;
        out.Ki(idx) = model.globals.Ki;
        out.Cai(idx) = model.globals.Cai;
        out.CaNSR(idx) = model.globals.CaNSR;
        out.CaLTRPN(idx) = model.globals.CaLTRPN;
        out.CaHTRPN(idx) = model.globals.CaHTRPN;
        out.xKs(idx) = model.globals.xKs;
        
        out.XKr(idx,0) = model.globals.Kr[0];
        out.XKr(idx,1) = model.globals.Kr[1];
        out.XKr(idx,2) = model.globals.Kr[2];
        out.XKr(idx,3) = model.globals.Kr[3];
        out.XKr(idx,4) = model.globals.Kr[4];

        for (int j = 0; j < 10; ++j){
            out.XKv14(idx,j) = model.globals.Kv14[j];
            out.XKv43(idx,j) = model.globals.Kv43[j];
        }

        for (int j = 0; j < nCRU; ++j){
            out.CaJSR(idx,j) = model.CRUs.CaJSR(j);
            for (int k = 0; k < 4; ++k){
                out.CaSS(idx,j,k) = model.CRUs.CaSS(j,k);
                out.LCC(idx,j,k) = model.CRUs.LCC(j,k);
                out.LCC_activation(idx,j,k) = model.CRUs.LCC_activation(j,k);
                out.RyR(idx,j,k,0) = model.CRUs.RyR(j,k,0);
                out.RyR(idx,j,k,1) = model.CRUs.RyR(j,k,1);
                out.RyR(idx,j,k,2) = model.CRUs.RyR(j,k,2);
                out.RyR(idx,j,k,3) = model.CRUs.RyR(j,k,3);
                out.RyR(idx,j,k,4) = model.CRUs.RyR(j,k,4);
                out.RyR(idx,j,k,5) = model.CRUs.RyR(j,k,5);
                out.ClCh(idx,j,k) = model.CRUs.ClCh(j,k);
            }
        }
    }

};

int main(int argc, char** argv)
{
    //if (argc < 2){
    //    std::cout << "An output filename must be provided as an argument." << std::endl;
    //    return 0;
    //}
    //else if (argc > 2){
    //    std::cout << "Too many arguments!" << std::endl;
    //    return 0;
    //}


    //std::string fname = argv[1];
    //GW_model<double> model(1000);

    //std::ofstream file;
    //file.open(fname, std::ofstream::out | std::ofstream::trunc );
    //file << std::setprecision(12);
    //model.euler_write(1e-3, 500000, [](double t){ return (t < 2) ? 35.0 : 0.0; }, file, 1000);
    //file.close();

    PyGW model(1000);

    auto out = model.run(1e-3, 100000, Ist, 1);
    std::cout << out.V(10000) << std::endl;


    return 0;
}