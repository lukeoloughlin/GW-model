#include "GW.hpp"
#include <iostream>
#include <iomanip>
#include <functional>
#include <fstream>
#include <string>

using namespace GW;

double Ist(double t) { return (t < 2.0) ? 35.0 : 0.0; }

int main(int argc, char** argv)
{
    if (argc < 2){
        std::cout << "An output filename must be provided as an argument." << std::endl;
        return 0;
    }
    else if (argc > 2){
        std::cout << "Too many arguments!" << std::endl;
        return 0;
    }


    std::string fname = argv[1];
    GW_model<double> model(1000);

    std::ofstream file;
    file.open(fname, std::ofstream::out | std::ofstream::trunc );
    file << std::setprecision(12);
    model.euler_write(1e-3, 500000, [](double t){ return (t < 2) ? 35.0 : 0.0; }, file, 1000);
    file.close();
    std::cout << model.globals.V << std::endl;


    return 0;
}