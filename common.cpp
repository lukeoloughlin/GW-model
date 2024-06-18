#include "common.hpp"
#include <iostream>

using namespace std;

void test_Nernst()
{
    double Xod = 1.0, zd = 1.0;
    float Xof = 1.0, zf = 1.0;

    double Xid = 0.1;
    float Xif = 0.1;

    double Xi_err = -0.1;

    double RT_F = 8.314 * 310.0 / 96.5;

    cout << "Using double: " << Nernst<double>(Xid, Xod, RT_F, zd) << endl;
    cout << "Using float: " << Nernst<float>(Xif, Xof, float(RT_F), zf) << endl;

    try
    {
        cout << Nernst<double>(Xi_err, Xod, RT_F, zd) << endl;
    }
    catch(const std::exception& e)
    {
        std::cout << "Caught error in call to Nernst" << endl;
    }
}

void test_urand()
{
    double u;
    for (int i = 0; i < 10; i++){
        u = urand<double>();
        cout << u << endl;
    }
}

int main(int argc, char* argv[])
{
    test_Nernst();
    cout << "Nernst test passed!" << endl << endl;
    test_urand();
    return 0;
}