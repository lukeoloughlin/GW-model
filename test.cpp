#include "ndarray.hpp"
#include <iostream>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;


int main()
{
    double sum = 0;
    const int N = 50;
    double data[N*N];
    int dims[2] = { N, N };
    NDArrayMap<double,2> stack_alloc(data, dims, N*N);
    stack_alloc.set_to_zeros();
    NDArray<double,2> heap_alloc(N,N);

    auto t1 = Clock::now();
    for (int i = 0; i < 1000; i++){
        sum += stack_alloc.sum();
    }
    auto t2 = Clock::now();

    std::cout << "Dummy output: " << sum << std::endl;
    std::cout << "Stack time: " << 1e-3 * float(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()) / 1000.0 << " microseconds" << std::endl; 
    
    t1 = Clock::now(); 
    for (int i = 0; i < 1000; i++){
        sum += heap_alloc.sum();
    }
    t2 = Clock::now();

    std::cout << "Dummy output: " << sum << std::endl;
    std::cout << "Dynamic time: " << 1e-3 * float(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()) / 1000.0 << " microseconds" << std::endl << std::endl << std::endl; 

    std::cout << "+= performance:" << std::endl << std::endl;

    t1 = Clock::now(); 
    for (int i = 0; i < 100000; i++){
        heap_alloc += (1.0+1e-10);
    } 
    t2 = Clock::now();
    
    std::cout << "Dummy output: " << heap_alloc(0,0) << std::endl;
    std::cout << "+= operator time (average): " << 1e-3 * float(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()) * 1e-5 << " microseconds" << std::endl << std::endl; 
    
    heap_alloc.set_to_zeros();

    t1 = Clock::now(); 
    for (int i = 0; i < 100000; i++){
        heap_alloc.add(1.0-1e-10);
    } 
    t2 = Clock::now();
    
    std::cout << "Dummy output: " << heap_alloc(0,0) << std::endl;
    std::cout << "add method time (average): " << 1e-3 * float(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()) * 1e-5 << " microseconds" << std::endl << std::endl << std::endl;; 


    std::cout << "*= performance:" << std::endl << std::endl;

    heap_alloc.set_to_ones();

    t1 = Clock::now(); 
    for (int i = 0; i < 100000; i++){
        heap_alloc *= (1.0+1e-10);
    } 
    t2 = Clock::now();

    
    std::cout << "Dummy output: " << heap_alloc(0,0) << std::endl;
    std::cout << "*= operator time (average): " << 1e-3 * float(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()) * 1e-5 << " microseconds" << std::endl << std::endl; 
    
    heap_alloc.set_to_ones();

    t1 = Clock::now(); 
    for (int i = 0; i < 100000; i++){
        heap_alloc.mul(1.0-1e-10);
    } 
    t2 = Clock::now();
    
    
    std::cout << "Dummy output: " << heap_alloc(0,0) << std::endl;
    std::cout << "mul method time (average): " << 1e-3 * float(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()) * 1e-5 << " microseconds" << std::endl << std::endl; 
    
    std::cout << "/= performance:" << std::endl << std::endl;
    

    heap_alloc.set_to_ones();

    t1 = Clock::now(); 
    for (int i = 0; i < 100000; i++){
        heap_alloc /= (1.0+1e-10);
    } 
    t2 = Clock::now();
    
    std::cout << "Dummy output: " << heap_alloc(0,0) << std::endl;
    std::cout << "/= operator time (average): " << 1e-3 * float(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()) * 1e-5 << " microseconds" << std::endl << std::endl; 


    heap_alloc.set_to_ones();

    t1 = Clock::now(); 
    for (int i = 0; i < 100000; i++){
        heap_alloc.div(1.0-1e-10);
    } 
    t2 = Clock::now();
    
    std::cout << "Dummy output: " << heap_alloc(0,0) << std::endl;
    std::cout << "div method time (average): " << 1e-3 * float(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()) * 1e-5 << " microseconds" << std::endl << std::endl; 


    

    return 0;
}