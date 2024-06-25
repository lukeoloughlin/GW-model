#include "ndarray.hpp"


template <typename T, size_t N>
T NDArrayBase<T,N>::sum() const {
    T acc = 0;
    #pragma omp simd reduction(+:acc)
    for (size_t i = 0; i < size; ++i)
        acc += storage[i];
    return acc;
}


template <typename T, size_t N>
void NDArrayBase<T,N>::add(const T val) {
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] += val;
}

template <typename T, size_t N>
void NDArrayBase<T,N>::sub(const T val) {
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] -= val;
}

template <typename T, size_t N>
void NDArrayBase<T,N>::mul(const T val) {
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] *= val;
}

template <typename T, size_t N>
void NDArrayBase<T,N>::div(const T val) {
    const T val_inv = 1 / val;
    mul(val_inv);
}

template <typename T, size_t N>
void NDArrayBase<T,N>::set_to(const T val) {
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] = val;
}


template <typename T, size_t N>
NDArrayBase<T,N>& NDArrayBase<T,N>::operator+=(const T val){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] += val;
    return *this;
}

template <typename T, size_t N>
NDArrayBase<T,N>& NDArrayBase<T,N>::operator+=(const NDArrayBase<T,N> &val){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] += val[i];
    return *this;
}

template <typename T, size_t N>
NDArrayBase<T,N>& NDArrayBase<T,N>::operator-=(const T val){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] -= val;
    return *this;
}

template <typename T, size_t N>
NDArrayBase<T,N>& NDArrayBase<T,N>::operator-=(const NDArrayBase<T,N> &val){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] -= val[i];
    return *this;
}

template <typename T, size_t N>
NDArrayBase<T,N>& NDArrayBase<T,N>::operator*=(const T val){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] *= val;
    return *this;
}

template <typename T, size_t N>
NDArrayBase<T,N>& NDArrayBase<T,N>::operator*=(const NDArrayBase<T,N> &val){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] *= val[i];
    return *this;
}

template <typename T, size_t N>
NDArrayBase<T,N>& NDArrayBase<T,N>::operator/=(const T val){
    const T val_inv = 1 / val;
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] *= val;
    return *this;
}

template <typename T, size_t N>
NDArrayBase<T,N>& NDArrayBase<T,N>::operator/=(const NDArrayBase<T,N> &val){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] /= val[i];
    return *this;
}

template <typename T, size_t N>
void NDArrayBase<T,N>::exp(){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] = std::exp(storage[i]);
}

template <typename T, size_t N>
void NDArrayBase<T,N>::log(){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] = std::log(storage[i]);
}

template <typename T, size_t N>
void NDArrayBase<T,N>::sin(){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] = std::sin(storage[i]);
}

template <typename T, size_t N>
void NDArrayBase<T,N>::cos(){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] = std::tan(storage[i]);
}

template <typename T, size_t N>
void NDArrayBase<T,N>::tan(){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] = std::tan(storage[i]);
}

template <typename T, size_t N>
void NDArrayBase<T,N>::sinh(){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] = std::sinh(storage[i]);
}

template <typename T, size_t N>
void NDArrayBase<T,N>::cosh(){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] = std::cosh(storage[i]);
}

template <typename T, size_t N>
void NDArrayBase<T,N>::tanh(){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] = std::tanh(storage[i]);
}

template <typename T, size_t N>
void NDArrayBase<T,N>::pow(const T exponent){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] = std::pow(storage[i], exponent);
}

template <typename T, size_t N>
void NDArrayBase<T,N>::abs(){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] = std::abs(storage[i]);
}