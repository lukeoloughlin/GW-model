#include "ndarray.hpp"

template <typename T, size_t N>
bool NDArrayBase<T,N>::assert_equal_shape(const NDArrayBase<T,N>& other) const {
    for (int i = 0; i < N; ++i){
        if (dims[i] != other.dims[i])
            return false;
    }
    return true;
}

template <typename T, size_t N>
void NDArrayBase<T,N>::copy_from_array(const T* const arr){
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] = arr[i];
}

template <typename T, size_t N>
T NDArrayBase<T,N>::sum() const {
    T acc = 0;
    #pragma omp simd reduction(+:acc)
    for (size_t i = 0; i < size; ++i)
        acc += storage[i];
    return acc;
}


template <typename T, size_t N>
void NDArrayBase<T,N>::inv() {
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] = 1 / storage[i];
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
    std::cout << "Called *= v1" << std::endl;
    #pragma omp simd
    for (size_t i = 0; i < size; ++i)
        storage[i] *= val;
    return *this;
}

template <typename T, size_t N>
NDArrayBase<T,N>& NDArrayBase<T,N>::operator*=(const NDArrayBase<T,N> &val){
    std::cout << "Called *= v2" << std::endl;
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


template <typename T, typename S, size_t N>
NDArrayBase<T,N> operator+(const NDArrayBase<T,N>& arr, const S& arr_or_T){
    std::cout << "Called + v1" << std::endl;
    NDArray<T,N> out = arr;
    out += arr_or_T;
    return out;
} 

template <typename T, typename S, size_t N>
NDArrayBase<T,N> operator+(const S& arr_or_T, NDArrayBase<T,N>& arr){
    std::cout << "Called + v2" << std::endl;
    NDArray<T,N> out = arr;
    out += arr_or_T;
    return out;
} 


template <typename T, typename S, size_t N>
NDArray<T,N> operator+(NDArray<T,N>&& arr, const S& arr_or_T){
    std::cout << "Called + v3" << std::endl;
    arr += arr_or_T;
    return std::move(arr);
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator+(const S& arr_or_T, NDArray<T,N>&& arr){
    std::cout << "Called + v4" << std::endl;
    arr += arr_or_T;
    return std::move(arr);
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator+(NDArray<T,N>&& arr, S&& arr_or_T){
    std::cout << "Called + v5" << std::endl;
    arr += arr_or_T;
    return std::move(arr);
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator+(S&& arr_or_T, NDArray<T,N>&& arr){
    std::cout << "Called + v6" << std::endl;
    arr += arr_or_T;
    return std::move(arr);
}


template <typename T, typename S, size_t N>
NDArray<T,N> operator-(const NDArray<T,N>& arr, const S& arr_or_T){
    std::cout << "Called - v1" << std::endl;
    NDArray<T,N> out = arr;
    out -= arr_or_T;
    return out;
} 

template <typename T, typename S, size_t N>
NDArray<T,N> operator-(const S& arr_or_T, NDArray<T,N>& arr){
    std::cout << "Called - v2" << std::endl;
    NDArray<T,N> out = arr;
    out *= -1;
    out += arr_or_T;
    return out;
} 


template <typename T, typename S, size_t N>
NDArray<T,N> operator-(NDArray<T,N>&& arr, const S& arr_or_T){
    std::cout << "Called - v3" << std::endl;
    arr -= arr_or_T;
    return std::move(arr);
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator-(const S& arr_or_T, NDArray<T,N>&& arr){
    std::cout << "Called - v4" << std::endl;
    arr *= -1;
    arr += arr_or_T;
    return std::move(arr);
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator-(NDArray<T,N>&& arr, S&& arr_or_T){
    std::cout << "Called - v5" << std::endl;
    arr -= arr_or_T;
    return std::move(arr);
}




template <typename T, typename S, size_t N>
NDArray<T,N> operator*(const NDArray<T,N>& arr, const S& arr_or_T){
    std::cout << "Called * v1" << std::endl;
    NDArray<T,N> out = arr;
    out *= arr_or_T;
    return out;
} 

template <typename T, typename S, size_t N>
NDArray<T,N> operator*(const S& arr_or_T, NDArray<T,N>& arr){
    std::cout << "Called * v2" << std::endl;
    NDArray<T,N> out = arr;
    out *= arr_or_T;
    return out;
} 


template <typename T, typename S, size_t N>
NDArray<T,N> operator*(NDArray<T,N>&& arr, const S& arr_or_T){
    std::cout << "Called * v3" << std::endl;
    arr *= arr_or_T;
    return std::move(arr);
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator*(const S& arr_or_T, NDArray<T,N>&& arr){
    std::cout << "Called * v4" << std::endl;
    arr *= arr_or_T;
    return std::move(arr);
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator*(NDArray<T,N>&& arr, S&& arr_or_T){
    std::cout << "Called * v5" << std::endl;
    arr *= arr_or_T;
    return std::move(arr);
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator*(S&& arr_or_T, NDArray<T,N>&& arr){
    std::cout << "Called * v6" << std::endl;
    return std::move(arr *= arr_or_T);
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator/(const NDArray<T,N>& arr, const S& arr_or_T){
    NDArray<T,N> out = arr;
    out /= arr_or_T;
    return out;
} 

template <typename T, typename S, size_t N>
NDArray<T,N> operator/(const S& arr_or_T, NDArray<T,N>& arr){
    NDArray<T,N> out = arr;
    out.inv();
    out *= arr_or_T;
    return out;
} 


template <typename T, typename S, size_t N>
NDArray<T,N> operator/(NDArray<T,N>&& arr, const S& arr_or_T){
    arr /= arr_or_T;
    return std::move(arr);
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator/(const S& arr_or_T, NDArray<T,N>&& arr){
    arr.inv();
    arr *= arr_or_T;
    return std::move(arr);
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator/(NDArray<T,N>&& arr, S&& arr_or_T){
    arr /= arr_or_T;
    return std::move(arr);
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator/(S&& arr_or_T, NDArray<T,N>&& arr){
    arr.inv();
    arr *= arr_or_T;
    return std::move(arr);
}