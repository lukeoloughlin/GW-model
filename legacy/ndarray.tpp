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
NDArrayBase<T,N>& NDArrayBase<T,N>::copy(NDArrayBase<T,N>& other){
    if (assert_equal_shape(other)){
        copy_from_array(other.storage);
        return *this;
    }
    else {
        std::cout << "Attempted to copy from array of different shape. Exiting program." << std::endl;
        exit(EXIT_FAILURE);
    }
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



template <typename T, typename S, size_t N>
NDArray<T,N> operator+(const NDArray<T,N>& arr, const S& arr_or_T){
    NDArray<T,N> out = arr;
    out += arr_or_T;
    return out;
} 

template <typename T, typename S, size_t N>
NDArray<T,N> operator+(const S& arr_or_T, NDArray<T,N>& arr){
    NDArray<T,N> out = arr;
    out += arr_or_T;
    return out;
} 

template <typename T, typename S, size_t N>
NDArray<T,N> operator+(NDArray<T,N>&& arr, const S& arr_or_T){
    arr += arr_or_T;
    return arr;
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator+(const S& arr_or_T, NDArray<T,N>&& arr){
    arr += arr_or_T;
    return arr;
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator+(NDArray<T,N>&& arr, S&& arr_or_T){
    arr += arr_or_T;
    return arr;
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator+(S&& arr_or_T, NDArray<T,N>&& arr){
    arr += arr_or_T;
    return arr;
}


template <typename T, typename S, size_t N>
NDArray<T,N> operator-(const NDArray<T,N>& arr, const S& arr_or_T){
    NDArray<T,N> out = arr;
    out -= arr_or_T;
    return out;
} 

template <typename T, typename S, size_t N>
NDArray<T,N> operator-(const S& arr_or_T, NDArray<T,N>& arr){
    NDArray<T,N> out = arr;
    out *= -1;
    out += arr_or_T;
    return out;
} 

template <typename T, typename S, size_t N>
NDArray<T,N> operator-(NDArray<T,N>&& arr, const S& arr_or_T){
    arr -= arr_or_T;
    return std::move(arr);
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator-(const S& arr_or_T, NDArray<T,N>&& arr){
    arr *= -1;
    arr += arr_or_T;
    return arr;
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator-(NDArray<T,N>&& arr, S&& arr_or_T){
    arr -= arr_or_T;
    return arr;
}


template <typename T, typename S, size_t N>
NDArray<T,N> operator*(const NDArray<T,N>& arr, const S& arr_or_T){
    NDArray<T,N> out = arr;
    out *= arr_or_T;
    return out;
} 

template <typename T, typename S, size_t N>
NDArray<T,N> operator*(const S& arr_or_T, NDArray<T,N>& arr){
    NDArray<T,N> out = arr;
    out *= arr_or_T;
    return out;
} 

template <typename T, typename S, size_t N>
NDArray<T,N> operator*(NDArray<T,N>&& arr, const S& arr_or_T){
    arr *= arr_or_T;
    return arr;
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator*(const S& arr_or_T, NDArray<T,N>&& arr){
    arr *= arr_or_T;
    return arr;
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator*(NDArray<T,N>&& arr, S&& arr_or_T){
    arr *= arr_or_T;
    return arr;
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator*(S&& arr_or_T, NDArray<T,N>&& arr){
    arr *= arr_or_T;
    return arr;
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
    return arr;
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator/(const S& arr_or_T, NDArray<T,N>&& arr){
    arr.inv();
    arr *= arr_or_T;
    return arr;
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator/(NDArray<T,N>&& arr, S&& arr_or_T){
    arr /= arr_or_T;
    return arr;
}

template <typename T, typename S, size_t N>
NDArray<T,N> operator/(S&& arr_or_T, NDArray<T,N>&& arr){
    arr.inv();
    arr *= arr_or_T;
    return arr;
}