#ifndef NDARRAY_H
#define NDARRAY_H

#include <vector>
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>

template <typename T, std::size_t N>
class NDArrayBase {

protected:
    T* storage;
    int dims[N];
    size_t size;
    
    bool is_map = false;


    void set_storage(T* storage_){ storage = storage_; }

    void set_dims(const int dims_[N]){
        for (size_t i = 0; i < N; ++i)
            dims[i] = dims_[i];
    }

    void set_size(size_t size_){ size = size_; }

    bool assert_equal_shape(const NDArrayBase<T,N>& other) const;

    void copy_from_array(const T* const arr);


    template <typename First, typename... Indices>
    inline int to_linear_index(int carry, const int i, const First first, const Indices... indices) const {
        if (i == 0)
            return to_linear_index(first, 1, indices...);
        else
            return to_linear_index(first + dims[i] * carry, i+1, indices...);
    }

    template <typename Last>
    inline int to_linear_index(int carry, const int i, const Last last) const {
        return last + carry*dims[i];
    }

public:
    
    template <typename... Indices>
    inline T operator() (Indices... indices) const { 
        int idx = to_linear_index(0, 0, indices...);
        return storage[idx];
    }
    
    // Writeable access
    template <typename... Indices>
    inline T& operator() (Indices... indices) {
        int idx = to_linear_index(0, 0, indices...);
        return storage[idx];
    }

    inline T operator[] (int i) const { return storage[i]; }
    
    inline NDArrayBase& operator+=(const T val);
    inline NDArrayBase& operator+=(const NDArrayBase<T,N> &val);

    inline NDArrayBase& operator-=(const T val);
    inline NDArrayBase& operator-=(const NDArrayBase<T,N> &val);


    inline NDArrayBase& operator*=(const T val);
    inline NDArrayBase& operator*=(const NDArrayBase<T,N> &val);

    inline NDArrayBase& operator/=(const T val);
    inline NDArrayBase& operator/=(const NDArrayBase<T,N> &val);

    inline NDArrayBase& copy(NDArrayBase& other);
    
    template <typename... Indices>
    inline void set(const T val, Indices... indices){
        int idx = to_linear_index(indices...);
        storage[idx] = val;
    }

    inline T sum() const; 
    inline void inv();
    inline void add(const T val);
    inline void sub(const T val);    
    inline void mul(const T val);
    inline void div(const T val);
    inline void set_to(const T val);
    inline void set_to_zeros(){ set_to(0); }
    inline void set_to_ones(){ set_to(1); }

    T* data() const { return storage; }

    T maximum() const { return *std::max_element(storage, storage+size); }
    T minimum() const { return *std::min_element(storage, storage+size); }

    unsigned int shape(const unsigned int dim)
    {
        if (dim > size){
            throw(std::invalid_argument("tried to access non-existant dimension of NDArray"));
        }
        return dims[dim];
    }


    template <typename S, size_t M>
    friend class NDArray;

    template <typename S, size_t M>
    friend class NDArrayMap;

};


/*
A multidimensional array class. Essentially just holds an array and provides multidimensional indexing semantics.
The NDArrayMap is essentially a view of an array that has already been allocated and does not manage the memory of the
underlying array. E.g. in the snippet below:

    std::vector<double> x(1000);
    int dims[2] = { 10, 100 };
    NDArrayMap<double,2> xmap(x,dims);

xmap holds a view of the data held by x as a 10x100 matrix, and the deallocation of the memory assocated with this view is handled by
std::vector.
*/
template <typename T, std::size_t N>
class NDArrayMap : public NDArrayBase<T,N>
{
public:
    NDArrayMap() = default;
    
    template <typename... Dimensions>
    NDArrayMap(T* arr, Dimensions... indices){
        size_t size_ = 1;
        int i = 0;
        int dims_[N];
        for (const auto p : {indices...}){
            dims_[i] = p;
            size_ *= p;
            ++i;
        }
        this->set_storage(arr);
        this->set_dims(dims_);
        this->set_size(size_);
        this->is_map = true;
        
    }

    NDArrayMap(T* arr, const int dims_[N], const int size_) { 
        this->set_storage(arr);
        this->set_dims(dims_);
        this->set_size((size_t)size_);
        this->is_map = true;
    }

    NDArrayMap(std::vector<T> &arr, const int dims_[N]){
        this->set_storage(arr.data());
        this->set_dims(dims_);
        this->set_size(arr.size());
        this->is_map = true;
    }

    NDArrayMap& operator=(NDArrayBase<T,N>&& other) noexcept {
        // Guard self assignment
        if (this == &other)
            return *this;
        
        if (assert_equal_shape(other)) {
            if (other.is_map) {
                this->set_storage(other.storage);
                other.set_storage(nullptr);
            }
            else { // If we try to move the storage of a temporary NDArray we will get a memory leak, so just copy the elements.
                copy_from_array(other.copy_from_array);
            }
        }

        std::cout << "Tried to move an NDArrayMap of size " << other.size << " into a location with size " << this->size << std::endl;
        exit(EXIT_FAILURE);

        return *this;
    }

    NDArrayMap& operator=(const NDArrayBase<T,N>& other) { // The copy operator
        if (this == &other)
            return *this;
        
        if (assert_equal_shape(other))
            copy_from_array(other.copy_from_array);
        else {
            std::cout << "Tried to copy an NDArrayMap of size " << other.size << " into a location with size " << this->size << std::endl;
            exit(EXIT_FAILURE);
        }

        return *this;
    }
};

/*
The NDArray manages its own memory unlike the NDArrayMap class. When initialised, the underlying array is allocated on the heap and lives for as long as
the NDArray object. E.g. 

// once x goes out of scope, the array hled by the NDArray x.data() is deallocated by the destructor.
void f(){
    NDArray<double,2> x(10,100); // a 10x100 matrix of zeros
    // Do stuff
} 

int main{
    NDArray<double,2>* x = new NDArray<double, 2>(10,100); // x.data() lives as long as x does in the program
    // Do stuff
    delete x; // x.data() is deleted when x is deleted.
    return 0;
}
*/

template <typename T, std::size_t N>
class NDArray : public NDArrayBase<T,N> {
public:
    NDArray() = default;

    NDArray(const NDArray<T,N>& other) {
        this->storage = new T[other.size];
        this->size = other.size;
        this->copy_from_array(other.storage);
        std::memcpy(this->dims, other.dims, N*sizeof(int));
    }

    
    NDArray(NDArray<T,N>&& other) {
        this->size = other.size;
        other.size = 0;

        this->storage = other.storage;
        other.storage = nullptr;

        std::memcpy(this->dims, other.dims, N*sizeof(int));
        std::memset(other.dims, 0, N*sizeof(int));
    }

    template <typename... Dimensions>
    NDArray(Dimensions... indices) {
        static_assert(sizeof...(indices) == N, "Attempted to initialise NDArray array with invalid number of dimensions. "
                                                "Make sure that the number of dimensions passed to the construct and the template parameter N match.");
        size_t sz = 1;
        int dims_[N];
        int i = 0;
        for (const auto p : {indices...}){
            sz *= p;
            dims_[i] = p;
            ++i;
        }

        T* arr = new T[sz];

        this->set_size(sz);
        this->set_dims(dims_);
        this->set_storage(arr);

        this->set_to_zeros();    
    }
    
    ~NDArray() { delete[] this->storage; }

    
    NDArray& operator=(NDArray<T,N>&& other) noexcept { // The move operator
        if (this == &other)
            return *this;

        if (this->assert_equal_shape(other)){
            if (other.is_map){ // If we try to move an NDArrayMap into the NDArray we will get a segfault when the destructor is called, so we just copy the elements.
                this->copy_from_array(other.storage); 
            }
            else {
                delete[] this->storage;
                this->set_storage(other.storage);
                other.set_storage(nullptr);
            }
            return *this;
        }
        std::cout << "Tried to move an NDArray of size " << other.size << " into a location with size " << this->size << std::endl;
        exit(EXIT_FAILURE);
    }

    NDArray& operator=(const NDArray<T,N>& other) { // The copy operator
        if (this == &other)
            return *this;
        
        if (this->assert_equal_shape(other))
            this->copy_from_array(other.storage);
        else {
            std::cout << "Tried to copy an NDArray of size " << other.size << " into a location with size " << this->size << std::endl;
            exit(EXIT_FAILURE);
        }

        return *this;
    }
};


template <typename T, size_t N>
NDArray<T,N> operator+(const NDArray<T,N>& arr1, const NDArray<T,N>& arr2);
template <typename T, size_t N>
NDArray<T,N> operator+(NDArray<T,N>&& arr1, const NDArray<T,N>& arr2);
template <typename T, size_t N>
NDArray<T,N> operator+(NDArray<T,N>& arr1, const NDArray<T,N>&& arr2);
template <typename T, size_t N>
NDArray<T,N> operator+(NDArray<T,N>&& arr1, NDArray<T,N>&& arr2);

template <typename T, size_t N>
NDArray<T,N> operator+(const NDArray<T,N>& arr, const T& val);
template <typename T, size_t N>
NDArray<T,N> operator+(const T& val, NDArray<T,N>& arr);
template <typename T, size_t N>
NDArray<T,N> operator+(T&& val, const NDArray<T,N>& arr);
template <typename T, size_t N>
NDArray<T,N> operator+(const NDArray<T,N>& arr, T&& val);
template <typename T, size_t N>
NDArray<T,N> operator+(const T& val, NDArray<T,N>&& arr);
template <typename T, size_t N>
NDArray<T,N> operator+(NDArray<T,N>&& arr, const T& val);
template <typename T, size_t N>
NDArray<T,N> operator+(T&& val, NDArray<T,N>&& arr);
template <typename T, size_t N>
NDArray<T,N> operator+(NDArray<T,N>&& arr, T&& val);

template <typename T, typename S, size_t N>
NDArray<T,N> operator-(const NDArray<T,N>& arr, const S& arr_or_T);
template <typename T, typename S, size_t N>
NDArray<T,N> operator-(const S& arr_or_T, NDArray<T,N>& arr);
template <typename T, typename S, size_t N>
NDArray<T,N> operator-(NDArray<T,N>&& arr, const S& arr_or_T);
template <typename T, typename S, size_t N>
NDArray<T,N> operator-(const S& arr_or_T, NDArray<T,N>&& arr);
template <typename T, typename S, size_t N>
NDArray<T,N> operator-(NDArray<T,N>&& arr, S&& arr_or_T);
template <typename T, typename S, size_t N>
NDArray<T,N> operator-(S&& arr_or_T, NDArray<T,N>&& arr);

template <typename T, typename S, size_t N>
NDArray<T,N> operator*(const NDArray<T,N>& arr, const S& arr_or_T);
template <typename T, typename S, size_t N>
NDArray<T,N> operator*(const S& arr_or_T, NDArray<T,N>& arr);
template <typename T, typename S, size_t N>
NDArray<T,N> operator*(NDArray<T,N>&& arr, const S& arr_or_T);
template <typename T, typename S, size_t N>
NDArray<T,N> operator*(const S& arr_or_T, NDArray<T,N>&& arr);
template <typename T, typename S, size_t N>
NDArray<T,N> operator*(NDArray<T,N>&& arr, S&& arr_or_T);
template <typename T, typename S, size_t N>
NDArray<T,N> operator*(S&& arr_or_T, NDArray<T,N>&& arr);

template <typename T, typename S, size_t N>
NDArray<T,N> operator/(const NDArray<T,N>& arr, const S& arr_or_T);
template <typename T, typename S, size_t N>
NDArray<T,N> operator/(const S& arr_or_T, NDArray<T,N>& arr);
template <typename T, typename S, size_t N>
NDArray<T,N> operator/(NDArray<T,N>&& arr, const S& arr_or_T);
template <typename T, typename S, size_t N>
NDArray<T,N> operator/(const S& arr_or_T, NDArray<T,N>&& arr);
template <typename T, typename S, size_t N>
NDArray<T,N> operator/(NDArray<T,N>&& arr, S&& arr_or_T);
template <typename T, typename S, size_t N>
NDArray<T,N> operator/(S&& arr_or_T, NDArray<T,N>&& arr);


#include "ndarray.tpp"

#endif