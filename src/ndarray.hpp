#ifndef _NDARRAYH
#define _NDARRAYH

#include <vector>
#include <assert.h>
#include <cmath>



template <typename T, std::size_t N>
class NDArrayBase
{
protected:
    T* storage;
    int dims[N];
    size_t size;

    void set_storage(T* storage_){ storage = storage_; }

    void set_dims(const int dims_[N]){
        for (unsigned int i = 0; i < N; ++i){
            dims[i] = dims_[i];
        }
    }

    void set_size(size_t size_){ size = size_; }

    // Segfaults if storage was not heap allocated.
    void delete_storage() { delete[] storage; }

    template <typename... Indices>
    inline int to_linear_index(const Indices... indices) const {
        static_assert(sizeof...(indices) == N, "Attempted to index array with invalid number of dimensions.");
        int idx = 0;
        int i = 0;
        for (const auto p : {indices...}){
            assert((p > 0, "Invalid indexing argument. A non-positive index was passed."));
            if (i == 0)
                idx += p;
            else {
                idx *= dims[i];
                idx += p;
            }
            ++i;
        }
        return idx;
    }


public:
    // Read only access
    template <typename... Indices>
    inline T get(Indices... indices) const {
        int idx = to_linear_index(indices...);
        return storage[idx];
    }
    
    template <typename... Indices>
    T operator() (Indices... indices) const { 
        int idx = to_linear_index(indices...);
        return storage[idx];
    }
    
    // Writeable access
    template <typename... Indices>
    T& operator() (Indices... indices) {
        int idx = to_linear_index(indices...);
        return storage[idx];
    }

    T operator[] (int i) const { return storage[i]; }

    
    template <typename... Indices>
    void set(const T val, Indices... indices){
        int idx = to_linear_index(indices...);
        storage[idx] = val;
    }

    T sum() const {
        T acc = 0;
        #pragma omp simd reduction(+:acc)
        for (unsigned int i = 0; i < size; ++i){
            acc += storage[i];
        }
        return acc;
    }

    void add(const T val){
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] += val;
        }
    }
    
    void sub(const T val){
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] -= val;
        }
    }

    
    void mul(const T val){
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] *= val;
        }
    }
    
    void div(const T val){
        const T vali = 1 / val;
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] *= vali;
        }
    }
    
    void set_to_val(const T val){
        memset(storage, val, size * sizeof(T));
        //#pragma omp simd
        //for (unsigned int i = 0; i < size; ++i){
        //    storage[i] = val;
        //}
    }

    void set_to_zeros(){ set_to_val(0.0); }
    void set_to_ones(){ set_to_val(1.0); }
    T* data() const { return storage; }

    
    NDArrayBase& operator+=(const T val){
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] += val;
        }
        return *this;
    }

    NDArrayBase& operator+=(const NDArrayBase<T,N> &val){
        //const T* const val_data = val.data();
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] += val[i];
        }
        return *this;
    }

    NDArrayBase& operator-=(const T val){
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] -= val;
        }
        return *this;
    }

    NDArrayBase& operator-=(const NDArrayBase<T,N> &val){
        //const T* const val_data = val.data();
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] -= val[i];
        }
        return *this;
    }

    NDArrayBase& operator*=(const T val){
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] *= val;
        }
        return *this;
    }

    NDArrayBase& operator*=(const NDArrayBase<T,N> &val){
        //const T* const val_data = val.data();
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] *= val[i];
        }
        return *this;
    }

    NDArrayBase& operator/=(const T val){
        const T vali = 1 / val;
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] *= vali;
        }
        return *this;
    }

    NDArrayBase& operator/=(const NDArrayBase<T,N> &val){
        //const T* const val_data = val.data();
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] /= val[i];
        }
        return *this;
    }

    void exp(){
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] = std::exp(storage[i]);
        }
    }

    void log(){
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] = std::log(storage[i]);
        }
    }

    void sin(){
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] = std::sin(storage[i]);
        }
    }

    void cos(){
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] = std::cos(storage[i]);
        }
    }

    void tan(){
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] = std::tan(storage[i]);
        }
    }

    void sinh(){
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] = std::sinh(storage[i]);
        }
    }

    void cosh(){
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] = std::cosh(storage[i]);
        }
    }

    void tanh(){
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] = std::tanh(storage[i]);
        }
    }

    void pow(const T exponent){
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] = std::pow(storage[i], exponent);
        }
    }

    void abs(){
        #pragma omp simd
        for (unsigned int i = 0; i < size; ++i){
            storage[i] = std::abs(storage[i]);
        }
    }

    T maximum() const { return *std::max_element(storage, storage+size); }
    T minimum() const { return *std::min_element(storage, storage+size); }

    unsigned int shape(const unsigned int dim)
    {
        if (dim > size){
            throw(std::invalid_argument("tried to access non-existant dimension of NDArray"));
        }
        return dims[dim];
    }





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
    NDArrayMap<T,N>() { };
    
    template <typename... Dimensions>
    NDArrayMap<T,N>(T* arr, Dimensions... indices){
        int size_ = 1;
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
        
    }

    NDArrayMap<T,N>(T* arr, const int dims_[N], const int size_) { 
        this->set_storage(arr);
        this->set_dims(dims_);
        this->set_size(size_);
    }

    NDArrayMap<T,N>(std::vector<T> &arr, const int dims_[N]){
        this->set_storage(arr.data());
        this->set_dims(dims_);
        this->set_size(arr.size());
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
    template <typename... Dimensions>
    NDArray<T,N>(Dimensions... indices){
        static_assert(sizeof...(indices) == N, "Attempted to initialise NDArray array with invalid number of dimensions. "
                                                "Make sure that the number of dimensions passed to the construct and the template parameter N match.");
        int sz = 1;
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
    
    ~NDArray<T,N>() { this->delete_storage(); }
};



#endif