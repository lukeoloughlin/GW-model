#ifndef NDARRAY_H
#define NDARRAY_H

#include <vector>
#include <assert.h>
#include <cmath>
#include <algorithm>
#include <cstring>


template <typename T, std::size_t N>
class NDArrayBase {
protected:
    T* storage;
    int dims[N];
    size_t size;

    bool is_dynamically_allocated;

    void set_storage(T* storage_){ storage = storage_; }

    void set_dims(const int dims_[N]){
        for (size_t i = 0; i < N; ++i)
            dims[i] = dims_[i];
    }

    void set_size(size_t size_){ size = size_; }

    void delete_storage() {
        if (is_dynamically_allocated) // Will segfault if you try to call delete[] if not dynamically allocated
            delete[] storage; 
    }

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
    inline T operator() (Indices... indices) const { 
        int idx = to_linear_index(indices...);
        return storage[idx];
    }
    
    // Writeable access
    template <typename... Indices>
    inline T& operator() (Indices... indices) {
        int idx = to_linear_index(indices...);
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

    // move assignment
    NDArray& operator=(NDArray&& other) noexcept {
        // Guard self assignment
        if (this == &other)
            return *this;
        
        if (this->is_dynamically_allocated != other.is_dynamically_allocated)
            exit(EXIT_FAILURE);
 
        delete_storage();   // release resource in *this if appropriate
        storage = std::exchange(other.storage, nullptr); // leave other in valid state
        dims = std::exchange(other.dims, nullptr);
        size = std::exchange(other.size, 0);
        return *this;
    }

    
    template <typename... Indices>
    inline void set(const T val, Indices... indices){
        int idx = to_linear_index(indices...);
        storage[idx] = val;
    }

    inline T sum() const; 
    inline void add(const T val);
    inline void sub(const T val);    
    inline void mul(const T val);
    inline void div(const T val);
    inline void set_to(const T val);
    inline void set_to_zeros(){ set_to_val(0); }
    inline void set_to_ones(){ set_to_val(1); }

    T* data() const { return storage; }

    inline void exp();
    inline void log();
    inline void sin();
    inline void cos();
    inline void tan();
    inline void sinh();
    inline void cosh();
    inline void tanh();
    inline void pow(const T exponent);
    inline void abs();

    template <typename LambdaType>
    inline void map(LambdaType&& fn);

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
    
template <typename T, size_t N>
inline NDArrayBase<T,N> operator+(const NDArrayBase<T,N>&, const NDArrayBase<T,N>&); // Add two lvalue references
template <typename T, size_t N>
inline NDArrayBase<T,N> operator+(const NDArrayBase<T,N>&, const T&);
template <typename T, size_t N>
inline NDArrayBase<T,N> operator+(const T&, const NDArrayBase<T,N>&);

template <typename T, size_t N>
inline NDArrayBase<T,N> operator+(NDArrayBase<T,N>&&, const NDArrayBase<T,N>&); // Add an rvalue reference to an lvalue reference on the left
template <typename T, size_t N>
inline NDArrayBase<T,N> operator+(NDArrayBase<T,N>&&, const T&); 
template <typename T, size_t N>
inline NDArrayBase<T,N> operator+(T&&, const NDArrayBase<T,N>&); 

template <typename T, size_t N>
inline NDArrayBase<T,N> operator+(const NDArrayBase<T,N>&, NDArrayBase<T,N>&&); // Add an lvalue reference to an rvalue reference on the right
template <typename T, size_t N>
inline NDArrayBase<T,N> operator+(const NDArrayBase<T,N>&, T&&);
template <typename T, size_t N>
inline NDArrayBase<T,N> operator+(const T&, NDArrayBase<T,N>&&);

template <typename T, size_t N>
inline NDArrayBase<T,N> operator+(NDArrayBase<T,N>&&, NDArrayBase<T,N>&&); // Add two rvalue references
template <typename T, size_t N>
inline NDArrayBase<T,N> operator+(NDArrayBase<T,N>&&, T&&); 
template <typename T, size_t N>
inline NDArrayBase<T,N> operator+(T&&, NDArrayBase<T,N>&&);



template <typename T, size_t N>
inline NDArrayBase<T,N> operator-(const NDArrayBase<T,N>&, const NDArrayBase<T,N>&); // Add two lvalue references
template <typename T, size_t N>
inline NDArrayBase<T,N> operator-(const NDArrayBase<T,N>&, const T&);
template <typename T, size_t N>
inline NDArrayBase<T,N> operator-(const T&, const NDArrayBase<T,N>&);

template <typename T, size_t N>
inline NDArrayBase<T,N> operator-(NDArrayBase<T,N>&&, const NDArrayBase<T,N>&); // Add an rvalue reference to an lvalue reference on the left
template <typename T, size_t N>
inline NDArrayBase<T,N> operator-(NDArrayBase<T,N>&&, const T&); 
template <typename T, size_t N>
inline NDArrayBase<T,N> operator-(T&&, const NDArrayBase<T,N>&); 

template <typename T, size_t N>
inline NDArrayBase<T,N> operator-(const NDArrayBase<T,N>&, NDArrayBase<T,N>&&); // Add an lvalue reference to an rvalue reference on the right
template <typename T, size_t N>
inline NDArrayBase<T,N> operator-(const NDArrayBase<T,N>&, T&&);
template <typename T, size_t N>
inline NDArrayBase<T,N> operator-(const T&, NDArrayBase<T,N>&&);

template <typename T, size_t N>
inline NDArrayBase<T,N> operator-(NDArrayBase<T,N>&&, NDArrayBase<T,N>&&); // Add two rvalue references
template <typename T, size_t N>
inline NDArrayBase<T,N> operator-(NDArrayBase<T,N>&&, T&&); 
template <typename T, size_t N>
inline NDArrayBase<T,N> operator-(T&&, NDArrayBase<T,N>&&);



template <typename T, size_t N>
inline NDArrayBase<T,N> operator*(const NDArrayBase<T,N>&, const NDArrayBase<T,N>&); // Add two lvalue references
template <typename T, size_t N>
inline NDArrayBase<T,N> operator*(const NDArrayBase<T,N>&, const T&);
template <typename T, size_t N>
inline NDArrayBase<T,N> operator*(const T&, const NDArrayBase<T,N>&);

template <typename T, size_t N>
inline NDArrayBase<T,N> operator*(NDArrayBase<T,N>&&, const NDArrayBase<T,N>&); // Add an rvalue reference to an lvalue reference on the left
template <typename T, size_t N>
inline NDArrayBase<T,N> operator*(NDArrayBase<T,N>&&, const T&); 
template <typename T, size_t N>
inline NDArrayBase<T,N> operator*(T&&, const NDArrayBase<T,N>&); 

template <typename T, size_t N>
inline NDArrayBase<T,N> operator*(const NDArrayBase<T,N>&, NDArrayBase<T,N>&&); // Add an lvalue reference to an rvalue reference on the right
template <typename T, size_t N>
inline NDArrayBase<T,N> operator*(const NDArrayBase<T,N>&, T&&);
template <typename T, size_t N>
inline NDArrayBase<T,N> operator*(const T&, NDArrayBase<T,N>&&);

template <typename T, size_t N>
inline NDArrayBase<T,N> operator*(NDArrayBase<T,N>&&, NDArrayBase<T,N>&&); // Add two rvalue references
template <typename T, size_t N>
inline NDArrayBase<T,N> operator*(NDArrayBase<T,N>&&, T&&); 
template <typename T, size_t N>
inline NDArrayBase<T,N> operator*(T&&, NDArrayBase<T,N>&&);



template <typename T, size_t N>
inline NDArrayBase<T,N> operator/(const NDArrayBase<T,N>&, const NDArrayBase<T,N>&); // Add two lvalue references
template <typename T, size_t N>
inline NDArrayBase<T,N> operator/(const NDArrayBase<T,N>&, const T&);
template <typename T, size_t N>
inline NDArrayBase<T,N> operator/(const T&, const NDArrayBase<T,N>&);

template <typename T, size_t N>
inline NDArrayBase<T,N> operator/(NDArrayBase<T,N>&&, const NDArrayBase<T,N>&); // Add an rvalue reference to an lvalue reference on the left
template <typename T, size_t N>
inline NDArrayBase<T,N> operator/(NDArrayBase<T,N>&&, const T&); 
template <typename T, size_t N>
inline NDArrayBase<T,N> operator/(T&&, const NDArrayBase<T,N>&); 

template <typename T, size_t N>
inline NDArrayBase<T,N> operator/(const NDArrayBase<T,N>&, NDArrayBase<T,N>&&); // Add an lvalue reference to an rvalue reference on the right
template <typename T, size_t N>
inline NDArrayBase<T,N> operator/(const NDArrayBase<T,N>&, T&&);
template <typename T, size_t N>
inline NDArrayBase<T,N> operator/(const T&, NDArrayBase<T,N>&&);

template <typename T, size_t N>
inline NDArrayBase<T,N> operator/(NDArrayBase<T,N>&&, NDArrayBase<T,N>&&); // Add two rvalue references
template <typename T, size_t N>
inline NDArrayBase<T,N> operator/(NDArrayBase<T,N>&&, T&&); 
template <typename T, size_t N>
inline NDArrayBase<T,N> operator/(T&&, NDArrayBase<T,N>&&);

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
        this->is_dynamically_allocated = false;
        
    }

    NDArrayMap<T,N>(T* arr, const int dims_[N], const int size_) { 
        this->set_storage(arr);
        this->set_dims(dims_);
        this->set_size(size_);
        this->is_dynamically_allocated = false;
    }

    NDArrayMap<T,N>(std::vector<T> &arr, const int dims_[N]){
        this->set_storage(arr.data());
        this->set_dims(dims_);
        this->set_size(arr.size());
        this->is_dynamically_allocated = false;
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
        this->is_dynamically_allocated = true;

        this->set_to_zeros();
    }
    
    ~NDArray<T,N>() { this->delete_storage(); }
};



#endif