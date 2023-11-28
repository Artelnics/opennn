//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D Y N A M I C  T E N S O R  C L A S S  H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com


#ifndef DYNAMIC_TENSOR_H
#define DYNAMIC_TENSOR_H


#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

#include "config.h"
#include "tensor_utilities.h"

namespace opennn{


template <typename T>
class DynamicTensor {


public:

    DynamicTensor()
    {

    }

    DynamicTensor(const Tensor<Index, 1>& new_dimensions)
    {
        set_dimensions(new_dimensions);
    }

    DynamicTensor(const initializer_list<Index>& new_dimensions_array)
    {
        Tensor<Index, 1> new_dimensions_tensor(new_dimensions_array.size());
        new_dimensions_tensor.setValues(new_dimensions_array);

        set_dimensions(new_dimensions_tensor);
    }


    DynamicTensor& operator = (const DynamicTensor& other)
    {
        if(this != &other)
        {
            const Tensor<bool, 0> different_dimensions = (dimensions != other.dimensions).all();

            if(different_dimensions(0))
            {
                dimensions = other.dimensions;
            }

            if(data != nullptr)
            {
                free(data);
            }

            const Tensor<Index, 0> size = dimensions.prod();

            data = (T*) malloc(static_cast<size_t>(size(0)*sizeof(T)));

            memcpy(data, other.data, static_cast<size_t>(size(0)*sizeof(T)) );
        }

        return *this;
    }

    bool operator != (DynamicTensor& other)
    {
        if((dimensions != other.dimensions)(0))
        {
            return true;
        }

        const Tensor<Index, 0> size = dimensions.prod();

        for(Index i = 0; i < size(0); i++)
        {
            if(*(data + i) != *(other.data + i))
            {
                return true;
            }
        }

        return false;
    }


    ostream& operator << (ostream& os)
    {

    }

/*
    DynamicTensor(const T* new_data, const Tensor<Index, 1>& new_dimensions)
    {
        /// @todo check if data size matches dimensions
        data = (T*) new_data;

        dimensions = new_dimensions;
    }

    DynamicTensor(const Tensor<T, 2>& new_tensor_2)
    {
        dimensions = opennn::get_dimensions(new_tensor_2);

        data = (T*) new_tensor_2.data();
    }

    DynamicTensor(const Tensor<T, 4>& new_tensor_4)
    {
        dimensions = opennn::get_dimensions(new_tensor_4);

        data = (T*) new_tensor_4.data();
    }
*/
    virtual ~DynamicTensor()
    {
        free(data);
    }

    T* get_data() const
    {
        return data;
    }

    const Tensor<Index, 1>& get_dimensions() const
    {
        return dimensions;
    }


    Index get_dimension(const Index& index) const
    {
        return dimensions(index);
    }

    void set_data(const T* new_data)
    {
        free(data);

        data = (T*) new_data;
    }


    void set_dimensions(const Tensor<Index, 1> new_dimensions)
    {
        free(data);

        dimensions = new_dimensions;

        const Tensor<Index, 0> size = dimensions.prod();

        data = (T*) malloc(static_cast<size_t>(size(0)*sizeof(T)));
    }

    template <int rank>
    TensorMap<Tensor<T, rank>> to_tensor_map() const
    {
//        const int rank = dimensions.size();

        std::array<Index, rank> sizes;

        for (Index i = 0; i < dimensions.size(); ++i) {
            sizes[i] = dimensions(i);
        }

        return TensorMap<Tensor<T, rank>>(data, sizes);
    }

private:

    T* data = nullptr;

    Tensor<Index, 1> dimensions;
};
};


#endif // DYNAMIC_TENSOR_H
