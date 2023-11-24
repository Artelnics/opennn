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

    DynamicTensor& operator = (const DynamicTensor& other)
    {
        if (this != &other)
        {
            dimensions = other.dimensions;
            data = other.data;
        }

        return *this;
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

    const Tensor<Index, 1> get_dimensions() const
    {
        return dimensions;
    }


    Index get_dimension(const Index& index) const
    {
        return dimensions(index);
    }

    void set_data(const T* new_data)
    {
        data = (T*) new_data;
    }


    void set_dimensions(const Tensor<Index, 1> new_dimensions)
    {
        free(data);

        dimensions = new_dimensions;

        const Tensor<Index, 0> size = dimensions.prod();

        data = (T*) malloc(static_cast<size_t>(size(0)*sizeof(T)));
    }


    TensorMap<Tensor<T, 2>> to_tensor_map_2() const
    {
        return TensorMap<Tensor<T, 2>>(data, dimensions(0), dimensions(1));
    }

    TensorMap<Tensor<T, 3>> to_tensor_map_3() const
    {
        return TensorMap<Tensor<T, 3>>(data, dimensions(0), dimensions(1), dimensions(2));
    }

    TensorMap<Tensor<T, 4>> to_tensor_map_4() const
    {
        return TensorMap<Tensor<T, 4>>(data, dimensions(0), dimensions(1), dimensions(2), dimensions(3));
    }


private:

    T* data = nullptr;

    Tensor<Index, 1> dimensions;
};
};


#endif // DYNAMIC_TENSOR_H
