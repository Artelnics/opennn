//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H

//// System includes

#include <cmath>
#include <cstdlib>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "config.h"
//#include "convolutional_layer.h"
#include "layer.h"
#include "flatten_layer.h"

#include "statistics.h"

namespace opennn
{

class ConvolutionalLayer;

struct PoolingLayerForwardPropagation;
struct PoolingLayerBackPropagation;

/// This class represents the Pooling Layer in Convolutional Neural Network(CNN).
/// Pooling: is the procross_entropy_errors of merging, ie, reducing the size of the data and remove some noise by different processes.

class PoolingLayer : public Layer
{

public:

    /// Enumeration of the available methods for pooling data.

    enum class PoolingMethod{NoPooling, MaxPooling, AveragePooling};

    // Constructors

    explicit PoolingLayer();

    explicit PoolingLayer(const Tensor<Index, 1>&);

    explicit PoolingLayer(const Tensor<Index, 1>&, const Tensor<Index, 1>&);

    // Get methods

    Tensor<Index, 1> get_inputs_dimensions() const;
    Tensor<Index, 1> get_outputs_dimensions() const;

    Index get_inputs_number() const;

    Index get_channels_number() const;

    Index get_inputs_rows_number() const;

    Index get_inputs_columns_number() const;

    Index get_neurons_number() const;

    Index get_outputs_rows_number() const;

    Index get_outputs_columns_number() const;

    Index get_padding_width() const;

    Index get_row_stride() const;

    Index get_column_stride() const;

    Index get_pool_rows_number() const;

    Index get_pool_columns_number() const;

    Index get_parameters_number() const;

    Tensor<type, 1> get_parameters() const;

    PoolingMethod get_pooling_method() const;

    string write_pooling_method() const;

    // Set methods

    void set_inputs_number(const Index&) {}
    void set_neurons_number(const Index&) {}

    void set_inputs_dimensions(const Tensor<Index, 1>&);

    void set_padding_width(const Index&);

    void set_row_stride(const Index&);

    void set_column_stride(const Index&);

    void set_pool_size(const Index&, const Index&);

    void set_pooling_method(const PoolingMethod&);
    void set_pooling_method(const string&);

    void set_name(const string& new_layer_name);

    void set_default();

    // Outputs

    Tensor<type, 4> calculate_no_pooling_outputs(const Tensor<type, 4>&) const;

    Tensor<type, 4> calculate_max_pooling_outputs(const Tensor<type, 4>&) const;

    Tensor<type, 4> calculate_average_pooling_outputs(const Tensor<type, 4>&) const;

    // First order activations

    void forward_propagate(type*,
                           const Tensor<Index, 1>&,
                           LayerForwardPropagation*,
                           const bool&) final;

    void forward_propagate_no_pooling(type*,
                           const Tensor<Index, 1>&,
                           LayerForwardPropagation*,
                           const bool&);

    void forward_propagate_max_pooling(type*,
                           const Tensor<Index, 1>&,
                           LayerForwardPropagation*,
                           const bool&);

    void forward_propagate_average_pooling(type*,
                           const Tensor<Index, 1>&,
                           LayerForwardPropagation*,
                           const bool&);

    // Delta methods

    void calculate_hidden_delta(LayerForwardPropagation*,
                                LayerBackPropagation*,
                                LayerBackPropagation*) const;

    void calculate_hidden_delta_convolutional(LayerForwardPropagation*,
                                LayerBackPropagation*,
                                LayerBackPropagation*) const;

    void calculate_hidden_delta_pooling(LayerForwardPropagation*,
                                LayerBackPropagation*,
                                LayerBackPropagation*) const;

    void calculate_hidden_delta_flatten(LayerForwardPropagation*,
                                LayerBackPropagation*,
                                LayerBackPropagation*) const;

    // Serialization methods

    void from_XML(const tinyxml2::XMLDocument&) final;
    void write_XML(tinyxml2::XMLPrinter&) const final;

protected:

    Tensor<Index, 1> inputs_dimensions;

    Index pool_rows_number = 2;

    Index pool_columns_number = 2;

    Index padding_width = 0;

    Index row_stride = 1;

    Index column_stride = 1;

    PoolingMethod pooling_method = PoolingMethod::AveragePooling;

    const Eigen::array<ptrdiff_t, 4> convolution_dimensions = {0, 1, 2, 3};

    const Eigen::array<ptrdiff_t, 4> patch_dimensions ={1, pool_rows_number, pool_columns_number, 1};

    const Eigen::array<ptrdiff_t, 2> max_pooling_reduce_dimensions ={1, 2};

#ifdef OPENNN_CUDA
#include "../../opennn-cuda/opennn-cuda/pooling_layer_cuda.h"
#endif

};


struct PoolingLayerForwardPropagation : LayerForwardPropagation
{
    // Default constructor

    explicit PoolingLayerForwardPropagation()
        : LayerForwardPropagation()
    {
    }

    // Constructor

    explicit PoolingLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }

    Eigen::array<ptrdiff_t, 4> get_inputs_dimensions_array() const
    {
        PoolingLayer* pooling_layer_pointer = static_cast<PoolingLayer*>(layer_pointer);

        const Index inputs_rows_number = pooling_layer_pointer->get_inputs_rows_number();
        const Index inputs_columns_number = pooling_layer_pointer->get_inputs_columns_number();
        const Index inputs_channels_number = pooling_layer_pointer->get_channels_number();

        return Eigen::array<ptrdiff_t, 4>({batch_samples_number,
                                           inputs_rows_number,
                                           inputs_columns_number,
                                           inputs_channels_number});
    }

    Eigen::array<ptrdiff_t, 4> get_outputs_dimensions_array() const
    {
        PoolingLayer* pooling_layer_pointer = static_cast<PoolingLayer*>(layer_pointer);

        const Index oututs_columns_number =  pooling_layer_pointer->get_outputs_columns_number();
        const Index oututs_rows_number = pooling_layer_pointer->get_outputs_rows_number();
        const Index oututs_channels_number = pooling_layer_pointer->get_channels_number();

        return Eigen::array<ptrdiff_t, 4>({batch_samples_number,
                                           oututs_columns_number,
                                           oututs_rows_number,
                                           oututs_channels_number});
    }

    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
        batch_samples_number = new_batch_samples_number;

        layer_pointer = new_layer_pointer;

        const PoolingLayer* pooling_layer_pointer = static_cast<PoolingLayer*>(layer_pointer);

        const Index outputs_rows_number = pooling_layer_pointer->get_outputs_rows_number();

        const Index outputs_columns_number = pooling_layer_pointer->get_outputs_columns_number();

        const Index channels_number = pooling_layer_pointer->get_channels_number();

        const Index patches_number = 0;

        outputs_dimensions.resize(4);

        outputs_dimensions.setValues({batch_samples_number,
                                      outputs_rows_number,
                                      outputs_columns_number,
                                      channels_number});

        outputs_data = (type*) malloc(static_cast<size_t>(batch_samples_number
                                                          *outputs_rows_number
                                                          *outputs_columns_number
                                                          *channels_number*sizeof(type)));



        patches.resize(patches_number, 2, 2, 2, 2); ///@todo extract_patches
    }


    void print() const
    {
        cout << "Pooling layer forward propagation" << endl;

        cout << "Outputs dimensions:" << endl;
        cout << outputs_dimensions << endl;

        cout << "Outputs:" << endl;

        cout << TensorMap<Tensor<type,4>>(outputs_data, get_outputs_dimensions_array()) << endl;

        cout << "Patches" << endl;
        cout << patches << endl;
     }

    Tensor<type, 5> patches;
};


struct PoolingLayerBackPropagation : LayerBackPropagation
{

    explicit PoolingLayerBackPropagation() : LayerBackPropagation()
    {
    }

    virtual ~PoolingLayerBackPropagation()
    {
    }

    explicit PoolingLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
        batch_samples_number = new_batch_samples_number;

        layer_pointer = new_layer_pointer;

        const PoolingLayer* pooling_layer_pointer = static_cast<PoolingLayer*>(layer_pointer);

        const Index outputs_rows_number = pooling_layer_pointer->get_outputs_rows_number();
        const Index outputs_columns_number = pooling_layer_pointer->get_outputs_columns_number();

        deltas_dimensions.resize(4);
/*
        deltas_dimensions.setValues({batch_samples_number,
                                     kernels_number,
                                     outputs_rows_number,
                                     outputs_columns_number});

        deltas_data = (type*)malloc(static_cast<size_t>(batch_samples_number*kernels_number*outputs_rows_number*outputs_columns_number*sizeof(type)));

        deltas_times_activations_derivatives.resize(batch_samples_number,
                                                    kernels_number,
                                                    outputs_rows_number,
                                                    outputs_columns_number);
*/
    }


    void print() const
    {
        cout << "Deltas:" << endl;
        //cout << deltas << endl;

    }
};

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/struct_convolutional_layer_cuda.h"
#endif


}

#endif // POOLING_LAYER_H
