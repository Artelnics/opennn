//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef ADDITIONLAYER_H
#define ADDITIONLAYER_H

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
#include "convolutional_layer.h"
#include "layer.h"
#include "flatten_layer.h"

#include "statistics.h"

namespace opennn
{

struct AdditionLayerForwardPropagation;
struct AdditionLayerBackPropagation;

/// This class represents the Pooling Layer in Convolutional Neural Network(CNN).
/// Pooling: is the procross_entropy_errors of merging, ie, reducing the size of the data and remove some noise by different processes.

class AdditionLayer : public Layer
{

public:

    // Constructors

    explicit AdditionLayer();

    explicit AdditionLayer(const Tensor<Index, 1>&);

    explicit AdditionLayer(const Tensor<Index, 1>&, const Tensor<Index, 1>&);

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

    Index get_parameters_number() const;

    Tensor<type, 1> get_parameters() const;

    // Set methods

    void set(const Tensor<Index, 1>&, const Tensor<Index, 1>&);

    void set_inputs_number(const Index&) {}
    void set_neurons_number(const Index&) {}
    void set_name(const string&);

    void set_inputs_dimensions(const Tensor<Index, 1>&);

    void set_default();

    // Outputs

    // First order activations

    void forward_propagate(const Tensor<DynamicTensor<type>, 1>&,
                           LayerForwardPropagation*,
                           const bool&) final;

    // Delta methods

    void calculate_hidden_delta(LayerForwardPropagation*,
                                LayerBackPropagation*,
                                LayerBackPropagation*) const;

    // Serialization methods

    void from_XML(const tinyxml2::XMLDocument&) final;
    void write_XML(tinyxml2::XMLPrinter&) const final;

protected:

    Tensor<Index, 1> inputs_dimensions;

//#ifdef OPENNN_CUDA
//#include "../../opennn-cuda/opennn-cuda/pooling_layer_cuda.h"
//#endif

};


struct AdditionLayerForwardPropagation : LayerForwardPropagation
{
    // Default constructor

    explicit AdditionLayerForwardPropagation()
        : LayerForwardPropagation()
    {
    }

    // Constructor

    explicit AdditionLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }

    Eigen::array<ptrdiff_t, 4> get_inputs_dimensions_array() const
    {
        AdditionLayer* addition_layer_pointer = static_cast<AdditionLayer*>(layer_pointer);

        const Index inputs_rows_number = addition_layer_pointer->get_inputs_rows_number();
        const Index inputs_columns_number = addition_layer_pointer->get_inputs_columns_number();
        const Index inputs_channels_number = addition_layer_pointer->get_channels_number();

        return Eigen::array<ptrdiff_t, 4>({batch_samples_number,
                                           inputs_rows_number,
                                           inputs_columns_number,
                                           inputs_channels_number});
    }

    Eigen::array<ptrdiff_t, 4> get_outputs_dimensions_array() const
    {
        AdditionLayer* addition_layer_pointer = static_cast<AdditionLayer*>(layer_pointer);

        const Index oututs_columns_number =  addition_layer_pointer->get_outputs_columns_number();
        const Index oututs_rows_number = addition_layer_pointer->get_outputs_rows_number();
        const Index outputs_channels_number = addition_layer_pointer->get_channels_number();

        return Eigen::array<ptrdiff_t, 4>({batch_samples_number,
                                           oututs_rows_number,
                                           oututs_columns_number,
                                           outputs_channels_number});
    }

    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
        batch_samples_number = new_batch_samples_number;

        layer_pointer = new_layer_pointer;

        const AdditionLayer* addition_layer_pointer = static_cast<AdditionLayer*>(layer_pointer);

        const Index outputs_rows_number = addition_layer_pointer->get_outputs_rows_number();

        const Index outputs_columns_number = addition_layer_pointer->get_outputs_columns_number();

        const Index channels_number = addition_layer_pointer->get_channels_number();

        outputs.resize(1);
        Tensor<Index, 1> output_dimensions(4);
        output_dimensions.setValues({batch_samples_number,
                                     outputs_rows_number,
                                     outputs_columns_number,
                                     channels_number});
        outputs(0).set_dimensions(output_dimensions);
    }


    void print() const
    {
        cout << "Addition layer forward propagation" << endl;

        cout << "Outputs dimensions:" << endl;
        cout << outputs[0].get_dimensions() << endl;

        cout << "Outputs:" << endl;

        cout << outputs(0).to_tensor_map<4>() << endl;
     }
};


struct AdditionLayerBackPropagation : LayerBackPropagation
{

    explicit AdditionLayerBackPropagation() : LayerBackPropagation()
    {
    }

    virtual ~AdditionLayerBackPropagation()
    {
    }

    explicit AdditionLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
        batch_samples_number = new_batch_samples_number;

        layer_pointer = new_layer_pointer;


        const AdditionLayer* pooling_layer_pointer = static_cast<AdditionLayer*>(layer_pointer);

        const Index outputs_rows_number = pooling_layer_pointer->get_outputs_rows_number();
        const Index outputs_columns_number = pooling_layer_pointer->get_outputs_columns_number();

        deltas_dimensions.resize(4);
/*
        deltas_dimensions.setValues({batch_samples_number,
                                     kernels_number,
                                     outputs_rows_number,
                                     outputs_columns_number});

        deltas_data = (type*)malloc(static_cast<size_t>(batch_samples_number*kernels_number*outputs_rows_number*outputs_columns_number*sizeof(type)));
*/
    }


    void print() const
    {
        cout << "Deltas:" << endl;
        //cout << deltas << endl;

    }
};

//#ifdef OPENNN_CUDA
//    #include "../../opennn-cuda/opennn-cuda/struct_convolutional_layer_cuda.h"
//#endif


}

#endif // POOLING_LAYER_H
