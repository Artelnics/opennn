//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef ADDITIONLAYER_H
#define ADDITIONLAYER_H

// System includes

#include <cmath>
#include <cstdlib>
#include <ctype.h>
#include <fstream>
#include <sstream>

#include <iostream>
#include <string>

// OpenNN includes

#include "config.h"
#include "layer.h"

#include "convolutional_layer.h"
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

    Index get_inputs_raw_variables_number() const;

    Index get_neurons_number() const;

    Index get_outputs_rows_number() const;

    Index get_outputs_raw_variables_number() const;

    Index get_parameters_number() const final;

    Tensor<type, 1> get_parameters() const final;

    // Set methods

    void set(const Tensor<Index, 1>&, const Tensor<Index, 1>&);

    void set_inputs_number(const Index&) {}
    void set_neurons_number(const Index&) {}
    void set_name(const string&);

    void set_inputs_dimensions(const Tensor<Index, 1>&);

    void set_default();

    // Outputs

    // First order activations

    void forward_propagate(const pair<type*, dimensions>&,
                           LayerForwardPropagation*,
                           const bool&) final;

    // Delta methods

    void calculate_hidden_delta(LayerForwardPropagation*,
                                LayerBackPropagation*,
                                LayerBackPropagation*) const final;

    // Serialization methods

    void from_XML(const tinyxml2::XMLDocument&) final;
    void write_XML(tinyxml2::XMLPrinter&) const final;

protected:

    Tensor<Index, 1> inputs_dimensions;

//#ifdef OPENNN_CUDA
//#include "../../opennn-cuda/opennn-cuda/addition_layer_cuda.h"
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

    explicit AdditionLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer) final
    {
        batch_samples_number = new_batch_samples_number;

        layer = new_layer;

        const AdditionLayer* addition_layer = static_cast<AdditionLayer*>(layer);

        const Index outputs_rows_number = addition_layer->get_outputs_rows_number();

        const Index outputs_raw_variables_number = addition_layer->get_outputs_raw_variables_number();

        const Index channels_number = addition_layer->get_channels_number();

        outputs.resize(batch_samples_number,
                       outputs_rows_number,
                       outputs_raw_variables_number,
                       channels_number);
    }


    void print() const
    {
        cout << "Addition layer forward propagation" << endl;

        cout << "Outputs:" << endl;
        cout << outputs(0) << endl;
     }

    Tensor<type, 4> outputs;
};


struct AdditionLayerBackPropagation : LayerBackPropagation
{

    explicit AdditionLayerBackPropagation() : LayerBackPropagation()
    {
    }

    virtual ~AdditionLayerBackPropagation()
    {
    }

    explicit AdditionLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer);
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer) final
    {
        batch_samples_number = new_batch_samples_number;

        layer = new_layer;
    }


    void print() const
    {
        cout << "Deltas:" << endl;
        //cout << deltas << endl;
    }
};

//#ifdef OPENNN_CUDA
//    #include "../../opennn-cuda/opennn-cuda/struct_addition_layer_cuda.h"
//#endif


}

#endif // POOLING_LAYER_H
