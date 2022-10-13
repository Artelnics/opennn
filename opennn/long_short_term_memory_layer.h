//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O N G   S H O R T   T E R M   M E M O R Y   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LONGSHORTTERMMEMORYLAYER_H
#define LONGSHORTTERMMEMORYLAYER_H

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "config.h"
#include "tensor_utilities.h"
#include "layer.h"

#include "probabilistic_layer.h"
#include "perceptron_layer.h"

namespace opennn
{

struct LongShortTermMemoryLayerForwardPropagation;
struct LongShortTermMemoryLayerBackPropagation;


/// This class represents a layer of neurons.
/// Layers of neurons will be used to construct multilayer neurons.

class LongShortTermMemoryLayer : public Layer
{

public:

    /// Enumeration of the available activation functions for the long-short-term memory layer.

    enum class ActivationFunction{Threshold, SymmetricThreshold, Logistic, HyperbolicTangent,
                            Linear, RectifiedLinear, ExponentialLinear, ScaledExponentialLinear,
                            SoftPlus, SoftSign, HardSigmoid};

   // Constructors

   explicit LongShortTermMemoryLayer();

   explicit LongShortTermMemoryLayer(const Index&, const Index&);


   // Get methods

   bool is_empty() const;

   Index get_inputs_number() const final;
   Index get_neurons_number() const final;

   // Parameters

   Tensor<type, 1> get_input_biases() const;
   Tensor<type, 1> get_forget_biases() const;
   Tensor<type, 1> get_state_biases() const;
   Tensor<type, 1> get_output_biases() const;

   Tensor<type, 2> get_input_weights() const;
   Tensor<type, 2> get_forget_weights() const;
   Tensor<type, 2> get_state_weights() const;
   Tensor<type, 2> get_output_weights() const;

   Tensor<type, 2> get_input_recurrent_weights() const;
   Tensor<type, 2> get_forget_recurrent_weights() const;
   Tensor<type, 2> get_state_recurrent_weights() const;
   Tensor<type, 2> get_output_recurrent_weights() const;

   Index get_timesteps() const;

   Index get_parameters_number() const override;
   Tensor<type, 1> get_parameters() const final;

   Tensor< TensorMap< Tensor<type, 1> >*, 1> get_layer_parameters() final;

   // Activation functions

   const LongShortTermMemoryLayer::ActivationFunction& get_activation_function() const;
   const LongShortTermMemoryLayer::ActivationFunction& get_recurrent_activation_function() const;

   string write_activation_function() const;
   string write_recurrent_activation_function() const;
   // Display messages

   const bool& get_display() const;

   // Set methods

   void set();
   void set(const Index&, const Index&);
   void set(const LongShortTermMemoryLayer&);

   void set_default();
   void set_name(const string&);

   // Architecture

   void set_inputs_number(const Index&) final;
   void set_neurons_number(const Index&) final;
   void set_input_shape(const Tensor<Index, 1>&);

   // Parameters

   void set_input_biases(const Tensor<type, 1>&);
   void set_forget_biases(const Tensor<type, 1>&);
   void set_state_biases(const Tensor<type, 1>&);
   void set_output_biases(const Tensor<type, 1>&);

   void set_input_weights(const Tensor<type, 2>&);
   void set_forget_weights(const Tensor<type, 2>&);
   void set_state_weights(const Tensor<type, 2>&);
   void set_output_weights(const Tensor<type, 2>&);

   void set_input_recurrent_weights(const Tensor<type, 2>&);
   void set_forget_recurrent_weights(const Tensor<type, 2>&);
   void set_state_recurrent_weights(const Tensor<type, 2>&);
   void set_output_recurrent_weights(const Tensor<type, 2>&);

   void set_parameters(const Tensor<type, 1>&, const Index& = 0) final;

   // Activation functions

   void set_activation_function(const ActivationFunction&);
   void set_activation_function(const string&);

   void set_recurrent_activation_function(const ActivationFunction&);
   void set_recurrent_activation_function(const string&);

   void set_timesteps(const Index&);

   // Display messages

   void set_display(const bool&);

   // Parameters initialization methods

   void set_biases_constant(const type&);

   void set_forget_biases_constant(const type&);
   void set_input_biases_constant(const type&);
   void set_state_biases_constant(const type&);
   void set_output_biases_constant(const type&);

   void set_weights_constant(const type&);

   void set_forget_weights_constant(const type&);
   void set_input_weights_constant(const type&);
   void set_state_weights_constant(const type&);
   void set_output_weights_constant(const type&);

   void set_recurrent_weights_constant(const type&);

   void set_forget_recurrent_weights_constant(const type&);
   void set_input_recurrent_weights_constant(const type&);
   void set_state_recurrent_weights_constant(const type&);
   void set_output_recurrent_weights_constant(const type&);

   void set_hidden_states_constant(const type&);
   void set_cell_states_constant(const type&);

   void set_parameters_constant(const type&) final;

   void set_parameters_random() final;

   // Long short-term memory layer combinations

   void calculate_combinations(type*, const Tensor<Index, 1>&,
                               const Tensor<type, 2>&,
                               const Tensor<type, 2>&,
                               const Tensor<type, 1>&,
                               type*, const Tensor<Index, 1>&);

   // Long short-term memory layer activations

   void calculate_activations(type*, Tensor<Index,1>&, type*, Tensor<Index,1>&);

   Tensor<type, 1> calculate_activations(Tensor<type, 1>&) const;

   void calculate_recurrent_activations(type*, Tensor<Index, 1>&, type*, Tensor<Index, 1>&);

   // Long short-term memory layer derivatives

   void calculate_activations_derivatives(type*, Tensor<Index, 1>&,
                                          type*, Tensor<Index, 1>&,
                                          type*, Tensor<Index, 1>&);

   void calculate_recurrent_activations_derivatives(type*, Tensor<Index, 1>&,
                                          type*, Tensor<Index, 1>&,
                                          type*, Tensor<Index, 1>&);

   // Long short-term memory layer outputs

   void calculate_outputs(type*, const Tensor<Index, 1>&, type*, const Tensor<Index, 1>&) final;

   void calculate_hidden_delta(LayerForwardPropagation*,
                               LayerBackPropagation*,
                               LayerBackPropagation*) const final;


   void calculate_hidden_delta_perceptron(PerceptronLayerForwardPropagation*,
                                          PerceptronLayerBackPropagation*,
                                          LongShortTermMemoryLayerBackPropagation*) const;


   void calculate_hidden_delta_probabilistic(ProbabilisticLayerForwardPropagation*,
                                             ProbabilisticLayerBackPropagation*,
                                             LongShortTermMemoryLayerBackPropagation*) const;

   // Forward propagate

   void forward_propagate(type*, const Tensor<Index, 1>&, LayerForwardPropagation*) final;

   void forward_propagate(type*, const Tensor<Index, 1>&, Tensor<type, 1>&, LayerForwardPropagation*) final;

   // Eror gradient

   void insert_gradient(LayerBackPropagation*, const Index& , Tensor<type, 1>&) const final;

   void calculate_error_gradient(type*, LayerForwardPropagation*, LayerBackPropagation*) const final;

   void calculate_forget_weights_error_gradient(const Tensor<type, 2>&,
                                                LongShortTermMemoryLayerForwardPropagation*,
                                                LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_input_weights_error_gradient(const Tensor<type, 2>&,
                                               LongShortTermMemoryLayerForwardPropagation*,
                                               LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_state_weights_error_gradient(const Tensor<type, 2>&,
                                               LongShortTermMemoryLayerForwardPropagation*,
                                               LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_output_weights_error_gradient(const Tensor<type, 2>&,
                                                LongShortTermMemoryLayerForwardPropagation*,
                                                LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_forget_recurrent_weights_error_gradient(const Tensor<type, 2>&,
                                                          LongShortTermMemoryLayerForwardPropagation*,
                                                          LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_input_recurrent_weights_error_gradient(const Tensor<type, 2>&,
                                                         LongShortTermMemoryLayerForwardPropagation*,
                                                         LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_state_recurrent_weights_error_gradient(const Tensor<type, 2>&,
                                                         LongShortTermMemoryLayerForwardPropagation*,
                                                         LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_output_recurrent_weights_error_gradient(const Tensor<type, 2>&,
                                                          LongShortTermMemoryLayerForwardPropagation*,
                                                          LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_forget_biases_error_gradient(const Tensor<type, 2>&,
                                               LongShortTermMemoryLayerForwardPropagation*,
                                               LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_input_biases_error_gradient(const Tensor<type, 2>&,
                                              LongShortTermMemoryLayerForwardPropagation*,
                                              LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_state_biases_error_gradient(const Tensor<type, 2>&,
                                              LongShortTermMemoryLayerForwardPropagation*,
                                              LongShortTermMemoryLayerBackPropagation*) const;

   void calculate_output_biases_error_gradient(const Tensor<type, 2>&,
                                               LongShortTermMemoryLayerForwardPropagation*,
                                               LongShortTermMemoryLayerBackPropagation*) const;

   // Expression methods

   string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const final;
   string write_recurrent_activation_function_expression() const;
   string write_activation_function_expression() const;

   string write_expression_c() const final;
   string write_combinations_c() const;

   string write_expression_python() const final;
   string write_combinations_python() const;

   // Serialization methods

   void from_XML(const tinyxml2::XMLDocument&) final;

   void write_XML(tinyxml2::XMLPrinter&) const final;

protected:

   Index timesteps = 3;

   Tensor<type, 1> input_biases;
   Tensor<type, 1> forget_biases;
   Tensor<type, 1> state_biases;
   Tensor<type, 1> output_biases;

   Tensor<type, 2> input_weights;
   Tensor<type, 2> forget_weights;
   Tensor<type, 2> state_weights;
   Tensor<type, 2> output_weights;

   Tensor<type, 2> forget_recurrent_weights;
   Tensor<type, 2> input_recurrent_weights;
   Tensor<type, 2> state_recurrent_weights;
   Tensor<type, 2> output_recurrent_weights;

   /// Activation function variable.

   ActivationFunction activation_function = ActivationFunction::HyperbolicTangent;
   ActivationFunction recurrent_activation_function = ActivationFunction::HardSigmoid;

   Index batch;
   Index variables;

   Tensor<type, 1> hidden_states;
   Tensor<type, 1> cell_states;

   /// Display messages to screen.

   bool display = true;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/long_short_term_memory_layer_cuda.h"
#endif

};


struct LongShortTermMemoryLayerForwardPropagation : LayerForwardPropagation
{
    explicit LongShortTermMemoryLayerForwardPropagation() : LayerForwardPropagation()
    {
    }

    explicit LongShortTermMemoryLayerForwardPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerForwardPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
        layer_pointer = new_layer_pointer;

        const Index inputs_number = layer_pointer->get_inputs_number();
        const Index neurons_number = layer_pointer->get_neurons_number();

        batch_samples_number = new_batch_samples_number;

        // Outputs

        outputs_dimensions.resize(2);
        outputs_dimensions.setValues({batch_samples_number, neurons_number});

        //delete outputs_data;

        outputs_data = (type*)malloc(static_cast<size_t>(batch_samples_number*neurons_number*sizeof(type)));

        // Rest of quantities

        previous_hidden_state_activations.resize(neurons_number);
        previous_cell_state_activations.resize(neurons_number);

        current_inputs.resize(inputs_number);

        current_forget_combinations.resize(neurons_number);
        current_input_combinations.resize(neurons_number);
        current_state_combinations.resize(neurons_number);
        current_output_combinations.resize(neurons_number);

        current_forget_activations.resize(neurons_number);
        current_input_activations.resize(neurons_number);
        current_state_activations.resize(neurons_number);
        current_output_activations.resize(neurons_number);

        current_cell_state_activations.resize(neurons_number);

        current_forget_activations_derivatives.resize(neurons_number);
        current_input_activations_derivatives.resize(neurons_number);
        current_state_activations_derivatives.resize(neurons_number);
        current_output_activations_derivatives.resize(neurons_number);
        current_hidden_states_derivatives.resize(neurons_number);

        forget_activations.resize(batch_samples_number, neurons_number);
        input_activations.resize(batch_samples_number, neurons_number);
        state_activations.resize(batch_samples_number, neurons_number);
        output_activations.resize(batch_samples_number, neurons_number);
        cell_states_activations.resize(batch_samples_number, neurons_number);
        hidden_states_activations.resize(batch_samples_number, neurons_number);

        forget_activations_derivatives.resize(batch_samples_number, neurons_number);
        input_activations_derivatives.resize(batch_samples_number, neurons_number);
        state_activations_derivatives.resize(batch_samples_number, neurons_number);
        output_activations_derivatives.resize(batch_samples_number, neurons_number);
        cell_states_activations_derivatives.resize(batch_samples_number, neurons_number);
        hidden_states_activations_derivatives.resize(batch_samples_number, neurons_number);

        combinations.resize(batch_samples_number, neurons_number);
    }

    void print() const
    {
        cout << "Combinations: " << endl;
        cout << combinations << endl;

        cout << "Current inputs: " << endl;
        cout << current_inputs << endl;

        cout << "Current input combinations: " << endl;
        cout << current_input_combinations << endl;

        cout << "Current input activations: " << endl;
        cout << current_input_activations << endl;

        cout << "Current input activations derivatives: " << endl;
        cout << current_input_activations_derivatives << endl;
     }

    Tensor<type, 2> combinations;

    Tensor<type, 1> previous_hidden_state_activations;
    Tensor<type, 1> previous_cell_state_activations;

    Tensor<type, 1> current_inputs;

    Tensor<type, 1> current_forget_combinations;
    Tensor<type, 1> current_input_combinations;
    Tensor<type, 1> current_state_combinations;
    Tensor<type, 1> current_output_combinations;

    Tensor<type, 1> current_forget_activations;
    Tensor<type, 1> current_input_activations;
    Tensor<type, 1> current_state_activations;
    Tensor<type, 1> current_output_activations;

    Tensor<type, 1> current_forget_activations_derivatives;
    Tensor<type, 1> current_input_activations_derivatives;
    Tensor<type, 1> current_state_activations_derivatives;
    Tensor<type, 1> current_output_activations_derivatives;

    Tensor<type, 1> current_hidden_states_derivatives;

    Tensor<type, 1> current_cell_state_activations;

    Tensor<type, 2, RowMajor> forget_activations;
    Tensor<type, 2, RowMajor> input_activations;
    Tensor<type, 2, RowMajor> state_activations;
    Tensor<type, 2, RowMajor> output_activations;
    Tensor<type, 2, RowMajor> cell_states_activations;
    Tensor<type, 2, RowMajor> hidden_states_activations;

    Tensor<type, 2, RowMajor> forget_activations_derivatives;
    Tensor<type, 2, RowMajor> input_activations_derivatives;
    Tensor<type, 2, RowMajor> state_activations_derivatives;
    Tensor<type, 2, RowMajor> output_activations_derivatives;
    Tensor<type, 2, RowMajor> cell_states_activations_derivatives;
    Tensor<type, 2, RowMajor> hidden_states_activations_derivatives;
};


struct LongShortTermMemoryLayerBackPropagation : LayerBackPropagation
{
    explicit LongShortTermMemoryLayerBackPropagation() : LayerBackPropagation()
    {
    }


    explicit LongShortTermMemoryLayerBackPropagation(const Index& new_batch_samples_number, Layer* new_layer_pointer)
        : LayerBackPropagation()
    {
        set(new_batch_samples_number, new_layer_pointer);
    }


    void set(const Index& new_batch_samples_number, Layer* new_layer_pointer)
    {
        layer_pointer = new_layer_pointer;

        batch_samples_number = new_batch_samples_number;

        const Index neurons_number = layer_pointer->get_neurons_number();
        const Index inputs_number = layer_pointer->get_inputs_number();

        deltas_dimensions.resize(2);
        deltas_dimensions.setValues({batch_samples_number, neurons_number});

        //delete deltas_data;
        deltas_data = (type*)malloc(static_cast<size_t>(batch_samples_number*neurons_number*sizeof(type)));

        current_layer_deltas.resize(neurons_number);

        forget_weights_derivatives.resize(inputs_number*neurons_number);
        input_weights_derivatives.resize(inputs_number*neurons_number);
        state_weights_derivatives.resize(inputs_number*neurons_number);
        output_weights_derivatives.resize(inputs_number*neurons_number);

        forget_recurrent_weights_derivatives.resize(neurons_number*neurons_number);
        input_recurrent_weights_derivatives.resize(neurons_number*neurons_number);
        state_recurrent_weights_derivatives.resize(neurons_number*neurons_number);
        output_recurrent_weights_derivatives.resize(neurons_number*neurons_number);

        forget_biases_derivatives.resize(neurons_number);
        input_biases_derivatives.resize(neurons_number);
        state_biases_derivatives.resize(neurons_number);
        output_biases_derivatives.resize(neurons_number);

        input_combinations_biases_derivatives.resize(neurons_number, neurons_number);
        forget_combinations_biases_derivatives.resize(neurons_number, neurons_number);
        state_combinations_biases_derivatives.resize(neurons_number, neurons_number);
        output_combinations_biases_derivatives.resize(neurons_number, neurons_number);

        hidden_states_biases_derivatives.resize(neurons_number, neurons_number);
        cell_state_biases_derivatives.resize(neurons_number, neurons_number);

        input_combinations_weights_derivatives.resize(inputs_number*neurons_number, neurons_number);
        forget_combinations_weights_derivatives.resize(inputs_number*neurons_number, neurons_number);
        state_combinations_weights_derivatives.resize(inputs_number*neurons_number, neurons_number);
        output_combinations_weights_derivatives.resize(inputs_number*neurons_number, neurons_number);

        hidden_states_weights_derivatives.resize(inputs_number*neurons_number, neurons_number);
        cell_state_weights_derivatives.resize(inputs_number*neurons_number, neurons_number);

        input_combinations_recurrent_weights_derivatives.resize(neurons_number*neurons_number, neurons_number);
        forget_combinations_recurrent_weights_derivatives.resize(neurons_number*neurons_number, neurons_number);
        state_combinations_recurrent_weights_derivatives.resize(neurons_number*neurons_number, neurons_number);
        output_combinations_recurrent_weights_derivatives.resize(neurons_number*neurons_number, neurons_number);

        hidden_states_recurrent_weights_derivatives.resize(neurons_number*neurons_number, neurons_number);
        cell_state_recurrent_weights_derivatives.resize(neurons_number*neurons_number, neurons_number);
    }

    void print() const
    {
    }

    Tensor< TensorMap< Tensor<type, 1> >*, 1> get_layer_gradient()
    {
        Tensor< TensorMap< Tensor<type, 1> >*, 1> layer_gradient(30);

        layer_gradient(0) = new TensorMap<Tensor<type, 1>>(input_biases_derivatives.data(), input_biases_derivatives.size());
        layer_gradient(1) = new TensorMap<Tensor<type, 1>>(forget_biases_derivatives.data(), forget_biases_derivatives.size());
        layer_gradient(2) = new TensorMap<Tensor<type, 1>>(state_biases_derivatives.data(), state_biases_derivatives.size());
        layer_gradient(3) = new TensorMap<Tensor<type, 1>>(output_biases_derivatives.data(), output_biases_derivatives.size());

        layer_gradient(4) = new TensorMap<Tensor<type, 1>>(input_weights_derivatives.data(), input_weights_derivatives.size());
        layer_gradient(5) = new TensorMap<Tensor<type, 1>>(forget_weights_derivatives.data(), forget_weights_derivatives.size());
        layer_gradient(6) = new TensorMap<Tensor<type, 1>>(state_weights_derivatives.data(), state_weights_derivatives.size());
        layer_gradient(7) = new TensorMap<Tensor<type, 1>>(output_weights_derivatives.data(), output_weights_derivatives.size());

        layer_gradient(8) = new TensorMap<Tensor<type, 1>>(input_recurrent_weights_derivatives.data(), input_recurrent_weights_derivatives.size());
        layer_gradient(9) = new TensorMap<Tensor<type, 1>>(forget_recurrent_weights_derivatives.data(), forget_recurrent_weights_derivatives.size());
        layer_gradient(10) = new TensorMap<Tensor<type, 1>>(state_recurrent_weights_derivatives.data(), state_recurrent_weights_derivatives.size());
        layer_gradient(11) = new TensorMap<Tensor<type, 1>>(output_recurrent_weights_derivatives.data(), output_recurrent_weights_derivatives.size());

        layer_gradient(12) = new TensorMap<Tensor<type, 1>>(input_combinations_biases_derivatives.data(), input_combinations_biases_derivatives.size());
        layer_gradient(13) = new TensorMap<Tensor<type, 1>>(forget_combinations_biases_derivatives.data(), forget_combinations_biases_derivatives.size());
        layer_gradient(14) = new TensorMap<Tensor<type, 1>>(state_combinations_biases_derivatives.data(), state_combinations_biases_derivatives.size());
        layer_gradient(15) = new TensorMap<Tensor<type, 1>>(output_combinations_biases_derivatives.data(), output_combinations_biases_derivatives.size());

        layer_gradient(16) = new TensorMap<Tensor<type, 1>>(hidden_states_biases_derivatives.data(), hidden_states_biases_derivatives.size());
        layer_gradient(17) = new TensorMap<Tensor<type, 1>>(cell_state_biases_derivatives.data(), cell_state_biases_derivatives.size());

        layer_gradient(18) = new TensorMap<Tensor<type, 1>>(input_combinations_weights_derivatives.data(), input_combinations_weights_derivatives.size());
        layer_gradient(19) = new TensorMap<Tensor<type, 1>>(forget_combinations_weights_derivatives.data(), forget_combinations_weights_derivatives.size());
        layer_gradient(20) = new TensorMap<Tensor<type, 1>>(state_combinations_weights_derivatives.data(), state_combinations_weights_derivatives.size());
        layer_gradient(21) = new TensorMap<Tensor<type, 1>>(output_combinations_weights_derivatives.data(), output_combinations_weights_derivatives.size());

        layer_gradient(22) = new TensorMap<Tensor<type, 1>>(hidden_states_weights_derivatives.data(), hidden_states_weights_derivatives.size());
        layer_gradient(23) = new TensorMap<Tensor<type, 1>>(cell_state_weights_derivatives.data(), cell_state_weights_derivatives.size());

        layer_gradient(24) = new TensorMap<Tensor<type, 1>>(input_combinations_recurrent_weights_derivatives.data(), input_combinations_recurrent_weights_derivatives.size());
        layer_gradient(25) = new TensorMap<Tensor<type, 1>>(forget_combinations_recurrent_weights_derivatives.data(), forget_combinations_recurrent_weights_derivatives.size());
        layer_gradient(26) = new TensorMap<Tensor<type, 1>>(state_combinations_recurrent_weights_derivatives.data(), state_combinations_recurrent_weights_derivatives.size());
        layer_gradient(27) = new TensorMap<Tensor<type, 1>>(output_combinations_recurrent_weights_derivatives.data(), output_combinations_recurrent_weights_derivatives.size());

        layer_gradient(28) = new TensorMap<Tensor<type, 1>>(hidden_states_recurrent_weights_derivatives.data(), hidden_states_recurrent_weights_derivatives.size());
        layer_gradient(29) = new TensorMap<Tensor<type, 1>>(cell_state_recurrent_weights_derivatives.data(), cell_state_recurrent_weights_derivatives.size());

        return layer_gradient;
    }

    Tensor<type, 1> current_layer_deltas;

    Tensor<type, 1> forget_weights_derivatives;
    Tensor<type, 1> input_weights_derivatives;
    Tensor<type, 1> state_weights_derivatives;
    Tensor<type, 1> output_weights_derivatives;

    Tensor<type, 1> forget_recurrent_weights_derivatives;
    Tensor<type, 1> input_recurrent_weights_derivatives;
    Tensor<type, 1> state_recurrent_weights_derivatives;
    Tensor<type, 1> output_recurrent_weights_derivatives;

    Tensor<type, 1> forget_biases_derivatives;
    Tensor<type, 1> input_biases_derivatives;
    Tensor<type, 1> state_biases_derivatives;
    Tensor<type, 1> output_biases_derivatives;

    Tensor<type, 2> input_combinations_biases_derivatives;
    Tensor<type, 2> forget_combinations_biases_derivatives;
    Tensor<type, 2> state_combinations_biases_derivatives;
    Tensor<type, 2> output_combinations_biases_derivatives;

    Tensor<type, 2> hidden_states_biases_derivatives;
    Tensor<type, 2> cell_state_biases_derivatives;

    Tensor<type, 2> input_combinations_weights_derivatives;
    Tensor<type, 2> forget_combinations_weights_derivatives;
    Tensor<type, 2> state_combinations_weights_derivatives;
    Tensor<type, 2> output_combinations_weights_derivatives;

    Tensor<type, 2> hidden_states_weights_derivatives;
    Tensor<type, 2> cell_state_weights_derivatives;

    Tensor<type, 2> input_combinations_recurrent_weights_derivatives;
    Tensor<type, 2> forget_combinations_recurrent_weights_derivatives;
    Tensor<type, 2> state_combinations_recurrent_weights_derivatives;
    Tensor<type, 2> output_combinations_recurrent_weights_derivatives;

    Tensor<type, 2> hidden_states_recurrent_weights_derivatives;
    Tensor<type, 2> cell_state_recurrent_weights_derivatives;
};


}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software

// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
