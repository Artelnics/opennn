//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P E R C E P T R O N   L A Y E R   C L A S S   H E A D E R             
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef PERCEPTRONLAYER_H
#define PERCEPTRONLAYER_H

#include "layer.h"

namespace opennn
{

//#ifdef OPENNN_CUDA
//    struct PerceptronLayerForwardPropagationCuda;
//    struct PerceptronLayerBackPropagationCuda;
//#endif

class Perceptron : public Layer
{

public:

    enum class Activation {
        Logistic,
        HyperbolicTangent,
        Linear,
        RectifiedLinear,
        ExponentialLinear,
        ScaledExponentialLinear,
        SoftPlus,
        SoftSign,
        HardSigmoid
    };

    Perceptron(const dimensions& = {0},
                    const dimensions& = {0},
                    const Activation& = Perceptron::Activation::HyperbolicTangent,
                    const string& = "perceptron_layer");

    dimensions get_input_dimensions() const override;
    dimensions get_output_dimensions() const override;

    Tensor<type, 1> get_parameters() const override;

    Index get_parameters_number() const override;
    type get_dropout_rate() const;

    const Perceptron::Activation& get_activation_function() const;

    string get_activation_function_string() const;

    void set(const dimensions& = {0},
             const dimensions& = {0},
             const Perceptron::Activation & = Perceptron::Activation::HyperbolicTangent,
             const string& = "perceptron_layer");

    void set_input_dimensions(const dimensions&) override;
    void set_output_dimensions(const dimensions&) override;

    void set_parameters(const Tensor<type, 1>&, Index&) override;
    void set_parameters_constant(const type&) override;
    void set_parameters_random() override;

    void set_activation_function(const Activation&);
    void set_activation_function(const string&);
    void set_dropout_rate(const type&);

    void calculate_combinations(const Tensor<type, 2>&,
                                Tensor<type, 2>&) const;

    void normalization(Tensor<type, 1>& means, Tensor<type, 1>& standard_deviations, const Tensor<type, 2>& inputs, Tensor<type, 2>& outputs) const
    {

        const Eigen::array<Index, 2> rows({ outputs.dimension(0), 1 });

        const Eigen::array<int, 1> axis_x({ 0 });

        means.device(*thread_pool_device) = outputs.mean(axis_x);

        standard_deviations.device(*thread_pool_device)
            = (outputs - means.broadcast(rows)).square().mean(axis_x).sqrt();
        
       
//        outputs = inputs - means.broadcast(Eigen::array<Index, 2>({ outputs.dimension(0), 1 }));
            //shifts.broadcast(rows);
                //+ (outputs - means.broadcast(rows))*scales.broadcast(rows)/standard_deviations.broadcast(rows);
        
    }


    void calculate_activations(Tensor<type, 2>&,
                               Tensor<type, 2>&) const;

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) override;

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void back_propagate_lm(const vector<pair<type*, dimensions>>&,
                           const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           unique_ptr<LayerBackPropagationLM>&) const override;

    void insert_gradient(unique_ptr<LayerBackPropagation>&,
                         Index&,
                         Tensor<type, 1>&) const override;

    void insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>&,
                                           const Index&,
                                           Tensor<type, 2>&) const override;

    string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const override;

    string get_activation_function_string_expression() const;

    void print() const override;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

#ifdef OPENNN_CUDA

public:

    void forward_propagate_cuda(const Tensor<pair<type*, dimensions>, 1>&,
                                LayerForwardPropagationCuda*,
                                const bool&) final;

    void back_propagate_cuda(const Tensor<pair<type*, dimensions>, 1>&,
                             const Tensor<pair<type*, dimensions>, 1>&,
                             LayerForwardPropagationCuda*,
                             LayerBackPropagationCuda*) const final;

            void insert_gradient_cuda(LayerBackPropagationCuda*, const Index&, float*) const;

            void set_parameters_cuda(const float*, const Index&);

            void get_parameters_cuda(const Tensor<type, 1>&, const Index&);

            void allocate_parameters_device();
            void free_parameters_device();
            void copy_parameters_device();
            void copy_parameters_host();

            float* get_synaptic_weights_device() const;
            float* get_biases_device() const;

            Index get_neurons_number() const;
            Index get_inputs_number() const;
            Index get_synaptic_weights_number() const;
            Index get_biases_number() const;
            ActivationFunction get_activation_function() const;

protected:

    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;
    cudnnOpTensorDescriptor_t operator_sum_descriptor;

private:

    float* biases_device = nullptr;
    float* synaptic_weights_device = nullptr;

#endif

private:

    Tensor<type, 1> biases;

    Tensor<type, 2> weights;

    Tensor<type, 1> scales;
    Tensor<type, 1> shifts;

    Activation activation_function = Activation::HyperbolicTangent;

    type dropout_rate = type(0);

    const Eigen::array<Index, 1> sum_dimensions_1 = {0};
};


struct PerceptronForwardPropagation : LayerForwardPropagation
{
    PerceptronForwardPropagation(const Index& = 0, Layer* = nullptr);

    pair<type*, dimensions> get_outputs_pair() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 1> means;
    Tensor<type, 1> standard_deviations;

    Tensor<type, 2> outputs;

    Tensor<type, 2> activation_derivatives;
};


struct PerceptronBackPropagation : LayerBackPropagation
{
    PerceptronBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 2> combination_derivatives;
    Tensor<type, 2> input_derivatives;

    Tensor<type, 1> bias_derivatives;
    Tensor<type, 2> weight_derivatives;
};


struct PerceptronLayerBackPropagationLM : LayerBackPropagationLM
{
    PerceptronLayerBackPropagationLM(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 2> combination_derivatives;
    Tensor<type, 2> input_derivatives;

    Tensor<type, 2> squared_errors_Jacobian;
};


#ifdef OPENNN_CUDA

struct PerceptronLayerForwardPropagationCuda : public LayerForwardPropagationCuda
{
    explicit PerceptronLayerForwardPropagationCuda(const Index& new_batch_samples_number, Layer* new_layer);

    void set(const Index& new_batch_samples_number, Layer* new_layer);

    void print() const override;

    void free() override;

    std::pair<type*, dimensions> get_outputs_pair() const override;

    type* combinations_cuda = nullptr;

    cudnnActivationDescriptor_t activation_descriptor = nullptr;

    cudnnTensorDescriptor_t outputs_tensor_descriptor = nullptr;
    cudnnTensorDescriptor_t outputs_batch_tensor_descriptor = nullptr;
    cudnnTensorDescriptor_t biases_batch_tensor_descriptor = nullptr;
};


struct PerceptronLayerBackPropagationCuda : public LayerBackPropagationCuda
{
    explicit PerceptronLayerBackPropagationCuda(const Index& new_batch_samples_number, Layer* new_layer);

    void set(const Index& new_batch_samples_number, Layer* new_layer);

    void print() const override;

    void free() override;

    float* biases_derivatives_cuda = nullptr;
    float* synaptic_weights_derivatives_cuda = nullptr;
    float* error_combinations_derivatives_cuda = nullptr;
    float* ones = nullptr;
    float one = 1.0f;

    cudnnTensorDescriptor_t error_combinations_derivatives_tensor_descriptor = nullptr;
    cudnnTensorDescriptor_t deltas_tensor_descriptor = nullptr;
};

#endif


}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
