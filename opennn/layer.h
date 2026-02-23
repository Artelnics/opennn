//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensors.h"
#include "random_utilities.h"

namespace opennn
{

struct LayerForwardPropagation;
struct LayerBackPropagation;
struct LayerBackPropagationLM;

struct LayerForwardPropagationCuda;
struct LayerBackPropagationCuda;

#ifdef _MSC_VER
#define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE __attribute__((always_inline)) inline
#else
#define FORCE_INLINE inline
#endif

class Layer
{

public:

    Layer();
    virtual ~Layer();

    const string& get_label() const;

    bool get_display() const;

    const string& get_name() const;

    virtual void set_input_shape(const Shape&);
    virtual void set_output_shape(const Shape&);

    void set_label(const string&);

    void set_display(bool);

    virtual void set_parameters_random();

    virtual void set_parameters_glorot();

    Index get_parameters_number();

    virtual vector<TensorView*> get_parameter_views()
    {
        return vector<TensorView*>();
    }

    //virtual pair

    virtual Shape get_input_shape() const = 0;
    virtual Shape get_output_shape() const = 0;

    Index get_inputs_number() const;

    Index get_outputs_number() const;

    // Forward propagation

    virtual void forward_propagate(const vector<TensorView>&,
                                   unique_ptr<LayerForwardPropagation>&,
                                   bool) = 0;

    // Back propagation

    virtual void back_propagate(const vector<TensorView>&,
                                const vector<TensorView>&,
                                unique_ptr<LayerForwardPropagation>&,
                                unique_ptr<LayerBackPropagation>&) const {}

    virtual void back_propagate_lm(const vector<TensorView>&,
                                   const vector<TensorView>&,
                                   unique_ptr<LayerForwardPropagation>&,
                                   unique_ptr<LayerBackPropagationLM>&) const {}

    virtual void insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>&,
                                                   Index,
                                                   MatrixR&) const {}

    virtual void from_XML(const tinyxml2::XMLDocument&) {}

    virtual void to_XML(tinyxml2::XMLPrinter&) const {}

    virtual string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const;

    virtual void print() const {}

    vector<string> get_default_feature_names() const;

    vector<string> get_default_output_names() const;

    bool get_is_trainable() const;

protected:

    string label = "my_layer";

    string name = "layer";

    bool is_trainable = true;

    Tensor2 empty_2;
    Tensor3 empty_3;
    Tensor4 empty_4;

    bool display = true;

    template <int Rank>
    void calculate_activations(const string& activation_function,
                               TensorMapR<Rank> activations,
                               TensorMapR<Rank> activation_derivatives) const
    {
        if (activation_function == "Linear")
            linear(activations, activation_derivatives);
        else if (activation_function == "Sigmoid")
            logistic(activations, activation_derivatives);
        else if (activation_function == "Softmax")
        {
            if constexpr (Rank == 2)
            {
                MatrixMap activations_matrix(activations.data(), activations.dimension(0), activations.dimension(1));
                softmax(activations_matrix);
            }
            else
                softmax(activations);
        }
        else if (activation_function == "Competitive")
            throw runtime_error("Competitive 3d not implemented");
        else if (activation_function == "HyperbolicTangent")
            hyperbolic_tangent(activations, activation_derivatives);
        else if (activation_function == "RectifiedLinear")
            rectified_linear(activations, activation_derivatives);
        else if (activation_function == "ScaledExponentialLinear")
            exponential_linear(activations, activation_derivatives);
        else
            throw runtime_error("Unknown activation: " + activation_function);
    }


    template <int Rank>
    FORCE_INLINE void binary(TensorR<Rank>& y, TensorR<Rank>& dy_dx, type threshold) const
    {
        y.device(get_device()) = (y < threshold).select(type(0), type(1));

        if (dy_dx.size() == 0) return;

        dy_dx.setConstant(type(0));
    }


    template <int Rank>
    FORCE_INLINE void linear(TensorMapR<Rank>, TensorMapR<Rank> dy_dx) const
    {
        if (dy_dx.size() == 0) return;

        dy_dx.setConstant(type(1));
    }


    template <int Rank>
    FORCE_INLINE void exponential_linear(TensorMapR<Rank> y, TensorMapR<Rank> dy_dx) const
    {
        const type alpha = type(1);

        y.device(get_device()) = (y > type(0)).select(y, alpha * (y.exp() - type(1)));

        if (dy_dx.size() == 0) return;

        dy_dx.device(get_device()) = (y > type(0)).select(dy_dx.constant(type(1)), y + alpha);
    }


    template <int Rank>
    FORCE_INLINE void hyperbolic_tangent(TensorMapR<Rank> y, TensorMapR<Rank> dy_dx) const
    {
        y.device(get_device()) = y.tanh();

        if (dy_dx.size() == 0) return;

        dy_dx.device(get_device()) = (type(1) - y.square()).eval();
    }


    template <int Rank>
    FORCE_INLINE void logistic(TensorMapR<Rank> y, TensorMapR<Rank> dy_dx) const
    {
        y.device(get_device()) = (type(1) + (-y).exp()).inverse();

        if (dy_dx.size() == 0) return;

        dy_dx.device(get_device()) = (y * (type(1) - y)).eval();
    }


    template <int Rank>
    FORCE_INLINE void rectified_linear(TensorMapR<Rank> y, TensorMapR<Rank> dy_dx) const
    {
        y.device(get_device()) = y.cwiseMax(type(0));

        if (dy_dx.size() == 0) return;

        dy_dx.device(get_device()) = (y > type(0)).select(dy_dx.constant(type(1)), dy_dx.constant(type(0)));
    }


    template <int Rank>
    FORCE_INLINE void leaky_rectified_linear(TensorMapR<Rank> y, TensorMapR<Rank> dy_dx, type slope) const
    {
        y.device(get_device()) = (y > type(0)).select(y, slope * y);

        if (dy_dx.size() == 0) return;

        dy_dx.device(get_device()) = (y > type(0)).select(dy_dx.constant(type(1)), dy_dx.constant(type(slope)));
    }


    template <int Rank>
    FORCE_INLINE void scaled_exponential_linear(TensorMapR<Rank> y, TensorMapR<Rank> dy_dx) const
    {
        const type lambda = type(1.0507);

        const type alpha = type(1.6733);

        y.device(get_device()) = (y > type(0)).select(lambda * y, lambda * alpha * (y.exp() - type(1)));

        if (dy_dx.size() == 0) return;

        dy_dx.device(get_device()) = (y > type(0)).select(dy_dx.constant(lambda), y + alpha * lambda);
    }

    void softmax(MatrixMap) const;
    void softmax(TensorMap3) const;
    void softmax(TensorMap4) const;

    void softmax_derivatives_times_tensor(const TensorMap3, TensorMap3, VectorMap) const;

    void add_gradients(const vector<TensorView>&) const;

    template <int Rank>
    void normalize_batch(
        TensorMapR<Rank> outputs,
        TensorMapR<Rank> normalized_outputs,
        VectorMap batch_means,
        VectorMap batch_variances,
        VectorR running_means,
        VectorR running_variances,
        const VectorMap gammas,
        const VectorMap betas,
        bool is_training,
        const type momentum = type(0.9),
        const type epsilon = type(1e-5)) const
    {
        const Index neurons = running_means.size();
        const Index total_rows = outputs.size() / neurons;

        MatrixMap outputs_mat(outputs.data(), total_rows, neurons);
        MatrixMap norm_mat(normalized_outputs.data(), total_rows, neurons);

        if(is_training)
        {
            batch_means = outputs_mat.colwise().mean();

            norm_mat = outputs_mat.rowwise() - batch_means.transpose();

            batch_variances = (norm_mat.array().square().colwise().mean() + epsilon).sqrt();

            norm_mat.array().rowwise() /= batch_variances.transpose().array();

            running_means = running_means * momentum + batch_means * (type(1) - momentum);
            running_variances = running_variances * momentum + batch_variances * (type(1) - momentum);
        }
        else
            norm_mat.array() = (outputs_mat.rowwise() - running_means.transpose()).array().rowwise() /
                               (running_variances.transpose().array() + epsilon);

        outputs_mat.array() = (norm_mat.array().rowwise() * gammas.transpose().array()).rowwise() +
                              betas.transpose().array();
    }


    template <int Rank>
    void dropout(TensorMapR<Rank> tensor, type dropout_rate) const
    {
        const type scaling_factor = type(1) / (type(1) - dropout_rate);

        #pragma omp parallel
        {
            for(Index i = 0; i < tensor.size(); i++)
                tensor(i) = (random_uniform(0, 1) < dropout_rate)
                    ? 0
                    : tensor(i) * scaling_factor;
        }
    }


    template <int Rank>
    void calculate_combinations(const TensorMapR<Rank> inputs,
                                const MatrixMap weights,
                                const VectorMap biases,
                                TensorMapR<Rank> outputs) const
    {
        const Index inputs_size = weights.rows();
        const Index outputs_size = weights.cols();
        const Index total_rows = inputs.size() / inputs_size;

        const Map<const Matrix<type, Dynamic, Dynamic, ColMajor, AlignedMax>>
            inputs_matrix(inputs.data(), total_rows, inputs_size);

        Map<Matrix<type, Dynamic, Dynamic, ColMajor, AlignedMax>>
            outputs_matrix(outputs.data(), total_rows, outputs_size);

        outputs_matrix.noalias() = (inputs_matrix * weights).rowwise() + biases.transpose();
    }

#ifdef OPENNN_CUDA

public:

    virtual void forward_propagate(const vector<TensorViewCuda>&,
                                   unique_ptr<LayerForwardPropagationCuda>&,
                                   bool)
    {
        throw runtime_error("CUDA forward propagation not implemented for layer type: " + get_name());
    }

    virtual void back_propagate(const vector<TensorViewCuda>&,
                                const vector<TensorViewCuda>&,
                                unique_ptr<LayerForwardPropagationCuda>&,
                                unique_ptr<LayerBackPropagationCuda>&) const 
    {
        throw runtime_error("CUDA back propagation not implemented for layer type: " + get_name());
    }

    virtual vector<TensorViewCuda*> get_parameter_views_device()
    {
        return vector<TensorViewCuda*>();
    }

    virtual void free() {}

    virtual void print_parameters_cuda() {}

protected:

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float beta_add = 1.0f;

#endif

};


struct LayerForwardPropagation
{
    LayerForwardPropagation() {}
    virtual ~LayerForwardPropagation() = default;

    void set(const Index = 0, Layer* = nullptr);
    virtual void initialize() = 0;

    virtual vector<TensorView*> get_workspace_views();

    TensorView get_outputs() const;

    virtual void print() const {}

    Index batch_size = 0;

    Layer* layer = nullptr;

    TensorView outputs;
};


struct LayerBackPropagation
{
    LayerBackPropagation() {}
    virtual ~LayerBackPropagation() = default;

    void set(const Index = 0, Layer* = nullptr);
    virtual void initialize() = 0;

    virtual vector<TensorView*> get_gradient_views();

    vector<TensorView> get_input_gradients() const;

    virtual void print() const {}

    Index batch_size = 0;

    Layer* layer = nullptr;

    bool is_first_layer = false;

    vector<TensorView> input_gradients;
    vector<VectorR> input_gradients_memory;
};


struct LayerBackPropagationLM
{
    LayerBackPropagationLM() {}
    virtual ~LayerBackPropagationLM() = default;

    void set(const Index = 0, Layer* = nullptr);
    virtual void initialize() = 0;

    virtual vector<TensorView*> get_gradient_views();

    vector<TensorView> get_input_gradients() const;

    virtual void print() const {}

    Index batch_size = 0;

    Layer* layer = nullptr;

    bool is_first_layer = false;

    vector<TensorView> input_gradients;
    vector<VectorR> input_gradients_memory;
};


#ifdef OPENNN_CUDA

struct LayerForwardPropagationCuda
{
    LayerForwardPropagationCuda() {}
    virtual ~LayerForwardPropagationCuda() {}

    void set(const Index = 0, Layer* = nullptr);

    virtual void initialize() = 0;

    virtual vector<TensorViewCuda*> get_workspace_views();

    TensorViewCuda get_outputs() const;

    virtual void print() const {}

    virtual void free() {}

    Index batch_size = 0;

    Layer* layer = nullptr;

    TensorViewCuda outputs;
};


struct LayerBackPropagationCuda
{
    LayerBackPropagationCuda() {}

    virtual ~LayerBackPropagationCuda() {}

    void set(const Index = 0, Layer* = nullptr);
    virtual void initialize() = 0;

    virtual vector<TensorViewCuda*> get_workspace_views()
    {
		return vector<TensorViewCuda*>();
    };

    vector<TensorViewCuda> get_input_gradient_views() const;

    virtual void print() const {}

	virtual void free() {}

    Index batch_size = 0;

    Layer* layer = nullptr;

    bool is_first_layer = false;

    vector<TensorCuda> input_gradients;
};

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
