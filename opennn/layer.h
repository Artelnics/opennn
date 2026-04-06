//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_utilities.h"
#include "random_utilities.h"

namespace opennn
{

struct ForwardPropagation;
struct BackPropagation;

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

    virtual ~Layer() = default;

    const string& get_label() const;

    const string& get_name() const;

    virtual void set_input_shape(const Shape&);
    virtual void set_output_shape(const Shape&);

    void set_label(const string&);

    virtual void set_parameters_random();

    virtual void set_parameters_glorot();

    Index get_parameters_number() const;

    virtual vector<Shape> get_parameter_shapes() const
    {
        return {};
    }

    virtual vector<Shape> get_forward_shapes(Index) const
    {
        return {};
    }

    virtual vector<Shape> get_backward_shapes(Index) const
    {
        return {};
    }

    virtual Shape get_input_shape() const
    {
        return input_shape;
    }

    virtual Shape get_output_shape() const = 0;

    Index get_inputs_number() const;

    Index get_outputs_number() const;

    // Forward propagation

    virtual void forward_propagate(ForwardPropagation&, size_t, bool) = 0;

    // Back propagation

    virtual void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const {}

    virtual void from_XML(const tinyxml2::XMLDocument&) {}

    virtual void to_XML(tinyxml2::XMLPrinter&) const {}

    virtual string get_expression(const vector<string>& = vector<string>(), const vector<string>& = vector<string>()) const;

    virtual void print() const {}

    vector<string> get_default_feature_names() const;

    vector<string> get_default_output_names() const;

    bool get_is_trainable() const;

    type* link_parameters(type* pointer);

protected:

    Layer() = default;

    Shape input_shape;

    string label = "my_layer";

    string name = "layer";

    bool is_trainable = true;

    // @todo set this

    bool is_first_layer = false;

    vector<TensorView> parameters;

    Tensor2 empty_2;
    Tensor3 empty_3;
    Tensor4 empty_4;

/*
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
*/
    void add_gradients(const vector<TensorView>&) const;

    template <int Rank>
    void normalize_batch(
        TensorMapR<Rank> outputs,
        TensorMapR<Rank> normalized_outputs,
        VectorMap means,
        VectorMap variances,
        VectorR running_means,
        VectorR running_variances,
        const VectorMap gammas,
        const VectorMap betas,
        bool is_training,
        const type momentum = type(0.9)) const
    {
        const Index neurons = running_means.size();
        const Index total_rows = outputs.size() / neurons;

        MatrixMap outputs_mat(outputs.data(), total_rows, neurons);
        MatrixMap norm_mat(normalized_outputs.data(), total_rows, neurons);

        if(is_training)
        {
            means = outputs_mat.colwise().mean();

            norm_mat = outputs_mat.rowwise() - means.transpose();

            variances = (norm_mat.array().square().colwise().mean() + EPSILON).sqrt();

            norm_mat.array().rowwise() /= variances.transpose().array();

            running_means = running_means * momentum + means * (type(1) - momentum);
            running_variances = running_variances * momentum + variances * (type(1) - momentum);
        }
        else
            norm_mat.array() = (outputs_mat.rowwise() - running_means.transpose()).array().rowwise() /
                               (running_variances.transpose().array() + EPSILON);

        outputs_mat.array() = (norm_mat.array().rowwise() * gammas.transpose().array()).rowwise() +
                              betas.transpose().array();
    }
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
