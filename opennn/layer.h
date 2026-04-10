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

    const string& get_label() const { return label; }

    const string& get_name() const { return name; }

    virtual void set_input_shape(const Shape&);
    virtual void set_output_shape(const Shape&);

    void set_label(const string& new_label) { label = new_label; }

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

    Index get_inputs_number() const { return get_input_shape().size(); }

    Index get_outputs_number() const { return get_output_shape().size(); }

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

    bool get_is_trainable() const { return is_trainable; }

    type* link_parameters(type* pointer);

    vector<TensorView>& get_parameter_views() { return parameters; }

protected:

    Layer() = default;

    Shape input_shape;

    string label = "my_layer";

    string name = "layer";

    bool is_trainable = true;

    bool is_first_layer = false;

    vector<TensorView> parameters;

    Tensor2 empty_2;
    Tensor3 empty_3;
    Tensor4 empty_4;
    
    void add_gradients(const vector<TensorView>&) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
