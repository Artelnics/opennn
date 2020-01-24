//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "layer.h"

namespace OpenNN {

/// Default constructor.
/// It creates a layer object with zero parameters.
/// It also initializes the rest of class members to their default values.

Layer::Type Layer::get_type () const
{
    return layer_type;
}


/// Takes the type of layer used by the model.

string Layer::get_type_string() const
{
    switch(layer_type)
    {
        case PrincipalComponents:
        {
            return "PrincipalComponents";
        }
        case Convolutional:
        {
            return "Convolutional";
        }
        case Perceptron:
        {
            return "Perceptron";
        }
        case Bounding:
        {
            return "Bounding";
        }
        case Pooling:
        {
            return "Pooling";
        }
        case Probabilistic:
        {
            return "Probabilistic";
        }
        case LongShortTermMemory:
        {
            return "LongShortTermMemory";
        }
        case Recurrent:
        {
            return "Recurrent";
        }
        case Scaling:
        {
            return "Scaling";
        }
        case Unscaling:
        {
            return "Unscaling";
        }
    }

    return string();
}


void Layer::initialize_parameters(const double&)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "initialize_parameters(const double&) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


void Layer::set_parameters_random()
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_parameters_random() method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


void Layer::set_parameters(const Tensor<type, 1>&)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_parameters(const Tensor<type, 1>&) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


int Layer::get_parameters_number() const
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "get_parameters_number() method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


Tensor<type, 1> Layer::get_parameters() const
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "get_parameters() method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


Tensor<type, 2> Layer::calculate_outputs(const Tensor<type, 2> &)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "calculate_outputs(const Tensor<type, 2> &) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


Tensor<type, 2> Layer::calculate_outputs(const Tensor<type, 2> &, const Tensor<type, 1> &)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "calculate_outputs(const Tensor<type, 2> &, const Tensor<type, 1> &) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


Tensor<type, 1> Layer::calculate_error_gradient(const Tensor<type, 2>&,
                                               const Layer::ForwardPropagation&,
                                               const Tensor<type, 2>&)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "calculate_error_gradient(const Tensor<type, 2>&, const Layer::ForwardPropagation&, const Tensor<type, 2>&) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


Layer::ForwardPropagation Layer::calculate_forward_propagation(const Tensor<type, 2>&)
 {
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "calculate_forward_propagation(const Tensor<type, 2>&) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
 }


Tensor<type, 2> Layer::calculate_output_delta(const Tensor<type, 2> &, const Tensor<type, 2> &) const
 {
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "calculate_output_delta(const Tensor<type, 2> &, const Tensor<type, 2> &) const method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


Tensor<type, 2> Layer::calculate_hidden_delta(Layer *,
                                             const Tensor<type, 2> &,
                                             const Tensor<type, 2> &,
                                             const Tensor<type, 2> &) const
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "calculate_hidden_delta(Layer *, const Tensor<type, 2> &, const Tensor<type, 2> &, const Tensor<type, 2> &) const method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


Tensor<int, 1> Layer::get_input_variables_dimensions() const
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "get_input_variables_dimensions() const method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


int Layer::get_inputs_number() const
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "get_inputs_number() const method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


int Layer::get_neurons_number() const
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "get_neurons_number() const method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


void Layer::set_inputs_number(const int &)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_inputs_number(const int &) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


void Layer::set_neurons_number(const int &)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_neurons_number(const int &) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


string Layer::object_to_string() const
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "to_string() const method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}
}
