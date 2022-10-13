//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "layer.h"

namespace opennn
{

Layer::~Layer()
{
    delete thread_pool;
    delete thread_pool_device;
}


/// Default constructor.
/// It creates a layer object with zero parameters.
/// It also initializes the rest of the class members to their default values.

Layer::Type Layer::get_type () const
{
    return layer_type;
}


/// Takes the type of layer used by the model.

string Layer::get_type_string() const
{
    switch(layer_type)
    {
    case Type::Perceptron:
        return "Perceptron";

    case Type::Bounding:
        return "Bounding";

    case Type::Pooling:
        return "Pooling";

    case Type::Probabilistic:
        return "Probabilistic";

    case Type::Convolutional:
        return "Convolutional";

    case Type::LongShortTermMemory:
        return "LongShortTermMemory";

    case Type::Recurrent:
        return "Recurrent";

    case Type::Scaling:
        return "Scaling";

    case Type::Unscaling:
        return "Unscaling";

    case Type::Flatten:
        return "Flatten";

    case Type::Resnet50:
        return "Resnet50";

    default:
        return "Unkown type";
    }
}


void Layer::set_threads_number(const int& new_threads_number)
{
    if(thread_pool != nullptr) delete this->thread_pool;
    if(thread_pool_device != nullptr) delete this->thread_pool_device;

    thread_pool = new ThreadPool(new_threads_number);
    thread_pool_device = new ThreadPoolDevice(thread_pool, new_threads_number);
}


void Layer::set_parameters_constant(const type&)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_parameters_constant(const type&) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw invalid_argument(buffer.str());
}


void Layer::set_parameters_random()
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_parameters_random() method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw invalid_argument(buffer.str());
}


void Layer::set_parameters(const Tensor<type, 1>&, const Index&)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_parameters(const Tensor<type, 1>&) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw invalid_argument(buffer.str());
}


Tensor< TensorMap< Tensor<type, 1>>*, 1> Layer::get_layer_parameters()
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "virtual Tensor< TensorMap< Tensor<type, 1> >, 1> get_layer_parameters() method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw invalid_argument(buffer.str());
}


Index Layer::get_parameters_number() const
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "get_parameters_number() method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw invalid_argument(buffer.str());
}


Tensor<type, 1> Layer::get_parameters() const
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "get_parameters() method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw invalid_argument(buffer.str());
}


void Layer::calculate_outputs(type*, const Tensor<Index, 1>&,  type*, const Tensor<Index, 1>&)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "calculate_outputs(const Tensor<type, 2>&) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw invalid_argument(buffer.str());
};


void Layer::forward_propagate(type*, const Tensor<Index, 1>&, LayerForwardPropagation*)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "forward_propagate(type*, const Tensor<Index, 1>&, LayerForwardPropagation*) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw invalid_argument(buffer.str());
}


void Layer::forward_propagate(type*, const Tensor<Index, 1>&, Tensor<type, 1>&, LayerForwardPropagation*)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "forward_propagate(type*, const Tensor<Index, 1>&, LayerForwardPropagation*) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw invalid_argument(buffer.str());
}


/// Returns the number of inputs

Index Layer::get_inputs_number() const
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "get_inputs_number() const method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw invalid_argument(buffer.str());
}


Index Layer::get_neurons_number() const
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "get_neurons_number() const method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw invalid_argument(buffer.str());
}


void Layer::set_inputs_number(const Index& )
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_inputs_number(const Index& ) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw invalid_argument(buffer.str());
}


void Layer::set_neurons_number(const Index& )
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_neurons_number(const Index& ) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw invalid_argument(buffer.str());
}


void Layer::linear(type* x_data, const Tensor<Index, 1>& x_dimensions, type* y_data, const Tensor<Index, 1>& y_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (x_dimensions== y_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::linear(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "X and Y vector must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    // Apply function

    const Tensor<Index, 0>  size = x_dimensions.prod();

    memcpy(y_data, x_data, static_cast<size_t>(size(0))*sizeof(type));
}


void Layer::linear_derivatives(type* combinations_data,
                               const Tensor<Index, 1>& combinations_dimensions,
                               type* activations_data,
                               const Tensor<Index, 1>& activations_dimensions,
                               type* activations_derivatives_data,
                               const Tensor<Index, 1>& activations_derivatives_dimensions) const
{
    // Check equal sizes and ranks

    if(combinations_dimensions.size() != activations_dimensions.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"

               << "void Layer::linear_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,\n"
               << "                               type* activations_data, Tensor<Index, 1>& activations_dimensions,\n "
               << "                               type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions)\n"
               << "Combinations and activations vectors must have the same rank.\n";

        throw invalid_argument(buffer.str());
    }

    Tensor<bool, 0> same_dimensions = (combinations_dimensions== activations_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"

               << "void Layer::linear_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,\n"
               << "                               type* activations_data, Tensor<Index, 1>& activations_dimensions,\n"
               << "                               type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions)\n"
               << "Combinations and activations vectors must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    // Apply function

    const Tensor<Index, 0>  size = combinations_dimensions.prod();

    const Tensor<Index, 0>  derivatives_size = activations_derivatives_dimensions.prod();

    memcpy(activations_data, combinations_data, static_cast<size_t>(size(0))*sizeof(type));

    fill(activations_derivatives_data, activations_derivatives_data + derivatives_size(0), 1);
}


void Layer::logistic(type* x_data, const Tensor<Index, 1>& x_dimensions, type* y_data, const Tensor<Index, 1>& y_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (x_dimensions== y_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::linear(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "X and Y vector must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    // Apply function

    const Index rank = x_dimensions.size();  

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> x(x_data, x_dimensions(0));
        TensorMap<Tensor<type, 1>> y(y_data, y_dimensions(0));
        y.device(*thread_pool_device) = (type(1) + x.exp().inverse()).inverse();
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> x(x_data, x_dimensions(0), x_dimensions(1));
        TensorMap<Tensor<type, 2>> y(y_data, y_dimensions(0), y_dimensions(1));
        y.device(*thread_pool_device) = (type(1) + x.exp().inverse()).inverse();
    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> x(x_data, x_dimensions(0), x_dimensions(1), x_dimensions(2), x_dimensions(3));
        TensorMap<Tensor<type, 4>> y(y_data, y_dimensions(0), y_dimensions(1), y_dimensions(2), y_dimensions(3));
        y.device(*thread_pool_device) = (type(1) + x.exp().inverse()).inverse();
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::logistic(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "Logisitic function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::logistic_derivatives(type* combinations_data, const Tensor<Index, 1>& combinations_dimensions,
                                 type* activations_data, const Tensor<Index, 1>& activations_dimensions,
                                 type* activations_derivatives_data, const Tensor<Index, 1>& activations_derivatives_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (combinations_dimensions== activations_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"

               << "void Layer::linear_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                               type* activations_data, Tensor<Index, 1>& activations_dimensions,  "
               << "                               type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "Combinations and activations must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    const Index rank = combinations_dimensions.size();
    const Index derivatives_rank = activations_derivatives_dimensions.size();

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> combinations(combinations_data, combinations_dimensions(0));
        TensorMap<Tensor<type, 1>> activations(activations_data, activations_dimensions(0));
        TensorMap<Tensor<type, 1>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0));

        activations.device(*thread_pool_device) = (type(1) + combinations.exp().inverse()).inverse();

        activations_derivatives.device(*thread_pool_device) = activations*(type(1) - activations);

    }
    else if(rank == 2 && derivatives_rank == 2)
    {
        const TensorMap<Tensor<type, 2>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1));
        TensorMap<Tensor<type, 2>> activations(activations_data, activations_dimensions(0), activations_dimensions(1));
        TensorMap<Tensor<type, 2>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1));

        activations.device(*thread_pool_device) = (type(1) + combinations.exp().inverse()).inverse();

        activations_derivatives.device(*thread_pool_device) = activations*(type(1) - activations);

    }
    else if(rank == 2 && derivatives_rank == 3)
    {
        const TensorMap<Tensor<type, 2>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1));
        TensorMap<Tensor<type, 2>> activations(activations_data, activations_dimensions(0), activations_dimensions(1));
        TensorMap<Tensor<type, 3>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1), activations_derivatives_dimensions(2));

        // Activations

        activations.device(*thread_pool_device) = (type(1) + combinations.exp().inverse()).inverse();

        // Activations Derivatives

        Tensor<type, 2> derivatives_2d(activations.dimension(0), activations.dimension(1));

        derivatives_2d.device(*thread_pool_device) = activations*(type(1) - activations);

        copy(derivatives_2d.data(), derivatives_2d.data() + derivatives_2d.size(), activations_derivatives.data());
    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1), combinations_dimensions(2), combinations_dimensions(3));
        TensorMap<Tensor<type, 4>> activations(activations_data, activations_dimensions(0), activations_dimensions(1), activations_dimensions(2), activations_dimensions(3));
        TensorMap<Tensor<type, 4>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1), activations_derivatives_dimensions(2), activations_derivatives_dimensions(3));

        activations.device(*thread_pool_device) = (type(1) + combinations.exp().inverse()).inverse();

        activations_derivatives.device(*thread_pool_device) = activations*(type(1) - activations);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::logistic(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "Logisitic function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::hard_sigmoid(type* x_data, const Tensor<Index, 1>& x_dimensions, type* y_data, const Tensor<Index, 1>& y_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (x_dimensions== y_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::hard_sigmoid(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "X and Y vector must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    // Apply function

    const Index rank = x_dimensions.size();

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> x(x_data, x_dimensions(0));
        TensorMap<Tensor<type, 1>> y(y_data, y_dimensions(0));

        const Tensor<bool, 1> if_sentence = x < x.constant(type(-2.5));
        const Tensor<bool, 1> elif_sentence = x > x.constant(type(2.5));

        Tensor<type, 1> f1(x.dimension(0));
        Tensor<type, 1> f2(x.dimension(0));
        Tensor<type, 1> f3(x.dimension(0));

        f1.setZero();
        f2.setConstant(type(1));
        f3 = static_cast<type>(0.2) * x + static_cast<type>(0.5);

        y.device(*thread_pool_device) = if_sentence.select(f1,elif_sentence.select(f2,f3));
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> x(x_data, x_dimensions(0), x_dimensions(1));
        TensorMap<Tensor<type, 2>> y(y_data, y_dimensions(0), y_dimensions(1));

        const Index n = x.size();

    #pragma omp parallel for

        for(Index i = 0; i < n; i++)
        {
            if(x(i) < static_cast<type>(-2.5))
            {
                y(i) = type(0);
            }
            else if(x(i) > static_cast<type>(2.5))
            {
                y(i) = type(1);
            }
            else
            {
                y(i) = static_cast<type>(0.2) * x(i) + static_cast<type>(0.5);
            }
        }

    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> x(x_data, x_dimensions(0), x_dimensions(1), x_dimensions(2), x_dimensions(3));
        TensorMap<Tensor<type, 4>> y(y_data, y_dimensions(0), y_dimensions(1), y_dimensions(2), y_dimensions(3));

        const Tensor<bool, 4> if_lower = x < x.constant(type(-2.5));
        const Tensor<bool, 4> if_greater = x > x.constant(type(2.5));
        const Tensor<bool, 4> if_middle = x < x.constant(type(-2.5)) && x > x.constant(type(2.5));

        Tensor<type, 4> f_lower(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));
        Tensor<type, 4> f_greater(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));
        Tensor<type, 4> f_middle(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));
        Tensor<type, 4> f_equal(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

        f_lower = x.constant(type(0));
        f_greater = x.constant(type(1));
        f_middle = static_cast<type>(0.2) * x + static_cast<type>(0.5);
        f_equal = x;

        y.device(*thread_pool_device) = if_lower.select(f_lower, f_equal);
        y.device(*thread_pool_device) = if_greater.select(f_greater, f_equal);
        y.device(*thread_pool_device) = if_middle.select(f_middle, f_equal);    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::hard_sigmoid(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "Hard sigmoid function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::hard_sigmoid_derivatives(type* combinations_data,
                                     const Tensor<Index, 1>& combinations_dimensions,
                                     type* activations_data,
                                     const Tensor<Index, 1>& activations_dimensions,
                                     type* activations_derivatives_data,
                                     const Tensor<Index, 1>& activations_derivatives_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (combinations_dimensions== activations_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::hard_sigmoid_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                           type* activations_data, Tensor<Index, 1>& activations_dimensions,"
               << "                                           type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "X and Y vector must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    const Index rank = combinations_dimensions.size();

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> combinations(combinations_data, combinations_dimensions(0));
        TensorMap<Tensor<type, 1>> activations(activations_data, activations_dimensions(0));
        TensorMap<Tensor<type, 1>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0));

        const Tensor<bool, 1> if_sentence = combinations < combinations.constant(type(-2.5));

        const Tensor<bool, 1> elif_sentence = combinations > combinations.constant(type(2.5));

        const Tensor<bool, 1> if_sentence_2 = combinations < combinations.constant(type(-2.5)) || combinations > combinations.constant(type(2.5));

        // Sentences

        Tensor<type, 1> f1(combinations.dimension(0));
        f1.setZero();

        Tensor<type, 1> f2(combinations.dimension(0));
        f2.setConstant(type(1));

        Tensor<type, 1> f3(combinations.dimension(0));
        f3 = static_cast<type>(0.2) * combinations + static_cast<type>(0.5);

        Tensor<type, 1> f4(combinations.dimension(0));
        f4.setConstant(type(0));

        Tensor<type, 1> f5(combinations.dimension(0));
        f5.setConstant(static_cast<type>(0.2));

        // Activations

        activations.device(*thread_pool_device) = if_sentence.select(f1, elif_sentence.select(f2, f3));

        // Activations Derivatives

        activations_derivatives.device(*thread_pool_device) = if_sentence_2.select(f4, f5);
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1));
        TensorMap<Tensor<type, 2>> activations(activations_data, activations_dimensions(0), activations_dimensions(1));
        TensorMap<Tensor<type, 2>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1));

        const Tensor<bool, 2> if_sentence = combinations < combinations.constant(type(-2.5));

        const Tensor<bool, 2> elif_sentence = combinations > combinations.constant(type(2.5));

        const Tensor<bool, 2> if_sentence_2 = combinations < combinations.constant(type(-2.5)) || combinations > combinations.constant(type(2.5));

        // Sentences

        Tensor<type, 2> f1(combinations.dimension(0), combinations.dimension(1));
        f1.setZero();

        Tensor<type, 2> f2(combinations.dimension(0), combinations.dimension(1));
        f2.setConstant(type(1));

        Tensor<type, 2> f3(combinations.dimension(0), combinations.dimension(1));
        f3 = static_cast<type>(0.2) * combinations + static_cast<type>(0.5);

        Tensor<type, 2> f4(combinations.dimension(0), combinations.dimension(1));
        f4.setConstant(type(0));

        Tensor<type, 2> f5(combinations.dimension(0), combinations.dimension(1));
        f5.setConstant(static_cast<type>(0.2));

        // Activations

        activations.device(*thread_pool_device) = if_sentence.select(f1, elif_sentence.select(f2, f3));

        // Activations Derivatives

        activations_derivatives.device(*thread_pool_device) = if_sentence_2.select(f4, f5);
    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1), combinations_dimensions(2), combinations_dimensions(3));
        TensorMap<Tensor<type, 4>> activations(activations_data, activations_dimensions(0), activations_dimensions(1), activations_dimensions(2), activations_dimensions(3));
        TensorMap<Tensor<type, 4>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1), activations_derivatives_dimensions(2), activations_derivatives_dimensions(3));

        // Conditions

        const Tensor<bool, 4> if_sentence = combinations < combinations.constant(type(-2.5));
        const Tensor<bool, 4> elif_sentence = combinations > combinations.constant(type(2.5));
        const Tensor<bool, 4> if_sentence_2 = combinations < combinations.constant(type(-2.5)) || combinations > combinations.constant(type(2.5));

        // Sentences

        Tensor<type, 4> f1(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));
        f1.setZero();

        Tensor<type, 4> f2(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));
        f2.setConstant(type(1));

        Tensor<type, 4> f3(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));
        f3 = static_cast<type>(0.2) * combinations + static_cast<type>(0.5);

        Tensor<type, 4> f4(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));
        f4.setConstant(type(0));

        Tensor<type, 4> f5(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));
        f5.setConstant(static_cast<type>(0.2));

        // Activations

        activations.device(*thread_pool_device) = if_sentence.select(f1, elif_sentence.select(f2, f3));

        // Activations derivatives

        activations_derivatives.device(*thread_pool_device) = if_sentence_2.select(f4, f5);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::hard_sigmoid_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                           type* activations_data, Tensor<Index, 1>& activations_dimensions,"
               << "                                           type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "Hard sigmoid function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::hyperbolic_tangent(type* x_data, const Tensor<Index, 1>& x_dimensions,
                               type* y_data, const Tensor<Index, 1>& y_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (x_dimensions== y_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::hyperbolic_tangent(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "X and Y vector must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    // Apply function

    const Index rank = x_dimensions.size();

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> x(x_data, x_dimensions(0));
        TensorMap<Tensor<type, 1>> y(y_data, y_dimensions(0));
        y.device(*thread_pool_device) = x.tanh();
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> x(x_data, x_dimensions(0), x_dimensions(1));
        TensorMap<Tensor<type, 2>> y(y_data, y_dimensions(0), y_dimensions(1));
        y.device(*thread_pool_device) = x.tanh();
    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> x(x_data, x_dimensions(0), x_dimensions(1), x_dimensions(2), x_dimensions(3));
        TensorMap<Tensor<type, 4>> y(y_data, y_dimensions(0), y_dimensions(1), y_dimensions(2), y_dimensions(3));
        y.device(*thread_pool_device) = x.tanh();
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::hyperbolic_tangent(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "Hyperbolic tangent function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::hyperbolic_tangent_derivatives(type* combinations_data,
                                           const Tensor<Index, 1>& combinations_dimensions,
                                           type* activations_data,
                                           const Tensor<Index, 1>& activations_dimensions,
                                           type* activations_derivatives_data,
                                           const Tensor<Index, 1>& activations_derivatives_dimensions) const
{
    // Check equal sizes and ranks

    if(combinations_dimensions.size() != activations_dimensions.size())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::hyperbolic_tangent_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                           type* activations_data, Tensor<Index, 1>& activations_dimensions,"
               << "                                           type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "X and Y vector must have the same ranks.\n";

        throw invalid_argument(buffer.str());
    }

    const Tensor<bool, 0> same_dimensions = (combinations_dimensions== activations_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::hyperbolic_tangent_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                           type* activations_data, Tensor<Index, 1>& activations_dimensions,"
               << "                                           type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "X and Y vector must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    const Index rank = combinations_dimensions.size();

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> combinations(combinations_data, combinations_dimensions(0));
        TensorMap<Tensor<type, 1>> activations(activations_data, activations_dimensions(0));
        TensorMap<Tensor<type, 1>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0));

        activations.device(*thread_pool_device) = combinations.tanh();

        // Activations Derivatives

        activations_derivatives.device(*thread_pool_device) = type(1) - activations.square();
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1));
        TensorMap<Tensor<type, 2>> activations(activations_data, activations_dimensions(0), activations_dimensions(1));
        TensorMap<Tensor<type, 2>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1));

        activations.device(*thread_pool_device) = combinations.tanh();

        // Activations Derivatives

        activations_derivatives.device(*thread_pool_device) = type(1) - activations.square();
    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1), combinations_dimensions(2), combinations_dimensions(3));
        TensorMap<Tensor<type, 4>> activations(activations_data, activations_dimensions(0), activations_dimensions(1), activations_dimensions(2), activations_dimensions(3));
        TensorMap<Tensor<type, 4>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1), activations_derivatives_dimensions(2), activations_derivatives_dimensions(3));

        activations.device(*thread_pool_device) = combinations.tanh();

        // Activations Derivatives

        activations_derivatives.device(*thread_pool_device) = type(1) - activations.square();
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::hyperbolic_tangent_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                           type* activations_data, Tensor<Index, 1>& activations_dimensions,"
               << "                                           type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "Hyperbolic tangent function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::threshold(type* x_data, const Tensor<Index, 1>& x_dimensions,
                      type* y_data, const Tensor<Index, 1>& y_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (x_dimensions== y_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::threshold(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "X and Y vector must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    // Apply function

    const Index rank = x_dimensions.size();

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> x(x_data, x_dimensions(0));
        TensorMap<Tensor<type, 1>> y(y_data, y_dimensions(0));

        const Tensor<bool, 1> if_sentence = x >= x.constant(type(0));

        Tensor<type, 1> ones(x.dimension(0));
        ones.setConstant(type(1));

        Tensor<type, 1> zeros(x.dimension(0));
        zeros.setConstant(type(0));

        y.device(*thread_pool_device) = if_sentence.select(ones, zeros);
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> x(x_data, x_dimensions(0), x_dimensions(1));
        TensorMap<Tensor<type, 2>> y(y_data, y_dimensions(0), y_dimensions(1));

        const Tensor<bool, 2> if_sentence = x >= x.constant(type(0));

        Tensor<type, 2> ones(x.dimension(0), x.dimension(1));
        ones.setConstant(type(1));

        Tensor<type, 2> zeros(x.dimension(0), x.dimension(1));
        zeros.setConstant(type(0));

        y.device(*thread_pool_device) = if_sentence.select(ones, zeros);
    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> x(x_data, x_dimensions(0), x_dimensions(1), x_dimensions(2), x_dimensions(3));
        TensorMap<Tensor<type, 4>> y(y_data, y_dimensions(0), y_dimensions(1), y_dimensions(2), y_dimensions(3));

        const Tensor<bool, 4> if_sentence = (x >= x.constant(type(0)));

        Tensor<type, 4> ones(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));
        ones.setConstant(type(1));

        Tensor<type, 4> zeros(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));
        zeros.setConstant(type(0));

        y.device(*thread_pool_device) = if_sentence.select(ones, zeros);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::threshold(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "Threshold function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::threshold_derivatives(type* combinations_data,
                                 const Tensor<Index, 1>& combinations_dimensions,
                                 type* activations_data,
                                 const Tensor<Index, 1>& activations_dimensions,
                                 type* activations_derivatives_data,
                                 const Tensor<Index, 1>& activations_derivatives_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (combinations_dimensions== activations_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::threshold_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                  type* activations_data, Tensor<Index, 1>& activations_dimensions,  "
               << "                                  type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "Combinations and activations must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    threshold(combinations_data, combinations_dimensions, activations_data, activations_dimensions);

    const Tensor<Index, 0> activations_derivatives_size = activations_derivatives_dimensions.prod();

    fill(activations_derivatives_data, activations_derivatives_data + activations_derivatives_size(0), 0);
}


void Layer::symmetric_threshold(type* x_data, const Tensor<Index, 1>& x_dimensions,
                                type* y_data, const Tensor<Index, 1>& y_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (x_dimensions== y_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::symmetric_threshold(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "X and Y vector must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    // Apply function

    const Index rank = x_dimensions.size();

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> x(x_data, x_dimensions(0));
        TensorMap<Tensor<type, 1>> y(y_data, y_dimensions(0));

        const Tensor<bool, 1> if_sentence = x > x.constant(type(0));

        Tensor<type, 1> ones(x.dimension(0));
        ones.setConstant(type(1));

        y.device(*thread_pool_device) = if_sentence.select(ones, -ones);
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> x(x_data, x_dimensions(0), x_dimensions(1));
        TensorMap<Tensor<type, 2>> y(y_data, y_dimensions(0), y_dimensions(1));

        const Tensor<bool, 2> if_sentence = x > x.constant(type(0));

        Tensor<type, 2> ones(x.dimension(0), x.dimension(1));

        ones.setConstant(type(1));

        y.device(*thread_pool_device) = if_sentence.select(ones, -ones);
    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> x(x_data, x_dimensions(0), x_dimensions(1), x_dimensions(2), x_dimensions(3));
        TensorMap<Tensor<type, 4>> y(y_data, y_dimensions(0), y_dimensions(1), y_dimensions(2), y_dimensions(3));

        const Tensor<bool, 4> if_sentence = x > x.constant(type(0));

        Tensor<type, 4> ones(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

        ones.setConstant(type(1));

        y.device(*thread_pool_device) = if_sentence.select(ones, -ones);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::threshold(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "Symmetric threshold function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::symmetric_threshold_derivatives(type* combinations_data,
                                 const Tensor<Index, 1>& combinations_dimensions,
                                 type* activations_data,
                                 const Tensor<Index, 1>& activations_dimensions,
                                 type* activations_derivatives_data,
                                 const Tensor<Index, 1>& activations_derivatives_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (combinations_dimensions== activations_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::symmetric_threshold_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                  type* activations_data, Tensor<Index, 1>& activations_dimensions,  "
               << "                                  type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "Combinations and activations must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    symmetric_threshold(combinations_data, combinations_dimensions, activations_data, activations_dimensions);

    const Tensor<Index, 0> activations_derivatives_size = activations_derivatives_dimensions.prod();

    fill(activations_derivatives_data, activations_derivatives_data + activations_derivatives_size(0), 0);
}


void Layer::rectified_linear(type* x_data, const Tensor<Index, 1>& x_dimensions, type* y_data, const Tensor<Index, 1>& y_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (x_dimensions== y_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::rectified_linear(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "X and Y vector must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    // Apply function

    const Index rank = x_dimensions.size();

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> x(x_data, x_dimensions(0));
        TensorMap<Tensor<type, 1>> y(y_data, y_dimensions(0));

        const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

        Tensor<type, 1> zeros(x.dimension(0));
        zeros.setConstant(type(0));

        y.device(*thread_pool_device) = if_sentence.select(zeros, x);
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> x(x_data, x_dimensions(0), x_dimensions(1));
        TensorMap<Tensor<type, 2>> y(y_data, y_dimensions(0), y_dimensions(1));

        const Tensor<bool, 2> if_sentence = x < x.constant(type(0));

        Tensor<type, 2> zeros(x.dimension(0), x.dimension(1));
        zeros.setConstant(type(0));

        y.device(*thread_pool_device) = if_sentence.select(zeros, x);
    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> x(x_data, x_dimensions(0), x_dimensions(1), x_dimensions(2), x_dimensions(3));
        TensorMap<Tensor<type, 4>> y(y_data, y_dimensions(0), y_dimensions(1), y_dimensions(2), y_dimensions(3));

        const Tensor<bool, 4> if_sentence = x < x.constant(type(0));

        Tensor<type, 4> zeros(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));
        zeros.setConstant(type(0));

        y.device(*thread_pool_device) = if_sentence.select(zeros, x);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::rectified_linear(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "Rectified linearfunction is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::rectified_linear_derivatives(type* combinations_data, const Tensor<Index, 1>& combinations_dimensions,
                                 type* activations_data, const Tensor<Index, 1>& activations_dimensions,
                                 type* activations_derivatives_data, const Tensor<Index, 1>& activations_derivatives_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (combinations_dimensions== activations_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::rectified_linear_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                  type* activations_data, Tensor<Index, 1>& activations_dimensions,  "
               << "                                  type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "Combinations and activations must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    const Index rank = combinations_dimensions.size();

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> combinations(combinations_data, combinations_dimensions(0));
        TensorMap<Tensor<type, 1>> activations(activations_data, activations_dimensions(0));
        TensorMap<Tensor<type, 1>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0));

        const Tensor<bool, 1> if_sentence = combinations < combinations.constant(type(0));

        Tensor<type, 1> zeros(combinations.dimension(0));
        zeros.setZero();

        Tensor<type, 1> ones(combinations.dimension(0));
        ones.setConstant(type(1));

        activations.device(*thread_pool_device) = if_sentence.select(zeros, combinations);
        activations_derivatives.device(*thread_pool_device) = if_sentence.select(zeros, ones);
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1));
        TensorMap<Tensor<type, 2>> activations(activations_data, activations_dimensions(0), activations_dimensions(1));
        TensorMap<Tensor<type, 2>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1));

        const Tensor<bool, 2> if_sentence = combinations < combinations.constant(type(0));

        Tensor<type, 2> zeros(combinations.dimension(0), combinations.dimension(1));
        zeros.setZero();

        Tensor<type, 2> ones(combinations.dimension(0), combinations.dimension(1));
        ones.setConstant(type(1));

        activations.device(*thread_pool_device) = if_sentence.select(zeros, combinations);
        activations_derivatives.device(*thread_pool_device) = if_sentence.select(zeros, ones);
    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1), combinations_dimensions(2), combinations_dimensions(3));
        TensorMap<Tensor<type, 4>> activations(activations_data, activations_dimensions(0), activations_dimensions(1), activations_dimensions(2), activations_dimensions(3));
        TensorMap<Tensor<type, 4>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1), activations_derivatives_dimensions(2), activations_derivatives_dimensions(3));

        const Tensor<bool, 4> if_sentence = combinations < combinations.constant(type(0));

        Tensor<type, 4> zeros(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));
        zeros.setZero();

        Tensor<type, 4> ones(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));
        ones.setConstant(type(1));

        activations.device(*thread_pool_device) = if_sentence.select(zeros, combinations);
        activations_derivatives.device(*thread_pool_device) = if_sentence.select(zeros, ones);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::rectified_linear_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                           type* activations_data, Tensor<Index, 1>& activations_dimensions,"
               << "                                           type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "Rectified linear function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::scaled_exponential_linear(type* x_data, const Tensor<Index, 1>& x_dimensions,
                                      type* y_data, const Tensor<Index, 1>& y_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (x_dimensions== y_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::scaled_exponential_linear(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "X and Y vector must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    // Apply function

    const Index rank = x_dimensions.size();

    const type lambda = static_cast<type>(1.0507);

    const type alpha = static_cast<type>(1.67326);

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> x(x_data, x_dimensions(0));
        TensorMap<Tensor<type, 1>> y(y_data, y_dimensions(0));

        const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

        Tensor<type, 1> f_1(x.dimension(0));

        Tensor<type, 1> f_2(x.dimension(0));

        f_1 = lambda*alpha*(x.exp()-static_cast<type>(1.0));

        f_2 = lambda*x;

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> x(x_data, x_dimensions(0), x_dimensions(1));
        TensorMap<Tensor<type, 2>> y(y_data, y_dimensions(0), y_dimensions(1));

        const Tensor<bool, 2> if_sentence = x < x.constant(type(0));

        Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

        Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

        f_1 = lambda*alpha*(x.exp() - static_cast<type>(1.0));

        f_2 = lambda*x;

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> x(x_data, x_dimensions(0), x_dimensions(1), x_dimensions(2), x_dimensions(3));
        TensorMap<Tensor<type, 4>> y(y_data, y_dimensions(0), y_dimensions(1), y_dimensions(2), y_dimensions(3));

        const Tensor<bool, 4> if_sentence = x < x.constant(type(0));

        Tensor<type, 4> f_1(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

        Tensor<type, 4> f_2(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

        f_1 = lambda*alpha*(x.exp() - static_cast<type>(1.0));

        f_2 = lambda*x;

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::scaled_exponential_linear(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "Scaled exponential linear function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::scaled_exponential_linear_derivatives(type* combinations_data, const Tensor<Index, 1>& combinations_dimensions,
                                 type* activations_data, const Tensor<Index, 1>& activations_dimensions,
                                 type* activations_derivatives_data, const Tensor<Index, 1>& activations_derivatives_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (combinations_dimensions== activations_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::scaled_exponential_linear_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                  type* activations_data, Tensor<Index, 1>& activations_dimensions,  "
               << "                                  type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "Combinations and activations must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    const Index rank = combinations_dimensions.size();

    const type lambda = static_cast<type>(1.0507);

    const type alpha = static_cast<type>(1.67326);

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> combinations(combinations_data, combinations_dimensions(0));
        TensorMap<Tensor<type, 1>> activations(activations_data, activations_dimensions(0));
        TensorMap<Tensor<type, 1>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0));

        const Tensor<bool, 1> if_sentence = combinations < combinations.constant(type(0));

        Tensor<type, 1> f_1(combinations.dimension(0));

        Tensor<type, 1> f_2(combinations.dimension(0));

        f_1 = lambda*alpha*(combinations.exp()-static_cast<type>(1.0));

        f_2 = lambda*combinations;

        // Activations

        activations.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = lambda*alpha*combinations.exp();

        f_2 = combinations.constant(type(1))*lambda;

        activations_derivatives.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1));
        TensorMap<Tensor<type, 2>> activations(activations_data, activations_dimensions(0), activations_dimensions(1));
        TensorMap<Tensor<type, 2>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1));

        const Tensor<bool, 2> if_sentence = combinations < combinations.constant(type(0));

        Tensor<type, 2> f_1(combinations.dimension(0), combinations.dimension(1));

        Tensor<type, 2> f_2(combinations.dimension(0), combinations.dimension(1));

        f_1 = lambda*alpha*(combinations.exp()-static_cast<type>(1.0));

        f_2 = lambda*combinations;

        // Activations

        activations.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = lambda*alpha*combinations.exp();

        f_2 = combinations.constant(type(1))*lambda;

        activations_derivatives.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1), combinations_dimensions(2), combinations_dimensions(3));
        TensorMap<Tensor<type, 4>> activations(activations_data, activations_dimensions(0), activations_dimensions(1), activations_dimensions(2), activations_dimensions(3));
        TensorMap<Tensor<type, 4>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1), activations_derivatives_dimensions(2), activations_derivatives_dimensions(3));

        const Tensor<bool, 4> if_sentence = combinations < combinations.constant(type(0));

        Tensor<type, 4> f_1(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));

        Tensor<type, 4> f_2(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));

        f_1 = lambda*alpha*(combinations.exp()-static_cast<type>(1.0));

        f_2 = lambda*combinations;

        // Activations

        activations.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = lambda*alpha*combinations.exp();

        f_2 = combinations.constant(type(1))*lambda;

        activations_derivatives.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::scaled_exponential_linear_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                           type* activations_data, Tensor<Index, 1>& activations_dimensions,"
               << "                                           type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "Scaled exponential linear function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::soft_plus(type* x_data, const Tensor<Index, 1>& x_dimensions,
                      type* y_data, const Tensor<Index, 1>& y_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (x_dimensions== y_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::soft_plus(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "X and Y vector must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    // Apply function

    const Index rank = x_dimensions.size();

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> x(x_data, x_dimensions(0));
        TensorMap<Tensor<type, 1>> y(y_data, y_dimensions(0));

        y.device(*thread_pool_device) = (x.constant(type(1)) + x.exp()).log();
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> x(x_data, x_dimensions(0), x_dimensions(1));
        TensorMap<Tensor<type, 2>> y(y_data, y_dimensions(0), y_dimensions(1));

        y.device(*thread_pool_device) = (x.constant(type(1)) + x.exp()).log();
    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> x(x_data, x_dimensions(0), x_dimensions(1), x_dimensions(2), x_dimensions(3));
        TensorMap<Tensor<type, 4>> y(y_data, y_dimensions(0), y_dimensions(1), y_dimensions(2), y_dimensions(3));

        y.device(*thread_pool_device) = (x.constant(type(1)) + x.exp()).log();
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::soft_plus(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "Soft plus is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::soft_plus_derivatives(type* combinations_data,
                                  const Tensor<Index, 1>& combinations_dimensions,
                                  type* activations_data,
                                  const Tensor<Index, 1>& activations_dimensions,
                                  type* activations_derivatives_data,
                                  const Tensor<Index, 1>& activations_derivatives_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (combinations_dimensions== activations_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::soft_plus_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                  type* activations_data, Tensor<Index, 1>& activations_dimensions,  "
               << "                                  type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "Combinations and activations must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    const Index rank = combinations_dimensions.size();

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> combinations(combinations_data, combinations_dimensions(0));
        TensorMap<Tensor<type, 1>> activations(activations_data, activations_dimensions(0));
        TensorMap<Tensor<type, 1>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0));

        activations.device(*thread_pool_device) = (combinations.constant(type(1)) + combinations.exp()).log();
        activations_derivatives.device(*thread_pool_device) = static_cast<type>(1.0) / (static_cast<type>(1.0) + combinations.exp().inverse());
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1));
        TensorMap<Tensor<type, 2>> activations(activations_data, activations_dimensions(0), activations_dimensions(1));
        TensorMap<Tensor<type, 2>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1));

        activations.device(*thread_pool_device)
                = (combinations.constant(type(1)) + combinations.exp()).log();

        activations_derivatives.device(*thread_pool_device)
                = static_cast<type>(1.0) / (static_cast<type>(1.0) + combinations.exp().inverse());

    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1), combinations_dimensions(2), combinations_dimensions(3));
        TensorMap<Tensor<type, 4>> activations(activations_data, activations_dimensions(0), activations_dimensions(1), activations_dimensions(2), activations_dimensions(3));
        TensorMap<Tensor<type, 4>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1), activations_derivatives_dimensions(2), activations_derivatives_dimensions(3));

        activations.device(*thread_pool_device) = (combinations.constant(type(1)) + combinations.exp()).log();
        activations_derivatives.device(*thread_pool_device) = static_cast<type>(1.0) / (static_cast<type>(1.0) + combinations.exp().inverse());
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::soft_plus_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                           type* activations_data, Tensor<Index, 1>& activations_dimensions,"
               << "                                           type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "Soft plus function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::soft_sign(type* x_data, const Tensor<Index, 1>& x_dimensions,
                      type* y_data, const Tensor<Index, 1>& y_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (x_dimensions== y_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::soft_sign(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "X and Y vector must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    // Apply function

    const Index rank = x_dimensions.size();

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> x(x_data, x_dimensions(0));
        TensorMap<Tensor<type, 1>> y(y_data, y_dimensions(0));

        const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

        Tensor<type, 1> f_1(x.dimension(0));

        Tensor<type, 1> f_2(x.dimension(0));

        f_1 = x / (static_cast<type>(1) - x);

        f_2 = x / (static_cast<type>(1) + x);

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> x(x_data, x_dimensions(0), x_dimensions(1));
        TensorMap<Tensor<type, 2>> y(y_data, y_dimensions(0), y_dimensions(1));

        const Tensor<bool, 2> if_sentence = x < x.constant(type(0));

        Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

        Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

        f_1 = x / (static_cast<type>(1) - x);

        f_2 = x / (static_cast<type>(1) + x);

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> x(x_data, x_dimensions(0), x_dimensions(1), x_dimensions(2), x_dimensions(3));
        TensorMap<Tensor<type, 4>> y(y_data, y_dimensions(0), y_dimensions(1), y_dimensions(2), y_dimensions(3));

        const Tensor<bool, 4> if_sentence = x < x.constant(type(0));

        Tensor<type, 4> f_1(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

        Tensor<type, 4> f_2(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

        f_1 = x / (static_cast<type>(1) - x);

        f_2 = x / (static_cast<type>(1) + x);

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::soft_sign(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "Soft sign function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::soft_sign_derivatives(type* combinations_data, const Tensor<Index, 1>& combinations_dimensions,
                                 type* activations_data, const Tensor<Index, 1>& activations_dimensions,
                                 type* activations_derivatives_data, const Tensor<Index, 1>& activations_derivatives_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (combinations_dimensions== activations_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::soft_sign_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                  type* activations_data, Tensor<Index, 1>& activations_dimensions,  "
               << "                                  type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "Combinations and activations must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    const Index rank = combinations_dimensions.size();

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> combinations(combinations_data, combinations_dimensions(0));
        TensorMap<Tensor<type, 1>> activations(activations_data, activations_dimensions(0));
        TensorMap<Tensor<type, 1>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0));

        const Tensor<bool, 1> if_sentence = combinations < combinations.constant(type(0));

        Tensor<type, 1> f_1(combinations.dimension(0));

        Tensor<type, 1> f_2(combinations.dimension(0));

        f_1 = combinations / (static_cast<type>(1) - combinations);

        f_2 = combinations / (static_cast<type>(1) + combinations);

        // Activations

        activations.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = static_cast<type>(1.0) / (static_cast<type>(1.0) - combinations).pow(type(2));

        f_2 = static_cast<type>(1.0) / (static_cast<type>(1.0) + combinations).pow(type(2));

        activations_derivatives.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1));
        TensorMap<Tensor<type, 2>> activations(activations_data, activations_dimensions(0), activations_dimensions(1));
        TensorMap<Tensor<type, 2>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1));

        const Tensor<bool, 2> if_sentence = combinations < combinations.constant(type(0));

        Tensor<type, 2> f_1(combinations.dimension(0), combinations.dimension(1));

        Tensor<type, 2> f_2(combinations.dimension(0), combinations.dimension(1));

        f_1 = combinations / (static_cast<type>(1) - combinations);

        f_2 = combinations / (static_cast<type>(1) + combinations);

        // Activations

        activations.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = static_cast<type>(1.0) / (static_cast<type>(1.0) - combinations).pow(type(2));

        f_2 = static_cast<type>(1.0) / (static_cast<type>(1.0) + combinations).pow(type(2));

        activations_derivatives.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1), combinations_dimensions(2), combinations_dimensions(3));
        TensorMap<Tensor<type, 4>> activations(activations_data, activations_dimensions(0), activations_dimensions(1), activations_dimensions(2), activations_dimensions(3));
        TensorMap<Tensor<type, 4>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1), activations_derivatives_dimensions(2), activations_derivatives_dimensions(3));

        const Tensor<bool, 4> if_sentence = combinations < combinations.constant(type(0));

        Tensor<type, 4> f_1(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));

        Tensor<type, 4> f_2(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));

        f_1 = combinations / (static_cast<type>(1) - combinations);

        f_2 = combinations / (static_cast<type>(1) + combinations);

        // Activations

        activations.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = static_cast<type>(1.0) / (static_cast<type>(1.0) - combinations).pow(type(2));

        f_2 = static_cast<type>(1.0) / (static_cast<type>(1.0) + combinations).pow(type(2));

        activations_derivatives.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::soft_sign_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                           type* activations_data, Tensor<Index, 1>& activations_dimensions,"
               << "                                           type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "Soft sign function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::exponential_linear(type* x_data, const Tensor<Index, 1>& x_dimensions,
                               type* y_data, const Tensor<Index, 1>& y_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (x_dimensions== y_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::exponential_linear(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "X and Y vector must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    // Apply function

    const Index rank = x_dimensions.size();

    const type alpha = static_cast<type>(1.0);

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> x(x_data, x_dimensions(0));
        TensorMap<Tensor<type, 1>> y(y_data, y_dimensions(0));

        const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

        Tensor<type, 1> f_1(x.dimension(0));

        Tensor<type, 1> f_2(x.dimension(0));

        f_1.device(*thread_pool_device) = alpha*(x.exp() - static_cast<type>(1));

        f_2 = x;

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> x(x_data, x_dimensions(0), x_dimensions(1));
        TensorMap<Tensor<type, 2>> y(y_data, y_dimensions(0), y_dimensions(1));

        const Tensor<bool, 2> if_sentence = x < x.constant(type(0));

        Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

        Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

        f_1 = alpha*(x.exp() - static_cast<type>(1));

        f_2 = x;

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> x(x_data, x_dimensions(0), x_dimensions(1), x_dimensions(2), x_dimensions(3));
        TensorMap<Tensor<type, 4>> y(y_data, y_dimensions(0), y_dimensions(1), y_dimensions(2), y_dimensions(3));

        const Tensor<bool, 4> if_sentence = x < x.constant(type(0));

        Tensor<type, 4> f_1(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

        Tensor<type, 4> f_2(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

        f_1 = alpha*(x.exp() - static_cast<type>(1));

        f_2 = x;

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::exponential_linear(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "Exponential linear is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::exponential_linear_derivatives(type* combinations_data, const Tensor<Index, 1>& combinations_dimensions,
                                 type* activations_data, const Tensor<Index, 1>& activations_dimensions,
                                 type* activations_derivatives_data, const Tensor<Index, 1>& activations_derivatives_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (combinations_dimensions== activations_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::exponential_linear_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                  type* activations_data, Tensor<Index, 1>& activations_dimensions,  "
               << "                                  type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "Combinations and activations must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    const Index rank = combinations_dimensions.size();

    const type alpha = static_cast<type>(1.0);

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> combinations(combinations_data, combinations_dimensions(0));
        TensorMap<Tensor<type, 1>> activations(activations_data, activations_dimensions(0));
        TensorMap<Tensor<type, 1>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0));

        const Tensor<bool, 1> if_sentence = combinations < combinations.constant(type(0));

        Tensor<type, 1> f_1(combinations.dimension(0));
        f_1 = alpha*(combinations.exp() - static_cast<type>(1));

        Tensor<type, 1> f_2(combinations.dimension(0));
        f_2 = combinations;

        // Activations

        activations.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = alpha * combinations.exp();

        f_2 = combinations.constant(type(1));

        activations_derivatives.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1));
        TensorMap<Tensor<type, 2>> activations(activations_data, activations_dimensions(0), activations_dimensions(1));
        TensorMap<Tensor<type, 2>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1));

        const Tensor<bool, 2> if_sentence = combinations < combinations.constant(type(0));

        Tensor<type, 2> f_1(combinations.dimension(0), combinations.dimension(1));

        Tensor<type, 2> f_2(combinations.dimension(0), combinations.dimension(1));

        f_1 = alpha*(combinations.exp() - static_cast<type>(1));

        f_2 = combinations;

        // Activations

        activations.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = alpha * combinations.exp();

        f_2 = combinations.constant(type(1));

        activations_derivatives.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else if(rank == 4)
    {
        const TensorMap<Tensor<type, 4>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1), combinations_dimensions(2), combinations_dimensions(3));
        TensorMap<Tensor<type, 4>> activations(activations_data, activations_dimensions(0), activations_dimensions(1), activations_dimensions(2), activations_dimensions(3));
        TensorMap<Tensor<type, 4>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1), activations_derivatives_dimensions(2), activations_derivatives_dimensions(3));

        const Tensor<bool, 4> if_sentence = combinations < combinations.constant(type(0));

        Tensor<type, 4> f_1(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));

        Tensor<type, 4> f_2(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));

        f_1 = alpha*(combinations.exp() - static_cast<type>(1));

        f_2 = combinations;

        // Activations

        activations.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = alpha * combinations.exp();

        f_2 = combinations.constant(type(1));

        activations_derivatives.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::exponential_linear_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                           type* activations_data, Tensor<Index, 1>& activations_dimensions,"
               << "                                           type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "Exponential linear function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::softmax(type* x_data, const Tensor<Index, 1>& x_dimensions,
                    type* y_data, const Tensor<Index, 1>& y_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (x_dimensions== y_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::softmax(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "X and Y vector must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    // Apply function

    const Index rank = x_dimensions.size();

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> x(x_data, x_dimensions(0));
        TensorMap<Tensor<type, 1>> y(y_data, y_dimensions(0));

        Tensor<type, 0> sum;

        sum.device(*thread_pool_device) = x.exp().sum();

        y.device(*thread_pool_device) = x.exp() / sum(0);
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> x(x_data, x_dimensions(0), x_dimensions(1));
        TensorMap<Tensor<type, 2>> y(y_data, y_dimensions(0), y_dimensions(1));

        const Index columns_number = x.dimension(1);
        const Index rows_number = x.dimension(0);

        y.device(*thread_pool_device) = x.exp();

        Tensor<type, 1> inverse_sums(rows_number);
        inverse_sums.setZero();

        Eigen::array<int, 1> dims({1}); // Eigen reduction cols Axis
        inverse_sums = y.sum(dims).inverse();

    #pragma omp parallel for
        for (Index i = 0; i < columns_number; i++)
        {
            const TensorMap<Tensor<type, 1>> single_col(y.data()+rows_number*i, rows_number);

            const Tensor<type, 1> tmp_result = single_col*inverse_sums;

            memcpy(y.data() + rows_number*i,
                   tmp_result.data(), static_cast<size_t>(rows_number)*sizeof(type));
        }

    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::softmax(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "Softmax function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::softmax_derivatives(type* combinations_data, const Tensor<Index, 1>& combinations_dimensions,
                                 type* activations_data, const Tensor<Index, 1>& activations_dimensions,
                                 type* activations_derivatives_data, const Tensor<Index, 1>& activations_derivatives_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (combinations_dimensions== activations_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::softmax_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                  type* activations_data, Tensor<Index, 1>& activations_dimensions,  "
               << "                                  type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "Combinations and activations must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    const Index rank = combinations_dimensions.size();

    const Index activations_derivatives_rank = activations_derivatives_dimensions.size();

    if(rank == 2)
    {
        softmax(combinations_data, combinations_dimensions, activations_data, activations_dimensions);

        if(activations_derivatives_rank == 2)
        {
            const TensorMap<Tensor<type, 2>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1));
            TensorMap<Tensor<type, 2>> activations(activations_data, activations_dimensions(0), activations_dimensions(1));
            TensorMap<Tensor<type, 2>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1));

            const Index dim = combinations.dimension(1);

            const Index matrix_number = activations.dimension(0);

            type delta = type(0);
            Index index= 0;

            for(Index row = 0; row < matrix_number; row++)
            {
                for(Index i = 0; i < dim; i++)
                {
                    for(Index j = 0; j < dim; j++)
                    {
                        (i == j) ? delta = type(1) : delta = type(0);

                        // row, i, j

                        activations_derivatives(index) = activations(row,j) * (delta - activations(row,i));
                        index++;
                    }
                }
            }
        }
        else if(activations_derivatives_rank == 3)
        {
            const TensorMap<Tensor<type, 2>> combinations(combinations_data, combinations_dimensions(0), combinations_dimensions(1));
            TensorMap<Tensor<type, 2>> activations(activations_data, activations_dimensions(0), activations_dimensions(1));
            TensorMap<Tensor<type, 3>> activations_derivatives(activations_derivatives_data, activations_derivatives_dimensions(0), activations_derivatives_dimensions(1), activations_derivatives_dimensions(2));

            const Index dim = combinations.dimension(1);

            const Index matrix_number = activations.dimension(0);

            type delta = type(0);
            Index index= 0;

            for(Index row = 0; row < matrix_number; row++)
            {
                for(Index i = 0; i < dim; i++)
                {
                    for(Index j = 0; j < dim; j++)
                    {
                        (i == j) ? delta = type(1) : delta = type(0);

                        // row, i, j

                        activations_derivatives(index) = activations(row,j) * (delta - activations(row,i));
                        index++;
                    }
                }
            }
        }

    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::softmax_derivatives(type* combinations_data, Tensor<Index, 1>& combinations_dimensions,"
               << "                                           type* activations_data, Tensor<Index, 1>& activations_dimensions,"
               << "                                           type* activations_derivatives_data, Tensor<Index, 1>& activations_derivatives_dimensions) "
               << "Softmax function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::binary(type* x_data, const Tensor<Index, 1>& x_dimensions, type* y_data, const Tensor<Index, 1>& y_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (x_dimensions== y_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::binary(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "X and Y vector must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    // Apply function

    const Index rank = x_dimensions.size();

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> x(x_data, x_dimensions(0));
        TensorMap<Tensor<type, 1>> y(y_data, y_dimensions(0));

        const Tensor<bool, 1> if_sentence = x < x.constant(type(0.5));

        Tensor<type, 1> f_1(x.dimension(0));

        Tensor<type, 1> f_2(x.dimension(0));

        f_1 = x.constant(type(false));

        f_2 = x.constant(type(true));

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> x(x_data, x_dimensions(0), x_dimensions(1));
        TensorMap<Tensor<type, 2>> y(y_data, y_dimensions(0), y_dimensions(1));

        const Tensor<bool, 2> if_sentence = x < x.constant(type(0));

        Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

        Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

        f_1 = x.constant(type(false));

        f_2 = x.constant(type(true));

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::exponential_linear(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "Binary function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


void Layer::competitive(type* x_data, const Tensor<Index, 1>& x_dimensions, type* y_data, const Tensor<Index, 1>& y_dimensions) const
{
    // Check equal sizes and ranks

    const Tensor<bool, 0> same_dimensions = (x_dimensions== y_dimensions).all();

    if(!same_dimensions(0))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::binary(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "X and Y vector must have the same dimensions.\n";

        throw invalid_argument(buffer.str());
    }

    // Apply function

    const Index rank = x_dimensions.size();

    if(rank == 1)
    {
        const TensorMap<Tensor<type, 1>> x(x_data, x_dimensions(0));
        TensorMap<Tensor<type, 1>> y(y_data, y_dimensions(0));

        y.setZero();

        const Index index = maximal_index(x);

        y(index) = type(1);
    }
    else if(rank == 2)
    {
        const TensorMap<Tensor<type, 2>> x(x_data, x_dimensions(0), x_dimensions(1));
        TensorMap<Tensor<type, 2>> y(y_data, y_dimensions(0), y_dimensions(1));

        const Index samples_number = x.dimension(0);

        Index maximum_index = 0;

        y.setZero();

        for(Index i = 0; i < samples_number; i++)
        {
            maximum_index = maximal_index(x.chip(i, 1));

            y(i, maximum_index) = type(1);
        }
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer class.\n"
               << "void Layer::exponential_linear(type* x_data, Tensor<Index, 1>& x_dimensions, type* y_data, Tensor<Index, 1>& y_dimensions) const.\n"
               << "Binary function is not implemented for rank " << rank << ".\n";

        throw invalid_argument(buffer.str());
    }
}


}

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
