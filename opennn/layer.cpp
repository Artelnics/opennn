//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "layer.h"

namespace OpenNN
{

Layer::~Layer()
{
    delete thread_pool;
    delete thread_pool_device;
}


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
    }

    return string();
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


void Layer::set_parameters(const Tensor<type, 1>&, const Index&)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_parameters(const Tensor<type, 1>&) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


Index Layer::get_parameters_number() const
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


Tensor<type, 2> Layer::calculate_outputs(const Tensor<type, 2>&)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "calculate_outputs(const Tensor<type, 2>&) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


/// Returns the number of inputs

Index Layer::get_inputs_number() const
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "get_inputs_number() const method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


Index Layer::get_neurons_number() const
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "get_neurons_number() const method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


/// Returns the number of layer's synaptic weights

Index Layer::get_synaptic_weights_number() const
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "get_synaptic_weight_number() const method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


void Layer::set_inputs_number(const Index& )
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_inputs_number(const Index& ) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


void Layer::set_neurons_number(const Index& )
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_neurons_number(const Index& ) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


void Layer::hard_sigmoid(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{
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


void Layer::hyperbolic_tangent(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{
    y.device(*thread_pool_device) = x.tanh();
}


void Layer::logistic(const Tensor<type, 1>& x, Tensor<type, 1>& y)const
{
    y.device(*thread_pool_device) = (type(1) + x.exp().inverse()).inverse();
}


void Layer::linear(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{
    y = x;
}


void Layer::threshold(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{
    const Tensor<bool, 1> if_sentence = x >= x.constant(type(0));

    Tensor<type, 1> ones(x.dimension(0));
    ones.setConstant(type(1));

    Tensor<type, 1> zeros(x.dimension(0));
    zeros.setConstant(type(0));

    y.device(*thread_pool_device) = if_sentence.select(ones, zeros);
}


void Layer::symmetric_threshold(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{

    const Tensor<bool, 1> if_sentence = x > x.constant(type(0));

    Tensor<type, 1> ones(x.dimension(0));
    ones.setConstant(type(1));

    y.device(*thread_pool_device) = if_sentence.select(ones, -ones);
}


void Layer::rectified_linear(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{
    const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

    Tensor<type, 1> zeros(x.dimension(0));

    zeros.setConstant(type(0));

    y.device(*thread_pool_device) = if_sentence.select(zeros, x);
}


void Layer::scaled_exponential_linear(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{

    const type lambda = static_cast<type>(1.0507);

    const type alpha = static_cast<type>(1.67326);

    const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

    Tensor<type, 1> f_1(x.dimension(0));

    Tensor<type, 1> f_2(x.dimension(0));

    f_1 = lambda*alpha*(x.exp()-static_cast<type>(1.0));

    f_2 = lambda*x;

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


void Layer::soft_plus(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{
    y.device(*thread_pool_device) = (x.constant(type(1)) + x.exp()).log();

}


void Layer::soft_sign(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{

    const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

    Tensor<type, 1> f_1(x.dimension(0));

    Tensor<type, 1> f_2(x.dimension(0));

    f_1 = x / (static_cast<type>(1) - x);

    f_2 = x / (static_cast<type>(1) + x);

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


void Layer::exponential_linear(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{
    const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

    const type alpha = static_cast<type>(1.0);

    Tensor<type, 1> f_1(x.dimension(0));

    Tensor<type, 1> f_2(x.dimension(0));

    f_1.device(*thread_pool_device) = alpha*(x.exp() - static_cast<type>(1));

    f_2 = x;

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


void Layer::binary(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{
    const Tensor<bool, 1> if_sentence = x < x.constant(type(0.5));

    Tensor<type, 1> f_1(x.dimension(0));

    Tensor<type, 1> f_2(x.dimension(0));

    f_1 = x.constant(type(false));

    f_2 = x.constant(type(true));

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


void Layer::competitive(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{
    y.setZero();

    const Index index = maximal_index(x);

    y(index) = type(1);
}


void Layer::softmax(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{
    Tensor<type, 0> sum;

    sum.device(*thread_pool_device) = x.exp().sum();

    y.device(*thread_pool_device) = x.exp() / sum(0);
}


void Layer::hard_sigmoid_derivatives(const Tensor<type, 1>& combinations,
                                     Tensor<type, 1>& activations,
                                     Tensor<type, 1>& activations_derivatives) const
{
    // Conditions

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

void Layer::hyperbolic_tangent_derivatives(const Tensor<type, 1>& combinations,
                                           Tensor<type, 1>& activations,
                                           Tensor<type, 1>& activations_derivatives) const
{
    // Activations

    activations.device(*thread_pool_device) = combinations.tanh();

    // Activations Derivatives

    activations_derivatives.device(*thread_pool_device) = type(1) - activations.square();
}


void Layer::logistic_derivatives(const Tensor<type, 1>& combinations,
                                 Tensor<type, 1>& activations,
                                 Tensor<type, 1>& activations_derivatives) const
{
    activations.device(*thread_pool_device) = (type(1) + combinations.exp().inverse()).inverse();

    activations_derivatives.device(*thread_pool_device) = activations*(type(1) - activations);
}


void Layer::linear_derivatives(const Tensor<type, 1>& combinations,
                               Tensor<type, 1>& activations,
                               Tensor<type, 1>& activations_derivatives) const
{
    activations = combinations;

    activations_derivatives.setConstant(type(1));
}


void Layer::threshold_derivatives(const Tensor<type, 1>& combinations,
                                  Tensor<type, 1>& activations,
                                  Tensor<type, 1>& activations_derivatives) const
{
    const Tensor<bool, 1> if_sentence = combinations > combinations.constant(type(0));

    Tensor<type, 1> ones(combinations.dimension(0));
    ones.setConstant(type(1));

    Tensor<type, 1> zeros(combinations.dimension(0));
    zeros.setConstant(type(0));

    // Activations

    activations.device(*thread_pool_device) = if_sentence.select(ones, zeros);

    // Activations Derivatives

    activations_derivatives.setZero();
}


void Layer::symmetric_threshold_derivatives(const Tensor<type, 1>& combinations,
                                            Tensor<type, 1>& activations,
                                            Tensor<type, 1>& activations_derivatives) const
{
    const Tensor<bool, 1> if_sentence = combinations > combinations.constant(type(0));

    Tensor<type, 1> ones(combinations.dimension(0));

    ones.setConstant(type(1));

    // Activations

    activations.device(*thread_pool_device) = if_sentence.select(ones, -ones);

    // Activations Derivatives

    activations_derivatives.setZero();
}


void Layer::rectified_linear_derivatives(const Tensor<type, 1>& combinations,
                                         Tensor<type, 1>& activations,
                                         Tensor<type, 1>& activations_derivatives) const
{
    const Tensor<bool, 1> if_sentence = combinations < combinations.constant(type(0));

    Tensor<type, 1> zeros(combinations.dimension(0));
    zeros.setZero();

    Tensor<type, 1> ones(combinations.dimension(0));
    ones.setConstant(type(1));

    // Activations

    activations.device(*thread_pool_device) = if_sentence.select(zeros, combinations);

    // Activations Derivatives

    activations_derivatives.device(*thread_pool_device) = if_sentence.select(zeros, ones);
}


void Layer::scaled_exponential_linear_derivatives(const Tensor<type, 1>& combinations,
                                                  Tensor<type, 1>& activations,
                                                  Tensor<type, 1>& activations_derivatives) const
{
    const type lambda = static_cast<type>(1.0507);

    const type alpha = static_cast<type>(1.67326);

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


void Layer::soft_plus_derivatives(const Tensor<type, 1>& combinations,
                                  Tensor<type, 1>& activations,
                                  Tensor<type, 1>& activations_derivatives) const
{
    activations.device(*thread_pool_device) = (combinations.constant(type(1)) + combinations.exp()).log();

    activations_derivatives.device(*thread_pool_device) = static_cast<type>(1.0) / (static_cast<type>(1.0) + combinations.exp().inverse());
}


void Layer::soft_sign_derivatives(const Tensor<type, 1>& combinations,
                                  Tensor<type, 1>& activations,
                                  Tensor<type, 1>& activations_derivatives) const
{
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


void Layer::exponential_linear_derivatives(const Tensor<type, 1>& combinations,
                                           Tensor<type, 1>& activations,
                                           Tensor<type, 1>& activations_derivatives) const
{

    const type alpha = static_cast<type>(1.0);

    const Tensor<bool, 1> if_sentence = combinations < combinations.constant(type(0));

    Tensor<type, 1> f_1(combinations.dimension(0));

    Tensor<type, 1> f_2(combinations.dimension(0));

    f_1 = alpha*(combinations.exp() - static_cast<type>(1));

    f_2 = combinations;

    // Activations

    activations.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

    // Activations Derivatives

    f_1 = alpha * combinations.exp();

    f_2 = combinations.constant(type(1));

    activations_derivatives.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


// Activations 2d

void Layer::hard_sigmoid(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
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


void Layer::hyperbolic_tangent(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    y.device(*thread_pool_device) = x.tanh();
}


void Layer::logistic(const Tensor<type, 2>& x, Tensor<type, 2>& y)const
{
    y.device(*thread_pool_device) = (type(1) + x.exp().inverse()).inverse();
}


void Layer::linear(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    y = x;
}


void Layer::threshold(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    const Tensor<bool, 2> if_sentence = x >= x.constant(type(0));

    Tensor<type, 2> ones(x.dimension(0), x.dimension(1));
    ones.setConstant(type(1));

    Tensor<type, 2> zeros(x.dimension(0), x.dimension(1));
    zeros.setConstant(type(0));

    y.device(*thread_pool_device) = if_sentence.select(ones, zeros);
}


void Layer::symmetric_threshold(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    const Tensor<bool, 2> if_sentence = x > x.constant(type(0));

    Tensor<type, 2> ones(x.dimension(0), x.dimension(1));

    ones.setConstant(type(1));

    y.device(*thread_pool_device) = if_sentence.select(ones, -ones);
}


void Layer::rectified_linear(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    const Tensor<bool, 2> if_sentence = x < x.constant(type(0));

    Tensor<type, 2> zeros(x.dimension(0), x.dimension(1));

    zeros.setConstant(type(0));

    y.device(*thread_pool_device) = if_sentence.select(zeros, x);
}


void Layer::scaled_exponential_linear(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    const type lambda = static_cast<type>(1.0507);

    const type alpha = static_cast<type>(1.67326);

    const Tensor<bool, 2> if_sentence = x < x.constant(type(0));

    Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

    Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

    f_1 = lambda*alpha*(x.exp() - static_cast<type>(1.0));

    f_2 = lambda*x;

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


void Layer::soft_plus(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    y.device(*thread_pool_device) = (x.constant(type(1)) + x.exp()).log();
}


void Layer::soft_sign(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    const Tensor<bool, 2> if_sentence = x < x.constant(type(0));

    Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

    Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

    f_1 = x / (static_cast<type>(1) - x);

    f_2 = x / (static_cast<type>(1) + x);

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


void Layer::exponential_linear(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    const Tensor<bool, 2> if_sentence = x < x.constant(type(0));

    const type alpha = static_cast<type>(1.0);

    Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

    Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

    f_1 = alpha*(x.exp() - static_cast<type>(1));

    f_2 = x;

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


void Layer::binary(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    const Tensor<bool, 2> if_sentence = x < x.constant(type(0));

    Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

    Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

    f_1 = x.constant(type(false));

    f_2 = x.constant(type(true));

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


/// @todo exception with several maximum indices

void Layer::competitive(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    const Index samples_number = x.dimension(0);

    Index maximum_index = 0;

    y.setZero();

    for(Index i = 0; i < samples_number; i++)
    {
        maximum_index = maximal_index(x.chip(i, 1));

        y(i, maximum_index) = type(1);
    }

}


void Layer::softmax(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    const Index columns_number = x.dimension(1);

    const Index rows_number = y.dimension(0);

    // Activations

    y.device(*thread_pool_device) = x.exp();

    Tensor<type, 1> sums(rows_number);
    sums.setZero();

    for(Index i = 0; i< rows_number; i++)
    {
        for(Index j = 0; j < columns_number; j++)
        {
            sums[i] +=  y(i,j);
        }
    }

    for(Index i = 0; i< rows_number; i++)
    {
        for(Index j = 0; j < columns_number; j++)
        {
            y(i, j) = y(i, j) / sums(i);
        }
    }
}


// Activations derivatives 2d

void Layer::hard_sigmoid_derivatives(const Tensor<type, 2>& combinations,
                                     Tensor<type, 2>& activations,
                                     Tensor<type, 2>& activations_derivatives) const
{
    // Conditions

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


void Layer::hyperbolic_tangent_derivatives(const Tensor<type, 2>& combinations,
                                           Tensor<type, 2>& activations,
                                           Tensor<type, 2>& activations_derivatives) const
{
    // Activations

    activations.device(*thread_pool_device) = combinations.tanh();

    // Activations Derivatives

    activations_derivatives.device(*thread_pool_device) = type(1) - activations.square();
}


void Layer::logistic_derivatives(const Tensor<type, 2>& combinations,
                                 Tensor<type, 2>& activations,
                                 Tensor<type, 2>& activations_derivatives) const
{
    // Activations

    activations.device(*thread_pool_device) = (type(1) + combinations.exp().inverse()).inverse();

    // Activations Derivatives

    activations_derivatives.device(*thread_pool_device) = activations*(type(1) - activations);
}


void Layer::linear_derivatives(const Tensor<type, 2>& combinations,
                               Tensor<type, 2>& activations,
                               Tensor<type, 2>& activations_derivatives) const
{
    activations = combinations;

    activations_derivatives.setConstant(type(1));
}



void Layer::threshold_derivatives(const Tensor<type, 2>& combinations,
                                  Tensor<type, 2>& activations,
                                  Tensor<type, 2>& activations_derivatives) const
{

    const Tensor<bool, 2> if_sentence = combinations > combinations.constant(type(0));

    Tensor<type, 2> ones(combinations.dimension(0), combinations.dimension(1));
    ones.setConstant(type(1));

    Tensor<type, 2> zeros(combinations.dimension(0), combinations.dimension(1));
    zeros.setConstant(type(0));

    // Activations

    activations.device(*thread_pool_device) = if_sentence.select(ones, zeros);

    // Activations Derivatives

    activations_derivatives.setZero();

}


void Layer::symmetric_threshold_derivatives(const Tensor<type, 2>& combinations,
                                            Tensor<type, 2>& activations,
                                            Tensor<type, 2>& activations_derivatives) const
{

    const Tensor<bool, 2> if_sentence = combinations > combinations.constant(type(0));

    Tensor<type, 2> ones(combinations.dimension(0), combinations.dimension(1));

    ones.setConstant(type(1));

    // Activations

    activations.device(*thread_pool_device) = if_sentence.select(ones, -ones);

    // Activations Derivatives

    activations_derivatives.setZero();
}


void Layer::rectified_linear_derivatives(const Tensor<type, 2>& combinations,
                                         Tensor<type, 2>& activations,
                                         Tensor<type, 2>& activations_derivatives) const
{
    const Tensor<bool, 2> if_sentence = combinations < combinations.constant(type(0));

    Tensor<type, 2> zeros(combinations.dimension(0), combinations.dimension(1));
    zeros.setZero();

    Tensor<type, 2> ones(combinations.dimension(0), combinations.dimension(1));
    ones.setConstant(type(1));

    // Activations

    activations.device(*thread_pool_device) = if_sentence.select(zeros, combinations);

    // Activations Derivatives

    activations_derivatives.device(*thread_pool_device) = if_sentence.select(zeros, ones);
}


void Layer::scaled_exponential_linear_derivatives(const Tensor<type, 2>& combinations,
                                                  Tensor<type, 2>& activations,
                                                  Tensor<type, 2>& activations_derivatives) const
{
    const type lambda = static_cast<type>(1.0507);

    const type alpha = static_cast<type>(1.67326);

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


void Layer::soft_plus_derivatives(const Tensor<type, 2>& combinations,
                                  Tensor<type, 2>& activations,
                                  Tensor<type, 2>& activations_derivatives) const
{
    activations.device(*thread_pool_device)
            = (combinations.constant(type(1)) + combinations.exp()).log();

    activations_derivatives.device(*thread_pool_device)
            = static_cast<type>(1.0) / (static_cast<type>(1.0) + combinations.exp().inverse());
}


void Layer::soft_sign_derivatives(const Tensor<type, 2>& combinations,
                                  Tensor<type, 2>& activations,
                                  Tensor<type, 2>& activations_derivatives) const
{

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


void Layer::exponential_linear_derivatives(const Tensor<type, 2>& combinations,
                                           Tensor<type, 2>& activations,
                                           Tensor<type, 2>& activations_derivatives) const
{
    const type alpha = static_cast<type>(1.0);

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


void Layer::logistic_derivatives(const Tensor<type, 2>& combinations,
                                 Tensor<type, 2>& activations,
                                 Tensor<type, 3>& activations_derivatives) const
{
    // Activations

    activations.device(*thread_pool_device) = (type(1) + combinations.exp().inverse()).inverse();

    // Activations Derivatives

    Tensor<type, 2> derivatives_2d(activations.dimension(0), activations.dimension(1));

    derivatives_2d.device(*thread_pool_device) = activations*(type(1) - activations);

    memcpy(activations_derivatives.data(), derivatives_2d.data(), static_cast<size_t>(derivatives_2d.size())*sizeof(type));
}


void Layer::softmax_derivatives(const Tensor<type, 2>& combinations,
                                 Tensor<type, 2>& activations,
                                 Tensor<type, 3>& activations_derivatives) const
{

    // 20 seconds!!!

        const Index columns_number = combinations.dimension(1);

        const Index rows_number = activations.dimension(0);

        // Activations : 14 seconds

        activations.device(*thread_pool_device) = combinations.exp().sum(Eigen::array<Index, 1>({1})).reshape(Eigen::array<Index,2>{rows_number,1}).broadcast(Eigen::array<Index, 2>({1, columns_number}));

        activations.device(*thread_pool_device) = combinations.exp()/activations;


        // Activations derivatives : 6 seconds

        kronecker_product(activations, activations_derivatives);

        #pragma omp parallel for

        for(Index i = 0; i < activations_derivatives.dimension(2); i++)
        {
            for(Index j = 0; j < activations_derivatives.dimension(1); j++)
            {
                activations_derivatives(j,j,i) += activations(i,j);
            }
        }

    /*
     const Index dim = combinations.dimension(1);

     const Index rows_number = activations.dimension(0);

     //Activations

     activations.device(*thread_pool_device) = combinations.exp();

     Tensor<type, 1> sums(rows_number);

     sums.setZero();

     #pragma omp parallel for

     for(Index i = 0; i< rows_number; i++)
     {
         for(Index j = 0; j < dim; j++)
         {
             sums[i] +=  activations(i,j);
         }
     }

     #pragma omp parallel for

     for(Index i = 0; i< rows_number; i++)
     {
         for(Index j = 0; j < dim; j++)
         {
             activations(i, j) /= sums(i);
         }
     }

     /// @todo Add #pragma parallel for in loops.

     // Activations derivatives

     type delta = type(0);
     Index index= 0;

     for(Index row = 0; row < rows_number; row++)
     {
         for(Index i = 0; i < dim; i++)
         {
             for(Index j = 0; j < dim; j++)
             {
                 (i == j) ? delta = type(1) : delta = type(0);

                 // row, i, j

                 activations_derivatives(index) = activations(row,i) * (delta - activations(row,j));

                 index++;
             }
         }
     }
     */
}


// Activations 4d

void Layer::linear(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{
    y = x;
}


void Layer::logistic(const Tensor<type, 4>& x, Tensor<type, 4>& y)const
{
    y.device(*thread_pool_device) = (type(1) + x.exp().inverse()).inverse();
}


void Layer::hyperbolic_tangent(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{
    y.device(*thread_pool_device) = x.tanh();
}


void Layer::threshold(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{
    const Tensor<bool, 4> if_sentence = (x >= x.constant(type(0)));

    Tensor<type, 4> ones(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));
    ones.setConstant(type(1));

    Tensor<type, 4> zeros(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));
    zeros.setConstant(type(0));

    y.device(*thread_pool_device) = if_sentence.select(ones, zeros);
}


void Layer::symmetric_threshold(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{
    const Tensor<bool, 4> if_sentence = x > x.constant(type(0));

    Tensor<type, 4> ones(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

    ones.setConstant(type(1));

    y.device(*thread_pool_device) = if_sentence.select(ones, -ones);
}


void Layer::rectified_linear(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{
    const Tensor<bool, 4> if_sentence = x < x.constant(type(0));

    Tensor<type, 4> zeros(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

    zeros.setConstant(type(0));

    y.device(*thread_pool_device) = if_sentence.select(zeros, x);
}


void Layer::scaled_exponential_linear(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{
    const type lambda = static_cast<type>(1.0507);

    const type alpha = static_cast<type>(1.67326);

    const Tensor<bool, 4> if_sentence = x < x.constant(type(0));

    Tensor<type, 4> f_1(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

    Tensor<type, 4> f_2(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

    f_1 = lambda*alpha*(x.exp() - static_cast<type>(1.0));

    f_2 = lambda*x;

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


void Layer::soft_plus(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{
    y.device(*thread_pool_device) = (x.constant(type(1)) + x.exp()).log();
}


void Layer::soft_sign(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{
    const Tensor<bool, 4> if_sentence = x < x.constant(type(0));

    Tensor<type, 4> f_1(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

    Tensor<type, 4> f_2(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

    f_1 = x / (static_cast<type>(1) - x);

    f_2 = x / (static_cast<type>(1) + x);

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


/// @todo Test this method.

void Layer::hard_sigmoid(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{
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
    y.device(*thread_pool_device) = if_middle.select(f_middle, f_equal);
}


void Layer::exponential_linear(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{

    const Tensor<bool, 4> if_sentence = x < x.constant(type(0));

    const type alpha = static_cast<type>(1.0);

    Tensor<type, 4> f_1(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

    Tensor<type, 4> f_2(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

    f_1 = alpha*(x.exp() - static_cast<type>(1));

    f_2 = x;

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


void Layer::linear_derivatives(const Tensor<type, 4>& combinations,
                               Tensor<type, 4>& activations,
                               Tensor<type, 4>& activations_derivatives) const
{
    activations = combinations;

    activations_derivatives.setConstant(type(1));
}


void Layer::logistic_derivatives(const Tensor<type, 4>& combinations,
                                 Tensor<type, 4>& activations,
                                 Tensor<type, 4>& activations_derivatives) const
{
    // Activations

    activations.device(*thread_pool_device) = (type(1) + combinations.exp().inverse()).inverse();

    // Activations Derivatives

    activations_derivatives.device(*thread_pool_device) = activations*(type(1) - activations);
}


void Layer::hyperbolic_tangent_derivatives(const Tensor<type, 4>& combinations,
                                           Tensor<type, 4>& activations,
                                           Tensor<type, 4>& activations_derivatives) const
{
    // Activations

    activations.device(*thread_pool_device) = combinations.tanh();

    // Activations Derivatives

    activations_derivatives.device(*thread_pool_device) = type(1) - activations.square();
}


void Layer::threshold_derivatives(const Tensor<type, 4>& combinations,
                                  Tensor<type, 4>& activations,
                                  Tensor<type, 4>& activations_derivatives) const
{
    const Tensor<bool, 4> if_sentence = combinations > combinations.constant(type(0));

    Tensor<type, 4> ones(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));
    ones.setConstant(type(1));

    Tensor<type, 4> zeros(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));
    zeros.setConstant(type(0));

    // Activations

    activations.device(*thread_pool_device) = if_sentence.select(ones, zeros);

    // Activations Derivatives

    activations_derivatives.setZero();
}


void Layer::symmetric_threshold_derivatives(const Tensor<type, 4>& combinations,
                                            Tensor<type, 4>& activations,
                                            Tensor<type, 4>& activations_derivatives) const
{
    const Tensor<bool, 4> if_sentence = combinations > combinations.constant(type(0));

    Tensor<type, 4> ones(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));

    ones.setConstant(type(1));

    // Activations

    activations.device(*thread_pool_device) = if_sentence.select(ones, -ones);

    // Activations Derivatives

    activations_derivatives.setZero();
}


void Layer::rectified_linear_derivatives(const Tensor<type, 4>& combinations,
                                         Tensor<type, 4>& activations,
                                         Tensor<type, 4>& activations_derivatives) const
{

    const Tensor<bool, 4> if_sentence = combinations < combinations.constant(type(0));

    Tensor<type, 4> zeros(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));
    zeros.setZero();

    Tensor<type, 4> ones(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));
    ones.setConstant(type(1));

    // Activations

    activations.device(*thread_pool_device) = if_sentence.select(zeros, combinations);

    // Activations Derivatives

    activations_derivatives.device(*thread_pool_device) = if_sentence.select(zeros, ones);
}


void Layer::scaled_exponential_linear_derivatives(const Tensor<type, 4>& combinations,
                                                  Tensor<type, 4>& activations,
                                                  Tensor<type, 4>& activations_derivatives) const
{

    const type lambda = static_cast<type>(1.0507);

    const type alpha = static_cast<type>(1.67326);

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


void Layer::soft_plus_derivatives(const Tensor<type, 4>& combinations,
                                  Tensor<type, 4>& activations,
                                  Tensor<type, 4>& activations_derivatives) const
{

    // Activations

    activations.device(*thread_pool_device) = (combinations.constant(type(1)) + combinations.exp()).log();

    // Activations Derivatives

    activations_derivatives.device(*thread_pool_device) = static_cast<type>(1.0) / (static_cast<type>(1.0) + combinations.exp().inverse());

}


void Layer::soft_sign_derivatives(const Tensor<type, 4>& combinations,
                                  Tensor<type, 4>& activations,
                                  Tensor<type, 4>& activations_derivatives) const
{

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


void Layer::hard_sigmoid_derivatives(const Tensor<type, 4>& combinations,
                                     Tensor<type, 4>& activations,
                                     Tensor<type, 4>& activations_derivatives) const
{

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


void Layer::exponential_linear_derivatives(const Tensor<type, 4>& combinations,
                                           Tensor<type, 4>& activations,
                                           Tensor<type, 4>& activations_derivatives) const
{

    const type alpha = static_cast<type>(1.0);

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

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2021 Artificial Intelligence Techniques, SL.
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
