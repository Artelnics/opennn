//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <ctype.h>

#include "layer.h"
#include "tensor_utilities.h"
#include "statistics.h"
#include "scaling.h"
#include <tuple>

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

    case Type::Scaling2D:
        return "Scaling2D";

    case Type::Scaling4D:
        return "Scaling4D";

    case Type::Unscaling:
        return "Unscaling";

    case Type::Flatten:
        return "Flatten";

    case Type::RegionProposal:
        return "RegionProposal";

    case Type::NonMaxSuppression:
        return "NonMaxSuppression";

    default:
        return "Unkown type";
    }
}


void Layer::set_threads_number(const int& new_threads_number)
{
    if(thread_pool != nullptr) delete thread_pool;
    if(thread_pool_device != nullptr) delete thread_pool_device;

    thread_pool = new ThreadPool(new_threads_number);
    thread_pool_device = new ThreadPoolDevice(thread_pool, new_threads_number);
}


void Layer::set_parameters_constant(const type&)
{
}


void Layer::set_parameters_random()
{
}


void Layer::set_parameters(const Tensor<type, 1>&, const Index&)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_parameters(const Tensor<type, 1>&) method.\n"
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


void Layer::forward_propagate(const pair<type*, dimensions>&, LayerForwardPropagation*, const bool&)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "forward_propagate(type*, const Tensor<Index, 1>&, LayerForwardPropagation*) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw invalid_argument(buffer.str());
}


void Layer::forward_propagate(const pair<type*, dimensions>&, Tensor<type, 1>&, LayerForwardPropagation*)
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


void Layer::set_inputs_number(const Index&)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_inputs_number(const Index&) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw invalid_argument(buffer.str());
}


void Layer::set_neurons_number(const Index&)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_neurons_number(const Index&) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw invalid_argument(buffer.str());
}


void Layer::linear(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{
    y.device(*thread_pool_device) = x;
}


void Layer::linear_derivatives(const Tensor<type, 1>& x, Tensor<type, 1>& y, Tensor<type, 1>& dy_dx) const
{
    y.device(*thread_pool_device) = x;

    dy_dx.setConstant(type(1));
}


void Layer::logistic(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{
    y.device(*thread_pool_device) = type(1)/(type(1) + (-x).exp());

    //    y.device(*thread_pool_device) = (type(1) + x.exp().inverse()).inverse();
}


void Layer::logistic_derivatives(const Tensor<type, 1>& x, Tensor<type, 1>& y, Tensor<type, 1>& dy_dx) const
{
    logistic(x, y);

    dy_dx.device(*thread_pool_device) = y*(type(1) - y);
}


void Layer::hard_sigmoid(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{
    //    x.cwiseMin(type(2.5)).cwiseMax(type(-2.5)).cwiseProduct(type(0.2)) + type(0.5);

    const Tensor<bool, 1> if_sentence = x < x.constant(type(-2.5));
    const Tensor<bool, 1> elif_sentence = x > x.constant(type(2.5));

    Tensor<type, 1> f1(x.dimension(0));
    Tensor<type, 1> f2(x.dimension(0));
    Tensor<type, 1> f3(x.dimension(0));

    f1.setZero();

    f2.setConstant(type(1));

    f3 = type(0.2) * x + type(0.5);

    y.device(*thread_pool_device) = if_sentence.select(f1,elif_sentence.select(f2, f3));
}


void Layer::hard_sigmoid_derivatives(const Tensor<type, 1>& x,
                                     Tensor<type, 1>& y,
                                     Tensor<type, 1>& dy_dx) const
{
    const Tensor<bool, 1> if_sentence = x < x.constant(type(-2.5));

    const Tensor<bool, 1> elif_sentence = x > x.constant(type(2.5));

    const Tensor<bool, 1> if_sentence_2 = x < x.constant(type(-2.5)) || x > x.constant(type(2.5));

    // Sentences

    Tensor<type, 1> f1(x.dimension(0));
    f1.setZero();

    Tensor<type, 1> f2(x.dimension(0));
    f2.setConstant(type(1));

    Tensor<type, 1> f3 = type(0.2) * x + type(0.5);

    Tensor<type, 1> f4(x.dimension(0));
    f4.setConstant(type(0));

    Tensor<type, 1> f5(x.dimension(0));
    f5.setConstant(type(0.2));

    y.device(*thread_pool_device) = if_sentence.select(f1, elif_sentence.select(f2, f3));

    dy_dx.device(*thread_pool_device) = if_sentence_2.select(f4, f5);
}


void Layer::hyperbolic_tangent(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{
    y.device(*thread_pool_device) = x.tanh();
}


void Layer::hyperbolic_tangent_derivatives(const Tensor<type, 1>& x, Tensor<type, 1>& y, Tensor<type, 1>& dy_dx) const
{
    y.device(*thread_pool_device) = x.tanh();

    dy_dx.device(*thread_pool_device) = type(1) - y.square();
}


void Layer::threshold(const Tensor<type, 1>& x,
                      Tensor<type, 1>& y) const
{
    const Tensor<bool, 1> if_sentence = x >= x.constant(type(0));

    Tensor<type, 1> ones(x.dimension(0));
    ones.setConstant(type(1));

    Tensor<type, 1> zeros(x.dimension(0));
    zeros.setConstant(type(0));

    y.device(*thread_pool_device) = if_sentence.select(ones, zeros);
}


void Layer::symmetric_threshold(const Tensor<type, 1>& x,
                                Tensor<type, 1>& y) const
{
    const Tensor<bool, 1> if_sentence = x > x.constant(type(0));

    Tensor<type, 1> ones(x.dimension(0));
    ones.setConstant(type(1));

    y.device(*thread_pool_device) = if_sentence.select(ones, -ones);
}


void Layer::rectified_linear(const Tensor<type, 1>& x,
                             Tensor<type, 1>& y) const
{
    const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

    Tensor<type, 1> zeros(x.dimension(0));
    zeros.setConstant(type(0));

    y.device(*thread_pool_device) = if_sentence.select(zeros, x);

}


void Layer::rectified_linear_derivatives(const Tensor<type, 1>& x,
                                         Tensor<type, 1>& y,
                                         Tensor<type, 1>& dy_dx) const
{
    const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

    Tensor<type, 1> zeros(x.dimension(0));
    zeros.setZero();

    Tensor<type, 1> ones(x.dimension(0));
    ones.setConstant(type(1));

    y.device(*thread_pool_device) = if_sentence.select(zeros, x);
    dy_dx.device(*thread_pool_device) = if_sentence.select(zeros, ones);
}


void Layer::scaled_exponential_linear(const Tensor<type, 1>& x,
                                      Tensor<type, 1>& y) const
{
    const type lambda = type(1.0507);

    const type alpha = type(1.67326);

    const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

    const Tensor<type, 1> f_1 = lambda*alpha*(x.exp()-type(1.0));

    Tensor<type, 1> f_2 = lambda*x;

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


void Layer::scaled_exponential_linear_derivatives(const Tensor<type, 1>& x,
                                                  Tensor<type, 1>& y,
                                                  Tensor<type, 1>& dy_dx) const
{
    const type lambda = type(1.0507);

    const type alpha = type(1.67326);

    const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

    Tensor<type, 1> f_1 = lambda*alpha*(x.exp()-type(1.0));

    Tensor<type, 1> f_2 = lambda*x;

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

    f_1 = lambda*alpha*x.exp();

    f_2 = x.constant(type(1))*lambda;

    dy_dx.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


void Layer::soft_plus(const Tensor<type, 1>& x,
                      Tensor<type, 1>& y) const
{
    y.device(*thread_pool_device) = (x.constant(type(1)) + x.exp()).log();
}


void Layer::soft_plus_derivatives(const Tensor<type, 1>& x,
                                  Tensor<type, 1>& y,
                                  Tensor<type, 1>& dy_dx) const
{
    y.device(*thread_pool_device) = (x.constant(type(1)) + x.exp()).log();

    dy_dx.device(*thread_pool_device) = type(1.0) / (type(1.0) + x.exp().inverse());
}


void Layer::soft_sign(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{
    const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

    const Tensor<type, 1> f_1 = x / (type(1) - x);

    const Tensor<type, 1> f_2 = x / (type(1) + x);

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


void Layer::soft_sign_derivatives(const Tensor<type, 1>& x,
                                  Tensor<type, 1>& y,
                                  Tensor<type, 1>& dy_dx) const
{
    const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

    Tensor<type, 1> f_1 = x / (type(1) - x);

    Tensor<type, 1> f_2 = x / (type(1) + x);

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

    // Activations Derivatives

    f_1 = type(1.0) / (type(1.0) - x).pow(type(2));

    f_2 = type(1.0) / (type(1.0) + x).pow(type(2));

    dy_dx.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


void Layer::exponential_linear(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{
    const type alpha = type(1.0);

    y.device(*thread_pool_device) = x;
    /*
    y.device(*thread_pool_device) = y.select(y < 0, alpha * (y.exp() - type(1)));


    const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

    Tensor<type, 1> f_1(x.dimension(0));

    Tensor<type, 1> f_2(x.dimension(0));

    f_1.device(*thread_pool_device) = alpha*(x.exp() - type(1));

    f_2 = x;

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    */
}


void Layer::exponential_linear_derivatives(const Tensor<type, 1>& x,
                                           Tensor<type, 1>& y,
                                           Tensor<type, 1>& dy_dx) const
{
    const type alpha = type(1.0);

    const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

    Tensor<type, 1> f_1(x.dimension(0));
    f_1 = alpha*(x.exp() - type(1));

    Tensor<type, 1> f_2(x.dimension(0));
    f_2 = x;

    // Activations

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

    // Activations Derivatives

    f_1 = alpha * x.exp();

    f_2 = x.constant(type(1));

    dy_dx.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


void Layer::binary(const Tensor<type, 1>& x, Tensor<type, 1>& y) const
{
    const Tensor<bool, 1> if_sentence = x < x.constant(type(0.5));

    const Tensor<type, 1> f_1 = x.constant(type(false));

    const Tensor<type, 1> f_2 = x.constant(type(true));

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


void Layer::exponential_linear(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    const type alpha = type(1.0);

    const Tensor<bool, 2> if_sentence = x < x.constant(type(0));

    Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

    Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

    f_1 = alpha*(x.exp() - type(1));

    f_2 = x;

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


void Layer::hard_sigmoid(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    const Index n = x.size();

#pragma omp parallel for

    for(Index i = 0; i < n; i++)
    {
        if(x(i) < type(-2.5))
        {
            y(i) = type(0);
        }
        else if(x(i) > type(2.5))
        {
            y(i) = type(1);
        }
        else
        {
            y(i) = type(0.2) * x(i) + type(0.5);
        }
    }
}


void Layer::hyperbolic_tangent(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    y.device(*thread_pool_device) = x.tanh();
}


void Layer::linear(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    y.device(*thread_pool_device) = x;
}


void Layer::logistic(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{

}


void Layer::rectified_linear(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    const Tensor<bool, 2> if_sentence = x < x.constant(type(0));

    Tensor<type, 2> zeros(x.dimension(0), x.dimension(1));
    zeros.setConstant(type(0));

    y.device(*thread_pool_device) = if_sentence.select(zeros, x);
}


void Layer::leaky_rectified_linear(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{

}


void Layer::scaled_exponential_linear(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    /*
    const Tensor<bool, 2> if_sentence = x < x.constant(type(0));

    Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

    Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

    f_1 = lambda*alpha*(x.exp() - type(1.0));

    f_2 = lambda*x;

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
*/
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

    f_1 = x / (type(1) - x);

    f_2 = x / (type(1) + x);

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


void Layer::symmetric_threshold(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    const Tensor<bool, 2> if_sentence = x > x.constant(type(0));

    Tensor<type, 2> ones(x.dimension(0), x.dimension(1));

    ones.setConstant(type(1));

    y.device(*thread_pool_device) = if_sentence.select(ones, -ones);
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


void Layer::exponential_linear_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y, Tensor<type, 2>& dy_dx) const
{
    /*
    const Tensor<bool, 2> if_sentence = x < x.constant(type(0));

    Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

    Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

    f_1 = alpha*(x.exp() - type(1));

    f_2 = x;

    // Activations

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

    // Activations Derivatives

    f_1 = alpha * x.exp();

    f_2 = x.constant(type(1));

    dy_dx.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
*/
}


void Layer::hard_sigmoid_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y, Tensor<type, 2>& dy_dx) const
{
    const Tensor<bool, 2> if_sentence = x < x.constant(type(-2.5));

    const Tensor<bool, 2> elif_sentence = x > x.constant(type(2.5));

    const Tensor<bool, 2> if_sentence_2 = x < x.constant(type(-2.5)) || x > x.constant(type(2.5));

    // Sentences

    Tensor<type, 2> f1(x.dimension(0), x.dimension(1));
    f1.setZero();

    Tensor<type, 2> f2(x.dimension(0), x.dimension(1));
    f2.setConstant(type(1));

    Tensor<type, 2> f3(x.dimension(0), x.dimension(1));
    f3 = type(0.2) * x + type(0.5);

    Tensor<type, 2> f4(x.dimension(0), x.dimension(1));
    f4.setConstant(type(0));

    Tensor<type, 2> f5(x.dimension(0), x.dimension(1));
    f5.setConstant(type(0.2));

    // Activations

    y.device(*thread_pool_device) = if_sentence.select(f1, elif_sentence.select(f2, f3));

    // Activations Derivatives

    dy_dx.device(*thread_pool_device) = if_sentence_2.select(f4, f5);
}


void Layer::hyperbolic_tangent_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y, Tensor<type, 2>& dy_dx) const
{
    y.device(*thread_pool_device) = x.tanh();

    dy_dx.device(*thread_pool_device) = type(1) - y.square();
}


void Layer::linear_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y, Tensor<type, 2>& dy_dx) const
{
    y.device(*thread_pool_device) = x;

    dy_dx.setConstant(type(1));
}


void Layer::logistic_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y, Tensor<type, 2>& dy_dx) const
{
    logistic(x, y);

    dy_dx.device(*thread_pool_device) = y*(type(1) - y);
}


void Layer::rectified_linear_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y, Tensor<type, 2>& dy_dx) const
{
    const Tensor<bool, 2> if_sentence = x < x.constant(type(0));

    Tensor<type, 2> zeros(x.dimension(0), x.dimension(1));
    zeros.setZero();

    Tensor<type, 2> ones(x.dimension(0), x.dimension(1));
    ones.setConstant(type(1));

    y.device(*thread_pool_device) = if_sentence.select(zeros, x);
    dy_dx.device(*thread_pool_device) = if_sentence.select(zeros, ones);
}


void Layer::leaky_rectified_linear_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y, Tensor<type, 2>& dy_dx) const
{

}


void Layer::scaled_exponential_linear_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y, Tensor<type, 2>& dy_dx) const
{
    /*
    const Tensor<bool, 2> if_sentence = x < x.constant(type(0));

    Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

    Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

    f_1 = lambda*alpha*(x.exp()-type(1.0));

    f_2 = lambda*x;

    // Activations

    activations.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

    // Activations Derivatives

    f_1 = lambda*alpha*x.exp();

    f_2 = x.constant(type(1))*lambda;

    activations_derivatives.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
*/
}


void Layer::soft_plus_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y, Tensor<type, 2>& dy_dx) const
{
    y.device(*thread_pool_device)
        = (x.constant(type(1)) + x.exp()).log();

    dy_dx.device(*thread_pool_device)
        = type(1.0) / (type(1.0) + x.exp().inverse());
}


void Layer::soft_sign_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y, Tensor<type, 2>& dy_dx) const
{
    /*
    const Tensor<bool, 2> if_sentence = combinations < combinations.constant(type(0));

    Tensor<type, 2> f_1(combinations.dimension(0), combinations.dimension(1));

    Tensor<type, 2> f_2(combinations.dimension(0), combinations.dimension(1));

    f_1 = combinations / (type(1) - combinations);

    f_2 = combinations / (type(1) + combinations);

    // Activations

    activations.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

    // Activations Derivatives

    f_1 = type(1.0) / (type(1.0) - combinations).pow(type(2));

    f_2 = type(1.0) / (type(1.0) + combinations).pow(type(2));

    activations_derivatives.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
*/
}


void Layer::hyperbolic_tangent(const Tensor<type, 3>& x, Tensor<type, 3>& y) const
{
    y.device(*thread_pool_device) = x.tanh();
}


void Layer::linear(const Tensor<type, 3>& x, Tensor<type, 3>& y) const
{
    y.device(*thread_pool_device) = x;
}

void Layer::rectified_linear(const Tensor<type, 3>& x, Tensor<type, 3>& y) const
{
    const Tensor<bool, 3> if_sentence = x < x.constant(type(0));

    Tensor<type, 3> zeros(x.dimension(0), x.dimension(1), x.dimension(2));
    zeros.setConstant(type(0));

    y.device(*thread_pool_device) = if_sentence.select(zeros, x);
}


void Layer::hyperbolic_tangent_derivatives(const Tensor<type, 3>& x, Tensor<type, 3>& y, Tensor<type, 3>& dy_dx) const
{
    y.device(*thread_pool_device) = x.tanh();

    dy_dx.device(*thread_pool_device) = type(1) - y.square();
}


void Layer::linear_derivatives(const Tensor<type, 3>& x, Tensor<type, 3>& y, Tensor<type, 3>& dy_dx) const
{
    y.device(*thread_pool_device) = x;

    dy_dx.setConstant(type(1));
}


void Layer::rectified_linear_derivatives(const Tensor<type, 3>& x, Tensor<type, 3>& y, Tensor<type, 3>& dy_dx) const
{
    const Tensor<bool, 3> if_sentence = x < x.constant(type(0));

    Tensor<type, 3> zeros(x.dimension(0), x.dimension(1), x.dimension(2));
    zeros.setZero();

    Tensor<type, 3> ones(x.dimension(0), x.dimension(1), x.dimension(2));
    ones.setConstant(type(1));

    y.device(*thread_pool_device) = if_sentence.select(zeros, x);
    dy_dx.device(*thread_pool_device) = if_sentence.select(zeros, ones);
}


void Layer::binary(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{

}


void Layer::exponential_linear(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{
    /*
    const Tensor<bool, 4> if_sentence = x < x.constant(type(0));

    Tensor<type, 4> f_1(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

    Tensor<type, 4> f_2(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

    f_1 = alpha*(x.exp() - type(1));

    f_2 = x;

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
*/
}


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
    f_middle = type(0.2) * x + type(0.5);
    f_equal = x;

    y.device(*thread_pool_device) = if_lower.select(f_lower, f_equal);
    y.device(*thread_pool_device) = if_greater.select(f_greater, f_equal);
    y.device(*thread_pool_device) = if_middle.select(f_middle, f_equal);

}


void Layer::hyperbolic_tangent(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{
    y.device(*thread_pool_device) = x.tanh();
}


void Layer::linear(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{
    y.device(*thread_pool_device) = x;
}


void Layer::logistic(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{

}


void Layer::rectified_linear(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{
    y.setZero();

    y.device(*thread_pool_device) = y.cwiseMax(x);
}


void Layer::leaky_rectified_linear(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{

}


void Layer::scaled_exponential_linear(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{
    /*
    const Tensor<bool, 4> if_sentence = x < x.constant(type(0));

    Tensor<type, 4> f_1(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

    Tensor<type, 4> f_2(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

    f_1 = lambda*alpha*(x.exp() - type(1.0));

    f_2 = lambda*x;

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
*/
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

    f_1 = x / (type(1) - x);

    f_2 = x / (type(1) + x);

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}


void Layer::symmetric_threshold(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
{
    const Tensor<bool, 4> if_sentence = x > x.constant(type(0));

    Tensor<type, 4> ones(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

    ones.setConstant(type(1));

    y.device(*thread_pool_device) = if_sentence.select(ones, -ones);
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


void Layer::exponential_linear_derivatives(const Tensor<type, 4>& x, Tensor<type, 4>& y, Tensor<type, 4>& dy_dx) const
{
    /*
    const Tensor<bool, 4> if_sentence = combinations < combinations.constant(type(0));

    Tensor<type, 4> f_1(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));

    Tensor<type, 4> f_2(combinations.dimension(0), combinations.dimension(1), combinations.dimension(2), combinations.dimension(3));

    f_1 = alpha*(combinations.exp() - type(1));

    f_2 = combinations;

    // Activations

    activations.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

    // Activations Derivatives

    f_1 = alpha * combinations.exp();

    f_2 = combinations.constant(type(1));

    activations_derivatives.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
*/
}


void Layer::hard_sigmoid_derivatives(const Tensor<type, 4>& x, Tensor<type, 4>& y, Tensor<type, 4>& dy_dx) const
{
    // Conditions

    const Tensor<bool, 4> if_sentence = x < x.constant(type(-2.5));
    const Tensor<bool, 4> elif_sentence = x > x.constant(type(2.5));
    const Tensor<bool, 4> if_sentence_2 = x < x.constant(type(-2.5)) || x > x.constant(type(2.5));

    // Sentences

    Tensor<type, 4> f1(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));
    f1.setZero();

    Tensor<type, 4> f2(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));
    f2.setConstant(type(1));

    Tensor<type, 4> f3(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));
    f3 = type(0.2) * x + type(0.5);

    Tensor<type, 4> f4(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));
    f4.setConstant(type(0));

    Tensor<type, 4> f5(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));
    f5.setConstant(type(0.2));

    // Activations

    y.device(*thread_pool_device) = if_sentence.select(f1, elif_sentence.select(f2, f3));

    // Activations derivatives

    dy_dx.device(*thread_pool_device) = if_sentence_2.select(f4, f5);

}


void Layer::hyperbolic_tangent_derivatives(const Tensor<type, 4>& x, Tensor<type, 4>& y, Tensor<type, 4>& dy_dx) const
{
    y.device(*thread_pool_device) = x.tanh();
}


void Layer::linear_derivatives(const Tensor<type, 4>& x, Tensor<type, 4>& y, Tensor<type, 4>& dy_dx) const
{
    y.device(*thread_pool_device) = x;

    dy_dx.setConstant(type(1));
}


void Layer::logistic_derivatives(const Tensor<type, 4>& x, Tensor<type, 4>& y, Tensor<type, 4>& dy_dx) const
{
    logistic(x, y);

    dy_dx.device(*thread_pool_device) = y*(type(1) - y);
}


void Layer::rectified_linear_derivatives(const Tensor<type, 4>& x, Tensor<type, 4>& y, Tensor<type, 4>& dy_dx) const
{
    const Tensor<bool, 4> if_sentence = x < x.constant(type(0));

    Tensor<type, 4> zeros(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));
    zeros.setZero();

    Tensor<type, 4> ones(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));
    ones.setConstant(type(1));

    y.device(*thread_pool_device) = if_sentence.select(zeros, x);
    dy_dx.device(*thread_pool_device) = if_sentence.select(zeros, ones);
}


void Layer::leaky_rectified_linear_derivatives(const Tensor<type, 4>& x, Tensor<type, 4>& y, Tensor<type, 4>& dy_dx) const
{

}


void Layer::scaled_exponential_linear_derivatives(const Tensor<type, 4>& x, Tensor<type, 4>& y, Tensor<type, 4>& dy_dx) const
{
    /*
    const Tensor<bool, 4> if_sentence = x < x.constant(type(0));

    Tensor<type, 4> f_1(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

    Tensor<type, 4> f_2(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

    f_1 = lambda*alpha*(x.exp()-type(1.0));

    f_2 = lambda*x;

    // Activations

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

    // Activations Derivatives

    f_1 = lambda*alpha*x.exp();

    f_2 = x.constant(type(1))*lambda;

    dy_dx.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
*/
}


void Layer::soft_plus_derivatives(const Tensor<type, 4>& x, Tensor<type, 4>& y, Tensor<type, 4>& dy_dx) const
{
    y.device(*thread_pool_device) = (x.constant(type(1)) + x.exp()).log();

    dy_dx.device(*thread_pool_device) = type(1.0) / (type(1.0) + x.exp().inverse());
}


void Layer::soft_sign_derivatives(const Tensor<type, 4>& x, Tensor<type, 4>& y, Tensor<type, 4>& dy_dx) const
{
    const Tensor<bool, 4> if_sentence = x < x.constant(type(0));

    Tensor<type, 4> f_1(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

    Tensor<type, 4> f_2(x.dimension(0), x.dimension(1), x.dimension(2), x.dimension(3));

    f_1 = x / (type(1) - x);

    f_2 = x / (type(1) + x);

    // Activations

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

    // Activations Derivatives

    f_1 = type(1.0) / (type(1.0) - x).pow(type(2));

    f_2 = type(1.0) / (type(1.0) + x).pow(type(2));

    dy_dx.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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
