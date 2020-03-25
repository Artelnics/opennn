//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "layer.h"

#include "statistics.h"

namespace OpenNN
{

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
        return "PrincipalComponents";

    case Convolutional:
        return "Convolutional";

    case Perceptron:
        return "Perceptron";

    case Bounding:
        return "Bounding";

    case Pooling:
        return "Pooling";

    case Probabilistic:
        return "Probabilistic";

    case LongShortTermMemory:
        return "LongShortTermMemory";

    case Recurrent:
        return "Recurrent";

    case Scaling:
        return "Scaling";

    case Unscaling:
        return "Unscaling";
    }

    return string();
}

void Layer::set_device_pointer(Device* new_device_pointer)
{
    device_pointer = new_device_pointer;
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

/*
Layer::ForwardPropagation Layer::forward_propagate(const Tensor<type, 2>&)
 {
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "forward_propagate(const Tensor<type, 2>&) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
 }
*/

Tensor<Index, 1> Layer::get_input_variables_dimensions() const
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "get_input_variables_dimensions() const method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


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


void Layer::set_inputs_number(const Index &)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_inputs_number(const Index &) method.\n"
           << "This method is not implemented in the layer type (" << get_type_string() << ").\n";

    throw logic_error(buffer.str());
}


void Layer::set_neurons_number(const Index &)
{
    ostringstream buffer;

    buffer << "OpenNN Exception: Layer class.\n"
           << "set_neurons_number(const Index &) method.\n"
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


void Layer::hard_sigmoid(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    const Index n = x.size();

    #pragma omp parallel for

    for(Index i = 0; i < n; i++)
    {
        if(x(i) < static_cast<type>(-2.5))
        {
            y(i) = 0;
        }
        else if(x(i) > static_cast<type>(2.5))
        {
            y(i) = 1;
        }
        else
        {
            y(i) = static_cast<type>(0.2) * x(i) + static_cast<type>(0.5);
        }
    }
}


void Layer::hyperbolic_tangent(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        y.device(*default_device) = x.tanh();

        return;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        y.device(*thread_pool_device) = x.tanh();

        return;
    }

    case Device::EigenGpu:
    {
        return;
    }
    }
}


void Layer::logistic(const Tensor<type, 2>& x, Tensor<type, 2>& y)const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        y.device(*default_device) = (1 + x.exp().inverse()).inverse();

        return;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        y.device(*thread_pool_device) = (1 + x.exp().inverse()).inverse();

        return;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        return;
    }
    }
}


void Layer::linear(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    y = x;
}


void Layer::threshold(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        const Tensor<bool, 2> if_sentence = x > x.constant(0);

        Tensor<type, 2> ones(x.dimension(0), x.dimension(1));
        ones.setConstant(1);

        Tensor<type, 2> zeros(x.dimension(0), x.dimension(1));
        zeros.setConstant(0);

        y.device(*default_device) = if_sentence.select(ones, zeros);

        break;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        const Tensor<bool, 2> if_sentence = x >= x.constant(0);

        Tensor<type, 2> ones(x.dimension(0), x.dimension(1));
        ones.setConstant(1);

        Tensor<type, 2> zeros(x.dimension(0), x.dimension(1));
        zeros.setConstant(0);

        y.device(*thread_pool_device) = if_sentence.select(ones, zeros);

        break;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        break;
    }
    }
}


void Layer::symmetric_threshold(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        const Tensor<bool, 2> if_sentence = x > x.constant(0);

        Tensor<type, 2> ones(x.dimension(0), x.dimension(1));

        ones.setConstant(1);

        y.device(*default_device) = if_sentence.select(ones, -ones);

        break;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        const Tensor<bool, 2> if_sentence = x > x.constant(0);

        Tensor<type, 2> ones(x.dimension(0), x.dimension(1));

        ones.setConstant(1);

        y.device(*thread_pool_device) = if_sentence.select(ones, -ones);

        break;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        break;
    }
    }
}


void Layer::rectified_linear(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        const Tensor<bool, 2> if_sentence = x < x.constant(0);

        Tensor<type, 2> zeros(x.dimension(0), x.dimension(1));

        zeros.setConstant(0);

        y.device(*default_device) = if_sentence.select(zeros, x);

        break;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        const Tensor<bool, 2> if_sentence = x < x.constant(0);

        Tensor<type, 2> zeros(x.dimension(0), x.dimension(1));

        zeros.setConstant(0);

        y.device(*thread_pool_device) = if_sentence.select(zeros, x);

        break;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        break;
    }
    }
}


void Layer::scaled_exponential_linear(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        const type lambda = static_cast<type>(1.0507);

        const type alpha = static_cast<type>(1.67326);

        const Tensor<bool, 2> if_sentence = x < x.constant(0);

        Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

        Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

        f_1 = lambda*alpha*(x.exp()-static_cast<type>(1.0));

        f_2 = lambda*x;

        y.device(*default_device) = if_sentence.select(f_1, f_2);

        break;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        const type lambda = static_cast<type>(1.0507);

        const type alpha = static_cast<type>(1.67326);

        const Tensor<bool, 2> if_sentence = x < x.constant(0);

        Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

        Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

        f_1 = lambda*alpha*(x.exp() - static_cast<type>(1.0));

        f_2 = lambda*x;

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        break;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


        break;
    }
    }
}


void Layer::soft_plus(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        y.device(*default_device) = (x.constant(1) + x.exp()).log();

        break;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        y.device(*thread_pool_device) = (x.constant(1) + x.exp()).log();

        break;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        break;
    }
    }
}


void Layer::soft_sign(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        const Tensor<bool, 2> if_sentence = x < x.constant(0);

        Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

        Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

        f_1 = x / (static_cast<type>(1) - x);

        f_2 = x / (static_cast<type>(1) + x);

        y.device(*default_device) = if_sentence.select(f_1, f_2);

        break;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        const Tensor<bool, 2> if_sentence = x < x.constant(0);

        Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

        Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

        f_1 = x / (static_cast<type>(1) - x);

        f_2 = x / (static_cast<type>(1) + x);

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        break;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        break;
    }
    }
}


void Layer::exponential_linear(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        const Tensor<bool, 2> if_sentence = x < x.constant(0);

        const type alpha = static_cast<type>(1.0);

        Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

        Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

        f_1 = alpha*(x.exp() - static_cast<type>(1));

        f_2 = x;

        y.device(*default_device) = if_sentence.select(f_1, f_2);

        break;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        const Tensor<bool, 2> if_sentence = x < x.constant(0);

        const type alpha = static_cast<type>(1.0);

        Tensor<type, 2> f_1(x.dimension(0), x.dimension(1));

        Tensor<type, 2> f_2(x.dimension(0), x.dimension(1));

        f_1 = alpha*(x.exp() - static_cast<type>(1));

        f_2 = x;

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        break;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        break;
    }
    }
}

/// @todo Ternary operator

void Layer::binary(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
//        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        break;
    }

    case Device::EigenSimpleThreadPool:
    {
//        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        break;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        break;
    }
    }

    const Index n = x.size();

    #pragma omp parallel for

    for(Index i = 0; i < n; i++)
    {
        x(i) < static_cast<type>(0.5) ? y(i) = false : y (i) = true;
    }
}


/// @todo exception with several maximum indices

void Layer::competitive(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        const Index rows_number = x.dimension(0);

        y.setZero();

        for(Index i = 0; i < rows_number; i++)
        {
        Index maximal_index_ = maximal_index(x.chip(i,0));
        y(i,maximal_index_) = 1;
        }

        y.device(*default_device) = y;

        break;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        const Index rows_number = x.dimension(0);

        y.setZero();

        for(Index i = 0; i < rows_number; i++)
        {
        Index maximal_index_ = maximal_index(x.chip(i,0));
        y(i,maximal_index_) = 1;
        }

        y.device(*thread_pool_device) = y;

        break;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        break;
    }
    }


}


void Layer::softmax(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    Tensor<type, 0> sum;

    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        sum.device(*default_device) = x.exp().sum();

        y.device(*default_device) = x.exp() / sum(0);

        break;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        sum.device(*thread_pool_device) = x.exp().sum();

        y.device(*thread_pool_device) = x.exp() / sum(0);

        break;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        break;
    }

    }
}


void Layer::hard_sigmoid_derivatives(const Tensor<type, 2>& combinations,
                                     Tensor<type, 2>& activations,
                                     Tensor<type, 2>& activations_derivatives) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        // Activations

        const Tensor<bool, 2> if_sentence = combinations < combinations.constant(-2.5);

        Tensor<type, 2> f1(combinations.dimension(0), combinations.dimension(1));
        f1.setZero();

        const Tensor<bool, 2> elif_sentence = combinations > combinations.constant(2.5);

        Tensor<type, 2> f2(combinations.dimension(0), combinations.dimension(1));
        f2.setConstant(1);

        Tensor<type, 2> f3(combinations.dimension(0), combinations.dimension(1));
        f3 = static_cast<type>(0.2) * combinations + static_cast<type>(0.5);

        activations.device(*default_device) = if_sentence.select(f1, elif_sentence.select(f2, f3));

        // Activations Derivatives

        const Tensor<bool, 2> if_sentence_2 = combinations < combinations.constant(-2.5) || combinations > combinations.constant(2.5);

        Tensor<type, 2> f4(combinations.dimension(0), combinations.dimension(1));
        f4.setConstant(0.0);

        Tensor<type, 2> f5(combinations.dimension(0), combinations.dimension(1));
        f5.setConstant(static_cast<type>(0.2));

        activations_derivatives.device(*default_device) = if_sentence_2.select(f4, f5);

        return;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        // Activations

        const Tensor<bool, 2> if_sentence = combinations < combinations.constant(-2.5);

        Tensor<type, 2> f1(combinations.dimension(0), combinations.dimension(1));
        f1.setZero();

        const Tensor<bool, 2> elif_sentence = combinations > combinations.constant(2.5);

        Tensor<type, 2> f2(combinations.dimension(0), combinations.dimension(1));
        f2.setConstant(1);

        Tensor<type, 2> f3(combinations.dimension(0), combinations.dimension(1));
        f3 = static_cast<type>(0.2) * combinations + static_cast<type>(0.5);

        activations.device(*thread_pool_device) = if_sentence.select(f1, elif_sentence.select(f2, f3));

        // Activations Derivatives

        const Tensor<bool, 2> if_sentence_2 = combinations < combinations.constant(-2.5) || combinations > combinations.constant(2.5);

        Tensor<type, 2> f4(combinations.dimension(0), combinations.dimension(1));
        f4.setConstant(0.0);

        Tensor<type, 2> f5(combinations.dimension(0), combinations.dimension(1));
        f5.setConstant(static_cast<type>(0.2));

        activations_derivatives.device(*thread_pool_device) = if_sentence_2.select(f4, f5);

        return;
    }

    case Device::EigenGpu:
    {
        return;
    }
    }


}

void Layer::hyperbolic_tangent_derivatives(const Tensor<type, 2>& combinations,
                                           Tensor<type, 2>& activations,
                                           Tensor<type, 2>& activations_derivatives) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        activations.device(*default_device) = combinations.tanh();

        activations_derivatives.device(*default_device) = 1 - activations.square();

        return;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        activations.device(*thread_pool_device) = combinations.tanh();

        activations_derivatives.setConstant(1);

        //        activations_derivatives.device(*thread_pool_device) = 1 - activations.square();
        activations_derivatives.device(*thread_pool_device) -= activations.square();

        return;
    }

    case Device::EigenGpu:
    {
        return;
    }
    }

}


void Layer::logistic_derivatives(const Tensor<type, 2>& combinations,
                                 Tensor<type, 2>& activations,
                                 Tensor<type, 2>& activations_derivatives) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        // Activations

        activations.device(*default_device) = (1 + combinations.exp().inverse()).inverse();

        // Activations Derivatives

        activations_derivatives.device(*default_device) = combinations.exp().inverse() / (static_cast<type>(1.0) + combinations.exp().inverse()).pow(2);

        return;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        // Activations

        activations.device(*thread_pool_device) = (1 + combinations.exp().inverse()).inverse();

        // Activations Derivatives

        activations_derivatives.device(*thread_pool_device) = combinations.exp().inverse() / (static_cast<type>(1.0) + combinations.exp().inverse()).pow(2);

        return;
    }

    case Device::EigenGpu:
    {
//      GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        return;
    }
    }

}


void Layer::linear_derivatives(const Tensor<type, 2>& combinations,
                               Tensor<type, 2>& activations,
                               Tensor<type, 2>& activations_derivatives) const
{
    activations = combinations;

    activations_derivatives.setConstant(1.0);
}



void Layer::threshold_derivatives(const Tensor<type, 2>& combinations,
                                  Tensor<type, 2>& activations,
                                  Tensor<type, 2>& activations_derivatives) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        // Activations

        const Tensor<bool, 2> if_sentence = combinations > combinations.constant(0);

        Tensor<type, 2> ones(combinations.dimension(0), combinations.dimension(1));
        ones.setConstant(1);

        Tensor<type, 2> zeros(combinations.dimension(0), combinations.dimension(1));
        zeros.setConstant(0);

        activations.device(*default_device) = if_sentence.select(ones, zeros);

        // Activations Derivatives

        activations_derivatives.setZero();

        break;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        // Activations

        const Tensor<bool, 2> if_sentence = combinations > combinations.constant(0);

        Tensor<type, 2> ones(combinations.dimension(0), combinations.dimension(1));
        ones.setConstant(1);

        Tensor<type, 2> zeros(combinations.dimension(0), combinations.dimension(1));
        zeros.setConstant(0);

        activations.device(*thread_pool_device) = if_sentence.select(ones, zeros);

        // Activations Derivatives

        activations_derivatives.setZero();


        break;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        break;
    }
    }

}


void Layer::symmetric_threshold_derivatives(const Tensor<type, 2>& combinations,
                                            Tensor<type, 2>& activations,
                                            Tensor<type, 2>& activations_derivatives) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        // Activations

        const Tensor<bool, 2> if_sentence = combinations > combinations.constant(0);

        Tensor<type, 2> ones(combinations.dimension(0), combinations.dimension(1));

        ones.setConstant(1);

        activations.device(*default_device) = if_sentence.select(ones, -ones);

        // Activations Derivatives

        activations_derivatives.setZero();

        break;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        // Activations

        const Tensor<bool, 2> if_sentence = combinations > combinations.constant(0);

        Tensor<type, 2> ones(combinations.dimension(0), combinations.dimension(1));

        ones.setConstant(1);

        activations.device(*thread_pool_device) = if_sentence.select(ones, -ones);

        // Activations Derivatives

        activations_derivatives.setZero();

        break;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        break;
    }
    }

}


void Layer::rectified_linear_derivatives(const Tensor<type, 2>& combinations,
                                         Tensor<type, 2>& activations,
                                         Tensor<type, 2>& activations_derivatives) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        // Activations

        const Tensor<bool, 2> if_sentence = combinations < combinations.constant(0);

        Tensor<type, 2> zeros(combinations.dimension(0), combinations.dimension(1));
        zeros.setZero();

        Tensor<type, 2> ones(combinations.dimension(0), combinations.dimension(1));
        ones.setConstant(1.);

        activations.device(*default_device) = if_sentence.select(zeros, combinations);

        // Activations Derivatives

        activations_derivatives.device(*default_device) = if_sentence.select(zeros, ones);

        break;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        // Activations

        const Tensor<bool, 2> if_sentence = combinations < combinations.constant(0);

        Tensor<type, 2> zeros(combinations.dimension(0), combinations.dimension(1));
        zeros.setZero();

        Tensor<type, 2> ones(combinations.dimension(0), combinations.dimension(1));
        ones.setConstant(1.);

        activations.device(*thread_pool_device) = if_sentence.select(zeros, combinations);

        // Activations Derivatives

        activations_derivatives.device(*thread_pool_device) = if_sentence.select(zeros, ones);

        break;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        break;
    }
    }

}


void Layer::scaled_exponential_linear_derivatives(const Tensor<type, 2>& combinations,
                                                  Tensor<type, 2>& activations,
                                                  Tensor<type, 2>& activations_derivatives) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        // Activations

        const type lambda = static_cast<type>(1.0507);

        const type alpha = static_cast<type>(1.67326);

        const Tensor<bool, 2> if_sentence = combinations < combinations.constant(0);

        Tensor<type, 2> f_1(combinations.dimension(0), combinations.dimension(1));

        Tensor<type, 2> f_2(combinations.dimension(0), combinations.dimension(1));

        f_1 = lambda*alpha*(combinations.exp()-static_cast<type>(1.0));

        f_2 = lambda*combinations;

        activations.device(*default_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = lambda*alpha*combinations.exp();

        f_2 = combinations.constant(1)*lambda;

        activations_derivatives.device(*default_device) = if_sentence.select(f_1, f_2);

        break;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        // Activations

        const type lambda = static_cast<type>(1.0507);

        const type alpha = static_cast<type>(1.67326);

        const Tensor<bool, 2> if_sentence = combinations < combinations.constant(0);

        Tensor<type, 2> f_1(combinations.dimension(0), combinations.dimension(1));

        Tensor<type, 2> f_2(combinations.dimension(0), combinations.dimension(1));

        f_1 = lambda*alpha*(combinations.exp()-static_cast<type>(1.0));

        f_2 = lambda*combinations;

        activations.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = lambda*alpha*combinations.exp();

        f_2 = combinations.constant(1)*lambda;

        activations_derivatives.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        break;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


        break;
    }
    }

}


void Layer::soft_plus_derivatives(const Tensor<type, 2>& combinations,
                                  Tensor<type, 2>& activations,
                                  Tensor<type, 2>& activations_derivatives) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        // Activations

        activations.device(*default_device) = (combinations.constant(1) + combinations.exp()).log();

        // Activations Derivatives

        activations_derivatives.device(*default_device) = static_cast<type>(1.0) / (static_cast<type>(1.0) + combinations.exp().inverse());

        break;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        // Activations

        activations.device(*thread_pool_device) = (combinations.constant(1) + combinations.exp()).log();

        // Activations Derivatives

        activations_derivatives.device(*thread_pool_device) = static_cast<type>(1.0) / (static_cast<type>(1.0) + combinations.exp().inverse());

        break;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        break;
    }
    }

}


void Layer::soft_sign_derivatives(const Tensor<type, 2>& combinations,
                                  Tensor<type, 2>& activations,
                                  Tensor<type, 2>& activations_derivatives) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        // Activations

        const Tensor<bool, 2> if_sentence = combinations < combinations.constant(0);

        Tensor<type, 2> f_1(combinations.dimension(0), combinations.dimension(1));

        Tensor<type, 2> f_2(combinations.dimension(0), combinations.dimension(1));

        f_1 = combinations / (static_cast<type>(1) - combinations);

        f_2 = combinations / (static_cast<type>(1) + combinations);

        activations.device(*default_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = static_cast<type>(1.0) / (static_cast<type>(1.0) - combinations).pow(2);

        f_2 = static_cast<type>(1.0) / (static_cast<type>(1.0) + combinations).pow(2);

        activations_derivatives.device(*default_device) = if_sentence.select(f_1, f_2);

        break;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();
        // Activations

        const Tensor<bool, 2> if_sentence = combinations < combinations.constant(0);

        Tensor<type, 2> f_1(combinations.dimension(0), combinations.dimension(1));

        Tensor<type, 2> f_2(combinations.dimension(0), combinations.dimension(1));

        f_1 = combinations / (static_cast<type>(1) - combinations);

        f_2 = combinations / (static_cast<type>(1) + combinations);

        activations.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = static_cast<type>(1.0) / (static_cast<type>(1.0) - combinations).pow(2);

        f_2 = static_cast<type>(1.0) / (static_cast<type>(1.0) + combinations).pow(2);

        activations_derivatives.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        break;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        break;
    }
    }

}


void Layer::exponential_linear_derivatives(const Tensor<type, 2>& combinations,
                                           Tensor<type, 2>& activations,
                                           Tensor<type, 2>& activations_derivatives) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        // Activations

        const Tensor<bool, 2> if_sentence = combinations <= combinations.constant(0);

        const type alpha = static_cast<type>(1.0);

        Tensor<type, 2> f_1(combinations.dimension(0), combinations.dimension(1));

        Tensor<type, 2> f_2(combinations.dimension(0), combinations.dimension(1));

        f_1 = alpha*(combinations.exp() - static_cast<type>(1));

        f_2 = combinations;

        activations.device(*default_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = alpha * combinations.exp();

        f_2 = combinations.constant(1.);

        activations_derivatives.device(*default_device) = if_sentence.select(f_1, f_2);

        break;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        // Activations

        const Tensor<bool, 2> if_sentence = combinations < combinations.constant(0);

        const type alpha = static_cast<type>(1.0);

        Tensor<type, 2> f_1(combinations.dimension(0), combinations.dimension(1));

        Tensor<type, 2> f_2(combinations.dimension(0), combinations.dimension(1));

        f_1 = alpha*(combinations.exp() - static_cast<type>(1));

        f_2 = combinations;

        activations.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = alpha * combinations.exp();

        f_2 = combinations.constant(1.);

        activations_derivatives.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        break;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        break;
    }
    }

}



void Layer::logistic_derivatives(const Tensor<type, 2>& combinations,
                                 Tensor<type, 2>& activations,
                                 Tensor<type, 3>& activations_derivatives) const
{
    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        // Activations

        activations.device(*default_device) = (1 + combinations.exp().inverse()).inverse();

        // Activations Derivatives

        Index dim = combinations.dimension(1);

        Tensor<type,2> ad_2d = combinations.exp().inverse() / (static_cast<type>(1.0) + combinations.exp().inverse()).pow(2);

        Tensor<type,3> ad_3d(1, dim, 1);
        ad_3d.setZero();

        ad_3d.chip(0,0) = ad_2d;

        activations_derivatives.device(*default_device) = ad_3d;

        return;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        // Activations

        activations.device(*thread_pool_device) = (1 + combinations.exp().inverse()).inverse();

        // Activations Derivatives

        Index dim = combinations.dimension(1);

        Tensor<type,2> ad_2d = combinations.exp().inverse() / (static_cast<type>(1.0) + combinations.exp().inverse()).pow(2);

        Tensor<type,3> ad_3d(1, dim, 1);
        ad_3d.setZero();

        ad_3d.chip(0,0) = ad_2d;

        activations_derivatives.device(*thread_pool_device) = ad_3d;

        return;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        return;
    }
    }

}


void Layer::softmax_derivatives(const Tensor<type, 2>& combinations,
                                 Tensor<type, 2>& activations,
                                 Tensor<type, 3>& activations_derivatives) const
{
    Tensor<type, 0> sum;

    const Index dim = combinations.dimension(1);

    switch(device_pointer->get_type())
    {
    case Device::EigenDefault:
    {
//        DefaultDevice* default_device = device_pointer->get_eigen_default_device();

        return;
    }

    case Device::EigenSimpleThreadPool:
    {
        ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

        //Activations

        sum.device(*thread_pool_device) = combinations.exp().sum();

        activations.device(*thread_pool_device) = combinations.exp() / sum(0);

        //Activations derivatives

        Tensor<type,3> activations_derivatives_(dim, dim, 1);

        for(Index i = 0; i < dim; i++)
        {            
            for(Index j = 0; j < dim; j++)
            {                
                Index delta = 0;
                if(i == j) delta = 1;

                activations_derivatives_(i,j,0) = activations(i) * (delta - activations(j));
            }
       }

        activations_derivatives.device(*thread_pool_device) = activations_derivatives_;

        return;
    }

    case Device::EigenGpu:
    {
//        GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

        return;
    }
}

    /*
#ifdef __OPENNN_DEBUG__

    if(combinations.dimension(0) != activations.dimension(2))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: Layer Class.\n"
               << "void softmax_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) method.\n"
               << "Number of rows in x("<< combinations.dimension(0)
               << ") must be equal to number of rows in y(" <<activations.dimension(2)<< ").\n";

        throw logic_error(buffer.str());
    }

#endif


    return;
*/
}




}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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

