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
        case PrincipalComponents: return "PrincipalComponents";

        case Convolutional: return "Convolutional";

        case Perceptron: return "Perceptron";

        case Bounding: return "Bounding";

        case Pooling: return "Pooling";

        case Probabilistic: return "Probabilistic";

        case LongShortTermMemory: return "LongShortTermMemory";

        case Recurrent: return "Recurrent";

        case Scaling: return "Scaling";

        case Unscaling: return "Unscaling";
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


void Layer::set_parameters(const Tensor<type, 1>&)
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
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

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

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

            y.device(*thread_pool_device) = x.tanh();

             break;
         }

        case Device::EigenGpu:
        {
             //GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

             //y.device(gpu_device) = x.tanh();

             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
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

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    const Index n = x.size();

    #pragma omp parallel for

    for(Index i = 0; i < n; i++)
    {
        y(i) = 1.0 / (1.0 + exp(-x(i)));
    }

}


void Layer::linear(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    y = x;
}


void Layer::threshold(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    const Index n = x.size();

    #pragma omp parallel for

    for(Index i = 0; i < n; i++)
    {
         y(i) = x(i) < 0 ? -1 : 1;
    }
}


void Layer::symmetric_threshold(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    const Index n = x.size();

    #pragma omp parallel for

     for(Index i = 0; i < n; i++)
     {
         y(i) = x(i) < 0 ? -1 : 1;
     }
}


void Layer::rectified_linear(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    const Index n = x.size();

    #pragma omp parallel for

    for(Index i = 0; i < n; i++)
    {
        y(i) = x(i) < 0 ? 0.0 : x(i);
    }
}


void Layer::scaled_exponential_linear(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    const Index n = x.size();

    const type lambda = static_cast<type>(1.0507);
    const type alpha = static_cast<type>(1.67326);

    #pragma omp parallel for

    for(Index i = 0; i < n; i++)
    {
        x(i) < 0 ? y(i) = lambda * alpha * (exp(x(i)) - 1) : y(i) = lambda * x(i);
    }
}


void Layer::soft_plus(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    y = (x.constant(1) + x.exp()).log();
}


void Layer::soft_sign(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    const Index n = x.size();

    #pragma omp parallel for

    for(Index i = 0; i < n; i++)
    {
       x(i) < 0 ? y(i) = x(i) / (1.0 - x(i)) : y(i) = x(i) / (1.0 + x(i));
    }

}


void Layer::exponential_linear(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    const Index n = x.size();

    const type alpha = 1.0;

    #pragma omp parallel for

    for(Index i = 0; i < n; i++)
    {
        x(i) < 0 ? y(i) = alpha * (exp(x(i)) - 1) : y(i) = x(i);
    }
}


void Layer::logistic_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    const Index n = x.size();

    #pragma omp parallel for

    for(Index i = 0; i < n; i++)
    {
        const type exponential = exp(-x(i));

        y(i) = exponential/((1.0 + exponential)*(static_cast<type>(1.0) + exponential));
    }

}


void Layer::threshold_derivatives(const Tensor<type, 2>&, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    y.setZero();
}


void Layer::symmetric_threshold_derivatives(const Tensor<type, 2>&, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    y.setZero();
}


void Layer::linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             y.setConstant(1.0);

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            y.setConstant(1.0);

             break;
         }

        case Device::EigenGpu:
        {

             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }
}


void Layer::hyperbolic_tangent_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             y.device(*default_device) = x.constant(1.0) - x.tanh().square();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

            y.device(*thread_pool_device) = x.constant(1.0) - x.tanh().square();

            break;
         }

        case Device::EigenGpu:
        {
             //GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();

             //y.device(*gpu_device) = x.constant(1.0) - x.tanh().square();

             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }
}


void Layer::rectified_linear_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    const Index n = x.size();

    #pragma omp parallel for

    for(Index i = 0; i < n; i++)
    {
        x(i) < 0 ? y(i) = 0.0 : y(i) = 1.0;
    }

}


void Layer::scaled_exponential_linear_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    const Index n = x.size();

    const type lambda = static_cast<type>(1.0507);
    const type alpha = static_cast<type>(1.67326);

    #pragma omp parallel for

    for(Index i = 0; i < n; i++)
    {
        x(i) < 0 ? y(i) = lambda * alpha * exp(x(i)) : y(i) = lambda;
    }

}


void Layer::soft_plus_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    const Index n = x.size();

    #pragma omp parallel for

    for(Index i = 0; i < n; i++)
    {
        y(i) = 1.0/(1.0 + exp(-x(i)));
    }

}


void Layer::soft_sign_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    const Index n = x.size();

    #pragma omp parallel for

    for(Index i = 0; i < n; i++)
    {
       x(i) < 0 ? y(i) = static_cast<type>(1.0) / pow(1 - x(i), 2) : y(i) = static_cast<type>(1.0) / pow(1 + x(i), 2);
    }
}


void Layer::hard_sigmoid_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    const Index n = x.size();

    #pragma omp parallel for

    for(Index i = 0; i < n; i++)
    {
        x(i) < static_cast<type>(-2.5) || x(i) > static_cast<type>(2.5) ? y(i) = 0.0 : y(i) = static_cast<type>(0.2);
    }

}


void Layer::exponential_linear_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    const Index n = x.size();

    const type alpha = 1.0;

    #pragma omp parallel for

    for(Index i = 0; i < n; i++)
    {
        x(i) < 0 ? y(i) = alpha * exp(x(i)) : y(i) = 1.0;
    }
}


/// @todo Fails

void Layer::softmax_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

#ifdef __OPENNN_DEBUG__

    if(x.dimension(0) != y.dimension(0))
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Functions.\n"
              << "void softmax_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) method.\n"
              << "Number of rows in x must be equal to number of rows in d.\n";

       throw logic_error(buffer.str());
    }

#endif

    const Index n = x.dimension(0);

    const Index columns_number = x.dimension(1);

    #pragma omp parallel for

    for(Index i = 0; i < n; i ++)
    {
/*
        const Tensor<type, 1> softmax_values = softmax(x.get_matrix(0).chip(i, 0));

        for(Index j = 0; j < columns_number; j++)
        {
            for(Index k = 0; k < columns_number; k++)
            {
                if(j == k)
                {
                    y(j,k,i) = softmax_values[j]*(1.0 - softmax_values[j]);
                }
                else
                {
                    y(j,k,i) = -softmax_values[j] * softmax_values[k];
                }
            }
        }
*/
    }
}


void Layer::binary(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    const Index n = x.size();

    #pragma omp parallel for

    for(Index i = 0; i < n; i++)
    {
        x(i) < static_cast<type>(0.5) ? y(i) = false : y (i) = true;
    }
}


void Layer::competitive(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    const Index rows_number = x.dimension(0);
/*
    #pragma omp parallel for

    for(Index i = 0; i < rows_number; i++)
    {
        const Index maximal_index = OpenNN::maximal_index(x.get_matrix(0).chip(i, 0));

        y(i, maximal_index) = 1;
    }
*/
}


void Layer::softmax(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
{
    switch(device_pointer->get_type())
    {
         case Device::EigenDefault:
         {
             DefaultDevice* default_device = device_pointer->get_eigen_default_device();

             break;
         }

         case Device::EigenSimpleThreadPool:
         {
            ThreadPoolDevice* thread_pool_device = device_pointer->get_eigen_thread_pool_device();

             break;
         }

        case Device::EigenGpu:
        {
             GpuDevice* gpu_device = device_pointer->get_eigen_gpu_device();


             break;
        }

         default:
         {
            ostringstream buffer;

            buffer << "OpenNN Exception: Layer class.\n"
                   << "void calculate_activations(const Tensor<type, 2>&, Tensor<type, 2>&) const method.\n"
                   << "Unknown device.\n";

            throw logic_error(buffer.str());
        }
    }

    const Index rows_number = x.dimension(0);
    const Index columns_number = x.dimension(1);

    #pragma omp parallel for

  for(Index j = 0; j < rows_number; j++)
  {
      type sum = static_cast<type>(0.0);

      for(Index i = 0; i < columns_number; i++)
      {
        sum += exp(x(j,i));
      }

      for(Index i = 0; i < columns_number; i++)
      {
        y(j,i) = exp(x(j,i)) / sum;
      }
  }
}

}
