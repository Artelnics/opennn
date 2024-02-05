//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LAYER_H
#define LAYER_H

// System includes

#include <iostream>
#include <string>
#include <sstream>
#include <iostream>

// OpenNN includes

#include "config.h"
#include "tensor_utilities.h"


namespace opennn {

class Layer;

struct LayerForwardPropagation;
struct LayerBackPropagation;
struct LayerBackPropagationLM;

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/struct_layer_cuda.h"
#endif


/// This abstract class represents the concept of layer of neurons in OpenNN.

/// A layer is a group of neurons having connections to the same inputs and sending outputs to the same destinations.

class Layer
{

public:

    /// This enumeration represents the possible types of layers.

    enum class Type{Scaling2D,
                    Scaling4D,
                    Convolutional,
                    Perceptron,
                    Perceptron3D,
                    Pooling,
                    Probabilistic,
                    Probabilistic3D,
                    LongShortTermMemory,
                    Recurrent,
                    Unscaling,
                    Bounding,
                    Flatten,
                    RegionProposal,
                    NonMaxSuppression,
                    MultiheadAttention,
                    Embedding};

    // Constructor

    explicit Layer()   
    {
        const int n = omp_get_max_threads();

        thread_pool = new ThreadPool(n);
        thread_pool_device = new ThreadPoolDevice(thread_pool, n);
    }

    // Destructor

    virtual ~Layer();

    string get_name() const
    {
        return layer_name;
    }

    // Get neurons number

    virtual Index get_inputs_number() const;
    virtual Index get_neurons_number() const;

    virtual void set_inputs_number(const Index&);
    virtual void set_neurons_number(const Index&);

    // Layer type

    Type get_type() const;

    string get_type_string() const;

    // Parameters initialization methods

    virtual void set_parameters_constant(const type&);

    virtual void set_parameters_random();

    // Architecture

    virtual Index get_parameters_number() const;
    virtual Tensor<type, 1> get_parameters() const;

    virtual void set_parameters(const Tensor<type, 1>&, const Index&);

    void set_threads_number(const int&);

    // Forward propagation

    virtual void forward_propagate(const pair<type*, dimensions>&,
                                   LayerForwardPropagation*,
                                   const bool&) = 0;

    virtual void forward_propagate(const pair<type*, dimensions>&,
                                   Tensor<type, 1>&,
                                   LayerForwardPropagation*);

    // Back propagation

    virtual void calculate_hidden_delta(LayerForwardPropagation*,
                                        LayerBackPropagation*,
                                        LayerBackPropagation*) const {}

    virtual void calculate_hidden_delta_lm(LayerForwardPropagation*,
                                           LayerBackPropagationLM*,
                                           LayerBackPropagationLM*) const {}

    virtual void calculate_error_gradient(const pair<type*, dimensions>&,
                                          LayerForwardPropagation*,
                                          LayerBackPropagation*) const {}

    virtual void insert_gradient(LayerBackPropagation*,
                                 const Index&,
                                 Tensor<type, 1>&) const {}

    virtual void calculate_squared_errors_Jacobian_lm(const Tensor<type, 2>&,
                                                      LayerForwardPropagation*,
                                                      LayerBackPropagationLM*) {}

    virtual void insert_squared_errors_Jacobian_lm(LayerBackPropagationLM*,
                                                   const Index&,
                                                   Tensor<type, 2>&) const {}

    // Serialization methods

    virtual void from_XML(const tinyxml2::XMLDocument&) {}

    virtual void write_XML(tinyxml2::XMLPrinter&) const {}

    // Expression methods

    virtual string write_expression(const Tensor<string, 1>&, const Tensor<string, 1>&) const {return string();}

protected:

    ThreadPool* thread_pool = nullptr;
    ThreadPoolDevice* thread_pool_device = nullptr;

    /// Layer name.

    string layer_name = "layer";

    /// Layer type.

    Type layer_type;

    /// Activation functions

    template <int rank>
    void linear(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        y.device(*thread_pool_device) = x;
    }


    template <int rank>
    void binary(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        /*
        const Tensor<bool, 1> if_sentence = x < x.constant(type(0.5));

        const Tensor<type, 1> f_1 = x.constant(type(false));

        const Tensor<type, 1> f_2 = x.constant(type(true));

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
*/
    }


    template <int rank>
    void exponential_linear(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        y.device(*thread_pool_device) = x;

        /*
         
                 const type alpha = type(1);

    y.device(*thread_pool_device) = y.select(y < 0, alpha * (y.exp() - type(1)));

    const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

    Tensor<type, 1> f_1(x.dimension(0));

    Tensor<type, 1> f_2(x.dimension(0));

    f_1.device(*thread_pool_device) = alpha*(x.exp() - type(1));

    f_2 = x;

    y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
    */

    }


    template <int rank>
    void hard_sigmoid(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        y.device(*thread_pool_device) = (type(0.5) + x*type(0.2)).cwiseMin(type(2.5)).cwiseMax(type(-2.5));
    }


    template <int rank>
    void hyperbolic_tangent(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        y.device(*thread_pool_device) = x.tanh();
    }


    template <int rank>
    void logistic(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        y.device(*thread_pool_device) = type(1)/(type(1) - x.exp());
    }


    template <int rank>
    void rectified_linear(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        y.device(*thread_pool_device) = x.cwiseMax(type(0));
    }


    template <int rank>
    void leaky_rectified_linear(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {

    }


    template <int rank>
    void scaled_exponential_linear(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        /*
        const type lambda = type(1.0507);

        const type alpha = type(1.67326);

        const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

        const Tensor<type, 1> f_1 = lambda*alpha*(x.exp()-type(1));

        Tensor<type, 1> f_2 = lambda*x;

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
*/
    }


    template <int rank>
    void soft_plus(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        y.device(*thread_pool_device) = (type(1) + x.exp()).log();
    }


    template <int rank>
    void soft_sign(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
/*
        const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

        const Tensor<type, 1> f_1 = x / (type(1) - x);

        const Tensor<type, 1> f_2 = x / (type(1) + x);

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
*/
    }


    template <int rank>
    void symmetric_threshold(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
/*
        const Tensor<bool, 1> if_sentence = x > x.constant(type(0));

        Tensor<type, 1> ones(x.dimension(0));
        ones.setConstant(type(1));

        y.device(*thread_pool_device) = if_sentence.select(ones, -ones);
*/
    }


    template <int rank>
    void threshold(const Tensor<type, rank>& x, Tensor<type, rank>& y) const
    {
        /*
        const Tensor<bool, 1> if_sentence = x >= x.constant(type(0));

        Tensor<type, 1> ones(x.dimension(0));
        ones.setConstant(type(1));

        Tensor<type, 1> zeros(x.dimension(0));
        zeros.setConstant(type(0));

        y.device(*thread_pool_device) = if_sentence.select(ones, zeros);
*/
    }


    void softmax(const Tensor<type, 2>& x, Tensor<type, 2>& y) const
    {
        const Index rows_number = x.dimension(0);
        const Index columns_number = x.dimension(1);

        const Eigen::array<Index, 1> last_dimension{ {2} };
        const Eigen::array<Index, 2> range_2{ {rows_number, 1} };
        const Eigen::array<Index, 2> expand_last_dim{ {1, columns_number} };

        y.device(*thread_pool_device) = x - x.maximum(last_dimension)
                                             .reshape(range_2)
                                             .broadcast(expand_last_dim);

        y.device(*thread_pool_device) = y.exp();

        Tensor<type, 2> y_sum = y.sum(last_dimension)
                                 .reshape(range_2)
                                 .broadcast(expand_last_dim);

        y.device(*thread_pool_device) = y / y.sum(last_dimension)
                                             .reshape(range_2)
                                             .broadcast(expand_last_dim);
    }


    void softmax(const Tensor<type, 3>& x, Tensor<type, 3>& y) const
    {
        const Index rows_number = x.dimension(0);
        const Index columns_number = x.dimension(1);
        const Index channels_number = x.dimension(2);

        const Eigen::array<Index, 1> last_dimension{ {2} };
        const Eigen::array<Index, 3> range_3{ {rows_number, columns_number, 1} };
        const Eigen::array<Index, 3> expand_last_dim{ {1, 1, channels_number} };

        y.device(*thread_pool_device) = x - x.maximum(last_dimension)
                                             .reshape(range_3)
                                             .broadcast(expand_last_dim);

        y.device(*thread_pool_device) = y.exp();

        Tensor<type, 3> y_sum = y.sum(last_dimension)
                                 .reshape(range_3)
                                 .broadcast(expand_last_dim);

        y.device(*thread_pool_device) = y / y_sum;
    }


    void softmax(const Tensor<type, 4>& x, Tensor<type, 4>& y) const
    {
        const Index rows_number = x.dimension(0);
        const Index columns_number = x.dimension(1);
        const Index channels_number = x.dimension(2);
        const Index blocks_number = x.dimension(3);

        const Eigen::array<Index, 1> last_dimension{ {3} };
        const Eigen::array<Index, 4> range_3{ {rows_number, columns_number, channels_number, 1} };
        const Eigen::array<Index, 4> expand_last_dim{ {1, 1, 1, blocks_number} };

        y.device(*thread_pool_device) = x - x.maximum(last_dimension)
            .reshape(range_3)
            .broadcast(expand_last_dim);

        y.device(*thread_pool_device) = y.exp();

        Tensor<type, 3> y_sum = y.sum(last_dimension)
            .reshape(range_3)
            .broadcast(expand_last_dim);

        y.device(*thread_pool_device) = y / y_sum;
    }


    template <int rank>
    void linear_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = x;

        dy_dx.setConstant(type(1));
    }


    template <int rank>
    void exponential_linear_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        /*
        const type alpha = type(1);

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
*/
    }


    template <int rank>
    void hard_sigmoid_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = (x*type(0.2) + type(0.5)).cwiseMin(type(2.5)).cwiseMax(type(-2.5));

        dy_dx.setConstant(type(0.2));
        dy_dx.device(*thread_pool_device) = dy_dx.cwiseMax(x < type(-2.5)).cwiseMin(x > type(2.5));
    }


    template <int rank>
    void hyperbolic_tangent_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = x.tanh();

        dy_dx.device(*thread_pool_device) = type(1) - y.square();
    }


    template <int rank>
    void logistic_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        logistic(x, y);

        dy_dx.device(*thread_pool_device) = y*(type(1) - y);
    }


    template <int rank>
    void rectified_linear_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = x.cwiseMax(type(0));

        dy_dx.device(*thread_pool_device) = y.cwiseMax(y > type(0));
    }


    template <int rank>
    void leaky_rectified_linear_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {

    }


    template <int rank>
    void scaled_exponential_linear_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        /*
        const type lambda = type(1.0507);

        const type alpha = type(1.67326);

        const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

        Tensor<type, 1> f_1 = lambda*alpha*(x.exp()-type(1));

        Tensor<type, 1> f_2 = lambda*x;

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        f_1 = lambda*alpha*x.exp();

        f_2 = x.constant(type(1))*lambda;

        dy_dx.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
*/
    }


    template <int rank>
    void soft_plus_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = (x.constant(type(1)) + x.exp()).log();

        dy_dx.device(*thread_pool_device) = type(1) / (type(1) + x.exp().inverse());

    }


    template <int rank>
    void soft_sign_derivatives(const Tensor<type, rank>& x, Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
/*
        const Tensor<bool, 1> if_sentence = x < x.constant(type(0));

        Tensor<type, 1> f_1 = x / (type(1) - x);

        Tensor<type, 1> f_2 = x / (type(1) + x);

        y.device(*thread_pool_device) = if_sentence.select(f_1, f_2);

        // Activations Derivatives

        f_1 = type(1) / (type(1) - x).pow(type(2));

        f_2 = type(1) / (type(1) + x).pow(type(2));

        dy_dx.device(*thread_pool_device) = if_sentence.select(f_1, f_2);
*/
    }


    void softmax_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y, Tensor<type, 3>& dy_dx) const
    {
        const Index rows_number = x.dimension(0);
        const Index columns_number = x.dimension(1);

        softmax(x, y);

        dy_dx.setZero();

        Tensor<type, 1> y_row(columns_number);
        Tensor<type, 2> dy_dx_matrix(columns_number, columns_number);

        for (Index i = 0; i < rows_number; i++)
        {
            y_row = y.chip(i, 0);

            dy_dx_matrix = -kronecker_product(y_row, y_row);

            sum_diagonal(dy_dx_matrix, y_row);

            dy_dx.chip(i, 0) = dy_dx_matrix;
        }
    }


    void softmax_derivatives(const Tensor<type, 3>& x, Tensor<type, 3>& y, Tensor<type, 4>& dy_dx) const
    {
        const Index rows_number = x.dimension(0);
        const Index columns_number = x.dimension(1);
        const Index channels_number = x.dimension(2);

        softmax(x, y);

        dy_dx.setZero();

        Tensor<type, 2> y_row(columns_number, channels_number);
        Tensor<type, 1> y_element(channels_number);
        Tensor<type, 2> dy_dx_element(channels_number, channels_number);

        for (Index i = 0; i < rows_number; i++)
        {
            y_row = y.chip(i, 0);

            for (Index j = 0; j < columns_number; j++)
            {
                y_element = y_row.chip(j, 0);

                dy_dx_element = -kronecker_product(y_element, y_element);

                sum_diagonal(dy_dx_element, y_element);

                dy_dx.chip(i, 0).chip(j, 0) = dy_dx_element;
            }
        }
    }

    const Eigen::array<IndexPair<Index>, 1> A_BT = {IndexPair<Index>(1, 1)};
    const Eigen::array<IndexPair<Index>, 1> AT_B = {IndexPair<Index>(0, 0)};
    const Eigen::array<IndexPair<Index>, 1> A_B = {IndexPair<Index>(1, 0)};

#ifdef OPENNN_CUDA
    #include "../../opennn-cuda/opennn-cuda/layer_cuda.h"
#else
};
#endif
}

#endif // LAYER_H
