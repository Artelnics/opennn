//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef LAYER_H
#define LAYER_H

#include <string>

#include "config.h"
#include "tinyxml2.h"
#include "layer_forward_propagation.h"
#include "layer_back_propagation.h"
#include "layer_back_propagation_lm.h"

using namespace tinyxml2;

namespace opennn
{

#ifdef OPENNN_CUDA
struct LayerForwardPropagationCuda;
struct LayerBackPropagationCuda;
#endif

class Layer
{

public:

    enum class Type{None,
                    Scaling2D,
                    Scaling4D,
                    Addition3D,
                    Normalization3D,
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
                    NonMaxSuppression,
                    MultiheadAttention,
                    Embedding};

    explicit Layer();

    string get_name() const;

    const bool& get_display() const;

    string layer_type_to_string(const Layer::Type&);
    Type string_to_layer_type(const string&);

    Type get_type() const;

    string get_type_string() const;

    virtual void set_input_dimensions(const dimensions&);
    virtual void set_output_dimensions(const dimensions&);

    void set_name(const string&);

    void set_display(const bool&);

    // Parameters initialization

    virtual void set_parameters_constant(const type&);
    virtual void set_parameters_random();

    // Architecture

    virtual Index get_parameters_number() const;
    virtual Tensor<type, 1> get_parameters() const;

    virtual dimensions get_input_dimensions() const;
    virtual dimensions get_output_dimensions() const;

    virtual void set_parameters(const Tensor<type, 1>&, const Index&);

    void set_threads_number(const int&);

    // Forward propagation

    virtual void forward_propagate(const vector<pair<type*, dimensions>>&,
                                   unique_ptr<LayerForwardPropagation>&,
                                   const bool&) = 0;

    // Back propagation

    virtual void back_propagate(const vector<pair<type*, dimensions>>&,
                                const vector<pair<type*, dimensions>>&,
                                unique_ptr<LayerForwardPropagation>&,
                                unique_ptr<LayerBackPropagation>&) const {}

    virtual void back_propagate_lm(const vector<pair<type*, dimensions>>&,
                                   const vector<pair<type*, dimensions>>&,
                                   unique_ptr<LayerForwardPropagation>&,
                                   unique_ptr<LayerBackPropagationLM>&) const {}

    virtual void insert_gradient(unique_ptr<LayerBackPropagation>&,
                                 const Index&,
                                 Tensor<type, 1>&) const {}

    virtual void calculate_squared_errors_Jacobian_lm(const Tensor<type, 2>&,
                                                      unique_ptr<LayerForwardPropagation>&,
                                                      unique_ptr<LayerBackPropagationLM>&) {}

    virtual void insert_squared_errors_Jacobian_lm(unique_ptr<LayerBackPropagationLM>&,
                                                   const Index&,
                                                   Tensor<type, 2>&) const {}

    // Serialization

    virtual void from_XML(const XMLDocument&) {}

    virtual void to_XML(XMLPrinter&) const {}

    // Expression

    virtual string get_expression(const vector<string>&, const vector<string>&) const 
    {
        return string();
    }

    virtual void print() const {}

protected:

    unique_ptr<ThreadPool> thread_pool;
    unique_ptr<ThreadPoolDevice> thread_pool_device;

    string name = "layer";

    Type layer_type = Type::None;

    bool display = true;

    template <int rank>
    void binary(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx, type threshold) const
    {
        y.device(*thread_pool_device) = (y < threshold).select(type(0), type(1));

        if (dy_dx.size() == 0) return;

        dy_dx.setConstant(type(0));
    }


    template <int rank>
    void linear(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        if (dy_dx.size() == 0) return;

        dy_dx.setConstant(type(1));
    }


    template <int rank>
    void exponential_linear(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        const type alpha = type(1);

        y.device(*thread_pool_device) = (y > type(0)).select(y, alpha * (y.exp() - type(1)));

        if (dy_dx.size() == 0) return;
        
        dy_dx.device(*thread_pool_device) = (y > type(0)).select(dy_dx.constant(type(1)), y + alpha);
    }


    template <int rank>
    void hard_sigmoid(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = ((y*type(0.2) + type(0.5)).cwiseMin(type(2.5)).cwiseMax(type(-2.5))).eval();

        if (dy_dx.size() == 0) return;

        dy_dx.device(*thread_pool_device)
            = (y > type(0) && y < type(1)).select(dy_dx.constant(type(0.2)), dy_dx.constant(type(0)));
    }


    template <int rank>
    void hyperbolic_tangent(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = y.tanh();

        if (dy_dx.size() == 0) return;

        dy_dx.device(*thread_pool_device) = (type(1) - y.square()).eval();
    }


    template <int rank>
    void logistic(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = (type(1) + (-y).exp()).inverse();

        if (dy_dx.size() == 0) return;

        dy_dx.device(*thread_pool_device) = (y * (type(1) - y)).eval();
    }


    template <int rank>
    void rectified_linear(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = y.cwiseMax(type(0));

        if (dy_dx.size() == 0) return;

        dy_dx.device(*thread_pool_device) = (y > type(0)).select(dy_dx.constant(type(1)), dy_dx.constant(type(0)));
    }


    template <int rank>
    void leaky_rectified_linear(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx, type slope) const
    {
        y.device(*thread_pool_device) = (y > type(0)).select(y, slope * y);

        if (dy_dx.size() == 0) return;

        dy_dx.device(*thread_pool_device) = (y > type(0)).select(dy_dx.constant(type(1)), dy_dx.constant(type(slope)));
    }


    template <int rank>
    void scaled_exponential_linear(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        const type lambda = type(1.0507);

        const type alpha = type(1.6733);

        y.device(*thread_pool_device) = (y > type(0)).select(lambda * y, lambda * alpha * (y.exp() - type(1)));

        if (dy_dx.size() == 0) return;

        dy_dx.device(*thread_pool_device) = (y > type(0)).select(dy_dx.constant(lambda), y + alpha * lambda);
    }


    template <int rank>
    void soft_plus(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = (type(1) + y.exp()).log();

        if (dy_dx.size() == 0) return;

        dy_dx.device(*thread_pool_device) = type(1) - (-y).exp();
    }


    template <int rank>
    void soft_sign(Tensor<type, rank>& y, Tensor<type, rank>& dy_dx) const
    {
        y.device(*thread_pool_device) = (y / (1 + y.abs())).eval();

        if (dy_dx.size() == 0) return;

        dy_dx.device(*thread_pool_device) = (type(1) + (y / type(1) - y).abs()).pow(-2);
    }


    void competitive(Tensor<type, 2>&) const;

    void softmax(Tensor<type, 2>&) const;
    void softmax(Tensor<type, 3>&) const;
    void softmax(Tensor<type, 4>&) const;

    void softmax_derivatives_times_tensor(const Tensor<type, 3>&, const Tensor<type, 3>&, TensorMap<Tensor<type, 3>>&, Tensor<type, 1>&) const;

    const Eigen::array<IndexPair<Index>, 1> A_BT = {IndexPair<Index>(1, 1)};
    const Eigen::array<IndexPair<Index>, 1> AT_B = {IndexPair<Index>(0, 0)};
    const Eigen::array<IndexPair<Index>, 1> A_B = {IndexPair<Index>(1, 0)};

#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/layer_cuda.h"
#endif

};


#ifdef OPENNN_CUDA
#include "../../opennn_cuda/opennn_cuda/layer_forward_propagation_cuda.h"
#include "../../opennn_cuda/opennn_cuda/layer_back_propagation_cuda.h"
#endif

}

#endif // LAYER_H
