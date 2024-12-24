//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H

#include "layer.h"

namespace opennn
{

class ConvolutionalLayer;

#ifdef OPENNN_CUDA
struct PoolingLayerForwardPropagationCuda;
struct PoolingLayerBackPropagationCuda;
#endif

class PoolingLayer : public Layer
{

public:

    enum class PoolingMethod{MaxPooling, AveragePooling};

    PoolingLayer(const dimensions& = {2, 2, 1}, // Input dimensions {height,width,channels}
                          const dimensions& = { 2, 2 },  // Pool dimensions {pool_height,pool_width}
                          const dimensions& = { 2, 2 },  // Stride dimensions {row_stride, column_stride}
                          const dimensions& = { 0, 0 },  // Padding dimensions {padding_height, padding_width}
                          const PoolingMethod& = PoolingMethod::MaxPooling,
                          const string = "pooling_layer");

    dimensions get_input_dimensions() const override;
    dimensions get_output_dimensions() const override;

    Index get_input_height() const;
    Index get_input_width() const;

    Index get_output_height() const;
    Index get_output_width() const;

    Index get_channels_number() const;

    Index get_padding_height() const;
    Index get_padding_width() const;

    Index get_row_stride() const;
    Index get_column_stride() const;

    Index get_pool_height() const;
    Index get_pool_width() const;

    PoolingMethod get_pooling_method() const;

    string write_pooling_method() const;

    void set(const dimensions& = {0, 0, 0},
             const dimensions& = {1, 1},
             const dimensions& = {1, 1},
             const dimensions& = {0, 0},
             const PoolingMethod& = PoolingMethod::MaxPooling,
             const string = "pooling_layer");

    void set_input_dimensions(const dimensions&) override;

    void set_padding_height(const Index&);
    void set_padding_width(const Index&);

    void set_row_stride(const Index&);
    void set_column_stride(const Index&);

    void set_pool_size(const Index&, const Index&);

    void set_pooling_method(const PoolingMethod&);
    void set_pooling_method(const string&);

    void forward_propagate(const vector<pair<type*, dimensions>>&,
                           unique_ptr<LayerForwardPropagation>&,
                           const bool&) override;

    void forward_propagate_max_pooling(const Tensor<type, 4>&,
                                       unique_ptr<LayerForwardPropagation>&,
                                       const bool&) const;

    void forward_propagate_average_pooling(const Tensor<type, 4>&,
                                           unique_ptr<LayerForwardPropagation>&,
                                           const bool&) const;

    void back_propagate(const vector<pair<type*, dimensions>>&,
                        const vector<pair<type*, dimensions>>&,
                        unique_ptr<LayerForwardPropagation>&,
                        unique_ptr<LayerBackPropagation>&) const override;

    void back_propagate_max_pooling(const Tensor<type, 4>&,
                                    const Tensor<type, 4>&,
                                    unique_ptr<LayerForwardPropagation>&,
                                    unique_ptr<LayerBackPropagation>&) const;

    void back_propagate_average_pooling(const Tensor<type, 4>&,
                                        const Tensor<type, 4>&,
                                        unique_ptr<LayerBackPropagation>&) const;

    void from_XML(const XMLDocument&) override;
    void to_XML(XMLPrinter&) const override;

    void print() const override;

    #ifdef OPENNN_CUDA
        #include "../../opennn_cuda/opennn_cuda/pooling_layer_cuda.h"
    #endif

private:

    dimensions input_dimensions;

    Index pool_height = 1;

    Index pool_width = 1;

    Index padding_height = 0;

    Index padding_width = 0;

    Index row_stride = 1;

    Index column_stride = 1;

    PoolingMethod pooling_method = PoolingMethod::AveragePooling;

    const Eigen::array<ptrdiff_t, 2> pooling_dimensions = {1, 2};
};


struct PoolingLayerForwardPropagation : LayerForwardPropagation
{   
    PoolingLayerForwardPropagation(const Index& = 0, Layer* = nullptr);
    
    pair<type*, dimensions> get_outputs_pair() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 4> outputs;

    Tensor<type, 5> image_patches;

    Tensor<Index, 4> maximal_indices;
};


struct PoolingLayerBackPropagation : LayerBackPropagation
{
    PoolingLayerBackPropagation(const Index& = 0, Layer* = nullptr);

    vector<pair<type*, dimensions>> get_input_derivative_pairs() const override;

    void set(const Index& = 0, Layer* = nullptr);

    void print() const override;

    Tensor<type, 4> deltas_by_pool_size;

    Tensor<type, 4> input_derivatives;
};


#ifdef OPENNN_CUDA
    #include "../../opennn_cuda/opennn_cuda/pooling_layer_forward_propagation_cuda.h"
    #include "../../opennn_cuda/opennn_cuda/pooling_layer_back_propagation_cuda.h"
#endif

}

#endif // POOLING_LAYER_H
