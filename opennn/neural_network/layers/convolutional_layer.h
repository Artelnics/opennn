//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O N V O L U T I O N A L   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <cstdio>
#include "opennn/neural_network/layers/layer.h"
#include "opennn/neural_network/operators/activation_operator.h"
#include "opennn/neural_network/operators/batch_norm_operator.h"
#include "opennn/neural_network/operators/convolution_operator.h"

namespace opennn
{

class Convolutional final : public Layer
{
public:

    Convolutional(const Shape& = {3, 3, 1},
                  const Shape& = {3, 3, 1, 1},
                  const string& = "Identity",
                  const Shape& = {1, 1},
                  const string& = "Valid",
                  bool = false,
                  const string& = "convolutional_layer");

    Shape get_input_shape() const noexcept override
    { return {convolution.input_height, convolution.input_width, input_channels}; }
    Shape get_output_shape() const override;

    Index get_output_height() const;
    Index get_output_width() const;

    Index get_kernel_height() const noexcept { return convolution.kernel_height; }
    Index get_kernel_width() const noexcept { return convolution.kernel_width; }
    Index get_kernel_channels() const noexcept { return convolution.kernel_channels; }
    Index get_kernels_number() const noexcept { return convolution.kernels_number; }

    Index get_row_stride() const noexcept { return convolution.row_stride; }
    Index get_column_stride() const noexcept { return convolution.column_stride; }

    Index get_padding_height() const;
    Index get_padding_width() const;

    ActivationFunction get_activation_function() const noexcept { return activation_operator.activation_function; }
    ActivationFunction get_output_activation() const noexcept override { return activation_operator.activation_function; }

    bool get_batch_normalization() const { return batch_norm.active(); }

    bool get_residual() const noexcept { return residual; }
    Index get_sources_number() const noexcept override { return residual ? 2 : 1; }
    void set_residual(bool);

    vector<TensorSpec> get_forward_specs(Index) const override;
    vector<TensorSpec> get_backward_specs(Index) const override;
    bool backward_uses_forward_output() const noexcept override { return get_output_activation() != ActivationFunction::Identity; }
    bool backward_uses_input(size_t input) const noexcept override { return input != 1 || !residual; }
    ForwardSlotKind get_forward_slot_kind(size_t spec) const override
    {
        return spec == size_t(ReluMask) - 1 ? ForwardSlotKind::TrainingOnly : ForwardSlotKind::Pooled;
    }
    bool folds_input_delta_addend(size_t input) const noexcept override { return input == 0; }
    size_t get_recomputable_forward_slot() const noexcept override
    {
        return batch_norm.active() ? size_t(0) : SIZE_MAX;
    }
    void recompute_forward_slot(ForwardPropagation&, size_t) override;

    void set(const Shape& = {0, 0, 0},
             const Shape& = {3, 3, 1, 1},
             const string& = "Identity",
             const Shape& = {1, 1},
             const string& = "Valid",
             bool = false,
             const string& = "convolutional_layer");

    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 3); }

    void apply_input_shape(const Shape&) override;

    void on_compute_dtype_changed() override { update_convolution_operator(); }

    void set_activation_function(const string&);
    void set_batch_normalization(bool);

    void load_darknet_weights(FILE*);

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;
    void on_loaded() override { update_convolution_operator(); }

    void forward_propagate(ForwardPropagation&, size_t, bool) override;

private:

#ifdef OPENNN_HAS_CUDA
    Buffer folded_parameters;
    bool   folded_dirty = true;

    bool forward_propagate_folded(ForwardPropagation&, size_t);
#endif

    Index input_channels = 0;

    bool use_padding = false;
    bool residual = false;

    ConvolutionOperator convolution;
    ActivationOperator  activation_operator;
    BatchNormalizationOperator   batch_norm;

    enum Forward {Input, ConvolutionView, BatchNormMean, BatchNormInverseVariance, ReluMask, Output};

    void update_convolution_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
