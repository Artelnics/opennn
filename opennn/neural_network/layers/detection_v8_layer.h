//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E T E C T I O N   V 8   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/detection_head.h"
#include "opennn/neural_network/layers/layer.h"
#include "opennn/neural_network/operators/operator.h"

namespace opennn
{

struct DetectionV8Operator : Operator
{
    Index grid_size      = 0;
    Index grid_width     = 0;
    Index classes_number = 0;
    Index reg_max        = 1;

    void set(const Shape&, Index reg_max);

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;
};

class DetectionV8 final : public Layer, public DetectionHeadEndpoint
{
public:

    DetectionV8(const Shape& = {}, const string& = "detection_v8");
    DetectionV8(const Shape&, Index reg_max, const string& = "detection_v8");

    Shape get_output_shape() const override { return input_shape; }
    Index get_classes_number() const { return detection.classes_number; }
    Index get_reg_max() const { return detection.reg_max; }
    DetectionHeadMetadata get_detection_head_metadata() const noexcept override
    {
        return {DetectionHeadKind::AnchorFree,
                1,
                detection.classes_number,
                detection.reg_max,
                DetectionClassActivation::Sigmoid};
    }

    void set(const Shape&, const string&);
    void set(const Shape&, Index reg_max, const string&);
    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 3); }

    void on_compute_dtype_changed() override
    {
        throw_if(get_compute_dtype() != Type::FP32,
                 "{} layer supports FP32 activations only; compile the network with Type::FP32.",
                 get_name());
    }

    void apply_input_shape(const Shape& new_input_shape) override { set(new_input_shape, detection.reg_max, label); }

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    DetectionV8Operator detection;

    void configure_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
