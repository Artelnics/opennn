//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E T E C T I O N   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/detection_head.h"
#include "opennn/neural_network/layers/layer.h"
#include "opennn/neural_network/operators/operator.h"

namespace opennn
{

struct DetectionOperator : Operator
{
    using ClassActivation = DetectionClassActivation;

    Index grid_size = 0;
    Index grid_width = 0;
    Index boxes_per_cell = 0;
    Index classes_number = 0;
    ClassActivation class_activation = ClassActivation::Softmax;

    vector<array<float, 2>> anchors;
    // Device mirror of anchors. DetectionOperator::set invalidates it whenever
    // the host configuration changes.
    Buffer device_anchors{Device::CUDA};

    void set(const Shape&, const vector<array<float, 2>>&);

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;
};

class Detection final : public Layer, public DetectionHeadEndpoint
{
public:

    using ClassActivation = DetectionOperator::ClassActivation;

    Detection(const Shape& = {},
              const vector<array<float, 2>>& = {},
              const string& = "detection_layer");

    Shape get_output_shape() const override { return input_shape; }
    const vector<array<float, 2>>& get_anchors() const { return detection.anchors; }
    ClassActivation get_class_activation() const { return detection.class_activation; }
    DetectionHeadMetadata get_detection_head_metadata() const noexcept override
    {
        return {DetectionHeadKind::AnchorBased,
                detection.boxes_per_cell,
                detection.classes_number,
                1,
                detection.class_activation};
    }

    void set(const Shape&, const vector<array<float, 2>>&, const string&);
    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 3); }

    // FP32 only: this layer's kernels reinterpret their slots as float, so a
    // BF16 compute dtype - which compile() applies to every layer with no
    // capability check - handed them half-width buffers to read as full-width
    // ones. Refusing is the difference between a message and silent corruption.
    void on_compute_dtype_changed() override
    {
        throw_if(get_compute_dtype() != Type::FP32,
                 "{} layer supports FP32 activations only; compile the network with Type::FP32.",
                 get_name());
    }


    void apply_input_shape(const Shape&) override;
    void set_class_activation(ClassActivation new_class_activation) { detection.class_activation = new_class_activation; }

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    DetectionOperator detection;

    void configure_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
