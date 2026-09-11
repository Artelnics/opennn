// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/network/layers/layer.h"
#include "opennn/network/operators/operator.h"

namespace opennn
{

struct NonMaxSuppressionOperator : Operator
{
    Index grid_size = 0;
    Index grid_width = 0;
    Index boxes_per_cell = 0;
    Index classes_number = 0;
    float confidence_threshold = 0.5f;
    float iou_threshold = 0.4f;

    void set(const Shape&,
             Index,
             float,
             float);

    void forward_propagate(ForwardPropagation&, size_t, ForwardPropagationMode) override;

private:
    void apply(const TensorView&, TensorView&) const;
};

class NonMaxSuppression final : public Layer
{
public:

    NonMaxSuppression(const Shape& = {},
                      Index boxes_per_cell = 1,
                      float confidence_threshold = 0.5f,
                      float iou_threshold = 0.4f,
                      const string& = "non_max_suppression_layer");

    Shape get_output_shape() const override;

    void set(const Shape&,
             Index,
             float,
             float,
             const string&);

    bool accepts_input_rank(Index rank) const override { return is_one_of(rank, 3); }

    void apply_input_shape(const Shape&) override;

    float get_confidence_threshold() const { return nms.confidence_threshold; }
    void  set_confidence_threshold(float t) { nms.confidence_threshold = t; }

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    NonMaxSuppressionOperator nms;

    void configure_operator();
};

}
