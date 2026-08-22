//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U P S A M P L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/neural_network/layers/layer.h"
#include "opennn/neural_network/operators/operator.h"

namespace opennn
{

struct UpsamplingOperator : Operator
{
    Index input_height = 0;
    Index input_width = 0;
    Index channels = 0;
    Index scale_factor = 2;

    void set(Index, Index, Index, Index);

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;
};

class Upsampling final : public Layer
{
public:

    Upsampling(const Shape& = {},
               Index scale_factor = 2,
               const string& = "upsampling_layer");

    Shape get_output_shape() const override;

    void set(const Shape&, Index, const string&);
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
    void set_scale_factor(Index);

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    UpsamplingOperator upsampling;

    void configure_operator();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
