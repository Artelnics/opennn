//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "statistics.h"
#include "layer.h"
#include "operators.h"
#include "variable.h"

namespace opennn
{

class Scaling final : public Layer
{
public:

    Scaling(const Shape& = {});

    Shape get_output_shape() const override { return input_shape; }

    VectorR get_minimums() const;
    VectorR get_maximums() const;
    VectorR get_means() const;
    VectorR get_standard_deviations() const;

    const vector<ScalerMethod>& get_scalers() const { return scalers; }

    float get_min_range() const { return scale_op.min_range; }
    float get_max_range() const { return scale_op.max_range; }

    void set(const Shape& = {});

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;

    void set_descriptives(const vector<Descriptives>&);

    void set_min_max_range(float min, float max) { scale_op.min_range = min; scale_op.max_range = max; }

    void set_scalers(const vector<string>&);
    void set_scalers(const string&);

    void forward_propagate(ForwardPropagation&, size_t, bool) noexcept override;

    string write_no_scaling_expression(const vector<string>&, const vector<string>&) const;
    string write_minimum_maximum_expression(const vector<string>&, const vector<string>&) const;
    string write_mean_standard_deviation_expression(const vector<string>&, const vector<string>&) const;
    string write_standard_deviation_expression(const vector<string>&, const vector<string>&) const;

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

private:

    vector<ScalerMethod> scalers;

    Scale scale_op;

    void flush_scalers_to_states();
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
