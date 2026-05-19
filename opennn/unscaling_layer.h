//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U N S C A L I N G   L A Y E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "layer.h"
#include "operators.h"
#include "scaling.h"
#include "statistics.h"
#include "variable.h"

namespace opennn
{

class Unscaling final : public Layer
{
public:

    Unscaling(const Shape& = {0}, const string& = "unscaling_layer");

    Shape get_input_shape()  const override { return { Index(scalers.size()) }; }
    Shape get_output_shape() const override { return { Index(scalers.size()) }; }

    const vector<Descriptives>& get_descriptives() const { return descriptives; }
    const vector<ScalerMethod>& get_scalers()      const { return scalers; }

    VectorR get_minimums()            const;
    VectorR get_maximums()            const;
    VectorR get_means()               const;
    VectorR get_standard_deviations() const;

    float get_min_range() const { return min_range; }
    float get_max_range() const { return max_range; }

    void set(Index = 0, const string& = "unscaling_layer");

    void set_input_shape(const Shape&) override;
    void set_output_shape(const Shape&) override;

    void set_descriptives(const vector<Descriptives>&);
    void set_min_max_range(float min, float max);
    void set_scalers(const vector<string>&);
    void set_scalers(const string&);

    float* link_states(float*) override;

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

    string write_expression(const vector<string>& input_names,
                            const vector<string>& output_names) const override;

private:

    vector<Descriptives> descriptives;
    vector<ScalerMethod> scalers;
    float min_range = -1.0f;
    float max_range = 1.0f;

    Buffer op_storage;
    bool   op_storage_dirty = true;

    UnscaleOp unscale_op;

    void refresh_op_storage(Device device);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
