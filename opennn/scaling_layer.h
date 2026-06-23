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
#include "scale_operator.h"
#include "variable.h"

namespace opennn
{

class Scaling : public Layer
{
public:

    Scaling(const Shape& = {});

    Shape get_output_shape() const override { return input_shape; }

    const vector<Descriptives>& get_descriptives() const { return descriptives; }
    const vector<ScalerMethod>& get_scalers()      const { return scalers; }

    VectorR get_minimums()            const;
    VectorR get_maximums()            const;
    VectorR get_means()               const;
    VectorR get_standard_deviations() const;

    float get_min_range() const { return min_range; }
    float get_max_range() const { return max_range; }

    void set(const Shape& = {});
    void set_input_shape(const Shape&) override;

    void set_descriptives(const vector<Descriptives>&);
    void set_scalers(const vector<string>&);
    void set_scalers(const string&);

    float* link_states(float*, Device) override;

    void read_JSON_body(const Json*) override;
    void write_JSON_body(JsonWriter&) const override;

    string write_expression(const vector<string>& input_names,
                            const vector<string>& output_names) const override;

protected:

    Scaling(LayerType);

    vector<Descriptives> descriptives;
    vector<ScalerMethod> scalers;
    float min_range = -1.0f;
    float max_range = 1.0f;

    Buffer op_storage;
    Device op_storage_device = Device::CPU;
    bool   op_storage_dirty = true;

    ScaleOp scale_op;

    void refresh_op_storage(Device device);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
