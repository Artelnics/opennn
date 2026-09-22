// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#pragma once

#include "opennn/core/opennn_types.h"

namespace opennn
{

class Network;

// A network written as an ONNX model (IR 7, opset 13), encoded directly so that
// no onnx dependency is needed. The graph has one float input "input"
// [batch, inputs] and one float output "output" [batch, outputs], and it
// reproduces Network::calculate_outputs: scaling, dense layers, unscaling and
// clamping are all part of it. The input and output feature names are stored as
// the "input_names" and "output_names" metadata entries.

struct OnnxModel
{
    string bytes;
    vector<string> layers;
    Index nodes_number = 0;
};

// Supports sequential networks of Scaling, Dense, Unscaling and Clamping layers
// with FP32 parameters, and throws naming the first layer it cannot represent.
OnnxModel build_onnx_model(const Network&);

void save_onnx_model(const Network&, const filesystem::path&);

}
