//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E G I S T R Y   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <memory>
#include <string>

#include "opennn/core/enum_map.h"

namespace opennn
{

enum class LayerType
{
    Activation,
    Addition,
    Clamping,
    Concatenation,
    Convolutional,
    Dense,
    Detection,
    DetectionV8,
    Embedding,
    Flatten,
    LongShortTermMemory,
    MultiHeadAttention,
    Normalization3d,
    GroupedQueryAttention,
    NonMaxSuppression,
    Pooling,
    Pooling3d,
    Recurrent,
    Scaling,
    Tokenizer,
    Unscaling,
    Upsampling,
    C2PSA,
    Count
};

class Layer;
class Optimizer;
class InputsSelection;

const EnumMap<LayerType>& layer_type_map();
const std::string& layer_type_to_string(LayerType);
LayerType string_to_layer_type(const std::string&);

std::unique_ptr<Layer> create_layer(const std::string& name);
std::unique_ptr<Optimizer> create_optimizer(const std::string& name);
std::unique_ptr<InputsSelection> create_inputs_selection(const std::string& name);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
