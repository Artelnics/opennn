//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L L M   T E S T   H E L P E R S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <memory>
#include <vector>

#include "opennn/core/tensor_types.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/models/models.h"

namespace opennn_test
{

// The shape of a small Qwen3 built for a test, plus the prompt/decode lengths
// the incremental-decoding cases step through.
struct Dims
{
    opennn::Index seq, vocab, hidden, layers, q_heads, kv_heads, head_dim, intermediate;
    opennn::Index prompt1, decodes, prompt2;
};

inline constexpr Dims TINY { 16, 50, 32, 2, 4, 2, 8, 64, 5, 2, 8 };
inline constexpr Dims WIDE { 32, 50, 32, 2, 4, 2, 8, 64, 20, 2, 24 };

std::unique_ptr<opennn::Qwen3> make_qwen(const Dims& dims);

// Deterministic parameters: the same seed on both sides of any comparison.
void fill_parameters(opennn::NeuralNetwork& network);

std::unique_ptr<opennn::Qwen3> make_filled_qwen(const Dims& dims);

// One forward pass over `ids`, appended after `past` cached tokens.
void run(opennn::NeuralNetwork& network,
         opennn::ForwardPropagation& forward_propagation,
         std::vector<float>& window,
         const std::vector<opennn::Index>& ids,
         opennn::Index past);

// One row of the logits, brought back to the host and widened to fp32 if the
// network produced bf16.
std::vector<float> logits_row(const opennn::ForwardPropagation& forward_propagation,
                              opennn::Index position);

float max_difference(const std::vector<float>& a, const std::vector<float>& b);

// Round every parameter through bf16, so an fp32 run can be compared against
// a bf16 one without the rounding itself showing up as a difference.
void round_parameters_to_bf16(opennn::NeuralNetwork& network);

}
