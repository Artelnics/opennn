#pragma once

#include <cstdlib>
#include <string>

#include "opennn/neural_network/standard_networks.h"

namespace opennn::benchmark
{

inline Index configure_transformer_sdpa(Transformer& transformer)
{
    const char* value = getenv("OPENNN_SDPA_MIN");
    const Index minimum_sequence_length =
        value ? Index(stoll(value)) : Index(128);

    transformer.set_attention_sdpa_min_sequence_length(
        minimum_sequence_length);
    return minimum_sequence_length;
}

}
