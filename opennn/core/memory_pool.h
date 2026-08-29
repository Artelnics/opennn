//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E M O R Y   P O O L
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/opennn_types.h"

namespace opennn
{

enum class MemoryPoolStrategy
{
    Chronological,
    Compact,
    ChronologicalLargestFirst,
    EarliestEndFirst,
    LatestEndFirst,
    LongestLifetimeFirst,
    ShortestLifetimeFirst
};

constexpr Index backward_step(Index layers_number, Index layer) noexcept
{
    return 2 * layers_number - 1 - layer;
}

struct MemoryPoolEntry
{
    Index bytes = 0;
    Index first_step = 0;
    Index last_step = 0;
};

struct MemoryPoolPlan
{
    vector<Index> byte_offsets;
    Index peak_bytes = 0;
    Index lower_bound_live_bytes = 0;

    Index fragmentation_bytes() const noexcept
    {
        return peak_bytes - lower_bound_live_bytes;
    }
};

MemoryPoolPlan plan_memory_pool(
    const vector<MemoryPoolEntry>&,
    MemoryPoolStrategy = MemoryPoolStrategy::Chronological);

MemoryPoolPlan plan_memory_pool_best(const vector<MemoryPoolEntry>&);

Index find_memory_pool_overlay(
    const vector<MemoryPoolEntry>&,
    const MemoryPoolPlan&,
    Index bytes,
    Index first_step,
    Index second_step);

}
