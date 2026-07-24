//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E M O R Y   P O O L
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn_types.h"

namespace opennn
{

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

// First-fit interval allocator shared by forward inference activations and
// backward deltas. Entries starting at a step are allocated before entries
// ending at that step are released, so producer/consumer buffers that meet at
// one execution step can never alias.
MemoryPoolPlan plan_memory_pool(const vector<MemoryPoolEntry>&);

}
