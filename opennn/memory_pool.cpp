//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E M O R Y   P O O L
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "memory_pool.h"

namespace opennn
{

MemoryPoolPlan plan_memory_pool(const vector<MemoryPoolEntry>& entries)
{
    MemoryPoolPlan plan;
    plan.byte_offsets.assign(entries.size(), Index(-1));
    if (entries.empty()) return plan;

    Index last_execution_step = 0;
    for (const MemoryPoolEntry& entry : entries)
    {
        throw_if(entry.bytes < 0,
                 "plan_memory_pool: entry size cannot be negative.");
        throw_if(entry.first_step < 0 || entry.last_step < entry.first_step,
                 format("plan_memory_pool: invalid lifetime [{}, {}].",
                        entry.first_step, entry.last_step));
        last_execution_step = max(last_execution_step, entry.last_step);
    }

    vector<vector<size_t>> entries_starting_at(size_t(last_execution_step + 1));
    vector<vector<size_t>> entries_ending_at(size_t(last_execution_step + 1));

    for (size_t entry_index = 0; entry_index < entries.size(); ++entry_index)
    {
        if (entries[entry_index].bytes == 0) continue;
        entries_starting_at[size_t(entries[entry_index].first_step)].push_back(entry_index);
        entries_ending_at[size_t(entries[entry_index].last_step)].push_back(entry_index);
    }

    Index live_bytes = 0;
    vector<pair<Index, Index>> free_blocks = {{0, numeric_limits<Index>::max()}};

    for (Index execution_step = 0; execution_step <= last_execution_step; ++execution_step)
    {
        for (const size_t entry_index : entries_starting_at[size_t(execution_step)])
        {
            const Index bytes = entries[entry_index].bytes;
            live_bytes += bytes;

            auto block = ranges::find_if(free_blocks,
                                         [bytes](const auto& candidate)
                                         {
                                             return candidate.second >= bytes;
                                         });
            throw_if(block == free_blocks.end(),
                     "plan_memory_pool: address space exhausted.");

            plan.byte_offsets[entry_index] = block->first;
            if (block->second == bytes)
                free_blocks.erase(block);
            else
            {
                block->first += bytes;
                block->second -= bytes;
            }

            plan.peak_bytes = max(plan.peak_bytes,
                                  plan.byte_offsets[entry_index] + bytes);
        }

        plan.lower_bound_live_bytes = max(plan.lower_bound_live_bytes, live_bytes);

        for (const size_t entry_index : entries_ending_at[size_t(execution_step)])
        {
            const Index bytes = entries[entry_index].bytes;
            live_bytes -= bytes;

            auto block = ranges::lower_bound(free_blocks,
                                             plan.byte_offsets[entry_index],
                                             {},
                                             &pair<Index, Index>::first);
            block = free_blocks.insert(block, {plan.byte_offsets[entry_index], bytes});

            if (block + 1 != free_blocks.end()
                && block->first + block->second == (block + 1)->first)
            {
                block->second += (block + 1)->second;
                free_blocks.erase(block + 1);
            }

            if (block != free_blocks.begin()
                && (block - 1)->first + (block - 1)->second == block->first)
            {
                (block - 1)->second += block->second;
                free_blocks.erase(block);
            }
        }
    }

    return plan;
}

}
