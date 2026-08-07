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

namespace
{

Index lowest_free_offset(const vector<pair<Index, Index>>& occupied, Index bytes)
{
    Index offset = 0;
    for (const auto& [begin, end] : occupied)
    {
        throw_if(offset > numeric_limits<Index>::max() - bytes,
                 "memory pool: address space exhausted.");
        if (begin >= offset + bytes) break;
        if (end > offset) offset = end;
    }

    throw_if(offset > numeric_limits<Index>::max() - bytes,
             "memory pool: address space exhausted.");
    return offset;
}

}

MemoryPoolPlan plan_memory_pool(const vector<MemoryPoolEntry>& entries,
                                MemoryPoolStrategy strategy)
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
                 "plan_memory_pool: invalid lifetime [{}, {}].",
                        entry.first_step, entry.last_step);
        last_execution_step = max(last_execution_step, entry.last_step);
    }

    vector<Index> live_bytes_delta(size_t(last_execution_step + 2), Index(0));
    for (const MemoryPoolEntry& entry : entries)
    {
        live_bytes_delta[size_t(entry.first_step)] += entry.bytes;
        live_bytes_delta[size_t(entry.last_step + 1)] -= entry.bytes;
    }

    Index live_bytes = 0;
    for (Index step = 0; step <= last_execution_step; ++step)
    {
        live_bytes += live_bytes_delta[size_t(step)];
        plan.lower_bound_live_bytes =
            max(plan.lower_bound_live_bytes, live_bytes);
    }

    vector<size_t> allocation_order(entries.size());
    iota(allocation_order.begin(), allocation_order.end(), size_t(0));
    ranges::sort(allocation_order, [&](size_t left, size_t right)
    {
        if (strategy == MemoryPoolStrategy::Chronological)
            return entries[left].first_step != entries[right].first_step
                ? entries[left].first_step < entries[right].first_step
                : left < right;
        if (entries[left].bytes != entries[right].bytes)
            return entries[left].bytes > entries[right].bytes;
        if (entries[left].first_step != entries[right].first_step)
            return entries[left].first_step < entries[right].first_step;
        if (entries[left].last_step != entries[right].last_step)
            return entries[left].last_step > entries[right].last_step;
        return left < right;
    });

    vector<size_t> placed_entries;
    placed_entries.reserve(entries.size());

    const auto lifetimes_overlap = [&](size_t left, size_t right)
    {
        return entries[left].first_step <= entries[right].last_step
            && entries[right].first_step <= entries[left].last_step;
    };

    for (const size_t entry_index : allocation_order)
    {
        const Index bytes = entries[entry_index].bytes;
        if (bytes == 0) continue;

        vector<pair<Index, Index>> occupied_blocks;
        occupied_blocks.reserve(placed_entries.size());
        for (const size_t placed_index : placed_entries)
        {
            if (!lifetimes_overlap(entry_index, placed_index)) continue;
            const Index begin = plan.byte_offsets[placed_index];
            occupied_blocks.push_back(
                {begin, begin + entries[placed_index].bytes});
        }
        ranges::sort(occupied_blocks);

        const Index offset = lowest_free_offset(occupied_blocks, bytes);
        plan.byte_offsets[entry_index] = offset;
        plan.peak_bytes = max(plan.peak_bytes, offset + bytes);
        placed_entries.push_back(entry_index);
    }

    return plan;
}

Index find_memory_pool_overlay(const vector<MemoryPoolEntry>& entries,
                               const MemoryPoolPlan& plan,
                               const Index bytes,
                               const Index first_step,
                               const Index second_step)
{
    throw_if(entries.size() != plan.byte_offsets.size(),
             "find_memory_pool_overlay: entries and offsets must have equal size.");
    if (bytes <= 0 || bytes > plan.peak_bytes) return Index(-1);

    const auto live_at = [](const MemoryPoolEntry& entry, Index step)
    {
        return entry.first_step <= step && step <= entry.last_step;
    };

    vector<pair<Index, Index>> occupied;
    occupied.reserve(entries.size());
    for (size_t i = 0; i < entries.size(); ++i)
    {
        const MemoryPoolEntry& entry = entries[i];
        if (entry.bytes == 0) continue;
        if (!live_at(entry, first_step) && !live_at(entry, second_step)) continue;
        occupied.push_back({plan.byte_offsets[i], plan.byte_offsets[i] + entry.bytes});
    }
    ranges::sort(occupied);

    const Index offset = lowest_free_offset(occupied, bytes);
    return offset + bytes <= plan.peak_bytes ? offset : Index(-1);
}

}
