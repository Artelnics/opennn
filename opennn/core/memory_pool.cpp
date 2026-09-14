// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/core/memory_pool.h"

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
        offset = std::max(offset, end);
    }

    throw_if(offset > numeric_limits<Index>::max() - bytes,
             "memory pool: address space exhausted.");
    return offset;
}

Index memory_pool_live_bytes_lower_bound(const vector<MemoryPoolEntry>& entries)
{
    Index last_step = 0;
    for(const MemoryPoolEntry& entry : entries)
    {
        throw_if(entry.bytes < 0, "plan_memory_pool: entry size cannot be negative.");
        throw_if(entry.first_step < 0 || entry.last_step < entry.first_step,
                 "plan_memory_pool: invalid lifetime [{}, {}].",
                 entry.first_step, entry.last_step);
        last_step = max(last_step, entry.last_step);
    }

    vector<Index> delta(size_t(last_step + 2), 0);
    for(const MemoryPoolEntry& entry : entries)
    {
        delta[size_t(entry.first_step)] += entry.bytes;
        delta[size_t(entry.last_step + 1)] -= entry.bytes;
    }

    Index live = 0;
    Index lower_bound = 0;
    for(Index step = 0; step <= last_step; ++step)
    {
        live += delta[size_t(step)];
        lower_bound = max(lower_bound, live);
    }
    return lower_bound;
}

bool memory_pool_entry_less(const vector<MemoryPoolEntry>& entries,
                            MemoryPoolStrategy strategy,
                            size_t left, size_t right)
{
    const MemoryPoolEntry& a = entries[left];
    const MemoryPoolEntry& b = entries[right];

    if(strategy == MemoryPoolStrategy::Chronological)
        return a.first_step != b.first_step ? a.first_step < b.first_step : left < right;

    if(strategy == MemoryPoolStrategy::Compact)
    {
        if(a.bytes != b.bytes) return a.bytes > b.bytes;
        if(a.first_step != b.first_step) return a.first_step < b.first_step;
        if(a.last_step != b.last_step) return a.last_step > b.last_step;
        return left < right;
    }

    if(strategy == MemoryPoolStrategy::ChronologicalLargestFirst)
    {
        if(a.first_step != b.first_step) return a.first_step < b.first_step;
        if(a.bytes != b.bytes) return a.bytes > b.bytes;
        if(a.last_step != b.last_step) return a.last_step > b.last_step;
        return left < right;
    }

    if(is_one_of(strategy, MemoryPoolStrategy::EarliestEndFirst,
                 MemoryPoolStrategy::LatestEndFirst))
    {
        const bool earliest = strategy == MemoryPoolStrategy::EarliestEndFirst;
        if(a.last_step != b.last_step)
            return earliest ? a.last_step < b.last_step : a.last_step > b.last_step;
        if(a.bytes != b.bytes) return earliest ? a.bytes < b.bytes : a.bytes > b.bytes;
        if(a.first_step != b.first_step) return a.first_step < b.first_step;
        return left < right;
    }

    const Index a_lifetime = a.last_step - a.first_step;
    const Index b_lifetime = b.last_step - b.first_step;
    if(a_lifetime != b_lifetime)
        return strategy == MemoryPoolStrategy::LongestLifetimeFirst
            ? a_lifetime > b_lifetime : a_lifetime < b_lifetime;
    if(a.bytes != b.bytes) return a.bytes > b.bytes;
    if(a.first_step != b.first_step) return a.first_step < b.first_step;
    if(a.last_step != b.last_step) return a.last_step > b.last_step;
    return left < right;
}

}

MemoryPoolPlan plan_memory_pool(const vector<MemoryPoolEntry>& entries,
                                MemoryPoolStrategy strategy)
{
    MemoryPoolPlan plan;
    plan.byte_offsets.assign(entries.size(), -1);

    if(entries.empty()) return plan;

    plan.lower_bound_live_bytes = memory_pool_live_bytes_lower_bound(entries);

    vector<size_t> allocation_order(entries.size());
    iota(allocation_order.begin(), allocation_order.end(), 0);

    ranges::sort(allocation_order, [&](size_t left, size_t right) {
        return memory_pool_entry_less(entries, strategy, left, right);
    });

    vector<size_t> placed_entries;
    placed_entries.reserve(entries.size());

    for(const size_t entry_index : allocation_order)
    {
        const MemoryPoolEntry& entry = entries[entry_index];

        if(entry.bytes == 0) continue;

        vector<pair<Index, Index>> occupied_blocks;
        occupied_blocks.reserve(placed_entries.size());

        for(const size_t placed_index : placed_entries)
        {
            const MemoryPoolEntry& placed_entry = entries[placed_index];

            if(entry.first_step > placed_entry.last_step ||
               placed_entry.first_step > entry.last_step)
            {
                continue;
            }

            const Index begin = plan.byte_offsets[placed_index];
            occupied_blocks.emplace_back(begin, begin + placed_entry.bytes);
        }

        ranges::sort(occupied_blocks);

        const Index offset = lowest_free_offset(occupied_blocks, entry.bytes);

        plan.byte_offsets[entry_index] = offset;
        plan.peak_bytes = max(plan.peak_bytes, offset + entry.bytes);

        placed_entries.push_back(entry_index);
    }

    return plan;
}

MemoryPoolPlan plan_memory_pool_best(const vector<MemoryPoolEntry>& entries)
{
    constexpr std::array strategies{
        MemoryPoolStrategy::Chronological,
        MemoryPoolStrategy::Compact,
        MemoryPoolStrategy::ChronologicalLargestFirst,
        MemoryPoolStrategy::EarliestEndFirst,
        MemoryPoolStrategy::LatestEndFirst,
        MemoryPoolStrategy::LongestLifetimeFirst,
        MemoryPoolStrategy::ShortestLifetimeFirst
    };

    MemoryPoolPlan best = plan_memory_pool(entries, strategies.front());

    for(const MemoryPoolStrategy strategy : strategies | views::drop(1))
    {
        MemoryPoolPlan candidate = plan_memory_pool(entries, strategy);
        if(candidate.peak_bytes < best.peak_bytes)
            best = std::move(candidate);
    }

    return best;
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
