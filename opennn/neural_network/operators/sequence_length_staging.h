//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S E Q U E N C E   L E N G T H   S T A G I N G   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/device_backend.h"
#include "opennn/core/tensor_types.h"

namespace opennn
{

// Uploads host-side sequence lengths without blocking the compute stream.
// The pinned-memory ring prevents a slot from being reused while its previous
// asynchronous copy is still in flight.
class SequenceLengthStaging
{
public:

    SequenceLengthStaging() = default;
    SequenceLengthStaging(const SequenceLengthStaging&) = delete;
    SequenceLengthStaging& operator=(const SequenceLengthStaging&) = delete;

    ~SequenceLengthStaging() { if (pinned) device::deallocate_pinned_host(pinned); }

    // A pinned host slot for `count` ints, free to overwrite: the copy that last
    // read it has been waited on. Fill it, issue the copy, then mark_copied().
    // For callers who own the device side -- the SDPA graph keeps its two length
    // tensors inside its cache entry, and fills one slot for both.
    int* acquire(const Index count)
    {
        if (count > capacity)
        {
            device::synchronize(device::get_compute_stream());
            if (pinned) device::deallocate_pinned_host(pinned);
            pinned = static_cast<int*>(device::allocate_pinned_host(slots * count * Index(sizeof(int))));
            capacity = count;
        }

        slot = (slot + 1) % slots;
        if (copy_done[slot]) device::synchronize_event(copy_done[slot]);
        else copy_done[slot].create();

        return pinned + slot * capacity;
    }

    void mark_copied() { device::record_event(copy_done[slot], device::get_compute_stream()); }

    // The whole errand for callers who just want the lengths on the device:
    // null for an empty batch, otherwise a pointer valid until this instance
    // stages again.
    const int* stage(const vector<Index>& lengths)
    {
        const Index batch_size = Index(lengths.size());
        if (batch_size == 0) return nullptr;

        int* const host_slot = acquire(batch_size);

        const Index bytes = batch_size * Index(sizeof(int));
        device_lengths.grow_to(bytes);

        ranges::transform(lengths, host_slot, [](const Index length) { return int(length); });

        device::copy_async(device_lengths.data(), host_slot, bytes,
                           device::CopyKind::HostToDevice, device::get_compute_stream());
        mark_copied();

        return device_lengths.as<int>();
    }

private:

    static constexpr int slots = 4;

    int* pinned = nullptr;
    Index capacity = 0;
    int slot = slots - 1;
    CudaEvent copy_done[slots];

    // Only stage() uses this; acquire() callers bring their own.
    Buffer device_lengths{Device::CUDA};
};

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
