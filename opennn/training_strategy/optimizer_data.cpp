//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z E R   D A T A
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/training_strategy/optimizer.h"

#include "opennn/core/device_backend.h"
#include "opennn/core/memory_debug.h"

namespace opennn
{

namespace
{

Type get_slot_type(const vector<Type>& slot_types, size_t slot)
{
    return slot < slot_types.size() ? slot_types[slot] : Type::FP32;
}

}

void OptimizerData::set(const vector<Shape>& slot_shapes, Device device)
{
    set(slot_shapes, {}, device);
}

void OptimizerData::set(const vector<Shape>& slot_shapes,
                        const vector<Type>& slot_types,
                        Device device)
{
    Index total_bytes = 0;

    for (size_t slot = 0; slot < slot_shapes.size(); slot++)
        total_bytes = detail::checked_index_add(
            total_bytes,
            get_aligned_bytes(slot_shapes[slot].size(), get_slot_type(slot_types, slot)),
            "OptimizerData::set");

    data.resize_bytes(total_bytes, device);
    memory_debug::record("optimizer", "OptimizerData::data", total_bytes,
                         format("slots={}", slot_shapes.size()));

    if (total_bytes > 0)
    {
        if (device == Device::CUDA)
            opennn::device::set_zero_async(data.data(), total_bytes, device::get_compute_stream());
        else
            data.setZero();
    }

    views.clear();
    views.reserve(slot_shapes.size());

    uint8_t* cursor = data.as<uint8_t>();

    for (size_t slot = 0; slot < slot_shapes.size(); slot++)
    {
        const Shape& shape = slot_shapes[slot];
        const Type type = get_slot_type(slot_types, slot);

        if (shape.size() > 0)
        {
            views.emplace_back(cursor, shape, type, data.get_device());
            cursor += get_aligned_bytes(shape.size(), type);
        }
        else
        {
            views.emplace_back();
        }
    }
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
