// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#include "opennn/training/optimizer.h"

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

    data.setZero();

    views.assign(slot_shapes.size(), TensorView{});

    uint8_t* cursor = data.as<uint8_t>();

    for (size_t slot = 0; slot < slot_shapes.size(); slot++)
    {
        const Shape& shape = slot_shapes[slot];
        if (shape.size() == 0) continue;
        const Type type = get_slot_type(slot_types, slot);

        views[slot] = TensorView(cursor, shape, type, data.get_device());
        cursor += get_aligned_bytes(shape.size(), type);
    }
}

}
