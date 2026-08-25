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

void OptimizerData::set(const vector<Shape>& slot_shapes, Device device)
{
    const Index total_bytes = get_aligned_bytes(slot_shapes, Type::FP32);

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

    for (const Shape& shape : slot_shapes)
    {
        if (shape.size() > 0)
        {
            views.emplace_back(cursor, shape, Type::FP32, data.get_device());
            cursor += get_aligned_bytes(shape.size(), Type::FP32);
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
