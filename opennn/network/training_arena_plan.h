// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/core/memory_pool.h"

namespace opennn
{

class Loss;
struct BackPropagation;
struct ForwardPropagation;

// Cold metadata shared by forward and backward construction. Runtime tensor
// access never goes through this object.
class TrainingArenaPlan
{
public:
    TrainingArenaPlan(Index batch_size, Loss&, bool include_gradient);
    ~TrainingArenaPlan();

    TrainingArenaPlan(const TrainingArenaPlan&) = delete;
    TrainingArenaPlan& operator=(const TrainingArenaPlan&) = delete;

private:
    struct Impl;
    unique_ptr<Impl> impl;

    span<const MemoryPoolEntry> co_planned_lifetimes() const noexcept;
    bool uses_joint_gradient() const noexcept;
    void bind_offsets(span<const Index>);

    friend struct BackPropagation;
    friend struct ForwardPropagation;
};

}
