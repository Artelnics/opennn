// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#pragma once

#include "opennn/core/opennn_types.h"

namespace opennn
{
struct MemoryPoolEntry;
}

namespace opennn::memory_debug
{

bool enabled();

void reset();

void record(const string&,
            const string&,
            Index,
            const string& note = {});

void record_pool_lifetimes(const string&,
                           const vector<MemoryPoolEntry>&,
                           const string&);

void print(ostream&);

}
