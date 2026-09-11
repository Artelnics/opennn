// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

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
