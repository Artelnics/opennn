//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E M O R Y   D E B U G   U T I L I T I E S

#pragma once

#include "opennn/core/opennn_types.h"
#include "opennn/core/configuration.h"

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
