//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E M O R Y   D E B U G   U T I L I T I E S

#pragma once

#include "opennn_types.h"
#include "configuration.h"

namespace opennn::memory_debug
{

bool enabled();

void reset();

void record(const string&,
            const string&,
            Index,
            const string& note = {});

void print(ostream&);

}

