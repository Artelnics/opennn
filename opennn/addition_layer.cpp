//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A D D I T I O N   L A Y E R   3 D   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "addition_layer.h"

namespace opennn
{
    using Addition3d = Addition<3>;
    using Addition4d = Addition<4>;

    REGISTER(Layer, Addition3d, "Addition3d")
    REGISTER(Layer, Addition4d, "Addition4d")

    template class Addition<3>;
    template class Addition<4>;
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
