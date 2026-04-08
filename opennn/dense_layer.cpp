//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E N S E   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "dense_layer.h"

namespace opennn
{
    using Dense2d = Dense<2>;
    using Dense3d = Dense<3>;

    REGISTER(Layer, Dense2d, "Dense2d")
    REGISTER(Layer, Dense3d, "Dense3d")

    template class Dense<2>;
    template class Dense<3>;

    void reference_dense_layer() { }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
