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

    using AdditionForwardPropagation3d = AdditionForwardPropagation<3>;
    using AdditionForwardPropagation4d = AdditionForwardPropagation<4>;

    using AdditionBackPropagation3d = AdditionBackPropagation<3>;
    using AdditionBackPropagation4d = AdditionBackPropagation<4>;

    REGISTER(Layer, Addition3d, "Addition3d")
    REGISTER(LayerForwardPropagation, AdditionForwardPropagation3d, "Addition3d")
    REGISTER(LayerBackPropagation, AdditionBackPropagation3d, "Addition3d")

    REGISTER(Layer, Addition4d, "Addition4d")
    REGISTER(LayerForwardPropagation, AdditionForwardPropagation4d, "Addition4d")
    REGISTER(LayerBackPropagation, AdditionBackPropagation4d, "Addition4d")

#ifdef OPENNN_CUDA

    using AdditionForwardPropagationCuda3d = AdditionForwardPropagationCuda<3>;
    using AdditionBackPropagationCuda3d = AdditionBackPropagationCuda<3>;

    using AdditionForwardPropagationCuda4d = AdditionForwardPropagationCuda<4>;
    using AdditionBackPropagationCuda4d = AdditionBackPropagationCuda<4>;

    REGISTER(LayerForwardPropagationCuda, AdditionForwardPropagationCuda3d, "Addition3d")
    REGISTER(LayerBackPropagationCuda, AdditionBackPropagationCuda3d, "Addition3d")

    REGISTER(LayerForwardPropagationCuda, AdditionForwardPropagationCuda4d, "Addition4d")
    REGISTER(LayerBackPropagationCuda, AdditionBackPropagationCuda4d, "Addition4d")

#endif

    template class Addition<3>;
    template class Addition<4>;

    template struct AdditionForwardPropagation<3>;
    template struct AdditionForwardPropagation<4>;

    template struct AdditionBackPropagation<3>;
    template struct AdditionBackPropagation<4>;

#ifdef OPENNN_CUDA

    template struct AdditionForwardPropagationCuda<3>;
    template struct AdditionForwardPropagationCuda<4>;

    template struct AdditionBackPropagationCuda<3>;
    template struct AdditionBackPropagationCuda<4>;

#endif

    // Linker fix: Ensures the static registration macros in this file are run.
    void reference_addition_layer() { }

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
