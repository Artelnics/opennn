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
    using DenseForwardPropagation2d = DenseForwardPropagation<2>;
    using DenseBackPropagation2d = DenseBackPropagation<2>;

    using Dense3d = Dense<3>;
    using DenseForwardPropagation3d = DenseForwardPropagation<3>;
    using DenseBackPropagation3d = DenseBackPropagation<3>;

    using DenseBackPropagationLM2d = DenseBackPropagationLM;

#ifdef OPENNN_CUDA
    using DenseForwardPropagationCuda2d = DenseForwardPropagationCuda<2>;
    using DenseBackPropagationCuda2d = DenseBackPropagationCuda<2>;
    using DenseForwardPropagationCuda3d = DenseForwardPropagationCuda<3>;
    using DenseBackPropagationCuda3d = DenseBackPropagationCuda<3>;
#endif

    REGISTER(Layer, Dense2d, "Dense2d")
    REGISTER(LayerForwardPropagation, DenseForwardPropagation2d, "Dense2d")
    REGISTER(LayerBackPropagation, DenseBackPropagation2d, "Dense2d")

    if (get_output_dimensions()[0] == 1 && new_activation_function == "Softmax")
        activation_function = "Logistic";

#ifdef OPENNN_CUDA
    REGISTER(LayerForwardPropagationCuda, DenseForwardPropagationCuda2d, "Dense2d")
    REGISTER(LayerBackPropagationCuda, DenseBackPropagationCuda2d, "Dense2d")
#endif

    REGISTER(Layer, Dense3d, "Dense3d")
    REGISTER(LayerForwardPropagation, DenseForwardPropagation3d, "Dense3d")
    REGISTER(LayerBackPropagation, DenseBackPropagation3d, "Dense3d")

#ifdef OPENNN_CUDA
    REGISTER(LayerForwardPropagationCuda, DenseForwardPropagationCuda3d, "Dense3d")
    REGISTER(LayerBackPropagationCuda, DenseBackPropagationCuda3d, "Dense3d")
#endif

    template class Dense<2>;
    template class Dense<3>;

    template struct DenseForwardPropagation<2>;
    template struct DenseForwardPropagation<3>;

    template struct DenseBackPropagation<2>;
    template struct DenseBackPropagation<3>;

#ifdef OPENNN_CUDA
    template struct DenseForwardPropagationCuda<2>;
    template struct DenseForwardPropagationCuda<3>;
    template struct DenseBackPropagationCuda<2>;
    template struct DenseBackPropagationCuda<3>;
#endif

    struct DenseBackPropagationLM;

    void reference_dense_layer() { }

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software Foundation.
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
