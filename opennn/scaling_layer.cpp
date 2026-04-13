//   OpenNN: Open Neural Networks Library
//   www.opennnn.net
//
//   S C A L I N G   L A Y E R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "scaling_layer.h"

namespace opennn
{
	using Scaling2d = Scaling<2>;
	using Scaling3d = Scaling<3>;
	using Scaling4d = Scaling<4>;

	using ScalingForwardPropagation2d = ScalingForwardPropagation<2>;
	using ScalingForwardPropagation3d = ScalingForwardPropagation<3>;
	using ScalingForwardPropagation4d = ScalingForwardPropagation<4>;

	REGISTER(Layer, Scaling2d, "Scaling2d")
	REGISTER(LayerForwardPropagation, ScalingForwardPropagation2d, "Scaling2d")

	REGISTER(Layer, Scaling3d, "Scaling3d")
	REGISTER(LayerForwardPropagation, ScalingForwardPropagation3d, "Scaling3d")

	REGISTER(Layer, Scaling4d, "Scaling4d")
	REGISTER(LayerForwardPropagation, ScalingForwardPropagation4d, "Scaling4d")

#ifdef OPENNN_CUDA

	using ScalingForwardPropagationCuda2d = ScalingForwardPropagationCuda<2>;
	using ScalingForwardPropagationCuda3d = ScalingForwardPropagationCuda<3>;
	using ScalingForwardPropagationCuda4d = ScalingForwardPropagationCuda<4>;

	REGISTER(LayerForwardPropagationCuda, ScalingForwardPropagationCuda2d, "Scaling2d")
	REGISTER(LayerForwardPropagationCuda, ScalingForwardPropagationCuda3d, "Scaling3d")
	REGISTER(LayerForwardPropagationCuda, ScalingForwardPropagationCuda4d, "Scaling4d")

#endif // OPENNN_CUDA

	template class Scaling<2>;
	template class Scaling<3>;
	template class Scaling<4>;

	template struct ScalingForwardPropagation<2>;
	template struct ScalingForwardPropagation<3>;
	template struct ScalingForwardPropagation<4>;

#ifdef OPENNN_CUDA

	template struct ScalingForwardPropagationCuda<2>;
	template struct ScalingForwardPropagationCuda<3>;
	template struct ScalingForwardPropagationCuda<4>;

#endif // OPENNN_CUDA

	void reference_scaling_layer() { }

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
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
