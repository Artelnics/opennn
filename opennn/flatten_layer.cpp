//   OpenNN: Open Neural Networks Library
//   www.opennnn.net
//
//   F L A T T E N   L A Y E R   R E G I S T R A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "flatten_layer.h"

namespace opennn
{
	using Flatten2d = Flatten<2>;
	using Flatten3d = Flatten<3>;
	using Flatten4d = Flatten<4>;

	using FlattenForwardPropagation2d = FlattenForwardPropagation<2>;
	using FlattenBackPropagation2d = FlattenBackPropagation<2>;

	using FlattenForwardPropagation3d = FlattenForwardPropagation<3>;
	using FlattenBackPropagation3d = FlattenBackPropagation<3>;

	using FlattenForwardPropagation4d = FlattenForwardPropagation<4>;
	using FlattenBackPropagation4d = FlattenBackPropagation<4>;

	REGISTER(Layer, Flatten2d, "Flatten2d")
	REGISTER(LayerForwardPropagation, FlattenForwardPropagation2d, "Flatten2d")
	REGISTER(LayerBackPropagation, FlattenBackPropagation2d, "Flatten2d")

	REGISTER(Layer, Flatten3d, "Flatten3d")
	REGISTER(LayerForwardPropagation, FlattenForwardPropagation3d, "Flatten3d")
	REGISTER(LayerBackPropagation, FlattenBackPropagation3d, "Flatten3d")

	REGISTER(Layer, Flatten4d, "Flatten4d")
	REGISTER(LayerForwardPropagation, FlattenForwardPropagation4d, "Flatten4d")
	REGISTER(LayerBackPropagation, FlattenBackPropagation4d, "Flatten4d")

#ifdef OPENNN_CUDA

	using FlattenForwardPropagationCuda2d = FlattenForwardPropagationCuda<2>;
	using FlattenBackPropagationCuda2d = FlattenBackPropagationCuda<2>;

	using FlattenForwardPropagationCuda3d = FlattenForwardPropagationCuda<3>;
	using FlattenBackPropagationCuda3d = FlattenBackPropagationCuda<3>;

	using FlattenForwardPropagationCuda4d = FlattenForwardPropagationCuda<4>;
	using FlattenBackPropagationCuda4d = FlattenBackPropagationCuda<4>;

	REGISTER(LayerForwardPropagationCuda, FlattenForwardPropagationCuda2d, "Flatten2d")
	REGISTER(LayerBackPropagationCuda, FlattenBackPropagationCuda2d, "Flatten2d")

	REGISTER(LayerForwardPropagationCuda, FlattenForwardPropagationCuda3d, "Flatten3d")
	REGISTER(LayerBackPropagationCuda, FlattenBackPropagationCuda3d, "Flatten3d")

	REGISTER(LayerForwardPropagationCuda, FlattenForwardPropagationCuda4d, "Flatten4d")
	REGISTER(LayerBackPropagationCuda, FlattenBackPropagationCuda4d, "Flatten4d")

#endif // OPENNN_CUDA

	template class Flatten<2>;
	template class Flatten<3>;
	template class Flatten<4>;

	template struct FlattenForwardPropagation<2>;
	template struct FlattenForwardPropagation<3>;
	template struct FlattenForwardPropagation<4>;

	template struct FlattenBackPropagation<2>;
	template struct FlattenBackPropagation<3>;
	template struct FlattenBackPropagation<4>;

#ifdef OPENNN_CUDA

	template struct FlattenForwardPropagationCuda<2>;
	template struct FlattenForwardPropagationCuda<3>;
	template struct FlattenForwardPropagationCuda<4>;

	template struct FlattenBackPropagationCuda<2>;
	template struct FlattenBackPropagationCuda<3>;
	template struct FlattenBackPropagationCuda<4>;

#endif // OPENNN_CUDA

	// Linker fix: Ensures the static registration macros in this file are run.
	void reference_flatten_layer() { }

} // namespace opennn
