//   OpenNN: Open Neural Networks Library
//   www.opennnn.net
//
//   F L A T TEN   L A Y E R   R E G I S T R A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "registry.h"
#include "flatten_layer.h"

namespace opennn
{
	using Flatten2d = Flatten<2>;
	using Flatten3d = Flatten<3>;

	using FlattenForwardPropagation2d = FlattenForwardPropagation<2>;
	using FlattenBackPropagation2d = FlattenBackPropagation<2>;

	using FlattenForwardPropagation3d = FlattenForwardPropagation<3>;
	using FlattenBackPropagation3d = FlattenBackPropagation<3>;

	REGISTER(Layer, Flatten2d, "Flatten2d")
	REGISTER(LayerForwardPropagation, FlattenForwardPropagation2d, "Flatten2d")
	REGISTER(LayerBackPropagation, FlattenBackPropagation2d, "Flatten2d")

	REGISTER(Layer, Flatten3d, "Flatten3d")
	REGISTER(LayerForwardPropagation, FlattenForwardPropagation3d, "Flatten3d")
	REGISTER(LayerBackPropagation, FlattenBackPropagation3d, "Flatten3d")

#ifdef OPENNN_CUDA

	using FlattenForwardPropagationCuda2d = FlattenForwardPropagationCuda<2>;
	using FlattenBackPropagationCuda2d = FlattenBackPropagationCuda<2>;

	using FlattenForwardPropagationCuda3d = FlattenForwardPropagationCuda<3>;
	using FlattenBackPropagationCuda3d = FlattenBackPropagationCuda<3>;

	REGISTER(LayerForwardPropagationCuda, FlattenForwardPropagationCuda2d, "Flatten2d")
	REGISTER(LayerBackPropagationCuda, FlattenBackPropagationCuda2d, "Flatten2d")

	REGISTER(LayerForwardPropagationCuda, FlattenForwardPropagationCuda3d, "Flatten3d")
	REGISTER(LayerBackPropagationCuda, FlattenBackPropagationCuda3d, "Flatten3d")

#endif // OPENNN_CUDA

} // namespace opennn