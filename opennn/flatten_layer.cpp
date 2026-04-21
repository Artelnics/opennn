//   OpenNN: Open Neural Networks Library
//   www.opennn.net
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

	REGISTER(Layer, Flatten2d, "Flatten2d")
	REGISTER(Layer, Flatten3d, "Flatten3d")
	REGISTER(Layer, Flatten4d, "Flatten4d")

	template class Flatten<2>;
	template class Flatten<3>;
	template class Flatten<4>;
}

