//   OpenNN: Open Neural Networks Library
//   www.opennn.net
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

	REGISTER(Layer, Scaling2d, "Scaling2d")
	REGISTER(Layer, Scaling3d, "Scaling3d")
	REGISTER(Layer, Scaling4d, "Scaling4d")

	template class Scaling<2>;
	template class Scaling<3>;
	template class Scaling<4>;

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
