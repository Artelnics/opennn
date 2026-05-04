//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N U M E R I C A L   D E R I V A T I V E S   ( T E S T   H E L P E R )
//
//   Test-side helpers for verifying analytical gradients against finite
//   differences. Not part of the runtime library.

#pragma once

#include "../opennn/pch.h"
#include "../opennn/loss.h"

namespace opennn
{

float   calculate_numerical_error(Loss& loss);
VectorR calculate_gradient(Loss& loss);
VectorR calculate_numerical_gradient(Loss& loss);
VectorR calculate_numerical_input_deltas(Loss& loss);
MatrixR calculate_numerical_hessian(Loss& loss);
MatrixR calculate_inverse_hessian(Loss& loss);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
