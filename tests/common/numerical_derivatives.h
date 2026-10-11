// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, S.L. and Roberto Lopez

//   Test-side helpers for verifying analytical gradients against finite
//   differences. Not part of the runtime library.

#pragma once

#include "opennn/core/pch.h"
#include "opennn/training/loss.h"

namespace opennn
{

// Central-difference step for a value: 2^-10, scaled by |x| beyond 1.
float   central_difference_step(float x);

float   calculate_numerical_error(Loss& loss);
VectorR calculate_gradient(Loss& loss);
VectorR calculate_numerical_gradient(Loss& loss);

}
