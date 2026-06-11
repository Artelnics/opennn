//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C P U   M A T H   B A C K E N D
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_operations.h"

namespace opennn::cpu_math
{

bool try_activation_forward(TensorView&, ActivationFunction);

bool try_linear_forward(const TensorView& input,
                        const TensorView& weights,
                        const TensorView& bias,
                        TensorView& output,
                        cublasLtEpilogue_t epilogue);

}
