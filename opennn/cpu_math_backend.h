//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C P U   M A T H   B A C K E N D
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

namespace opennn
{

enum class ActivationFunction;
struct TensorView;

}

namespace opennn::cpu_math
{

// Fast MKL VML path (CPU). Off by default; set from code (no environment var).
void set_mkl_fast_vml(bool);
bool mkl_fast_vml_enabled();

bool try_activation_forward(TensorView&, ActivationFunction);

bool try_linear_forward(const TensorView& input,
                        const TensorView& weights,
                        const TensorView& bias,
                        TensorView& output,
                        bool fuse_relu);

}
