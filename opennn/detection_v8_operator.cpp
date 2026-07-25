//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E T E C T I O N   V 8   O P E R A T O R   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "detection_v8_operator.h"
#include "tensor_operations.h"
#include "forward_propagation.h"
#include "back_propagation.h"
#ifdef OPENNN_HAS_CUDA
#include "kernel.cuh"
#include <cuda_runtime.h>
#endif

namespace opennn
{

void DetectionV8Operator::set(const Shape& input_shape)
{
    throw_if(input_shape.rank != 3,
             "DetectionV8Operator: input shape must be rank 3.");
    throw_if(input_shape[2] <= 4,
             "DetectionV8Operator: channels must be > 4 (need at least 1 class).");

    grid_size      = input_shape[0];
    grid_width     = input_shape[1];
    classes_number = input_shape[2] - 4;
}

void DetectionV8Operator::forward_propagate(ForwardPropagation& forward_propagation, size_t layer, bool)
{
    const TensorView& input  = get_input(forward_propagation, layer);
    TensorView&       output = get_output(forward_propagation, layer);

#ifdef OPENNN_HAS_CUDA
    if (input.is_cuda())
    {
        detection_v8_forward_cuda(input.shape[0], grid_size, grid_width, classes_number,
                                  input.as<float>(), output.as<float>());
        return;
    }
#endif

    copy(input, output);
    activation_forward(output, ActivationFunction::Sigmoid);
}

void DetectionV8Operator::back_propagate(ForwardPropagation& forward_propagation,
                                          BackPropagation&    back_propagation,
                                          size_t              layer) const
{
    const TensorView& output       = get_output(forward_propagation, layer);
    const TensorView& output_delta = get_output_delta(back_propagation, layer);
    TensorView&       input_delta  = get_input_delta(back_propagation, layer);

    if (input_delta.empty()) return;

#ifdef OPENNN_HAS_CUDA
    if (output_delta.is_cuda())
    {
        detection_v8_backward_cuda(output.shape[0], grid_size, grid_width, classes_number,
                                   output.as<float>(), output_delta.as<float>(),
                                   input_delta.as<float>());
        return;
    }
#endif

    copy(output_delta, input_delta);
    activation_backward(output, input_delta, ActivationFunction::Sigmoid);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
