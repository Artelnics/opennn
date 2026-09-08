#ifndef KERNEL_DETECTION_CUH
#define KERNEL_DETECTION_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

void detection_forward_cuda(const Index batch_size,
                            const Index grid_size,
                            const Index boxes_per_cell,
                            const Index classes_number,
                            const int class_activation,
                            const float* anchors,
                            const float* input,
                            float* output);

void detection_backward_cuda(const Index batch_size,
                             const Index grid_size,
                             const Index boxes_per_cell,
                             const Index classes_number,
                             const int class_activation,
                             const float* output,
                             const float* output_delta,
                             float* input_delta);

void detection_v8_forward_cuda(Index batch_size,
                               Index grid_size,
                               Index grid_width,
                               Index classes_number,
                               Index reg_max,
                               const float* input,
                               float* output);

void detection_v8_backward_cuda(Index batch_size,
                                Index grid_size,
                                Index grid_width,
                                Index classes_number,
                                Index reg_max,
                                const float* output,
                                const float* output_delta,
                                float* input_delta);

#endif

#endif
