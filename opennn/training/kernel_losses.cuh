#ifndef KERNEL_LOSSES_CUH
#define KERNEL_LOSSES_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

template<typename TIn, typename TOut>
void scaled_diff_cuda_typed(const Index n, const TIn* input, const float* target,
                            float scale, TOut* output);

template<typename TIn, typename TOut>
void mean_squared_error_metrics_gradient_cuda(const Index n, const Index batch,
                                              const TIn* input, const float* target,
                                              TOut* delta, float* error_sum);

template<typename T>
void mean_absolute_error_gradient_cuda(const Index, T*, const float*, const T*, float);

template<typename T>
void binary_cross_entropy_cuda(const Index, float*, const float*, const T*, const float);

template<typename T>
void binary_cross_entropy_gradient_cuda(const Index, T*, const float*, const T*, const float, const float);

template<typename T>
void categorical_cross_entropy_cuda(const Index, float*, const float*, const T*, const float);

template<typename T>
void categorical_cross_entropy_gradient_cuda(const Index, T*, const float*, const T*, const float);

template<typename T>
void weighted_squared_error_cuda(const Index, float*, const float*, const T*, const float, const float);

template<typename T>
void weighted_squared_error_gradient_cuda(const Index, T*, const float*, const T*, const float, const float, const float);

template<typename T>
void cross_entropy_3d_multiple_forward_cuda(const Index, const int, const T*, const float*, float*, float*, float*);

template<typename T>
void cross_entropy_3d_metrics_cuda(const Index, const int, const T*, const float*, float*);

template<typename T>
void cross_entropy_3d_multiple_backward_cuda(const Index, const int, const T*, const float*, T*, const float scale,
                                             const float* active_count_device = nullptr);

void accumulate_scaled_metric_cuda(const float*, float, float*);

void accumulate_cross_entropy_3d_metrics_cuda(const float*, float*, float*);

template<typename T>
void l1_gradient_cuda(const Index, T*, const T*, const float);

void yolo_error_cuda(const float* output, const float* target, float* error_accumulator,
                     int batch, int grid, int boxes_per_cell, int values_per_box,
                     int classes_number, int sigmoid_classes,
                     float lambda_giou, float lambda_noobj, float lambda_class,
                     float focal_gamma, float obj_focal_gamma);

void yolo_gradient_cuda(const float* output, const float* target, float* delta,
                        int batch, int grid, int boxes_per_cell, int values_per_box,
                        int classes_number, int sigmoid_classes, float inv_batch,
                        float lambda_giou, float lambda_noobj, float lambda_class,
                        float focal_gamma, float obj_focal_gamma);

void yolo_assemble_head_target_cuda(const float* target_flat, float* head_target,
                                    Index batch, Index per_sample_floats,
                                    Index head_offset, Index head_floats);

#endif

#endif
