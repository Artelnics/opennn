#ifndef KERNEL_RECURRENT_CUH
#define KERNEL_RECURRENT_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

inline constexpr int RNN_COPY_MAX_REGIONS = 16;

struct RnnCopySpec
{
    const void* src = nullptr;
    void*       dst = nullptr;
    int rows = 0;
    int cols = 0;
    int transpose = 0;
    int src_stride = 0;
    int dst_stride = 0;
    int src_bf16 = 0;
    int dst_bf16 = 0;
};

void rnn_copy_regions_cuda(const RnnCopySpec* specs, int count,
                           cudaStream_t stream = nullptr);

template<typename T>
void gather_time_slice_cuda(const Index batch, const Index time_steps,
                            const Index features, const Index t,
                            const T* src, T* dst);

// Writes only the (batch, features) slice at time step t of dst (batch, time_steps,
// features); the caller pre-zeroes dst when the other steps must read as zero.
template<typename T>
void scatter_time_slice_cuda(const Index batch, const Index time_steps,
                             const Index features, const Index t,
                             const T* src, T* dst);

// cuDNN's fastest recurrent layout is time-major while OpenNN tensors are
// batch-major. These kernels transpose only the two outer sequence axes and
// keep each feature vector contiguous.
template<typename T>
void batch_time_to_time_batch_cuda(const Index batch, const Index time_steps,
                                   const Index features, const T* src, T* dst);

template<typename T>
void batch_time_to_time_batch_padded_cuda(const Index batch, const Index time_steps,
                                          const Index features,
                                          const Index padded_features,
                                          const T* src, T* dst);

template<typename T>
void time_batch_to_batch_time_cuda(const Index batch, const Index time_steps,
                                   const Index features, const T* src, T* dst);

template<typename T>
void time_batch_to_batch_time_cropped_cuda(const Index batch, const Index time_steps,
                                           const Index features,
                                           const Index padded_features,
                                           const T* src, T* dst);

template<typename T>
void gather_time_major_slice_cuda(const Index batch, const Index time_steps,
                                  const Index features, const Index t,
                                  const T* src, T* dst);

template<typename T>
void scatter_time_major_slice_cuda(const Index batch, const Index time_steps,
                                   const Index features, const Index t,
                                   const T* src, T* dst);

template<typename T>
void rnn_step_fused_forward_cuda(const Index batch,
                                 const Index in_features,
                                 const Index out_features,
                                 const T* step_input,
                                 const T* prev_hidden,
                                 const T* W_in,
                                 const T* W_rec,
                                 const T* bias,
                                 T* step_hidden,
                                 T* derivs_or_null,
                                 const int activation_id);

template<typename T>
void rnn_step_fused_backward_pre_cuda(const Index batch,
                                      const Index out_features,
                                      const Index time_steps,
                                      const Index t,
                                      const T* output_delta,
                                      const T* next_carry,
                                      const T* activation_derivatives,
                                      T* delta);

#endif

#endif
