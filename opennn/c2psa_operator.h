//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C 2 P S A   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

struct C2PSAOperator : Operator
{
    Index h = 0, w = 0, channels = 0;

    TensorView Wq, Wk, Wv, Wout;
    TensorView dWq, dWk, dWv, dWout;

    void set(Index new_h, Index new_w, Index new_channels);

    vector<TensorSpec> parameter_specs() const override;
    void link_parameters(span<const TensorView>) override;
    void link_gradients (span<const TensorView>) override;
    void set_parameters_random() override;
    void set_parameters_glorot() override;

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;

private:
    // CPU scratch retained across forward→backward for the GPU training path.
    // Single-threaded network computation makes this safe.
    mutable vector<float> cpu_x, cpu_xa, cpu_Q, cpu_K, cpu_V, cpu_A, cpu_cat, cpu_out;
    mutable vector<float> cpu_Wq, cpu_Wk, cpu_Wv, cpu_Wout;

    // GPU scratch: attn_v + backward temporaries in one flat allocation.
    mutable Buffer gpu_scratch;

    void run_fwd_cpu(Index B, float scale) const;
    void run_bwd_cpu(const float* dout, float* din,
                     float* dWq_out, float* dWk_out,
                     float* dWv_out, float* dWout_out,
                     Index B, float scale) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
