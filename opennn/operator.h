//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_types.h"
#include "tensor_operations.h"
#include "enum_map.h"
#include "forward_propagation.h"
#include "back_propagation.h"

namespace opennn
{

class Json;
class JsonWriter;

template<typename Handle>
struct CudnnDescriptor
{
    Handle handle = nullptr;
#ifdef OPENNN_HAS_CUDA
    cudnnStatus_t (*deleter)(Handle) = nullptr;
#else
    void (*deleter)(Handle) = nullptr;
#endif

    CudnnDescriptor() = default;

    CudnnDescriptor(CudnnDescriptor&& other) noexcept
        : handle(other.handle), deleter(other.deleter)
    {
        other.handle = nullptr;
        other.deleter = nullptr;
    }

    CudnnDescriptor& operator=(CudnnDescriptor&& other) noexcept
    {
        if (this != &other)
        {
            reset();
            handle = other.handle;
            deleter = other.deleter;
            other.handle = nullptr;
            other.deleter = nullptr;
        }
        return *this;
    }

    CudnnDescriptor(const CudnnDescriptor&) = delete;
    CudnnDescriptor& operator=(const CudnnDescriptor&) = delete;

    ~CudnnDescriptor() { reset(); }

    void reset()
    {
        if (handle && deleter) deleter(handle);
        handle = nullptr;
        deleter = nullptr;
    }

    operator Handle() const { return handle; }
    explicit operator bool() const { return handle != nullptr; }
};

struct Operator
{
    virtual ~Operator() = default;

    virtual vector<TensorSpec> parameter_specs() const { return {}; }
    virtual vector<TensorSpec> state_specs()     const { return {}; }

    virtual void link_parameters(span<const TensorView>) {}
    virtual void link_gradients (span<const TensorView>) {}
    virtual void link_states    (span<const TensorView>) {}

    virtual void set_parameters_random() {}
    virtual void set_parameters_glorot() {}

    virtual void forward_propagate(ForwardPropagation&, size_t, bool) {}
    virtual void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const {}

    virtual void to_JSON  (JsonWriter&) const {}
    virtual void from_JSON(const Json*)       {}
    virtual void load_state_from_JSON(const Json*) {}

    virtual void destroy_cuda() {}

    vector<size_t> input_slots = {0};
    vector<size_t> output_slots = {1};

    vector<size_t> input_delta_slots = {1};
    vector<size_t> output_delta_slots = {0};

    TensorView& get_input(ForwardPropagation& fp, size_t layer, size_t slot_index = 0) const noexcept
    {
        const size_t slot = input_slots[slot_index];
        return slot == 0 ? fp.input_views[layer][0] : fp.forward_slots[layer][slot];
    }

    vector<TensorView>& get_inputs(ForwardPropagation& fp, size_t layer, size_t = 0) const noexcept
    {
        return fp.input_views[layer];
    }

    TensorView& get_output(ForwardPropagation& fp, size_t layer, size_t slot_index = 0) const noexcept
    {
        return fp.forward_slots[layer][output_slots[slot_index]];
    }

    TensorView& get_output_delta(BackPropagation& bp, size_t layer, size_t slot_index = 0) const noexcept
    {
        const size_t slot = output_delta_slots[slot_index];
        return slot == 0 ? bp.layer_output_deltas[layer] : bp.backward_slots[layer][slot];
    }

    TensorView& get_input_delta(BackPropagation& bp, size_t layer, size_t slot_index = 0) const noexcept
    {
        return bp.backward_slots[layer][input_delta_slots[slot_index]];
    }
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
