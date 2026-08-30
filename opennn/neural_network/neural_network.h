//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N E U R A L   N E T W O R K   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <cstdint>
#include <functional>

#include "opennn/core/configuration.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/variable.h"
#include "opennn/neural_network/layers/layer.h"

namespace opennn
{

enum class NetworkTask
{
    Generic,
    Approximation,
    Classification,
    Forecasting,
    AutoAssociation,
    ImageClassification,
    ObjectDetection,
    TextClassification,
    LanguageModeling
};

class NeuralNetwork
{

public:

    struct HostParametersGuard
    {
        explicit HostParametersGuard(NeuralNetwork& n)
            : network(n), was_on_device(n.parameters.get_device() == Device::CUDA)
        {
            if (was_on_device) network.copy_parameters_host();
        }

        ~HostParametersGuard()
        {
            if (!was_on_device) return;
            try { network.copy_parameters_device(); } catch (...) {}
        }

        HostParametersGuard(const HostParametersGuard&) = delete;
        HostParametersGuard& operator=(const HostParametersGuard&) = delete;

        NeuralNetwork& network;
        const bool was_on_device;
    };

    NeuralNetwork();

    virtual ~NeuralNetwork() = default;

    NeuralNetwork(const filesystem::path&);

    NetworkTask get_task() const noexcept { return task; }
    void set_task(NetworkTask new_task) noexcept { task = new_task; }

    Index add_layer(unique_ptr<Layer>,
                    const vector<Index>& = {});

    const EffectiveConfig& get_config() const noexcept { return config; }
    Device get_device() const noexcept { return config.device; }
    bool is_gpu() const noexcept { return config.device == Device::CUDA; }
    bool is_cpu() const noexcept { return config.device == Device::CPU; }

    Type get_training_type()  const noexcept { return config.training_type; }

    void set_training_activation_recomputation(bool enabled) noexcept
    {
        training_activation_recomputation = enabled;
    }
    bool get_training_activation_recomputation() const noexcept
    {
        return training_activation_recomputation;
    }

    void warn_if_stale_configuration() const;

    vector<vector<TensorSpec>> get_parameter_specs() const
    {
        return collect_layer_specs([](const Layer& layer) { return layer.get_parameter_specs(); });
    }

    vector<vector<TensorSpec>> get_state_specs() const
    {
        return collect_layer_specs([](const Layer& layer) { return layer.get_state_specs(); });
    }

    vector<vector<TensorSpec>> get_forward_specs(Index batch_size) const
    {
        auto specs = collect_layer_specs([batch_size](const Layer& layer) { return layer.get_forward_specs(batch_size); });
        if (!is_gpu()) force_specs_to_fp32(specs);
        return specs;
    }

    vector<vector<TensorSpec>> get_backward_specs(Index batch_size) const
    {
        auto specs = collect_layer_specs([batch_size](const Layer& layer) { return layer.get_backward_specs(batch_size); });
        if (!is_gpu()) force_specs_to_fp32(specs);
        return specs;
    }

    Index get_states_size() const     { return get_aligned_size(get_state_specs()); }

    void compile();
    void compile(Device device);
    bool has(const string&) const;
    bool has(LayerType) const;
    bool has_recurrent_layers() const;

    bool is_empty() const noexcept { return layers.empty(); }

    // Anything derived from the parameters -- a packed cuDNN weight space, a
    // batch-norm fold, a quantized copy -- has to know when they change. There
    // is no way to observe that passively, because callers write straight
    // through the mutable handles below, so handing one out counts as a change.
    // That over-counts the few readers that take a mutable handle and only read,
    // which costs those callers a recompute and cannot make a cache stale. The
    // reverse policy, trusting writers to announce themselves, is one missed
    // call site away from silently wrong weights.
    // A handle retained across an execution has already consumed that one
    // invalidation; call mark_parameters_changed() before executing after any
    // later writes through the retained handle.
    //
    // Cache holders should store the version they were built at and compare, not
    // assume any particular increment.
    uint64_t get_parameters_version() const noexcept { return parameters_version; }

    void mark_parameters_changed() noexcept { ++parameters_version; }

    float* get_parameters_data() { mark_parameters_changed(); return parameters.as<float>(); }
    const float* get_parameters_data() const { return parameters.as<float>(); }
    VectorMap get_parameters_map() &
    {
        mark_parameters_changed();
        return parameters.as_vector();
    }
    ConstVectorMap get_parameters_map() const &
    {
        return parameters.as_vector();
    }
    Index get_parameters_buffer_size() const noexcept { return parameters.size_in_floats(); }
    Device get_parameters_device() const noexcept { return parameters.get_device(); }
    float* get_states_data() { return states.as<float>(); }
    const float* get_states_data() const { return states.as<float>(); }
    Index get_states_buffer_size() const noexcept { return states.size_in_floats(); }

    const vector<Variable>& get_input_variables() const noexcept { return input_variables; }
    vector<string> get_input_feature_names() const { return get_variable_feature_names(input_variables); }

    const vector<Variable>& get_output_variables() const noexcept { return output_variables; }
    vector<string> get_output_feature_names() const { return get_variable_feature_names(output_variables); }

    const vector<unique_ptr<Layer>>& get_layers() const noexcept { return layers; }
    const unique_ptr<Layer>& get_layer(const Index layer_index) const { return layers[layer_index]; }
    const unique_ptr<Layer>& get_layer(const string&) const;

    Index get_layer_index(const string&) const;

    const vector<vector<Index>>& get_source_layers() const noexcept { return source_layers; }

    Layer* get_first(const string&);
    Layer* get_first(LayerType);
    const Layer* get_first(const string&) const;
    const Layer* get_first(LayerType) const;

    void set_input_variables(const vector<Variable>& new_input_variables) { input_variables = new_input_variables; }
    void set_output_variables(const vector<Variable>& new_output_variables) { output_variables = new_output_variables; }

    void set_input_names(const vector<string>&);
    void set_output_names(const vector<string>&);

    void set_input_shape(const Shape&);

    void clear();
    void steal_from(NeuralNetwork& src);

    Index get_layers_number() const noexcept { return ssize(layers); }
    Index get_layers_number(const string&) const;
    Index get_layers_number(LayerType) const;

    Index get_first_trainable_layer_index() const;
    Index get_last_trainable_layer_index() const;


    Index get_inputs_number() const { return get_input_shape().size(); }
    Index get_outputs_number() const { return get_output_shape().size(); }

    Shape get_input_shape() const;
    Shape get_output_shape() const;

    ActivationFunction get_output_activation() const;
    Index get_parameters_number() const;

    void set_parameters(const VectorR&);
    void set_states(const VectorR&);
    void set_parameters_random() { initialize_parameters(&Operator::set_parameters_random); }
    void set_parameters_glorot() { initialize_parameters(&Operator::set_parameters_glorot); }
    void set_parameters_pytorch() { initialize_parameters(&Operator::set_parameters_pytorch); }
    void link_parameters();

    enum class ParameterStorage
    {
        Host,
        DeviceMaster,
        DeviceMasterWithMirror,
        DeviceCompact
    };

    ParameterStorage get_parameter_storage() const noexcept
    {
        if (parameters.get_device() != Device::CUDA)
            return ParameterStorage::Host;

        if (!parameters.owns_memory())
            return ParameterStorage::DeviceCompact;

        return parameters_bf16_mirror.empty() && parameters_int8_storage.empty()
            ? ParameterStorage::DeviceMaster
            : ParameterStorage::DeviceMasterWithMirror;
    }

    bool fp32_master_released() const noexcept
    {
        return get_parameter_storage() == ParameterStorage::DeviceCompact;
    }

    // Whether the derived storage this precision needs has been built yet. A
    // separate question from where the master lives: FP32 needs nothing derived,
    // so it is always ready.
    bool low_precision_storage_ready() const noexcept
    {
        if (config.training_type == Type::BF16) return !parameters_bf16_mirror.empty();
        if (config.training_type == Type::INT8) return !parameters_int8_storage.empty();
        return true;
    }

    void link_gradients(const Buffer&) const;
    void link_gradients(span<const TensorView> layer_gradients) const;

    void link_states();
    void link_states(Device);
    MatrixR calculate_outputs(const vector<TensorView>&);

    void calculate_outputs(const vector<TensorView>&, MatrixR& outputs);

    TensorView calculate_outputs_resident(const vector<TensorView>&,
                                          ForwardPropagation&,
                                          bool upload_parameters = true);

    MatrixR calculate_outputs(const MatrixR&);

    MatrixR calculate_outputs(const Tensor3&);

    MatrixR calculate_outputs(const Tensor4&);

    void calculate_outputs(const MatrixR&, MatrixR& outputs);

    void calculate_outputs(const Tensor3&, MatrixR& outputs);

    void calculate_outputs(const Tensor4&, MatrixR& outputs);

    Tensor3 calculate_outputs(const Tensor3&, const Tensor3&);

    void calculate_outputs(const Tensor3&, const Tensor3&, Tensor3& outputs);

    void from_JSON(const JsonDocument&);

    void to_JSON(JsonWriter&) const;

    void save(const filesystem::path&) const;
    void save_parameters_binary(const filesystem::path&) const;
    void save_states_binary(const filesystem::path&) const;

    void load(const filesystem::path&);
    void load_parameters_binary(const filesystem::path&);

    void load_parameters_bf16_inference_binary(const filesystem::path&);
    void load_states_binary(const filesystem::path&);

    void forward_propagate(const vector<TensorView>&,
                          ForwardPropagation&,
                          ForwardPropagationMode = ForwardPropagationMode::Inference) const;

    void forward_propagate(const vector<TensorView>&,
                          ForwardPropagation&,
                          ForwardPropagationMode,
                          Index,
                          Index) const;

    void forward_propagate(const vector<TensorView>&,
                          const VectorR&,
                          ForwardPropagation&);

    void cast_parameters_to_bf16();

    void release_bf16_fp32_parameter_master_for_inference();

    void upload_parameters_bf16_inference();

    void upload_parameters_int8_inference();

    bfloat16* get_parameters_bf16_mirror_data()
    {
        return config.training_type == Type::BF16 && parameters.owns_memory()
            ? parameters_bf16_mirror.as<bfloat16>()
            : nullptr;
    }

    void copy_parameters_device();
    void copy_parameters_host();

    void copy_states_device();
    void copy_states_host();

    vector<string> get_layer_labels() const;

protected:

    explicit NeuralNetwork(NetworkTask);
    NeuralNetwork(const filesystem::path&, NetworkTask);

    NetworkTask task = NetworkTask::Generic;

    vector<Variable> input_variables;
    vector<Variable> output_variables;

    vector<unique_ptr<Layer>> layers;

    vector<vector<Index>> source_layers;

    struct DeviceResidency
    {
        const void* parameters = nullptr;
        const void* bf16_mirror = nullptr;
        const void* fp32_inference = nullptr;
        const void* int8_storage = nullptr;
        const void* states = nullptr;

        friend bool operator==(const DeviceResidency&, const DeviceResidency&) = default;
    };

    DeviceResidency get_device_residency() const noexcept;

    // Bumped whenever a mutable handle to the parameters is handed out, so a
    // derived cache can tell that what it was built from has moved on.
    uint64_t parameters_version = 1;

    Buffer parameters;
    Buffer parameters_bf16_mirror{Device::CUDA};
    Buffer parameters_fp32_inference_storage{Device::CUDA};
    Buffer parameters_int8_storage{Device::CUDA};

    bool parameters_bf16_mirror_compact = false;

    Buffer states;

    EffectiveConfig config;

    bool training_activation_recomputation = false;

    mutable bool stale_configuration_warned = false;

    mutable const void* linked_gradient_base = nullptr;

private:

    void compile(EffectiveConfig);

    MatrixR calculate_outputs_device(const vector<TensorView>&, ForwardPropagation&);

    void calculate_outputs_device(const vector<TensorView>&, ForwardPropagation&,
                                  MatrixR& outputs);

    struct HostStatesGuard
    {
        explicit HostStatesGuard(NeuralNetwork& n)
            : HostStatesGuard(n, n.states.get_device() == Device::CUDA) {}

        HostStatesGuard(NeuralNetwork& n, bool stage)
            : network(n), was_on_device(stage)
        {
            if (was_on_device) network.copy_states_host();
        }

        ~HostStatesGuard()
        {
            if (!was_on_device) return;
            try { network.copy_states_device(); } catch (...) {}
        }

        HostStatesGuard(const HostStatesGuard&) = delete;
        HostStatesGuard& operator=(const HostStatesGuard&) = delete;

        NeuralNetwork& network;
        const bool was_on_device;
    };

    void initialize_parameters(void (Operator::*)());

    uint64_t parameter_layout_fingerprint() const;
    uint64_t state_layout_fingerprint() const;

    void clear_low_precision_parameter_storage();
    void activate_transposed_inference_weights();

    struct ParameterSlot
    {
        Layer* layer = nullptr;
        Shape shape;
        Type dtype = Type::FP32;
        bool tied = false;
        Index scale_channels = 0;
        int   scale_axis = 0;
        Index master_offset = 0;
        Index bf16_offset = 0;
        Index int8_offset = 0;
        Index fp32_offset = 0;
    };

    struct ParameterSlotTotals
    {
        Index bf16_elements = 0;
        Index int8_elements = 0;
        Index fp32_elements = 0;
    };

    ParameterSlotTotals for_each_parameter_slot(
        const function<void(const ParameterSlot&)>& visit,
        const function<void(Layer&)>& begin_layer = {}) const;

    void allocate_compact_parameter_storage(const ParameterSlotTotals&);
    void use_compact_parameter_storage();

    static void force_specs_to_fp32(vector<vector<TensorSpec>>& specs)
    {
        for (auto& layer_specs : specs)
            for (auto& spec : layer_specs)
                if (spec.dtype == Type::BF16)
                    spec.dtype = Type::FP32;
    }

    template<typename Fn>
    vector<vector<TensorSpec>> collect_layer_specs(Fn fn) const
    {
        vector<vector<TensorSpec>> out(layers.size());
        ranges::transform(layers, out.begin(),
                          [&](const unique_ptr<Layer>& layer) { return fn(*layer); });
        return out;
    }
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
