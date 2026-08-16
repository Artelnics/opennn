#include "tests/pch.h"

#include "opennn/core/json.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/neural_network/standard_networks.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/layers/embedding_layer.h"
#include "opennn/neural_network/layers/layer.h"
#include "opennn/neural_network/layers/long_short_term_memory_layer.h"
#include "opennn/neural_network/layers/recurrent_layer.h"
#include "opennn/neural_network/layers/scaling_layer.h"
#include "opennn/neural_network/layers/tokenizer_layer.h"
#include "opennn/dataset/dataset.h"

using namespace opennn;

namespace
{

vector<char> read_snapshot_bytes(const filesystem::path& path)
{
    ifstream input(path, ios::binary | ios::ate);
    throw_if(!input.is_open(), "Cannot open test snapshot: {}", path.string());

    const streamoff file_size = input.tellg();
    throw_if(file_size < 0, "Cannot determine test snapshot size: {}", path.string());
    input.seekg(0);

    vector<char> bytes(static_cast<size_t>(file_size));
    input.read(bytes.data(), file_size);
    throw_if(!input, "Cannot read test snapshot: {}", path.string());
    return bytes;
}

void write_snapshot_bytes(const filesystem::path& path, const vector<char>& bytes)
{
    ofstream output(path, ios::binary | ios::trunc);
    throw_if(!output.is_open(), "Cannot open test snapshot: {}", path.string());
    output.write(bytes.data(), streamsize(bytes.size()));
    throw_if(!output, "Cannot write test snapshot: {}", path.string());
}

enum class SnapshotKind { Parameters, States };

void configure_snapshot_network(NeuralNetwork& network, const Shape& input_shape,
                                const Shape& output_shape, SnapshotKind kind)
{
    network.add_layer(make_unique<opennn::Dense>(
        input_shape, output_shape, "Identity", kind == SnapshotKind::States));
    network.compile();
}

Index snapshot_size(const NeuralNetwork& network, SnapshotKind kind)
{
    return kind == SnapshotKind::Parameters
         ? network.get_parameters_buffer_size()
         : network.get_states_buffer_size();
}

const float* snapshot_data(const NeuralNetwork& network, SnapshotKind kind)
{
    return kind == SnapshotKind::Parameters
         ? network.get_parameters_data()
         : network.get_states_data();
}

void set_snapshot(NeuralNetwork& network, SnapshotKind kind, const VectorR& values)
{
    if (kind == SnapshotKind::Parameters) network.set_parameters(values);
    else                                  network.set_states(values);
}

void save_snapshot(const NeuralNetwork& network, SnapshotKind kind,
                   const filesystem::path& path)
{
    if (kind == SnapshotKind::Parameters) network.save_parameters_binary(path);
    else                                  network.save_states_binary(path);
}

void load_snapshot(NeuralNetwork& network, SnapshotKind kind,
                   const filesystem::path& path)
{
    if (kind == SnapshotKind::Parameters) network.load_parameters_binary(path);
    else                                  network.load_states_binary(path);
}

void expect_snapshot_values(const NeuralNetwork& network, SnapshotKind kind,
                            const VectorR& expected)
{
    const float* actual = snapshot_data(network, kind);
    for (Index i = 0; i < expected.size(); ++i)
        EXPECT_FLOAT_EQ(actual[i], expected(i));
}

void validate_snapshot_format(SnapshotKind kind, string_view file_stem,
                              float scale, float offset, float sentinel_value)
{
    const filesystem::path directory = filesystem::temp_directory_path();
    const string stem(file_stem);
    const filesystem::path versioned_path = directory / (stem + ".bin");
    const filesystem::path corrupted_path = directory / (stem + "_corrupted.bin");
    const filesystem::path future_path = directory / (stem + "_future.bin");
    const filesystem::path legacy_path = directory / (stem + "_legacy.bin");
    const initializer_list<filesystem::path> paths = {
        versioned_path, corrupted_path, future_path, legacy_path};

    error_code error;
    for (const filesystem::path& path : paths) filesystem::remove(path, error);

    NeuralNetwork source;
    configure_snapshot_network(source, Shape{2}, Shape{3}, kind);

    VectorR expected(snapshot_size(source, kind));
    for (Index i = 0; i < expected.size(); ++i)
        expected(i) = scale * float(i + 1) + offset;
    set_snapshot(source, kind, expected);
    save_snapshot(source, kind, versioned_path);

    NeuralNetwork compatible;
    configure_snapshot_network(compatible, Shape{2}, Shape{3}, kind);
    load_snapshot(compatible, kind, versioned_path);
    expect_snapshot_values(compatible, kind, expected);

    NeuralNetwork different_layout;
    configure_snapshot_network(different_layout, Shape{3}, Shape{2}, kind);
    ASSERT_EQ(snapshot_size(different_layout, kind), expected.size());
    EXPECT_THROW(load_snapshot(different_layout, kind, versioned_path), runtime_error);

    const vector<char> file_bytes = read_snapshot_bytes(versioned_path);
    ASSERT_GT(file_bytes.size(), expected.size() * sizeof(float));

    vector<char> corrupted_bytes = file_bytes;
    corrupted_bytes.back() ^= char(0x01);
    write_snapshot_bytes(corrupted_path, corrupted_bytes);

    const VectorR sentinel = VectorR::Constant(expected.size(), sentinel_value);
    set_snapshot(compatible, kind, sentinel);
    EXPECT_THROW(load_snapshot(compatible, kind, corrupted_path), runtime_error);
    expect_snapshot_values(compatible, kind, sentinel);

    vector<char> future_bytes = file_bytes;
    ASSERT_GT(future_bytes.size(), size_t(8));
    future_bytes[8] = char(2);
    future_bytes[9] = future_bytes[10] = future_bytes[11] = 0;
    write_snapshot_bytes(future_path, future_bytes);
    EXPECT_THROW(load_snapshot(compatible, kind, future_path), runtime_error);

    ofstream legacy(legacy_path, ios::binary | ios::trunc);
    ASSERT_TRUE(legacy.is_open());
    legacy.write(reinterpret_cast<const char*>(expected.data()),
                 streamsize(expected.size() * Index(sizeof(float))));
    ASSERT_TRUE(legacy.good());
    legacy.close();

    load_snapshot(compatible, kind, legacy_path);
    expect_snapshot_values(compatible, kind, expected);

    for (const filesystem::path& path : paths) filesystem::remove(path, error);
}

#ifdef OPENNN_HAS_CUDA
void validate_cuda_snapshot_roundtrip(SnapshotKind kind, string_view file_name,
                                      float scale, float offset, float sentinel_value)
{
    const filesystem::path path = filesystem::temp_directory_path() / file_name;
    error_code error;
    filesystem::remove(path, error);

    NeuralNetwork network;
    configure_snapshot_network(network, Shape{2}, Shape{3}, kind);

    VectorR expected(snapshot_size(network, kind));
    for (Index i = 0; i < expected.size(); ++i)
        expected(i) = scale * float(i + 1) + offset;
    set_snapshot(network, kind, expected);

    if (kind == SnapshotKind::Parameters) network.copy_parameters_device();
    else                                  network.copy_states_device();

    save_snapshot(network, kind, path);
    set_snapshot(network, kind, VectorR::Constant(expected.size(), sentinel_value));
    load_snapshot(network, kind, path);

    if (kind == SnapshotKind::Parameters) network.copy_parameters_host();
    else                                  network.copy_states_host();

    expect_snapshot_values(network, kind, expected);
    filesystem::remove(path, error);
}
#endif

}

TEST(NeuralNetworkTest, DefaultConstructor)
{
    NeuralNetwork neural_network;

    EXPECT_EQ(neural_network.is_empty(), true);
    EXPECT_EQ(neural_network.get_layers_number(), 0);
    EXPECT_EQ(neural_network.get_task(), NetworkTask::Generic);
}

TEST(NeuralNetworkTest, RejectsWrongSourceCount)
{
    NeuralNetwork neural_network;

    EXPECT_THROW(neural_network.add_layer(
                     make_unique<opennn::Dense>(Shape{2}, Shape{2}, "Identity"),
                     {-1, -2}),
                 runtime_error);

    EXPECT_THROW(neural_network.add_layer(nullptr), runtime_error);
}

TEST(NeuralNetworkTest, DetectsRecurrentLayersByCapability)
{
    const auto has_recurrent_layer = [](unique_ptr<Layer> layer)
    {
        NeuralNetwork network;
        network.add_layer(std::move(layer));
        return network.has_recurrent_layers();
    };

    EXPECT_FALSE(has_recurrent_layer(
        make_unique<opennn::Dense>(Shape{3}, Shape{2}, "Identity")));
    EXPECT_TRUE(has_recurrent_layer(
        make_unique<Recurrent>(Shape{4, 3}, Shape{2})));
    EXPECT_TRUE(has_recurrent_layer(
        make_unique<LongShortTermMemory>(Shape{4, 3}, Shape{2})));
}

TEST(NeuralNetworkTest, InputCountIsTheExternalShapeSize)
{
    const auto inputs_number = [](unique_ptr<Layer> layer)
    {
        NeuralNetwork network;
        network.add_layer(std::move(layer));
        return network.get_inputs_number();
    };

    EXPECT_EQ(inputs_number(
        make_unique<opennn::Dense>(Shape{5}, Shape{2}, "Identity")), 5);
    EXPECT_EQ(inputs_number(
        make_unique<Recurrent>(Shape{4, 3}, Shape{2})), 12);
    EXPECT_EQ(inputs_number(
        make_unique<LongShortTermMemory>(Shape{4, 3}, Shape{2})), 12);
    EXPECT_EQ(inputs_number(
        make_unique<Scaling>(Shape{2, 3, 4})), 24);
    EXPECT_EQ(inputs_number(
        make_unique<Embedding>(Shape{100, 7}, 8)), 7);
}

TEST(NeuralNetworkTest, DiscreteInputLayersRejectBf16InputCasts)
{
    const opennn::Dense dense(Shape{1}, Shape{1}, "Identity");
    const Embedding embedding(Shape{512, 1}, 2);
    const Tokenizer tokenizer(Shape{1});

    EXPECT_TRUE(dense.allows_bf16_input_cast(0));
    EXPECT_FALSE(embedding.allows_bf16_input_cast(0));
    EXPECT_FALSE(tokenizer.allows_bf16_input_cast(0));
}

TEST(NeuralNetworkTest, SetInputShapePropagatesFromTheFirstExternalInput)
{
    NeuralNetwork scaled;
    scaled.add_layer(make_unique<Scaling>(Shape{2}));
    scaled.add_layer(make_unique<opennn::Dense>(Shape{2}, Shape{3}, "Identity"));
    scaled.add_layer(make_unique<opennn::Dense>(Shape{3}, Shape{1}, "Identity"));

    scaled.set_input_shape(Shape{5});

    EXPECT_EQ(scaled.get_layer(0)->get_input_shape(), Shape{5});
    EXPECT_EQ(scaled.get_layer(1)->get_input_shape(), Shape{5});
    EXPECT_EQ(scaled.get_layer(2)->get_input_shape(), Shape{3});

    NeuralNetwork unscaled;
    unscaled.add_layer(make_unique<opennn::Dense>(Shape{2}, Shape{3}, "Identity"));
    unscaled.add_layer(make_unique<opennn::Dense>(Shape{3}, Shape{1}, "Identity"));

    unscaled.set_input_shape(Shape{7});

    EXPECT_EQ(unscaled.get_layer(0)->get_input_shape(), Shape{7});
    EXPECT_EQ(unscaled.get_layer(1)->get_input_shape(), Shape{3});
}

TEST(NeuralNetworkTest, SetInputShapeDoesNotRequireTrainableLayers)
{
    NeuralNetwork empty;
    EXPECT_NO_THROW(empty.set_input_shape(Shape{4}));

    NeuralNetwork preprocessing_only;
    preprocessing_only.add_layer(make_unique<Scaling>(Shape{2}));

    EXPECT_NO_THROW(preprocessing_only.set_input_shape(Shape{6}));
    EXPECT_EQ(preprocessing_only.get_input_shape(), Shape{6});
}

TEST(NeuralNetworkTest, SetInputShapeLeavesOtherExternalInputsUnchanged)
{
    NeuralNetwork network;
    network.add_layer(
        make_unique<opennn::Dense>(Shape{2}, Shape{3}, "Identity"), {-1});
    network.add_layer(
        make_unique<opennn::Dense>(Shape{4}, Shape{3}, "Identity"), {-2});

    network.set_input_shape(Shape{5});

    EXPECT_EQ(network.get_layer(0)->get_input_shape(), Shape{5});
    EXPECT_EQ(network.get_layer(1)->get_input_shape(), Shape{4});
}

TEST(NeuralNetworkTest, PreScaledInputBoundaryIsPlannedOnce)
{
    NeuralNetwork network;
    network.add_layer(make_unique<Scaling>(Shape{1}));
    network.add_layer(make_unique<opennn::Dense>(Shape{1}, Shape{1}, "Identity"));
    network.compile();

    ForwardPropagation raw_training(
        2, &network, ForwardPropagationMode::Training, {}, false);
    ForwardPropagation preprocessed_inference(
        2, &network, ForwardPropagationMode::Inference, {}, true);

    EXPECT_EQ(raw_training.get_execution_start_layer(), 0);
    EXPECT_EQ(preprocessed_inference.get_execution_start_layer(), 1);
    EXPECT_FALSE(raw_training.slots[0].back().empty());
    EXPECT_TRUE(preprocessed_inference.slots[0].back().empty());

    Tensor2 inputs(2, 1);
    inputs.setValues({{2.0f}, {3.0f}});
    const vector<TensorView> input_views = {
        TensorView(inputs.data(), {2, 1})};

    network.forward_propagate(input_views, raw_training, true);
    network.forward_propagate(input_views, preprocessed_inference, false);

    EXPECT_NE(raw_training.inputs[1][0].get_data(), inputs.data());
    EXPECT_EQ(preprocessed_inference.inputs[1][0].get_data(), inputs.data());
}

TEST(NeuralNetworkTest, PreScaledBoundaryLeavesTextInputPipelineActive)
{
    TextClassificationNetwork network(Shape{16, 4, 8}, Shape{2}, Shape{2});

    ASSERT_FALSE(network.get_layer(0)->skip_for_pre_scaled_input());

    ForwardPropagation propagation(
        3, &network, ForwardPropagationMode::Training, {}, true);

    EXPECT_EQ(propagation.get_execution_start_layer(), 0);
}

TEST(NeuralNetworkTest, PreScaledInputIsOutputWhenEveryLayerIsSkipped)
{
    NeuralNetwork network;
    network.add_layer(make_unique<Scaling>(Shape{1}));
    network.compile();

    ForwardPropagation propagation(
        2, &network, ForwardPropagationMode::Inference, {}, true);

    Tensor2 inputs(2, 1);
    inputs.setValues({{2.0f}, {3.0f}});
    const vector<TensorView> input_views = {
        TensorView(inputs.data(), {2, 1})};

    network.forward_propagate(input_views, propagation, false);

    const TensorView outputs = propagation.get_outputs();
    ASSERT_EQ(outputs.get_data(), inputs.data());
    EXPECT_EQ(outputs.get_shape(), (Shape{2, 1}));
}

TEST(NeuralNetworkTest, SerializesNetworkTask)
{
    NeuralNetwork neural_network;
    neural_network.set_task(NetworkTask::TextClassification);

    JsonWriter writer;
    neural_network.to_JSON(writer);

    JsonDocument document;
    document.root = Json::parse(writer.c_str());

    NeuralNetwork loaded;
    loaded.from_JSON(document);

    EXPECT_EQ(loaded.get_task(), NetworkTask::TextClassification);
}

TEST(NeuralNetworkTest, SerializesTiedWeightRelationships)
{
    NeuralNetwork neural_network;

    auto embedding = make_unique<Embedding>(Shape{11, 4}, 6, "embedding");
    Layer* embedding_source = embedding.get();
    neural_network.add_layer(std::move(embedding));

    auto output = make_unique<opennn::Dense>(Shape{4, 6}, Shape{11}, "Identity", false, "output");
    output->set_use_bias(false);
    output->set_tied_weight_source(embedding_source);
    neural_network.add_layer(std::move(output));
    neural_network.compile();

    JsonWriter writer;
    neural_network.to_JSON(writer);

    JsonDocument document;
    document.root = Json::parse(writer.c_str());

    NeuralNetwork restored;
    restored.from_JSON(document);

    ASSERT_EQ(restored.get_layers_number(), 2);
    const Layer::TiedWeight tied_weight = restored.get_layer(1)->get_tied_weight();
    EXPECT_EQ(tied_weight.source, restored.get_layer(0).get());
    EXPECT_EQ(tied_weight.spec_index, 0);
    EXPECT_EQ(tied_weight.source_spec_index, 0);

    JsonWriter restored_writer;
    restored.to_JSON(restored_writer);
    EXPECT_EQ(restored_writer.c_str(), writer.c_str());
}

TEST(NeuralNetworkTest, CompleteSaveLoadPreservesModelOwnedState)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const filesystem::path directory = filesystem::temp_directory_path();
    const filesystem::path model_path = directory / "opennn_complete_persistence_test.json";
    filesystem::path parameters_path = model_path;
    parameters_path.replace_extension(".bin");
    const filesystem::path states_path = directory / "opennn_complete_persistence_test.states.bin";

    error_code error;
    filesystem::remove(model_path, error);
    filesystem::remove(parameters_path, error);
    filesystem::remove(states_path, error);

    NeuralNetwork network;
    network.set_task(NetworkTask::Classification);
    network.set_training_activation_recomputation(true);

    auto scaling = make_unique<Scaling>(Shape{3});
    scaling->set_descriptives({
        Descriptives(-3.14159274f, 7.12345695f, 0.123456791f, 1.23456788f),
        Descriptives(-2.71828175f, 9.87654305f, -0.987654328f, 2.34567881f),
        Descriptives(-1.41421354f, 6.78901243f, 0.333333343f, 3.45678902f)
    });
    scaling->set_scalers({"MeanStandardDeviation", "MinimumMaximum", "StandardDeviation"});
    network.add_layer(std::move(scaling));

    auto dense = make_unique<opennn::Dense>(Shape{3}, Shape{2}, "ReLU", true, "output");
    dense->set_momentum(0.234567895f);
    network.add_layer(std::move(dense));

    Variable count("count", "Decoder", VariableType::Integer, "Logarithm");
    Variable category("category", "Input", VariableType::Categorical, "None", {"red", "green"});
    Variable target("scores", "InputTarget", VariableType::Numeric, "StandardDeviation");
    target.features = 2;
    network.set_input_variables({count, category});
    network.set_output_variables({target});

    network.compile();

    VectorR parameters(network.get_parameters_buffer_size());
    for (Index i = 0; i < parameters.size(); ++i)
        parameters(i) = 0.013579246f * float(i + 1) - 0.246813580f;
    network.set_parameters(parameters);

    VectorR states = VectorR::Zero(network.get_states_buffer_size());
    ASSERT_GT(states.size(), get_aligned_size(2) + 1);
    states(0) = 0.123456791f;
    states(1) = -0.987654328f;
    states(get_aligned_size(2)) = 1.23456788f;
    states(get_aligned_size(2) + 1) = 2.34567881f;
    network.set_states(states);

    MatrixR inputs(2, 3);
    inputs << -1.25f, 0.5f, 2.75f,
               3.5f, -0.75f, 1.125f;
    const MatrixR expected_outputs = network.calculate_outputs(inputs);

    network.save(model_path);
    network.save_states_binary(states_path);

    ASSERT_TRUE(filesystem::exists(model_path));
    ASSERT_GT(filesystem::file_size(parameters_path),
              uintmax_t(parameters.size() * Index(sizeof(float))));
    ASSERT_GT(filesystem::file_size(states_path),
              uintmax_t(states.size() * Index(sizeof(float))));

    const JsonDocument saved_document = load_json_file(model_path);
    const Json* saved_root = get_json_root(saved_document, "NeuralNetwork");
    ASSERT_NE(saved_root, nullptr);
    EXPECT_TRUE(read_json_bool(saved_root, "TrainingActivationRecomputation"));
    EXPECT_FALSE(saved_root->has("Device"));
    EXPECT_FALSE(saved_root->has("TrainingType"));

    NeuralNetwork restored;
    restored.load(model_path);

    EXPECT_EQ(restored.get_task(), NetworkTask::Classification);
    EXPECT_TRUE(restored.get_training_activation_recomputation());
    EXPECT_EQ(restored.get_device(), Device::CPU);
    EXPECT_EQ(restored.get_training_type(), Type::FP32);

    ASSERT_EQ(restored.get_input_variables().size(), 2);
    const Variable& restored_count = restored.get_input_variables()[0];
    EXPECT_EQ(restored_count.name, "count");
    EXPECT_EQ(restored_count.role, VariableRole::Decoder);
    EXPECT_EQ(restored_count.type, VariableType::Integer);
    EXPECT_EQ(restored_count.scaler, ScalerMethod::Logarithm);

    const Variable& restored_category = restored.get_input_variables()[1];
    EXPECT_EQ(restored_category.role, VariableRole::Input);
    EXPECT_EQ(restored_category.type, VariableType::Categorical);
    EXPECT_EQ(restored_category.scaler, ScalerMethod::None);
    EXPECT_EQ(restored_category.categories, vector<string>({"red", "green"}));

    ASSERT_EQ(restored.get_output_variables().size(), 1);
    const Variable& restored_target = restored.get_output_variables()[0];
    EXPECT_EQ(restored_target.role, VariableRole::InputTarget);
    EXPECT_EQ(restored_target.type, VariableType::Numeric);
    EXPECT_EQ(restored_target.scaler, ScalerMethod::StandardDeviation);
    EXPECT_EQ(restored_target.features, 2);

    ASSERT_EQ(restored.get_parameters_buffer_size(), parameters.size());
    ASSERT_EQ(restored.get_states_buffer_size(), states.size());
    for (Index i = 0; i < parameters.size(); ++i)
        EXPECT_FLOAT_EQ(restored.get_parameters_data()[i], parameters(i));
    for (Index i = 0; i < states.size(); ++i)
        EXPECT_FLOAT_EQ(restored.get_states_data()[i], states(i));

    JsonWriter original_writer;
    JsonWriter restored_writer;
    network.to_JSON(original_writer);
    restored.to_JSON(restored_writer);
    EXPECT_EQ(restored_writer.c_str(), original_writer.c_str());

    const MatrixR actual_outputs = restored.calculate_outputs(inputs);
    ASSERT_EQ(actual_outputs.rows(), expected_outputs.rows());
    ASSERT_EQ(actual_outputs.cols(), expected_outputs.cols());
    for (Index i = 0; i < actual_outputs.size(); ++i)
        EXPECT_FLOAT_EQ(actual_outputs(i), expected_outputs(i));

    restored.set_states(VectorR::Zero(states.size()));
    restored.load_states_binary(states_path);
    for (Index i = 0; i < states.size(); ++i)
        EXPECT_FLOAT_EQ(restored.get_states_data()[i], states(i));

    EXPECT_THROW(network.save(directory), runtime_error);

    filesystem::remove(model_path, error);
    filesystem::remove(parameters_path, error);
    filesystem::remove(states_path, error);
    Configuration::instance().set();
}

TEST(NeuralNetworkTest, ModelSaveCommitsOrRecoversJsonAndParametersTogether)
{
    Configuration::instance().set(Device::CPU, Type::FP32);

    const filesystem::path directory = filesystem::temp_directory_path();
    const filesystem::path model_path = directory / "opennn_atomic_model_save_test.json";
    const filesystem::path candidate_path =
        directory / "opennn_atomic_model_save_candidate.json";
    filesystem::path parameter_path = model_path;
    parameter_path.replace_extension(".bin");
    filesystem::path candidate_parameter_path = candidate_path;
    candidate_parameter_path.replace_extension(".bin");

    const auto suffixed = [](filesystem::path path, string_view suffix)
    {
        path += suffix;
        return path;
    };
    const filesystem::path model_backup = suffixed(model_path, ".bak");
    const filesystem::path parameter_backup = suffixed(parameter_path, ".bak");
    const filesystem::path marker_path = suffixed(model_path, ".save-transaction");

    const vector<filesystem::path> artifacts = {
        model_path,
        parameter_path,
        suffixed(model_path, ".tmp"),
        suffixed(parameter_path, ".tmp"),
        model_backup,
        parameter_backup,
        marker_path,
        suffixed(marker_path, ".tmp"),
        candidate_path,
        candidate_parameter_path,
        suffixed(candidate_path, ".tmp"),
        suffixed(candidate_parameter_path, ".tmp"),
        suffixed(candidate_path, ".bak"),
        suffixed(candidate_parameter_path, ".bak"),
        suffixed(candidate_path, ".save-transaction"),
        suffixed(suffixed(candidate_path, ".save-transaction"), ".tmp")
    };

    error_code error;
    for (const filesystem::path& path : artifacts) filesystem::remove(path, error);

    NeuralNetwork original;
    original.set_task(NetworkTask::Classification);
    original.add_layer(make_unique<opennn::Dense>(Shape{2}, Shape{3}, "Identity"));
    original.compile();
    const VectorR original_parameters =
        VectorR::LinSpaced(original.get_parameters_buffer_size(), -0.75f, 0.25f);
    original.set_parameters(original_parameters);

    NeuralNetwork replacement;
    replacement.set_task(NetworkTask::Forecasting);
    replacement.add_layer(make_unique<opennn::Dense>(Shape{2}, Shape{3}, "Identity"));
    replacement.compile();
    const VectorR replacement_parameters =
        VectorR::LinSpaced(replacement.get_parameters_buffer_size(), 1.25f, 2.25f);
    replacement.set_parameters(replacement_parameters);

    original.save(model_path);
    replacement.save(model_path);

    NeuralNetwork committed;
    committed.load(model_path);
    EXPECT_EQ(committed.get_task(), NetworkTask::Forecasting);
    for (Index i = 0; i < replacement_parameters.size(); ++i)
        EXPECT_FLOAT_EQ(committed.get_parameters_data()[i], replacement_parameters(i));
    EXPECT_FALSE(filesystem::exists(suffixed(model_path, ".tmp")));
    EXPECT_FALSE(filesystem::exists(suffixed(parameter_path, ".tmp")));
    EXPECT_FALSE(filesystem::exists(model_backup));
    EXPECT_FALSE(filesystem::exists(parameter_backup));
    EXPECT_FALSE(filesystem::exists(marker_path));

    original.save(model_path);
    replacement.save(candidate_path);

    // Simulate interruption after both replacements but before the marker is
    // removed. The next load must roll the pair back to the old generation.
    filesystem::rename(model_path, model_backup);
    filesystem::rename(parameter_path, parameter_backup);
    filesystem::rename(candidate_path, model_path);
    filesystem::rename(candidate_parameter_path, parameter_path);
    ofstream marker(marker_path, ios::trunc);
    ASSERT_TRUE(marker.is_open());
    marker << "OPENNN_SAVE_TRANSACTION_V1\n1 1\n";
    marker.close();
    ASSERT_TRUE(marker.good());

    NeuralNetwork recovered;
    recovered.load(model_path);
    EXPECT_EQ(recovered.get_task(), NetworkTask::Classification);
    for (Index i = 0; i < original_parameters.size(); ++i)
        EXPECT_FLOAT_EQ(recovered.get_parameters_data()[i], original_parameters(i));

    EXPECT_FALSE(filesystem::exists(suffixed(model_path, ".tmp")));
    EXPECT_FALSE(filesystem::exists(suffixed(parameter_path, ".tmp")));
    EXPECT_FALSE(filesystem::exists(model_backup));
    EXPECT_FALSE(filesystem::exists(parameter_backup));
    EXPECT_FALSE(filesystem::exists(marker_path));

    for (const filesystem::path& path : artifacts) filesystem::remove(path, error);
    Configuration::instance().set();
}

TEST(NeuralNetworkTest, ParameterSnapshotsValidateVersionIntegrityAndLayout)
{
    Configuration::instance().set(Device::CPU, Type::FP32);
    validate_snapshot_format(SnapshotKind::Parameters,
                             "opennn_parameter_format_test",
                             0.017f, -0.31f, 42.0f);
    Configuration::instance().set();
}

TEST(NeuralNetworkTest, StateSnapshotsValidateVersionIntegrityAndLayout)
{
    Configuration::instance().set(Device::CPU, Type::FP32);
    validate_snapshot_format(SnapshotKind::States,
                             "opennn_state_format_test",
                             0.021f, 0.19f, 23.0f);
    Configuration::instance().set();
}

#ifdef OPENNN_HAS_CUDA
TEST(NeuralNetworkTest, VersionedParameterSnapshotRoundTripsCudaStorage)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);
    validate_cuda_snapshot_roundtrip(
        SnapshotKind::Parameters, "opennn_parameter_format_cuda.bin",
        0.029f, -0.43f, 17.0f);
    Configuration::instance().set();
}

TEST(NeuralNetworkTest, VersionedStateSnapshotRoundTripsCudaStorage)
{
    Configuration::instance().set(Device::CUDA, Type::FP32);
    validate_cuda_snapshot_roundtrip(
        SnapshotKind::States, "opennn_state_format_cuda.bin",
        0.031f, 0.27f, 29.0f);
    Configuration::instance().set();
}
#endif

TEST(NeuralNetworkTest, ApproximationConstructor)
{
    ApproximationNetwork neural_network({ 1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 5);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(3)->get_name(), "Unscaling");
    EXPECT_EQ(neural_network.get_layer(4)->get_name(), "Bounding");
    EXPECT_EQ(neural_network.get_task(), NetworkTask::Approximation);
}

TEST(NeuralNetworkTest, ClassificationConstructor)
{
    ClassificationNetwork neural_network({ 1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 3);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_task(), NetworkTask::Classification);
}

TEST(NeuralNetworkTest, AproximationConstructor)
{
    ApproximationNetwork neural_network({ 1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 5);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(3)->get_name(), "Unscaling");
    EXPECT_EQ(neural_network.get_layer(4)->get_name(), "Bounding");
}

TEST(NeuralNetworkTest, ForecastingConstructor)
{
    ForecastingNetwork neural_network({ 1,1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 5);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Recurrent");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(3)->get_name(), "Unscaling");
    EXPECT_EQ(neural_network.get_layer(4)->get_name(), "Bounding");
    EXPECT_EQ(neural_network.get_task(), NetworkTask::Forecasting);
}

TEST(NeuralNetworkTest, AutoAssociationConstructor)
{
    AutoAssociationNetwork neural_network({ 1 }, { 4 }, { 2 });

    EXPECT_EQ(neural_network.get_layers_number(), 6);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(3)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(4)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(5)->get_name(), "Unscaling");
    EXPECT_EQ(neural_network.get_task(), NetworkTask::AutoAssociation);
}

TEST(NeuralNetworkTest, AutoAssociationSymmetricEncoderConstructor)
{
    AutoAssociationNetwork neural_network({140}, {32, 16, 8}, "ReLU", "Sigmoid");

    ASSERT_EQ(neural_network.get_layers_number(), 8);
    EXPECT_EQ(neural_network.get_input_shape(), Shape({140}));
    EXPECT_EQ(neural_network.get_output_shape(), Shape({140}));

    const vector<Shape> expected_shapes = {
        {140}, {32}, {16}, {8}, {16}, {32}, {140}, {140}
    };

    for (Index i = 0; i < neural_network.get_layers_number(); ++i)
        EXPECT_EQ(neural_network.get_layer(i)->get_output_shape(), expected_shapes[size_t(i)]);

    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(3)->get_label(), "bottleneck_layer");

    for (Index i = 1; i <= 5; ++i)
    {
        const opennn::Dense* dense =
            dynamic_cast<const opennn::Dense*>(neural_network.get_layer(i).get());
        ASSERT_NE(dense, nullptr);
        EXPECT_EQ(dense->get_activation_function(), ActivationFunction::ReLU);
        EXPECT_TRUE(dense->get_use_bias());
    }

    const opennn::Dense* output =
        dynamic_cast<const opennn::Dense*>(neural_network.get_layer(6).get());
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->get_activation_function(), ActivationFunction::Sigmoid);
    EXPECT_TRUE(output->get_use_bias());
}

TEST(NeuralNetworkTest, AutoAssociationSymmetricEncoderRejectsEmptyEncoder)
{
    EXPECT_THROW(AutoAssociationNetwork({140}, {}, "ReLU", "Sigmoid"), runtime_error);
}

TEST(NeuralNetworkTest, ImageClassificationConstructor)
{
    const Index height = 3;
    const Index width = 3;
    const Index channels = 1;

    const Index complexity = 1;

    const Index outputs_number = 1;

    ImageClassificationNetwork neural_network({height, width, channels}, { complexity }, { outputs_number });

    EXPECT_EQ(neural_network.get_layers_number(), 6);
    EXPECT_EQ(neural_network.get_layer(0)->get_name(), "Scaling");
    EXPECT_EQ(neural_network.get_layer(1)->get_name(), "Convolutional");
    EXPECT_EQ(neural_network.get_layer(2)->get_name(), "Pooling");
    EXPECT_EQ(neural_network.get_layer(3)->get_name(), "Flatten");
    EXPECT_EQ(neural_network.get_layer(4)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(4)->get_label(), "dense_2d_layer_1");
    EXPECT_EQ(neural_network.get_layer(5)->get_name(), "Dense");
    EXPECT_EQ(neural_network.get_layer(5)->get_label(), "classification_layer");
    EXPECT_EQ(neural_network.get_task(), NetworkTask::ImageClassification);
}

TEST(NeuralNetworkTest, ForwardPropagate)
{
    const Index samples_number = 5;
    const Index inputs_number = 2;
    const Index outputs_number = 1;
    const Index neurons_number = 1;

    ApproximationNetwork neural_network_aproximation({inputs_number}, {neurons_number}, {outputs_number});
    neural_network_aproximation.set_parameters_random();

    MatrixR input_data(samples_number, inputs_number);
    input_data << 0, 0,
                  1, 1,
                  2, 2,
                  3, 3,
                  4, 4;

    MatrixR result = neural_network_aproximation.calculate_outputs(input_data);

    EXPECT_EQ(result.rows(), samples_number);
    EXPECT_EQ(result.cols(), outputs_number);

    ClassificationNetwork neural_network_classification({inputs_number}, {neurons_number}, {outputs_number});

    MatrixR result_classification = neural_network_classification.calculate_outputs(input_data);

    EXPECT_EQ(result_classification.rows(), samples_number);
    EXPECT_EQ(result_classification.cols(), outputs_number);
}

TEST(NeuralNetworkTest, CalculateOutputsEmpty)
{
    NeuralNetwork neural_network;

    MatrixR inputs;

    const MatrixR outputs = neural_network.calculate_outputs(inputs);

    EXPECT_EQ(outputs.size(), 0);
}
