// Paired training-quality driver. The Python runner owns scoring and provenance.
// Arguments: manifest output_directory epochs batch seed device precision learning_rate
#include "../tools/quality_dataset.h"
#include "../tools/quality_initialization.h"
#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/models/models.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/network/layers/lstm_layer.h"
#include "opennn/network/operators/embedding_lookup_operator.h"
#include "opennn/training/training.h"
#include "opennn/training/adam.h"
#include "opennn/training/sgd.h"
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

using namespace opennn;

namespace
{
unique_ptr<Network> build(const Json& manifest, const quality::Dataset& data)
{
    const auto model = manifest.at("model").as_string();
    const auto& config = manifest.at("profile");
    if (model == "cnn")
    {
        vector<Index> blocks;
        for (const auto& item : config.at("blocks").as_array()) blocks.push_back(item.as_long());
        auto network = make_unique<ResNet>(data.get_input_shape(), blocks,
            Shape{64,128,256,512}, data.get_target_shape(), true);
        // ResNet's generic fixed-range random initializer is independent of
        // convolution fan-in. Use the fan-aware engine initializer for a deep
        // paired training-quality run.
        network->set_parameters_pytorch();
        return network;
    }
    if (model == "transformer")
    {
        const Index sequence = config.at("sequence").as_long();
        const Index vocabulary = manifest.at("vocabulary").as_array().size();
        auto network = make_unique<Transformer>(sequence, sequence, vocabulary, vocabulary,
            config.at("d_model").as_long(), config.at("heads").as_long(),
            config.at("ff").as_long(), config.at("layers").as_long());
        network->set_dropout_rate(0.0f);
        network->set_parameters_pytorch();
        // PyTorch nn.Embedding starts from N(0, 1). The default Glorot
        // initialization here makes scaled token embeddings much smaller than
        // positional encodings, so initialize the paired token tables alike.
        network->mark_parameters_changed();
        {
            const Network::HostParametersGuard guard(*network);
            for (const char* label : {"encoder_embedding", "decoder_embedding"})
            {
                auto& operators = network->get_layer(label)->get_operators();
                auto* embedding = dynamic_cast<EmbeddingLookupOperator*>(operators.front());
                if (!embedding) throw runtime_error("Transformer embedding lookup missing");
                embedding->set_parameters_random();
            }
        }
        return network;
    }
    auto network = make_unique<Network>();
    Shape current = data.get_input_shape();
    if (model == "lstm")
    {
        network->set_task(NetworkTask::Forecasting);
        auto recurrent = make_unique<LSTM>(current, Shape{config.at("hidden").as_long()}, "Tanh", "Sigmoid", "lstm");
        recurrent->set_return_sequences(false);
        network->add_layer(std::move(recurrent));
    }
    else
    {
        network->set_task(NetworkTask::Classification);
        for (Index i = 0; i < config.at("layers").as_long(); ++i)
        {
            network->add_layer(make_unique<opennn::Dense>(current, Shape{config.at("hidden").as_long()},
                "ReLU", BatchNormalization::No, "dense_" + to_string(i)));
            current = network->get_output_shape();
        }
    }
    network->add_layer(make_unique<opennn::Dense>(network->get_output_shape(), data.get_target_shape(),
        model == "dense" ? "Sigmoid" : "Identity", BatchNormalization::No, "output"));
    network->compile();
    if (model == "lstm") network->set_parameters_pytorch();
    else network->set_parameters_glorot();
    return network;
}

void write_values(ofstream& output, const float* values, Index count)
{
    for (Index i = 0; i < count; ++i)
        if (!std::isfinite(values[i])) throw runtime_error("Nonfinite prediction");
    output.write(reinterpret_cast<const char*>(values), count * sizeof(float));
    if (!output) throw runtime_error("Cannot write predictions");
}

void initial_predictions(Network& network, const quality::Dataset& data,
                         const Json& manifest, const filesystem::path& directory)
{
    const Index count = min(Index(8), data.get_samples_number());
    vector<Index> indices(size_t(count), Index(0));
    iota(indices.begin(), indices.end(), Index(0));
    const auto input = data.read("x", indices);
    const string model = manifest.at("model").as_string();
    ofstream output(directory / "initial_predictions.bin", ios::binary);
    Json dimensions = Json::make_array();
    if (model == "transformer")
    {
        const Index sequence = manifest.at("profile").at("sequence").as_long();
        const auto shifted = data.read("decoder", indices);
        Tensor3 source(count, sequence, 1), decoder(count, sequence, 1);
        std::copy(input.begin(), input.end(), source.data());
        std::copy(shifted.begin(), shifted.end(), decoder.data());
        const Tensor3 logits = network.calculate_outputs(decoder, source);
        write_values(output, logits.data(), logits.size());
        dimensions = json_array(vector<Index>{count, sequence, logits.dimension(2)});
    }
    else
    {
        MatrixR aligned(count, input.size() / size_t(count));
        std::copy(input.begin(), input.end(), aligned.data());
        vector<TensorView> views{TensorView(aligned.data(), Shape{count}.append(data.get_input_shape()),
            Type::FP32, Device::CPU)};
        const MatrixR values = network.calculate_outputs(views);
        write_values(output, values.data(), values.size());
        dimensions = json_array(vector<Index>{count, values.cols()});
    }
    Json record = Json::make_object();
    record.set("shape", std::move(dimensions)).set("dtype", "float32").set("samples", count)
        .set("output", model == "transformer" ? "teacher_forced_logits" : model == "lstm" ? "regression" : "probabilities")
        .set("mode", "evaluation");
    ofstream(directory / "initial_predictions.json") << record.dump();
}

void diagnostic_steps(Network& network, quality::Dataset& data, const Json& manifest,
                       Adam& optimizer, Loss& loss, Index requested_batch, Index steps,
                       const filesystem::path& directory)
{
    if (steps < 1) throw runtime_error("Diagnostic steps must be positive");
    const Index count = min(requested_batch, data.get_samples_number());
    vector<Index> indices(size_t(count), Index(0));
    iota(indices.begin(), indices.end(), Index(0));
    Batch batch(count, &data, network.get_config());
    batch.fill(indices, data.get_feature_selection());
    if (network.is_gpu())
    {
        batch.upload_to_device_batch_async(batch, device::get_compute_stream());
        batch.wait_h2d_on_compute_stream();
    }
    // The diagnostic batch keeps raw image bytes; include the network's input
    // scaling so this is mathematically the same /255 preparation as training.
    TrainingContext context(count, loss, false);
    loss.set_normalization_coefficient();
    OptimizerData moments;
    const Index parameters = network.get_parameters_buffer_size();
    moments.set({Shape{parameters}, Shape{parameters}, Shape{}}, network.get_device());
    quality::SharedInitialization tensors(network, manifest);
    const string model = manifest.at("model").as_string();
    Json records = Json::make_array();
    for (Index step = 0; step < steps; ++step)
    {
        const filesystem::path output = directory / "diagnostic" / ("step_" + to_string(step));
        filesystem::create_directories(output);
        network.forward_propagate(batch.get_inputs(), context.forward, ForwardPropagationMode::Training);
        const TensorView values = context.forward.get_outputs();
        vector<float> host(size_t(values.size()));
        if (values.get_device() == Device::CPU)
            std::copy_n(values.as<float>(), host.size(), host.data());
        else copy_device_to_host_float(values.get_data(), values.get_type(), values.size(),
            host.data(), device::get_compute_stream());
        ofstream binary(output / "outputs.bin", ios::binary);
        write_values(binary, host.data(), host.size());
        Json specification = Json::make_object();
        specification.set("shape", json_array(values.get_shape())).set("dtype", "float32")
            .set("samples", count).set("mode", "training")
            .set("output", model == "transformer" ? "logits" : model == "lstm" ? "regression" : "probabilities");
        ofstream(output / "outputs.json") << specification.dump();
        loss.back_propagate(batch, context.forward, context.backward);
        if (!std::isfinite(context.backward.metrics.loss_value)) throw runtime_error("Nonfinite diagnostic loss");
        tensors.save(output / "gradients", true);
        optimizer.update_parameters(context.backward, moments);
        tensors.save(output / "parameters");
        Json record = Json::make_object();
        record.set("step", step).set("loss", context.backward.metrics.loss_value).set("samples", count);
        records.push_back(std::move(record));
    }
    Json diagnostic = Json::make_object();
    diagnostic.set("steps", std::move(records)).set("mode", "fixed_batch_training");
    ofstream(directory / "diagnostic.json") << diagnostic.dump();
}

Json teacher_forced_probe(Network& network, const quality::Dataset& data, Index sequence)
{
    const Index count = min(Index(8), data.get_samples_number());
    vector<Index> indices;
    indices.reserve(size_t(count));
    for (Index i = 0; i < count; ++i)
        indices.push_back(i);

    const auto x = data.read("x", indices);
    const auto shifted = data.read("decoder", indices);
    const auto targets = data.read("y", indices);
    Tensor3 source(count, sequence, 1), decoder(count, sequence, 1);
    std::copy(x.begin(), x.end(), source.data());
    std::copy(shifted.begin(), shifted.end(), decoder.data());
    const Tensor3 logits = network.calculate_outputs(decoder, source);
    const Index vocabulary = logits.dimension(2);

    Index active = 0, correct = 0;
    double loss_sum = 0.0;
    std::map<Index, Index> predicted_counts;
    for (Index sample = 0; sample < count; ++sample)
        for (Index position = 0; position < sequence; ++position)
        {
            const Index target = static_cast<Index>(targets[size_t(sample * sequence + position)]);
            if (target <= 0 || target >= vocabulary) continue;
            float maximum = -numeric_limits<float>::infinity();
            Index best = 0;
            for (Index token = 0; token < vocabulary; ++token)
            {
                const float value = logits(sample, position, token);
                if (value > maximum) { maximum = value; best = token; }
            }
            double exponential_sum = 0.0;
            for (Index token = 0; token < vocabulary; ++token)
                exponential_sum += exp(double(logits(sample, position, token) - maximum));
            loss_sum += log(exponential_sum) + maximum - logits(sample, position, target);
            ++active;
            correct += best == target;
            ++predicted_counts[best];
        }

    Index dominant_token = -1, dominant_count = 0;
    for (const auto& [token, frequency] : predicted_counts)
        if (frequency > dominant_count) { dominant_token = token; dominant_count = frequency; }
    Json record = Json::make_object();
    record.set("samples", count).set("active_tokens", active)
        .set("mean_cross_entropy", active ? loss_sum / active : 0.0)
        .set("token_accuracy_percent", active ? 100.0 * correct / active : 0.0)
        .set("dominant_prediction_id", dominant_token)
        .set("dominant_prediction_fraction", active ? double(dominant_count) / active : 0.0)
        .set("unique_prediction_ids", Index(predicted_counts.size()));
    return record;
}

Json cnn_mode_probe(Network& network, const quality::Dataset& data, bool training_mode)
{
    const Index count = min(Index(32), data.get_samples_number());
    vector<Index> indices(size_t(count), Index(0));
    for (Index i = 0; i < count; ++i)
        indices[size_t(i)] = i * data.get_samples_number() / count;
    const auto input = data.read("x", indices);
    const auto target = data.read("y", indices);
    MatrixR aligned(count, input.size() / size_t(count));
    std::copy(input.begin(), input.end(), aligned.data());
    if (training_mode) aligned.array() /= 255.0f;
    vector<TensorView> views{TensorView(aligned.data(),
        Shape{count}.append(data.get_input_shape()), Type::FP32, Device::CPU)};
    MatrixR probabilities;
    if (training_mode)
    {
        ForwardPropagation forward(count, &network,
            ForwardPropagationMode::Training, {}, training_mode);
        network.forward_propagate(views, forward, ForwardPropagationMode::Training);
        const TensorView output = forward.get_outputs();
        probabilities.resize(count, output.size() / count);
        copy_device_to_host_float(output.get_data(), output.get_type(), output.size(),
            probabilities.data(), device::get_compute_stream());
    }
    else probabilities = network.calculate_outputs(views);

    Index correct = 0;
    double max_probability = 0.0;
    map<Index, Index> counts;
    for (Index i = 0; i < count; ++i)
    {
        Index predicted = 0, expected = 0;
        for (Index j = 1; j < probabilities.cols(); ++j)
        {
            if (probabilities(i, j) > probabilities(i, predicted)) predicted = j;
            if (target[size_t(i * probabilities.cols() + j)] >
                target[size_t(i * probabilities.cols() + expected)]) expected = j;
        }
        correct += predicted == expected;
        max_probability += probabilities(i, predicted);
        ++counts[predicted];
    }
    Index dominant = 0;
    for (const auto& [_, frequency] : counts) dominant = max(dominant, frequency);
    Json record = Json::make_object();
    record.set("samples", count)
        .set("accuracy_percent", 100.0 * correct / count)
        .set("unique_predictions", Index(counts.size()))
        .set("dominant_fraction", double(dominant) / count)
        .set("mean_max_probability", max_probability / count);
    return record;
}

void recalibrate_cnn_batchnorm(Network& network, const quality::Dataset& data, Index batch)
{
    vector<BatchNormalizationOperator*> normalizations;
    for (const auto& layer : network.get_layers())
        for (Operator* op : layer->get_operators())
            if (auto* normalization = dynamic_cast<BatchNormalizationOperator*>(op))
                if (normalization->active()) normalizations.push_back(normalization);
    if (normalizations.empty()) throw runtime_error("CNN has no BatchNorm state to recalibrate");
    const bool states_on_device = network.is_gpu();
    if (states_on_device) network.copy_states_host();
    for (auto* normalization : normalizations) normalization->initialize_states();
    if (states_on_device) network.copy_states_device();
    Index seen = 0;
    for (Index start = 0; start < data.get_samples_number(); start += batch)
    {
        const Index count = min(batch, data.get_samples_number() - start);
        seen += count;
        const float momentum = float(count) / float(seen);
        for (auto* normalization : normalizations)
            // Exact cumulative averaging starts with momentum 1. The public
            // setter reserves that endpoint even though every update backend
            // accepts it, so assign the public field for this calibration pass.
            normalization->momentum = momentum;
        vector<Index> indices(size_t(count), Index(0));
        iota(indices.begin(), indices.end(), start);
        const auto input = data.read("x", indices);
        MatrixR aligned(count, input.size() / size_t(count));
        std::copy(input.begin(), input.end(), aligned.data());
        aligned.array() /= 255.0f;
        vector<TensorView> views{TensorView(aligned.data(),
            Shape{count}.append(data.get_input_shape()), Type::FP32, Device::CPU)};
        ForwardPropagation forward(count, &network, ForwardPropagationMode::Training, {}, true);
        network.forward_propagate(views, forward, ForwardPropagationMode::Training);
    }
    for (auto* normalization : normalizations)
        normalization->momentum = 0.1f;
}
}

int main(int argc, char** argv)
{
    try
    {
        if (argc != 9) throw runtime_error("usage: quality_opennn manifest out epochs batch seed cpu|cuda fp32|bf16 learning_rate");
        if (std::endian::native != std::endian::little) throw runtime_error("Quality tensors require little-endian host");
        const filesystem::path manifest_path(argv[1]), out(argv[2]);
        const Index epochs = stoll(argv[3]), batch = stoll(argv[4]), seed = stoll(argv[5]);
        const string device_name(argv[6]), precision_name(argv[7]);
        if (epochs < 1 || batch < 1 || (device_name != "cpu" && device_name != "cuda")
            || (precision_name != "fp32" && precision_name != "bf16")) throw runtime_error("Invalid quality options");
        const bool gpu = device_name == "cuda";
        const char* deterministic_env = std::getenv("OPENNN_QUALITY_DETERMINISTIC");
        const bool deterministic = deterministic_env && string_view(deterministic_env) == "1";
        if (!gpu && precision_name != "fp32") throw runtime_error("CPU quality uses FP32");
        Configuration::instance().set(gpu ? Device::CUDA : Device::CPU,
            precision_name == "bf16" ? Type::BF16 : Type::FP32);
        Configuration::instance().set_blas(Blas::Mkl);
        set_seed(seed);
        JsonDocument document; document.load(manifest_path);
        const auto& manifest = document.get_root();
        const auto model = manifest.at("model").as_string();
        quality::Dataset train(manifest_path.parent_path(), manifest.at("train"));
        quality::Dataset test(manifest_path.parent_path(), manifest.at("test"));
        auto network = build(manifest, train);
        if (network->is_gpu() != gpu) throw runtime_error("Unexpected device fallback");
        filesystem::create_directories(out);
        const char* checkpoint_env = std::getenv("OPENNN_QUALITY_LOAD_CHECKPOINT");
        const bool evaluation_only = checkpoint_env && *checkpoint_env;
        const filesystem::path checkpoint = evaluation_only ? checkpoint_env : "";
        Json initialization = Json::make_object();
        initialization.set("mode", "native");
        const char* initialization_env = std::getenv("OPENNN_QUALITY_INITIALIZATION");
        if (initialization_env && *initialization_env)
        {
            if (evaluation_only) throw runtime_error("Shared initialization cannot be combined with checkpoint evaluation");
            quality::SharedInitialization(*network, manifest).load(initialization_env, model, seed);
            filesystem::path path(initialization_env);
            if (filesystem::is_directory(path)) path /= "manifest.json";
            JsonDocument initial_document;
            initial_document.load(path);
            initialization.set("mode", "shared").set("manifest", path.string()).set("seed", seed)
                .set("tensor_count", initial_document.get_root().at("tensors").as_array().size());
        }
        if (!evaluation_only && std::getenv("OPENNN_QUALITY_EXPORT_INITIALIZATION"))
            quality::SharedInitialization(*network, manifest).save(out / "initialization");
        if (!evaluation_only && std::getenv("OPENNN_QUALITY_INITIAL_PREDICTIONS"))
            initial_predictions(*network, test, manifest, out);
        ofstream history(out / "history.csv");
        history << "epoch,training_loss\n";
        Training training(network.get(), &train);
        training.set_loss(model == "transformer" ? "CrossEntropyError3d" : model == "lstm" ? "MeanSquaredError" : "CrossEntropy");
        const char* weight_decay_env = std::getenv("OPENNN_QUALITY_WEIGHT_DECAY");
        const float weight_decay = weight_decay_env && *weight_decay_env
            ? stof(weight_decay_env) : 0.0f;
        if (!std::isfinite(weight_decay) || weight_decay < 0.0f)
            throw runtime_error("Weight decay must be finite and nonnegative");
        training.get_loss()->set_regularization(
            weight_decay > 0.0f ? "L2" : "NoRegularization");
        training.get_loss()->set_regularization_weight(weight_decay);
        const char* optimizer_env = std::getenv("OPENNN_QUALITY_OPTIMIZER");
        const string optimizer_name = optimizer_env && *optimizer_env ? optimizer_env : "adam";
        vector<Index> learning_rate_milestones;
        if (const char* milestones_env = std::getenv("OPENNN_QUALITY_LR_MILESTONES");
            milestones_env && *milestones_env)
        {
            stringstream stream(milestones_env);
            string value;
            while (getline(stream, value, ','))
                learning_rate_milestones.push_back(stoll(value));
        }
        const char* gamma_env = std::getenv("OPENNN_QUALITY_LR_GAMMA");
        const float learning_rate_gamma = gamma_env && *gamma_env ? stof(gamma_env) : 0.1f;
        if (!std::is_sorted(learning_rate_milestones.begin(), learning_rate_milestones.end())
            || adjacent_find(learning_rate_milestones.begin(), learning_rate_milestones.end())
                != learning_rate_milestones.end()
            || any_of(learning_rate_milestones.begin(), learning_rate_milestones.end(),
                [epochs](Index value) { return value <= 0 || value >= epochs; }))
            throw runtime_error("Learning-rate milestones must be distinct, increasing, and inside the training budget");
        if (!std::isfinite(learning_rate_gamma) || learning_rate_gamma <= 0.0f
            || learning_rate_gamma >= 1.0f)
            throw runtime_error("Learning-rate gamma must be finite and in (0, 1)");
        if (!learning_rate_milestones.empty() && optimizer_name != "sgd")
            throw runtime_error("Learning-rate milestones currently require SGD");
        const char* adam_epsilon_env = std::getenv("OPENNN_QUALITY_ADAM_EPSILON");
        const float adam_epsilon = adam_epsilon_env && *adam_epsilon_env
            ? stof(adam_epsilon_env) : EPSILON;
        if (!std::isfinite(adam_epsilon) || adam_epsilon <= 0.0f)
            throw runtime_error("Invalid Adam epsilon");
        if (optimizer_name != "adam" && optimizer_name != "sgd")
            throw runtime_error("Unsupported quality optimizer: " + optimizer_name);
        training.set_optimization_algorithm(optimizer_name == "adam" ? "Adam" : "SGD");
        auto* optimizer = training.get_optimization_algorithm();
        SGD* sgd_optimizer = nullptr;
        if (auto* adam = dynamic_cast<Adam*>(optimizer))
        {
            adam->set_learning_rate(stof(argv[8]));
            adam->set_beta_1(0.9f); adam->set_beta_2(0.999f);
            adam->set_epsilon(adam_epsilon);
            adam->set_bf16_first_moment(false);
        }
        else
        {
            auto* sgd = dynamic_cast<SGD*>(optimizer);
            sgd_optimizer = sgd;
            const char* momentum_env = std::getenv("OPENNN_QUALITY_MOMENTUM");
            const float momentum = momentum_env && *momentum_env ? stof(momentum_env) : 0.9f;
            if (momentum < 0.0f || momentum >= 1.0f) throw runtime_error("Invalid SGD momentum");
            sgd->set_initial_learning_rate(stof(argv[8]));
            sgd->set_initial_decay(0.0f);
            sgd->set_momentum(momentum);
            sgd->set_nesterov(false);
        }
        optimizer->set_batch_size(batch);
        optimizer->set_maximum_epochs(epochs);
        optimizer->set_maximum_time(numeric_limits<float>::max());
        optimizer->set_loss_goal(-numeric_limits<float>::max());
        optimizer->set_restore_best(false);
        optimizer->set_gradient_clip_norm(0.0f);
        optimizer->set_shuffle(false); // Same prepared sample order and all tail samples.
        optimizer->set_cuda_graph(false); // Quality, not throughput or cache warm-up.
        optimizer->set_display(false);
        const char* diagnostic_env = std::getenv("OPENNN_QUALITY_DIAGNOSTIC_STEPS");
        const Index diagnostic_count = diagnostic_env && *diagnostic_env ? stoll(diagnostic_env) : 0;
        if (diagnostic_count < 0) throw runtime_error("Diagnostic step count must be nonnegative");
        if (diagnostic_count > 0)
        {
            if (evaluation_only) throw runtime_error("Diagnostics cannot be combined with checkpoint evaluation");
            auto* adam = dynamic_cast<Adam*>(optimizer);
            if (!adam) throw runtime_error("Numerical diagnostics currently require Adam");
            diagnostic_steps(*network, train, manifest, *adam, *training.get_loss(),
                batch, diagnostic_count, out);
            Json record = Json::make_object();
            record.set("engine", "opennn").set("model", model).set("device", device_name)
                .set("precision", precision_name).set("seed", seed).set("diagnostic_only", true)
                .set("deterministic", deterministic)
                .set("optimizer", optimizer_name)
                .set("adam_epsilon", adam_epsilon)
                .set("weight_decay", weight_decay)
                .set("batchnorm_recalibrated", false)
                .set("diagnostic_steps", diagnostic_count)
                .set("initialization", std::move(initialization)).set("epochs", 0).set("status", "ok");
            ofstream(out / "driver.json") << record.dump();
            cout << "RESULT=DIAGNOSTIC\n";
            return 0;
        }
        Index completed = 0;
        optimizer->post_epoch_callback = [&](Index epoch, float loss, float, Network*) {
            if (!std::isfinite(loss)) throw runtime_error("Nonfinite training loss");
            ++completed;
            history << epoch + 1 << ',' << setprecision(9) << loss << '\n' << flush;
            cout << "epoch=" << epoch + 1 << " training_loss=" << loss << '\n' << flush;
            if (sgd_optimizer && binary_search(learning_rate_milestones.begin(),
                                               learning_rate_milestones.end(), epoch + 1))
                sgd_optimizer->set_initial_learning_rate(
                    sgd_optimizer->get_initial_learning_rate() * learning_rate_gamma);
        };
        if (evaluation_only)
        {
            network->load_parameters_binary(checkpoint / "parameters.bin");
            network->load_states_binary(checkpoint / "states.bin");
            completed = epochs; // Training budget of the checkpoint, checked by the parent runner.
        }
        else
        {
            training.train();
            if (completed != epochs) throw runtime_error("Training stopped before the requested budget");
            if (std::getenv("OPENNN_QUALITY_SAVE_CHECKPOINT"))
            {
                network->save_parameters_binary(out / "parameters.bin");
                network->save_states_binary(out / "states.bin");
            }
        }
        const bool batchnorm_recalibrated = model == "cnn"
            && std::getenv("OPENNN_QUALITY_RECALIBRATE_BATCHNORM");
        if (batchnorm_recalibrated) recalibrate_cnn_batchnorm(*network, train, batch);
        if (std::getenv("OPENNN_QUALITY_EXPORT_FINAL"))
            quality::SharedInitialization(*network, manifest).save(out / "final");
        if (model == "cnn" && std::getenv("OPENNN_QUALITY_CNN_PROBE"))
        {
            const filesystem::path snapshot = out / "probe-states.bin";
            network->save_states_binary(snapshot);
            Json probe = Json::make_object();
            probe.set("train_inference", cnn_mode_probe(*network, train, false))
                 .set("test_inference", cnn_mode_probe(*network, test, false))
                 .set("train_batch", cnn_mode_probe(*network, train, true))
                 .set("test_batch", cnn_mode_probe(*network, test, true));
            network->load_states_binary(snapshot);
            ofstream(out / "cnn_mode_probe.json") << probe.dump();
            if (string_view(std::getenv("OPENNN_QUALITY_CNN_PROBE")) == "only")
            {
                cout << "RESULT=PROBE\n";
                return 0;
            }
        }
        if (model == "transformer" && std::getenv("OPENNN_QUALITY_PROBE"))
        {
            const Index sequence = manifest.at("profile").at("sequence").as_long();
            Json probe = Json::make_object();
            probe.set("train", teacher_forced_probe(*network, train, sequence))
                 .set("test", teacher_forced_probe(*network, test, sequence));
            ofstream(out / "teacher_forced_probe.json") << probe.dump();
            if (string_view(std::getenv("OPENNN_QUALITY_PROBE")) == "only")
            {
                cout << "RESULT=PROBE\n";
                return 0;
            }
        }
        ofstream predictions(out / "predictions.bin", ios::binary);
        for (Index start = 0; start < test.get_samples_number(); start += batch)
        {
            const Index count = min(batch, test.get_samples_number() - start);
            vector<Index> indices(size_t(count), Index(0));
            iota(indices.begin(), indices.end(), start);
            auto input = test.read("x", indices);
            if (model == "transformer")
            {
                const Index sequence = manifest.at("profile").at("sequence").as_long();
                Tensor3 source(count, sequence, 1), decoder(count, sequence, 1);
                std::copy(input.begin(), input.end(), source.data());
                decoder.setZero();
                vector<float> generated(size_t(count * sequence), 0.0f);
                vector<bool> done(size_t(count), false);
                for (Index i = 0; i < count; ++i) decoder(i, 0, 0) = 2.0f;
                for (Index position = 0; position < sequence; ++position)
                {
                    // Transformer binds its decoder to external input 0 and
                    // its encoder to input 1, matching Batch::get_inputs().
                    const Tensor3 logits = network->calculate_outputs(decoder, source);
                    const Index vocab = logits.dimension(2);
                    for (Index i = 0; i < count; ++i)
                    {
                        if (done[i]) continue;
                        Index best = 0;
                        for (Index token = 0; token < vocab; ++token)
                        {
                            if (!std::isfinite(logits(i, position, token))) throw runtime_error("Nonfinite generation logits");
                            if (logits(i, position, token) > logits(i, position, best)) best = token;
                        }
                        generated[i * sequence + position] = float(best);
                        done[i] = best == 3;
                        if (position + 1 < sequence) decoder(i, position + 1, 0) = float(best);
                    }
                    if (all_of(done.begin(), done.end(), [](bool value) { return value; })) break;
                }
                write_values(predictions, generated.data(), generated.size());
            }
            else
            {
                Shape shape = Shape{count}.append(test.get_input_shape());
                // TensorView's Eigen maps require aligned storage; std::vector
                // does not guarantee the SIMD alignment of MatrixR/Tensor4.
                MatrixR aligned(count, input.size() / count);
                std::copy(input.begin(), input.end(), aligned.data());
                vector<TensorView> views{TensorView(aligned.data(), shape, Type::FP32, Device::CPU)};
                const MatrixR values = network->calculate_outputs(views);
                write_values(predictions, values.data(), values.size());
            }
        }
        predictions.close();
        Json record = Json::make_object();
        record.set("engine", "opennn").set("device", device_name).set("precision", precision_name)
            .set("model", model).set("seed", seed).set("epochs", completed)
            .set("deterministic", deterministic)
            .set("optimizer", optimizer_name)
            .set("adam_epsilon", adam_epsilon)
            .set("weight_decay", weight_decay)
            .set("learning_rate_milestones", json_array(learning_rate_milestones))
            .set("learning_rate_gamma", learning_rate_gamma)
            .set("batchnorm_recalibrated", batchnorm_recalibrated)
            .set("parameters", network->get_parameters_number()).set("train_samples", train.get_samples_number())
            .set("test_samples", test.get_samples_number()).set("evaluation_only", evaluation_only)
            .set("initialization", std::move(initialization))
            .set("status", "ok");
        ofstream(out / "driver.json") << record.dump();
        cout << "RESULT=OK\n";
        return 0;
    }
    catch (const exception& error)
    {
        cerr << error.what() << '\n';
        cout << "RESULT=ERROR\n";
        return 1;
    }
}
