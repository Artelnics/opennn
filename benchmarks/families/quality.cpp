// Paired training-quality driver. The Python runner owns scoring and provenance.
// Arguments: manifest output_directory epochs batch seed device precision learning_rate
#include "../tools/quality_dataset.h"
#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/models/models.h"
#include "opennn/network/layers/dense_layer.h"
#include "opennn/network/layers/lstm_layer.h"
#include "opennn/training/training.h"
#include "opennn/training/adam.h"
#include <bit>
#include <cmath>
#include <iomanip>
#include <iostream>

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
        return make_unique<ResNet>(data.get_input_shape(), blocks, Shape{64,128,256,512}, data.get_target_shape(), true);
    }
    if (model == "transformer")
    {
        const Index sequence = config.at("sequence").as_long();
        const Index vocabulary = manifest.at("vocabulary").as_array().size();
        auto network = make_unique<Transformer>(sequence, sequence, vocabulary, vocabulary,
            config.at("d_model").as_long(), config.at("heads").as_long(),
            config.at("ff").as_long(), config.at("layers").as_long());
        network->set_dropout_rate(0.0f);
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
        ofstream history(out / "history.csv");
        history << "epoch,training_loss\n";
        Training training(network.get(), &train);
        training.set_loss(model == "transformer" ? "CrossEntropyError3d" : model == "lstm" ? "MeanSquaredError" : "CrossEntropy");
        training.get_loss()->set_regularization("NoRegularization");
        training.set_optimization_algorithm("Adam");
        auto* optimizer = dynamic_cast<Adam*>(training.get_optimization_algorithm());
        optimizer->set_learning_rate(stof(argv[8]));
        optimizer->set_beta_1(0.9f); optimizer->set_beta_2(0.999f);
        optimizer->set_bf16_first_moment(false);
        optimizer->set_batch_size(batch);
        optimizer->set_maximum_epochs(epochs);
        optimizer->set_maximum_time(numeric_limits<float>::max());
        optimizer->set_loss_goal(-numeric_limits<float>::max());
        optimizer->set_restore_best(false);
        optimizer->set_gradient_clip_norm(0.0f);
        optimizer->set_shuffle(false); // Same prepared sample order and all tail samples.
        optimizer->set_cuda_graph(false); // Quality, not throughput or cache warm-up.
        optimizer->set_display(false);
        Index completed = 0;
        optimizer->post_epoch_callback = [&](Index epoch, float loss, float, Network*) {
            if (!std::isfinite(loss)) throw runtime_error("Nonfinite training loss");
            ++completed;
            history << epoch + 1 << ',' << setprecision(9) << loss << '\n' << flush;
            cout << "epoch=" << epoch + 1 << " training_loss=" << loss << '\n' << flush;
        };
        training.train();
        if (completed != epochs) throw runtime_error("Training stopped before the requested budget");
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
                    const Tensor3 logits = network->calculate_outputs(source, decoder);
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
            .set("parameters", network->get_parameters_number()).set("train_samples", train.get_samples_number())
            .set("test_samples", test.get_samples_number()).set("status", "ok");
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
