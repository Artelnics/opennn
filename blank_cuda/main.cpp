//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   C U D A
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

#include "../opennn/image_dataset.h"
#include "../opennn/image_utilities.h"
#include "../opennn/language_dataset.h"
#include "../opennn/tabular_dataset.h"
#include "../opennn/standard_networks.h"
#include "../opennn/scaling_layer.h"
#include "../opennn/convolutional_layer.h"
#include "../opennn/pooling_layer.h"
#include "../opennn/flatten_layer.h"
#include "../opennn/dense_layer.h"
#include "../opennn/addition_layer.h"
#include "../opennn/activation_layer.h"
#include "../opennn/training_strategy.h"
#include "../opennn/testing_analysis.h"
#include "../opennn/stochastic_gradient_descent.h"
#include "../opennn/adaptive_moment_estimation.h"
#include "../opennn/random_utilities.h"
#include "../opennn/transformer_decoder.h"

using namespace opennn;
using namespace std::chrono;

#if 0
// ----- SDPA vs unfused benchmark scaffolding (temporary) -----
// Writes N random (input, target) pairs to dst, each pair containing exactly
// `tokens_per_side` tokens drawn from a vocabulary of `vocab_size` synthetic
// words. LanguageDataset then pads up to maximum_input_sequence_length =
// tokens_per_side + 2, giving deterministic control over the model seq_len.
static void write_synthetic_pairs(const std::filesystem::path& dst,
                                  Index n_pairs,
                                  Index tokens_per_side,
                                  Index vocab_size)
{
    if (std::filesystem::exists(dst)) return;

    std::ofstream out(dst);
    if (!out.is_open())
        throw runtime_error("write_synthetic_pairs: cannot open " + dst.string());

    for (Index i = 0; i < n_pairs; ++i)
    {
        for (Index j = 0; j < tokens_per_side; ++j)
        {
            if (j > 0) out << ' ';
            out << "tok" << random_integer(0, int(vocab_size) - 1);
        }
        out << '\t';
        for (Index j = 0; j < tokens_per_side; ++j)
        {
            if (j > 0) out << ' ';
            out << "tok" << random_integer(0, int(vocab_size) - 1);
        }
        out << '\n';
    }
}
// ----- end SDPA bench scaffolding -----

// tinychat.txt format: each line is one conversation with multiple
// `[INST] user [/INST] assistant` turns concatenated. We extract every turn as
// an independent (user, assistant) pair and write them tab-separated, which is
// the format LanguageDataset::read_csv expects (one input/target per line).
//
// `max_pairs == 0` means no limit. Output is cached at `dst`; subsequent runs
// skip preprocessing.
static void prepare_tinychat_pairs(const std::filesystem::path& src,
                                   const std::filesystem::path& dst,
                                   Index max_pairs = 0)
{
    if (std::filesystem::exists(dst))
    {
        cout << "Reusing cached pairs file at " << dst << endl;
        return;
    }

    cout << "Preprocessing " << src << " -> " << dst
         << (max_pairs > 0 ? " (limit " + to_string(max_pairs) + " pairs)" : "")
         << " ..." << endl;

    std::ifstream in(src);
    if (!in.is_open()) throw runtime_error("Cannot open " + src.string());

    std::ofstream out(dst);
    if (!out.is_open()) throw runtime_error("Cannot open " + dst.string() + " for writing");

    const string open_tag  = "[INST]";
    const string close_tag = "[/INST]";

    auto sanitize = [](string& s)
    {
        const size_t first = s.find_first_not_of(" \t\r\n");
        const size_t last  = s.find_last_not_of(" \t\r\n");
        if (first == string::npos) { s.clear(); return; }
        s = s.substr(first, last - first + 1);
        for (char& c : s) if (c == '\t' || c == '\n' || c == '\r') c = ' ';
    };

    Index pair_count = 0;
    string line;
    while (std::getline(in, line))
    {
        if (max_pairs > 0 && pair_count >= max_pairs) break;

        size_t cursor = 0;
        while (cursor < line.size())
        {
            if (max_pairs > 0 && pair_count >= max_pairs) break;

            const size_t open_i = line.find(open_tag, cursor);
            if (open_i == string::npos) break;
            const size_t close_i = line.find(close_tag, open_i + open_tag.size());
            if (close_i == string::npos) break;
            const size_t next_open = line.find(open_tag, close_i + close_tag.size());
            const size_t resp_end  = (next_open == string::npos) ? line.size() : next_open;

            string user      = line.substr(open_i  + open_tag.size(),  close_i - (open_i  + open_tag.size()));
            string assistant = line.substr(close_i + close_tag.size(), resp_end - (close_i + close_tag.size()));
            sanitize(user);
            sanitize(assistant);

            if (!user.empty() && !assistant.empty())
            {
                out << user << '\t' << assistant << '\n';
                ++pair_count;
            }

            cursor = (next_open == string::npos) ? line.size() : next_open;
        }
    }

    cout << "Wrote " << pair_count << " (user, assistant) pairs to " << dst << endl;
}
#endif

int main()
{
    try
    {
#if 0
        cout << "OpenNN. ResNet-50 ImageNet single-image test." << endl;

        Configuration::instance().set(Device::CPU, Type::FP32);

        const filesystem::path imagenette_path = "/home/artelnics/Documents/imagenette2_bmp_224";
        const filesystem::path resnet50_dir = "/home/artelnics/Documents/resnet-50";
        const filesystem::path parameters_path = resnet50_dir / "resnet50_imagenet1k_v2_opennn_parameters.bin";
        const filesystem::path states_path = resnet50_dir / "resnet50_imagenet1k_v2_opennn_states.bin";
        const filesystem::path categories_path = resnet50_dir / "imagenet1k_categories.txt";

        if (!filesystem::exists(imagenette_path))
            throw runtime_error("Imagenette folder not found: " + imagenette_path.string());
        if (!filesystem::exists(parameters_path))
            throw runtime_error("ResNet-50 parameters not found: " + parameters_path.string());
        if (!filesystem::exists(states_path))
            throw runtime_error("ResNet-50 states not found: " + states_path.string());
        if (!filesystem::exists(categories_path))
            throw runtime_error("ImageNet categories file not found: " + categories_path.string());

        vector<string> categories;
        {
            ifstream in(categories_path);
            string line;
            while (getline(in, line)) categories.push_back(line);
        }
        if (categories.size() != 1000)
            throw runtime_error("Expected 1000 ImageNet categories, got " + to_string(categories.size()));

        ResNet network({224, 224, 3}, {3, 4, 6, 3}, Shape{64, 128, 256, 512}, Shape{1000}, true);

        auto* scaling = dynamic_cast<Scaling*>(network.get_first(LayerType::Scaling));
        if (!scaling)
            throw runtime_error("ResNet scaling layer not found.");

        scaling->set_descriptives({
            Descriptives(0.0f, 255.0f, 0.485f * 255.0f, 0.229f * 255.0f),
            Descriptives(0.0f, 255.0f, 0.456f * 255.0f, 0.224f * 255.0f),
            Descriptives(0.0f, 255.0f, 0.406f * 255.0f, 0.225f * 255.0f)
        });
        scaling->set_scalers("MeanStandardDeviation");

        cout << "ResNet-50 params=" << network.get_parameters_number()
             << " (buffer=" << network.get_parameters_size() << ")" << endl;
        cout << "Loading parameters from " << parameters_path << endl;
        network.load_parameters_binary(parameters_path);
        cout << "Loading states from " << states_path << endl;
        network.load_states_binary(states_path);

        TestingAnalysis testing_analysis(&network, nullptr);

        vector<filesystem::path> images;
        for (const auto& entry : filesystem::directory_iterator(imagenette_path))
        {
            if (!entry.is_directory() || entry.path().filename().string().starts_with("."))
                continue;

            for (const auto& image_entry : filesystem::recursive_directory_iterator(entry.path()))
                if (image_entry.is_regular_file() && is_supported_image_file(image_entry.path()))
                    images.push_back(image_entry.path());
        }

        sort(images.begin(), images.end());
        if (images.empty())
            throw runtime_error("No images found in " + imagenette_path.string());

        mt19937 rng(random_device{}());
        uniform_int_distribution<size_t> pick(0, images.size() - 1);
        const filesystem::path image_path = images[pick(rng)];

        Tensor4 image(1, 224, 224, 3);
        load_image(image_path, image.data(), 224, 224, 3, false);
        MatrixR output = network.calculate_outputs(image);

        vector<Index> top_indices(1000);
        iota(top_indices.begin(), top_indices.end(), 0);
        partial_sort(top_indices.begin(),
                     top_indices.begin() + 5,
                     top_indices.end(),
                     [&](Index a, Index b) { return output(0, a) > output(0, b); });

        cout << "\nImage: " << image_path << endl;
        cout << "Folder label: " << image_path.parent_path().filename().string() << endl;
        cout << "\nTop-5 predictions:" << endl;

        cout << fixed << setprecision(6);
        for (Index i = 0; i < 5; ++i)
        {
            const Index class_index = top_indices[size_t(i)];
            cout << i + 1 << ". [" << class_index << "] "
                 << categories[size_t(class_index)]
                 << " = " << output(0, class_index) << endl;
        }

        cout << "Bye!" << endl;
        return 0;
#endif

        cout << "OpenNN. HIGGS 5x300 DNN GPU FP32 batch-1M benchmark." << endl;

#ifdef OPENNN_HAS_CUDA
        Configuration::instance().set(Device::CUDA, Type::FP32);
        Backend::instance();
#else
        throw runtime_error("OpenNN was built without CUDA support.");
#endif

        const filesystem::path dataset_path = "/home/artelnics/Documents/HIGGS.csv";

        TabularDataset dataset(dataset_path, ",", false, false);

        vector<Index> input_variables(28);
        iota(input_variables.begin(), input_variables.end(), 1);
        dataset.set_variable_indices(input_variables, {0});
        dataset.set_variable_type(0, VariableType::Binary);
        dataset.split_samples_sequential(10.0f / 11.0f, 0.5f / 11.0f, 0.5f / 11.0f);

        const Index training_samples = dataset.get_sample_indices("Training").size();
        const Index validation_samples = dataset.get_sample_indices("Validation").size();
        const Index testing_samples = dataset.get_sample_indices("Testing").size();

        cout << "[DATASET] train=" << training_samples
             << " val="          << validation_samples
             << " test="         << testing_samples
             << " input="        << dataset.get_shape("Input")[0]
             << " target="       << dataset.get_shape("Target")[0]
             << endl;

        NeuralNetwork network;
        network.add_layer(make_unique<Scaling>(dataset.get_shape("Input")));

        for (Index i = 0; i < 5; ++i)
            network.add_layer(make_unique<opennn::Dense>(network.get_output_shape(),
                                                         Shape{300},
                                                         "Tanh",
                                                         false,
                                                         format("higgs_hidden_{}", i + 1)));

        network.add_layer(make_unique<opennn::Dense>(network.get_output_shape(),
                                                     dataset.get_shape("Target"),
                                                     "Sigmoid",
                                                     false,
                                                     "higgs_output"));
        network.compile();
        network.set_parameters_glorot();

        cout << "HIGGS DNN params=" << network.get_parameters_number()
             << " (buffer=" << network.get_parameters_size() << ")" << endl;

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.set_loss("CrossEntropy");
        training_strategy.set_optimization_algorithm("StochasticGradientDescent");
        training_strategy.get_loss()->set_regularization("NoRegularization");

        auto* sgd = dynamic_cast<StochasticGradientDescent*>(training_strategy.get_optimization_algorithm());
        if (!sgd)
            throw runtime_error("StochasticGradientDescent optimizer not found.");

        sgd->set_batch_size(1'000'000);
        sgd->set_initial_learning_rate(0.05f);
        sgd->set_initial_decay(0.0202f);
        sgd->set_momentum(0.9f);
        sgd->set_nesterov(false);
        sgd->set_num_workers(8);
        sgd->set_maximum_epochs(4);
        sgd->set_maximum_validation_failures(100);
        sgd->set_display_period(1);

        const auto t0 = steady_clock::now();
        training_strategy.train();
        const auto t1 = steady_clock::now();
        const double training_time = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;

        cout << "\nTotal training time: " << training_time << " s" << endl;

        cout << "Bye!" << endl;
        return 0;

#if 0
        cout << "OpenNN. HIGGS 5x300 DNN benchmark." << endl;

#ifdef OPENNN_HAS_CUDA
        Configuration::instance().set(Device::CUDA, Type::FP32);
        Backend::instance();
#endif

        const filesystem::path dataset_path = "/home/artelnics/Documents/HIGGS.csv";
        const filesystem::path parameters_path = "/home/artelnics/Documents/higgs_5x300_dnn_parameters.bin";
        const filesystem::path threshold_path = "/home/artelnics/Documents/higgs_5x300_dnn_threshold.txt";

        TabularDataset dataset(dataset_path, ",", false, false);

        vector<Index> input_variables(28);
        iota(input_variables.begin(), input_variables.end(), 1);
        dataset.set_variable_indices(input_variables, {0});
        dataset.set_variable_type(0, VariableType::Binary);
        dataset.split_samples_sequential(10.0f / 11.0f, 0.5f / 11.0f, 0.5f / 11.0f);

        cout << "[DATASET] train=" << dataset.get_sample_indices("Training").size()
             << " val="            << dataset.get_sample_indices("Validation").size()
             << " test="           << dataset.get_sample_indices("Testing").size()
             << " input="          << dataset.get_shape("Input")[0]
             << " target="         << dataset.get_shape("Target")[0]
             << endl;

        NeuralNetwork network;
        network.add_layer(make_unique<Scaling>(dataset.get_shape("Input")));

        for (Index i = 0; i < 5; ++i)
            network.add_layer(make_unique<opennn::Dense>(network.get_output_shape(),
                                                         Shape{300},
                                                         "Tanh",
                                                         false,
                                                         format("higgs_hidden_{}", i + 1)));

        network.add_layer(make_unique<opennn::Dense>(network.get_output_shape(),
                                                     dataset.get_shape("Target"),
                                                     "Sigmoid",
                                                     false,
                                                     "higgs_output"));
        network.compile();
        network.set_parameters_glorot();

        cout << "HIGGS DNN params=" << network.get_parameters_number()
             << " (buffer=" << network.get_parameters_size() << ")" << endl;

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.set_loss("CrossEntropy");
        training_strategy.set_optimization_algorithm("StochasticGradientDescent");
        training_strategy.get_loss()->set_regularization("L2");
        training_strategy.get_loss()->set_regularization_weight(1.0e-5f);

        auto* sgd = dynamic_cast<StochasticGradientDescent*>(training_strategy.get_optimization_algorithm());
        if (!sgd)
            throw runtime_error("StochasticGradientDescent optimizer not found.");

        sgd->set_batch_size(100);
        sgd->set_initial_learning_rate(0.05f);
        sgd->set_initial_decay(0.0202f);
        sgd->set_momentum(0.9f);
        sgd->set_nesterov(false);
        sgd->set_num_workers(8);
        sgd->set_maximum_epochs(100);
        sgd->set_maximum_validation_failures(200);
        sgd->set_display_period(1);

        const auto t0 = steady_clock::now();
        training_strategy.train();
        const auto t1 = steady_clock::now();

        cout << "\nTotal training time: "
             << duration_cast<milliseconds>(t1 - t0).count() / 1000.0 << " s" << endl;

        network.save_parameters_binary(parameters_path);
        cout << "Saved parameters to " << parameters_path
             << " (" << filesystem::file_size(parameters_path) / (1024 * 1024)
             << " MiB)" << endl;

        TestingAnalysis testing_analysis(&network, &dataset);
        testing_analysis.set_batch_size(10000);
        const TestingAnalysis::RocAnalysis roc_analysis = testing_analysis.perform_roc_analysis();

        cout << "\nROC analysis:" << endl;
        cout << "Role: Testing" << endl;
        cout << "AUC: " << roc_analysis.area_under_curve << endl;
        cout << "Confidence limit: " << roc_analysis.confidence_limit << endl;
        cout << "Optimal threshold: " << roc_analysis.optimal_threshold << endl;

        ofstream threshold_file(threshold_path);
        if (!threshold_file.is_open())
            throw runtime_error("Cannot open threshold file for writing: " + threshold_path.string());
        threshold_file << setprecision(9) << roc_analysis.optimal_threshold << '\n';
        cout << "Saved optimal threshold to " << threshold_path << endl;

        cout << "\nConfusion matrix at threshold 0.5:\n"
             << testing_analysis.calculate_confusion(0.5f) << endl;
        cout << "\nConfusion matrix at optimal threshold:\n"
             << testing_analysis.calculate_confusion(roc_analysis.optimal_threshold) << endl;

        cout << "Bye!" << endl;
        return 0;

        /*
        cout << "OpenNN. train_1000_filter ResNet-18 ImageNet transfer learning." << endl;

#ifdef OPENNN_HAS_CUDA
        Configuration::instance().set(Device::CUDA, Type::BF16);
        Backend::instance();
#endif

        const filesystem::path dataset_path = "/home/artelnics/Documents/train_1000_filter";
        const filesystem::path pretrained_dir = "/home/artelnics/Documents/opennn_pretrained";
        const filesystem::path pretrained_parameters_path =
            pretrained_dir / "resnet18_imagenet1k_to_1_opennn_parameters.bin";
        const filesystem::path pretrained_states_path =
            pretrained_dir / "resnet18_imagenet1k_to_1_opennn_states.bin";
        const filesystem::path trained_parameters_path =
            "/home/artelnics/Documents/train_1000_filter_resnet18_transfer_lr3e6_best_parameters.bin";
        const filesystem::path trained_states_path =
            "/home/artelnics/Documents/train_1000_filter_resnet18_transfer_lr3e6_best_states.bin";
        const filesystem::path trained_threshold_path =
            "/home/artelnics/Documents/train_1000_filter_resnet18_transfer_lr3e6_best_threshold.txt";

        if (!filesystem::exists(dataset_path))
            throw runtime_error("Dataset folder not found: " + dataset_path.string());
        if (!filesystem::exists(pretrained_parameters_path))
            throw runtime_error("Pretrained parameters not found: " + pretrained_parameters_path.string());
        if (!filesystem::exists(pretrained_states_path))
            throw runtime_error("Pretrained states not found: " + pretrained_states_path.string());

        ImageDataset dataset(dataset_path);
        dataset.split_samples_random(0.80, 0.10, 0.10);

        AugmentationSettings augmentation;
        augmentation.enabled = true;
        augmentation.reflection_axis_x = true;
        augmentation.rotation_minimum = -5.0f;
        augmentation.rotation_maximum = 5.0f;
        augmentation.horizontal_translation_minimum = -4.0f;
        augmentation.horizontal_translation_maximum = 4.0f;
        augmentation.vertical_translation_minimum = -4.0f;
        augmentation.vertical_translation_maximum = 4.0f;
        dataset.set_augmentation(augmentation);

        const Shape input_shape = dataset.get_shape("Input");
        const Shape target_shape = dataset.get_shape("Target");

        cout << "[DATASET] train=" << dataset.get_samples_number("Training")
             << " val="            << dataset.get_samples_number("Validation")
             << " test="           << dataset.get_samples_number("Testing")
             << " input="          << input_shape[0] << "x" << input_shape[1] << "x" << input_shape[2]
             << " target="         << target_shape[0] << endl;

        ResNet network(input_shape,
                       {2, 2, 2, 2},
                       Shape{64, 128, 256, 512},
                       target_shape,
                       false);

        auto* scaling = dynamic_cast<Scaling*>(network.get_first(LayerType::Scaling));
        if (!scaling)
            throw runtime_error("ResNet scaling layer not found.");

        scaling->set_descriptives({
            Descriptives(0.0f, 255.0f, 0.485f * 255.0f, 0.229f * 255.0f),
            Descriptives(0.0f, 255.0f, 0.456f * 255.0f, 0.224f * 255.0f),
            Descriptives(0.0f, 255.0f, 0.406f * 255.0f, 0.225f * 255.0f)
        });
        scaling->set_scalers("MeanStandardDeviation");

        cout << "Transfer ResNet-18 params=" << network.get_parameters_number()
             << " (buffer=" << network.get_parameters_size() << ")" << endl;
        cout << "Loading pretrained parameters from " << pretrained_parameters_path << endl;
        network.load_parameters_binary(pretrained_parameters_path);
        cout << "Loading pretrained states from " << pretrained_states_path << endl;
        network.load_states_binary(pretrained_states_path);

        const bool evaluate_saved_parameters_only =
            getenv("OPENNN_EVALUATE_SAVED_PARAMETERS") != nullptr;

        if (evaluate_saved_parameters_only)
        {
            if (!filesystem::exists(trained_parameters_path))
                throw runtime_error("Saved transfer parameters not found: " + trained_parameters_path.string());
            if (!filesystem::exists(trained_states_path))
                throw runtime_error("Saved transfer states not found: " + trained_states_path.string());

            cout << "Loading trained transfer parameters from " << trained_parameters_path << endl;
            network.load_parameters_binary(trained_parameters_path);
            cout << "Loading trained transfer states from " << trained_states_path << endl;
            network.load_states_binary(trained_states_path);
        }
        else
        {
            TrainingStrategy training_strategy(&network, &dataset);
            training_strategy.set_loss("CrossEntropy");
            training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
            training_strategy.get_loss()->set_regularization("None");

            auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
            if (!adam) throw runtime_error("AdaptiveMomentEstimation optimizer not found.");

            adam->set_batch_size(16);
            adam->set_learning_rate(3.0e-6f);
            adam->set_num_workers(8);
            adam->set_maximum_epochs(48);
            adam->set_maximum_validation_failures(8);
            adam->set_display_period(1);

            const auto t0 = steady_clock::now();
            training_strategy.train();
            const auto t1 = steady_clock::now();

            const double training_seconds = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;
            cout << "\nTotal training time: " << training_seconds << " s" << endl;

            network.save_parameters_binary(trained_parameters_path);
            cout << "Saved parameters to " << trained_parameters_path
                 << " (" << filesystem::file_size(trained_parameters_path) / (1024 * 1024)
                 << " MiB)" << endl;

            network.save_states_binary(trained_states_path);
            cout << "Saved states to " << trained_states_path
                 << " (" << filesystem::file_size(trained_states_path) / 1024
                 << " KiB)" << endl;
        }

        TestingAnalysis testing_analysis(&network, &dataset);
        testing_analysis.set_batch_size(32);

        auto get_targets_and_outputs_batched = [&](const string& role) -> pair<MatrixR, MatrixR>
        {
            const vector<Index> sample_indices = dataset.get_sample_indices(role);
            const vector<Index> input_indices = dataset.get_feature_indices("Input");
            const vector<Index> target_indices = dataset.get_feature_indices("Target");

            MatrixR targets(ssize(sample_indices), ssize(target_indices));
            MatrixR outputs(ssize(sample_indices), target_shape.size());

            const Index batch_size = 32;
            for (Index begin = 0; begin < ssize(sample_indices); begin += batch_size)
            {
                const Index current_batch_size = min(batch_size, ssize(sample_indices) - begin);
                vector<Index> batch_indices(sample_indices.begin() + begin,
                                            sample_indices.begin() + begin + current_batch_size);

                MatrixR batch_targets(current_batch_size, ssize(target_indices));
                dataset.fill_targets(batch_indices, target_indices,
                                     batch_targets.data(), false, true);

                Tensor4 batch_inputs(current_batch_size, input_shape[0], input_shape[1], input_shape[2]);
                dataset.fill_inputs(batch_indices, input_indices,
                                    batch_inputs.data(), false, true);

                const MatrixR batch_outputs = network.calculate_outputs(batch_inputs);
                targets.middleRows(begin, current_batch_size) = batch_targets;
                outputs.middleRows(begin, current_batch_size) = batch_outputs;
            }

            return {targets, outputs};
        };

        const string evaluation_role = "Testing";
        const auto [testing_targets, testing_outputs] = get_targets_and_outputs_batched(evaluation_role);
        const MatrixR roc_curve = testing_analysis.calculate_roc_curve(testing_targets, testing_outputs);

        TestingAnalysis::RocAnalysis roc_analysis;
        roc_analysis.roc_curve = roc_curve;
        roc_analysis.area_under_curve = testing_analysis.calculate_area_under_curve(roc_curve);
        roc_analysis.confidence_limit =
            testing_analysis.calculate_area_under_curve_confidence_limit(testing_targets, testing_outputs);
        roc_analysis.optimal_threshold = testing_analysis.calculate_optimal_threshold(roc_curve);

        MatrixR inverted_outputs = MatrixR::Ones(testing_outputs.rows(), testing_outputs.cols()) - testing_outputs;
        const MatrixR inverted_roc_curve = testing_analysis.calculate_roc_curve(testing_targets, inverted_outputs);
        const float inverted_auc = testing_analysis.calculate_area_under_curve(inverted_roc_curve);

        cout << "\nROC analysis:" << endl;
        cout << "Role: " << evaluation_role << endl;
        cout << "AUC: " << roc_analysis.area_under_curve << endl;
        cout << "AUC with inverted score: " << inverted_auc << endl;
        cout << "Confidence limit: " << roc_analysis.confidence_limit << endl;
        cout << "Optimal threshold: " << roc_analysis.optimal_threshold << endl;

        ofstream threshold_file(trained_threshold_path);
        if (!threshold_file.is_open())
            throw runtime_error("Cannot open threshold file for writing: " + trained_threshold_path.string());
        threshold_file << setprecision(9) << roc_analysis.optimal_threshold << '\n';
        cout << "Saved optimal threshold to " << trained_threshold_path << endl;

        cout << "\nConfusion matrix at threshold 0.5:\n"
             << testing_analysis.calculate_confusion(testing_targets,
                                                    testing_outputs,
                                                    0.5f) << endl;
        cout << "\nConfusion matrix at optimal threshold:\n"
             << testing_analysis.calculate_confusion(testing_targets,
                                                    testing_outputs,
                                                    roc_analysis.optimal_threshold) << endl;

        cout << "Bye!" << endl;
        return 0;
        }

        {
        cout << "OpenNN. ResNet-18 ImageNet pretrained Imagenette evaluation." << endl;

#ifdef OPENNN_HAS_CUDA
        Configuration::instance().set(Device::CUDA, Type::FP32);
        Backend::instance();
#endif

        const filesystem::path imagenette_path = "/home/artelnics/Documents/imagenette2_bmp_224";
        const filesystem::path pretrained_dir = "/home/artelnics/Documents/opennn_pretrained";
        const filesystem::path parameters_path = pretrained_dir / "resnet18_imagenet1k_opennn_parameters.bin";
        const filesystem::path states_path = pretrained_dir / "resnet18_imagenet1k_opennn_states.bin";

        if (!filesystem::exists(imagenette_path))
            throw runtime_error("Imagenette folder not found: " + imagenette_path.string());
        if (!filesystem::exists(parameters_path))
            throw runtime_error("Pretrained parameters not found: " + parameters_path.string());
        if (!filesystem::exists(states_path))
            throw runtime_error("Pretrained states not found: " + states_path.string());

        ResNet network({224, 224, 3}, {2, 2, 2, 2}, Shape{64, 128, 256, 512}, Shape{1000}, false);

        auto* scaling = dynamic_cast<Scaling*>(network.get_first(LayerType::Scaling));
        if (!scaling)
            throw runtime_error("ResNet scaling layer not found.");

        scaling->set_descriptives({
            Descriptives(0.0f, 255.0f, 0.485f * 255.0f, 0.229f * 255.0f),
            Descriptives(0.0f, 255.0f, 0.456f * 255.0f, 0.224f * 255.0f),
            Descriptives(0.0f, 255.0f, 0.406f * 255.0f, 0.225f * 255.0f)
        });
        scaling->set_scalers("MeanStandardDeviation");

        cout << "ResNet-18 params=" << network.get_parameters_number()
             << " (buffer=" << network.get_parameters_size() << ")" << endl;
        cout << "Loading " << parameters_path << endl;
        network.load_parameters_binary(parameters_path);
        cout << "Loading " << states_path << endl;
        network.load_states_binary(states_path);

        const unordered_map<string, Index> imagenette_to_imagenet = {
            {"tench", 0},
            {"english_springer", 217},
            {"cassette_player", 482},
            {"chain_saw", 491},
            {"church", 497},
            {"french_horn", 566},
            {"garbage_truck", 569},
            {"gas_pump", 571},
            {"golf_ball", 574},
            {"parachute", 701}
        };

        struct Sample { filesystem::path path; Index target; string label; };
        vector<Sample> samples;

        for (const auto& entry : filesystem::directory_iterator(imagenette_path))
        {
            if (!entry.is_directory() || entry.path().filename().string().starts_with("."))
                continue;

            const string label = entry.path().filename().string();
            const auto target_it = imagenette_to_imagenet.find(label);
            if (target_it == imagenette_to_imagenet.end())
                continue;

            for (const auto& image_entry : filesystem::recursive_directory_iterator(entry.path()))
                if (image_entry.is_regular_file() && is_supported_image_file(image_entry.path()))
                    samples.push_back({image_entry.path(), target_it->second, label});
        }

        ranges::sort(samples, {}, &Sample::path);

        cout << "Imagenette samples=" << samples.size() << endl;
        if (samples.empty())
            throw runtime_error("No Imagenette samples found.");

        const Index batch_size = 64;
        Index top1 = 0;
        Index top5 = 0;
        vector<Index> class_total(1000, 0);
        vector<Index> class_correct(1000, 0);

        vector<Index> top_indices(1000);
        iota(top_indices.begin(), top_indices.end(), 0);

        const auto t0 = steady_clock::now();

        for (Index begin = 0; begin < ssize(samples); begin += batch_size)
        {
            const Index current_batch_size = min(batch_size, ssize(samples) - begin);
            Tensor4 batch_inputs(current_batch_size, 224, 224, 3);

            for (Index i = 0; i < current_batch_size; ++i)
                load_image(samples[size_t(begin + i)].path,
                           batch_inputs.data() + i * 224 * 224 * 3,
                           224, 224, 3, false);

            const MatrixR outputs = network.calculate_outputs(batch_inputs);

            for (Index i = 0; i < current_batch_size; ++i)
            {
                const Index target = samples[size_t(begin + i)].target;
                ++class_total[size_t(target)];

                Index prediction = 0;
                outputs.row(i).maxCoeff(&prediction);
                if (prediction == target)
                {
                    ++top1;
                    ++class_correct[size_t(target)];
                }

                iota(top_indices.begin(), top_indices.end(), 0);
                nth_element(top_indices.begin(),
                            top_indices.begin() + 5,
                            top_indices.end(),
                            [&](Index a, Index b) { return outputs(i, a) > outputs(i, b); });
                if (find(top_indices.begin(), top_indices.begin() + 5, target) != top_indices.begin() + 5)
                    ++top5;
            }
        }

        const auto t1 = steady_clock::now();
        const double seconds = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;

        cout << fixed << setprecision(4);
        cout << "Top-1 accuracy: " << 100.0 * double(top1) / double(samples.size()) << " %" << endl;
        cout << "Top-5 accuracy: " << 100.0 * double(top5) / double(samples.size()) << " %" << endl;
        cout << "Elapsed: " << seconds << " s" << endl;

        cout << "\nPer-class top-1:" << endl;
        for (const auto& [label, target] : imagenette_to_imagenet)
        {
            const Index total = class_total[size_t(target)];
            if (total == 0) continue;
            cout << setw(18) << label << ": "
                 << 100.0 * double(class_correct[size_t(target)]) / double(total)
                 << " % (" << class_correct[size_t(target)] << "/" << total << ")" << endl;
        }

        cout << "Bye!" << endl;
        return 0;
        }

        
        cout << "OpenNN. train_1000_filter ResNet-18 clean test." << endl;

#ifdef OPENNN_HAS_CUDA
        // ============== train_1000_filter — binary PNG image training ==============
        //
        // Dataset layout:
        //   /home/artelnics/Documents/train_1000_filter/
        //       NEGATIVE/*.png
        //       POSITIVE/*.png
        //
        // ImageDataset now builds .cache/images.bin directly from PNG/BMP files.

        Configuration::instance().set(Device::CUDA, Type::BF16);
        Backend::instance();
        set_seed(42);

        const filesystem::path dataset_path = "/home/artelnics/Documents/train_1000_filter";
        if (!filesystem::exists(dataset_path))
            throw runtime_error("Dataset folder not found: " + dataset_path.string());

        ImageDataset dataset(dataset_path);
        dataset.split_samples_random(0.80, 0.10, 0.10);

        AugmentationSettings augmentation;
        augmentation.enabled = false;
        dataset.set_augmentation(augmentation);

        const Shape input_shape = dataset.get_shape("Input");
        const Shape target_shape = dataset.get_shape("Target");

        cout << "[DATASET] train=" << dataset.get_samples_number("Training")
             << " val="            << dataset.get_samples_number("Validation")
             << " test="           << dataset.get_samples_number("Testing")
             << " input="          << input_shape[0] << "x" << input_shape[1] << "x" << input_shape[2]
             << " target="         << target_shape[0] << endl;

        NeuralNetwork network;

        auto scaling = make_unique<Scaling>(input_shape);
        scaling->set_scalers("ImageMinMax");
        network.add_layer(move(scaling));

        auto add_conv = [&](Index input_index,
                            const Shape& kernel_shape,
                            const char* activation,
                            const Shape& stride,
                            const string& name) -> Index
        {
            network.add_layer(make_unique<Convolutional>(
                                  network.get_layer(input_index)->get_output_shape(),
                                  kernel_shape,
                                  activation,
                                  stride,
                                  "Same",
                                  true,
                                  name),
                              {input_index});
            return network.get_layers_number() - 1;
        };

        auto add_skip = [&](Index input_index,
                            Index in_channels,
                            Index out_channels,
                            Index stride,
                            const string& prefix) -> Index
        {
            if (stride == 1 && in_channels == out_channels)
                return input_index;

            return add_conv(input_index,
                            Shape{1, 1, in_channels, out_channels},
                            "Identity",
                            Shape{stride, stride},
                            prefix + "_skip");
        };

        auto add_basic_block = [&](Index input_index,
                                   size_t stage,
                                   Index block,
                                   Index filters) -> Index
        {
            const Shape block_input_shape = network.get_layer(input_index)->get_output_shape();
            const Index in_channels = block_input_shape[2];
            const Index stride = (stage > 0 && block == 0) ? 2 : 1;
            const string prefix = format("s{}b{}", stage, block);

            Index main_index = add_conv(input_index,
                                        Shape{3, 3, in_channels, filters},
                                        "ReLU",
                                        Shape{stride, stride},
                                        prefix + "_conv1");

            main_index = add_conv(main_index,
                                  Shape{3, 3, filters, filters},
                                  "Identity",
                                  Shape{1, 1},
                                  prefix + "_conv2");

            const Index skip_index = add_skip(input_index,
                                              in_channels,
                                              filters,
                                              stride,
                                              prefix);

            network.add_layer(make_unique<Addition>(
                                  network.get_layer(main_index)->get_output_shape(),
                                  prefix + "_add"),
                              {main_index, skip_index});
            const Index add_index = network.get_layers_number() - 1;

            network.add_layer(make_unique<Activation>(
                                  network.get_layer(add_index)->get_output_shape(),
                                  "ReLU",
                                  prefix + "_relu"),
                              {add_index});

            return network.get_layers_number() - 1;
        };

        Index last_index = add_conv(0,
                                    Shape{7, 7, input_shape[2], 64},
                                    "ReLU",
                                    Shape{2, 2},
                                    "stem_conv");

        network.add_layer(make_unique<Pooling>(network.get_layer(last_index)->get_output_shape(),
                                               Shape{3, 3},
                                               Shape{2, 2},
                                               Shape{1, 1},
                                               "MaxPooling",
                                               "stem_pool"),
                          {last_index});
        last_index = network.get_layers_number() - 1;

        const vector<Index> blocks_per_stage = {2, 2, 2, 2};
        const Shape filters_per_stage = {64, 128, 256, 512};

        for (size_t stage = 0; stage < blocks_per_stage.size(); ++stage)
            for (Index block = 0; block < blocks_per_stage[stage]; ++block)
                last_index = add_basic_block(last_index,
                                             stage,
                                             block,
                                             filters_per_stage[Index(stage)]);

        const Shape pre_pool_shape = network.get_layer(last_index)->get_output_shape();
        network.add_layer(make_unique<Pooling>(pre_pool_shape,
                                               Shape{pre_pool_shape[0], pre_pool_shape[1]},
                                               Shape{1, 1},
                                               Shape{0, 0},
                                               "AveragePooling",
                                               "global_avg_pool"),
                          {last_index});
        last_index = network.get_layers_number() - 1;

        network.add_layer(make_unique<Flatten>(network.get_layer(last_index)->get_output_shape()),
                          {last_index});
        last_index = network.get_layers_number() - 1;

        auto classifier_hidden = make_unique<opennn::Dense>(
            network.get_layer(last_index)->get_output_shape(),
            Shape{256},
            "ReLU",
            true,
            "classifier_hidden");
        classifier_hidden->set_dropout_rate(0.0f);
        network.add_layer(move(classifier_hidden), {last_index});
        last_index = network.get_layers_number() - 1;

        network.add_layer(make_unique<opennn::Dense>(network.get_layer(last_index)->get_output_shape(),
                                                     target_shape,
                                                     "Sigmoid",
                                                     false,
                                                     "classifier"),
                                                     {last_index});

        network.compile();
        network.set_parameters_random();

        cout << "Binary ResNet-18 clean params=" << network.get_parameters_number()
             << " (buffer=" << network.get_parameters_size() << ")" << endl;

        const filesystem::path parameters_path =
            "/home/artelnics/Documents/train_1000_filter_resnet18_clean_parameters.bin";
        const filesystem::path states_path =
            "/home/artelnics/Documents/train_1000_filter_resnet18_clean_states.bin";

        const bool evaluate_saved_parameters_only =
            std::getenv("OPENNN_EVALUATE_SAVED_PARAMETERS") != nullptr;

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.set_loss("CrossEntropy");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        training_strategy.get_loss()->set_regularization("None");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        if (!adam) throw runtime_error("AdaptiveMomentEstimation optimizer not found.");

        adam->set_batch_size(16);
        adam->set_learning_rate(3.0e-4f);
        adam->set_num_workers(8);
        adam->set_maximum_epochs(39);
        adam->set_maximum_validation_failures(8);
        adam->set_display_period(1);

        if (evaluate_saved_parameters_only)
        {
            if (!filesystem::exists(parameters_path))
                throw runtime_error("Saved parameters not found: " + parameters_path.string());

            cout << "Loading saved parameters from " << parameters_path
                 << " (skipping training)" << endl;
            network.load_parameters_binary(parameters_path);

            if (!filesystem::exists(states_path))
                throw runtime_error("Saved states not found: " + states_path.string());

            cout << "Loading saved states from " << states_path << endl;
            network.load_states_binary(states_path);
        }
        else
        {
            const auto t0 = steady_clock::now();
            training_strategy.train();
            const auto t1 = steady_clock::now();

            const double training_seconds = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;
            cout << "\nTotal training time: " << training_seconds << " s" << endl;

            network.save_parameters_binary(parameters_path);
            cout << "Saved parameters to " << parameters_path
                 << " (" << filesystem::file_size(parameters_path) / (1024 * 1024)
                 << " MiB)" << endl;

            network.save_states_binary(states_path);
            cout << "Saved states to " << states_path
                 << " (" << filesystem::file_size(states_path) / 1024
                 << " KiB)" << endl;
        }

        TestingAnalysis testing_analysis(&network, &dataset);
        testing_analysis.set_batch_size(32);

        auto get_targets_and_outputs_batched = [&](const string& role) -> pair<MatrixR, MatrixR>
        {
            const vector<Index> sample_indices = dataset.get_sample_indices(role);
            const vector<Index> input_indices = dataset.get_feature_indices("Input");
            const vector<Index> target_indices = dataset.get_feature_indices("Target");

            MatrixR targets(ssize(sample_indices), ssize(target_indices));
            MatrixR outputs(ssize(sample_indices), target_shape.size());

            const Index batch_size = 32;
            for (Index begin = 0; begin < ssize(sample_indices); begin += batch_size)
            {
                const Index current_batch_size = min(batch_size, ssize(sample_indices) - begin);
                vector<Index> batch_indices(sample_indices.begin() + begin,
                                            sample_indices.begin() + begin + current_batch_size);

                MatrixR batch_targets(current_batch_size, ssize(target_indices));
                dataset.fill_targets(batch_indices, target_indices,
                                     batch_targets.data(), false, true);

                Tensor4 batch_inputs(current_batch_size, input_shape[0], input_shape[1], input_shape[2]);
                dataset.fill_inputs(batch_indices, input_indices,
                                    batch_inputs.data(), false, true);

                const MatrixR batch_outputs = network.calculate_outputs(batch_inputs);
                targets.middleRows(begin, current_batch_size) = batch_targets;
                outputs.middleRows(begin, current_batch_size) = batch_outputs;
            }

            return {targets, outputs};
        };

        const string evaluation_role = "Testing";
        const auto [testing_targets, testing_outputs] = get_targets_and_outputs_batched(evaluation_role);
        const MatrixR roc_curve = testing_analysis.calculate_roc_curve(testing_targets, testing_outputs);

        TestingAnalysis::RocAnalysis roc_analysis;
        roc_analysis.roc_curve = roc_curve;
        roc_analysis.area_under_curve = testing_analysis.calculate_area_under_curve(roc_curve);
        roc_analysis.confidence_limit =
            testing_analysis.calculate_area_under_curve_confidence_limit(testing_targets, testing_outputs);
        roc_analysis.optimal_threshold = testing_analysis.calculate_optimal_threshold(roc_curve);

        cout << "\nROC analysis:" << endl;
        cout << "Role: " << evaluation_role << endl;
        cout << "AUC: " << roc_analysis.area_under_curve << endl;
        cout << "Confidence limit: " << roc_analysis.confidence_limit << endl;
        cout << "Optimal threshold: " << roc_analysis.optimal_threshold << endl;
        cout << "\nConfusion matrix at threshold 0.5:\n"
             << testing_analysis.calculate_confusion(testing_targets,
                                                    testing_outputs,
                                                    0.5f) << endl;
        cout << "\nConfusion matrix at optimal threshold:\n"
             << testing_analysis.calculate_confusion(testing_targets,
                                                    testing_outputs,
                                                    roc_analysis.optimal_threshold) << endl;

        Index positives = 0;
        Index negatives = 0;
        double positive_sum = 0.0;
        double negative_sum = 0.0;
        float positive_min = numeric_limits<float>::max();
        float positive_max = numeric_limits<float>::lowest();
        float negative_min = numeric_limits<float>::max();
        float negative_max = numeric_limits<float>::lowest();

        for (Index i = 0; i < testing_targets.rows(); ++i)
        {
            const float output = testing_outputs(i, 0);
            if (testing_targets(i, 0) >= 0.5f)
            {
                ++positives;
                positive_sum += output;
                positive_min = min(positive_min, output);
                positive_max = max(positive_max, output);
            }
            else
            {
                ++negatives;
                negative_sum += output;
                negative_min = min(negative_min, output);
                negative_max = max(negative_max, output);
            }
        }

        cout << "\nOutput stats:" << endl;
        cout << "Positive targets: mean=" << positive_sum / double(positives)
             << " min=" << positive_min
             << " max=" << positive_max << endl;
        cout << "Negative targets: mean=" << negative_sum / double(negatives)
             << " min=" << negative_min
             << " max=" << negative_max << endl;

#if 0
/*
        // ============== ResNet on Places365 ==============
        //
        // Folder layout expected by ImageDataset: one subdirectory per class,
        // each holding .bmp files. ImageDataset builds .cache/images.bin on
        // first construction and reuses it on subsequent runs.
        //
        // Places365 converted for OpenNN:
        // one root folder with one BMP subfolder per class.

        Configuration::instance().set(Device::CUDA, Type::BF16);

        const std::filesystem::path PLACES365_PATH =
            R"(\\Artelnics\data_sets\places365_bmp_224)";

        if (!std::filesystem::exists(PLACES365_PATH))
            throw runtime_error("Places365 folder not found at " + PLACES365_PATH.string()
                                + ". Place the dataset there (one subfolder per class with .bmp images) "
                                  "or edit PLACES365_PATH in blank_cuda/main.cpp.");

        ImageDataset dataset(PLACES365_PATH);
        dataset.split_samples_random(0.85, 0.1, 0.05);

        const Shape input_shape  = dataset.get_shape("Input");
        const Index num_classes  = dataset.get_shape("Target")[0];

        cout << "[PARITY] train="  << dataset.get_samples_number("Training")
             << " val="            << dataset.get_samples_number("Validation")
             << " input="          << input_shape[0] << "x" << input_shape[1] << "x" << input_shape[2]
             << " classes="        << num_classes << endl;

        // ResNet-50: bottleneck blocks, stage depths [3, 4, 6, 3].
        // Inner channels {64, 128, 256, 512}; with expansion=4 the outer
        // channels become {256, 512, 1024, 2048}.
        ResNet resnet(input_shape,
                      {3, 4, 6, 3},
                      Shape{64, 128, 256, 512},
                      Shape{num_classes},
                      true);

        TrainingStrategy training_strategy(&resnet, &dataset);
        training_strategy.set_loss("CrossEntropy");
        training_strategy.set_optimization_algorithm("StochasticGradientDescent");

        auto* sgd = dynamic_cast<StochasticGradientDescent*>(training_strategy.get_optimization_algorithm());
        if (!sgd) throw runtime_error("StochasticGradientDescent optimizer not found.");

        sgd->set_batch_size(16);
        sgd->set_initial_learning_rate(1.0e-2f);
        sgd->set_initial_decay(0.0f);
        sgd->set_momentum(0.9f);
        sgd->set_nesterov(true);
        sgd->set_maximum_epochs(10);
        sgd->set_display_period(1);

        cout << "ResNet-50 params=" << resnet.get_parameters_number()
             << " (buffer=" << resnet.get_parameters_size() << ")" << endl;

        const std::filesystem::path parameters_path =
            R"(\\Artelnics\data_sets\places365_resnet50_parameters.bin)";

        const auto t0 = steady_clock::now();
        training_strategy.train();
        const auto t1 = steady_clock::now();

        const double training_seconds = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;
        cout << "\nTotal training time: " << training_seconds << " s" << endl;

        resnet.save_parameters_binary(parameters_path);
        cout << "Saved parameters to " << parameters_path
             << " (" << std::filesystem::file_size(parameters_path) / (1024 * 1024)
             << " MiB)" << endl;

        // Confusion on the testing split.
        TestingAnalysis testing_analysis(&resnet, &dataset);
        testing_analysis.set_batch_size(16);
        cout << "\nConfusion matrix:\n" << testing_analysis.calculate_confusion() << endl;
        */
        
        // ====================  LLM experiments (commented out)  ====================
        // The blocks below were used to fine-tune chat-style Transformers on
        // tinychat / OASST2 / UltraChat combinations. Restored above is the
        // canonical translation benchmark. To re-run the LLM experiments,
        // uncomment the desired block.

        // ====================  TINYCHAT (instruction-tuning)  ====================
        /*
        Configuration::instance().set(Device::CUDA, Type::BF16);

        const std::filesystem::path tinychat_src = "/home/artelnics/Documents/openNN/opennn/temp/tinychat.txt";
        // Half-sized cache (~2M pairs) — keeps the previous full cache around if it exists.
        const std::filesystem::path tinychat_pairs = "/tmp/tinychat_pairs_2M.tsv";
        const Index max_pairs = 2'000'000;

        prepare_tinychat_pairs(tinychat_src, tinychat_pairs, max_pairs);

        LanguageDataset dataset(tinychat_pairs);
        dataset.split_samples_random(0.9, 0.05, 0.05);

        const Index input_vocab  = dataset.get_input_vocabulary_size();
        const Index output_vocab = dataset.get_target_vocabulary_size();
        const Index input_seq    = dataset.get_shape("Input")[0];
        const Index decoder_seq  = dataset.get_shape("Decoder")[0];
        const Index target_seq   = dataset.get_shape("Target")[0];

        if(decoder_seq != target_seq)
            throw runtime_error("Decoder and target sequence lengths must match.");

        const Index embedding_dimension     = 512;
        const Index heads_number            = 8;
        const Index feed_forward_dimension  = 2048;
        const Index layers_number           = 4;

        Transformer transformer(input_seq,
                                decoder_seq,
                                input_vocab,
                                output_vocab,
                                embedding_dimension,
                                heads_number,
                                feed_forward_dimension,
                                layers_number);

        transformer.set_dropout_rate(float(0.1));

        cout << "[TINYCHAT] train="    << dataset.get_samples_number("Training")
             << " val="                << dataset.get_samples_number("Validation")
             << " test="               << dataset.get_samples_number("Testing")
             << " input_vocab="        << input_vocab
             << " target_vocab="       << output_vocab
             << " input_seq="          << input_seq
             << " decoder_seq="        << decoder_seq
             << " params="             << transformer.get_parameters_number() << endl;

        const std::filesystem::path weights_file = "/tmp/tinychat_weights.bin";

        if (std::filesystem::exists(weights_file))
        {
            cout << "Loading cached weights from " << weights_file
                 << " (skipping training)" << endl;
            transformer.load_parameters_binary(weights_file);
        }
        else
        {
            TrainingStrategy training_strategy(&transformer, &dataset);

            training_strategy.set_loss("CrossEntropyError3d");
            training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

            auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
            if(!adam) throw runtime_error("AdaptiveMomentEstimation optimizer not found.");

            adam->set_batch_size(128);
            adam->set_learning_rate(float(5e-4));
            adam->set_maximum_epochs(0);
            adam->set_display_period(1);

            const auto t0 = steady_clock::now();
            training_strategy.train();
            const auto t1 = steady_clock::now();

            cout << "\nTotal training time (tinychat): "
                 << duration_cast<milliseconds>(t1 - t0).count() / 1000.0 << " s" << endl;

            std::ofstream weights_out(weights_file, std::ios::binary);
            if (!weights_out)
                throw runtime_error("Cannot open " + weights_file.string() + " for writing");
            const Index parameters_floats = transformer.get_parameters_size();
            weights_out.write(reinterpret_cast<const char*>(transformer.get_parameters_data()),
                              parameters_floats * sizeof(float));
            cout << "Saved " << parameters_floats << " floats ("
                 << parameters_floats * sizeof(float) << " bytes) to "
                 << weights_file << endl;
        }

        cout << "\n================ TINYCHAT CHAT ================\n";

        TransformerDecoder::SamplingConfig sampling_config;
        sampling_config.temperature        = 0.8f;
        sampling_config.top_p              = 0.9f;
        sampling_config.top_k              = 50;
        sampling_config.repetition_penalty = 1.15f;

        TransformerDecoder decoder(transformer, dataset);
        decoder.chat(sampling_config);

        cout << "===============================================\n";
        */

        // ====================  OASST2 (instruction-tuning)  ====================
        // Pre-extracted .tsv produced by temp/OASST/prepare_oasst2_pairs.py from
        // the OpenAssistant ready_for_export trees (English subset, length-capped).
        /*
        Configuration::instance().set(Device::CUDA, Type::BF16);

        set_seed(42);

        const std::filesystem::path oasst2_pairs = "/home/artelnics/Documents/openNN/opennn/temp/OASST/oasst2_pairs.tsv";

        LanguageDataset dataset(oasst2_pairs, std::numeric_limits<Index>::max(), 2);
        dataset.split_samples_random(0.9, 0.05, 0.05);

        const Index input_vocab  = dataset.get_input_vocabulary_size();
        const Index output_vocab = dataset.get_target_vocabulary_size();
        const Index input_seq    = dataset.get_shape("Input")[0];
        const Index decoder_seq  = dataset.get_shape("Decoder")[0];
        const Index target_seq   = dataset.get_shape("Target")[0];

        if(decoder_seq != target_seq)
            throw runtime_error("Decoder and target sequence lengths must match.");

        const Index embedding_dimension     = 768;
        const Index heads_number            = 12;
        const Index feed_forward_dimension  = 3072;
        const Index layers_number           = 6;

        Transformer transformer(input_seq,
                                decoder_seq,
                                input_vocab,
                                output_vocab,
                                embedding_dimension,
                                heads_number,
                                feed_forward_dimension,
                                layers_number);

        transformer.set_dropout_rate(float(0.1));

        cout << "[OASST2] train="     << dataset.get_samples_number("Training")
             << " val="               << dataset.get_samples_number("Validation")
             << " test="              << dataset.get_samples_number("Testing")
             << " input_vocab="       << input_vocab
             << " target_vocab="      << output_vocab
             << " input_seq="         << input_seq
             << " decoder_seq="       << decoder_seq
             << " params="            << transformer.get_parameters_number() << endl;

        const std::filesystem::path weights_file = "/tmp/oasst2_weights.bin";

        if (std::filesystem::exists(weights_file))
        {
            cout << "Loading cached weights from " << weights_file
                 << " (skipping training)" << endl;
            transformer.load_parameters_binary(weights_file);
        }
        else
        {
            TrainingStrategy training_strategy(&transformer, &dataset);

            training_strategy.set_loss("CrossEntropyError3d");
            training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

            auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
            if(!adam) throw runtime_error("AdaptiveMomentEstimation optimizer not found.");

            adam->set_batch_size(32);
            adam->set_learning_rate(float(5e-4));
            adam->set_maximum_epochs(29);
            adam->set_display_period(1);

            const auto t0 = steady_clock::now();
            training_strategy.train();
            const auto t1 = steady_clock::now();

            cout << "\nTotal training time (oasst2): "
                 << duration_cast<milliseconds>(t1 - t0).count() / 1000.0 << " s" << endl;

            std::ofstream weights_out(weights_file, std::ios::binary);
            if (!weights_out)
                throw runtime_error("Cannot open " + weights_file.string() + " for writing");
            const Index parameters_floats = transformer.get_parameters_size();
            weights_out.write(reinterpret_cast<const char*>(transformer.get_parameters_data()),
                              parameters_floats * sizeof(float));
            cout << "Saved " << parameters_floats << " floats ("
                 << parameters_floats * sizeof(float) << " bytes) to "
                 << weights_file << endl;
        }

        cout << "\n================ OASST2 CHAT ================\n";

        TransformerDecoder::SamplingConfig sampling_config;
        sampling_config.temperature        = 0.8f;
        sampling_config.top_p              = 0.9f;
        sampling_config.top_k              = 50;
        sampling_config.repetition_penalty = 1.15f;

        TransformerDecoder decoder(transformer, dataset);
        decoder.chat(sampling_config);

        cout << "=============================================\n";
        */

        // ====================  TINYCHAT + OASST2 + ULTRACHAT (combined3)  =======
        // Concatenation of all three datasets:
        //   tinychat   ~2.00M pairs   short Q&A volume
        //   OASST2       ~27K pairs   diverse human-written conversations
        //   UltraChat  ~274K pairs    high-quality synthetic multi-turn chat
        // Total ~2.30M pairs.
        //
        // Model bumped to ~110M params (d_model=640, layers=6, heads=10, ff=2560)
        // — bigger than the 57M proven config but conservative enough to fit in
        // 16 GB VRAM with batch=64 and vocab cap 25K. Designed for ~25-35h of
        // training on the 4080 (10 epochs). Won't OOM.
        /*
        Configuration::instance().set(Device::CUDA, Type::BF16);

        set_seed(42);

        const std::filesystem::path combined3_pairs = "/home/artelnics/Documents/openNN/opennn/temp/combined3_pairs.tsv";

        // Vocabulary cap 25K; drop hapax legomena (≥2 occurrences required).
        // 25K keeps all of tinychat (17K) plus the most frequent OASST2 +
        // UltraChat additions. The rare long tail is dropped to keep the
        // output-projection tensor manageable.
        LanguageDataset dataset(combined3_pairs, 25000, 2);
        dataset.split_samples_random(0.95, 0.025, 0.025);

        const Index input_vocab  = dataset.get_input_vocabulary_size();
        const Index output_vocab = dataset.get_target_vocabulary_size();
        const Index input_seq    = dataset.get_shape("Input")[0];
        const Index decoder_seq  = dataset.get_shape("Decoder")[0];
        const Index target_seq   = dataset.get_shape("Target")[0];

        if(decoder_seq != target_seq)
            throw runtime_error("Decoder and target sequence lengths must match.");

        const Index embedding_dimension     = 640;
        const Index heads_number            = 10;
        const Index feed_forward_dimension  = 2560;
        const Index layers_number           = 6;

        Transformer transformer(input_seq,
                                decoder_seq,
                                input_vocab,
                                output_vocab,
                                embedding_dimension,
                                heads_number,
                                feed_forward_dimension,
                                layers_number);

        transformer.set_dropout_rate(float(0.1));

        cout << "[COMBINED3] train="  << dataset.get_samples_number("Training")
             << " val="               << dataset.get_samples_number("Validation")
             << " test="              << dataset.get_samples_number("Testing")
             << " input_vocab="       << input_vocab
             << " target_vocab="      << output_vocab
             << " input_seq="         << input_seq
             << " decoder_seq="       << decoder_seq
             << " params="            << transformer.get_parameters_number() << endl;

        const std::filesystem::path weights_file = "/tmp/combined3_weights.bin";

        if (std::filesystem::exists(weights_file))
        {
            cout << "Loading cached weights from " << weights_file
                 << " (skipping training)" << endl;
            transformer.load_parameters_binary(weights_file);
        }
        else
        {
            TrainingStrategy training_strategy(&transformer, &dataset);

            training_strategy.set_loss("CrossEntropyError3d");
            training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

            auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
            if(!adam) throw runtime_error("AdaptiveMomentEstimation optimizer not found.");

            adam->set_batch_size(64);     // safe headroom for 110M params + vocab 25K + decoder_seq=251
            adam->set_learning_rate(float(5e-4));
            adam->set_maximum_epochs(9);   // 10 epochs (loop runs 0..9 inclusive)
            adam->set_display_period(1);

            const auto t0 = steady_clock::now();
            training_strategy.train();
            const auto t1 = steady_clock::now();

            cout << "\nTotal training time (combined3): "
                 << duration_cast<milliseconds>(t1 - t0).count() / 1000.0 << " s" << endl;

            std::ofstream weights_out(weights_file, std::ios::binary);
            if (!weights_out)
                throw runtime_error("Cannot open " + weights_file.string() + " for writing");
            const Index parameters_floats = transformer.get_parameters_size();
            weights_out.write(reinterpret_cast<const char*>(transformer.get_parameters_data()),
                              parameters_floats * sizeof(float));
            cout << "Saved " << parameters_floats << " floats ("
                 << parameters_floats * sizeof(float) << " bytes) to "
                 << weights_file << endl;
        }

        cout << "\n================ COMBINED3 CHAT ================\n";

        TransformerDecoder::SamplingConfig sampling_config;
        sampling_config.temperature        = 0.8f;
        sampling_config.top_p              = 0.9f;
        sampling_config.top_k              = 50;
        sampling_config.repetition_penalty = 1.15f;

        TransformerDecoder decoder(transformer, dataset);
        decoder.chat(sampling_config);

        cout << "===============================================\n";
        */

        cout << "Bye!" << endl;
        return 0;
#endif
    }
    catch(const exception& e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
