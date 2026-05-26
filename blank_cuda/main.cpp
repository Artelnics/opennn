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
#include <vector>

#include "../opennn/image_dataset.h"
#include "../opennn/language_dataset.h"
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

        Configuration::instance().set(Device::CUDA, Type::BF16, Type::BF16);
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

        Configuration::instance().set(Device::CUDA, Type::BF16, Type::BF16);

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
        // ============== SDPA vs unfused — seq_len sweep ==============
        // Drives a synthetic dataset where every pair has exactly seq_len-2
        // real tokens. Same Transformer config for every seq_len so the only
        // varying axis is the attention shape. The policy now lives in
        // MultiHeadAttention (sdpa_auto + sdpa_min_sequence_length); we toggle
        // it through Transformer::set_attention_sdpa_min_sequence_length to
        // force SDPA on (threshold=1) or off (threshold=MAX).

        Configuration::instance().set(Device::CUDA, Type::BF16, Type::BF16);
        Backend::instance();

        const vector<Index> seq_lens = { 64, 96, 128, 192, 256, 384, 512, 768 };
        const Index   n_pairs                = 1024;
        const Index   vocab_size             = 500;
        const Index   embedding_dimension    = 256;
        const Index   heads_number           = 8;
        const Index   feed_forward_dimension = 1024;
        const Index   layers_number          = 2;
        const Index   batch_size             = 32;
        const Index   warmup_epochs          = 2;
        const Index   timed_epochs           = 3;

        struct BenchResult { Index seq_len; double sdpa_sec; double unfused_sec; };
        vector<BenchResult> results;

        auto run_one = [&](Index seq_len, bool force_sdpa) -> double
        {
            const Index tokens_per_side = seq_len - 2;   // +2 for START/END pads
            const filesystem::path tsv = format("/tmp/sdpa_bench_seq{}.tsv", seq_len);

            set_seed(42);
            write_synthetic_pairs(tsv, n_pairs, tokens_per_side, vocab_size);

            set_seed(42);
            LanguageDataset dataset(tsv);
            dataset.split_samples_random(0.9f, 0.0f, 0.1f);

            Transformer transformer(dataset.get_shape("Input")[0],
                                    dataset.get_shape("Decoder")[0],
                                    dataset.get_input_vocabulary_size(),
                                    dataset.get_target_vocabulary_size(),
                                    embedding_dimension,
                                    heads_number,
                                    feed_forward_dimension,
                                    layers_number);
            transformer.set_dropout_rate(0.0f);

            // Layer-level policy: threshold=1 forces SDPA at every seq_len,
            // MAX disables it entirely. Must be set BEFORE training because
            // forward_scratch_specs() reads use_sdpa at compile time.
            if (force_sdpa)
                transformer.set_attention_sdpa_min_sequence_length(1);
            else
                transformer.set_attention_sdpa_auto(false);

            TrainingStrategy ts(&transformer, &dataset);
            ts.set_loss("CrossEntropyError3d");
            ts.set_optimization_algorithm("AdaptiveMomentEstimation");

            auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(ts.get_optimization_algorithm());
            if (!adam) throw runtime_error("Adam optimizer not found.");
            adam->set_batch_size(batch_size);
            adam->set_learning_rate(5e-4f);
            adam->set_display(false);

            adam->set_maximum_epochs(warmup_epochs - 1);  // train() runs max+1 epochs
            ts.train();

            adam->set_maximum_epochs(timed_epochs - 1);
            const auto t0 = steady_clock::now();
            ts.train();
            const auto t1 = steady_clock::now();

            return duration_cast<microseconds>(t1 - t0).count() / 1e6 / double(timed_epochs);
        };

        cout << "\n=================== SDPA vs UNFUSED benchmark ===================\n";
        cout << "config: emb=" << embedding_dimension
             << " heads=" << heads_number
             << " ffn=" << feed_forward_dimension
             << " layers=" << layers_number
             << " batch=" << batch_size
             << " samples=" << n_pairs
             << " warmup=" << warmup_epochs << "ep timed=" << timed_epochs << "ep (avg)\n\n";

        cout << left << setw(10) << "seq_len"
             << right << setw(14) << "sdpa (s/ep)"
             << setw(16) << "unfused (s/ep)"
             << setw(14) << "speedup" << "\n";
        cout << string(54, '-') << "\n";

        for (Index seq_len : seq_lens)
        {
            const double t_sdpa    = run_one(seq_len, /*force_sdpa=*/true);
            const double t_unfused = run_one(seq_len, /*force_sdpa=*/false);
            results.push_back({seq_len, t_sdpa, t_unfused});

            cout << left << setw(10) << seq_len
                 << right << fixed << setprecision(3) << setw(14) << t_sdpa
                 << setw(16) << t_unfused
                 << setw(13) << setprecision(2) << (t_unfused / t_sdpa) << "x" << "\n"
                 << flush;
        }
        cout << "=================================================================\n";
        

        // ====================  LLM experiments (commented out)  ====================
        // The blocks below were used to fine-tune chat-style Transformers on
        // tinychat / OASST2 / UltraChat combinations. Restored above is the
        // canonical translation benchmark. To re-run the LLM experiments,
        // uncomment the desired block.

        // ====================  TINYCHAT (instruction-tuning)  ====================
        /*
        Configuration::instance().set(Device::CUDA, Type::BF16, Type::BF16);

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
        Configuration::instance().set(Device::CUDA, Type::BF16, Type::BF16);

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
        Configuration::instance().set(Device::CUDA, Type::BF16, Type::BF16);

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

#endif
#endif

        cout << "Bye!" << endl;
        return 0;
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
