//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   C U D A
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <vector>
#include <filesystem>

#include "../opennn/configuration.h"
#include "../opennn/device_backend.h"
#include "../opennn/neural_network.h"
#include "../opennn/standard_networks.h"

#include "../opennn/tabular_dataset.h"
#include "../opennn/time_series_dataset.h"
#include "../opennn/image_dataset.h"
#include "../opennn/language_dataset.h"

#include "../opennn/scaling_layer.h"
#include "../opennn/dense_layer.h"
#include "../opennn/long_short_term_memory_layer.h"
#include "../opennn/recurrent_layer.h"

#include "../opennn/loss.h"
#include "../opennn/training_strategy.h"
#include "../opennn/testing_analysis.h"
#include "../opennn/stochastic_gradient_descent.h"
#include "../opennn/adaptive_moment_estimation.h"
#include "../opennn/transformer_decoder.h"
#include "../opennn/random_utilities.h"

using namespace opennn;
using namespace std::chrono;

int main(int argc, char** argv)
{
    try
    {
        // ====================================================================
        // blank_cuda — five GPU training benchmarks, all disabled (`#if 0`).
        // Enable exactly ONE block by switching its `#if 0` to `#if 1`.
        // ====================================================================

        // --------------------------------------------------------------------
        // 1) HIGGS — dense DNN, binary classification (TabularDataset)
        //    Dataset already exists at /tmp/HIGGS.csv (28 features, col 0 = label).
        //    Network: Scaling -> 5x Dense(300, tanh) -> Dense(1, sigmoid).
        //    Args: argv[1] = batch size, argv[2] = max epochs.
        // --------------------------------------------------------------------
#if 0
        const Index batch_size = argc > 1 ? Index(stoll(argv[1])) : Index(100);
        const Index maximum_epochs = argc > 2 ? Index(stoll(argv[2])) : Index(20);

        cout << "OpenNN. HIGGS 5x300 DNN GPU BF16 benchmark."
             << " batch=" << batch_size << " max_epochs=" << maximum_epochs << endl;

        Configuration::instance().set(Device::CUDA, Type::BF16);
        Backend::instance();
        set_seed(42);

        TabularDataset dataset("/home/artelnics/Documents/datasets/higgs/HIGGS.csv", ",", false, false);

        vector<Index> input_variables(28);
        iota(input_variables.begin(), input_variables.end(), 1);
        dataset.set_variable_indices(input_variables, {0});
        dataset.set_variable_type(0, VariableType::Binary);
        dataset.split_samples_sequential(10.0f / 11.0f, 0.5f / 11.0f, 0.5f / 11.0f);

        cout << "[DATASET] train=" << dataset.get_sample_indices("Training").size()
             << " val="            << dataset.get_sample_indices("Validation").size()
             << " test="           << dataset.get_sample_indices("Testing").size()
             << " input="          << dataset.get_shape("Input")[0]
             << " target="         << dataset.get_shape("Target")[0] << endl;

        NeuralNetwork network;
        network.add_layer(make_unique<Scaling>(dataset.get_shape("Input")));

        for (Index i = 0; i < 5; ++i)
            network.add_layer(make_unique<opennn::Dense>(network.get_output_shape(),
                                                         Shape{300}, "Tanh", false,
                                                         format("higgs_hidden_{}", i + 1)));

        network.add_layer(make_unique<opennn::Dense>(network.get_output_shape(),
                                                     dataset.get_shape("Target"),
                                                     "Sigmoid", false, "higgs_output"));
        network.compile();
        network.set_parameters_glorot();

        cout << "HIGGS DNN params=" << network.get_parameters_number() << endl;

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.set_loss("CrossEntropy");
        training_strategy.set_optimization_algorithm("StochasticGradientDescent");
        training_strategy.get_loss()->set_regularization("L2");
        training_strategy.get_loss()->set_regularization_weight(1.0e-5f);

        auto* sgd = dynamic_cast<StochasticGradientDescent*>(training_strategy.get_optimization_algorithm());
        if (!sgd) throw runtime_error("StochasticGradientDescent optimizer not found.");
        sgd->set_batch_size(batch_size);
        sgd->set_initial_learning_rate(0.05f);
        sgd->set_initial_decay(0.0202f);
        sgd->set_momentum(0.9f);
        sgd->set_nesterov(false);
        sgd->set_workers_number(8);
        sgd->set_maximum_epochs(maximum_epochs);
        sgd->set_maximum_validation_failures(100);
        sgd->set_display_period(1);

        const auto t0 = steady_clock::now();
        training_strategy.train();
        const auto t1 = steady_clock::now();
        cout << "\nTotal training time: "
             << duration_cast<milliseconds>(t1 - t0).count() / 1000.0 << " s" << endl;

        TestingAnalysis testing_analysis(&network, &dataset);
        testing_analysis.set_batch_size(10000);
        const TestingAnalysis::RocAnalysis roc = testing_analysis.perform_roc_analysis();
        cout << "\nAUC: " << roc.area_under_curve
             << "  optimal_threshold: " << roc.optimal_threshold << endl;
        cout << "\nConfusion (threshold 0.5):\n"
             << testing_analysis.calculate_confusion(0.5f) << endl;

        cout << "Bye!" << endl;
        return 0;
#endif

        // --------------------------------------------------------------------
        // 2) WEATHER — LSTM forecasting (TimeSeriesDataset)
        //    Jena Climate (Max Planck): 10-minute weather records, 14 numeric
        //    variables. The CSV was preprocessed (datasets/weather_jena/) to drop
        //    the "Date Time" column and move "T (degC)" to the last column, which
        //    TimeSeriesDataset forecasts by default (the other 13 vars are inputs).
        //    Network: ForecastingLstmNetwork (Scaling -> LSTM stack -> Dense).
        // --------------------------------------------------------------------
#if 0
        cout << "OpenNN. Weather (Jena) LSTM forecasting GPU FP32 benchmark." << endl;

        Configuration::instance().set(Device::CUDA, Type::FP32);
        Backend::instance();
        set_seed(42);

        const filesystem::path dataset_path =
            "/home/artelnics/Documents/datasets/weather_jena/jena_weather.csv";

        TimeSeriesDataset dataset(dataset_path, ",", /*has_header=*/true, /*has_sample_ids=*/false);
        // Target = last column ("T (degC)"); the other 13 variables are inputs.

        const Index past_time_steps   = 72;   // 12 h of history (10-min cadence)
        const Index future_time_steps = 1;    // forecast horizon
        dataset.set_past_time_steps(past_time_steps);
        dataset.set_future_time_steps(future_time_steps);
        dataset.set_multi_target(future_time_steps > 1);

        // Time series: keep temporal order, do not shuffle.
        dataset.split_samples_sequential(0.70f, 0.15f, 0.15f);

        const Shape hidden_units{64};   // LSTM hidden size(s); add more for a stacked LSTM

        ForecastingLstmNetwork network(dataset.get_input_shape(),
                                       hidden_units,
                                       dataset.get_target_shape());
        network.compile();
        network.set_parameters_glorot();

        cout << "Weather LSTM params=" << network.get_parameters_number() << endl;

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.set_loss("MeanSquaredError");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        if (!adam) throw runtime_error("AdaptiveMomentEstimation optimizer not found.");
        adam->set_batch_size(64);
        adam->set_learning_rate(0.01f);
        adam->set_maximum_epochs(300);
        adam->set_maximum_validation_failures(25);
        adam->set_display_period(10);

        const auto t0 = steady_clock::now();
        training_strategy.train();
        const auto t1 = steady_clock::now();
        cout << "\nTotal training time: "
             << duration_cast<milliseconds>(t1 - t0).count() / 1000.0 << " s" << endl;

        TestingAnalysis testing_analysis(&network, &dataset);
        const VectorR testing_errors = testing_analysis.calculate_errors("Testing");
        if (testing_errors.size() >= 3)
            cout << "Test RMSE: " << testing_errors(2) << endl;

        cout << "Bye!" << endl;
        return 0;
#endif

        // --------------------------------------------------------------------
        // 3) IMAGENET — ResNet-50 (ImageDataset)
        //    Full ImageNet-1k needs image-net.org credentials, so this uses
        //    Imagenette (fast.ai's public 10-class ImageNet subset) as a drop-in
        //    proxy. ImageDataset takes the root as ONE subfolder per class (no
        //    train/val split — split_samples_random does that), so train+val were
        //    merged into datasets/imagenette/ (10 class folders directly) and every image was
        //    preprocessed to a uniform 160x160 (ImageDataset adopts the first
        //    image's size for the whole set). ResNet-50 = bottleneck blocks
        //    {3,4,6,3}, filters {64,128,256,512}.
        // --------------------------------------------------------------------
#if 0
        cout << "OpenNN. ImageNet (Imagenette) ResNet-50 GPU FP32 benchmark." << endl;

        Configuration::instance().set(Device::CUDA, Type::FP32);  // switch to Type::BF16 for the fast tensor-core path
        Backend::instance();
        set_seed(42);

        // Class-per-subfolder root (10 classes). Swap for the real ImageNet-1k
        // train root once credentials / a mirror are available.
        const filesystem::path dataset_path =
            "/home/artelnics/Documents/datasets/imagenette";

        ImageDataset dataset(dataset_path);
        dataset.split_samples_random(0.80f, 0.10f, 0.10f);

        const Shape input_shape  = dataset.get_shape("Input");
        const Shape target_shape = dataset.get_shape("Target");

        cout << "[DATASET] train=" << dataset.get_samples_number("Training")
             << " val="            << dataset.get_samples_number("Validation")
             << " test="           << dataset.get_samples_number("Testing")
             << " input="          << input_shape[0] << "x" << input_shape[1] << "x" << input_shape[2]
             << " classes="        << target_shape[0] << endl;

        // ResNet-50 geometry (use_bottleneck = true). For a lighter run use
        // ResNet-18: blocks {2,2,2,2}, use_bottleneck = false.
        ResNet network(input_shape,
                       {3, 4, 6, 3},
                       Shape{64, 128, 256, 512},
                       target_shape,
                       /*use_bottleneck=*/true);

        auto* scaling = dynamic_cast<Scaling*>(network.get_first(LayerType::Scaling));
        if (!scaling) throw runtime_error("ResNet scaling layer not found.");
        scaling->set_descriptives({
            Descriptives(0.0f, 255.0f, 0.485f * 255.0f, 0.229f * 255.0f),
            Descriptives(0.0f, 255.0f, 0.456f * 255.0f, 0.224f * 255.0f),
            Descriptives(0.0f, 255.0f, 0.406f * 255.0f, 0.225f * 255.0f)
        });
        scaling->set_scalers("MeanStandardDeviation");

        cout << "ResNet-50 params=" << network.get_parameters_number() << endl;

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.set_loss("CrossEntropy");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        training_strategy.get_loss()->set_regularization("None");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        if (!adam) throw runtime_error("AdaptiveMomentEstimation optimizer not found.");
        adam->set_batch_size(32);   // ResNet-50 @160x160; raise if VRAM allows
        adam->set_learning_rate(1.0e-3f);
        adam->set_workers_number(8);
        adam->set_maximum_epochs(90);
        adam->set_maximum_validation_failures(10);
        adam->set_display_period(1);

        const auto t0 = steady_clock::now();
        training_strategy.train();
        const auto t1 = steady_clock::now();
        cout << "\nTotal training time: "
             << duration_cast<milliseconds>(t1 - t0).count() / 1000.0 << " s" << endl;

        TestingAnalysis testing_analysis(&network, &dataset);
        testing_analysis.set_batch_size(64);
        cout << "\nConfusion matrix:\n" << testing_analysis.calculate_confusion() << endl;

        cout << "Bye!" << endl;
        return 0;
#endif

        // --------------------------------------------------------------------
        // 4) EN -> DE TRANSLATION — Transformer (LanguageDataset)
        //    The "Attention Is All You Need" WMT14 English->German task. The raw
        //    WMT14 parallel corpora (Europarl + Common Crawl + News Commentary)
        //    were combined into tab-separated "english <TAB> german" files in
        //    datasets/wmt14_en_de/:
        //      - wmt14_en_de.txt         full ~4.5M pairs (authentic WMT14)
        //      - wmt14_en_de.subset.txt  100k clean News-Commentary pairs (used here)
        //    The model below is smaller than the paper so the subset trains in
        //    reasonable time; paper base is d_model=512, h=8, d_ff=2048, N=6.
        // --------------------------------------------------------------------
#if 1
        cout << "OpenNN. EN->DE Transformer GPU FP32 benchmark." << endl;

        Configuration::instance().set(Device::CUDA, Type::FP32);
        Backend::instance();
        set_seed(42);

        const filesystem::path dataset_path =
            "/home/artelnics/Documents/datasets/wmt14_en_de/wmt14_en_de.cap60.txt";

        LanguageDataset language_dataset(dataset_path, 37000);

        const Index input_vocabulary_size  = language_dataset.get_input_vocabulary_size();
        const Index output_vocabulary_size = language_dataset.get_target_vocabulary_size();
        const Index input_sequence_length   = language_dataset.get_shape("Input")[0];
        const Index decoder_sequence_length = language_dataset.get_shape("Decoder")[0];
        const Index target_sequence_length  = language_dataset.get_shape("Target")[0];

        if (decoder_sequence_length != target_sequence_length)
            throw runtime_error("Decoder and target sequence lengths must match.");

        // Paper-base transformer ("Attention Is All You Need"): 512/8/2048/6.
        const Index embedding_dimension    = 512;
        const Index heads_number           = 8;
        const Index feed_forward_dimension  = 2048;
        const Index layers_number           = 6;

        Transformer transformer(input_sequence_length,
                                decoder_sequence_length,
                                input_vocabulary_size,
                                output_vocabulary_size,
                                embedding_dimension,
                                heads_number,
                                feed_forward_dimension,
                                layers_number);

        cout << "Transformer params=" << transformer.get_parameters_number() << endl;

        const filesystem::path parameters_path =
            "/home/artelnics/Documents/datasets/wmt14_en_de/wmt14_en_de_parameters_paperbase.bin";

        if (filesystem::exists(parameters_path))
        {
            cout << "Found saved parameters at " << parameters_path
                 << "\n-> skipping training, loading weights for inference." << endl;
            transformer.load_parameters_binary(parameters_path);
        }
        else
        {
            cout << "No saved parameters at " << parameters_path
                 << "\n-> training from scratch." << endl;

            TrainingStrategy training_strategy(&transformer, &language_dataset);
            training_strategy.set_loss("CrossEntropyError3d");
            training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

            auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
            if (!adam) throw runtime_error("AdaptiveMomentEstimation optimizer not found.");
            adam->set_batch_size(64);
            adam->set_learning_rate(0.0005f);
            adam->set_maximum_epochs(1);
            adam->set_maximum_time(288000.0f);
            adam->set_maximum_validation_failures(1000000);
            adam->set_display_period(1);

            const auto t0 = steady_clock::now();
            training_strategy.train();
            const auto t1 = steady_clock::now();
            cout << "\nTotal training time: "
                 << duration_cast<milliseconds>(t1 - t0).count() / 1000.0 << " s" << endl;

            transformer.save_parameters_binary(parameters_path);
            cout << "Saved parameters (binary) to " << parameters_path << endl;
        }

        // Inference (TransformerDecoder is GPU-only): interactive EN->DE chat.
        // Type an English sentence and press Enter; empty line or Ctrl+D exits.
        TransformerDecoder decoder(transformer, language_dataset);
        cout << "\n================ EN -> DE CHAT ================" << endl;
        decoder.chat();

        return 0;
#endif

        // --------------------------------------------------------------------
        // 5) TOM & JERRY — CNN, binary image classification (ImageDataset)
        //    Dataset: datasets/tom_and_jerry_bmp/ with two class folders
        //    (jerry/, tom/). The dataset is NOT preprocessed here: OpenNN fixes
        //    the input size from the FIRST image of the set and resizes every
        //    other image to match it (resize_image() runs inside ImageDataset on
        //    load). ImageDataset uses StorageMode::BinaryFile by default — a
        //    disk-backed pixel cache at <root>/.cache/images.bin, built once then
        //    read on demand (no full in-RAM matrix).
        //    Network: ImageClassificationNetwork (a conv/pool CNN). NOTE: ResNet
        //    is not used here because its classifier is hardcoded to "Softmax",
        //    which is degenerate for a 1-output binary target; ImageClassification
        //    Network correctly uses Sigmoid when the output size is 1.
        //    Args: argv[1] = batch size (default 32), argv[2] = max epochs (50).
        // --------------------------------------------------------------------
#if 0
        const Index batch_size = argc > 1 ? Index(stoll(argv[1])) : Index(32);
        const Index maximum_epochs = argc > 2 ? Index(stoll(argv[2])) : Index(50);

        cout << "OpenNN. Tom & Jerry CNN GPU FP32 benchmark."
             << " batch=" << batch_size << " max_epochs=" << maximum_epochs << endl;

        Configuration::instance().set(Device::CUDA, Type::FP32);  // Type::BF16 for the tensor-core path
        Backend::instance();
        set_seed(42);

        const filesystem::path dataset_path =
            "/home/artelnics/Documents/datasets/tom_and_jerry_bmp";

        // StorageMode::BinaryFile (ImageDataset default). OpenNN reads the first
        // image to fix the input size, then resizes every other image to match.
        ImageDataset dataset(dataset_path);
        dataset.split_samples_random(0.80f, 0.10f, 0.10f);

        const Shape input_shape  = dataset.get_shape("Input");
        const Shape target_shape = dataset.get_shape("Target");

        cout << "[DATASET] train=" << dataset.get_samples_number("Training")
             << " val="            << dataset.get_samples_number("Validation")
             << " test="           << dataset.get_samples_number("Testing")
             << " input="          << input_shape[0] << "x" << input_shape[1] << "x" << input_shape[2]
             << " classes="        << target_shape[0] << endl;

        // Conv(3x3,ReLU,Same)+MaxPool(2x2) per filter count -> Dense(<=128,ReLU)
        // -> Dense(output, Sigmoid for binary / Softmax for multi-class).
        ImageClassificationNetwork network(input_shape,
                                           Shape{16, 32, 64, 128},
                                           target_shape);

        cout << "Tom & Jerry CNN params=" << network.get_parameters_number() << endl;

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.set_loss("CrossEntropy");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
        training_strategy.get_loss()->set_regularization("None");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        if (!adam) throw runtime_error("AdaptiveMomentEstimation optimizer not found.");
        adam->set_batch_size(batch_size);
        adam->set_learning_rate(1.0e-3f);
        adam->set_workers_number(8);
        adam->set_maximum_epochs(maximum_epochs);
        adam->set_maximum_validation_failures(10);
        adam->set_display_period(1);

        const auto t0 = steady_clock::now();
        training_strategy.train();
        const auto t1 = steady_clock::now();
        cout << "\nTotal training time: "
             << duration_cast<milliseconds>(t1 - t0).count() / 1000.0 << " s" << endl;

        TestingAnalysis testing_analysis(&network, &dataset);
        testing_analysis.set_batch_size(64);
        const TestingAnalysis::RocAnalysis roc = testing_analysis.perform_roc_analysis();
        cout << "\nAUC: " << roc.area_under_curve
             << "  optimal_threshold: " << roc.optimal_threshold << endl;
        cout << "\nConfusion (threshold 0.5):\n"
             << testing_analysis.calculate_confusion(0.5f) << endl;

        cout << "Bye!" << endl;
        return 0;
#endif

        cout << "blank_cuda: all five benchmark blocks are disabled (#if 0).\n"
                "Enable one by switching its `#if 0` to `#if 1` and rebuilding." << endl;
        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
