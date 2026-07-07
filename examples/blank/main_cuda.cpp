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
#include "../opennn/memory_debug.h"
#include "../opennn/neural_network.h"
#include "../opennn/standard_networks.h"

#include "../opennn/tabular_dataset.h"
#include "../opennn/time_series_dataset.h"
#include "../opennn/image_dataset.h"
#include "../opennn/language_dataset.h"
#include "../opennn/text_generation_dataset.h"

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
        // blank_cuda — seven GPU training benchmarks. Block 7 (GPT) is the
        // enabled one; blocks 1-6 are `#if 0`. Enable exactly ONE block at a
        // time by switching its `#if 0` to `#if 1` (and the others back to 0).
        // ====================================================================

        // --------------------------------------------------------------------
        // 1) HIGGS — dense DNN, binary classification (TabularDataset)
        //    Dataset already exists at /tmp/HIGGS.csv (28 features, col 0 = label).
        //    Network: Scaling -> 5x Dense(300, tanh) -> Dense(1, sigmoid).
        //    Args: argv[1] = batch size, argv[2] = max epochs.
        // --------------------------------------------------------------------
#if 0
        const Index batch_size = argc > 1 ? Index(stoll(argv[1])) : Index(100);
        const Index maximum_epochs = argc > 2 ? Index(stoll(argv[2])) : Index(1);
        const string precision = argc > 3 ? argv[3] : "fp32";
        const string mode = argc > 4 ? argv[4] : "probe";   // "speed"/"speedval" = throughput
        const bool speed_mode = (mode == "speed" || mode == "speedval");
        const bool speed_with_val = (mode == "speedval");
        const Type training_type = (precision == "bf16") ? Type::BF16 : Type::FP32;

        cout << "OpenNN. HIGGS 5x300 DNN GPU benchmark."
             << " batch=" << batch_size << " max_epochs=" << maximum_epochs
             << " precision=" << precision << endl;

        Configuration::instance().set(Device::CUDA, training_type);
        Backend::instance();
        set_seed(42);

        TabularDataset dataset("/home/artelnics/Documents/datasets/higgs/HIGGS.csv", ",", false, false);

        vector<Index> input_variables(28);
        iota(input_variables.begin(), input_variables.end(), 1);
        dataset.set_variable_indices(input_variables, {0});
        dataset.set_variable_type(0, VariableType::Binary);
        if (speed_mode)
        {
            // Throughput mode, GPU-resident data. "speed" = all-train (no validation);
            // "speedval" = keep the validation split so per-epoch validation runs.
            if (speed_with_val)
                dataset.split_samples_sequential(10.0f / 11.0f, 0.5f / 11.0f, 0.5f / 11.0f);
            else
                dataset.split_samples_sequential(1.0f, 0.0f, 0.0f);
            dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);
        }
        else
            dataset.split_samples_sequential(10.0f / 11.0f, 0.5f / 11.0f, 0.5f / 11.0f);

        cout << "[DATASET] train=" << dataset.get_sample_indices("Training").size()
             << " val="            << dataset.get_sample_indices("Validation").size()
             << " test="           << dataset.get_sample_indices("Testing").size()
             << " input="          << dataset.get_shape("Input")[0]
             << " target="         << dataset.get_shape("Target")[0] << endl;

        const string hidden_activation = argc > 6 ? argv[6] : "Tanh";

        NeuralNetwork network;
        network.add_layer(make_unique<Scaling>(dataset.get_shape("Input")));

        for (Index i = 0; i < 5; ++i)
            network.add_layer(make_unique<opennn::Dense>(network.get_output_shape(),
                                                         Shape{300}, hidden_activation, false,
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

        if (speed_mode)
        {
            const bool use_graph = argc > 5 ? (stoi(argv[5]) != 0) : true;
            sgd->set_cuda_graph(use_graph);
            const Index timed_epochs = maximum_epochs;
            sgd->set_maximum_epochs(2);             // warmup: absorb autotune / graph capture
            training_strategy.train();
            sgd->set_maximum_epochs(timed_epochs);
            const auto s0 = steady_clock::now();
            training_strategy.train();
            const auto s1 = steady_clock::now();
            const double epoch_s = (duration_cast<milliseconds>(s1 - s0).count() / 1000.0)
                                 / double(timed_epochs);
            const Index train_samples = dataset.get_sample_indices("Training").size();
            cout << "epoch_s=" << epoch_s
                 << " samples_per_sec=" << long(double(train_samples) / epoch_s) << endl;
            cout << "RESULT=OK" << endl;
            return 0;
        }

        const auto t0 = steady_clock::now();
        training_strategy.train();
        const auto t1 = steady_clock::now();
        cout << "\nTotal training time: "
             << duration_cast<milliseconds>(t1 - t0).count() / 1000.0 << " s" << endl;

        // Skip the ROC/confusion testing analysis in probe mode (epochs<=1):
        // the max-batch sweep only needs the training step to fit in memory.
        if (maximum_epochs > 1)
        {
            TestingAnalysis testing_analysis(&network, &dataset);
            testing_analysis.set_batch_size(10000);
            const TestingAnalysis::RocAnalysis roc = testing_analysis.perform_roc_analysis();
            cout << "\nAUC: " << roc.area_under_curve
                 << "  optimal_threshold: " << roc.optimal_threshold << endl;
            cout << "\nConfusion (threshold 0.5):\n"
                 << testing_analysis.calculate_confusion(0.5f) << endl;
        }

        cout << "Bye!" << endl;
        memory_debug::print(cout);
        cout << "RESULT=OK" << endl;
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
#if 0
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

        // --------------------------------------------------------------------
        // 6) CHATGPT — conversational assistant, Transformer (LanguageDataset)
        //    Seq2seq chatbot on the ENCODER-DECODER Transformer (for the
        //    decoder-only GPT see block 7): every line of
        //    the dataset is a single  prompt <TAB> response  pair.
        //    Data is already prepared at dataset_path: Stanford Alpaca (52k
        //    instruction->response) processed to that tab format, 47487 pairs,
        //    token-capped (prompt<=62, response<=126, OpenNN's tokenizer splits
        //    each punctuation char) -> in_len 64 / dec_len 128, fits batch 64
        //    on a 16 GB GPU. Longer caps overflow the paper-base model. Other
        //    good sources: OpenAssistant OASST1, Cornell Movie-Dialogs.
        //    Model: paper-base Transformer (512/8/2048/6); shrink d_model/layers
        //    for faster iteration. Trains with Adam + token cross-entropy, saves
        //    the weights, then drops into an interactive chat() loop.
        //    Args: argv[1]=batch (64), argv[2]=max_epochs (10), argv[3]=fp32|bf16.
        // --------------------------------------------------------------------
#if 0
        const Index batch_size     = argc > 1 ? Index(stoll(argv[1])) : Index(64);
        const Index maximum_epochs = argc > 2 ? Index(stoll(argv[2])) : Index(10);
        const string precision     = argc > 3 ? argv[3] : "bf16";
        const bool  use_graph      = argc > 4 ? (stoi(argv[4]) != 0) : true;
        const Type training_type   = (precision == "bf16") ? Type::BF16 : Type::FP32;

        cout << "OpenNN. ChatGPT-style conversational Transformer (" << precision
             << ") batch=" << batch_size << " max_epochs=" << maximum_epochs
             << " cuda_graph=" << use_graph << endl;

        Configuration::instance().set(Device::CUDA, training_type);
        Backend::instance();
        set_seed(42);

        // One pair per line:  prompt <TAB> response  (BinaryFile .bin cache).
        const filesystem::path dataset_path =
            "/home/artelnics/Documents/datasets/chat/chat_pairs.txt";
        const Index maximum_vocabulary_size = 30000;

        LanguageDataset language_dataset(dataset_path, maximum_vocabulary_size);
        language_dataset.set_sample_roles("Training");  // all-train (no early stop here)

        const Index input_vocabulary_size   = language_dataset.get_input_vocabulary_size();
        const Index output_vocabulary_size  = language_dataset.get_target_vocabulary_size();
        const Index input_sequence_length   = language_dataset.get_shape("Input")[0];
        const Index decoder_sequence_length = language_dataset.get_shape("Decoder")[0];

        cout << "[DATASET] train_pairs=" << language_dataset.get_samples_number("Training")
             << " in_vocab="  << input_vocabulary_size
             << " out_vocab=" << output_vocabulary_size
             << " in_len="    << input_sequence_length
             << " dec_len="   << decoder_sequence_length << endl;

        // Paper-base transformer ("Attention Is All You Need"): 512/8/2048/6.
        const Index embedding_dimension     = 512;
        const Index heads_number            = 8;
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
        transformer.set_dropout_rate(0.1f);

        cout << "Transformer params=" << transformer.get_parameters_number() << endl;

        const filesystem::path parameters_path =
            "/home/artelnics/Documents/datasets/chat/chat_parameters.bin";

        if (filesystem::exists(parameters_path))
        {
            cout << "Found saved parameters at " << parameters_path
                 << "\n-> skipping training, loading weights for chat." << endl;
            transformer.load_parameters_binary(parameters_path);
        }
        else
        {
            cout << "No saved parameters -> training from scratch." << endl;

            TrainingStrategy training_strategy(&transformer, &language_dataset);
            training_strategy.set_loss("CrossEntropyError3d");
            training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

            auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
            if (!adam) throw runtime_error("AdaptiveMomentEstimation optimizer not found.");
            adam->set_batch_size(batch_size);
            adam->set_learning_rate(0.0001f);
            adam->set_maximum_epochs(maximum_epochs);
            adam->set_maximum_validation_failures(1000000);
            adam->set_display_period(1);
            adam->set_cuda_graph(use_graph);

            const auto t0 = steady_clock::now();
            training_strategy.train();
            const auto t1 = steady_clock::now();
            const double train_s = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;
            cout << "\nTotal training time: " << train_s << " s" << endl;
            cout << "epoch_s=" << train_s / double(maximum_epochs) << "\n";

            // Per-Buffer memory breakdown (only prints when OPENNN_MEMORY_DEBUG=1).
            memory_debug::print(cout);

            transformer.save_parameters_binary(parameters_path);
            cout << "Saved parameters (binary) to " << parameters_path << endl;
        }

        // Interactive chat: type a prompt, press Enter; empty line / Ctrl+D exits.
        TransformerDecoder decoder(transformer, language_dataset);
        cout << "\n================ CHAT ================" << endl;
        decoder.chat();

        return 0;
#endif

        // --------------------------------------------------------------------
        // 7) GPT — decoder-only language model, GPT-2-small architecture
        //    (TextGenerationDataset + TextGenerationNetwork).
        //    Next-token prediction on WikiText-103 raw (~536 MB, ~110M
        //    word-level tokens of curated Wikipedia articles), prepared with:
        //      sed -e 's/ @-@ /-/g' -e 's/ @,@ /,/g' -e 's/ @\.@ /./g' \
        //          wiki.train.raw > wiki.train.txt   (Moses artifacts removed)
        //    Alternative corpus (English side of WMT14, parliamentary/news
        //    register): datasets/wmt14_en_de/wmt14_en.txt via argv[5].
        //    GPT-2 small dims: 12 layers / 12 heads / 768 emb / 3072 ff.
        //    Differences vs the real GPT-2: word-level vocabulary (no BPE),
        //    sinusoidal positions (not learned), untied output projection.
        //    Vocabulary capped at 50257 (GPT-2's vocab size).
        //    PRE-LN IS REQUIRED at these widths: post-LN without LR warmup
        //    freezes at the unigram plateau for embedding_dim >= 512 or
        //    ff_dim >= 2048 (verified empirically July 2026: post-LN
        //    768/12/3072 never breaks 6.76 on the WMT subset while pre-LN
        //    reaches 0.7 on shakespeare in 10 epochs; same reason GPT-2
        //    itself is pre-LN). Narrow nets (256/4/512) train either way.
        //    BF16 default (~4x faster than fp32, same convergence). An SDPA
        //    backward bug used to corrupt dQ/dK in pure-bf16 training (the
        //    graph read the head-merged O instead of the forward's BHSD O);
        //    fixed July 2026 in AttentionOperator::apply_sdpa_forward/backward
        //    (private O copy, mirroring the fp32 cast path). Validated: bf16
        //    12L subset now matches fp32 epoch-for-epoch (6.55 -> 2.67).
        //    seq 512 + batch 8 fits a 16 GB GPU; first run tokenizes and
        //    writes the .cache (one-off, minutes). ~24k iters/epoch on the
        //    full corpus; pass the .subset corpus for a quick trial. Weights save/load like block 6 (path derived from
        //    the corpus name). After training: sample generations + chat().
        //    CUDA graph stays OFF: the graph-epoch path collapses transformer
        //    train-to-quality convergence (see project notes).
        //    Args: argv[1]=batch (8), argv[2]=max_epochs (1), argv[3]=fp32|bf16,
        //          argv[4]=sequence_length (512), argv[5]=corpus path override,
        //          argv[6]=learning_rate (1e-4), argv[7]=layers_number (12),
        //          argv[8]=pre_normalization 0|1 (1 = pre-LN, the default),
        //          argv[9]=sdpa 0|1, argv[10]=dropout rate (0.1),
        //          argv[11]=embedding_dim (768), argv[12]=heads (12), argv[13]=ff_dim (3072),
        //          argv[14]=device cuda|cpu (cuda), argv[15]=grad_clip_norm (1.0).
        // --------------------------------------------------------------------
#if 1
        const Index batch_size      = argc > 1 ? Index(stoll(argv[1])) : Index(8);
        const Index maximum_epochs  = argc > 2 ? Index(stoll(argv[2])) : Index(1);
        const string precision      = argc > 3 ? argv[3] : "bf16";
        const Index sequence_length = argc > 4 ? Index(stoll(argv[4])) : Index(512);
        const filesystem::path corpus_path = argc > 5 ? argv[5]
            : "/home/artelnics/Documents/datasets/wikitext103/wiki.train.txt";
        const float learning_rate   = argc > 6 ? stof(argv[6]) : 0.0001f;
        const Index layers_number_arg = argc > 7 ? Index(stoll(argv[7])) : Index(12);
        const bool pre_normalization = argc > 8 ? (stoi(argv[8]) != 0) : true;
        const bool use_sdpa          = argc > 9 ? (stoi(argv[9]) != 0) : true;
        const float dropout_rate     = argc > 10 ? stof(argv[10]) : 0.1f;
        const Index embedding_dimension_arg    = argc > 11 ? Index(stoll(argv[11])) : Index(768);
        const Index heads_number_arg           = argc > 12 ? Index(stoll(argv[12])) : Index(12);
        const Index feed_forward_dimension_arg = argc > 13 ? Index(stoll(argv[13])) : Index(3072);
        const bool on_cpu           = argc > 14 && string(argv[14]) == "cpu";
        const float gradient_clip   = argc > 15 ? stof(argv[15]) : 1.0f;
        const Type training_type    = (precision == "bf16") ? Type::BF16 : Type::FP32;

        cout << "OpenNN. GPT decoder-only LM, GPT-2-small architecture (" << precision
             << ") batch=" << batch_size << " max_epochs=" << maximum_epochs
             << " seq=" << sequence_length << " lr=" << learning_rate
             << " layers=" << layers_number_arg << " preln=" << pre_normalization
             << " sdpa=" << use_sdpa << " dropout=" << dropout_rate
             << "\ncorpus=" << corpus_path << endl;

        Configuration::instance().set(on_cpu ? Device::CPU : Device::CUDA, training_type);
        if (!on_cpu) Backend::instance();
        set_seed(42);

        const Index maximum_vocabulary_size = 50257;

        TextGenerationDataset dataset(corpus_path, sequence_length, maximum_vocabulary_size);
        dataset.split_samples_random(0.95f, 0.04f, 0.01f);

        cout << "[DATASET] blocks=" << dataset.get_samples_number()
             << " train="  << dataset.get_samples_number("Training")
             << " val="    << dataset.get_samples_number("Validation")
             << " vocab="  << dataset.get_vocabulary_size()
             << " seq="    << dataset.get_sequence_length() << endl;

        // GPT-2 small: 12 layers / 12 heads / 768 emb / 3072 ff.
        const Index embedding_dimension    = embedding_dimension_arg;
        const Index heads_number           = heads_number_arg;
        const Index feed_forward_dimension = feed_forward_dimension_arg;
        const Index layers_number          = layers_number_arg;

        TextGenerationNetwork network(sequence_length,
                                      dataset.get_vocabulary_size(),
                                      embedding_dimension,
                                      heads_number,
                                      feed_forward_dimension,
                                      layers_number,
                                      pre_normalization);
        network.set_dropout_rate(dropout_rate);
        if (!use_sdpa) network.set_attention_sdpa_auto(false);

        cout << "GPT params=" << network.get_parameters_number() << endl;

        const filesystem::path parameters_path =
            corpus_path.string() + ".gpt_parameters.bin";

        if (filesystem::exists(parameters_path))
        {
            cout << "Found saved parameters at " << parameters_path
                 << "\n-> skipping training, loading weights for generation." << endl;
            network.load_parameters_binary(parameters_path);
        }
        else
        {
            cout << "No saved parameters -> training from scratch." << endl;

            TrainingStrategy training_strategy(&network, &dataset);
            training_strategy.set_loss("CrossEntropyError3d");
            training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

            auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
            if (!adam) throw runtime_error("AdaptiveMomentEstimation optimizer not found.");
            adam->set_batch_size(batch_size);
            adam->set_learning_rate(learning_rate);
            adam->set_gradient_clip_norm(gradient_clip);
            adam->set_maximum_epochs(maximum_epochs);
            adam->set_maximum_validation_failures(1000000);
            adam->set_display_period(1);

            const auto t0 = steady_clock::now();
            training_strategy.train();
            const auto t1 = steady_clock::now();
            const double train_s = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;
            cout << "\nTotal training time: " << train_s << " s" << endl;
            cout << "epoch_s=" << train_s / double(maximum_epochs) << "\n";

            memory_debug::print(cout);

            network.save_parameters_binary(parameters_path);
            cout << "Saved parameters (binary) to " << parameters_path << endl;
        }

        if (on_cpu) return 0;   // TransformerDecoder generation is GPU-only

        TransformerDecoder generator(network, dataset);

        TransformerDecoder::SamplingConfig sampling;
        sampling.temperature = 0.8f;
        sampling.top_k = 40;
        sampling.maximum_tokens = 40;

        cout << "\n================ GENERATED TEXT ================" << endl;

        const vector<string> prompts =
            {
                "the film received",
                "the city is located in",
                "world war ii began"
            };

        for (const string& prompt : prompts)
        {
            cout << "Prompt:    " << prompt << endl;
            cout << "Generated: " << generator.generate(prompt, sampling) << endl;
            cout << endl;
        }

        // Interactive: type a prompt, press Enter; empty line / Ctrl+D exits.
        cout << "================ GPT CHAT ================" << endl;
        generator.chat(sampling);

        return 0;
#endif

        cout << "blank_cuda: all seven benchmark blocks are disabled (#if 0).\n"
                "Enable one by switching its `#if 0` to `#if 1` and rebuilding." << endl;
        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;
        cout << "RESULT=ERROR" << endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
