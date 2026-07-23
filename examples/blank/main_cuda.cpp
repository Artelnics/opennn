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

#include "opennn/configuration.h"
#include "opennn/device_backend.h"
#include "opennn/memory_debug.h"
#include "opennn/neural_network.h"
#include "opennn/standard_networks.h"

#include "opennn/tabular_dataset.h"
#include "opennn/time_series_dataset.h"
#include "opennn/image_dataset.h"
#include "opennn/text_dataset.h"


#include "opennn/scaling_layer.h"
#include "opennn/dense_layer.h"
#include "opennn/long_short_term_memory_layer.h"
#include "opennn/recurrent_layer.h"

#include "opennn/loss.h"
#include "opennn/training_strategy.h"
#include "opennn/testing_analysis.h"
#include "opennn/stochastic_gradient_descent.h"
#include "opennn/adaptive_moment_estimation.h"
#include "opennn/random_utilities.h"

using namespace opennn;
using namespace chrono;

int main(int argc, char** argv)
{
    try
    {

#if 0
        const Index batch_size = argc > 1 ? Index(stoll(argv[1])) : Index(100);
        const Index maximum_epochs = argc > 2 ? Index(stoll(argv[2])) : Index(1);
        const string precision = argc > 3 ? argv[3] : "fp32";
        const string mode = argc > 4 ? argv[4] : "probe";
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
            sgd->set_maximum_epochs(2);
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

#if 0
        cout << "OpenNN. Weather (Jena) LSTM forecasting GPU FP32 benchmark." << endl;

        Configuration::instance().set(Device::CUDA, Type::FP32);
        Backend::instance();
        set_seed(42);

        const filesystem::path dataset_path =
            "/home/artelnics/Documents/datasets/weather_jena/jena_weather.csv";

        TimeSeriesDataset dataset(dataset_path, ",",                true,                    false);

        const Index past_time_steps   = 72;
        const Index future_time_steps = 1;
        dataset.set_past_time_steps(past_time_steps);
        dataset.set_future_time_steps(future_time_steps);
        dataset.set_multi_target(future_time_steps > 1);

        dataset.split_samples_sequential(0.70f, 0.15f, 0.15f);

        const Shape hidden_units{64};

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

#if 0
        cout << "OpenNN. ImageNet (Imagenette) ResNet-50 GPU FP32 benchmark." << endl;

        Configuration::instance().set(Device::CUDA, Type::FP32);
        Backend::instance();
        set_seed(42);

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

        ResNet network(input_shape,
                       {3, 4, 6, 3},
                       Shape{64, 128, 256, 512},
                       target_shape,
                                          true);

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
        adam->set_batch_size(32);
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

#if 0
        cout << "OpenNN. EN->DE Transformer GPU FP32 benchmark." << endl;

        Configuration::instance().set(Device::CUDA, Type::FP32);
        Backend::instance();
        set_seed(42);

        const filesystem::path dataset_path =
            "/home/artelnics/Documents/datasets/wmt14_en_de/wmt14_en_de.cap60.txt";

        const filesystem::path model_path =
            "/home/artelnics/Documents/datasets/wmt14_en_de/wmt14_en_de_model.json";

        if (filesystem::exists(model_path))
        {
            cout << "Found saved model at " << model_path
                 << "\n-> loading for inference; the corpus is never read." << endl;

            Transformer transformer(model_path);

            cout << "\n================ EN -> DE CHAT ================" << endl;
            transformer.chat();

            return 0;
        }

        unique_ptr<TextDataset> language_dataset = TextDataset::from_sequence_to_sequence(dataset_path, 37000);

        const Index input_vocabulary_size  = language_dataset->get_vocabulary_size();
        const Index output_vocabulary_size = language_dataset->get_target_vocabulary().size();
        const Index input_sequence_length   = language_dataset->get_shape("Input")[0];
        const Index decoder_sequence_length = language_dataset->get_shape("Decoder")[0];
        const Index target_sequence_length  = language_dataset->get_shape("Target")[0];

        if (decoder_sequence_length != target_sequence_length)
            throw runtime_error("Decoder and target sequence lengths must match.");

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

            TrainingStrategy training_strategy(&transformer, language_dataset.get());
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

        transformer.set_input_vocabulary(language_dataset->get_input_vocabulary());
        transformer.set_target_vocabulary(language_dataset->get_target_vocabulary());
        transformer.save(model_path);
        cout << "Saved self-contained model to " << model_path << endl;

        cout << "\n================ EN -> DE CHAT ================" << endl;
        transformer.chat();

        return 0;
#endif

#if 0
        const Index batch_size = argc > 1 ? Index(stoll(argv[1])) : Index(32);
        const Index maximum_epochs = argc > 2 ? Index(stoll(argv[2])) : Index(50);

        cout << "OpenNN. Tom & Jerry CNN GPU FP32 benchmark."
             << " batch=" << batch_size << " max_epochs=" << maximum_epochs << endl;

        Configuration::instance().set(Device::CUDA, Type::FP32);
        Backend::instance();
        set_seed(42);

        const filesystem::path dataset_path =
            "/home/artelnics/Documents/datasets/tom_and_jerry_bmp";

        ImageDataset dataset(dataset_path);
        dataset.split_samples_random(0.80f, 0.10f, 0.10f);

        const Shape input_shape  = dataset.get_shape("Input");
        const Shape target_shape = dataset.get_shape("Target");

        cout << "[DATASET] train=" << dataset.get_samples_number("Training")
             << " val="            << dataset.get_samples_number("Validation")
             << " test="           << dataset.get_samples_number("Testing")
             << " input="          << input_shape[0] << "x" << input_shape[1] << "x" << input_shape[2]
             << " classes="        << target_shape[0] << endl;

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

        const filesystem::path dataset_path =
            "/home/artelnics/Documents/datasets/chat/chat_pairs.txt";
        const Index maximum_vocabulary_size = 30000;

        const filesystem::path model_path =
            "/home/artelnics/Documents/datasets/chat/chat_model.json";

        if (filesystem::exists(model_path))
        {
            cout << "Found saved model at " << model_path
                 << "\n-> loading for chat; the corpus is never read." << endl;

            Transformer transformer(model_path);

            cout << "\n================ CHAT ================" << endl;
            transformer.chat();

            return 0;
        }

        unique_ptr<TextDataset> language_dataset = TextDataset::from_sequence_to_sequence(dataset_path, maximum_vocabulary_size);
        language_dataset->set_sample_roles("Training");

        const Index input_vocabulary_size   = language_dataset->get_vocabulary_size();
        const Index output_vocabulary_size  = language_dataset->get_target_vocabulary().size();
        const Index input_sequence_length   = language_dataset->get_shape("Input")[0];
        const Index decoder_sequence_length = language_dataset->get_shape("Decoder")[0];

        cout << "[DATASET] train_pairs=" << language_dataset->get_samples_number("Training")
             << " in_vocab="  << input_vocabulary_size
             << " out_vocab=" << output_vocabulary_size
             << " in_len="    << input_sequence_length
             << " dec_len="   << decoder_sequence_length << endl;

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

            TrainingStrategy training_strategy(&transformer, language_dataset.get());
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

            memory_debug::print(cout);

            transformer.save_parameters_binary(parameters_path);
            cout << "Saved parameters (binary) to " << parameters_path << endl;
        }

        transformer.set_input_vocabulary(language_dataset->get_input_vocabulary());
        transformer.set_target_vocabulary(language_dataset->get_target_vocabulary());
        transformer.save(model_path);
        cout << "Saved self-contained model to " << model_path << endl;

        cout << "\n================ CHAT ================" << endl;
        transformer.chat();

        return 0;
#endif

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
        const string tokenizer_kind = argc > 16 ? argv[16] : "word";
        const filesystem::path bpe_dir = argc > 17 ? argv[17]
            : "/home/artelnics/Documents/datasets/gpt2_tokenizer";
        const bool use_bpe          = tokenizer_kind == "bpe";
        const Type training_type    = (precision == "bf16") ? Type::BF16 : Type::FP32;

        cout << "OpenNN. GPT decoder-only LM, GPT-2-small architecture (" << precision
             << ") batch=" << batch_size << " max_epochs=" << maximum_epochs
             << " seq=" << sequence_length << " lr=" << learning_rate
             << " layers=" << layers_number_arg << " preln=" << pre_normalization
             << " sdpa=" << use_sdpa << " dropout=" << dropout_rate
             << " tokenizer=" << tokenizer_kind
             << "\ncorpus=" << corpus_path << endl;

        Configuration::instance().set(on_cpu ? Device::CPU : Device::CUDA, training_type);
        if (!on_cpu) Backend::instance();
        set_seed(42);

        const Index maximum_vocabulary_size = 50257;

        const filesystem::path model_path =
            corpus_path.string() + ".gpt_model_" + tokenizer_kind + ".json";

        if (!on_cpu && filesystem::exists(model_path))
        {
            cout << "Found saved model at " << model_path
                 << "\n-> loading for generation; the corpus is never read." << endl;

            TextGenerationNetwork network(model_path);

            SamplingConfig sampling;
            sampling.temperature = 0.8f;
            sampling.top_k = 40;
            sampling.maximum_tokens = 40;

            cout << "\n================ GENERATED TEXT ================" << endl;

            for (const string& prompt : {"the film received",
                                         "the city is located in",
                                         "world war ii began"})
            {
                cout << "Prompt:    " << prompt << endl;
                cout << "Generated: " << network.generate(prompt, sampling) << endl;
                cout << endl;
            }

            cout << "================ GPT CHAT ================" << endl;
            network.chat(sampling);

            return 0;
        }

        TextDataset dataset("", sequence_length, maximum_vocabulary_size);
        if (use_bpe)
        {
            auto tokenizer = make_unique<BytePairTokenizer>();
            tokenizer->load(bpe_dir / "vocab.json", bpe_dir / "merges.txt");
            dataset.set_tokenizer(move(tokenizer));
        }
        dataset.set_data_path(corpus_path);
        dataset.read_txt();
        dataset.split_samples_random(0.95f, 0.04f, 0.01f);

        cout << "[DATASET] blocks=" << dataset.get_samples_number()
             << " train="  << dataset.get_samples_number("Training")
             << " val="    << dataset.get_samples_number("Validation")
             << " vocab="  << dataset.get_vocabulary_size()
             << " seq="    << dataset.get_sequence_length() << endl;

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
            corpus_path.string() + ".gpt_parameters_" + tokenizer_kind + ".bin";

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

        if (use_bpe) network.set_tokenizer(dataset.get_tokenizer()->clone());
        else         network.set_vocabulary(dataset.get_vocabulary());

        network.save(model_path);
        cout << "Saved self-contained model to " << model_path << endl;

        if (on_cpu) return 0;

        SamplingConfig sampling;
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
            cout << "Generated: " << network.generate(prompt, sampling) << endl;
            cout << endl;
        }

        cout << "================ GPT CHAT ================" << endl;
        network.chat(sampling);

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
