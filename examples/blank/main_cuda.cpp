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
#include "opennn/chat.h"
#include "opennn/standard_networks.h"

#include "opennn/tabular_dataset.h"
#include "opennn/time_series_dataset.h"
#include "opennn/image_dataset.h"
#include "opennn/language_dataset.h"
#include "opennn/text_generation_dataset.h"

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
using namespace std::chrono;

int main(int argc, char** argv)
{
    try
    {












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
            ChatSession session(transformer);

            cout << "\n================ EN -> DE CHAT ================" << endl;
            session.chat();

            return 0;
        }

        LanguageDataset language_dataset(dataset_path, 37000);

        const Index input_vocabulary_size  = language_dataset.get_input_vocabulary_size();
        const Index output_vocabulary_size = language_dataset.get_target_vocabulary_size();
        const Index input_sequence_length   = language_dataset.get_shape("Input")[0];
        const Index decoder_sequence_length = language_dataset.get_shape("Decoder")[0];
        const Index target_sequence_length  = language_dataset.get_shape("Target")[0];

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



        transformer.set_input_vocabulary(language_dataset.get_input_vocabulary());
        transformer.set_target_vocabulary(language_dataset.get_target_vocabulary());
        transformer.save(model_path);
        cout << "Saved self-contained model to " << model_path << endl;



        cout << "\n================ EN -> DE CHAT ================" << endl;
        ChatSession session(transformer);
        session.chat();

        return 0;
#endif

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
