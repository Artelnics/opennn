//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   C U D A   —   Translation benchmark (refactor)
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <iomanip>
#include <chrono>

#include "../opennn/language_dataset.h"
#include "../opennn/standard_networks.h"
#include "../opennn/training_strategy.h"
#include "../opennn/adaptive_moment_estimation.h"
#include "../opennn/random_utilities.h"
#include "../opennn/tensor_utilities.h"
#include "../opennn/kernel.cuh"

using namespace opennn;
using namespace std::chrono;

#ifdef OPENNN_WITH_CUDA

static void benchmark_mask_softmax(int B, int H, int Sq, int Sk, int E,
                                   bool use_causal_mask,
                                   int iterations, int warmup)
{
    const Index n_logits = static_cast<Index>(B) * H * Sq * Sk;
    const Index n_pad    = static_cast<Index>(B) * Sk;
    const Index n_input  = static_cast<Index>(B) * Sk * E;

    Memory logits_ref, logits_run, source_mem, pad_mem;
    logits_ref.resize_device(n_logits);
    logits_run.resize_device(n_logits);
    source_mem.resize_device(n_input);
    pad_mem.resize_device(n_pad);

    set_seed(42);

    VectorR logits_h(n_logits);
    set_random_uniform(logits_h, type(-2), type(2));
    CHECK_CUDA(cudaMemcpy(logits_ref.device(), logits_h.data(), n_logits * sizeof(float), cudaMemcpyHostToDevice));

    VectorR source_h = VectorR::Zero(n_input);
    {
        Index pos = 0;
        for(Index b = 0; b < B; ++b)
            for(Index s = 0; s < Sk; ++s, ++pos)
                if(s < (Sk * 7) / 10)
                    for(Index e = 0; e < E; ++e)
                        source_h[pos * E + e] = type(0.5);
    }
    CHECK_CUDA(cudaMemcpy(source_mem.device(), source_h.data(), n_input * sizeof(float), cudaMemcpyHostToDevice));

    padding_mask_compute_cuda<float>(B, Sk, E, source_mem.device(), pad_mem.device());
    CHECK_CUDA(cudaDeviceSynchronize());

    auto reset = [&](float* dst) {
        CHECK_CUDA(cudaMemcpy(dst, logits_ref.device(), n_logits * sizeof(float), cudaMemcpyDeviceToDevice));
    };
    auto fused = [&](float* logits) {
        mask_softmax_fused_cuda<float>(B, H, Sq, Sk, logits, pad_mem.device(), use_causal_mask);
    };

    for(int i = 0; i < warmup; ++i) { reset(logits_run.device()); fused(logits_run.device()); }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    cudaEventRecord(e0);
    for(int i = 0; i < iterations; ++i) { reset(logits_run.device()); fused(logits_run.device()); }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);

    const float us = (ms / iterations) * 1000.0f;

    cout << "  B="  << setw(3) << B
         << " H="   << setw(2) << H
         << " Sq="  << setw(3) << Sq
         << " Sk="  << setw(3) << Sk
         << " causal=" << (use_causal_mask ? "Y" : "N")
         << " | fused: " << fixed << setprecision(2) << setw(6) << us << " us/iter"
         << "\n";

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
}

#endif

int main()
{
    try
    {
        cout << "OpenNN. Translation benchmark (refactor)." << endl;

#ifdef OPENNN_WITH_CUDA

        Device::instance().set(DeviceType::Gpu);

        cout << "[BENCH mask+softmax fused — multi-warp]  iterations=2000 warmup=20  seed=42\n";
        benchmark_mask_softmax(/*B*/ 64, /*H*/ 4, /*Sq*/  8, /*Sk*/  8, /*E*/ 128, /*causal*/ false, 2000, 20);
        benchmark_mask_softmax(/*B*/ 64, /*H*/ 4, /*Sq*/ 16, /*Sk*/ 16, /*E*/ 128, /*causal*/ false, 2000, 20);
        benchmark_mask_softmax(/*B*/ 64, /*H*/ 4, /*Sq*/ 16, /*Sk*/ 16, /*E*/ 128, /*causal*/ true,  2000, 20);
        benchmark_mask_softmax(/*B*/ 64, /*H*/ 4, /*Sq*/ 24, /*Sk*/ 24, /*E*/ 128, /*causal*/ false, 2000, 20);
        benchmark_mask_softmax(/*B*/ 64, /*H*/ 4, /*Sq*/ 32, /*Sk*/ 32, /*E*/ 128, /*causal*/ false, 2000, 20);
        benchmark_mask_softmax(/*B*/ 64, /*H*/ 4, /*Sq*/ 32, /*Sk*/ 32, /*E*/ 128, /*causal*/ true,  2000, 20);
        cout << "[/BENCH]\n\n";

        set_seed(42);

        LanguageDataset dataset("/home/artelnics/Documents/openNN/opennn/temp/translation_en_es.txt");
        dataset.split_samples_random(0.8, 0.1, 0.1);

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

        transformer.set_input_vocabulary(dataset.get_input_vocabulary());
        transformer.set_output_vocabulary(dataset.get_target_vocabulary());
        transformer.set_dropout_rate(type(0));

        TrainingStrategy training_strategy(&transformer, &dataset);

        training_strategy.set_loss("CrossEntropyError3d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        if(!adam) throw runtime_error("AdaptiveMomentEstimation optimizer not found.");

        adam->set_batch_size(128);
        adam->set_learning_rate(type(5e-4));
        adam->set_maximum_epochs(1);
        adam->set_display_period(1);

        cout << "[PARITY] train="  << dataset.get_samples_number("Training")
             << " val="            << dataset.get_samples_number("Validation")
             << " test="           << dataset.get_samples_number("Testing")
             << " input_vocab="    << input_vocab
             << " target_vocab="   << output_vocab
             << " input_seq="      << input_seq
             << " decoder_seq="    << decoder_seq
             << " params="         << transformer.get_parameters_number()
             << " (buffer="        << transformer.get_parameters_size() << ")" << endl;

        const auto t0 = steady_clock::now();
        training_strategy.train();
        const auto t1 = steady_clock::now();

        const double training_seconds = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;
        cout << "\nTotal training time (refactor): " << training_seconds << " s" << endl;

        cout << "\n================ TRANSFORMER PREDICTIONS ================\n";

        const vector<string> test_sources =
        {
            "the cat eats an apple",
            "a boy sings happily",
            "my friend writes a book",
            "the teacher speaks loudly"
        };

        for(Index i = 0; i < Index(test_sources.size()); ++i)
        {
            const string prediction = transformer.calculate_outputs(test_sources[i]);

            cout << "Sample " << i << endl;
            cout << "  Source:    " << test_sources[i] << endl;
            cout << "  Predicted: " << prediction << endl;
            cout << endl;
        }

        cout << "=========================================================\n";

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
