//   OpenNN Transformer energy-to-target driver, GPU.
//
//   Trains the ChatGPT-example Transformer (encoder-decoder, paper base
//   512/8/2048/6 by default) on the chat corpus (prompt <TAB> response) until
//   the epoch-mean token cross-entropy reaches a fixed target, in OpenNN's
//   fastest configuration (bf16 tensor-core path + CUDA graph). The Python
//   orchestrator samples GPU power while this runs and integrates it between
//   the TRAIN_START_UNIX / TRAIN_END_UNIX markers printed below.
//
//   usage: opennn_transformer_energy CORPUS probe
//          opennn_transformer_energy CORPUS TARGET [batch] [max_epochs] [lr]
//                                    [d_model] [heads] [ff] [layers] [seed]
//   env:   OPENNN_BF16=1  -> bf16 (else fp32)
//          OPENNN_GRAPH=0 -> disable the CUDA graph (default on)

#include <chrono>
#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include "opennn/standard_networks.h"
#include "opennn/text_dataset.h"
#include "opennn/training_strategy.h"
#include "opennn/adaptive_moment_estimation.h"
#include "opennn/configuration.h"
#include "opennn/random_utilities.h"

using namespace opennn;

static double unix_seconds()
{
    return chrono::duration<double>(
        chrono::system_clock::now().time_since_epoch()).count();
}

int main(int argc, char* argv[])
{
    cout << unitbuf;

    const string corpus = argc > 1 ? argv[1] : "chat_pairs.txt";
    const string target_arg = argc > 2 ? argv[2] : "probe";

    try
    {
        const int seed = argc > 10 ? stoi(argv[10]) : 42;
        set_seed(seed);

        const bool probe_only = (target_arg == "probe");

        const bool use_bf16 = []() {
            const char* v = getenv("OPENNN_BF16");
            return v && string(v) != "0";
        }();
        const bool use_graph = []() {
            const char* v = getenv("OPENNN_GRAPH");
            return !v || string(v) != "0";
        }();

        if (!probe_only)
            Configuration::instance().set(Device::CUDA, use_bf16 ? Type::BF16 : Type::FP32);

        unique_ptr<TextDataset> dataset = TextDataset::from_sequence_to_sequence(corpus, 30000);
        dataset->set_sample_roles("Training");

        const Index samples      = dataset->get_samples_number("Training");
        const Index input_vocab  = dataset->get_vocabulary_size();
        const Index output_vocab = dataset->get_target_vocabulary().size();
        const Index input_seq    = dataset->get_shape("Input")[0];
        const Index decoder_seq  = dataset->get_shape("Decoder")[0];

        cout << "samples=" << samples
                  << " input_vocab=" << input_vocab
                  << " output_vocab=" << output_vocab
                  << " input_seq=" << input_seq
                  << " decoder_seq=" << decoder_seq << "\n";

        if (probe_only)
        {
            cout << "RESULT=OK\n";
            return 0;
        }

        const float target      = stof(target_arg);
        const Index batch       = argc > 3 ? Index(stoll(argv[3])) : 128;
        const Index max_epochs  = argc > 4 ? Index(stoll(argv[4])) : 40;
        const float lr          = argc > 5 ? stof(argv[5]) : 5.0e-4f;
        const Index d_model     = argc > 6 ? Index(stoll(argv[6])) : 512;
        const Index heads       = argc > 7 ? Index(stoll(argv[7])) : 8;
        const Index ff          = argc > 8 ? Index(stoll(argv[8])) : 2048;
        const Index layers      = argc > 9 ? Index(stoll(argv[9])) : 6;

        cout << "precision=" << (use_bf16 ? "bf16" : "fp32")
                  << " cuda_graph=" << use_graph
                  << " target=" << target
                  << " batch=" << batch
                  << " max_epochs=" << max_epochs
                  << " lr=" << lr
                  << " d_model=" << d_model << " heads=" << heads
                  << " ff=" << ff << " layers=" << layers
                  << " seed=" << seed << "\n";

        Transformer transformer(input_seq, decoder_seq,
                                input_vocab, output_vocab,
                                d_model, heads, ff, layers);
        transformer.set_dropout_rate(0.0f);

        cout << "parameters=" << transformer.get_parameters_size() << "\n";

        TrainingStrategy training_strategy(&transformer, dataset.get());
        training_strategy.set_loss("CrossEntropyError3d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        if (!adam) throw runtime_error("Adam optimizer not found.");

        adam->set_batch_size(batch);
        adam->set_learning_rate(lr);
        adam->set_maximum_epochs(max_epochs);
        adam->set_maximum_validation_failures(1 << 30);
        adam->set_loss_goal(target);
        adam->set_display_period(1);
        adam->set_cuda_graph(use_graph);

        cout << "TRAIN_START_UNIX=" << fixed << unix_seconds() << "\n";
        const auto t0 = chrono::high_resolution_clock::now();

        const TrainingResult result = training_strategy.train();
        cudaDeviceSynchronize();

        const auto t1 = chrono::high_resolution_clock::now();
        cout << "TRAIN_END_UNIX=" << fixed << unix_seconds() << "\n";
        cout.unsetf(ios::fixed);

        const double wall_s = chrono::duration<double>(t1 - t0).count();
        const Index epochs = result.get_epochs_number();
        const bool reached = result.stopping_condition
            && *result.stopping_condition == StoppingCondition::LossGoal;

        cout << "loss_history=";
        for (Index e = 0; e < result.training_error_history.size(); ++e)
            cout << (e ? "," : "") << result.training_error_history(e);
        cout << "\n";

        cout << "epochs=" << epochs << "\n";
        cout << "final_error=" << result.get_training_error() << "\n";
        cout << "reached_goal=" << (reached ? 1 : 0) << "\n";
        cout << "wall_s=" << wall_s << "\n";
        cout << "samples_per_sec="
                  << double(samples) * double(epochs + 1) / wall_s << "\n";
        cout << "RESULT=OK\n";
        return 0;
    }
    catch (const exception& e)
    {
        cout << "FAIL: " << e.what() << "\n";
        cout << "RESULT=ERROR\n";
        return 1;
    }
}
