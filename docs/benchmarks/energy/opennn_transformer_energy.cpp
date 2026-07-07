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

#include "../../../opennn/standard_networks.h"
#include "../../../opennn/language_dataset.h"
#include "../../../opennn/training_strategy.h"
#include "../../../opennn/adaptive_moment_estimation.h"
#include "../../../opennn/configuration.h"
#include "../../../opennn/random_utilities.h"

using namespace opennn;

static double unix_seconds()
{
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

int main(int argc, char* argv[])
{
    std::cout << std::unitbuf;

    const std::string corpus = argc > 1 ? argv[1] : "chat_pairs.txt";
    const std::string target_arg = argc > 2 ? argv[2] : "probe";

    try
    {
        const int seed = argc > 10 ? std::stoi(argv[10]) : 42;
        set_seed(seed);

        const bool probe_only = (target_arg == "probe");

        const bool use_bf16 = []() {
            const char* v = std::getenv("OPENNN_BF16");
            return v && std::string(v) != "0";
        }();
        const bool use_graph = []() {
            const char* v = std::getenv("OPENNN_GRAPH");
            return !v || std::string(v) != "0";
        }();

        if (!probe_only)
            Configuration::instance().set(Device::CUDA, use_bf16 ? Type::BF16 : Type::FP32);

        // Vocab capped at 30000 to match the ChatGPT example (blank_cuda block 6).
        LanguageDataset dataset(corpus, 30000);
        dataset.set_sample_roles("Training");   // all-train, gate is the training CE

        const Index samples      = dataset.get_samples_number("Training");
        const Index input_vocab  = dataset.get_input_vocabulary_size();
        const Index output_vocab = dataset.get_target_vocabulary_size();
        const Index input_seq    = dataset.get_shape("Input")[0];
        const Index decoder_seq  = dataset.get_shape("Decoder")[0];

        std::cout << "samples=" << samples
                  << " input_vocab=" << input_vocab
                  << " output_vocab=" << output_vocab
                  << " input_seq=" << input_seq
                  << " decoder_seq=" << decoder_seq << "\n";

        if (probe_only)
        {
            std::cout << "RESULT=OK\n";
            return 0;
        }

        const float target      = std::stof(target_arg);
        const Index batch       = argc > 3 ? Index(std::stoll(argv[3])) : 128;
        const Index max_epochs  = argc > 4 ? Index(std::stoll(argv[4])) : 40;
        const float lr          = argc > 5 ? std::stof(argv[5]) : 5.0e-4f;
        const Index d_model     = argc > 6 ? Index(std::stoll(argv[6])) : 512;
        const Index heads       = argc > 7 ? Index(std::stoll(argv[7])) : 8;
        const Index ff          = argc > 8 ? Index(std::stoll(argv[8])) : 2048;
        const Index layers      = argc > 9 ? Index(std::stoll(argv[9])) : 6;

        std::cout << "precision=" << (use_bf16 ? "bf16" : "fp32")
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

        std::cout << "parameters=" << transformer.get_parameters_size() << "\n";

        TrainingStrategy training_strategy(&transformer, &dataset);
        training_strategy.set_loss("CrossEntropyError3d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        if (!adam) throw std::runtime_error("Adam optimizer not found.");

        adam->set_batch_size(batch);
        adam->set_learning_rate(lr);
        adam->set_maximum_epochs(max_epochs);
        adam->set_maximum_validation_failures(1 << 30);
        adam->set_loss_goal(target);
        adam->set_display_period(1);
        adam->set_cuda_graph(use_graph);

        // The energy window starts here: it includes OpenNN's in-train() warmup
        // (cuDNN plan selection, allocations, graph capture) -- real electricity
        // any user pays to train this model -- and excludes only the one-time
        // corpus tokenization (cached in tokens.bin, shared by every engine).
        std::cout << "TRAIN_START_UNIX=" << std::fixed << unix_seconds() << "\n";
        const auto t0 = std::chrono::high_resolution_clock::now();

        const TrainingResult result = training_strategy.train();
        cudaDeviceSynchronize();

        const auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "TRAIN_END_UNIX=" << std::fixed << unix_seconds() << "\n";
        std::cout.unsetf(std::ios::fixed);

        const double wall_s = std::chrono::duration<double>(t1 - t0).count();
        const Index epochs = result.get_epochs_number();
        const bool reached = result.stopping_condition
            && *result.stopping_condition == StoppingCondition::LossGoal;

        std::cout << "loss_history=";
        for (Index e = 0; e < result.training_error_history.size(); ++e)
            std::cout << (e ? "," : "") << result.training_error_history(e);
        std::cout << "\n";

        std::cout << "epochs=" << epochs << "\n";
        std::cout << "final_error=" << result.get_training_error() << "\n";
        std::cout << "reached_goal=" << (reached ? 1 : 0) << "\n";
        std::cout << "wall_s=" << wall_s << "\n";
        std::cout << "samples_per_sec="
                  << double(samples) * double(epochs + 1) / wall_s << "\n";
        std::cout << "RESULT=OK\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        std::cout << "RESULT=ERROR\n";
        return 1;
    }
}
