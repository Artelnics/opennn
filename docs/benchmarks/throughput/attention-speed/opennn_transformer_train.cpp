//   OpenNN Transformer TRAINING throughput, GPU. Fair counterpart to PyTorch's
//   training loop: same architecture, same synthetic corpus (token-for-token),
//   same optimizer (Adam) and loss (token cross-entropy over the vocabulary).
//
//   We time train() over a fixed epoch count and report samples/sec. train()
//   does one CUDA warmup epoch internally before its timed loop, and uses CUDA
//   graph capture for the optimizer step, so this measures the steady-state
//   forward+backward+update throughput of the resident GPU path.
//
//   The corpus is built by make_synthetic_corpus.py (tab-separated input<TAB>
//   target per line); LanguageDataset.read_txt derives vocab + sequence lengths.
//
//   usage: opennn_transformer_train CORPUS.txt [d_model] [heads] [ff] [layers] [batch] [epochs]
//   env:   OPENNN_BF16=1   -> train in bf16 (else fp32, via the fp32-via-bf16 SDPA path)
//          OPENNN_LR=<f>   -> Adam learning rate (default 1e-4)
//          OPENNN_SDPA_MIN -> lower the fused-attention sequence-length threshold

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

int main(int argc, char* argv[])
{
    std::cout << std::unitbuf;

    const std::string corpus = argc > 1 ? argv[1] : "synthetic_corpus.txt";
    const Index d_model = argc > 2 ? Index(std::stoll(argv[2])) : 256;
    const Index heads   = argc > 3 ? Index(std::stoll(argv[3])) : 8;
    const Index ff      = argc > 4 ? Index(std::stoll(argv[4])) : 1024;
    const Index layers  = argc > 5 ? Index(std::stoll(argv[5])) : 2;
    const Index batch   = argc > 6 ? Index(std::stoll(argv[6])) : 32;
    const Index epochs  = argc > 7 ? Index(std::stoll(argv[7])) : 30;

    try
    {
        set_seed(0);
        const bool use_bf16 = std::getenv("OPENNN_BF16") != nullptr;
        Configuration::instance().set(Device::CUDA, use_bf16 ? Type::BF16 : Type::FP32);

        LanguageDataset dataset(corpus);
        dataset.split_samples(1.0f, 0.0f, 0.0f);   // all samples Training

        const Index samples = dataset.get_samples_number("Training");
        const Index input_vocab  = dataset.get_input_vocabulary_size();
        const Index output_vocab = dataset.get_target_vocabulary_size();
        const Index input_seq    = dataset.get_shape("Input")[0];
        const Index decoder_seq  = dataset.get_shape("Decoder")[0];

        std::cout << "precision=" << (use_bf16 ? "bf16" : "fp32")
                  << " samples=" << samples
                  << " input_seq=" << input_seq << " decoder_seq=" << decoder_seq
                  << " input_vocab=" << input_vocab << " output_vocab=" << output_vocab
                  << " d_model=" << d_model << " heads=" << heads
                  << " ff=" << ff << " layers=" << layers
                  << " batch=" << batch << " epochs=" << epochs << "\n";

        Transformer transformer(input_seq, decoder_seq,
                                input_vocab, output_vocab,
                                d_model, heads, ff, layers);

        if (const char* e = std::getenv("OPENNN_SDPA_MIN"))
            transformer.set_attention_sdpa_min_sequence_length(Index(std::stoll(e)));

        std::cout << "parameters=" << transformer.get_parameters_size() << "\n";

        TrainingStrategy training_strategy(&transformer, &dataset);
        training_strategy.set_loss("CrossEntropyError3d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        if (!adam) throw std::runtime_error("Adam optimizer not found.");

        adam->set_batch_size(batch);
        const float lr = std::getenv("OPENNN_LR") ? std::stof(std::getenv("OPENNN_LR")) : 0.0001f;
        adam->set_learning_rate(lr);
        adam->set_maximum_epochs(epochs);
        adam->set_display(false);
        std::cout << "learning_rate=" << lr << "\n";

        const auto t0 = std::chrono::high_resolution_clock::now();
        const TrainingResult result = training_strategy.train();
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::high_resolution_clock::now();

        const double wall_s = std::chrono::duration<double>(t1 - t0).count();
        // epochs+1 timed passes over the data (train() runs epoch 0..maximum_epochs).
        const double total_samples = double(samples) * double(epochs + 1);
        const double samples_per_s = total_samples / wall_s;
        const double tokens_per_s  = samples_per_s * double(input_seq + decoder_seq);

        std::cout << "final_loss=" << result.loss << "\n";
        std::cout << "wall_s=" << wall_s << "\n";
        std::cout << "samples_per_sec=" << samples_per_s << "\n";
        std::cout << "tokens_per_sec=" << tokens_per_s << "\n";

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cout << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
