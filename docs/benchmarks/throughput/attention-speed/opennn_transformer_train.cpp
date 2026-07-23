//   OpenNN Transformer TRAINING throughput, GPU. Fair counterpart to PyTorch's
//   training loop: same architecture, same synthetic corpus (token-for-token),
//   same optimizer (Adam) and loss (token cross-entropy over the vocabulary).
//
//   We run one untimed warmup train() pass first (CUDA context, cuBLASLt /
//   cuDNN plan caches, allocator, graph capture -- the counterpart of the
//   excluded warmup epoch in the PyTorch/TensorFlow scripts), then time
//   train() over the requested epoch count and report samples/sec: the
//   steady-state forward+backward+update throughput of the resident GPU path.
//
//   The corpus is built by make_synthetic_corpus.py (tab-separated input<TAB>
//   target per line); TextDataset.read_txt derives vocab + sequence lengths.
//
//   usage: opennn_transformer_train CORPUS.txt [d_model] [heads] [ff] [layers] [batch] [epochs]
//   env:   OPENNN_BF16=1   -> train in bf16 (else fp32, via the fp32-via-bf16 SDPA path)
//          OPENNN_LR=<f>   -> Adam learning rate (default 1e-4)
//          OPENNN_SDPA_MIN -> lower the fused-attention sequence-length threshold
//          OPENNN_BENCH_DISPLAY=1 -> per-epoch optimizer display (timing diagnosis)

#include <algorithm>
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

int main(int argc, char* argv[])
{
    cout << unitbuf;

    const string corpus = argc > 1 ? argv[1] : "synthetic_corpus.txt";
    const Index d_model = argc > 2 ? Index(stoll(argv[2])) : 256;
    const Index heads   = argc > 3 ? Index(stoll(argv[3])) : 8;
    const Index ff      = argc > 4 ? Index(stoll(argv[4])) : 1024;
    const Index layers  = argc > 5 ? Index(stoll(argv[5])) : 2;
    const Index batch   = argc > 6 ? Index(stoll(argv[6])) : 32;
    const Index epochs  = argc > 7 ? Index(stoll(argv[7])) : 30;

    try
    {
        set_seed(0);
        const bool use_bf16 = getenv("OPENNN_BF16") != nullptr;
        Configuration::instance().set(Device::CUDA, use_bf16 ? Type::BF16 : Type::FP32);

        unique_ptr<TextDataset> dataset = TextDataset::from_sequence_to_sequence(corpus, 30000);
        dataset->split_samples(1.0f, 0.0f, 0.0f);

        const Index samples = dataset->get_samples_number("Training");
        const Index input_vocab  = dataset->get_vocabulary_size();
        const Index output_vocab = dataset->get_target_vocabulary().size();
        const Index input_seq    = dataset->get_shape("Input")[0];
        const Index decoder_seq  = dataset->get_shape("Decoder")[0];

        cout << "precision=" << (use_bf16 ? "bf16" : "fp32")
                  << " samples=" << samples
                  << " input_seq=" << input_seq << " decoder_seq=" << decoder_seq
                  << " input_vocab=" << input_vocab << " output_vocab=" << output_vocab
                  << " d_model=" << d_model << " heads=" << heads
                  << " ff=" << ff << " layers=" << layers
                  << " batch=" << batch << " epochs=" << epochs << "\n";

        Transformer transformer(input_seq, decoder_seq,
                                input_vocab, output_vocab,
                                d_model, heads, ff, layers);

        if (const char* e = getenv("OPENNN_SDPA_MIN"))
            transformer.set_attention_sdpa_min_sequence_length(Index(stoll(e)));

        cout << "parameters=" << transformer.get_parameters_size() << "\n";

        TrainingStrategy training_strategy(&transformer, dataset.get());
        training_strategy.set_loss("CrossEntropyError3d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        if (!adam) throw runtime_error("Adam optimizer not found.");

        adam->set_batch_size(batch);
        const float lr = getenv("OPENNN_LR") ? stof(getenv("OPENNN_LR")) : 0.0001f;
        adam->set_learning_rate(lr);
        adam->set_display(getenv("OPENNN_BENCH_DISPLAY") != nullptr);
        adam->set_display_period(1);
        cout << "learning_rate=" << lr << "\n";

        adam->set_maximum_epochs(0);
        training_strategy.train();
        cudaDeviceSynchronize();

        adam->set_maximum_epochs(epochs);

        const auto t0 = chrono::high_resolution_clock::now();
        const TrainingResult result = training_strategy.train();
        cudaDeviceSynchronize();
        const auto t1 = chrono::high_resolution_clock::now();

        const double wall_s = chrono::duration<double>(t1 - t0).count();
        const double timed_passes = double(max<Index>(Index(1), epochs));
        const double total_samples = double(samples) * timed_passes;
        const double samples_per_s = total_samples / wall_s;
        const double tokens_per_s  = samples_per_s * double(input_seq + decoder_seq);

        cout << "final_loss=" << result.loss << "\n";
        cout << "wall_s=" << wall_s << "\n";
        cout << "samples_per_sec=" << samples_per_s << "\n";
        cout << "tokens_per_sec=" << tokens_per_s << "\n";

        return 0;
    }
    catch (const exception& e)
    {
        cout << "FAIL: " << e.what() << "\n";
        return 1;
    }
}
