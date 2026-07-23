//   OpenNN Transformer max-batch / training-speed trial, GPU.
//
//   One process = one (batch, precision) attempt at the encoder-decoder
//   Transformer from *Attention Is All You Need* (paper base 512/8/2048/6),
//   trained sequence-to-sequence on the chat corpus (prompt <TAB> response).
//   CUDA graph is OFF (it does not help the transformer path -- big GEMMs, the
//   launch overhead a graph amortizes is already negligible).
//
//   The Python driver binary-searches batch (capacity) and, at a fixed batch,
//   reads samples_per_sec (throughput). To bound the work per trial, only the
//   first `train_samples` samples are used for training (the peak allocation is
//   already reached on the first forward+backward+Adam step), while the vocab
//   and sequence lengths come from the FULL corpus so every batch builds the
//   identical model.
//
//   mode "infer" measures forward-only capacity/throughput instead: no
//   optimizer state, no gradients, device-resident inputs and output
//   (calculate_outputs_resident), so the footprint is parameters + activations
//   only. Token values are random -- memory depends on the shapes alone. In
//   this mode `epochs` is reused as the number of timed forward iterations.
//
//   usage: opennn_transformer_maxbatch_trial CORPUS [d_model] [heads] [ff]
//                                            [layers] [batch] [train_samples]
//                                            [epochs] [train|infer]
//   env:   OPENNN_BF16=1  -> bf16 (else fp32)
//          OPENNN_TRANSFORMER_RELEASE_BF16_FP32_MASTER=0
//              keep the fp32 parameter master in BF16 inference (debug A/B)
//          OPENNN_TRANSFORMER_INFER_CUDA_GRAPH=1
//              capture/replay the resident inference forward pass

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "opennn/standard_networks.h"
#include "opennn/text_dataset.h"
#include "opennn/training_strategy.h"
#include "opennn/adaptive_moment_estimation.h"
#include "opennn/configuration.h"
#include "opennn/device_backend.h"
#include "opennn/forward_propagation.h"
#include "opennn/random_utilities.h"
#include "opennn/memory_debug.h"

using namespace opennn;

int main(int argc, char* argv[])
{
    cout << unitbuf;

    const string corpus = argc > 1 ? argv[1] : "chat_pairs.txt";
    const Index d_model       = argc > 2 ? Index(stoll(argv[2])) : 512;
    const Index heads         = argc > 3 ? Index(stoll(argv[3])) : 8;
    const Index ff            = argc > 4 ? Index(stoll(argv[4])) : 2048;
    const Index layers        = argc > 5 ? Index(stoll(argv[5])) : 6;
    const Index batch         = argc > 6 ? Index(stoll(argv[6])) : 32;
    const Index train_samples = argc > 7 ? Index(stoll(argv[7])) : 4096;
    const Index epochs        = argc > 8 ? Index(stoll(argv[8])) : 1;
    const string mode    = argc > 9 ? argv[9] : "train";

    try
    {
        set_seed(0);
        const bool use_bf16 = getenv("OPENNN_BF16") != nullptr;
        Configuration::instance().set(Device::CUDA, use_bf16 ? Type::BF16 : Type::FP32);

        unique_ptr<TextDataset> dataset = TextDataset::from_sequence_to_sequence(corpus, 30000);

        const Index total = dataset->get_samples_number();
        const float train_fraction = min(1.0f, float(train_samples) / float(total));
        dataset->split_samples(train_fraction, 0.0f, 1.0f - train_fraction);

        const Index samples      = dataset->get_samples_number("Training");
        const Index input_vocab  = dataset->get_vocabulary_size();
        const Index output_vocab = dataset->get_target_vocabulary().size();
        const Index input_seq    = dataset->get_shape("Input")[0];
        const Index decoder_seq  = dataset->get_shape("Decoder")[0];

        cout << "precision=" << (use_bf16 ? "bf16" : "fp32")
                  << " mode=" << mode
                  << " samples=" << samples
                  << " input_seq=" << input_seq << " decoder_seq=" << decoder_seq
                  << " input_vocab=" << input_vocab << " output_vocab=" << output_vocab
                  << " d_model=" << d_model << " heads=" << heads
                  << " ff=" << ff << " layers=" << layers
                  << " batch=" << batch << " epochs=" << epochs << "\n";

        Transformer transformer(input_seq, decoder_seq,
                                input_vocab, output_vocab,
                                d_model, heads, ff, layers);

        cout << "parameters=" << transformer.get_parameters_size() << "\n";

        if (mode == "infer")
        {
            const Index iterations = max<Index>(Index(1), epochs);

            mt19937 generator(0);
            uniform_int_distribution<int> input_token(0, int(input_vocab) - 1);
            uniform_int_distribution<int> output_token(0, int(output_vocab) - 1);

            MatrixR decoder_ids(batch, decoder_seq);
            MatrixR input_ids(batch, input_seq);
            for (Index i = 0; i < decoder_ids.size(); ++i)
                decoder_ids.data()[i] = float(output_token(generator));
            for (Index i = 0; i < input_ids.size(); ++i)
                input_ids.data()[i] = float(input_token(generator));

            const Index decoder_bytes = get_aligned_bytes(batch * decoder_seq, Type::FP32);
            const Index input_bytes = get_aligned_bytes(batch * input_seq, Type::FP32);

            Buffer arena(Device::CUDA);
            arena.resize_bytes(decoder_bytes + input_bytes, Device::CUDA);
            char* const base = arena.as<char>();

            TensorView decoder_view(base, {batch, decoder_seq}, Type::FP32, Device::CUDA);
            TensorView input_view(base + decoder_bytes, {batch, input_seq}, Type::FP32, Device::CUDA);

            cudaStream_t stream = Backend::get_compute_stream();
            device::copy_async(decoder_view.data, decoder_ids.data(), decoder_view.byte_size(),
                               device::CopyKind::HostToDevice, stream);
            device::copy_async(input_view.data, input_ids.data(), input_view.byte_size(),
                               device::CopyKind::HostToDevice, stream);

            const vector<TensorView> inputs = {decoder_view, input_view};

            bool parameters_uploaded = false;
            if (use_bf16)
            {
                const char* release_flag = getenv("OPENNN_TRANSFORMER_RELEASE_BF16_FP32_MASTER");
                const bool release_fp32_master = !release_flag || string(release_flag) != "0";
                if (release_fp32_master)
                {
                    transformer.copy_parameters_device();
                    transformer.copy_states_device();
                    transformer.release_bf16_fp32_parameter_master_for_inference();
                    parameters_uploaded = true;
                    cout << "bf16_fp32_master_released=1\n";
                }
            }

            ForwardPropagation forward_propagation(batch, &transformer);

            transformer.calculate_outputs_resident(inputs, forward_propagation, !parameters_uploaded);
            cudaDeviceSynchronize();

            const char* graph_flag = getenv("OPENNN_TRANSFORMER_INFER_CUDA_GRAPH");
            const bool try_cuda_graph =
                graph_flag != nullptr && string(graph_flag) != "0";
            device::GraphExecHandle infer_graph;
            if (try_cuda_graph)
            {
                try
                {
                    device::synchronize(stream);
                    device::StreamCapture capture(stream);
                    transformer.calculate_outputs_resident(inputs, forward_propagation, false);
                    const device::GraphHandle graph = capture.end();
                    device::instantiate_or_update(infer_graph, graph.get());
                    cudaDeviceSynchronize();
                    cout << "cuda_graph=1\n";
                }
                catch (const exception& capture_error)
                {
                    cout << "cuda_graph=0 reason=\"" << capture_error.what() << "\"\n";
                    infer_graph.reset();
                }
            }

            const auto t0 = chrono::high_resolution_clock::now();
            for (Index i = 0; i < iterations; ++i)
            {
                if (infer_graph)
                    device::launch_graph(infer_graph, stream);
                else
                    transformer.calculate_outputs_resident(inputs, forward_propagation, false);
            }
            cudaDeviceSynchronize();
            const auto t1 = chrono::high_resolution_clock::now();

            const TensorView output_view = forward_propagation.get_outputs();
            float probe[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const Index probe_size = min<Index>(Index(4), output_view.size());
            copy_device_to_host_float(output_view.data, output_view.type, probe_size, probe, stream);
            cudaStreamSynchronize(stream);
            for (Index i = 0; i < probe_size; ++i)
                if (!isfinite(probe[i]))
                    throw runtime_error("non-finite logits");

            const double wall_s = chrono::duration<double>(t1 - t0).count();
            const double samples_per_s = double(batch) * double(iterations) / wall_s;

            memory_debug::print(cout);

            cout << "wall_s=" << wall_s << "\n";
            cout << "samples_per_sec=" << samples_per_s << "\n";
            cout << "tokens_per_sec=" << samples_per_s * double(input_seq + decoder_seq) << "\n";
            cout << "RESULT=OK\n";
            return 0;
        }

        TrainingStrategy training_strategy(&transformer, dataset.get());
        training_strategy.set_loss("CrossEntropyError3d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(
            training_strategy.get_optimization_algorithm());
        if (!adam) throw runtime_error("Adam optimizer not found.");

        adam->set_batch_size(batch);
        adam->set_learning_rate(0.0001f);
        adam->set_maximum_validation_failures(1 << 30);
        adam->set_display(false);
        adam->set_cuda_graph(false);

        adam->set_maximum_epochs(0);
        training_strategy.train();
        cudaDeviceSynchronize();

        adam->set_maximum_epochs(epochs);
        const auto t0 = chrono::high_resolution_clock::now();
        const TrainingResult result = training_strategy.train();
        cudaDeviceSynchronize();
        const auto t1 = chrono::high_resolution_clock::now();

        const double wall_s = chrono::duration<double>(t1 - t0).count();
        const double total_samples = double(samples) * double(epochs + 1);
        const double samples_per_s = total_samples / wall_s;

        memory_debug::print(cout);

        cout << "final_loss=" << result.loss << "\n";
        cout << "wall_s=" << wall_s << "\n";
        cout << "samples_per_sec=" << samples_per_s << "\n";
        cout << "tokens_per_sec=" << samples_per_s * double(input_seq + decoder_seq) << "\n";
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
