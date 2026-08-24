// OpenNN GPU HIGGS dense inference benchmark: one main function, written the
// way a user writes a device-resident OpenNN application, plus a timer. The
// measurement protocol lives in run_higgs_infer.py and run_higgs_infer_sweep.py.
//
//   opennn_higgs_infer <test_csv> [batch[,batch...]] [runs] [fp32|bf16] [hidden] [hidden_layers] [activation]
//                      [resident|default|reuse|construct]
//
// Forward-only throughput of the canonical HIGGS dense classifier
// (28 -> hidden -> hidden -> 1, ReLU hidden, sigmoid output). The test split
// goes to the device once, in the network's compute type, and each batch is
// staged into a fixed buffer with one device-to-device copy before the captured
// CUDA graph replays: a warmup pass that captures the graph, then `runs` timed
// passes, their times printed in temporal order before the median. The batch
// sizes run inside one process so they share one load and one thermal window;
// each prints its own batch_<B>_... lines for the sweep runner.
//
// The last argument selects the API under measurement. `resident` is the
// published path, calculate_outputs_resident: the caller owns the
// ForwardPropagation and the results stay on the device. `default` is what
// ordinary code calls instead, calculate_outputs with host inputs and host
// outputs; `reuse` is the same call with a caller-provided output buffer; and
// `construct` only builds a ForwardPropagation per batch, which isolates the
// arena planning and allocation the default path pays on every call.
//
// Precision is fp32 or bf16, matching the autocast / mixed_bfloat16 cells of
// the PyTorch and TensorFlow drivers. The resident rows are held in the
// compute type, as PyTorch's driver does with PT_BF16_WEIGHTS; holding them as
// fp32 instead costs a cast kernel inside the graph on every batch and doubles
// the bytes of the staging copy. OPENNN_INFER_STAGE=fp32 restores that for the A/B.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#ifdef OPENNN_HAS_CUDA
#include <cuda_runtime.h>
#include "opennn/core/device_backend.h"
#endif

#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/tensor_types.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/neural_network.h"

using namespace opennn;
using clock_type = chrono::steady_clock;

int main(int argc, char* argv[])
{
    cout << unitbuf;

    try
    {
        if (argc < 2)
        {
            cerr << "usage: opennn_higgs_infer <test_csv> [batch[,batch...]] [runs] [fp32|bf16] [hidden] [hidden_layers]"
                    " [activation] [resident|default|reuse|construct]\n";
            return 2;
        }

        const string test_path = argv[1];
        const Index runs = argc > 3 ? max<Index>(Index(1), Index(stoll(argv[3]))) : 5;
        const string precision = argc > 4 ? argv[4] : "fp32";
        const Index hidden = argc > 5 ? Index(stoll(argv[5])) : 1024;
        const Index hidden_layers = argc > 6 ? Index(stoll(argv[6])) : 2;
        const string activation = argc > 7 ? argv[7] : "relu";
        const string api_path = argc > 8 ? argv[8] : "resident";

        vector<Index> batch_list;
        stringstream batch_text(argc > 2 ? argv[2] : "8192");
        for (string item; getline(batch_text, item, ',');)
            if (!item.empty()) batch_list.push_back(Index(stoll(item)));
        if (batch_list.empty()) batch_list.push_back(8192);

        set_seed(42);
        const Type inference_type = (precision == "bf16") ? Type::BF16 : Type::FP32;
        Configuration::instance().set(Device::CUDA, inference_type);

        // Parsing the 500k-row split is seconds of full-machine work right
        // before a GPU measurement, and a sweep pays it once per engine per
        // round, so the parsed floats are cached beside the CSV (<csv>.bin),
        // the way the Python drivers cache theirs as .npy, and re-read while
        // the cache is newer than the CSV. The cache holds exactly what the
        // parser produced, so nothing timed changes.
        Index samples = 0;
        Index columns = 0;
        vector<float> values;

        const char cache_magic[8] = {'O', 'N', 'N', 'T', 'B', 'L', '0', '1'};
        const filesystem::path cache_path(test_path + ".bin");
        error_code cache_error;
        const auto cache_time = filesystem::last_write_time(cache_path, cache_error);
        bool cached = false;

        if (!cache_error && cache_time >= filesystem::last_write_time(filesystem::path(test_path)))
        {
            ifstream file(cache_path, ios::binary);
            char magic[sizeof(cache_magic)] = {};
            int64_t rows = 0;
            int64_t cols = 0;

            file.read(magic, sizeof(magic));
            file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

            if (file && memcmp(magic, cache_magic, sizeof(magic)) == 0 && rows > 0 && cols > 0)
            {
                samples = Index(rows);
                columns = Index(cols);
                values.resize(size_t(rows) * size_t(cols));
                file.read(reinterpret_cast<char*>(values.data()),
                          streamsize(values.size() * sizeof(float)));
                cached = bool(file);
            }
        }

        if (!cached)
        {
            TabularDataset dataset(test_path, ",", false, false);
            const MatrixR& data = dataset.get_data();

            samples = data.rows();
            columns = data.cols();
            values.assign(data.data(), data.data() + data.size());

            ofstream file(cache_path, ios::binary);
            if (file)                           // read-only data directory: parse next time
            {
                const int64_t rows = int64_t(samples);
                const int64_t cols = int64_t(columns);
                file.write(cache_magic, sizeof(cache_magic));
                file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
                file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
                file.write(reinterpret_cast<const char*>(values.data()),
                           streamsize(values.size() * sizeof(float)));
            }
        }

        const Index inputs_number = columns - 1;         // the label column is ignored
        const MatrixR inputs = Eigen::Map<const MatrixR>(values.data(), samples, columns).leftCols(inputs_number);

        const char* const stage_request = getenv("OPENNN_INFER_STAGE");
        const Type staged_type = (stage_request && string(stage_request) == "fp32") ? Type::FP32 : inference_type;
        const Index staged_bytes = type_bytes(staged_type);

        cout << "engine=opennn\n";
        cout << "mode=infer\n";
        cout << "device=cuda\n";
        cout << "runs=" << runs << "\n";
        cout << "hidden=" << hidden << "\n";
        cout << "hidden_layers=" << hidden_layers << "\n";
        cout << "activation=" << activation << "\n";
        cout << "precision=" << precision << "\n";
        cout << "api_path=" << api_path << "\n";
        cout << "staged_type=" << (staged_type == Type::BF16 ? "bf16" : "fp32") << "\n";

        NeuralNetwork network;
        const string hidden_activation = (activation == "relu" || activation == "ReLU") ? "ReLU" : "Tanh";
        Shape current{inputs_number};

        for (Index i = 0; i < hidden_layers; ++i)
        {
            network.add_layer(make_unique<opennn::Dense>(current, Shape{hidden}, hidden_activation, false,
                                                         "higgs_dense_" + to_string(i + 1)));
            current = network.get_output_shape();
        }

        network.add_layer(make_unique<opennn::Dense>(current, Shape{1}, "Sigmoid", false, "higgs_output"));
        network.compile();
        network.set_parameters_glorot();
        cout << "parameters=" << network.get_parameters_number() << "\n";

        // The whole split goes to the device once; every batch size in the
        // sweep reads the same rows from it. Converting here rather than on the
        // device is deliberate: a cast on the device would have to live either
        // in the graph, where it is a node per batch, or in a pass of its own
        // over the whole split.
        vector<uint16_t> staged_host;
        const void* staged_source = inputs.data();

        if (staged_type == Type::BF16)
        {
            staged_host.resize(size_t(samples * inputs_number));
            float_2_bfloat16_host(samples * inputs_number, inputs.data(), staged_host.data());
            staged_source = staged_host.data();
        }

        Buffer inputs_device(Device::CUDA);
        inputs_device.resize_bytes(get_aligned_bytes(samples * inputs_number, staged_type), Device::CUDA);

#ifdef OPENNN_HAS_CUDA
        cudaStream_t stream = device::get_compute_stream();
        device::copy_async(inputs_device.data(),
                           staged_source,
                           samples * inputs_number * staged_bytes,
                           device::CopyKind::HostToDevice,
                           stream);
        cudaStreamSynchronize(stream);
#endif

        for (const Index batch : batch_list)
        {
            const Index processed = (samples / batch) * batch;

            if (processed <= 0)
            {
                cout << "batch_" << batch << "_error=batch larger than the test split\n";
                continue;
            }

            const Index batches = processed / batch;

            // Each batch size gets its own ForwardPropagation, so each one
            // captures its own graph on its first call.
            ForwardPropagation forward_propagation(batch, &network);
            forward_propagation.set_cuda_graph(true);

            Buffer staging_input;
            staging_input.resize_bytes(batch * inputs_number * staged_bytes, Device::CUDA);
            const TensorView staging_view(staging_input.data(),
                                          Shape{batch, inputs_number}, staged_type, Device::CUDA);

            bool parameters_uploaded = false;
            TensorView outputs;

            const auto run_pass = [&]
            {
#ifdef OPENNN_HAS_CUDA
                cudaStream_t compute = device::get_compute_stream();
#endif
                for (Index b = 0; b < batches; ++b)
                {
                    const Index start = b * batch;
#ifdef OPENNN_HAS_CUDA
                    device::copy_async(staging_input.data(),
                                       static_cast<const char*>(inputs_device.data())
                                           + start * inputs_number * staged_bytes,
                                       batch * inputs_number * staged_bytes,
                                       device::CopyKind::DeviceToDevice,
                                       compute);
#endif
                    outputs = network.calculate_outputs_resident({staging_view}, forward_propagation,
                                                                 !parameters_uploaded);
                    parameters_uploaded = true;
                }
            };

            const MatrixR host_batch = inputs.topRows(batch);
            MatrixR default_outputs;

            const auto run_default_pass = [&]
            {
                for (Index b = 0; b < batches; ++b)
                {
                    if (api_path == "construct")
                    {
                        ForwardPropagation probe(batch, &network, ForwardPropagationMode::Inference);
                        (void)probe.get_outputs();
                    }
                    else if (api_path == "reuse")
                        network.calculate_outputs(host_batch, default_outputs);
                    else
                        default_outputs = network.calculate_outputs(host_batch);
                }
            };

            if (api_path == "resident")
                run_pass();
            else
            {
                run_pass();                     // capture the graph, then leave it
                run_default_pass();
            }
#ifdef OPENNN_HAS_CUDA
            cudaDeviceSynchronize();
#endif

            vector<double> times;
            times.reserve(size_t(runs));

            for (Index r = 0; r < runs; ++r)
            {
                const auto t0 = clock_type::now();
                if (api_path == "resident")
                    run_pass();
                else
                    run_default_pass();
#ifdef OPENNN_HAS_CUDA
                cudaDeviceSynchronize();
#endif
                times.push_back(chrono::duration<double>(clock_type::now() - t0).count());
            }

            double checksum = 0.0;
            for (Index i = 0; i < Index(default_outputs.size()); ++i)
                checksum += double(default_outputs.data()[i]);

#ifdef OPENNN_HAS_CUDA
            // A few outputs come back to the host: a network that produces
            // non-finite values has no throughput worth reporting.
            float probe[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const Index probe_size = min<Index>(Index(4), batch);
            copy_device_to_host_float(outputs.get_data(), outputs.get_type(), probe_size, probe, stream);
            cudaStreamSynchronize(stream);
            for (Index i = 0; i < probe_size; ++i)
                if (!isfinite(probe[i]))
                    throw runtime_error("non-finite outputs");
#endif

            // In temporal order, before the sort: a median hides a drifting
            // machine entirely. If these fall monotonically the run is
            // measuring the clock, not the code.
            cout << "batch_" << batch << "_pass_times=";
            for (size_t i = 0; i < times.size(); ++i)
                cout << (i ? "," : "") << times[i];
            cout << "\n";

            sort(times.begin(), times.end());
            const double median_pass_s = times[times.size() / 2];
            const double samples_per_sec = double(processed) / median_pass_s;
            const double ms_per_batch = median_pass_s * 1000.0 / double(batches);

            if (batch_list.size() == 1)
            {
                cout << "samples=" << processed << "\n";
                cout << "batch=" << batch << "\n";
                if (api_path != "resident")
                    cout << "checksum=" << checksum << "\n";
                cout << "median_pass_s=" << median_pass_s << "\n";
                cout << "samples_per_sec=" << long(samples_per_sec) << "\n";
                cout << "ms_per_batch=" << ms_per_batch << "\n";
            }
            else
            {
                cout << "batch_" << batch << "_samples=" << processed << "\n";
                cout << "batch_" << batch << "_samples_per_sec=" << long(samples_per_sec)
                     << " median_pass_s=" << median_pass_s
                     << " ms_per_batch=" << ms_per_batch << "\n";
            }
        }

        cout << "RESULT=OK\n";
        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << "\n";
        cout << "RESULT=ERROR\n";
        return 1;
    }
}
