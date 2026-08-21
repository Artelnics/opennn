//   OpenNN GPU HIGGS dense inference-speed benchmark.
//
//   Forward-only throughput of the canonical HIGGS dense classifier
//   (28 -> hidden -> hidden -> 1, ReLU hidden, sigmoid output -- see
//   docs/benchmarks/throughput/higgs/README.md) on the GPU. The test CSV is
//   loaded once (features then last-column label; the label is ignored for the
//   speed measurement), the inputs are made device-resident, and the network
//   forward (calculate_outputs_resident) is replayed over batches: a warmup pass
//   plus N timed passes. No optimizer state, no gradients, no per-call H2D copy.
//
//   Precision is selectable fp32 or bf16, matching the autocast / mixed_bfloat16
//   used on the PyTorch and TensorFlow sides. It is selected exactly like
//   opennn_speed.cpp: Configuration::instance().set(Device::CUDA, type).
//
//   usage:  opennn_higgs_infer <test_csv> [batch[,batch...]] [runs] [fp32|bf16] [hidden] [hidden_layers] [activation]
//                              [resident|default|reuse|construct]
//
//   A comma-separated batch list is swept inside one process, the way the CPU
//   driver does: the batch sizes then share one load and one thermal window, so
//   a row of the comparison table is internally comparable. Per-pass times are
//   printed in temporal order before the median, so drift is visible in the
//   data rather than averaged away.
//
//   The resident rows are held in the network's own compute type, which is what
//   the PyTorch driver does for its bf16 cell (PT_BF16_WEIGHTS). Holding them as
//   fp32 instead costs a cast kernel inside the graph on every batch and doubles
//   the bytes of the staging copy. OPENNN_INFER_STAGE=fp32 restores that for the
//   A/B.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#ifdef OPENNN_HAS_CUDA
#include <cuda_runtime.h>
#include "opennn/core/device_backend.h"
#endif

#include "opennn/core/configuration.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/forward_propagation.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/core/tensor_types.h"

using namespace opennn;
using clock_type = chrono::steady_clock;

namespace
{

// Parsing the 500k-row split is seconds of full-machine work immediately before
// a GPU measurement, and a sweep pays it once per engine per round. The parsed
// floats are cached beside the CSV, the way the Python drivers cache theirs as
// .npy, and re-read whenever the cache is newer than the CSV. Nothing timed
// changes: the cache holds exactly what the parser produced. Same format and
// same file as the CPU driver's cache, deliberately - they share the split.
struct Table
{
    Index rows = 0;
    Index columns = 0;
    vector<float> values;
};

const char table_magic[8] = {'O', 'N', 'N', 'T', 'B', 'L', '0', '1'};

bool read_table_cache(const filesystem::path& cache, Table& table)
{
    ifstream file(cache, ios::binary);
    if (!file) return false;

    char magic[sizeof(table_magic)] = {};
    int64_t rows = 0;
    int64_t columns = 0;

    file.read(magic, sizeof(magic));
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&columns), sizeof(columns));

    if (!file || memcmp(magic, table_magic, sizeof(magic)) != 0 || rows <= 0 || columns <= 0)
        return false;

    table.rows = Index(rows);
    table.columns = Index(columns);
    table.values.resize(size_t(rows) * size_t(columns));
    file.read(reinterpret_cast<char*>(table.values.data()),
              streamsize(table.values.size() * sizeof(float)));

    return bool(file);
}

void write_table_cache(const filesystem::path& cache, const Table& table)
{
    ofstream file(cache, ios::binary);
    if (!file) return;                          // read-only data directory: parse next time

    const int64_t rows = int64_t(table.rows);
    const int64_t columns = int64_t(table.columns);

    file.write(table_magic, sizeof(table_magic));
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&columns), sizeof(columns));
    file.write(reinterpret_cast<const char*>(table.values.data()),
               streamsize(table.values.size() * sizeof(float)));
}

Table load_table(const string& csv_path)
{
    const filesystem::path csv(csv_path);
    const filesystem::path cache = filesystem::path(csv_path + ".bin");

    Table table;

    error_code error;
    const auto cache_time = filesystem::last_write_time(cache, error);

    if (!error && cache_time >= filesystem::last_write_time(csv)
        && read_table_cache(cache, table))
        return table;

    TabularDataset dataset(csv_path, ",", false, false);
    const MatrixR& data = dataset.get_data();

    table.rows = data.rows();
    table.columns = data.cols();
    table.values.assign(data.data(), data.data() + data.size());

    write_table_cache(cache, table);

    return table;
}

vector<Index> parse_batches(const string& text)
{
    vector<Index> batches;
    size_t start = 0;

    while (start <= text.size())
    {
        const size_t comma = text.find(',', start);
        const string item = text.substr(start, comma == string::npos ? string::npos : comma - start);

        if (!item.empty()) batches.push_back(Index(stoll(item)));
        if (comma == string::npos) break;
        start = comma + 1;
    }

    if (batches.empty()) batches.push_back(8192);

    return batches;
}

unique_ptr<NeuralNetwork> make_network(const Shape& input_shape,
                                            const Shape& target_shape,
                                            Index hidden,
                                            Index hidden_layers,
                                            const string& activation)
{
    auto network = make_unique<NeuralNetwork>();
    Shape current = input_shape;
    const string hidden_activation = (activation == "relu" || activation == "ReLU")
        ? "ReLU"
        : "Tanh";

    for (Index i = 0; i < hidden_layers; ++i)
    {
        network->add_layer(make_unique<opennn::Dense>(
            current,
            Shape{hidden},
            hidden_activation,
            false,
            "higgs_dense_" + to_string(i + 1)));
        current = network->get_output_shape();
    }

    network->add_layer(make_unique<opennn::Dense>(
        current,
        target_shape,
        "Sigmoid",
        false,
        "higgs_output"));

    network->compile();
    network->set_parameters_glorot();
    return network;
}

}

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
        const vector<Index> batch_list = parse_batches(argc > 2 ? argv[2] : "8192");
        const Index runs  = argc > 3 ? max<Index>(Index(1), Index(stoll(argv[3]))) : 5;
        const string precision = argc > 4 ? argv[4] : "fp32";
        const Index hidden = argc > 5 ? Index(stoll(argv[5])) : 1024;
        const Index hidden_layers = argc > 6 ? Index(stoll(argv[6])) : 2;
        const string activation = argc > 7 ? argv[7] : "relu";

        // The benchmarks all measured calculate_outputs_resident, the expert
        // path where the caller owns the ForwardPropagation and results stay on
        // the device. `default` measures what ordinary code calls instead:
        // calculate_outputs takes host inputs and returns host outputs. `reuse`
        // is the same call with a caller-provided output buffer.
        const string api_path = argc > 8 ? string(argv[8]) : "resident";

        set_seed(42);
        const Type inference_type = (precision == "bf16") ? Type::BF16 : Type::FP32;
        Configuration::instance().set(Device::CUDA, inference_type);

        const Table table = load_table(test_path);
        const Index samples = table.rows;
        const Index inputs_number = table.columns - 1;
        const MatrixR inputs = Eigen::Map<const MatrixR>(table.values.data(),
                                                        table.rows,
                                                        table.columns).leftCols(inputs_number);

        const char* const stage_request = getenv("OPENNN_INFER_STAGE");
        const bool stage_as_compute_type = !(stage_request && string(stage_request) == "fp32");
        const Type staged_type = stage_as_compute_type ? inference_type : Type::FP32;
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

        auto network = make_network(Shape{inputs_number},
                                    Shape{table.columns - inputs_number},
                                    hidden,
                                    hidden_layers,
                                    activation);
        cout << "parameters=" << network->get_parameters_number() << "\n";

        // The whole split goes to the device once, in the compute type; every
        // batch size in the sweep reads the same rows from it. Converting here
        // rather than on the device is deliberate: a cast on the device would
        // have to live either in the graph, where it is a node per batch, or in
        // a pass of its own over the whole split.
        vector<uint16_t> staged_host;
        const void* staged_source = inputs.data();

        if (staged_type == Type::BF16)
        {
            staged_host.resize(size_t(samples * inputs_number));
            float_2_bfloat16_host(samples * inputs_number, inputs.data(), staged_host.data());
            staged_source = staged_host.data();
        }

        Buffer inputs_device(Device::CUDA);
        const Index input_bytes = get_aligned_bytes(samples * inputs_number, staged_type);
        inputs_device.resize_bytes(input_bytes, Device::CUDA);

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

            ForwardPropagation forward_propagation(batch, network.get());
            forward_propagation.set_cuda_graph(true);

            Buffer staging_input;
            staging_input.resize_bytes(batch * inputs_number * staged_bytes, Device::CUDA);
            const TensorView staging_view(staging_input.data(),
                                          Shape{batch, inputs_number}, staged_type, Device::CUDA);

            // Each batch size gets its own ForwardPropagation, so each one
            // captures its own graph on its first call.
            bool parameters_uploaded = false;
            const TensorView* last_outputs = nullptr;
            TensorView probe_view;

            auto run_pass = [&]()
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

                    const bool upload_parameters = !parameters_uploaded;
                    probe_view = network->calculate_outputs_resident(
                        {staging_view}, forward_propagation, upload_parameters);
                    parameters_uploaded = true;
                    last_outputs = &probe_view;
                }
            };

            const MatrixR host_batch = inputs.topRows(batch);
            MatrixR default_outputs;

            auto run_default_pass = [&]()
            {
                for (Index b = 0; b < batches; ++b)
                {
                    // `construct` isolates what the default path pays per call
                    // beyond the work itself: planning the arena and allocating
                    // it. No forward pass is run.
                    if (api_path == "construct")
                    {
                        ForwardPropagation probe(batch, network.get(),
                                                 ForwardPropagationMode::Inference);
                        (void)probe.get_outputs();
                    }
                    else if (api_path == "reuse")
                        network->calculate_outputs(host_batch, default_outputs);
                    else
                        default_outputs = network->calculate_outputs(host_batch);
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
                const auto t1 = clock_type::now();
                times.push_back(chrono::duration<double>(t1 - t0).count());
            }

            double default_checksum = 0.0;
            for (Index i = 0; i < Index(default_outputs.size()); ++i)
                default_checksum += double(default_outputs.data()[i]);

#ifdef OPENNN_HAS_CUDA
            if (last_outputs)
            {
                float probe[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                const Index probe_size = min<Index>(Index(4), batch);
                copy_device_to_host_float(last_outputs->get_data(), last_outputs->get_type(),
                                          probe_size, probe, stream);
                cudaStreamSynchronize(stream);
                for (Index i = 0; i < probe_size; ++i)
                    if (!isfinite(probe[i]))
                        throw runtime_error("non-finite outputs");
            }
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
                    cout << "checksum=" << default_checksum << "\n";
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

            cout.flush();
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
