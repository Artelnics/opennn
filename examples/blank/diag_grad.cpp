// Temporary diagnostic: dump the parameter gradient of one fixed batch to a
// binary file, with a per-view offset table on stdout. Run once with fp32 and
// once with bf16, then compare per-layer gradient agreement offline.
// Usage: diag_grad <fp32|bf16> <output.bin> [layers=12]

#include <bit>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "../opennn/configuration.h"
#include "../opennn/device_backend.h"
#include "../opennn/standard_networks.h"
#include "../opennn/text_generation_dataset.h"
#include "../opennn/loss.h"
#include "../opennn/forward_propagation.h"
#include "../opennn/back_propagation.h"
#include "../opennn/batch.h"
#include "../opennn/random_utilities.h"

using namespace std;
using namespace opennn;

int main(int argc, char** argv)
{
    try
    {
        const string precision    = argc > 1 ? argv[1] : "fp32";
        const string output_path  = argc > 2 ? argv[2] : "gradient.bin";
        const Index layers_number = argc > 3 ? Index(stoll(argv[3])) : Index(12);
        const bool use_sdpa       = argc > 4 ? (stoi(argv[4]) != 0) : true;

        Configuration::instance().set(Device::CUDA,
                                      precision == "bf16" ? Type::BF16 : Type::FP32);
        Backend::instance();
        set_seed(42);

        TextGenerationDataset dataset(
            "/home/artelnics/Documents/datasets/wmt14_en_de/wmt14_en.subset.txt", 512);

        TextGenerationNetwork network(512,
                                      dataset.get_vocabulary_size(),
                                      768, 12, 3072,
                                      layers_number,
                                      true);

        // Force bit-identical master weights across the fp32/bf16 runs: the
        // first run saves its init, later runs load it.
        const filesystem::path weights_path =
            filesystem::path(output_path).parent_path() / ("diag_weights_" + to_string(layers_number) + "L.bin");
        if (filesystem::exists(weights_path))
        {
            network.load_parameters_binary(weights_path);
            cout << "LOADED shared weights\n";
        }
        else
        {
            network.save_parameters_binary(weights_path);
            cout << "SAVED shared weights\n";
        }

        if (!use_sdpa) network.set_attention_sdpa_auto(false);

        network.copy_parameters_device();
        network.link_parameters();
        network.copy_states_device();
        network.link_states();

        Loss loss(&network, &dataset);
        loss.set_error("CrossEntropyError3d");

        const Index batch_size = 8;
        Batch batch(batch_size, &dataset, network.get_config());

        vector<Index> samples = dataset.get_sample_indices("Training");
        samples.resize(batch_size);

        batch.fill(samples,
                   dataset.get_feature_indices("Input"),
                   dataset.get_feature_indices("Decoder"),
                   dataset.get_feature_indices("Target"),
                   true);

        cudaStream_t stream = Backend::get_compute_stream();
        batch.copy_device_async(stream);
        device::synchronize(stream);

        ForwardPropagation forward(batch_size, &network);
        BackPropagation backward(batch_size, &loss);

        auto dump_view = [&](const TensorView& view, const string& path)
        {
            const size_t n = size_t(view.size());
            vector<float> host_data(n, 0.0f);
            if (view.is_bf16())
            {
                vector<uint16_t> raw(n, 0);
                device::copy_async(raw.data(), view.data, Index(n * sizeof(uint16_t)),
                                   device::CopyKind::DeviceToHost, stream);
                device::synchronize(stream);
                for (size_t i = 0; i < n; ++i)
                    host_data[i] = bit_cast<float>(uint32_t(raw[i]) << 16);
            }
            else
            {
                device::copy_async(host_data.data(), view.data, Index(n * sizeof(float)),
                                   device::CopyKind::DeviceToHost, stream);
                device::synchronize(stream);
            }
            ofstream f(path, ios::binary);
            f.write(reinterpret_cast<const char*>(host_data.data()),
                    streamsize(n * sizeof(float)));
        };

        network.forward_propagate(batch.get_inputs(), forward, true);
        device::synchronize(stream);

        dump_view(forward.get_last_trainable_layer_outputs(), output_path + ".softmax");

        for (size_t i = 0; i < forward.forward_slots.size(); ++i)
        {
            if (forward.forward_slots[i].size() <= 1) continue;
            const TensorView& out = forward.forward_slots[i].back();
            if (out.empty()) continue;
            dump_view(out, output_path + ".fwd" + to_string(i));
            cout << "FWD " << i << " " << network.get_layer(Index(i))->get_label()
                 << " " << out.size() << "\n";
        }

        loss.back_propagate(batch, forward, backward);
        device::synchronize(stream);

        dump_view(backward.get_output_delta(), output_path + ".delta");

        const Index total_floats = backward.gradient.size_in_floats();
        vector<float> host(static_cast<size_t>(total_floats), 0.0f);
        device::copy_async(host.data(), backward.gradient.as<float>(),
                           total_floats * Index(sizeof(float)),
                           device::CopyKind::DeviceToHost, stream);
        device::synchronize(stream);

        ofstream file(output_path, ios::binary);
        file.write(reinterpret_cast<const char*>(host.data()),
                   streamsize(size_t(total_floats) * sizeof(float)));

        const float* base = backward.gradient.as<float>();
        const auto& layers = network.get_layers();
        for (size_t i = 0; i < backward.gradient_views.size(); ++i)
            for (size_t j = 0; j < backward.gradient_views[i].size(); ++j)
            {
                const auto& view = backward.gradient_views[i][j];
                if (view.empty()) continue;
                cout << "VIEW " << i << " " << j << " "
                     << (view.as<float>() - base) << " " << view.size() << " "
                     << layers[i]->get_label() << "\n";
            }
        cout << "TOTAL " << total_floats << "\n";

        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;
        return 1;
    }
}
