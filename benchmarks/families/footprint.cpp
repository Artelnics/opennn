// What OpenNN costs before it does any work.
//
// The one part of the suite that cannot ride along on a training run:
// speed, peak memory and energy are all readings of a run in progress, whereas
// these three ask what the framework costs merely by existing.
//
//   footprint memory    CPU process resident set after empty objects
//   footprint startup   time to first prediction
//   footprint export    serialize a fresh small model as C and Python source
//
// Each mode is one process that does its thing and exits, because that is the
// only honest way to measure a cost paid at startup: anything sharing a
// process with another mode has already paid it.
//
// Note what `startup` can and cannot see. It times from main(), so it counts
// library initialisation, construction and the forward pass, but not the
// dynamic loader -- and for an engine whose shared objects run to hundreds of
// megabytes the loader is most of the answer. run.py records whole-process
// wall time alongside, which is the figure to compare.

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

#ifdef __linux__
#include <unistd.h>
#endif

#include "opennn/core/configuration.h"
#include "opennn/core/tensor_operations.h"
#include "opennn/core/random_utilities.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/network/model_expression.h"
#include "opennn/network/network.h"
#include "opennn/models/models.h"
#include "opennn/training/training.h"

using namespace opennn;
using clock_type = chrono::steady_clock;

namespace
{

optional<double> resident_mb()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
        return double(counters.WorkingSetSize) / (1024.0 * 1024.0);
#elif defined(__linux__)
    // statm reports pages; the second field is resident.
    ifstream statm("/proc/self/statm");
    long total = 0, resident = 0;
    const long page_size = sysconf(_SC_PAGESIZE);
    if ((statm >> total >> resident) && page_size > 0)
        return double(resident) * double(page_size) / (1024.0 * 1024.0);
#endif
    return nullopt;
}

int usage()
{
    cerr << "usage: footprint memory | startup | export\n";
    return 2;
}

}   // namespace

int main(int argc, char* argv[])
{
    const auto entered = clock_type::now();
    // Each engine at its best, as PROTOCOL.md requires. The library defaults to
    // Eigen so a plain build behaves like a plain build; a build that has the
    // MKL kernels is told to use them here rather than inheriting them.
    Configuration::instance().set_blas(Blas::Mkl);
    cout << "blas=" << (blas_mkl_available() ? "mkl" : "eigen") << "\n";

    const string mode = argc > 1 ? argv[1] : "";

    cout << "engine=opennn\nmode=" << mode << "\n";

    if (mode == "memory")
    {
        // Empty objects only: a network with no layers, a dataset with no
        // data, a training coordinator over both. This is the floor -- what an
        // application pays for linking the library and declaring intent,
        // before a single sample is loaded.
        Configuration::instance().set(Device::CPU, Type::FP32);

        Network network;
        TabularDataset dataset;
        Training training(&network, &dataset);

        const auto memory = resident_mb();
        cout << "device=cpu\nbaseline_ram_mb=";
        if (memory) cout << *memory;
        else cout << "null";
        cout << "\nbaseline_ram_metric=process_resident_set_mib\n"
             << "baseline_ram_note=" << (memory ? "available" : "resident_memory_query_unavailable") << "\n";
    }
    else if (mode == "startup")
    {
        Configuration::instance().set(Device::CPU, Type::FP32);

        ApproximationNetwork network({10}, {64}, {1});

        MatrixR input(1, 10);
        input.setOnes();

        const MatrixR output = network.calculate_outputs(input);

        const double seconds =
            chrono::duration<double>(clock_type::now() - entered).count();

        cout << "prediction=" << output(0, 0) << "\n"
             << "first_prediction_s=" << seconds << "\n"
             << "first_prediction_scope=main_entry_to_completed_prediction_excludes_dynamic_loader\n"
             << "device=cpu\n";
    }
    else if (mode == "export")
    {
        set_seed(42);
        Configuration::instance().set(Device::CPU, Type::FP32);

        // Both engines serialize a freshly initialized 3 -> 64 -> 1 Tanh
        // model. Training would make this process lifetime an unequal mixture
        // of training and export work. The output files live outside the tree.
        const filesystem::path scratch =
            filesystem::temp_directory_path() / "opennn_footprint";
        filesystem::create_directories(scratch);

        const string c_path = (scratch / "model.c").string();
        const string python_path = (scratch / "model.py").string();
        ApproximationNetwork network({3}, {64}, {1});
        network.set_input_names({"x0", "x1", "x2"});
        network.set_output_names({"y"});

        ModelExpression expression(&network);
        expression.save(c_path, ModelExpression::ProgrammingLanguage::C);
        expression.save(python_path, ModelExpression::ProgrammingLanguage::Python);

        const auto size_of = [](const string& path) -> long
        {
            ifstream file(path, ios::binary | ios::ate);
            return file ? long(file.tellg()) : -1;
        };

        // The C export needs only the C standard library. The Python export
        // is source without an OpenNN runtime, but imports NumPy and pandas.
        // File sizes here exclude those Python dependencies.
        cout << "export_c_bytes=" << size_of(c_path) << "\n"
             << "export_python_bytes=" << size_of(python_path) << "\n";
    }
    else
    {
        return usage();
    }

    cout << "RESULT=OK\n";

    return 0;
}
