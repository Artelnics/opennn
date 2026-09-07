//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   E N E R G Y   M E T E R

#include "opennn/core/cuda/energy_meter.h"

#ifdef OPENNN_HAS_CUDA
#include <cuda_runtime.h>
#include <nvml.h>
#include <chrono>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace opennn::device
{

#ifdef OPENNN_HAS_CUDA

namespace
{

// NVML is loaded at run time rather than linked, so a machine without the
// driver library still links and simply reports no meter. The driver ships it
// as libnvidia-ml.so.1 on Linux and nvml.dll (in System32) on Windows.
void* open_nvml_library() noexcept
{
#ifdef _WIN32
    return static_cast<void*>(LoadLibraryA("nvml.dll"));
#else
    return dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
}

void* nvml_symbol(void* library, const char* name) noexcept
{
#ifdef _WIN32
    return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(library), name));
#else
    return dlsym(library, name);
#endif
}

struct Nvml
{
    void* library = nullptr;
    nvmlDevice_t device = nullptr;
    decltype(&nvmlDeviceGetSamples) get_samples = nullptr;
    bool ready = false;

    Nvml()
    {
        library = open_nvml_library();
        if (!library) return;

        const auto init = reinterpret_cast<decltype(&nvmlInit_v2)>(nvml_symbol(library, "nvmlInit_v2"));
        const auto by_bus = reinterpret_cast<decltype(&nvmlDeviceGetHandleByPciBusId_v2)>(
            nvml_symbol(library, "nvmlDeviceGetHandleByPciBusId_v2"));
        get_samples = reinterpret_cast<decltype(&nvmlDeviceGetSamples)>(nvml_symbol(library, "nvmlDeviceGetSamples"));
        if (!init || !by_bus || !get_samples) return;
        if (init() != NVML_SUCCESS) return;

        int cuda_device = 0;
        char bus_id[32] = {};
        if (cudaGetDevice(&cuda_device) != cudaSuccess
            || cudaDeviceGetPCIBusId(bus_id, int(sizeof(bus_id)), cuda_device) != cudaSuccess)
        {
            cudaGetLastError();
            return;
        }
        if (by_bus(bus_id, &device) != NVML_SUCCESS) return;

        // A driver that does not keep the power ring answers NOT_SUPPORTED
        // here rather than later, mid-measurement. NOT_FOUND only means the
        // ring holds nothing newer than "now", which is fine.
        nvmlValueType_t type = NVML_VALUE_TYPE_DOUBLE;
        unsigned int count = 0;
        const nvmlReturn_t status = get_samples(device, NVML_TOTAL_POWER_SAMPLES, 0, &type, &count, nullptr);
        ready = status == NVML_SUCCESS || status == NVML_ERROR_NOT_FOUND;
    }
};

const Nvml& nvml()
{
    static const Nvml instance;
    return instance;
}

unsigned long long now_microseconds() noexcept
{
    return static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

}

bool energy_meter_available() noexcept
{
    return nvml().ready;
}

unsigned long long energy_meter_begin() noexcept
{
    return now_microseconds();
}

double energy_meter_window_watts(const unsigned long long begin, int& samples) noexcept
{
    samples = 0;
    const Nvml& meter = nvml();
    if (!meter.ready) return 0.0;

    nvmlValueType_t type = NVML_VALUE_TYPE_DOUBLE;
    unsigned int count = 0;
    if (meter.get_samples(meter.device, NVML_TOTAL_POWER_SAMPLES, begin, &type, &count, nullptr) != NVML_SUCCESS
        || count == 0)
        return 0.0;

    std::vector<nvmlSample_t> ring(count);
    if (meter.get_samples(meter.device, NVML_TOTAL_POWER_SAMPLES, begin, &type, &count, ring.data()) != NVML_SUCCESS
        || count == 0)
        return 0.0;

    // Power samples are milliwatts in the unsigned-int arm of the union.
    double milliwatts = 0.0;
    for (unsigned int i = 0; i < count; ++i) milliwatts += double(ring[i].sampleValue.uiVal);
    samples = int(count);
    return milliwatts / double(count) / 1000.0;
}

#else

bool energy_meter_available() noexcept { return false; }
unsigned long long energy_meter_begin() noexcept { return 0; }
double energy_meter_window_watts(unsigned long long, int& samples) noexcept { samples = 0; return 0.0; }

#endif

}
