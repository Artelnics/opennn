// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/core/cuda/energy_meter.h"

#ifdef OPENNN_HAS_CUDA
#include <cuda_runtime.h>
#include <nvml.h>
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
#include <chrono>
#include <vector>
#endif

namespace opennn::device
{

#ifdef OPENNN_HAS_CUDA

namespace
{

#ifdef _WIN32
using LibraryHandle = HMODULE;
#else
using LibraryHandle = void*;
#endif

LibraryHandle open_nvml_library()
{
#ifdef _WIN32
    return LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
#else
    return dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
}

template <typename Function>
Function load_nvml_symbol(const LibraryHandle library, const char* name)
{
#ifdef _WIN32
    return reinterpret_cast<Function>(GetProcAddress(library, name));
#else
    return reinterpret_cast<Function>(dlsym(library, name));
#endif
}

struct Nvml
{
    LibraryHandle library = nullptr;
    nvmlDevice_t device = nullptr;
    decltype(&nvmlDeviceGetSamples) get_samples = nullptr;
    bool ready = false;

    Nvml()
    {
        library = open_nvml_library();
        if (!library) return;

        const auto init = load_nvml_symbol<decltype(&nvmlInit_v2)>(library, "nvmlInit_v2");
        const auto by_bus = load_nvml_symbol<decltype(&nvmlDeviceGetHandleByPciBusId_v2)>(
            library, "nvmlDeviceGetHandleByPciBusId_v2");
        get_samples = load_nvml_symbol<decltype(&nvmlDeviceGetSamples)>(
            library, "nvmlDeviceGetSamples");
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
