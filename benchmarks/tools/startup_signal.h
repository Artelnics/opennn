#pragma once
#include <ctime>
#include <iostream>

inline long long monotonic_ns()
{
    timespec value{};
    clock_gettime(CLOCK_MONOTONIC, &value);
    return static_cast<long long>(value.tv_sec) * 1000000000LL + value.tv_nsec;
}

inline void startup_ready(long long entered, long long ready, const char* engine,
                          bool gpu, bool bf16, long long parameters,
                          long long stored_parameters, long long output_values, float first)
{
    std::cout << "STARTUP_READY {\"main_ns\":" << entered << ",\"ready_ns\":" << ready
              << ",\"engine\":\"" << engine << "\",\"device\":\"" << (gpu ? "cuda" : "cpu")
              << "\",\"precision\":\"" << (bf16 ? "bf16" : "fp32")
              << "\",\"parameters\":" << parameters << ",\"stored_parameters\":" << stored_parameters
              << ",\"output_values\":" << output_values << ",\"first\":" << first << "}" << std::endl;
}
