cmake_minimum_required(VERSION 3.18)

project(benchmarks)

# OpenNN benchmark drivers. Each source lives next to its write-up in
# docs/benchmarks/<benchmark>/ so the code ships with the methodology;
# binaries land in build/bin alongside the examples.

# opennn_capacity measures the peak working set via psapi (Windows-only).
if(WIN32)
    add_executable(opennn_capacity
        capacity/data-capacity/opennn_capacity.cpp)
    target_link_libraries(opennn_capacity PRIVATE opennn psapi)
endif()

add_executable(opennn_memory
    footprint/memory/opennn_memory.cpp)
if(WIN32)
    target_link_libraries(opennn_memory PRIVATE psapi)
endif()

add_executable(opennn_speed
    throughput/higgs-gpu/opennn_speed.cpp)

add_executable(opennn_higgs_cpu
    throughput/higgs/opennn_higgs_cpu.cpp)

add_executable(opennn_higgs_maxbatch_trial
    capacity/higgs-max-batch/opennn_higgs_maxbatch_trial.cpp)

add_executable(opennn_higgs_infer
    throughput/higgs-gpu/opennn_higgs_infer.cpp)

add_executable(opennn_resnet50_speed
    throughput/resnet50/opennn_resnet50_speed.cpp)

add_executable(opennn_resnet50_infer
    throughput/resnet50/opennn_resnet50_infer.cpp)

# Transformer benchmark binaries (inference, training, and the CPU/GPU validator).
add_executable(opennn_transformer_infer
    throughput/attention-speed/opennn_transformer_infer.cpp)

add_executable(opennn_attention_validate
    throughput/attention-speed/opennn_attention_validate.cpp)

add_executable(opennn_resnet50_maxbatch_trial
    capacity/resnet50-max-batch/opennn_resnet50_maxbatch_trial.cpp)

# These four include <cuda_runtime.h> unconditionally, so they only exist in
# CUDA builds; the rest of the suite guards its CUDA usage and builds CPU-only.
if(OPENNN_HAS_CUDA)
    add_executable(opennn_transformer_resident
        throughput/attention-speed/opennn_transformer_resident.cpp)

    add_executable(opennn_transformer_train
        throughput/attention-speed/opennn_transformer_train.cpp)

    add_executable(opennn_transformer_maxbatch_trial
        capacity/transformer-max-batch/opennn_transformer_maxbatch_trial.cpp)

    add_executable(opennn_transformer_energy
        energy/transformer-energy/opennn_transformer_energy.cpp)
endif()

# CPU quality benchmarks (HIGGS classification; precision is the documented regression exception).
add_executable(opennn_accuracy
    quality/accuracy/opennn_accuracy.cpp)

add_executable(opennn_convergence
    quality/convergence/opennn_convergence.cpp)

add_executable(opennn_precision
    quality/precision/opennn_precision.cpp)

# Footprint measurement binaries.
add_executable(opennn_startup
    footprint/startup/opennn_startup.cpp)

add_executable(opennn_export
    footprint/export/opennn_export.cpp)

# run_forecasting.py looks for this exact binary name in build/bin.
add_executable(recurrent_lstm_forecasting_benchmark
    quality/recurrent-lstm-forecasting/recurrent_lstm_forecasting_benchmark.cpp)

set(_benchmark_targets
        opennn_memory
        opennn_speed
        opennn_higgs_infer
        opennn_higgs_cpu
        opennn_higgs_maxbatch_trial
        opennn_resnet50_speed
        opennn_resnet50_infer
        opennn_resnet50_maxbatch_trial
        opennn_transformer_infer
        opennn_attention_validate
        opennn_accuracy
        opennn_convergence
        opennn_precision
        opennn_startup
        opennn_export
        recurrent_lstm_forecasting_benchmark)

if(OPENNN_HAS_CUDA)
    list(APPEND _benchmark_targets
        opennn_transformer_resident
        opennn_transformer_train
        opennn_transformer_maxbatch_trial
        opennn_transformer_energy)
endif()

foreach(benchmark_target ${_benchmark_targets})
    target_link_libraries(${benchmark_target} PRIVATE opennn)
endforeach()
