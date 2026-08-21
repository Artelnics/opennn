
# OpenNN benchmark drivers. Each source lives next to its write-up in
# docs/benchmarks/<benchmark>/ so the code ships with the methodology;
# binaries land in build/bin alongside the examples.

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

add_executable(opennn_transformer_infer
    throughput/attention-speed/opennn_transformer_infer.cpp)

add_executable(opennn_attention_validate
    throughput/attention-speed/opennn_attention_validate.cpp)

add_executable(opennn_resnet50_maxbatch_trial
    capacity/resnet50-max-batch/opennn_resnet50_maxbatch_trial.cpp)

if(OPENNN_HAS_CUDA)
    add_executable(opennn_transformer_resident
        throughput/attention-speed/opennn_transformer_resident.cpp)

    add_executable(opennn_qwen3_resident
        throughput/attention-speed/opennn_qwen3_resident.cpp)

    add_executable(opennn_transformer_train
        throughput/attention-speed/opennn_transformer_train.cpp)

    add_executable(opennn_transformer_maxbatch_trial
        capacity/transformer-max-batch/opennn_transformer_maxbatch_trial.cpp)

    add_executable(opennn_transformer_energy
        energy/transformer-energy/opennn_transformer_energy.cpp)
endif()

add_executable(opennn_accuracy
    quality/accuracy/opennn_accuracy.cpp)

add_executable(opennn_convergence
    quality/convergence/opennn_convergence.cpp)

add_executable(opennn_precision
    quality/precision/opennn_precision.cpp)

add_executable(opennn_startup
    footprint/startup/opennn_startup.cpp)

add_executable(opennn_export
    footprint/export/opennn_export.cpp)

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
    add_executable(cudnn_fusion_probe
        throughput/resnet50/cudnn_fusion_probe.cpp)
    # Builds cudnn-frontend graphs itself, so it needs the headers the library
    # keeps private.
    target_include_directories(cudnn_fusion_probe SYSTEM PRIVATE
        ${CUDNN_FRONTEND_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
    add_executable(pooling_probe
        throughput/resnet50/pooling_probe.cpp)
    target_include_directories(pooling_probe SYSTEM PRIVATE
        ${CUDNN_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
    list(APPEND _benchmark_targets
        cudnn_fusion_probe
        pooling_probe
        opennn_transformer_resident
        opennn_qwen3_resident
        opennn_transformer_train
        opennn_transformer_maxbatch_trial
        opennn_transformer_energy)
endif()

foreach(benchmark_target ${_benchmark_targets})
    target_link_libraries(${benchmark_target} PRIVATE opennn)
endforeach()

# Aggregate target so CI can compile every benchmark without running any of
# them (most need datasets that CI does not have). OpenNN_BUILD_BENCHMARKS
# defaults to OFF and CI never turned it on, so these targets went unbuilt for
# long enough to drift out of sync with library refactors and stop compiling.
add_custom_target(benchmarks)
add_dependencies(benchmarks ${_benchmark_targets})
