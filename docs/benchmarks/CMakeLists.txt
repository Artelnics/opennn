cmake_minimum_required(VERSION 3.18)

project(benchmarks)

# OpenNN benchmark drivers. Each source lives next to its write-up in
# docs/benchmarks/<benchmark>/ so the code ships with the methodology;
# binaries land in build/bin alongside the examples.

# opennn_capacity measures the peak working set via psapi (Windows-only).
if(WIN32)
    add_executable(opennn_capacity
        capacity/data-capacity/opennn_capacity.cpp)
    target_link_libraries(opennn_capacity PUBLIC opennn psapi)
endif()

add_executable(opennn_memory
    footprint/memory/opennn_memory.cpp)
if(WIN32)
    target_link_libraries(opennn_memory PUBLIC psapi)
endif()

add_executable(opennn_speed
    throughput/training-speed/opennn_speed.cpp)

add_executable(opennn_higgs_cpu
    throughput/higgs/opennn_higgs_cpu.cpp)

add_executable(opennn_higgs_maxbatch_trial
    capacity/higgs-max-batch/opennn_higgs_maxbatch_trial.cpp)

add_executable(opennn_inference
    throughput/inference-speed/opennn_inference.cpp)

add_executable(opennn_resnet50_speed
    throughput/resnet50-training-speed/opennn_resnet50_speed.cpp)

add_executable(opennn_resnet50_maxbatch_trial
    capacity/resnet50-max-batch/opennn_resnet50_maxbatch_trial.cpp)

add_executable(opennn_image_classification_maxbatch_trial
    capacity/image-classification-max-batch/opennn_image_classification_maxbatch_trial.cpp)

add_executable(opennn_transformer_maxbatch_trial
    capacity/transformer-max-batch/opennn_transformer_maxbatch_trial.cpp)

add_executable(opennn_transformer_energy
    energy/transformer-energy/opennn_transformer_energy.cpp)

# run_forecasting.py looks for this exact binary name in build/bin.
add_executable(recurrent_lstm_forecasting_benchmark
    quality/recurrent-lstm-forecasting/recurrent_lstm_forecasting_benchmark.cpp)

foreach(benchmark_target
        opennn_memory
        opennn_speed
        opennn_higgs_cpu
        opennn_higgs_maxbatch_trial
        opennn_inference
        opennn_resnet50_speed
        opennn_resnet50_maxbatch_trial
        opennn_image_classification_maxbatch_trial
        opennn_transformer_maxbatch_trial
        opennn_transformer_energy
        recurrent_lstm_forecasting_benchmark)
    target_link_libraries(${benchmark_target} PUBLIC opennn)
endforeach()
