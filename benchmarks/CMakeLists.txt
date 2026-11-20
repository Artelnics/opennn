# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

# run.py depends on the <family>_opennn executable names.
set(_benchmark_targets "")

# The CPU LSTM cells compare oneDNN's recurrent primitive with PyTorch's. A build
# without it measures the built-in path instead, at a third of the throughput.
if(NOT OpenNN_ONEDNN_FOUND)
    message(WARNING
        "OpenNN benchmarks: oneDNN is not linked, so the CPU LSTM cells measure "
        "the built-in recurrent path. Enable OpenNN_FETCH_ONEDNN or set OpenNN_ONEDNN_ROOT.")
endif()
if(NOT OpenNN_ENABLE_MKL)
    message(WARNING
        "OpenNN benchmarks: MKL is off, so the CPU cells run Eigen's BLAS, about 5% "
        "slower on LSTM training. Configure with OpenNN_ENABLE_MKL=ON.")
endif()

foreach(family dense cnn transformer lstm quality)
    set(target "${family}_opennn")
    add_executable(${target} "families/${family}.cpp")
    target_link_libraries(${target} PRIVATE opennn)
    list(APPEND _benchmark_targets ${target})
endforeach()

# Separate binaries preserve each application's native dependency footprint.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    foreach(family dense lstm cnn transformer)
        set(target "${family}_application_opennn")
        add_executable(${target} families/application.cpp)
        target_compile_definitions(${target} PRIVATE "FAMILY_${family}=1")
        target_link_libraries(${target} PRIVATE opennn)
        list(APPEND _benchmark_targets ${target})
    endforeach()
endif()

# Compile every benchmark without running datasets that may not be available.
add_custom_target(benchmarks DEPENDS ${_benchmark_targets})
