# The OpenNN side of the benchmark suite.
#
# One program per family, each driven by mode -- train, infer, capacity --
# rather than one program per metric. See PLAN.md; run.py locates these by
# name, so `<family>_opennn` is a contract rather than a convention.

add_executable(dense_opennn       families/dense.cpp)
add_executable(cnn_opennn         families/cnn.cpp)
add_executable(transformer_opennn families/transformer.cpp)
add_executable(lstm_opennn        families/lstm.cpp)

set(_benchmark_targets
        dense_opennn
        cnn_opennn
        transformer_opennn
        lstm_opennn)

# footprint/ is the one part of the previous suite not yet folded into a
# family. It asks what a framework costs *before* it runs anything -- baseline
# memory, time to first prediction, exported model size -- which cannot ride
# along on a training run the way speed, peak memory and energy now do.
add_executable(opennn_memory  footprint/memory/opennn_memory.cpp)
add_executable(opennn_startup footprint/startup/opennn_startup.cpp)
add_executable(opennn_export  footprint/export/opennn_export.cpp)

if(WIN32)
    target_link_libraries(opennn_memory PRIVATE psapi)
endif()

list(APPEND _benchmark_targets
        opennn_memory
        opennn_startup
        opennn_export)

foreach(benchmark_target ${_benchmark_targets})
    target_link_libraries(${benchmark_target} PRIVATE opennn)
endforeach()

# So CI can compile every benchmark without running any: most need datasets CI
# does not have. OpenNN_BUILD_BENCHMARKS defaults to OFF, and the previous
# generation went unbuilt long enough to stop compiling at all.
add_custom_target(benchmarks)
add_dependencies(benchmarks ${_benchmark_targets})
