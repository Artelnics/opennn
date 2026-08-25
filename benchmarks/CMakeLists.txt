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

# footprint asks what a framework costs *before* it runs anything, so unlike
# the families above it cannot ride along on a training run: each mode is its
# own process, because a cost paid at startup is already paid by anything
# sharing a process with it.
add_executable(footprint_opennn families/footprint.cpp)

if(WIN32)
    target_link_libraries(footprint_opennn PRIVATE psapi)
endif()

list(APPEND _benchmark_targets footprint_opennn)

foreach(benchmark_target ${_benchmark_targets})
    target_link_libraries(${benchmark_target} PRIVATE opennn)
endforeach()

# So CI can compile every benchmark without running any: most need datasets CI
# does not have. OpenNN_BUILD_BENCHMARKS defaults to OFF, and the previous
# generation went unbuilt long enough to stop compiling at all.
add_custom_target(benchmarks)
add_dependencies(benchmarks ${_benchmark_targets})
