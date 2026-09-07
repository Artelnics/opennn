# run.py depends on the <family>_opennn executable names.
set(_benchmark_targets "")

foreach(family dense cnn transformer lstm footprint qwen)
    set(target "${family}_opennn")
    add_executable(${target} "families/${family}.cpp")
    target_link_libraries(${target} PRIVATE opennn)
    list(APPEND _benchmark_targets ${target})
endforeach()

if(WIN32)
    target_link_libraries(footprint_opennn PRIVATE psapi)
endif()

# Compile every benchmark without running datasets that may not be available.
add_custom_target(benchmarks DEPENDS ${_benchmark_targets})
