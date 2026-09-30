option(OpenNN_BUILD_BLANK "Build the blank example template" ON)

function(opennn_example name)
    add_executable(${name} "${name}/main.cpp")
    target_link_libraries(${name} PRIVATE opennn)

    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${name}/data")
        # Re-stage changed data even when the executable does not need relinking.
        # Preserve ../data/<name> for single- and multi-configuration generators.
        file(GLOB_RECURSE data_files CONFIGURE_DEPENDS "${name}/data/*")
        set(data_script "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/prepare_data.cmake")
        set(data_stamp "${CMAKE_CURRENT_BINARY_DIR}/${name}-data-$<CONFIG>.stamp")
        add_custom_command(OUTPUT "${data_stamp}"
            COMMAND ${CMAKE_COMMAND}
                "-DOPENNN_DATA_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/${name}/data"
                "-DOPENNN_DATA_DESTINATION=$<TARGET_FILE_DIR:${name}>/../data/${name}"
                -P "${data_script}"
            COMMAND ${CMAKE_COMMAND} -E touch "${data_stamp}"
            DEPENDS ${data_files} "${data_script}"
            VERBATIM)
        add_custom_target(${name}_data DEPENDS "${data_stamp}")
        add_dependencies(${name} ${name}_data)
    endif()
endfunction()

set(_example_targets "")

if(OpenNN_BUILD_BLANK)
    opennn_example(blank)
    list(APPEND _example_targets blank)
endif()

foreach(example
    airfoil_self_noise
    amazon_reviews
    bert
    breast_cancer
    concrete
    electrocardiogram_anomaly_detection
    emotion_analysis
    gpt2
    iris_plant
    melanoma_cancer
    mnist
    qwen3
    translation
    yacht_hydrodynamics
    yolo)
    opennn_example(${example})
    list(APPEND _example_targets ${example})
endforeach()

if(MSVC)
    target_compile_options(yolo PRIVATE /W4 /w14061)
    target_precompile_headers(yolo REUSE_FROM opennn)
else()
    target_precompile_headers(yolo PRIVATE "${PROJECT_SOURCE_DIR}/opennn/core/pch.h")
endif()

# Compile every example without running it.
add_custom_target(examples DEPENDS ${_example_targets})
