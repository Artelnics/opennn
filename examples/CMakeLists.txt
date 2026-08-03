

if(NOT DEFINED OpenNN_BUILD_EXAMPLES)
    option(OpenNN_BUILD_EXAMPLES "Build OpenNN examples" ON)
endif()

option(OpenNN_BUILD_BLANK "Build OpenNN blank examples" ON)

function(opennn_example name)
    set(options COPY_DATA)
    cmake_parse_arguments(EXAMPLE "${options}" "" "" ${ARGN})

    add_executable(${name} "${CMAKE_CURRENT_SOURCE_DIR}/${name}/main.cpp")
    target_link_libraries(${name} PRIVATE opennn)

    if(EXAMPLE_COPY_DATA)
        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/${name}/data"
                "${CMAKE_BINARY_DIR}/data/${name}"
        )
    endif()
endfunction()

if(OpenNN_BUILD_EXAMPLES)
    if(OpenNN_BUILD_BLANK)
        add_subdirectory(blank)
    endif()

    set(examples_with_data
        airfoil_self_noise amazon_reviews bert breast_cancer emotion_analysis
        iris_plant melanoma_cancer mnist translation
    )

    foreach(example IN ITEMS
        airfoil_self_noise amazon_reviews bert breast_cancer emotion_analysis
        gpt2 qwen3 forecasting_tinyml iris_plant mnist melanoma_cancer translation
        yolo
    )
        if(example IN_LIST examples_with_data)
            opennn_example(${example} COPY_DATA)
        else()
            opennn_example(${example})
        endif()
    endforeach()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/beijing_pm25_forecasting/CMakeLists.txt")
        add_subdirectory(beijing_pm25_forecasting)
    endif()

    # REUSE_FROM requires identical compile flags. That holds on MSVC, while
    # Makefiles/GCC need a target-local PCH generated from the same header.
    if(MSVC)
        target_precompile_headers(yolo REUSE_FROM opennn)
    else()
        target_precompile_headers(yolo PRIVATE "${CMAKE_SOURCE_DIR}/opennn/pch.h")
    endif()
endif()
