option(OpenNN_BUILD_BLANK "Build the blank OpenNN example" ON)

function(opennn_example name)
    add_executable(${name} "${name}/main.cpp")
    target_link_libraries(${name} PRIVATE opennn)

    if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${name}/data")
        add_custom_command(TARGET ${name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/${name}/data"
                "${CMAKE_BINARY_DIR}/data/${name}")
    endif()
endfunction()

if(OpenNN_BUILD_BLANK)
    opennn_example(blank)
endif()

foreach(example
    airfoil_self_noise
    amazon_reviews
    bert
    breast_cancer
    ecg5000_anomaly_detection
    emotion_analysis
    forecasting_tinyml
    gpt2
    iris_plant
    melanoma_cancer
    mnist
    qwen3
    translation
    yolo)
    opennn_example(${example})
endforeach()

if(MSVC)
    target_compile_options(yolo PRIVATE /W4 /w14061)
    target_precompile_headers(yolo REUSE_FROM opennn)
else()
    target_precompile_headers(yolo PRIVATE "${PROJECT_SOURCE_DIR}/opennn/pch.h")
endif()
