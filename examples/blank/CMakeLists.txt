cmake_minimum_required(VERSION 3.18)

project(blank LANGUAGES CXX)

add_executable(blank main.cpp)
target_include_directories(blank PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../opennn
)
target_link_libraries(blank PRIVATE opennn)

if(OPENNN_HAS_CUDA AND OpenNN_BUILD_BLANK_CUDA)
    add_executable(blank_cuda main_cuda.cpp)
    target_include_directories(blank_cuda PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../opennn
    )
    target_link_libraries(blank_cuda PRIVATE
        opennn
        CUDA::cudart
        CUDA::cublas
    )
    target_compile_options(blank_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    if(HAVE_CUDNN)
        target_link_libraries(blank_cuda PRIVATE ${CUDNN_LIBRARY})
        target_include_directories(blank_cuda PRIVATE ${CUDNN_INCLUDE_DIR})
    endif()

    # Temporary bf16 gradient diagnostic (July 2026) — safe to delete.
    add_executable(diag_grad diag_grad.cpp)
    target_include_directories(diag_grad PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../opennn
    )
    target_link_libraries(diag_grad PRIVATE
        opennn
        CUDA::cudart
        CUDA::cublas
    )
    if(HAVE_CUDNN)
        target_link_libraries(diag_grad PRIVATE ${CUDNN_LIBRARY})
        target_include_directories(diag_grad PRIVATE ${CUDNN_INCLUDE_DIR})
    endif()
endif()






