cmake_minimum_required(VERSION 3.18)

project(blank_cuda LANGUAGES CXX CUDA)

add_executable(blank_cuda main.cpp)

target_include_directories(blank_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../opennn
)

target_link_libraries(blank_cuda PRIVATE
    opennn
    CUDA::cudart
    CUDA::cublas
)

target_compile_options(blank_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
add_compile_options(-Wno-interference-size)

if(HAVE_CUDNN)
    target_link_libraries(blank_cuda PRIVATE ${CUDNN_LIBRARY})
    target_include_directories(blank_cuda PRIVATE ${CUDNN_INCLUDE_DIR})
endif()
