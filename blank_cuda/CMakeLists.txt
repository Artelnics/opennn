project(blank_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 90)

add_executable(blank_cuda 
    main.cpp
    ../opennn/kernel.cu
)

target_compile_definitions(blank_cuda PRIVATE OPENNN_CUDA)

target_include_directories(blank_cuda PRIVATE ../opennn)

target_link_libraries(blank_cuda PRIVATE opennn CUDA::cudart)