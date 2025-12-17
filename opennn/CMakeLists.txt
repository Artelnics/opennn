cmake_minimum_required(VERSION 3.18)

project(opennn_library LANGUAGES CXX)

file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS "*.cpp")
file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS "*.h")

add_library(opennn STATIC ${CPP_SOURCES} ${HEADER_FILES})

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(opennn PUBLIC stdc++fs)
endif()

target_include_directories(opennn PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

if(APPLE)
    target_compile_options(opennn PRIVATE -Xpreprocessor -fopenmp)
elseif(OpenMP_FOUND)
    target_compile_options(opennn PRIVATE ${OpenMP_CXX_FLAGS})
    target_link_libraries(opennn PUBLIC OpenMP::OpenMP_CXX)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(opennn PRIVATE -Wno-interference-size)
endif()

if(OPENNN_WITH_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "*.cu" "*.cuh")
    target_sources(opennn PRIVATE ${CUDA_SOURCES})
    target_link_libraries(opennn PUBLIC CUDA::cudart CUDA::cublas)
    target_compile_options(opennn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    if(HAVE_CUDNN)
        target_link_libraries(opennn PUBLIC ${CUDNN_LIBRARY})
        target_include_directories(opennn PUBLIC ${CUDNN_INCLUDE_DIR})
    endif()
endif()

target_compile_options(opennn PRIVATE
    $<$<CONFIG:DEBUG>:-O0 -g>
    $<$<NOT:$<CONFIG:DEBUG>>:-O2>
)