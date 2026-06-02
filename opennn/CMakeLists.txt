cmake_minimum_required(VERSION 3.18)

project(opennn_library LANGUAGES CXX)

file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS "*.cpp")
file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS "*.h")

# Vision module: YOLO, ResNet, ImageClassification, Transformer, and their
# layers (convolutional, pooling, flatten, embedding, multihead-attention,
# normalization_3d, detection, transformer_decoder) plus image/yolo
# datasets. Default ON so colleagues building the full library are
# unaffected; set OFF (with -DOpenNN_BUILD_VISION=OFF) for tabular-only
# users who hit MSVC link.exe LNK1327 against the full vision lib.
option(OpenNN_BUILD_VISION "Build vision/seq layers, networks, and datasets" ON)
option(OpenNN_ENABLE_MKL "Use Intel MKL as Eigen BLAS/LAPACK backend" OFF)
set(OpenNN_MKL_ROOT "" CACHE PATH "Intel MKL root prefix (e.g. /opt/intel/oneapi/mkl/latest or extracted /usr prefix)")

if(NOT OpenNN_BUILD_VISION)
    set(_opennn_vision_pattern
        ".*/(convolutional_layer|pooling_layer|pooling_layer_3d|normalization_layer_3d|flatten_layer|embedding_layer|multihead_attention_layer|detection_layer|transformer_decoder|image_dataset|image_utilities|yolo_dataset)\\.cpp$")
    list(FILTER CPP_SOURCES EXCLUDE REGEX "${_opennn_vision_pattern}")
endif()

add_library(opennn STATIC ${CPP_SOURCES} ${HEADER_FILES})

if(NOT OpenNN_BUILD_VISION)
    target_compile_definitions(opennn PUBLIC OPENNN_NO_VISION)
    # Distinct output name so the slim lib does not clash with the full
    # opennn.lib (which may be locked by another VS process on the same
    # machine). Consumers of the slim build should point their CMake
    # OPENNN_LIB at .../Release/opennn_core.lib.
    set_target_properties(opennn PROPERTIES OUTPUT_NAME "opennn_core")
endif()

target_compile_features(opennn PUBLIC cxx_std_20)

# ZLIB is only needed by the PNG support in image_dataset / image_utilities,
# which are themselves excluded when the slim opennn_core lib is requested.
# Make the dependency conditional so tabular-only builds (which is how the
# IDC_benchmark and baseline_trainer consumers wire opennn in) do not need
# a system ZLIB install. Use -l<name> instead of the ZLIB::ZLIB imported
# target so Flatpak-host .so files with epoch-zero mtimes don't make GNU
# make fail with "No rule to make target" (same workaround as JPEG below).
if(OpenNN_BUILD_VISION)
    find_package(ZLIB REQUIRED)
    target_link_libraries(opennn PUBLIC z)
endif()

if(OpenNN_BUILD_VISION)
    find_package(JPEG REQUIRED)
    target_link_libraries(opennn PUBLIC jpeg)
endif()

# stdc++fs is only needed on GCC < 9; modern toolchains have std::filesystem
# in libstdc++ itself and the link library may not exist in the sysroot.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(opennn PUBLIC stdc++fs)
endif()

target_include_directories(opennn PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(opennn PUBLIC Eigen3::Eigen)

if(OpenNN_ENABLE_MKL)
    if(OpenNN_MKL_ROOT)
        set(_opennn_mkl_include "${OpenNN_MKL_ROOT}/include")
        if(EXISTS "${_opennn_mkl_include}/mkl/mkl.h")
            set(_opennn_mkl_include "${_opennn_mkl_include}/mkl")
        endif()
        set(_opennn_mkl_lib_dir "${OpenNN_MKL_ROOT}/lib")
        if(EXISTS "${OpenNN_MKL_ROOT}/lib/x86_64-linux-gnu/libmkl_rt.so")
            set(_opennn_mkl_lib_dir "${OpenNN_MKL_ROOT}/lib/x86_64-linux-gnu")
        endif()
    else()
        set(_opennn_mkl_include "/usr/include/mkl")
        set(_opennn_mkl_lib_dir "/usr/lib/x86_64-linux-gnu")
    endif()

    if(NOT EXISTS "${_opennn_mkl_include}/mkl.h")
        message(FATAL_ERROR "OpenNN_ENABLE_MKL=ON but mkl.h was not found. Set OpenNN_MKL_ROOT.")
    endif()
    if(NOT EXISTS "${_opennn_mkl_lib_dir}/libmkl_rt.so")
        message(FATAL_ERROR "OpenNN_ENABLE_MKL=ON but libmkl_rt.so was not found. Set OpenNN_MKL_ROOT.")
    endif()

    target_compile_definitions(opennn PUBLIC EIGEN_USE_MKL_ALL)
    target_include_directories(opennn SYSTEM PUBLIC "${_opennn_mkl_include}")
    target_link_libraries(opennn PUBLIC "${_opennn_mkl_lib_dir}/libmkl_rt.so" pthread m dl)
    target_link_options(opennn PUBLIC "-Wl,-rpath,${_opennn_mkl_lib_dir}")
endif()

if(APPLE)
    target_compile_options(opennn PRIVATE -Xpreprocessor -fopenmp)
elseif(OpenMP_FOUND)
    target_compile_options(opennn PRIVATE ${OpenMP_CXX_FLAGS})
    # Same epoch-zero mtime workaround as ZLib/JPEG.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_options(opennn PUBLIC -fopenmp)
        target_link_libraries(opennn PUBLIC gomp pthread)
    else()
        target_link_libraries(opennn PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
    target_compile_options(opennn PRIVATE -Wno-interference-size)
endif()

target_compile_options(opennn PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -Wswitch-enum>
    $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4 /w14061>
)

if(OPENNN_HAS_CUDA)
    target_compile_definitions(opennn PUBLIC OPENNN_HAS_CUDA)
    file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS "*.cu" "*.cuh")
    target_sources(opennn PRIVATE ${CUDA_SOURCES})
    target_link_libraries(opennn PUBLIC CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::nvrtc CUDA::cuda_driver)
    target_compile_options(opennn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    if(HAVE_CUDNN)
        target_link_libraries(opennn PUBLIC ${CUDNN_LIBRARY})
        target_include_directories(opennn PUBLIC ${CUDNN_INCLUDE_DIR})
    endif()
    # CUDNN_FRONTEND_INCLUDE_DIR is set by the root CMakeLists.txt's
    # FetchContent_MakeAvailable(cudnn_frontend) block, so it is always present here.
    target_include_directories(opennn PUBLIC ${CUDNN_FRONTEND_INCLUDE_DIR})
    target_compile_definitions(opennn PUBLIC HAVE_CUDNN_FRONTEND)
endif()

# Optimization & SIMD. Without host-native ISA flags Eigen falls back to the x86-64
# baseline (SSE2), costing 2-4x on matmul-heavy workloads. Mirrors opennn.pro.
# Guarded to CXX so nvcc is unaffected — it forwards its own host flags separately.
if(MSVC)
    target_compile_options(opennn PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:DEBUG>>>:/arch:AVX2>
    )
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    target_compile_options(opennn PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-O0 -g>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:DEBUG>>>:-O3 -mcpu=apple-m1>
    )
else()
    target_compile_options(opennn PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-O0 -g>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:DEBUG>>>:-O3 -march=native>
    )
endif()

# Disable Eigen's internal shape/index/aliasing assertions in release builds.
# NDEBUG (auto-added by CMake for non-Debug configs) only silences <cassert>;
# Eigen has its own eigen_assert macro controlled separately. PUBLIC so any
# consumer of the library sees the same Eigen behavior.
target_compile_definitions(opennn PUBLIC
    $<$<NOT:$<CONFIG:Debug>>:EIGEN_NO_DEBUG>
)

# Precompiled header for C++ TUs only (CUDA .cu files compile with nvcc and
# don't reliably reuse host PCH). pch.h pulls in CUDA / cuDNN / Eigen / STL.
target_precompile_headers(opennn PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/pch.h>
)

# operators.cpp pulls in cudnn-frontend, which has names (`half`, `bfloat16`)
# that collide with Eigen names brought into global scope by pch.h's
# `using namespace Eigen`. Disable PCH on this TU so cudnn-frontend is parsed
# in a clean scope and the file picks up Eigen/std the regular way.
if(OPENNN_HAS_CUDA)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/operators.cpp
        PROPERTIES SKIP_PRECOMPILE_HEADERS ON
    )
endif()

# Safe subset of -ffast-math: lets sqrt/exp/log/sin/cos vectorize without
# preserving errno or trapping FP exceptions. Excludes -ffinite-math-only
# (would break NaN handling in correlations.cpp / statistics.cpp) and
# -funsafe-math-optimizations (would reorder reductions).
if(NOT MSVC)
    target_compile_options(opennn PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:-fno-math-errno -fno-trapping-math>
    )
endif()
