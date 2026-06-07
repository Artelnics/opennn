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

# A static lib is an archive of every TU's objects, so the heavy Eigen / STL /
# cuDNN-frontend template instantiations are duplicated across all 72 objects
# (the linker only dedups them when building an exe/dll). A shared lib dedups
# at link time into one much smaller binary. Default OFF to keep the proven
# static build; enable with -DOpenNN_BUILD_SHARED=ON.
option(OpenNN_BUILD_SHARED "Build OpenNN as a shared library (DLL) instead of static" OFF)
if(OpenNN_BUILD_SHARED)
    set(_opennn_lib_type SHARED)
else()
    set(_opennn_lib_type STATIC)
endif()

add_library(opennn ${_opennn_lib_type} ${CPP_SOURCES} ${HEADER_FILES})

# The headers carry no __declspec(dllexport), so let CMake auto-generate the
# export table from every symbol when building the DLL on Windows.
if(OpenNN_BUILD_SHARED AND WIN32)
    set_target_properties(opennn PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(NOT OpenNN_BUILD_VISION)
    target_compile_definitions(opennn PUBLIC OPENNN_NO_VISION)
    # Distinct output name so the slim lib does not clash with the full
    # opennn.lib (which may be locked by another VS process on the same
    # machine). Consumers of the slim build should point their CMake
    # OPENNN_LIB at .../Release/opennn_core.lib.
    set_target_properties(opennn PROPERTIES OUTPUT_NAME "opennn_core")
endif()

target_compile_features(opennn PUBLIC cxx_std_20)

if(OpenNN_BUILD_VISION)
    include(FetchContent)

    # ZLIB is only needed by the PNG support in image_dataset /
    # image_utilities, which are excluded when the slim opennn_core lib is
    # requested. Keep system packages preferred, but fetch a pinned fallback so
    # Windows/CUDA builds do not require a separate zlib install.
    find_package(ZLIB QUIET)
    if(ZLIB_FOUND)
        target_include_directories(opennn PUBLIC ${ZLIB_INCLUDE_DIRS})
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            # Avoid imported-target absolute paths that can trip GNU make when
            # sysroot files have epoch-zero mtimes.
            target_link_libraries(opennn PUBLIC z)
        elseif(TARGET ZLIB::ZLIB)
            target_link_libraries(opennn PUBLIC ZLIB::ZLIB)
        else()
            target_link_libraries(opennn PUBLIC ${ZLIB_LIBRARIES})
        endif()
    else()
        FetchContent_Declare(
            opennn_zlib
            URL https://zlib.net/fossils/zlib-1.3.2.tar.gz
            URL_HASH SHA256=bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(opennn_zlib)
        target_include_directories(opennn SYSTEM PUBLIC
            ${opennn_zlib_SOURCE_DIR}
            ${opennn_zlib_BINARY_DIR})
        target_link_libraries(opennn PUBLIC zlibstatic)
    endif()

    find_package(JPEG QUIET)
    if(JPEG_FOUND)
        target_include_directories(opennn PUBLIC ${JPEG_INCLUDE_DIRS})
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_link_libraries(opennn PUBLIC jpeg)
        elseif(TARGET JPEG::JPEG)
            target_link_libraries(opennn PUBLIC JPEG::JPEG)
        else()
            target_link_libraries(opennn PUBLIC ${JPEG_LIBRARIES})
        endif()
    else()
        include(ExternalProject)
        set(_opennn_jpeg_install_dir
            ${CMAKE_BINARY_DIR}/_deps/opennn_libjpeg_turbo-install)
        set(_opennn_jpeg_include_dir
            ${_opennn_jpeg_install_dir}/${CMAKE_INSTALL_INCLUDEDIR})
        set(_opennn_jpeg_lib_dir
            ${_opennn_jpeg_install_dir}/${CMAKE_INSTALL_LIBDIR})
        if(MSVC)
            set(_opennn_jpeg_library ${_opennn_jpeg_lib_dir}/jpeg-static.lib)
        else()
            set(_opennn_jpeg_library
                ${_opennn_jpeg_lib_dir}/${CMAKE_STATIC_LIBRARY_PREFIX}jpeg${CMAKE_STATIC_LIBRARY_SUFFIX})
        endif()

        file(MAKE_DIRECTORY ${_opennn_jpeg_include_dir} ${_opennn_jpeg_lib_dir})

        set(_opennn_jpeg_cmake_args
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            -DENABLE_SHARED=OFF
            -DENABLE_STATIC=ON
            -DWITH_JAVA=OFF
            -DWITH_SIMD=OFF
            -DWITH_TESTS=OFF
            -DWITH_TURBOJPEG=OFF)

        if(MSVC)
            list(APPEND _opennn_jpeg_cmake_args -DWITH_CRT_DLL=ON)
        endif()

        ExternalProject_Add(
            opennn_libjpeg_turbo_ep
            URL https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/3.1.4.tar.gz
            URL_HASH SHA256=3a581ad5a3060cee69487aae0bc71a1f3be94b89f92215e32b43742ec4a5055c
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            INSTALL_DIR ${_opennn_jpeg_install_dir}
            CMAKE_ARGS ${_opennn_jpeg_cmake_args}
            BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config $<CONFIG>
            INSTALL_COMMAND
                ${CMAKE_COMMAND} --install <BINARY_DIR> --config $<CONFIG> --component include
                COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --config $<CONFIG> --component lib
            BUILD_BYPRODUCTS ${_opennn_jpeg_library}
        )
        add_library(opennn_jpeg STATIC IMPORTED GLOBAL)
        add_dependencies(opennn_jpeg opennn_libjpeg_turbo_ep)
        set_target_properties(opennn_jpeg PROPERTIES
            IMPORTED_LOCATION ${_opennn_jpeg_library})
        add_dependencies(opennn opennn_libjpeg_turbo_ep)
        target_include_directories(opennn SYSTEM PUBLIC ${_opennn_jpeg_include_dir})
        target_link_libraries(opennn PUBLIC opennn_jpeg)
    endif()
endif()

# stdc++fs is only needed on GCC < 9; modern toolchains have std::filesystem
# in libstdc++ itself and the link library may not exist in the sysroot.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(opennn PUBLIC stdc++fs)
endif()

# libstdc++'s parallel algorithms (std::execution::par, used in image_dataset.cpp)
# are implemented on top of Intel TBB, so GCC/Clang builds must link it. MSVC's
# parallel STL has no such dependency. PUBLIC so consumers inherit the link.
if(NOT MSVC)
    find_package(TBB QUIET)
    if(TBB_FOUND)
        target_link_libraries(opennn PUBLIC TBB::tbb)
    else()
        target_link_libraries(opennn PUBLIC tbb)
    endif()
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

    find_path(CUDNN_FRONTEND_INCLUDE_DIR NAMES cudnn_frontend.h)
    if(NOT CUDNN_FRONTEND_INCLUDE_DIR)
        message(FATAL_ERROR
            "OpenNN with OPENNN_HAS_CUDA=ON requires the cuDNN frontend headers "
            "(cudnn_frontend.h). Install from https://github.com/NVIDIA/cudnn-frontend "
            "and set CUDNN_FRONTEND_INCLUDE_DIR.")
    endif()
    # SYSTEM include so GCC suppresses third-party warnings (-Wswitch-enum,
    # -Wunused-parameter, etc.) from inside cudnn-frontend headers.
    target_include_directories(opennn SYSTEM PUBLIC ${CUDNN_FRONTEND_INCLUDE_DIR})
    # operators.cpp guards SDPA / Flash Attention with HAVE_CUDNN_FRONTEND.
    # The find_path above already errors out if the header is missing, so when
    # we get here it is always available — define the macro so the SDPA path
    # is actually compiled in the CMake build (qmake sets it in cuda.pri).

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
