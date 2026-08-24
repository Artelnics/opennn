
project(opennn_library VERSION 8.0.1 LANGUAGES CXX)

include(GNUInstallDirs)

file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS "*.cpp")
file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS "*.h")
list(FILTER CPP_SOURCES EXCLUDE REGEX "-DESKTOP-[^/\\\\]*\\.cpp$")
list(FILTER HEADER_FILES EXCLUDE REGEX "-DESKTOP-[^/\\\\]*\\.h$")

option(OpenNN_BUILD_VISION "Build vision/seq layers, networks, and datasets" ON)
option(OpenNN_ENABLE_MKL "Use Intel MKL as Eigen BLAS/LAPACK backend" OFF)
set(OpenNN_MKL_ROOT "" CACHE PATH "Intel MKL root prefix (e.g. /opt/intel/oneapi/mkl/latest or extracted /usr prefix)")
option(OpenNN_ENABLE_ONEDNN "Use oneDNN optimized CPU recurrent primitives" OFF)
set(OpenNN_ONEDNN_ROOT "" CACHE PATH "oneDNN prefix (for example /usr or an extracted package root/usr)")

if(NOT OpenNN_BUILD_VISION)
    set(_opennn_vision_pattern
        ".*/(convolutional_layer|pooling_layer|pooling_layer_3d|normalization_layer_3d|grouped_query_attention_layer|flatten_layer|embedding_layer|multihead_attention_layer|detection_layer|detection_v8_layer|detection_v8_operator|yolo_dataset)\\.cpp$")
    list(FILTER CPP_SOURCES EXCLUDE REGEX "${_opennn_vision_pattern}")
endif()

option(OpenNN_BUILD_SHARED "Build OpenNN as a shared library (DLL) instead of static" OFF)
option(OpenNN_INSTALL "Generate install rules for the OpenNN library" ON)
if(OpenNN_BUILD_SHARED)
    set(_opennn_lib_type SHARED)

    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    set(_opennn_lib_type STATIC)
endif()

add_library(opennn ${_opennn_lib_type} ${CPP_SOURCES} ${HEADER_FILES})
add_library(OpenNN::opennn ALIAS opennn)

if(OpenNN_BUILD_SHARED)
    set_target_properties(opennn PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR})
endif()

if(OpenNN_BUILD_SHARED AND WIN32)
    set_target_properties(opennn PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

if(NOT OpenNN_BUILD_VISION)
    target_compile_definitions(opennn PUBLIC OPENNN_NO_VISION)

    set_target_properties(opennn PROPERTIES OUTPUT_NAME "opennn_core")
endif()

target_compile_features(opennn PUBLIC cxx_std_20)

include(FetchContent)

set(ZLIB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZLIB_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ZLIB_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(ZLIB_INSTALL ${OpenNN_INSTALL} CACHE BOOL "" FORCE)
FetchContent_Declare(
    opennn_zlib
    URL https://zlib.net/fossils/zlib-1.3.2.tar.gz
    URL_HASH SHA256=bb329a0a2cd0274d05519d61c667c062e06990d72e125ee2dfa8de64f0119d16
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(opennn_zlib)
target_link_libraries(opennn PUBLIC
    $<BUILD_INTERFACE:ZLIB::ZLIBSTATIC>
    $<INSTALL_INTERFACE:ZLIB::ZLIBSTATIC>)

include(ExternalProject)
set(_opennn_jpeg_install_dir
    ${CMAKE_BINARY_DIR}/_deps/opennn_libjpeg_turbo-install)
set(_opennn_jpeg_include_dir
    ${_opennn_jpeg_install_dir}/${CMAKE_INSTALL_INCLUDEDIR})
set(_opennn_jpeg_lib_dir
    ${_opennn_jpeg_install_dir}/${CMAKE_INSTALL_LIBDIR})
if(MSVC OR (WIN32 AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC"))
    set(_opennn_jpeg_library ${_opennn_jpeg_lib_dir}/jpeg-static.lib)
else()
    set(_opennn_jpeg_library
        ${_opennn_jpeg_lib_dir}/${CMAKE_STATIC_LIBRARY_PREFIX}jpeg${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()
file(MAKE_DIRECTORY ${_opennn_jpeg_include_dir} ${_opennn_jpeg_lib_dir})

set(_opennn_jpeg_cmake_args
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
    -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
    -DENABLE_SHARED=OFF
    -DENABLE_STATIC=ON
    -DWITH_JAVA=OFF
    -DWITH_SIMD=OFF
    -DWITH_TESTS=OFF
    -DWITH_TOOLS=OFF
    -DWITH_TURBOJPEG=OFF)
if(MSVC)
    list(APPEND _opennn_jpeg_cmake_args -DWITH_CRT_DLL=ON)
endif()
if(OpenNN_BUILD_SHARED)
    list(APPEND _opennn_jpeg_cmake_args -DCMAKE_POSITION_INDEPENDENT_CODE=ON)
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
    BUILD_BYPRODUCTS ${_opennn_jpeg_library})
add_library(opennn_jpeg STATIC IMPORTED GLOBAL)
add_dependencies(opennn_jpeg opennn_libjpeg_turbo_ep)
set_target_properties(opennn_jpeg PROPERTIES
    IMPORTED_LOCATION ${_opennn_jpeg_library}
    INTERFACE_INCLUDE_DIRECTORIES ${_opennn_jpeg_include_dir})
add_dependencies(opennn opennn_libjpeg_turbo_ep)
target_link_libraries(opennn PUBLIC
    $<BUILD_INTERFACE:opennn_jpeg>
    $<INSTALL_INTERFACE:libjpeg-turbo::jpeg-static>)

target_include_directories(opennn PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(opennn PUBLIC Eigen3::Eigen)

# Parallel std algorithms. MSVC implements them in its own runtime; libstdc++
# declares them (so __cpp_lib_parallel_algorithm is defined either way) but
# resolves them through TBB at link time. Only advertise them when they will
# actually link, so parallel_algorithms.h can fall back to the serial calls.
if(MSVC)
    target_compile_definitions(opennn PUBLIC OPENNN_HAS_PARALLEL_ALGORITHMS)
else()
    find_package(TBB QUIET)
    if(TBB_FOUND)
        target_compile_definitions(opennn PUBLIC OPENNN_HAS_PARALLEL_ALGORITHMS)
        target_link_libraries(opennn PUBLIC TBB::tbb)
        message(STATUS "OpenNN: parallel std algorithms enabled (TBB found).")
    else()
        message(STATUS "OpenNN: TBB not found, std algorithms run serially.")
    endif()
endif()

if(OpenNN_ENABLE_MKL)

    if(NOT OpenNN_MKL_ROOT)
        if(DEFINED ENV{MKLROOT})
            set(OpenNN_MKL_ROOT "$ENV{MKLROOT}")
        elseif(DEFINED ENV{ONEAPI_ROOT} AND EXISTS "$ENV{ONEAPI_ROOT}/mkl/latest")
            set(OpenNN_MKL_ROOT "$ENV{ONEAPI_ROOT}/mkl/latest")
        endif()
    endif()

    if(NOT DEFINED MKL_LINK)
        set(MKL_LINK sdl)
    endif()
    if(NOT DEFINED MKL_INTERFACE)
        set(MKL_INTERFACE lp64)
    endif()
    find_package(MKL CONFIG REQUIRED
        HINTS "${OpenNN_MKL_ROOT}" "${OpenNN_MKL_ROOT}/lib/cmake/mkl")
    target_compile_definitions(opennn PUBLIC EIGEN_USE_MKL_ALL)
    target_link_libraries(opennn PUBLIC MKL::MKL)
    message(STATUS "OpenNN: MKL backend via MKLConfig.cmake (${MKL_ROOT})")
endif()

if(OpenNN_ENABLE_ONEDNN)
    find_path(OpenNN_ONEDNN_INCLUDE_DIR
        NAMES oneapi/dnnl/dnnl.hpp
        HINTS "${OpenNN_ONEDNN_ROOT}/include")
    find_library(OpenNN_ONEDNN_LIBRARY
        NAMES dnnl
        HINTS
            "${OpenNN_ONEDNN_ROOT}/lib"
            "${OpenNN_ONEDNN_ROOT}/lib64"
            "${OpenNN_ONEDNN_ROOT}/lib/x86_64-linux-gnu")
    if(NOT OpenNN_ONEDNN_INCLUDE_DIR OR NOT OpenNN_ONEDNN_LIBRARY)
        message(FATAL_ERROR
            "OpenNN_ENABLE_ONEDNN=ON but oneDNN headers or library were not found. "
            "Set OpenNN_ONEDNN_ROOT to the oneDNN installation prefix.")
    endif()

    target_compile_definitions(opennn PRIVATE OPENNN_HAS_ONEDNN)
    target_include_directories(opennn SYSTEM PRIVATE "${OpenNN_ONEDNN_INCLUDE_DIR}")
    target_link_libraries(opennn PUBLIC "${OpenNN_ONEDNN_LIBRARY}")
    message(STATUS "OpenNN: oneDNN CPU recurrent backend (${OpenNN_ONEDNN_LIBRARY})")
endif()

if(APPLE)
    target_compile_options(opennn PRIVATE -Xpreprocessor -fopenmp)
elseif(OpenMP_FOUND)
    # MSVC's plain -openmp is OpenMP 2.0 and rejects `#pragma omp simd` with
    # C7660, which stops the whole build the first time a kernel uses one.
    # -openmp:experimental is the same runtime plus the OpenMP 3/4 constructs,
    # and it replaces -openmp rather than joining it - so the imported target's
    # interface flag is patched too, or the plain one would arrive after this
    # one on every command line and win.
    if(MSVC)
        target_compile_options(opennn PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-openmp:experimental>")
        set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS
            "$<$<COMPILE_LANGUAGE:CXX>:-openmp:experimental>")
    else()
        target_compile_options(opennn PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>")
    endif()

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
    target_link_libraries(opennn PUBLIC
        CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::nvrtc CUDA::cuda_driver
        CUDNN::cudnn)
    target_compile_options(opennn PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    if(MSVC)

        target_compile_options(opennn PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    endif()

    if(NOT CUDNN_FRONTEND_INCLUDE_DIR)
        message(FATAL_ERROR
            "OpenNN with OPENNN_HAS_CUDA=ON requires the cuDNN frontend headers "
            "(cudnn_frontend.h). Install from https://github.com/NVIDIA/cudnn-frontend "
            "and set CUDNN_FRONTEND_INCLUDE_DIR.")
    endif()

    target_include_directories(opennn SYSTEM PRIVATE ${CUDNN_FRONTEND_INCLUDE_DIR})

    # CUTLASS is optional and header-only. Without it the narrow-contraction
    # dense forward compiles to a stub that declines every shape and cuBLASLt
    # runs everything, which is what the library did before. With it, a first
    # layer whose contraction is at most 32 is 1.03x to 1.48x faster - see
    # opennn/core/cuda/cutlass_narrow_gemm.cu for the measurement.
    if(OpenNN_CUTLASS_INCLUDE_DIR)
        if(NOT EXISTS "${OpenNN_CUTLASS_INCLUDE_DIR}/cutlass/cutlass.h")
            message(FATAL_ERROR
                "OpenNN_CUTLASS_INCLUDE_DIR is set but cutlass/cutlass.h is not under it: "
                "${OpenNN_CUTLASS_INCLUDE_DIR}")
        endif()
        target_include_directories(opennn SYSTEM PRIVATE ${OpenNN_CUTLASS_INCLUDE_DIR})
        set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/core/cuda/cutlass_narrow_gemm.cu
                     APPEND PROPERTY COMPILE_DEFINITIONS OPENNN_HAS_CUTLASS)
    endif()

    set_property(
        SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/neural_network/operators/attention_operator.cpp
        APPEND PROPERTY COMPILE_DEFINITIONS HAVE_CUDNN_FRONTEND)

    if(OPENNN_FLASH_ATTENTION_TARGET)
        target_link_libraries(opennn PRIVATE ${OPENNN_FLASH_ATTENTION_TARGET})
        target_compile_definitions(opennn PUBLIC
            $<BUILD_INTERFACE:OPENNN_HAS_FLASH_ATTENTION>)
    endif()
endif()

if(MSVC)
    target_compile_options(opennn PUBLIC
        $<BUILD_INTERFACE:$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:DEBUG>>>:/arch:AVX2>>
        # The cudnn-frontend graph API instantiates enough templates in one
        # translation unit to pass COFF's section limit: attention_operator.cpp
        # stopped fitting at frontend v1.27.0 with "fatal error C1128". The
        # limit is an object file format detail, not a sign of anything wrong
        # with the code, and raising it costs nothing.
        $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:/bigobj>>
    )
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    target_compile_options(opennn PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-O0 -g>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:DEBUG>>>:-O3>
    )
    target_compile_options(opennn PUBLIC
        $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:-mcpu=apple-m1>>
    )
else()
    target_compile_options(opennn PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>:-O0 -g>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:DEBUG>>>:-O3>
    )
    target_compile_options(opennn PUBLIC
        $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CXX>:-march=native>>
    )
endif()

target_compile_definitions(opennn PUBLIC
    $<$<NOT:$<CONFIG:Debug>>:EIGEN_NO_DEBUG>
)

target_compile_definitions(opennn PUBLIC
    EIGEN_MAX_ALIGN_BYTES=64
    EIGEN_USE_THREADS
)

target_precompile_headers(opennn PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/pch.h>
)

if(MSVC AND OPENNN_HAS_CUDA)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/neural_network/operators/attention_operator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/neural_network/operators/batch_norm_operator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/neural_network/operators/convolution_operator.cpp
        PROPERTIES SKIP_PRECOMPILE_HEADERS ON
    )
endif()

if(NOT MSVC)
    target_compile_options(opennn PRIVATE
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:-fno-math-errno -fno-trapping-math>
    )
endif()

if(OpenNN_INSTALL)
    include(CMakePackageConfigHelpers)

    # The FlashAttention-2 kernels go with it: a static opennn carries the calls
    # into them, so an installed one needs the archive that defines them.
    install(TARGETS opennn ${OPENNN_FLASH_ATTENTION_TARGET}
        EXPORT OpenNNTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    # Headers live in folders now, and every #include spells that folder out, so
    # the installed tree has to mirror the source tree instead of being flattened.
    foreach(_opennn_header IN LISTS HEADER_FILES)
        file(RELATIVE_PATH _opennn_header_relative
            ${CMAKE_CURRENT_SOURCE_DIR} ${_opennn_header})
        get_filename_component(_opennn_header_folder ${_opennn_header_relative} DIRECTORY)
        install(FILES ${_opennn_header}
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/opennn/${_opennn_header_folder})
    endforeach()

    install(DIRECTORY ${_opennn_jpeg_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(DIRECTORY ${_opennn_jpeg_install_dir}/${CMAKE_INSTALL_LIBDIR}/
        DESTINATION ${CMAKE_INSTALL_LIBDIR})

    install(EXPORT OpenNNTargets
        FILE OpenNNTargets.cmake
        NAMESPACE OpenNN::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenNN)

    set(_opennn_needs_openmp OFF)
    if(NOT APPLE AND OpenMP_FOUND AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(_opennn_needs_openmp ON)
    endif()
    set(_opennn_needs_cudnn OFF)
    if(OPENNN_HAS_CUDA AND HAVE_CUDNN)
        set(_opennn_needs_cudnn ON)
    endif()

    set(_opennn_needs_mkl OFF)
    if(OpenNN_ENABLE_MKL AND MKL_FOUND)
        set(_opennn_needs_mkl ON)
    endif()

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/OpenNNConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/OpenNNConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenNN)

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/OpenNNConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/OpenNNConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/OpenNNConfigVersion.cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindCUDNN.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenNN)
endif()
