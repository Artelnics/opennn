# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

# Build oneDNN from source as a static library, for when no installed oneDNN
# was found. Without it the CPU LSTM runs the built-in path, 0.30x-0.66x of
# PyTorch on the benchmark cell against 1.16x-1.79x with it -- and a missing
# optional library is easy not to notice.
#
# Only the LSTM layer uses oneDNN, so the build keeps the RNN primitive and the
# reorders that pack its weights, with the training workload for the backward
# pass. It runs on OpenMP, the runtime OpenNN itself uses, so the process keeps
# one thread pool rather than two over the same cores. No GPU or graph
# component. The JIT kernels dispatch on the CPU at run time, so the archive is
# as portable as OpenNN_CPU_TARGET=PORTABLE.
#
# Defines the imported target opennn_onednn and _opennn_onednn_include_dir,
# _opennn_onednn_library and _opennn_onednn_source_dir.

include(ExternalProject)
include(GNUInstallDirs)

set(OpenNN_ONEDNN_VERSION 3.11)
set(_opennn_onednn_install_dir ${CMAKE_BINARY_DIR}/_deps/opennn_onednn-install)
set(_opennn_onednn_include_dir ${_opennn_onednn_install_dir}/${CMAKE_INSTALL_INCLUDEDIR})
set(_opennn_onednn_lib_dir ${_opennn_onednn_install_dir}/${CMAKE_INSTALL_LIBDIR})
set(_opennn_onednn_library
    ${_opennn_onednn_lib_dir}/${CMAKE_STATIC_LIBRARY_PREFIX}dnnl${CMAKE_STATIC_LIBRARY_SUFFIX})
file(MAKE_DIRECTORY ${_opennn_onednn_include_dir} ${_opennn_onednn_lib_dir})

# A Debug oneDNN is many times slower and buys nothing when debugging OpenNN,
# so it is built Release -- except under MSVC, where the runtime library must
# match the configuration of everything it is linked with.
if(MSVC)
    set(_opennn_onednn_build_type ${CMAKE_BUILD_TYPE})
else()
    set(_opennn_onednn_build_type Release)
endif()

# Same compiler as OpenNN, so a MinGW or Clang build gets a oneDNN it can link.
set(_opennn_onednn_cmake_args
    -DCMAKE_BUILD_TYPE:STRING=${_opennn_onednn_build_type}
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    -DCMAKE_INSTALL_INCLUDEDIR=${CMAKE_INSTALL_INCLUDEDIR}
    -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
    -DDNNL_LIBRARY_TYPE=STATIC
    -DDNNL_CPU_RUNTIME=OMP
    -DDNNL_GPU_RUNTIME=NONE
    -DDNNL_ENABLE_WORKLOAD=TRAINING
    -DDNNL_ENABLE_PRIMITIVE=RNN|REORDER
    -DDNNL_ENABLE_JIT_PROFILING=OFF
    -DDNNL_ENABLE_ITT_TASKS=OFF
    -DDNNL_BUILD_EXAMPLES=OFF
    -DDNNL_BUILD_TESTS=OFF
    -DDNNL_BUILD_DOC=OFF
    -DONEDNN_BUILD_GRAPH=OFF)
if(MSVC AND DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    list(APPEND _opennn_onednn_cmake_args
        "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${CMAKE_MSVC_RUNTIME_LIBRARY}")
endif()
if(OpenNN_BUILD_SHARED)
    list(APPEND _opennn_onednn_cmake_args -DCMAKE_POSITION_INDEPENDENT_CODE=ON)
endif()

ExternalProject_Add(
    opennn_onednn_ep
    URL https://github.com/uxlfoundation/oneDNN/archive/refs/tags/v${OpenNN_ONEDNN_VERSION}.tar.gz
    URL_HASH SHA256=04df98b18300daf6c3aa7cc2d5e7ce8a8f430fed1787151daed0254d8dd4e64e
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    INSTALL_DIR ${_opennn_onednn_install_dir}
    LIST_SEPARATOR |
    CMAKE_ARGS ${_opennn_onednn_cmake_args}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config $<CONFIG>
    INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --config $<CONFIG>
    BUILD_BYPRODUCTS ${_opennn_onednn_library})

ExternalProject_Get_Property(opennn_onednn_ep SOURCE_DIR)
set(_opennn_onednn_source_dir ${SOURCE_DIR})

find_package(Threads REQUIRED)
add_library(opennn_onednn STATIC IMPORTED GLOBAL)
add_dependencies(opennn_onednn opennn_onednn_ep)
set_target_properties(opennn_onednn PROPERTIES
    IMPORTED_LOCATION ${_opennn_onednn_library}
    INTERFACE_LINK_LIBRARIES "Threads::Threads;${CMAKE_DL_LIBS}")
