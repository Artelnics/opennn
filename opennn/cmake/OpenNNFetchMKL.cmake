# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

# Download Intel oneMKL's static libraries, for OpenNN_ENABLE_MKL=ON builds that
# were given no installation. Intel publishes oneMKL only as binaries; its PyPI
# wheels are versioned zip archives with a published SHA-256, so they can be
# pinned like any other fetched dependency.
#
# Linked statically with the GNU OpenMP threading layer: nothing has to ship
# beside the program, and MKL's threads come from the same libgomp pool as
# OpenNN's and oneDNN's. The linker keeps only the kernels the program calls.
#
# Defines the imported target opennn_mkl and _opennn_mkl_include_dir,
# _opennn_mkl_lib_dir, _opennn_mkl_parts and _opennn_mkl_license.

include(FetchContent)

set(OpenNN_MKL_VERSION 2025.3.1)
set(_opennn_mkl_pypi https://files.pythonhosted.org/packages)

# A .whl is a zip archive; the download is named so the extract step knows it.
FetchContent_Declare(
    opennn_mkl_static
    URL ${_opennn_mkl_pypi}/6c/ca/3f030ea6a9690f7b9918961d1a8cb9854ae2b716c4b40745d3f8d555cfa5/mkl_static-${OpenNN_MKL_VERSION}-py2.py3-none-manylinux_2_28_x86_64.whl
    URL_HASH SHA256=a8f13291243aa78bae7c2d38092ae34c99cc133fcd426beca2421cc44517f355
    DOWNLOAD_NAME mkl_static-${OpenNN_MKL_VERSION}.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_Declare(
    opennn_mkl_include
    URL ${_opennn_mkl_pypi}/a6/43/7c107c3b05da5590797d34a48666b235b05265398feb48984a14dd1b458b/mkl_include-${OpenNN_MKL_VERSION}-py2.py3-none-manylinux_2_28_x86_64.whl
    URL_HASH SHA256=b81b08dfd64901a398bd1a751783e9a4de53d9e2ba2e019438dab9e3f8c7e1ea
    DOWNLOAD_NAME mkl_include-${OpenNN_MKL_VERSION}.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(opennn_mkl_static opennn_mkl_include)

set(_opennn_mkl_lib_dir
    ${opennn_mkl_static_SOURCE_DIR}/mkl_static-${OpenNN_MKL_VERSION}.data/data/lib)
set(_opennn_mkl_include_dir
    ${opennn_mkl_include_SOURCE_DIR}/mkl_include-${OpenNN_MKL_VERSION}.data/data/include)
set(_opennn_mkl_license
    ${opennn_mkl_static_SOURCE_DIR}/mkl_static-${OpenNN_MKL_VERSION}.dist-info/LICENSE.txt)

# LP64 interface, GNU OpenMP threading, core: the three archives depend on one
# another in a cycle, so they are linked as a group the linker rescans.
set(_opennn_mkl_parts intel_lp64 gnu_thread core)
set(_opennn_mkl_group "")
foreach(part ${_opennn_mkl_parts})
    set(library ${_opennn_mkl_lib_dir}/libmkl_${part}.a)
    if(NOT EXISTS ${library})
        message(FATAL_ERROR "OpenNN: the oneMKL ${OpenNN_MKL_VERSION} wheel has no ${library}")
    endif()
    add_library(opennn_mkl_${part} STATIC IMPORTED GLOBAL)
    set_target_properties(opennn_mkl_${part} PROPERTIES IMPORTED_LOCATION ${library})
    list(APPEND _opennn_mkl_group opennn_mkl_${part})
endforeach()
list(JOIN _opennn_mkl_group "," _opennn_mkl_group)

find_package(Threads REQUIRED)
add_library(opennn_mkl INTERFACE IMPORTED GLOBAL)
set_target_properties(opennn_mkl PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${_opennn_mkl_include_dir}
    INTERFACE_LINK_LIBRARIES
        "$<LINK_GROUP:RESCAN,${_opennn_mkl_group}>;Threads::Threads;m;${CMAKE_DL_LIBS}")
