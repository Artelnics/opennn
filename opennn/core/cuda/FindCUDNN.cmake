# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

# FindCUDNN
# ---------
# Locates the NVIDIA cuDNN library and headers and defines the imported
# target CUDNN::cudnn. Installed with the OpenNN package so that
# OpenNNConfig.cmake can find_dependency(CUDNN) on consumer machines.
#
# Result variables: CUDNN_FOUND, CUDNN_INCLUDE_DIR, CUDNN_LIBRARY,
# CUDNN_VERSION.

find_path(CUDNN_INCLUDE_DIR
    NAMES cudnn_version.h cudnn.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS}
    PATHS /usr/local/cuda/include /usr/include)

find_library(CUDNN_LIBRARY
    NAMES cudnn
    HINTS ${CUDAToolkit_LIBRARY_DIR}
    PATHS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)

# The "major.minor.patch" of the cuDNN headers in `dir`, or empty.
function(_opennn_cudnn_header_version dir out_var)
    set(${out_var} "" PARENT_SCOPE)
    set(header "${dir}/cudnn_version.h")
    if(NOT EXISTS "${header}")
        set(header "${dir}/cudnn.h")
    endif()
    if(NOT EXISTS "${header}")
        return()
    endif()
    file(STRINGS "${header}" lines
        REGEX "^#define[ \t]+CUDNN_(MAJOR|MINOR|PATCHLEVEL)[ \t]+[0-9]+$")
    foreach(line ${lines})
        if(line MATCHES "CUDNN_MAJOR[ \t]+([0-9]+)")
            set(major "${CMAKE_MATCH_1}")
        elseif(line MATCHES "CUDNN_MINOR[ \t]+([0-9]+)")
            set(minor "${CMAKE_MATCH_1}")
        elseif(line MATCHES "CUDNN_PATCHLEVEL[ \t]+([0-9]+)")
            set(patch "${CMAKE_MATCH_1}")
        endif()
    endforeach()
    if(DEFINED major)
        set(${out_var} "${major}.${minor}.${patch}" PARENT_SCOPE)
    endif()
endfunction()

if(CUDNN_INCLUDE_DIR)
    _opennn_cudnn_header_version("${CUDNN_INCLUDE_DIR}" CUDNN_VERSION)
    if(NOT CUDNN_VERSION)
        unset(CUDNN_VERSION)
    endif()
endif()

# Two ways to compile against one cuDNN and run another without the build
# saying so. Headers inside the CUDA toolkit can be found before
# CUDNN_INCLUDE_DIR, since a directory already on the path as a system include
# is not moved by a later -I. And a library can come from another release than
# the headers. Either one switches off code paths gated on the cuDNN version.
if(CUDNN_VERSION)
    foreach(_cudnn_toolkit_dir ${CUDAToolkit_INCLUDE_DIRS})
        if(NOT _cudnn_toolkit_dir STREQUAL CUDNN_INCLUDE_DIR)
            _opennn_cudnn_header_version("${_cudnn_toolkit_dir}" _cudnn_toolkit_version)
            if(_cudnn_toolkit_version AND NOT _cudnn_toolkit_version VERSION_EQUAL CUDNN_VERSION)
                message(WARNING
                    "cuDNN ${_cudnn_toolkit_version} headers in the CUDA toolkit "
                    "(${_cudnn_toolkit_dir}) can shadow CUDNN_INCLUDE_DIR (cuDNN "
                    "${CUDNN_VERSION}, ${CUDNN_INCLUDE_DIR}): the code would compile "
                    "against one and run the other. Install one cuDNN, or point "
                    "CUDNN_INCLUDE_DIR and CUDNN_LIBRARY at the toolkit's.")
            endif()
        endif()
    endforeach()
    if(CUDNN_LIBRARY)
        get_filename_component(_cudnn_library_real "${CUDNN_LIBRARY}" REALPATH)
        if(_cudnn_library_real MATCHES "cudnn[^/]*\\.so\\.([0-9]+\\.[0-9]+\\.[0-9]+)$"
           AND NOT CMAKE_MATCH_1 VERSION_EQUAL CUDNN_VERSION)
            message(WARNING
                "cuDNN library ${_cudnn_library_real} is ${CMAKE_MATCH_1}, but the "
                "headers in ${CUDNN_INCLUDE_DIR} are ${CUDNN_VERSION}.")
        endif()
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN
    REQUIRED_VARS CUDNN_LIBRARY CUDNN_INCLUDE_DIR
    VERSION_VAR CUDNN_VERSION)

if(CUDNN_FOUND AND NOT TARGET CUDNN::cudnn)
    add_library(CUDNN::cudnn UNKNOWN IMPORTED)
    set_target_properties(CUDNN::cudnn PROPERTIES
        IMPORTED_LOCATION "${CUDNN_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}")
endif()

mark_as_advanced(CUDNN_INCLUDE_DIR CUDNN_LIBRARY)
