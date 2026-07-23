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

if(CUDNN_INCLUDE_DIR)
    set(_cudnn_version_header "${CUDNN_INCLUDE_DIR}/cudnn_version.h")
    if(NOT EXISTS "${_cudnn_version_header}")
        set(_cudnn_version_header "${CUDNN_INCLUDE_DIR}/cudnn.h")
    endif()
    if(EXISTS "${_cudnn_version_header}")
        file(STRINGS "${_cudnn_version_header}" _cudnn_version_lines
            REGEX "^#define[ \t]+CUDNN_(MAJOR|MINOR|PATCHLEVEL)[ \t]+[0-9]+$")
        foreach(line ${_cudnn_version_lines})
            if(line MATCHES "CUDNN_MAJOR[ \t]+([0-9]+)")
                set(_cudnn_major "${CMAKE_MATCH_1}")
            elseif(line MATCHES "CUDNN_MINOR[ \t]+([0-9]+)")
                set(_cudnn_minor "${CMAKE_MATCH_1}")
            elseif(line MATCHES "CUDNN_PATCHLEVEL[ \t]+([0-9]+)")
                set(_cudnn_patch "${CMAKE_MATCH_1}")
            endif()
        endforeach()
        if(DEFINED _cudnn_major)
            set(CUDNN_VERSION "${_cudnn_major}.${_cudnn_minor}.${_cudnn_patch}")
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
