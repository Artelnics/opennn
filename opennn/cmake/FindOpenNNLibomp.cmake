# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

# Locate the macOS OpenMP runtime for both builds and installed consumers.
# Override LIBOMP_INCLUDE_DIR/LIBOMP_LIBRARY or use CMAKE_PREFIX_PATH.
# AppleClang has no OpenMP runtime of its own, so -Xpreprocessor -fopenmp
# needs LLVM's libomp. Homebrew installs it keg-only, outside its main
# include and lib directories. The two keg prefixes (Apple Silicon, then
# Intel) go in PATHS, which CMake searches after all of its own locations:
# a libomp exposed through CMAKE_PREFIX_PATH or the environment still wins,
# and CMAKE_FIND_ROOT_PATH applies to the kegs like any other extra path.
set(_opennn_libomp_kegs /opt/homebrew/opt/libomp /usr/local/opt/libomp)
set(_opennn_libomp_include_dirs ${_opennn_libomp_kegs})
list(TRANSFORM _opennn_libomp_include_dirs APPEND /include)
set(_opennn_libomp_library_dirs ${_opennn_libomp_kegs})
list(TRANSFORM _opennn_libomp_library_dirs APPEND /lib)

find_path(LIBOMP_INCLUDE_DIR
    NAMES omp.h
    PATHS ${_opennn_libomp_include_dirs})
find_library(LIBOMP_LIBRARY
    NAMES omp
    PATHS ${_opennn_libomp_library_dirs})

# Last resort: a file named exactly libomp.dylib, even when the prefix or
# suffix lists have been customised so that they no longer form that name.
# The same search runs again with the lists narrowed to lib and .dylib, so
# this candidate ranks below every regular match in every location. The
# lists are restored afterwards, including whether they were defined.
if("${LIBOMP_LIBRARY}" MATCHES "(^|-)NOTFOUND$")
    foreach(_opennn_libomp_list IN ITEMS PREFIXES SUFFIXES)
        if(DEFINED CMAKE_FIND_LIBRARY_${_opennn_libomp_list})
            set(_opennn_libomp_defined_${_opennn_libomp_list} ON)
            set(_opennn_libomp_saved_${_opennn_libomp_list}
                "${CMAKE_FIND_LIBRARY_${_opennn_libomp_list}}")
        else()
            set(_opennn_libomp_defined_${_opennn_libomp_list} OFF)
        endif()
    endforeach()

    set(CMAKE_FIND_LIBRARY_PREFIXES lib)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .dylib)
    find_library(LIBOMP_LIBRARY
        NAMES omp
        PATHS ${_opennn_libomp_library_dirs})

    foreach(_opennn_libomp_list IN ITEMS PREFIXES SUFFIXES)
        if(_opennn_libomp_defined_${_opennn_libomp_list})
            set(CMAKE_FIND_LIBRARY_${_opennn_libomp_list}
                "${_opennn_libomp_saved_${_opennn_libomp_list}}")
        else()
            unset(CMAKE_FIND_LIBRARY_${_opennn_libomp_list})
        endif()
        unset(_opennn_libomp_defined_${_opennn_libomp_list})
        unset(_opennn_libomp_saved_${_opennn_libomp_list})
    endforeach()
    unset(_opennn_libomp_list)
endif()

unset(_opennn_libomp_kegs)
unset(_opennn_libomp_include_dirs)
unset(_opennn_libomp_library_dirs)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenNNLibomp
    REQUIRED_VARS LIBOMP_INCLUDE_DIR LIBOMP_LIBRARY
    REASON_FAILURE_MESSAGE
        "Install libomp or set LIBOMP_INCLUDE_DIR and LIBOMP_LIBRARY to its installation.")

if(OpenNNLibomp_FOUND AND NOT TARGET OpenNN::libomp)
    add_library(OpenNN::libomp UNKNOWN IMPORTED)
    set_target_properties(OpenNN::libomp PROPERTIES
        IMPORTED_LOCATION "${LIBOMP_LIBRARY}")
endif()

mark_as_advanced(LIBOMP_INCLUDE_DIR LIBOMP_LIBRARY)
