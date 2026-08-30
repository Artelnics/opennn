cmake_minimum_required(VERSION 3.24)

set(_valid_modes quick cpu cuda full)
if(NOT DEFINED OPENNN_VERIFY_MODE)
    set(OPENNN_VERIFY_MODE quick)
endif()
string(TOLOWER "${OPENNN_VERIFY_MODE}" OPENNN_VERIFY_MODE)
if(NOT OPENNN_VERIFY_MODE IN_LIST _valid_modes)
    message(FATAL_ERROR
        "OPENNN_VERIFY_MODE must be one of: quick, cpu, cuda, full")
endif()

if(NOT DEFINED OPENNN_VERIFY_BACKEND)
    set(OPENNN_VERIFY_BACKEND cpu)
endif()
string(TOLOWER "${OPENNN_VERIFY_BACKEND}" OPENNN_VERIFY_BACKEND)
if(NOT OPENNN_VERIFY_BACKEND STREQUAL "cpu"
   AND NOT OPENNN_VERIFY_BACKEND STREQUAL "cuda")
    message(FATAL_ERROR "OPENNN_VERIFY_BACKEND must be cpu or cuda")
endif()

if(OPENNN_VERIFY_MODE STREQUAL "quick"
   AND (NOT DEFINED OPENNN_TEST_FILTER OR OPENNN_TEST_FILTER STREQUAL ""))
    message(FATAL_ERROR
        "Quick verification requires OPENNN_TEST_FILTER so it cannot accidentally run the full suite")
endif()

get_filename_component(_source_dir "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
file(TO_CMAKE_PATH "${_source_dir}" _source_dir)
string(SHA256 _source_hash "${_source_dir}")
string(SUBSTRING "${_source_hash}" 0 10 _source_hash)

if(DEFINED OPENNN_BUILD_ROOT AND NOT OPENNN_BUILD_ROOT STREQUAL "")
    set(_cache_root "${OPENNN_BUILD_ROOT}")
elseif(DEFINED ENV{OPENNN_BUILD_ROOT} AND NOT "$ENV{OPENNN_BUILD_ROOT}" STREQUAL "")
    set(_cache_root "$ENV{OPENNN_BUILD_ROOT}")
elseif(WIN32 AND DEFINED ENV{LOCALAPPDATA} AND NOT "$ENV{LOCALAPPDATA}" STREQUAL "")
    set(_cache_root "$ENV{LOCALAPPDATA}/OpenNN/build")
elseif(DEFINED ENV{XDG_CACHE_HOME} AND NOT "$ENV{XDG_CACHE_HOME}" STREQUAL "")
    set(_cache_root "$ENV{XDG_CACHE_HOME}/opennn/build")
elseif(DEFINED ENV{HOME} AND NOT "$ENV{HOME}" STREQUAL "")
    set(_cache_root "$ENV{HOME}/.cache/opennn/build")
else()
    set(_cache_root "${_source_dir}/build/verify-cache")
endif()
file(TO_CMAKE_PATH "${_cache_root}" _cache_root)
set(_build_root "${_cache_root}/${_source_hash}")

if(NOT DEFINED OPENNN_VERIFY_RECONFIGURE)
    set(OPENNN_VERIFY_RECONFIGURE OFF)
endif()
if(NOT DEFINED OPENNN_USE_SCCACHE)
    set(OPENNN_USE_SCCACHE ON)
endif()

set(_configure_common_args)
if(OPENNN_USE_SCCACHE)
    find_program(_sccache_program sccache)
    if(_sccache_program)
        list(APPEND _configure_common_args
            "-DCMAKE_CXX_COMPILER_LAUNCHER=${_sccache_program}")
        message(STATUS "Compiler cache: ${_sccache_program}")
    else()
        message(STATUS "Compiler cache: not installed (continuing without sccache)")
    endif()
endif()

function(_prepend_path path)
    if(NOT path STREQUAL "" AND IS_DIRECTORY "${path}")
        if(WIN32)
            set(ENV{PATH} "${path};$ENV{PATH}")
        else()
            set(ENV{PATH} "${path}:$ENV{PATH}")
        endif()
    endif()
endfunction()

function(_prepare_cuda configure_args_out)
    set(_cuda_args)
    set(_cuda_toolkit_major)

    if(DEFINED ENV{OPENNN_CUDA_ARCHITECTURES}
       AND NOT "$ENV{OPENNN_CUDA_ARCHITECTURES}" STREQUAL "")
        list(APPEND _cuda_args
            "-DCMAKE_CUDA_ARCHITECTURES=$ENV{OPENNN_CUDA_ARCHITECTURES}")
    endif()

    if(_sccache_program)
        list(APPEND _cuda_args
            "-DCMAKE_CUDA_COMPILER_LAUNCHER=${_sccache_program}")
    endif()

    if(WIN32)
        file(GLOB _cuda_roots LIST_DIRECTORIES true
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*")
        if(_cuda_roots)
            list(SORT _cuda_roots COMPARE NATURAL ORDER DESCENDING)
            foreach(_cuda_root IN LISTS _cuda_roots)
                if(EXISTS "${_cuda_root}/bin/nvcc.exe")
                    _prepend_path("${_cuda_root}/bin")
                    get_filename_component(_cuda_toolkit_version "${_cuda_root}" NAME)
                    if(_cuda_toolkit_version MATCHES "^v([0-9]+)\\.")
                        set(_cuda_toolkit_major "${CMAKE_MATCH_1}")
                    endif()
                    message(STATUS "CUDA toolkit: ${_cuda_root}")
                    break()
                endif()
            endforeach()
        endif()
    endif()

    set(_has_cudnn_include FALSE)
    set(_has_cudnn_library FALSE)
    if(DEFINED ENV{OPENNN_CUDNN_INCLUDE_DIR}
       AND NOT "$ENV{OPENNN_CUDNN_INCLUDE_DIR}" STREQUAL "")
        set(_has_cudnn_include TRUE)
    endif()
    if(DEFINED ENV{OPENNN_CUDNN_LIBRARY}
       AND NOT "$ENV{OPENNN_CUDNN_LIBRARY}" STREQUAL "")
        set(_has_cudnn_library TRUE)
    endif()

    if(_has_cudnn_include OR _has_cudnn_library)
        if(NOT _has_cudnn_include OR NOT _has_cudnn_library)
            message(FATAL_ERROR
                "Set both OPENNN_CUDNN_INCLUDE_DIR and OPENNN_CUDNN_LIBRARY")
        endif()
        list(APPEND _cuda_args
            "-DCUDNN_INCLUDE_DIR=$ENV{OPENNN_CUDNN_INCLUDE_DIR}"
            "-DCUDNN_LIBRARY=$ENV{OPENNN_CUDNN_LIBRARY}")
    elseif(WIN32)
        if(_cuda_toolkit_major STREQUAL "")
            message(FATAL_ERROR
                "Automatic cuDNN selection requires a CUDA toolkit under the standard "
                "Windows install directory. For a custom toolkit, set both "
                "OPENNN_CUDNN_INCLUDE_DIR and OPENNN_CUDNN_LIBRARY.")
        endif()
        file(GLOB _cudnn_headers
            "C:/Program Files/NVIDIA/CUDNN/v*/include/*/cudnn.h")
        set(_matching_cudnn_headers)
        foreach(_candidate IN LISTS _cudnn_headers)
            get_filename_component(_candidate_include "${_candidate}" DIRECTORY)
            get_filename_component(_candidate_cuda_version
                "${_candidate_include}" NAME)
            if(_cuda_toolkit_major STREQUAL ""
               OR _candidate_cuda_version MATCHES "^${_cuda_toolkit_major}\\.")
                list(APPEND _matching_cudnn_headers "${_candidate}")
            endif()
        endforeach()

        if(_matching_cudnn_headers)
            list(SORT _matching_cudnn_headers COMPARE NATURAL ORDER DESCENDING)
            list(GET _matching_cudnn_headers 0 _cudnn_header)
            get_filename_component(_cudnn_include "${_cudnn_header}" DIRECTORY)
            get_filename_component(_cudnn_cuda_version "${_cudnn_include}" NAME)
            get_filename_component(_cudnn_include_parent "${_cudnn_include}" DIRECTORY)
            get_filename_component(_cudnn_root "${_cudnn_include_parent}" DIRECTORY)
            set(_cudnn_library
                "${_cudnn_root}/lib/${_cudnn_cuda_version}/x64/cudnn.lib")
            if(EXISTS "${_cudnn_library}")
                list(APPEND _cuda_args
                    "-DCUDNN_INCLUDE_DIR=${_cudnn_include}"
                    "-DCUDNN_LIBRARY=${_cudnn_library}")
                _prepend_path("${_cudnn_root}/bin/${_cudnn_cuda_version}/x64")
                message(STATUS "cuDNN: ${_cudnn_root} (${_cudnn_cuda_version})")
            else()
                message(FATAL_ERROR
                    "The matching cuDNN header was found, but its library is missing: "
                    "${_cudnn_library}")
            endif()
        elseif(NOT _cuda_toolkit_major STREQUAL "")
            message(FATAL_ERROR
                "No cuDNN installation matching CUDA ${_cuda_toolkit_major}.x was found. "
                "Set OPENNN_CUDNN_INCLUDE_DIR and OPENNN_CUDNN_LIBRARY explicitly.")
        endif()
    endif()

    set(${configure_args_out} ${_cuda_args} PARENT_SCOPE)
endfunction()

function(_configure backend build_dir_out)
    set(_build_dir "${_build_root}/${backend}")
    set(_cache_file "${_build_dir}/CMakeCache.txt")

    if(backend STREQUAL "cpu")
        set(_preset verify-cpu)
        set(_backend_args)
    else()
        set(_preset verify-cuda)
        _prepare_cuda(_backend_args)
    endif()

    if(OPENNN_VERIFY_RECONFIGURE OR NOT EXISTS "${_cache_file}")
        message(STATUS "Configuring ${backend}: ${_build_dir}")
        execute_process(
            COMMAND "${CMAKE_COMMAND}" --preset "${_preset}"
                -S "${_source_dir}" -B "${_build_dir}"
                ${_configure_common_args} ${_backend_args}
            WORKING_DIRECTORY "${_source_dir}"
            RESULT_VARIABLE _configure_result
            COMMAND_ECHO STDOUT)
        if(NOT _configure_result EQUAL 0)
            message(FATAL_ERROR "${backend} configure failed (${_configure_result})")
        endif()
    else()
        message(STATUS "Reusing ${backend} configuration: ${_build_dir}")
    endif()

    set(${build_dir_out} "${_build_dir}" PARENT_SCOPE)
endfunction()

function(_build backend build_dir)
    set(_build_args --build "${build_dir}" --target opennn_tests)
    if(DEFINED OPENNN_VERIFY_JOBS AND NOT OPENNN_VERIFY_JOBS STREQUAL "")
        list(APPEND _build_args --parallel "${OPENNN_VERIFY_JOBS}")
    endif()

    message(STATUS "Building ${backend} opennn_tests")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" ${_build_args}
        WORKING_DIRECTORY "${_source_dir}"
        RESULT_VARIABLE _build_result
        COMMAND_ECHO STDOUT)
    if(NOT _build_result EQUAL 0)
        message(FATAL_ERROR "${backend} build failed (${_build_result})")
    endif()
endfunction()

function(_run_tests backend build_dir focused)
    if(WIN32)
        set(_test_executable "${build_dir}/bin/opennn_tests.exe")
    else()
        set(_test_executable "${build_dir}/bin/opennn_tests")
    endif()
    if(NOT EXISTS "${_test_executable}")
        message(FATAL_ERROR "Test executable not found: ${_test_executable}")
    endif()

    set(_test_args --gtest_brief=1)
    if(focused)
        list(APPEND _test_args
            "--gtest_filter=${OPENNN_TEST_FILTER}"
            --gtest_fail_fast
            --gtest_fail_if_no_test_selected)
        message(STATUS
            "Running focused ${backend} tests: ${OPENNN_TEST_FILTER}")
    else()
        message(STATUS "Running complete ${backend} suite")
    endif()

    execute_process(
        COMMAND "${_test_executable}" ${_test_args}
        WORKING_DIRECTORY "${_source_dir}"
        RESULT_VARIABLE _test_result
        COMMAND_ECHO STDOUT)
    if(NOT _test_result EQUAL 0)
        message(FATAL_ERROR "${backend} tests failed (${_test_result})")
    endif()
endfunction()

function(_verify backend focused)
    _configure("${backend}" _build_dir)
    _build("${backend}" "${_build_dir}")
    _run_tests("${backend}" "${_build_dir}" "${focused}")
endfunction()

message(STATUS "OpenNN source: ${_source_dir}")
message(STATUS "Verification cache: ${_build_root}")

if(OPENNN_VERIFY_MODE STREQUAL "quick")
    _verify("${OPENNN_VERIFY_BACKEND}" TRUE)
elseif(OPENNN_VERIFY_MODE STREQUAL "cpu")
    _verify(cpu FALSE)
elseif(OPENNN_VERIFY_MODE STREQUAL "cuda")
    _verify(cuda FALSE)
else()
    _verify(cpu FALSE)
    _verify(cuda FALSE)
endif()
