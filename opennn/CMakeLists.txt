cmake_minimum_required(VERSION 3.18)

project(opennn_library VERSION 8.0.1 LANGUAGES CXX)

include(GNUInstallDirs)

file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS "*.cpp")
file(GLOB_RECURSE HEADER_FILES CONFIGURE_DEPENDS "*.h")

# Vision module: YOLO, ResNet, ImageClassification, Transformer, and their
# layers (convolutional, pooling, flatten, embedding, multihead-attention,
# normalization_3d, detection) plus yolo_dataset.
# image_dataset and image_processing are always compiled so tabular pipelines
# that feed images can use ImageDataset without a full vision build. Default ON;
# set OFF (with -DOpenNN_BUILD_VISION=OFF) for tabular-only users who hit MSVC
# link.exe LNK1327 against the full vision lib.
option(OpenNN_BUILD_VISION "Build vision/seq layers, networks, and datasets" ON)
option(OpenNN_ENABLE_MKL "Use Intel MKL as Eigen BLAS/LAPACK backend" OFF)
set(OpenNN_MKL_ROOT "" CACHE PATH "Intel MKL root prefix (e.g. /opt/intel/oneapi/mkl/latest or extracted /usr prefix)")

if(NOT OpenNN_BUILD_VISION)
    set(_opennn_vision_pattern
        ".*/(convolutional_layer|pooling_layer|pooling_layer_3d|normalization_layer_3d|flatten_layer|embedding_layer|multihead_attention_layer|detection_layer|yolo_dataset)\\.cpp$")
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
    # Fallback deps (fetched zlib/libjpeg static archives) get linked into the
    # shared object, so they must be built as position-independent code too.
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

include(FetchContent)

# image_dataset and image_processing are always compiled; they need ZLIB (PNG)
# and JPEG. Keep system packages preferred, but fetch pinned fallbacks so
# Windows/CUDA builds do not require a separate install.
set(_opennn_needs_zlib_dependency OFF)
set(_opennn_needs_jpeg_dependency OFF)
set(_opennn_install_zlib_fallback OFF)
set(_opennn_install_jpeg_fallback OFF)

find_package(ZLIB QUIET)
    if(ZLIB_FOUND)
        target_include_directories(opennn PRIVATE ${ZLIB_INCLUDE_DIRS})
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            # Avoid imported-target absolute paths that can trip GNU make when
            # sysroot files have epoch-zero mtimes.
            target_link_libraries(opennn PUBLIC z)
        elseif(TARGET ZLIB::ZLIB)
            target_link_libraries(opennn PUBLIC ZLIB::ZLIB)
            set(_opennn_needs_zlib_dependency ON)
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
        # The fetched fallback lives in the build tree, so the install rules
        # copy its static archive (renamed to avoid clashing with a system
        # zlib) into the prefix and the install interface points at that copy.
        set(_opennn_zlib_installed_name
            ${CMAKE_STATIC_LIBRARY_PREFIX}opennn_zlib${CMAKE_STATIC_LIBRARY_SUFFIX})
        target_include_directories(opennn SYSTEM PRIVATE
            ${opennn_zlib_SOURCE_DIR}
            ${opennn_zlib_BINARY_DIR})
        target_link_libraries(opennn PUBLIC
            $<BUILD_INTERFACE:zlibstatic>
            $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBDIR}/${_opennn_zlib_installed_name}>)
        set(_opennn_install_zlib_fallback ON)
    endif()

    find_package(JPEG QUIET)
    if(JPEG_FOUND)
        target_include_directories(opennn PRIVATE ${JPEG_INCLUDE_DIRS})
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_link_libraries(opennn PUBLIC jpeg)
        elseif(TARGET JPEG::JPEG)
            target_link_libraries(opennn PUBLIC JPEG::JPEG)
            set(_opennn_needs_jpeg_dependency ON)
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
            BUILD_BYPRODUCTS ${_opennn_jpeg_library}
        )
        add_library(opennn_jpeg STATIC IMPORTED GLOBAL)
        add_dependencies(opennn_jpeg opennn_libjpeg_turbo_ep)
        set_target_properties(opennn_jpeg PROPERTIES
            IMPORTED_LOCATION ${_opennn_jpeg_library})
        add_dependencies(opennn opennn_libjpeg_turbo_ep)
        # Build-tree only fallback, same install treatment as the ZLIB one.
        set(_opennn_jpeg_installed_name
            ${CMAKE_STATIC_LIBRARY_PREFIX}opennn_jpeg${CMAKE_STATIC_LIBRARY_SUFFIX})
        target_include_directories(opennn SYSTEM PRIVATE
            ${_opennn_jpeg_include_dir})
        target_link_libraries(opennn PUBLIC
            $<BUILD_INTERFACE:opennn_jpeg>
            $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_LIBDIR}/${_opennn_jpeg_installed_name}>)
        set(_opennn_install_jpeg_fallback ON)
    endif()

# stdc++fs is only needed on GCC < 9; modern toolchains have std::filesystem
# in libstdc++ itself and the link library may not exist in the sysroot.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(opennn PUBLIC stdc++fs)
endif()

# Single public include root: the one supported spelling is
# #include "opennn/foo.h". Library-internal bare includes ("dataset.h")
# resolve through the quoted-include same-directory rule, not through
# these paths, so the opennn/ dir itself is not exposed.
target_include_directories(opennn PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(opennn PUBLIC Eigen3::Eigen)

if(OpenNN_ENABLE_MKL)
    # Allow OpenNN_MKL_ROOT, the MKLROOT env var (set by oneAPI's setvars), or
    # ONEAPI_ROOT to point at the install. Single-DLL mkl_rt is used everywhere:
    # it picks threading/interface layers at runtime, so we link one import lib.
    if(NOT OpenNN_MKL_ROOT)
        if(DEFINED ENV{MKLROOT})
            set(OpenNN_MKL_ROOT "$ENV{MKLROOT}")
        elseif(DEFINED ENV{ONEAPI_ROOT} AND EXISTS "$ENV{ONEAPI_ROOT}/mkl/latest")
            set(OpenNN_MKL_ROOT "$ENV{ONEAPI_ROOT}/mkl/latest")
        endif()
    endif()

    # Prefer oneAPI's own MKLConfig.cmake when discoverable: it resolves the
    # correct lib/include/redist layout (which differs across oneAPI versions)
    # and exposes MKL::MKL with the runtime DLL dir baked in.
    # Default to the single-DLL mkl_rt (sdl) with the LP64 interface: mkl_rt
    # picks LP64 at runtime unless told otherwise, so compiling against
    # MKLConfig's ilp64 default while running through mkl_rt corrupts every
    # LAPACK integer argument (e.g. geqp3 pivots used by Levenberg-Marquardt).
    if(NOT DEFINED MKL_LINK)
        set(MKL_LINK sdl)
    endif()
    if(NOT DEFINED MKL_INTERFACE)
        set(MKL_INTERFACE lp64)
    endif()
    find_package(MKL CONFIG QUIET
        HINTS "${OpenNN_MKL_ROOT}" "${OpenNN_MKL_ROOT}/lib/cmake/mkl")

    if(MKL_FOUND)
        target_compile_definitions(opennn PUBLIC EIGEN_USE_MKL_ALL)
        target_link_libraries(opennn PUBLIC MKL::MKL)
        message(STATUS "OpenNN: MKL backend via MKLConfig.cmake (${MKL_ROOT})")
    else()
        # Manual discovery. Headers and the mkl_rt import library have
        # platform- and layout-specific names and locations.
        if(NOT OpenNN_MKL_ROOT)
            if(WIN32)
                message(FATAL_ERROR "OpenNN_ENABLE_MKL=ON but MKL was not found. "
                    "Run oneAPI's setvars.bat (sets MKLROOT) or pass -DOpenNN_MKL_ROOT=<mkl>.")
            else()
                set(OpenNN_MKL_ROOT "/usr")
            endif()
        endif()

        set(_opennn_mkl_include "${OpenNN_MKL_ROOT}/include")
        if(EXISTS "${_opennn_mkl_include}/mkl/mkl.h")
            set(_opennn_mkl_include "${_opennn_mkl_include}/mkl")
        endif()

        # Candidate lib dirs cover oneAPI (lib/), classic (lib/intel64/), and
        # Debian's multiarch layout.
        find_library(_opennn_mkl_rt
            NAMES mkl_rt
            HINTS "${OpenNN_MKL_ROOT}"
            PATH_SUFFIXES lib lib/intel64 lib/x86_64-linux-gnu lib/intel64/gcc4.8
            NO_DEFAULT_PATH)

        if(NOT EXISTS "${_opennn_mkl_include}/mkl.h")
            message(FATAL_ERROR "OpenNN_ENABLE_MKL=ON but mkl.h was not found under "
                "${OpenNN_MKL_ROOT}. Set OpenNN_MKL_ROOT to the MKL prefix.")
        endif()
        if(NOT _opennn_mkl_rt)
            message(FATAL_ERROR "OpenNN_ENABLE_MKL=ON but the mkl_rt library was not found under "
                "${OpenNN_MKL_ROOT}. Set OpenNN_MKL_ROOT to the MKL prefix.")
        endif()

        target_compile_definitions(opennn PUBLIC EIGEN_USE_MKL_ALL)
        target_include_directories(opennn SYSTEM PUBLIC "${_opennn_mkl_include}")
        target_link_libraries(opennn PUBLIC "${_opennn_mkl_rt}")

        if(NOT WIN32)
            # mkl_rt dlopen()s its layer libraries at runtime; pthread/m/dl and an
            # rpath to the lib dir are needed so they resolve. On Windows the
            # matching mkl_rt.dll must instead be on PATH at run time (it lives in
            # the oneAPI redist/bin dir, added by setvars.bat).
            get_filename_component(_opennn_mkl_lib_dir "${_opennn_mkl_rt}" DIRECTORY)
            target_link_libraries(opennn PUBLIC pthread m dl)
            target_link_options(opennn PUBLIC "-Wl,-rpath,${_opennn_mkl_lib_dir}")
        endif()
        message(STATUS "OpenNN: MKL backend via manual discovery (${_opennn_mkl_rt})")
    endif()
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
        # Link through an imported target so the exported OpenNNTargets.cmake
        # references CUDNN::cudnn (resolved on the consumer machine by the
        # installed FindCUDNN.cmake) instead of a non-relocatable absolute path.
        if(NOT TARGET CUDNN::cudnn)
            add_library(CUDNN::cudnn UNKNOWN IMPORTED)
            set_target_properties(CUDNN::cudnn PROPERTIES
                IMPORTED_LOCATION "${CUDNN_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}")
        endif()
        target_link_libraries(opennn PUBLIC CUDNN::cudnn)
    endif()

    if(NOT CUDNN_FRONTEND_INCLUDE_DIR)
        message(FATAL_ERROR
            "OpenNN with OPENNN_HAS_CUDA=ON requires the cuDNN frontend headers "
            "(cudnn_frontend.h). Install from https://github.com/NVIDIA/cudnn-frontend "
            "and set CUDNN_FRONTEND_INCLUDE_DIR.")
    endif()
    # SYSTEM include so GCC suppresses third-party warnings (-Wswitch-enum,
    # -Wunused-parameter, etc.) from inside cudnn-frontend headers. PRIVATE:
    # only .cpp files include cudnn-frontend, no public header exposes it.
    target_include_directories(opennn SYSTEM PRIVATE ${CUDNN_FRONTEND_INCLUDE_DIR})
    # CUDA operators guard the cuDNN frontend paths with HAVE_CUDNN_FRONTEND.
    # The top-level build fetches these headers; standalone consumers can still
    # pass CUDNN_FRONTEND_INCLUDE_DIR explicitly.

    target_compile_definitions(opennn PRIVATE HAVE_CUDNN_FRONTEND)
endif()

# Optimization & SIMD. Without host-native ISA flags Eigen falls back to the x86-64
# baseline (SSE2), costing 2-4x on matmul-heavy workloads. Mirrors opennn.pro.
# Guarded to CXX so nvcc is unaffected — it forwards its own host flags separately.
# The ISA flag is PUBLIC behind BUILD_INTERFACE: every in-tree consumer
# (tests/examples/benchmarks) must instantiate Eigen with the same ISA and
# alignment as the library, but the installed package must not impose a host
# ISA on external consumers. Fetched third-party targets (gtest, zlib) no
# longer inherit it.
if(MSVC)
    target_compile_options(opennn PUBLIC
        $<BUILD_INTERFACE:$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:DEBUG>>>:/arch:AVX2>>
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

# Disable Eigen's internal shape/index/aliasing assertions in release builds.
# NDEBUG (auto-added by CMake for non-Debug configs) only silences <cassert>;
# Eigen has its own eigen_assert macro controlled separately. PUBLIC so any
# consumer of the library sees the same Eigen behavior.
target_compile_definitions(opennn PUBLIC
    $<$<NOT:$<CONFIG:Debug>>:EIGEN_NO_DEBUG>
)

# Eigen ABI/behavior configuration. These must reach every TU that includes
# Eigen — including consumer TUs that include Eigen BEFORE any OpenNN header,
# where the #defines inside our headers would arrive too late. The copies in
# opennn_types.h (#ifndef-guarded) remain for non-CMake consumers.
target_compile_definitions(opennn PUBLIC
    EIGEN_MAX_ALIGN_BYTES=64
    EIGEN_USE_THREADS
)

# Precompiled header: binary PCH on every compiler. The historical GCC
# failure — headers textually including pch.h while the binary PCH and the
# textual include kept separate #pragma once bookkeeping — is gone: no header
# includes pch.h anymore (they include opennn_types.h), and both files carry
# classic include guards, so any re-inclusion is macro-guarded.
target_precompile_headers(opennn PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/pch.h>
)

# On MSVC, these TUs use SKIP_PRECOMPILE_HEADERS to avoid the half/bfloat16
# conflict. On GCC (-include pch.h) the CUDA headers pulled in by pch.h define
# __half/__nv_bfloat16 before the cudnn-frontend include, so no disambiguation
# is needed — but the include order inside those three files is load-bearing.
if(MSVC AND OPENNN_HAS_CUDA)
    set_source_files_properties(
        ${CMAKE_CURRENT_SOURCE_DIR}/attention_operator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/batch_norm_operator.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/convolution_operator.cpp
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

# Install rules: library + headers + a relocatable CMake package so external
# projects can `find_package(OpenNN CONFIG)` and link OpenNN::opennn.
option(OpenNN_INSTALL "Generate install rules for the OpenNN library" ON)
if(OpenNN_INSTALL)
    include(CMakePackageConfigHelpers)

    install(TARGETS opennn
        EXPORT OpenNNTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

    install(FILES ${HEADER_FILES}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/opennn)

    if(_opennn_install_zlib_fallback)
        install(FILES $<TARGET_FILE:zlibstatic>
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RENAME ${_opennn_zlib_installed_name})
    endif()

    if(_opennn_install_jpeg_fallback)
        install(FILES ${_opennn_jpeg_library}
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RENAME ${_opennn_jpeg_installed_name})
    endif()

    install(EXPORT OpenNNTargets
        FILE OpenNNTargets.cmake
        NAMESPACE OpenNN::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenNN)

    # Which dependencies the exported target references as imported targets,
    # so OpenNNConfig.cmake only find_dependency()s what it actually needs.
    set(_opennn_needs_tbb OFF)
    if(NOT MSVC AND TBB_FOUND)
        set(_opennn_needs_tbb ON)
    endif()
    set(_opennn_needs_openmp OFF)
    if(NOT APPLE AND OpenMP_FOUND AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(_opennn_needs_openmp ON)
    endif()
    set(_opennn_needs_zlib ${_opennn_needs_zlib_dependency})
    set(_opennn_needs_jpeg ${_opennn_needs_jpeg_dependency})
    set(_opennn_needs_cudnn OFF)
    if(OPENNN_HAS_CUDA AND HAVE_CUDNN)
        set(_opennn_needs_cudnn ON)
    endif()
    # Only when MKLConfig.cmake provided MKL::MKL; the manual-discovery branch
    # exports the mkl_rt path directly and needs no find_dependency.
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
