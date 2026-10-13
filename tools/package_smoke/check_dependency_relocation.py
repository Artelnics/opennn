#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

"""Check package dependency discovery after both installation prefixes move.

Uses empty dependency fixtures and a header-only producer: this checks CMake
configuration and exports on any host, not native oneDNN or macOS execution.
The real installed library is compiled and linked by package_smoke separately.
"""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[2]

PRODUCER = r'''
cmake_minimum_required(VERSION 3.24)
project(DependencyProducer LANGUAGES NONE)
include(CMakePackageConfigHelpers)
list(PREPEND CMAKE_MODULE_PATH "${OPENNN_SOURCE}/opennn/cmake")
find_package(OpenNNOneDNN REQUIRED)
find_package(OpenNNLibomp REQUIRED)
add_library(opennn INTERFACE)
target_link_libraries(opennn INTERFACE OpenNN::oneDNN OpenNN::libomp)
install(TARGETS opennn EXPORT OpenNNTargets)
install(EXPORT OpenNNTargets NAMESPACE OpenNN:: DESTINATION lib/cmake/OpenNN)
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_INCLUDEDIR include)
set(_opennn_jpeg_filename jpeg.stub)
set(APPLE ON)
set(OpenNN_ONEDNN_FOUND ON)
set(_opennn_needs_openmp OFF)
set(_opennn_needs_cudnn OFF)
set(_opennn_needs_mkl OFF)
set(OPENNN_HAS_CUDA OFF)
set(TBB_FOUND OFF)
configure_package_config_file("${OPENNN_SOURCE}/opennn/OpenNNConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/OpenNNConfig.cmake"
    INSTALL_DESTINATION lib/cmake/OpenNN
    PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/OpenNNConfig.cmake"
    "${OPENNN_SOURCE}/opennn/cmake/FindOpenNNOneDNN.cmake"
    "${OPENNN_SOURCE}/opennn/cmake/FindOpenNNLibomp.cmake"
    DESTINATION lib/cmake/OpenNN)
'''

CONSUMER = r'''
cmake_minimum_required(VERSION 3.24)
project(DependencyConsumer LANGUAGES NONE)
find_package(OpenNN CONFIG REQUIRED)
# Repeat discovery to check that imported targets can be reused.
find_package(OpenNN CONFIG REQUIRED)
foreach(dependency OpenNN::oneDNN OpenNN::libomp)
    get_target_property(location ${dependency} IMPORTED_LOCATION)
    cmake_path(IS_PREFIX EXPECTED_DEPENDENCIES "${location}" NORMALIZE matches)
    if(NOT matches)
        message(FATAL_ERROR "${dependency} resolved outside the new prefix: ${location}")
    endif()
endforeach()
get_target_property(links OpenNN::opennn INTERFACE_LINK_LIBRARIES)
if(NOT links STREQUAL "OpenNN::oneDNN;OpenNN::libomp")
    message(FATAL_ERROR "Package did not preserve dependency targets: ${links}")
endif()
'''


def write(path, content=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main():
    cmake = shutil.which("cmake")
    if not cmake:
        raise SystemExit("CMake is required")
    env = os.environ.copy()
    for name in ("ONEDNN_ROOT", "DNNLROOT", "CMAKE_PREFIX_PATH", "CMAKE_TOOLCHAIN_FILE"):
        env.pop(name, None)

    def run(*args):
        subprocess.run([cmake, *map(str, args)], check=True, env=env)

    with tempfile.TemporaryDirectory(prefix="opennn-package-check-") as directory:
        work = Path(directory).resolve()
        dependencies = work / "producer-dependencies"
        package = work / "producer-package"
        write(dependencies / "include/oneapi/dnnl/dnnl.hpp")
        write(dependencies / "include/omp.h")
        for name in ("dnnl.lib", "libdnnl.so", "libdnnl.dylib", "omp.lib", "libomp.so", "libomp.dylib"):
            write(dependencies / "lib" / name)
        # Isolate these unrelated dependencies from the host environment.
        write(package / "lib/cmake/OpenNN/FindEigen3.cmake", "set(Eigen3_FOUND TRUE)\n")
        write(package / "lib/cmake/ZLIB/ZLIBConfig.cmake", "set(ZLIB_FOUND TRUE)\n")
        write(package / "lib/jpeg.stub")
        (package / "include").mkdir()
        write(work / "producer/CMakeLists.txt", PRODUCER)
        write(work / "consumer/CMakeLists.txt", CONSUMER)
        run("-S", work / "producer", "-B", work / "build-producer",
            f"-DOPENNN_SOURCE={ROOT.as_posix()}",
            f"-DCMAKE_PREFIX_PATH={dependencies.as_posix()}")
        run("--install", work / "build-producer", "--prefix", package)
        relocated_dependencies = work / "consumer-dependencies"
        relocated_package = work / "consumer-package"
        for source, destination in ((dependencies, relocated_dependencies), (package, relocated_package)):
            assert source.resolve().is_relative_to(work) and destination.resolve().is_relative_to(work)
            source.rename(destination)
        common = ["-S", work / "consumer",
                  f"-DOpenNN_DIR={relocated_package.as_posix()}/lib/cmake/OpenNN",
                  f"-DEXPECTED_DEPENDENCIES={relocated_dependencies.as_posix()}",
                  "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF"]
        run(*common, "-B", work / "build-prefix",
            f"-DCMAKE_PREFIX_PATH={relocated_package.as_posix()};{relocated_dependencies.as_posix()}")
        # Existing explicit include/library overrides remain supported.
        libomp_name = "omp.lib" if os.name == "nt" else "libomp.dylib"
        run(*common, "-B", work / "build-overrides",
            f"-DCMAKE_PREFIX_PATH={relocated_package.as_posix()}",
            f"-DOpenNN_ONEDNN_ROOT={relocated_dependencies.as_posix()}",
            f"-DLIBOMP_INCLUDE_DIR={relocated_dependencies.as_posix()}/include",
            f"-DLIBOMP_LIBRARY={relocated_dependencies.as_posix()}/lib/{libomp_name}")
    print("Dependency relocation and explicit overrides passed.")


if __name__ == "__main__":
    main()
