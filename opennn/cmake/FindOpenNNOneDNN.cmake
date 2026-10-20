# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

# Locate oneDNN for both builds and installed consumers. Preserve OpenNN's
# explicit include/library overrides and support CMAKE_PREFIX_PATH.
# $ENV{ONEDNN_ROOT} and $ENV{DNNLROOT} are what oneDNN's own vars scripts
# export, so a machine that has sourced one is already findable. A prefix
# in neither a standard location nor those variables still needs
# OpenNN_ONEDNN_ROOT or CMAKE_PREFIX_PATH -- AUTO finds an installed
# oneDNN, it does not go looking through home directories.
find_path(OpenNN_ONEDNN_INCLUDE_DIR
    NAMES oneapi/dnnl/dnnl.hpp
    HINTS
        "${OpenNN_ONEDNN_ROOT}/include"
        "$ENV{ONEDNN_ROOT}/include"
        "$ENV{DNNLROOT}/include")
find_library(OpenNN_ONEDNN_LIBRARY
    NAMES dnnl
    HINTS
        "${OpenNN_ONEDNN_ROOT}/lib"
        "${OpenNN_ONEDNN_ROOT}/lib64"
        "${OpenNN_ONEDNN_ROOT}/lib/x86_64-linux-gnu"
        "$ENV{ONEDNN_ROOT}/lib"
        "$ENV{DNNLROOT}/lib")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenNNOneDNN
    REQUIRED_VARS OpenNN_ONEDNN_INCLUDE_DIR OpenNN_ONEDNN_LIBRARY)

if(OpenNNOneDNN_FOUND AND NOT TARGET OpenNN::oneDNN)
    add_library(OpenNN::oneDNN UNKNOWN IMPORTED)
    set_target_properties(OpenNN::oneDNN PROPERTIES
        IMPORTED_LOCATION "${OpenNN_ONEDNN_LIBRARY}")
endif()

mark_as_advanced(OpenNN_ONEDNN_INCLUDE_DIR OpenNN_ONEDNN_LIBRARY)
