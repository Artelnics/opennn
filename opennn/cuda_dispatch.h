//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D A   D I S P A T C H   H E L P E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#ifdef OPENNN_HAS_CUDA

    #define TRY_GPU_DISPATCH(view, ...) \
        (::opennn::Configuration::instance().is_gpu() \
            ? ((view).dispatch(__VA_ARGS__), true) \
            : false)

    #define IF_GPU(...) \
        do { \
            if (::opennn::Configuration::instance().is_gpu()) { \
                __VA_ARGS__ \
            } \
        } while (0)

#else

    #define TRY_GPU_DISPATCH(view, ...) (false)

    #define IF_GPU(...) ((void)0)

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
