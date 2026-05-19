//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D A   D I S P A T C H   H E L P E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#ifdef OPENNN_HAS_CUDA

    // Read Configuration directly each call. The previous design cached the
    // value at first call (magic statics), but Configuration::set() can run
    // mid-process (examples, tests, multi-session runners) and the cache
    // would silently go stale. Configuration::is_gpu() is an atomic read on
    // the singleton — sub-nanosecond, and inlines away in tight dispatch.

    #define TRY_GPU_DISPATCH(view, ...) \
        (::opennn::is_gpu() \
            ? ((view).dispatch(__VA_ARGS__), true) \
            : false)

    #define IF_GPU(...) \
        do { \
            if (::opennn::is_gpu()) { \
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
