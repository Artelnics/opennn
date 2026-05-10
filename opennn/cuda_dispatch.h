//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D A   D I S P A T C H   H E L P E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#ifdef OPENNN_HAS_CUDA

namespace opennn {
// Cached at first call (C++11 magic statics make this thread-safe). Configuration
// is set once at startup, so the value never changes after init.
inline bool is_gpu_cached() {
    static const bool v = is_gpu();
    return v;
}
}

    #define TRY_GPU_DISPATCH(view, ...) \
        (::opennn::is_gpu_cached() \
            ? ((view).dispatch(__VA_ARGS__), true) \
            : false)

    #define IF_GPU(...) \
        do { \
            if (::opennn::is_gpu_cached()) { \
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
