//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D A   D I S P A T C H   H E L P E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#ifdef OPENNN_HAS_CUDA

    #define OPENNN_VIEW_IS_CUDA(view) ((view).is_cuda())

    #define TRY_GPU_DISPATCH(view, ...) \
        (OPENNN_VIEW_IS_CUDA(view) \
            ? ((view).dispatch(__VA_ARGS__), true) \
            : false)

#else

    #define OPENNN_VIEW_IS_CUDA(view) (false)
    #define TRY_GPU_DISPATCH(view, ...) (false)

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
