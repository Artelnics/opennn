//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D A   D I S P A T C H   H E L P E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_utilities.h"

// Macro (not function) because the lambda body references *_cuda<T> templates that
// are only declared when OPENNN_WITH_CUDA is set (via pch.h → kernel.cuh). A function
// helper would force the lambda to compile in the CPU TU, where those names don't exist.
// As a macro, the lambda is discarded token-by-token in the CPU build.
//
// Usage:
//   if (TRY_GPU_DISPATCH(output, [&](auto tag) {
//       using T = decltype(tag);
//       bounding_cuda<T>(output.size(), to_int(features), input.as<T>(), ..., output.as<T>());
//   })) return;
//   // CPU body follows.

// Variadic so the preprocessor keeps top-level commas inside the lambda body
// (e.g., `const int A = ..., B = ...;`) — they get folded into __VA_ARGS__.
#ifdef OPENNN_WITH_CUDA
    #define TRY_GPU_DISPATCH(view, ...) \
        (::opennn::Device::instance().is_gpu() \
            ? ((view).dispatch(__VA_ARGS__), true) \
            : false)
#else
    #define TRY_GPU_DISPATCH(view, ...) (false)
#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
