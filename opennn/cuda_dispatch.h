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
        (::opennn::Configuration::instance().is_gpu() \
            ? ((view).dispatch(__VA_ARGS__), true) \
            : false)
#else
    #define TRY_GPU_DISPATCH(view, ...) (false)
#endif

// IF_GPU(...) wraps the body in `if (Device::is_gpu()) { ... }` when CUDA is on,
// and discards it entirely when CUDA is off. Use for GPU branches that call
// cuDNN / cuBLAS / cuBLASLt directly (i.e. not the templated *_cuda<T> dispatch
// pattern that TRY_GPU_DISPATCH covers).
//
// Like TRY_GPU_DISPATCH, this is a macro — not a function — because the body
// often references types/functions that only exist when OPENNN_WITH_CUDA is
// defined (cudnnOpTensor, cublasLtMatmul, etc.). The macro form lets the
// preprocessor drop the body before the compiler ever sees those names.
//
// Usage:
//   IF_GPU({
//       CHECK_CUDNN(cudnnSomething(...));
//       return;
//   });
//   // CPU body follows.
#ifdef OPENNN_WITH_CUDA
    #define IF_GPU(body) do { if (::opennn::Configuration::instance().is_gpu()) body } while (0)
#else
    #define IF_GPU(body) ((void)0)
#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
