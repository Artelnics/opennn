// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#pragma once

#include "opennn/core/image_transform.h"

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

namespace opennn
{

void transform_images_cuda(const void*, void*, bool, const void*, uint8_t*,
                           const ImageTransformBatch&, cudaStream_t);

}

#endif
