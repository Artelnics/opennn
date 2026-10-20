// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

// Stand-in for ATen's unpack of a philox state; the whole of it lives in
// ../CUDAGeneratorImpl.h, which some FlashAttention-2 sources reach through
// this path instead.

#pragma once

#include <ATen/cuda/CUDAGeneratorImpl.h>
