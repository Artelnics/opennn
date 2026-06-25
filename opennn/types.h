//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T Y P E S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

namespace opennn
{

enum class Device { Auto, CPU, CUDA };
enum class Type { Auto, FP32, BF16 };

// Negative-side slope for LeakyReLU. 0.1 matches the Darknet/YOLO default.
inline constexpr float LEAKY_RELU_SLOPE = 0.1f;

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
