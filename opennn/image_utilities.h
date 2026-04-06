//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E N N N   I M A G E S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

Tensor3 load_image(const filesystem::path&);

Tensor3 resize_image(const Tensor3&, Index, Index);

void reflect_image_x(Tensor3&);
void reflect_image_y(Tensor3&);
void rotate_image(const Tensor3&, Tensor3&, type);
void translate_image_x(const Tensor3&, Tensor3&, Index);
void translate_image_y(const Tensor3&, Tensor3&, Index);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
