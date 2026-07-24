//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E T E C T I O N   V 8   O P E R A T O R   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "operator.h"

namespace opennn
{

// Anchor-free detection operator for YOLOv8-style heads.
// Input layout:  [B, G, G, 4+C]  — raw logits from concatenated box+class branches
// Output layout: [B, G, G, 4+C]  — decoded in-place
//   ch 0: sigmoid(tx) → x offset within grid cell ∈ [0,1]
//   ch 1: sigmoid(ty) → y offset within grid cell ∈ [0,1]
//   ch 2: sigmoid(tw) → normalized width ∈ [0,1]
//   ch 3: sigmoid(th) → normalized height ∈ [0,1]
//   ch 4..4+C-1: sigmoid(cls_c) → per-class probabilities
// No anchor parameters, no objectness channel.
struct DetectionV8Operator : Operator
{
    Index grid_size    = 0;
    Index grid_width   = 0;
    Index classes_number = 0;

    void set(const Shape&);

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
