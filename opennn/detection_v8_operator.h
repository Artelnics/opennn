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
// reg_max=1 (default, Phase 5a): sigmoid applied to all channels.
//   ch 0..3:        sigmoid(t) → x/y offset, w, h ∈ [0,1]
//   ch 4..4+C-1:    sigmoid(cls_c)
// reg_max>1 (DFL):  box channels pass through as raw logits; only class channels get sigmoid.
//   ch 0..4*RM-1:   raw DFL logits (RM = reg_max)
//   ch 4*RM..end:   sigmoid(cls_c)
struct DetectionV8Operator : Operator
{
    Index grid_size      = 0;
    Index grid_width     = 0;
    Index classes_number = 0;
    Index reg_max        = 1;

    void set(const Shape&, Index reg_max);

    void forward_propagate(ForwardPropagation&, size_t, bool) override;
    void back_propagate(ForwardPropagation&, BackPropagation&, size_t) const override;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
