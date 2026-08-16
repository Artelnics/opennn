//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D E T E C T I O N   H E A D   C O N T R A C T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/opennn_types.h"

namespace opennn
{

enum class DetectionHeadKind { AnchorBased, AnchorFree };
enum class DetectionClassActivation { Softmax, Sigmoid };

struct DetectionHeadMetadata
{
    DetectionHeadKind kind = DetectionHeadKind::AnchorBased;
    Index boxes_per_cell = 1;
    Index classes_number = 0;
    Index regression_bins = 1;
    DetectionClassActivation class_activation = DetectionClassActivation::Softmax;
};

class DetectionHeadEndpoint
{
public:
    virtual ~DetectionHeadEndpoint() = default;

    virtual DetectionHeadMetadata get_detection_head_metadata() const noexcept = 0;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
