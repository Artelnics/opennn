//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   K - M E A N S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_utilities.h"

namespace opennn
{

class KMeans
{

public:

    KMeans(Index clusters = 3, Index = 100);

    VectorI calculate_outputs(const MatrixR&);
    VectorR elbow_method(const MatrixR&, Index = 10);
    Index find_optimal_clusters(const VectorR&) const;

    VectorI get_cluster_labels() const;
    MatrixR get_cluster_centers() const;
    Index get_clusters_number() const;

    void fit(const MatrixR&);
    void set_cluster_number(const Index);
    void set_centers_random(const MatrixR&);

private:

    Index clusters_number;
    Index maximum_iterations;
    //string metric;

    MatrixR cluster_centers;
    VectorI rows_cluster_labels;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
