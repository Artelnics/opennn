//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   K - M E A N S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

namespace opennn
{

class KMeans
{

public:

    KMeans(Index clusters = 3, string distance_calculation_method = "euclidean", Index = 100);

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
    string metric;

    MatrixR cluster_centers;
    VectorI rows_cluster_labels;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
