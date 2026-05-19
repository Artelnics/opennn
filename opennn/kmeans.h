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

/// @brief K-means clustering utility that partitions samples into the requested number of clusters.
class KMeans
{

public:

    /// @brief Builds a K-means instance with the given cluster count and maximum number of iterations.
    KMeans(Index clusters = 3, Index = 100);

    /// @brief Assigns each row of the input matrix to its nearest cluster.
    /// @return Vector with the cluster index for every row.
    VectorI calculate_outputs(const MatrixR&);

    /// @brief Runs the elbow method on the supplied data over a range of cluster counts.
    /// @return Vector with the within-cluster distortion for each tested cluster count.
    VectorR elbow_method(const MatrixR&, Index = 10);

    /// @brief Returns the cluster count located at the elbow of the supplied distortion curve.
    Index find_optimal_clusters(const VectorR&) const;

    /// @brief Returns the cluster label assigned to each fitted sample.
    VectorI get_cluster_labels() const;

    /// @brief Returns the centroid of each cluster as rows of the returned matrix.
    MatrixR get_cluster_centers() const;

    /// @brief Returns the number of clusters configured for the algorithm.
    Index get_clusters_number() const;

    /// @brief Fits the K-means model on the supplied data matrix.
    void fit(const MatrixR&);

    /// @brief Sets the desired number of clusters.
    void set_cluster_number(const Index);

    /// @brief Initializes cluster centres by sampling at random from the supplied data.
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
