// OpenNN: Open Neural Networks Library
// www.opennn.net
//
//
//   Artificial Intelligence Techniques SL
// artelnics@artelnics.com

#include "kmeans.h"
#include "random_utilities.h"

namespace opennn
{

KMeans::KMeans(Index clusters,
               Index iterations_number)
    : clusters_number(clusters), maximum_iterations(iterations_number)
{
}

void KMeans::fit(const MatrixR& data)
{
    const Index rows_number = data.rows();
    const Index columns_number = data.cols();

    cluster_centers.resize(clusters_number, columns_number);
    rows_cluster_labels.resize(rows_number);

    set_centers_random(data);

    VectorI counts(clusters_number);

    for (Index iter = 0; iter < maximum_iterations; ++iter)
    {
        #pragma omp parallel for
        for (Index row_index = 0; row_index < rows_number; ++row_index)
            (cluster_centers.rowwise() - data.row(row_index)).rowwise().squaredNorm().minCoeff(&rows_cluster_labels(row_index));

        cluster_centers.setZero();
        counts.setZero();

        for (Index row_index = 0; row_index < rows_number; ++row_index)
        {
            const Index cluster_index = rows_cluster_labels(row_index);
            cluster_centers.row(cluster_index) += data.row(row_index);
            counts(cluster_index)++;
        }

        for (Index cluster_index = 0; cluster_index < clusters_number; ++cluster_index)
            if (counts(cluster_index) != 0)
                cluster_centers.row(cluster_index) /= float(counts(cluster_index));
    }
}

VectorI KMeans::calculate_outputs(const MatrixR& data)
{
    const Index rows_number = data.rows();

    VectorI predictions(rows_number);

    #pragma omp parallel for
    for (Index row_index = 0; row_index < rows_number; ++row_index)
        (cluster_centers.rowwise() - data.row(row_index)).rowwise().squaredNorm().minCoeff(&predictions(row_index));

    return predictions;
}

VectorR KMeans::elbow_method(const MatrixR& data, Index max_clusters)
{
    VectorR sum_squared_error_values(max_clusters);

    const Index rows_number = data.rows();
    const Index original_clusters_number = clusters_number;

    for (Index cluster_index = 1; cluster_index <= max_clusters; ++cluster_index)
    {
        clusters_number = cluster_index;
        fit(data);

        float sum_squared_error = 0.0f;
        #pragma omp parallel for reduction(+:sum_squared_error)
        for (Index row_index = 0; row_index < rows_number; ++row_index)
            sum_squared_error += (data.row(row_index) - cluster_centers.row(rows_cluster_labels(row_index))).squaredNorm();

        sum_squared_error_values(cluster_index - 1) = sum_squared_error;
    }

    clusters_number = original_clusters_number;

    return sum_squared_error_values;
}

Index KMeans::find_optimal_clusters(const VectorR& sum_squared_error_values) const
{
    const Index cluster_number = sum_squared_error_values.size();

    const float first_x = 1.0f;
    const float first_y = sum_squared_error_values(0);
    const float last_x = float(cluster_number);
    const float last_y = sum_squared_error_values(cluster_number - 1);

    const float delta_x = last_x - first_x;
    const float delta_y = last_y - first_y;
    const float cross_term = last_x * first_y - last_y * first_x;
    const float inv_line_length = 1.0f / sqrt(delta_y * delta_y + delta_x * delta_x);

    float max_distance = 0.0f;
    Index optimal_clusters_number = 1;

    for (Index cluster_index = 1; cluster_index <= cluster_number; ++cluster_index)
    {
        const float perpendicular_distance
            = abs(delta_y * float(cluster_index) - delta_x * sum_squared_error_values(cluster_index - 1) + cross_term) * inv_line_length;

        if (perpendicular_distance > max_distance)
        {
            max_distance = perpendicular_distance;
            optimal_clusters_number = cluster_index;
        }
    }

    return optimal_clusters_number;
}

MatrixR KMeans::get_cluster_centers() const
{
    return cluster_centers;
}

VectorI KMeans::get_cluster_labels() const
{
    return rows_cluster_labels;
}

Index KMeans::get_clusters_number() const
{
    return clusters_number;
}

void KMeans::set_cluster_number(const Index new_clusters_number)
{
    clusters_number = new_clusters_number;
}

void KMeans::set_centers_random(const MatrixR& data)
{
    const Index data_size = data.rows();

    for (Index i = 0; i < clusters_number; ++i)
        cluster_centers.row(i) = data.row(random_integer(0, data_size - 1));
}

}
