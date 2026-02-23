// OpenNN: Open Neural Networks Library
// www.opennn.net
//
// K - M E A N S   C L A S S
//
// Artificial Intelligence Techniques SL
// artelnics@artelnics.com

#include "kmeans.h"
#include "random_utilities.h"

namespace opennn
{

KMeans::KMeans(Index clusters,
               string distance_calculation_method,
               Index iterations_number)
    : clusters_number(clusters), maximum_iterations(iterations_number), metric(distance_calculation_method)
{
}


void KMeans::fit(const MatrixR& data)
{
    const Index rows_number = data.rows();
    const Index columns_number = data.cols();

    VectorR row(rows_number);
    VectorR center(columns_number);
    VectorR center_sum(columns_number);

    cluster_centers.resize(clusters_number, columns_number);
    rows_cluster_labels.resize(rows_number);

    set_centers_random(data);

    for(Index iterations_number = 0; iterations_number < maximum_iterations; iterations_number++)
    {              
        for(Index row_index = 0; row_index < rows_number; row_index++)
        {
            (cluster_centers.rowwise() - data.row(row_index)).rowwise().squaredNorm().minCoeff(&rows_cluster_labels(row_index));
        }

        for(Index cluster_index = 0; cluster_index < clusters_number; cluster_index++)
        {
            center_sum.setZero();

            Index count = 0;

            for(Index row_index = 0; row_index < rows_number; row_index++)
            {
                if(rows_cluster_labels(row_index) == cluster_index)
                {
                    row = data.row(row_index);

                    center_sum += row;
                    count++;
                }
            }

            if(count != 0)
            {
                center = center_sum / type(count);
                cluster_centers.row(cluster_index) = center;
            }
        }
    }
}


VectorI KMeans::calculate_outputs(const MatrixR& data)
{
    const Index rows_number = data.rows();
    VectorR row(data.cols());
    VectorR center;

    VectorI predictions(rows_number);

    for(Index row_index = 0; row_index < rows_number; row_index++)
    {
        row = data.row(row_index);
        center = cluster_centers.row(0);

        type minimum_distance = (row - center).norm();
        Index minimal_distance_cluster_index = 0;

        for(Index cluster_index = 1; cluster_index < clusters_number; cluster_index++)
        {
            center = cluster_centers.row(cluster_index);
            const type distance = (row - center).norm();

            if(distance < minimum_distance)
            {
                minimum_distance = distance;
                minimal_distance_cluster_index = cluster_index;
            }
        }

        predictions(row_index) = minimal_distance_cluster_index;
    }

    return predictions;
}


VectorR KMeans::elbow_method(const MatrixR& data, Index max_clusters)
{
    VectorR data_point;
    VectorR cluster_center;
    VectorR sum_squared_error_values(max_clusters);

    const Index rows_number = data.rows();

    Index original_clusters_number = clusters_number;
    type mean_squared_error;

    for(Index cluster_index = 1; cluster_index <= max_clusters; cluster_index++)
    {
        clusters_number = cluster_index;

        fit(data);

        mean_squared_error = type(0);

        for(Index row_index = 0; row_index < rows_number; row_index++)
        {
            data_point = data.row(row_index);
            cluster_center = cluster_centers.row(rows_cluster_labels(row_index));

            mean_squared_error += type(pow((data_point - cluster_center).norm(), 2));
        }

        sum_squared_error_values(cluster_index-1) = mean_squared_error;
    }

    clusters_number = original_clusters_number;

    return sum_squared_error_values;
}


Index KMeans::find_optimal_clusters(const VectorR& sum_squared_error_values) const
{
    const Index cluster_number = sum_squared_error_values.size();

    VectorR initial_endpoint(2);
    initial_endpoint << type(1), type(sum_squared_error_values(0));

    VectorR override_endpoint(2);
    override_endpoint << type(clusters_number), sum_squared_error_values(clusters_number - 1);

    type max_distance = type(0);
    Index optimal_clusters_number = 1;

    VectorR current_point(2);
    type perpendicular_distance;

    for(Index cluster_index = 1; cluster_index <= cluster_number; cluster_index++)
    {
        current_point << type(cluster_index), sum_squared_error_values(cluster_index - 1);
         
        perpendicular_distance
            = type(abs((override_endpoint(1) - initial_endpoint(1)) * current_point(0) -
                  (override_endpoint(0) - initial_endpoint(0)) * current_point(1) +
                   override_endpoint(0) * initial_endpoint(1) - override_endpoint(1) * initial_endpoint(0))) /
              type(sqrt(pow(override_endpoint(1) - initial_endpoint(1), 2) + pow(override_endpoint(0) - initial_endpoint(0), 2)));

        if(perpendicular_distance > max_distance)
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

    for(Index i = 0; i < clusters_number; i++)
        cluster_centers.row(i) = data.row(random_integer(0, data_size - 1));
}

}
