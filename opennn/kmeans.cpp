// OpenNN: Open Neural Networks Library
// www.opennn.net
//
// K - M E A N S   C L A S S
//
// Artificial Intelligence Techniques SL
// artelnics@artelnics.com

#include <string>
#include <omp.h>

#include "tensors.h"
#include "config.h"
#include "kmeans.h"

namespace opennn
{

KMeans::KMeans(Index clusters,
               string distance_calculation_method,
               Index iterations_number)
    : clusters_number(clusters), maximum_iterations(iterations_number), metric(distance_calculation_method)
{
}


void KMeans::fit(const Tensor<type, 2>& data)
{
    const Index rows_number = data.dimension(0);
    const Index columns_number = data.dimension(1);

    Tensor<type, 1> row(rows_number);
    Tensor<type, 1> center(columns_number);
    Tensor<type, 1> center_sum(columns_number);

    cluster_centers.resize(clusters_number, columns_number);
    rows_cluster_labels.resize(rows_number);

    set_centers_random(data);

    for(Index iterations_number = 0; iterations_number < maximum_iterations; iterations_number++)
    {              
        for(Index row_index = 0; row_index < rows_number; row_index++)
        {
            row = data.chip(row_index, 0);

            center = cluster_centers.chip(0, 0);

            type minimum_distance = l2_distance(row, center);

            Index minimal_distance_cluster_index = 0;

            for(Index cluster_index = 1; cluster_index < clusters_number; cluster_index++)
            {
                center = cluster_centers.chip(cluster_index, 0);

                const type distance = l2_distance(row, center);

                if(distance < minimum_distance)
                {
                    minimum_distance = distance;
                    minimal_distance_cluster_index = cluster_index;
                }
            }

            rows_cluster_labels(row_index) = minimal_distance_cluster_index;
        }

        for(Index cluster_index = 0; cluster_index < clusters_number; cluster_index++)
        {
            center_sum.setZero();

            Index count = 0;

            for(Index row_index = 0; row_index < rows_number; row_index++)
            {
                if(rows_cluster_labels(row_index) == cluster_index)
                {
                    row = data.chip(row_index, 0);

                    center_sum += row;
                    count++;
                }
            }

            if(count != 0)
            {
                center = center_sum / type(count);
                cluster_centers.chip(cluster_index, 0) = center;
            }
        }
    }
}


Tensor<Index, 1> KMeans::calculate_outputs(const Tensor<type, 2>& data)
{
    const Index rows_number = data.dimension(0);
    Tensor<type, 1> row(data.dimension(1));
    Tensor<type, 1> center;

    Tensor<Index, 1> predictions(rows_number);

    for(Index row_index = 0; row_index < rows_number; row_index++)
    {
        row = data.chip(row_index, 0);
        center = cluster_centers.chip(0, 0);

        type minimum_distance = l2_distance(row, center);
        Index minimal_distance_cluster_index = 0;

        for(Index cluster_index = 1; cluster_index < clusters_number; cluster_index++)
        {
            center = cluster_centers.chip(cluster_index, 0);
            const type distance = l2_distance(row, center);

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


Tensor<type, 1> KMeans::elbow_method(const Tensor<type, 2>& data, Index max_clusters)
{
    Tensor<type, 1> data_point;
    Tensor<type, 1> cluster_center;
    Tensor<type, 1> sum_squared_error_values(max_clusters);

    const Index rows_number = data.dimension(0);

    Index original_clusters_number = clusters_number;
    type sum_squared_error;

    for(Index cluster_index = 1; cluster_index <= max_clusters; cluster_index++)
    {
        clusters_number = cluster_index;

        fit(data);

        sum_squared_error = type(0);

        for(Index row_index = 0; row_index < rows_number; row_index++)
        {
            data_point = data.chip(row_index, 0);
            cluster_center = cluster_centers.chip(rows_cluster_labels(row_index), 0);

            sum_squared_error += type(pow(l2_distance(data_point, cluster_center), 2));
        }

        sum_squared_error_values(cluster_index-1) = sum_squared_error;
    }

    clusters_number = original_clusters_number;

    return sum_squared_error_values;
}


Index KMeans::find_optimal_clusters(const Tensor<type, 1>& sum_squared_error_values) const
{
    const Index cluster_number = sum_squared_error_values.dimension(0);

    Tensor<type, 1> initial_endpoint(2);
    initial_endpoint.setValues({ type(1), type(sum_squared_error_values(0)) });

    Tensor<type, 1> final_endpoint(2);
    final_endpoint.setValues({ type(clusters_number), sum_squared_error_values(clusters_number - 1) });

    type max_distance = type(0);
    Index optimal_clusters_number = 1;

    Tensor<type, 1> current_point(2);
    type perpendicular_distance;

    for(Index cluster_index = 1; cluster_index <= cluster_number; cluster_index++)
    {
        current_point.setValues({ type(cluster_index), sum_squared_error_values(cluster_index - 1) });
         
        perpendicular_distance
            = type(abs((final_endpoint(1) - initial_endpoint(1)) * current_point(0) -
                  (final_endpoint(0) - initial_endpoint(0)) * current_point(1) +
                   final_endpoint(0) * initial_endpoint(1) - final_endpoint(1) * initial_endpoint(0))) /
              type(sqrt(pow(final_endpoint(1) - initial_endpoint(1), 2) + pow(final_endpoint(0) - initial_endpoint(0), 2)));

        if(perpendicular_distance > max_distance)
        {
            max_distance = perpendicular_distance;
            optimal_clusters_number = cluster_index;
        }
    }

    return optimal_clusters_number;
}


Tensor<type, 2> KMeans::get_cluster_centers()
{
    return cluster_centers;
}


Tensor<Index, 1> KMeans::get_cluster_labels()
{
    return rows_cluster_labels;
}


Index KMeans::get_clusters_number() const
{
    return clusters_number;
}


void KMeans::set_cluster_number(const Index& new_clusters_number)
{
    clusters_number = new_clusters_number;
}


void KMeans::set_centers_random(const Tensor<type, 2>& data)
{
    const Index data_size = data.dimension(0);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> index_distribution(0, data_size - 1);

    for(Index i = 0; i < clusters_number; i++)
        cluster_centers.chip(i, 0) = data.chip(index_distribution(gen), 0);
}

}
