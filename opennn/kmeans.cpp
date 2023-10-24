// OpenNN: Open Neural Networks Library
// www.opennn.net
//
// K - M E A N S   C L A S S
//
// Artificial Intelligence Techniques SL
// artelnics@artelnics.com

#include "kmeans.h"

// System includes

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <omp.h>
#include <Eigen/Dense>
#include "unsupported/Eigen/CXX11/Tensor"
#include "config.h"

// OpenNN includes

#include "../../opennn/opennn/opennn.h"

using namespace OpenNN;

namespace OpenNN
{

type KMeans::euclidean_distance(const Tensor<type, 1>& a, const Tensor<type, 1>& b)
{
    const Tensor<type, 0> result_tensor = (a - b).square().sum();

    return sqrt(result_tensor(0));
}


KMeans::KMeans(Index clusters, string distance_calculation_metod, Index iterations_number)
    : clusters_number(clusters), maximum_iterations(iterations_number), metric(distance_calculation_metod)
{
}


void KMeans::fit(const Tensor<type, 2>& data)
{
    const Index rows_number = data.dimension(0);
    const Index columns_number = data.dimension(1);

    Tensor<type,1> row(rows_number);
    Tensor<type,1> center(columns_number);
    Tensor<type, 1> center_sum(columns_number);

    cluster_centers.resize(clusters_number, columns_number);
    rows_cluster_labels.resize(rows_number);

    set_centers_random(data);

    for (Index iterations_number = 0; iterations_number < maximum_iterations; iterations_number++)
    {
        for (Index row_index = 0; row_index < rows_number; row_index++)
        {
            row = data.chip(row_index, 0);

            center = cluster_centers.chip(0, 0);

            type minimum_distance = euclidean_distance(row, center);

            Index minimal_distance_cluster_index = 0;

            for(Index cluster_index = 1; cluster_index < clusters_number; cluster_index++)
            {
                center = cluster_centers.chip(cluster_index, 0);

                const type distance = euclidean_distance(row, center);

                if (distance < minimum_distance)
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

            for (Index row_index = 0; row_index < rows_number; row_index++)
            {
                if (rows_cluster_labels(row_index) == cluster_index)
                {
                    row = data.chip(row_index, 0);

                    center_sum += row;
                    count++;
                }
            }

            if (count != 0)
            {
                center = center_sum / static_cast<type>(count);
                cluster_centers.chip(cluster_index, 0) = center;
            }

        }
    }
}


Tensor<Index, 1> KMeans::predict(const Tensor<type, 2>& data)
{
    const Index rows_number = data.dimension(0);
    Tensor<type, 1> row(data.dimension(1));
    Tensor<type, 1> center;

    Tensor<Index, 1> predictions(rows_number);

    for (Index row_index = 0; row_index < rows_number; row_index++)
    {
        row = data.chip(row_index, 0);
        center = cluster_centers.chip(0, 0);

        type minimum_distance = euclidean_distance(row, center);
        Index minimal_distance_cluster_index = 0;

        for(Index cluster_index = 1; cluster_index < clusters_number; cluster_index++)
        {
            center = cluster_centers.chip(cluster_index, 0);
            const type distance = euclidean_distance(row, center);

            if (distance < minimum_distance)
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
    Tensor<type, 1> sum_squared_error_values(max_clusters);
    Tensor<type,1> data_point;
    Tensor<type,1> cluster_center;

    const Index rows_number = data.dimension(0);

    Index original_clusters_number = clusters_number;

    for (Index cluster_index = 1; cluster_index <= max_clusters; cluster_index++)
    {
        clusters_number = cluster_index;
        fit(data);

        type sum_squared_error = 0;

        for (Index row_index = 0; row_index < rows_number; row_index++)
        {
            data_point = data.chip(row_index, 0);
            cluster_center = cluster_centers.chip(rows_cluster_labels(row_index), 0);

            sum_squared_error += pow(euclidean_distance(data_point, cluster_center), 2);
        }

        sum_squared_error_values(cluster_index-1) = sum_squared_error;
    }

    clusters_number = original_clusters_number;
    return sum_squared_error_values;
}


Index KMeans::find_optimal_clusters(const Tensor<type, 1>& sum_squared_error_values)
{

    if (clusters_number < 2) return 0;

    type max_slope = 0;
    Index optimal_clusters_index = 0;

    for (Index clusters_index = 1; clusters_index < clusters_number - 1; clusters_index++)
    {
        type slope = sum_squared_error_values(clusters_index) - sum_squared_error_values(clusters_index + 1);

        if (slope > max_slope)
        {
            max_slope = slope;
            optimal_clusters_index = clusters_index;
        }
    }

    return optimal_clusters_index + 1;
}


Tensor<type, 2>  KMeans::get_cluster_centers()
    {
        return cluster_centers;
    }


Tensor<Index, 1>  KMeans::get_cluster_labels()
    {
        return rows_cluster_labels;
    }


void KMeans::set_centers_random(const Tensor<type, 2>& data)
{
    Index data_size = data.dimension(0);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> index_distribution(0, data_size - 1);

    for (Index i = 0; i < clusters_number; i++)
    {
        cluster_centers.chip(i, 0) = data.chip(index_distribution(gen), 0);
    }
}

}
