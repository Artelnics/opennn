//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   K - M E A N S   C L A S S                                             
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "k_means.h"

namespace OpenNN
{

/// @todo

KMeans::Results KMeans::calculate_k_means(const Tensor<type, 2>& matrix, const int& k) const
{
    Results k_means_results;
/*
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    vector<Tensor<int, 1>> clusters(k);

    Tensor<type, 2> previous_means(k, columns_number);
    Tensor<type, 2> means(k, columns_number);

    const Tensor<type, 1> minimums = OpenNN::columns_minimums(matrix);
    const Tensor<type, 1> maximums = OpenNN::columns_maximums(matrix);

    int iterations = 0;

    bool end = false;

    // Calculate initial means

    Tensor<int, 1> selected_rows(k);

    const int initial_center = calculate_random_uniform<int>(0, rows_number);

    previous_means.set_row(0, matrix.get_row(initial_center));
    selected_rows[0] = initial_center;

    for(int i = 1; i < k; i++)
    {
        Tensor<type, 1> minimum_distances(rows_number, 0.0);

        for(int j = 0; j < rows_number; j++)
        {
            Tensor<type, 1> distances(i, 0.0);

            const Tensor<type, 1> row_data = matrix.get_row(j);

            for(int l = 0; l < i; l++)
            {
                distances[l] = euclidean_distance(row_data, previous_means.get_row(l));
            }

            const double minimum_distance = minimum(distances);

            minimum_distances[static_cast<int>(j)] = minimum_distance;
        }

        int sample_index = calculate_sample_index_proportional_probability(minimum_distances);

        int random_failures = 0;

        while(selected_rows.contains(sample_index))
        {
            sample_index = calculate_sample_index_proportional_probability(minimum_distances);

            random_failures++;

            if(random_failures > 5)
            {
                Tensor<type, 1> new_row(columns_number);

                new_row.setRandom(minimums, maximums);

                previous_means.set_row(i, new_row);

                break;
            }
        }

        if(random_failures <= 5)
        {
            previous_means.set_row(i, matrix.get_row(sample_index));
        }
    }

    // Main loop

    while(!end)
    {
        clusters.clear();
        clusters.set(k);

 #pragma omp parallel for

        for(int i = 0; i < rows_number; i++)
        {
            Tensor<type, 1> distances(k, 0.0);

            const Tensor<type, 1> current_row = matrix.get_row(i);

            for(int j = 0; j < k; j++)
            {
                distances[j] = euclidean_distance(current_row, previous_means.get_row(j));
            }

            const int minimum_distance_index = minimal_index(distances);

  #pragma omp critical
            clusters[minimum_distance_index].push_back(static_cast<int>(i));
        }

        for(int i = 0; i < k; i++)
        {
            means.set_row(i, rows_means(matrix, clusters[i]));
        }

        if(previous_means == means)
        {
            end = true;
        }
        else if(iterations > 100)
        {
            end = true;
        }

        previous_means = means;
        iterations++;
    }

//    k_means_results.means = means;
    k_means_results.clusters = clusters;
*/
    return k_means_results;
}


int KMeans::calculate_sample_index_proportional_probability(const Tensor<type, 1>& vector) const
{
/*
    const int this_size = vector.size();

    Tensor<type, 1> cumulative = OpenNN::cumulative(vector);

    const double sum = vector.calculate_sum();

    const double random = calculate_random_uniform(0.,sum);

    int selected_index = 0;

    for(int i = 0; i < this_size; i++)
    {
        if(i == 0 && random < cumulative[0])
        {
            selected_index = i;
            break;
        }
        else if(random < cumulative[i] && random >= cumulative[i-1])
        {
            selected_index = i;
            break;
        }
    }

    return selected_index;
*/
    return 0;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
