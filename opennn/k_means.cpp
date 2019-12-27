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

KMeans::Results KMeans::calculate_k_means(const Matrix<double>& matrix, const size_t& k) const
{
    Results k_means_results;

    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    Vector<Vector<size_t>> clusters(k);

    Matrix<double> previous_means(k, columns_number);
    Matrix<double> means(k, columns_number);

    const Vector<double> minimums = OpenNN::columns_minimums(matrix);
    const Vector<double> maximums = OpenNN::columns_maximums(matrix);

    size_t iterations = 0;

    bool end = false;

    // Calculate initial means

    Vector<size_t> selected_rows(k);

    const size_t initial_center = calculate_random_uniform<size_t>(0, rows_number);

    previous_means.set_row(0, matrix.get_row(initial_center));
    selected_rows[0] = initial_center;

    for(size_t i = 1; i < k; i++)
    {
        Vector<double> minimum_distances(rows_number, 0.0);

        for(size_t j = 0; j < rows_number; j++)
        {
            Vector<double> distances(i, 0.0);

            const Vector<double> row_data = matrix.get_row(j);

            for(size_t l = 0; l < i; l++)
            {
                distances[l] = euclidean_distance(row_data, previous_means.get_row(l));
            }

            const double minimum_distance = minimum(distances);

            minimum_distances[static_cast<size_t>(j)] = minimum_distance;
        }

        size_t sample_index = calculate_sample_index_proportional_probability(minimum_distances);

        int random_failures = 0;

        while(selected_rows.contains(sample_index))
        {
            sample_index = calculate_sample_index_proportional_probability(minimum_distances);

            random_failures++;

            if(random_failures > 5)
            {
                Vector<double> new_row(columns_number);

                new_row.randomize_uniform(minimums, maximums);

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

        for(size_t i = 0; i < rows_number; i++)
        {
            Vector<double> distances(k, 0.0);

            const Vector<double> current_row = matrix.get_row(i);

            for(size_t j = 0; j < k; j++)
            {
                distances[j] = euclidean_distance(current_row, previous_means.get_row(j));
            }

            const size_t minimum_distance_index = minimal_index(distances);

  #pragma omp critical
            clusters[minimum_distance_index].push_back(static_cast<size_t>(i));
        }

        for(size_t i = 0; i < k; i++)
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

    return k_means_results;
}


size_t KMeans::calculate_sample_index_proportional_probability(const Vector<double>& vector) const
{
    const size_t this_size = vector.size();

    Vector<double> cumulative = OpenNN::cumulative(vector);

    const double sum = vector.calculate_sum();

    const double random = calculate_random_uniform(0.,sum);

    size_t selected_index = 0;

    for(size_t i = 0; i < this_size; i++)
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
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
