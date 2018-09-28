/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   K   N E A R E S T   N E I G H B O R S   C L A S S                                                          */
/*                                                                                                              */
/*   Javier Sanchez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   javiersanchez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

#include "k_nearest_neighbors.h"

namespace OpenNN
{

// CONSTRUCTOR

KNearestNeighbors::KNearestNeighbors()
{

}

// DataSet constructor

/// Initializes an instances using a DataSet.
/// @param new_data_set_pointer Pointer to the DataSet.

KNearestNeighbors::KNearestNeighbors(DataSet* new_data_set_pointer)
{
    data_set_pointer = new_data_set_pointer;
}

// DESTRUCTOR

KNearestNeighbors::~KNearestNeighbors()
{
}

// METHODS

// Get Methods

// size_t get_k() const

/// Returns the k-nearest neighbors number parameter

size_t KNearestNeighbors::get_k() const
{
    return k;
}

// Set Methods

void KNearestNeighbors::set_dataset(DataSet* dataset)
{
    data_set_pointer = dataset;
}

void KNearestNeighbors::set_k(size_t k_new)
{
    k = k_new;
}


// Vector<double> calculate_distances(const size_t&) const

/// Returns a matrix with the distances between every instance and the rest of the instances.
/// The number of rows is the number of instances in the data set.
/// The number of columns is the number of instances in the data set.

Matrix<double> KNearestNeighbors::calculate_instances_distances() const
{
    const Instances& instances = data_set_pointer->get_instances();

    const Variables& variables = data_set_pointer->get_variables();

    const Matrix<double>& data = data_set_pointer->get_data().delete_columns(variables.arrange_targets_indices());

    const size_t instances_number = instances.count_used_instances_number();
    const Vector<size_t> instances_indices = instances.arrange_used_indices();

    Matrix<double> distances(instances_number, instances_number, 0.0);

    Vector<double> instance;
    Vector<double> other_instance;

    for(size_t i = 0; i < instances_number; i++)
    {
        for(size_t j = 0; j < instances_number; j++)
        {
            distances(i,j) = data.calculate_distance(instances_indices[i], instances_indices[j]);
        }
    }

    return(distances);
}


//Vector<double> calculate_instance_distances(const Vector<double>& input) const

//// Returns the distance from a new instance to the rest of instances in the dataset
//// @param input New instance vector

Vector<double> KNearestNeighbors::calculate_instance_distances(const Vector<double>& input) const
{
    const Instances& instances = data_set_pointer->get_instances();

    const Variables& variables = data_set_pointer->get_variables();

    Matrix<double> data = data_set_pointer->get_data().delete_columns(variables.arrange_targets_indices());

    data.append_row(input);

    const size_t instances_number = instances.count_used_instances_number();
    const Vector<size_t> instances_indices = instances.arrange_used_indices();

    Vector<double> distances(instances_number, 0.0);

    #pragma omp parallel for

    for(int i=0; i<instances_number; i++)
    {
        distances[i] = input.calculate_distance(data.get_row(i));
    }

    return(distances);

}


// Matrix<size_t> calculate_k_nearest_neighbors(const Matrix<double>&) const

/// Returns a matrix with the k-nearest neighbors to every used instance in the data set.
/// Number of rows is the number of isntances in the data set.
/// Number of columns is the number of nearest neighbors to calculate.
/// @param distances Distances between every instance and the rest of them.

Matrix<size_t> KNearestNeighbors::calculate_k_nearest_neighbors(const Matrix<double>& distances) const
{

    const Instances& instances = data_set_pointer->get_instances();

    const size_t instances_number = instances.count_used_instances_number();

    Matrix<size_t> nearest_neighbors(instances_number, k);

    #pragma omp parallel for

    for(int i = 0; i < instances_number; i++)
    {
        const Vector<double> instance_distances = distances.get_row(i);

        const Vector<size_t> minimal_distances_indices = instance_distances.calculate_minimal_indices(k + 1);

        for(size_t j = 0; j < k; j++)
        {
            nearest_neighbors(i, j) = minimal_distances_indices[j + 1];
        }

        if(minimal_distances_indices[1] == i)
        {
            nearest_neighbors(i, 0) = minimal_distances_indices[0];
        }
    }

    return(nearest_neighbors);
}


//Vector<size_t> calculate_k_nearest_neighbors(const Vector<double>& distances) const

//// Returns the K-nearest neighbors for a single instance
//// @param distances Distance from the new instance to the other instances in the matrix

Vector<size_t> KNearestNeighbors::calculate_k_nearest_neighbors(const Vector<double>& distances) const
{
    const Instances& instances = data_set_pointer->get_instances();

    const size_t instances_number = instances.count_used_instances_number();

    const Vector<size_t> nearest_neighbors = distances.calculate_k_minimal_indices(k);

    return(nearest_neighbors);
}


// Vector<double> calculate_k_distances(const Matrix<double>&) const

/// Returns a vector with the k-distance of every instance in the data set, which is the distance between every
/// instance and k-th nearest neighbor.
/// @param distances Distances between every instance in the data set.

Vector<double> KNearestNeighbors::calculate_k_distances(const Matrix<double>& distances) const
{
    const size_t instances_number = instances.count_used_instances_number();

    const Matrix<size_t> nearest_neighbors = calculate_k_nearest_neighbors(distances);

    Vector<double> k_distances(instances_number);

    #pragma omp parallel for

    for(int i = 0; i < instances_number; i++)
    {
        const size_t maximal_index = nearest_neighbors(i, k - 1);

        k_distances[i] = distances.get_row(i)[maximal_index];
    }

    return(k_distances);
}


// Matrix<double> calculate_reachability_distance(const Matrix<double>&, Vector<double>&) const

/// Calculates the reachability distances for the instances in the data set.
/// @param distances Distances between every instance.

Matrix<double> KNearestNeighbors::calculate_reachability_distances(const Matrix<double>& distances, const Vector<double>& k_distances) const
{
    const size_t instances_number = instances.count_used_instances_number();

    Matrix<double> reachability_distances(instances_number, instances_number);

    for(size_t i = 0; i < instances_number; i++)
    {
        for(size_t j = i; j < instances_number; j++)
        {
            if(distances(i, j) <= k_distances[i])
            {
                reachability_distances(i, j) = k_distances[i];
                reachability_distances(j, i) = k_distances[i];
            }
            else if(distances(i, j) > k_distances[i])
            {
                reachability_distances(i, j) = distances(i, j);
                reachability_distances(j, i) = distances(j, i);
            }
         }
    }

    return(reachability_distances);
}


// Vector<double> calculate_reachability_density(const Matrix<double>&) const

/// Calculates reachability density for every element of the data set.
/// @param distances Distances between every instance in the data set.

Vector<double> KNearestNeighbors::calculate_reachability_density(const Matrix<double>& distances) const
{
   const size_t instances_number = instances.count_used_instances_number();

   const Vector<double> k_distances = calculate_k_distances(distances);

   const Matrix<double> reachability_distances = calculate_reachability_distances(distances, k_distances);

   const Matrix<size_t> nearest_neighbors_indices = calculate_k_nearest_neighbors(distances);

   Vector<double> reachability_density(instances_number);

   Vector<size_t> nearest_neighbors_instance;

   for(size_t i = 0; i < instances_number; i++)
   {
       nearest_neighbors_instance = nearest_neighbors_indices.get_row(i);

       reachability_density[i] = k/ reachability_distances.get_row(i).calculate_partial_sum(nearest_neighbors_instance);
   }

   return(reachability_density);
}


// Vector<double> calculate_output(const Vector<double>&) const

/// Returns the a vector with the probability of each class for a new instance
/// @param input New instance vector

Vector<double> KNearestNeighbors::calculate_output(const Vector<double>& input) const
{
    const Instances& instances = data_set_pointer->get_instances();

    const size_t instances_number = instances.count_used_instances_number();


    const Matrix<double>& data = data_set_pointer->get_data();

    const Variables& variables = data_set_pointer->get_variables();


    const Vector<double> distances = calculate_instance_distances(input);


    const Vector<size_t> k_neighbors = calculate_k_nearest_neighbors(distances);


    const Vector<size_t> targets_indexes = variables.arrange_targets_indices();

    const size_t targets_number = targets_indexes.size();

    Vector<double> categories(targets_number, 0.0);

    for(int i=0; i<k; i++)
    {
        const Vector<double> neighbor_data = data.get_row(k_neighbors[i]);
        for(size_t j=0; j<targets_number; j++)
        {
            categories[j] += neighbor_data[targets_indexes[j]];
        }
    }

    for(int i=0; i<targets_number; i++)
    {
        categories[i] = categories[i]/k;
    }

    return categories;
}


// Vector<double> calculate_output(const Matrix<double>&) const

/// Returns the probability of a batch of instances
/// @param input_data New data matrix

Matrix<double> KNearestNeighbors::calculate_output_data(const Matrix<double>& input_data) const
{
    const size_t input_instances_number = input_data.get_rows_number();


    Matrix<double> output;

    #pragma omp parallel for

    for(int i=0; i<input_instances_number; i++)
    {
        output.append_row(calculate_output(input_data.get_row(i)));
    }

    return output;

}


// Matrix<double> calculate_testing_output() const

/// Returns the classification probabilities of the testing instances of the DataSet
Matrix<double> KNearestNeighbors::calculate_testing_output() const
{
    const Matrix<double> testing_data = data_set_pointer->arrange_testing_target_data();

    const Matrix<double> testing_output_data = calculate_output_data(testing_data);

    return(testing_output_data);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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
