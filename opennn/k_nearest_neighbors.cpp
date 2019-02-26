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

/// It creates a k-nearest neighbor object and initializes the rest of class members to their default values.

KNearestNeighbors::KNearestNeighbors()
{
    data_set_pointer = nullptr;

    weights.set(1, 1, 0);

    scaling_method = Softmax;

    distance_method = Euclidean;

    k = 3;
}

// DataSet constructor

/// Initializes an instances using a DataSet.
/// @param new_data_set_pointer Pointer to the DataSet.

KNearestNeighbors::KNearestNeighbors(DataSet* new_data_set_pointer)
{
    data_set_pointer = new_data_set_pointer;

    weights.set(data_set_pointer->get_data().get_columns_number(), 1, 1);

    scaling_method = Softmax;

    distance_method = Euclidean;

    k = 3;
}

// DESTRUCTOR

/// K-nearest neighbor object destructor

KNearestNeighbors::~KNearestNeighbors()
{
}

// METHODS

// Get Methods

/// Returns the k-nearest neighbors number parameter

size_t KNearestNeighbors::get_k() const
{
    return k;
}

// Set Methods

/// Set the dataset pointer.
/// @param dataset Pointer to the DataSet.

void KNearestNeighbors::set_dataset(DataSet* dataset)
{
    data_set_pointer = dataset;
}


/// Set the k-nearest neighbors number parameter.
/// @param k_new is the new k-nearest neighbors number parameter.

void KNearestNeighbors::set_k(const size_t& k_new)
{
    k = k_new;
}


/// Set the matrix of weights to calculate the distances with weighted attributes.
/// @param new_weights is the new matrix of weights.

void KNearestNeighbors::set_weights(const Matrix<double>& new_weights)
{
    weights = new_weights;
}


/// Set the normalization method used for calculations.
/// @param new_method (enum Method) is the normalization method selected for the analysis.

void KNearestNeighbors::set_scaling_method(const ScalingMethod& new_method)
{
    scaling_method = new_method;
}


/// Set the normalization method used for calculations.
/// @param new_method (String) is the normalization method selected for the analysis.

void KNearestNeighbors::set_scaling_method(const string& new_method_string)
{
    if(new_method_string == "Softmax")
    {
        scaling_method = Softmax;
    }
    else if(new_method_string == "Unitary")
    {
        scaling_method = Unitary;
    }
    else if(new_method_string == "MinMax")
    {
        scaling_method = MinMax;
    }
    else if(new_method_string == "MeanStd")
    {
        scaling_method = MeanStd;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: KNearestNeighbors class.\n"
               << "void set_method(const string&) method.\n"
               << "Unknown method: " << new_method_string << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Set the distance method used for KNN.
/// @param new_method (enum Method) is the distance method selected for the analysis.

void KNearestNeighbors::set_distance_method(const DistanceMethod& new_method)
{
    distance_method = new_method;
}


/// Set the distance method used for KNN.
/// @param new_method (String) is the distance method selected for the analysis.

void KNearestNeighbors::set_distance_method(const string& new_method_string)
{
    if(new_method_string == "Euclidean")
    {
        distance_method = Euclidean;
    }
    else if(new_method_string == "Manhattan")
    {
        distance_method = Manhattan;
    }
    else
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: KNearestNeighbors class.\n"
               << "void set_distance_method(const string&) method.\n"
               << "Unknown method: " << new_method_string << ".\n";

        throw logic_error(buffer.str());
    }
}


/// Returns the matrix of attributes weights

Matrix<double> KNearestNeighbors::calculate_correlation_weights(void) const
{
    const Matrix<double> correlations = data_set_pointer->calculate_input_target_correlations().calculate_absolute_value();

    Matrix<double> scaled_correlations;

    if (scaling_method == Softmax)
    {
        scaled_correlations = correlations.calculate_softmax_columns();
    }
    else if (scaling_method == Unitary)
    {
        scaled_correlations = correlations.calculate_normalized_columns();
    }
    else if (scaling_method == MinMax)
    {
        scaled_correlations = correlations.calculate_scaled_minimum_maximum_0_1_columns();
    }
    else if (scaling_method == MeanStd)
    {
        scaled_correlations = correlations.calculate_scaled_mean_standard_deviation_columns();
    }

    return scaled_correlations;
}


/// Returns the distances weighted
/// @param k_nearest_distances is the vector of the k-nearest distances

Vector<double> KNearestNeighbors::calculate_distances_weights(const Vector<double>& k_nearest_distances) const
{
    Vector<double> distances_weights = k_nearest_distances.calculate_reverse_scaling();

    if (scaling_method == Softmax)
    {
        distances_weights = distances_weights.calculate_softmax();
    }
    else if (scaling_method == Unitary)
    {
        distances_weights = distances_weights.calculate_normalized();
    }
    else if (scaling_method == MinMax)
    {
        distances_weights = distances_weights.calculate_scaled_minimum_maximum_0_1();
    }
    else if (scaling_method == MeanStd)
    {
        distances_weights = distances_weights.calculate_scaled_mean_standard_deviation();
    }

    return distances_weights;
}


/// Calculates a set of outputs from the KNN in response to a set of inputs.
/// The format is a matrix, where each row contains the output for a single input.
/// @param inputs Matrix of inputs to the KNN.

Matrix<double> KNearestNeighbors::calculate_outputs(const Matrix<double>& input_data) const
{
    const size_t instances_number = input_data.get_rows_number();
    const size_t targets_number = data_set_pointer->get_variables().get_targets_number();

    Matrix<double> output_data(instances_number, targets_number, 0.0);

    #pragma omp parallel for

    for(int i = 0; i < static_cast<int>(instances_number); i++)
    {
        const Vector<double> inputs = input_data.get_row(static_cast<size_t>(i));

        const Vector<double> outputs = calculate_outputs(inputs);

        output_data.set_row(static_cast<size_t>(i), outputs);
    }

    return output_data;
}


/// Calculates a set of outputs from the KNN in response to the selection instances.
/// The format is a matrix, where each row contains the output for a single selection instance.

Matrix<double> KNearestNeighbors::calculate_selection_outputs(void) const
{
    return calculate_outputs(data_set_pointer->get_selection_inputs());
}


/// Calculates a set of outputs from the KNN in response to the testing instances.
/// The format is a matrix, where each row contains the output for a single testing instance.

Matrix<double> KNearestNeighbors::calculate_testing_outputs(void) const
{
    return calculate_outputs(data_set_pointer->get_testing_inputs());
}


/// Calculates a set of outputs from the KNN in response to a single input instance.
/// The format is a vector, where each element contains the output associated to each target.
/// @param inpus Vector of inputs to the KNN.

Vector<double> KNearestNeighbors::calculate_outputs(const Vector<double>& inputs) const
{
    const ShortNeighbors k_nearest_neighbors = calculate_k_nearest_neighbors_supervised(inputs);

    const Vector<double> outputs = calculate_outputs(k_nearest_neighbors);

    return outputs;
}


/// Calculates the neighbors structure from the KNN in response to a single input instance.
/// Neighbors structure contains k_nearest_distances matrix and k_nearest_neighbors matrix.
/// @param inpus Vector of inputs to the KNN.

KNearestNeighbors::ShortNeighbors KNearestNeighbors::calculate_k_nearest_neighbors_supervised(const Vector<double>& inputs) const
{
    const Matrix<double> training_inputs = data_set_pointer->get_training_inputs();

    const size_t targets_number = data_set_pointer->get_variables().get_targets_number();

    Matrix<double> k_nearest_distances(k,targets_number);
    Matrix<size_t> k_nearest_neighbors(k,targets_number);

    for (size_t i = 0; i < targets_number; i++)
    {
        Vector<double> weighted_distances;

        if (distance_method == Euclidean)
        {
            weighted_distances = training_inputs.calculate_euclidean_weighted_distance(inputs, weights.get_column(i));
        }
        else if (distance_method == Manhattan)
        {
            weighted_distances = training_inputs.calculate_manhattan_weighted_distance(inputs, weights.get_column(i));
        }

        const Vector<size_t> nearest_neighbors = weighted_distances.calculate_lower_indices(k);

        const Vector<double> nearest_distances = weighted_distances.calculate_lower_values(k);

        k_nearest_distances.set_column(i,nearest_distances);
        k_nearest_neighbors.set_column(i,nearest_neighbors);
    }

    ShortNeighbors neighbors;
    neighbors.distances = k_nearest_distances;
    neighbors.indices = k_nearest_neighbors;

    return neighbors;
}


/// Calculates the neighbors structure from the KNN in response to a single input instance.
/// Neighbors structure contains k_nearest_distances matrix and k_nearest_neighbors matrix.
/// @param inpus Vector of inputs to the KNN.

KNearestNeighbors::ShortNeighbors KNearestNeighbors::calculate_k_nearest_neighbors_unsupervised(const Vector<double>& inputs) const
{
    const Matrix<double> training_inputs = data_set_pointer->get_training_inputs();

    Matrix<double> k_nearest_distances(k,1);
    Matrix<size_t> k_nearest_neighbors(k,1);

    Vector<double> weighted_distances;

    if (distance_method == Euclidean)
    {
        weighted_distances = training_inputs.calculate_euclidean_weighted_distance(inputs, weights.get_column(0));
    }
    else if (distance_method == Manhattan)
    {
        weighted_distances = training_inputs.calculate_manhattan_weighted_distance(inputs, weights.get_column(0));
    }

    const Vector<size_t> nearest_neighbors = weighted_distances.calculate_lower_indices(k);

    const Vector<double> nearest_distances = weighted_distances.calculate_lower_values(k);

    k_nearest_distances.set_column(0,nearest_distances);
    k_nearest_neighbors.set_column(0,nearest_neighbors);

    ShortNeighbors neighbors;
    neighbors.distances = k_nearest_distances;
    neighbors.indices = k_nearest_neighbors;

    return neighbors;
}


KNearestNeighbors::LongNeighbors KNearestNeighbors::calculate_long_k_nearest_neighbors_unsupervised(const Vector<double>& inputs) const
{
    const Matrix<double> training_inputs = data_set_pointer->get_training_inputs();

    Vector< Matrix<double> > k_nearest_distances_matrix(1);
    Matrix<double> k_nearest_distances(k,1);
    Matrix<size_t> k_nearest_neighbors(k,1);

    Matrix<double> weighted_distances_matrix;
    Vector<double> weighted_distances;

    if (distance_method == Euclidean)
    {
        weighted_distances_matrix = training_inputs.calculate_euclidean_weighted_distance_matrix(inputs, weights.get_column(0));

        weighted_distances = weighted_distances_matrix.calculate_rows_sum().calculate_square_root_elements();
    }
    else if (distance_method == Manhattan)
    {
        weighted_distances_matrix = training_inputs.calculate_manhattan_weighted_distance_matrix(inputs, weights.get_column(0));

        weighted_distances = weighted_distances_matrix.calculate_rows_sum();
    }

    const Vector<size_t> nearest_neighbors = weighted_distances.calculate_lower_indices(k);

    const Vector<double> nearest_distances = weighted_distances.calculate_lower_values(k);

    k_nearest_distances_matrix[0] = weighted_distances_matrix.sort_rank_rows(nearest_neighbors);
    k_nearest_distances.set_column(0,nearest_distances);
    k_nearest_neighbors.set_column(0,nearest_neighbors);

    LongNeighbors neighbors;
    neighbors.distances_matrix = k_nearest_distances_matrix;
    neighbors.distances = k_nearest_distances;
    neighbors.indices = k_nearest_neighbors;

    return neighbors;
}


/// Calculates a set of outputs from the KNN in response to a Neighbors structure.
/// The format is a vector, where each element contains the output associated to each target.
/// @param k_nearest_neighbors is the Neighbors structure that contains k_nearest_distances matrix and k_nearest_neighbors matrix.

Vector<double> KNearestNeighbors::calculate_outputs(const ShortNeighbors& k_nearest_neighbors) const
{
    const Matrix<double> targets = data_set_pointer->get_training_targets();

    const size_t targets_number = data_set_pointer->get_variables().get_targets_number();

    Vector<double> output(targets_number,0.0);

    for (size_t i = 0; i < targets_number; i++)
    {
        const Vector<size_t> current_k_nearest_neighbors = k_nearest_neighbors.indices.get_column(i);

        const Vector<double> k_nearest_targets = targets.get_column(i).get_subvector(current_k_nearest_neighbors);

        const Vector<double> k_distances_weights = calculate_distances_weights(k_nearest_neighbors.distances.get_column(i));

        const double norm = k_distances_weights.calculate_sum();

        output[i] = (k_nearest_targets*k_distances_weights).calculate_sum()/norm;
    }

    return output;
}


/// Returns the vector of error statistics for the testing instances.
/// The format is a vector, where each element contains the error statistics associated to each target.

Vector< Statistics<double> > KNearestNeighbors::calculate_testing_error_statistics(void) const
{
    const Matrix<double> testing_targets = data_set_pointer->get_testing_targets();

    const Matrix<double> testing_outputs = calculate_testing_outputs();

    Vector< Statistics<double> > testing_error = calculate_error_statistics(testing_targets,testing_outputs);

    return testing_error;
}


/// Returns the vector of error statistics for the selection instances.
/// The format is a vector, where each element contains the error statistics associated to each target.

Vector< Statistics<double> > KNearestNeighbors::calculate_selection_error_statistics(void) const
{
    const Matrix<double> selection_targets = data_set_pointer->get_selection_targets();

    const Matrix<double> selection_outputs = calculate_selection_outputs();

    Vector< Statistics<double> > selection_error = calculate_error_statistics(selection_targets,selection_outputs);

    return selection_error;
}


/// Returns the vector of error statistics.
/// The format is a vector, where each element contains the error statistics associated to each target.
/// @param targets is the matrix of targets.
/// @param outputs is the matrix of outputs.

Vector< Statistics<double> > KNearestNeighbors::calculate_error_statistics(const Matrix<double>& targets,
                                                                           const Matrix<double>& outputs) const
{
    TestingAnalysis testing_analysis(data_set_pointer);

    Vector< Statistics<double> > error_statistics = testing_analysis.calculate_percentage_errors_statistics(targets,
                                                                                                           outputs);

    return error_statistics;
}


/// Sets the optimal k-nearest neighbors number by performing a k selection algorithm.
/// It calculates the mean selection error by performing the KNN with the parameter k sets from @param first to @param last values.
/// Then set the k-nearest neighbors number associated to the minimum mean selection error.
/// @param first is the initial k value.
/// @param last is the final k value.

void KNearestNeighbors::perform_k_selection(const size_t& first, const size_t& last)
{
    set_k(first);

    size_t optimal_k = first;

    double optimal_error = calculate_selection_error_statistics()[0].mean;

    for(size_t i = first+1; i < last+1; i++)
    {
        set_k(i);

        const double selection_errors_mean = calculate_selection_error_statistics()[0].mean;

        if (selection_errors_mean < optimal_error)
        {
            optimal_k = i;
            optimal_error = selection_errors_mean;
        }
    }

    set_k(optimal_k);
}

/*
/// Calculates the reachability distances for the instances in the data set.
/// @param distances Distances between every instance.

Matrix<double> KNearestNeighbors::calculate_reachability_distances(const Matrix<double>& distances, const Vector<double>& k_distances) const
{
    const size_t instances_number = data_set_pointer->get_instances_pointer()->get_used_instances_number();

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


/// Calculates reachability density for every element of the data set.
/// @param distances Distances between every instance in the data set.

Vector<double> KNearestNeighbors::calculate_reachability_density(const Matrix<double>& distances) const
{
   const size_t instances_number = data_set_pointer->get_instances_pointer()->get_used_instances_number();

   const Vector<double> k_distances = calculate_distances(distances); // @todo

   const Matrix<double> reachability_distances = calculate_reachability_distances(distances, k_distances);

   const Matrix<size_t> nearest_neighbors_indices = calculate_k_nearest_neighbors(distances);

   Vector<double> reachability_density(instances_number);

   Vector<size_t> nearest_neighbors_instance;

   for(size_t i = 0; i < instances_number; i++)
   {
       nearest_neighbors_instance = nearest_neighbors_indices.get_row(i);

       reachability_density[i] = k/reachability_distances.get_row(i).calculate_partial_sum(nearest_neighbors_instance);
   }

   return(reachability_density);
}
*/

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
