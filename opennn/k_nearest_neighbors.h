/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   K   N E A R E S T   N E I G H B O R S   H E A D E R                                                        */
/*                                                                                                              */
/*   Javier Sanchez, Alberto Quesada                                                                            */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   javiersanchez@artelnics.com, albertoquesada@artelnics.com                                                  */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef K_NEAREST_NEIGHBORS_H
#define K_NEAREST_NEIGHBORS_H

// System includes

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <ctime>
#include <exception>

// OpenNN includes

#include "opennn.h"

// TinyXml includes

#include "tinyxml2.h"

namespace OpenNN
{

class KNearestNeighbors
{

public:

    enum ScalingMethod{Softmax, Unitary, MinMax, MeanStd};
    enum DistanceMethod{Euclidean, Manhattan};

    // DEFAULT CONSTRUCTOR

    explicit KNearestNeighbors();

    // DATASET CONSTRUCTOR

    explicit KNearestNeighbors(DataSet*);

    // DESTRUCTOR

    virtual ~KNearestNeighbors();

    // STRUCTURES

    struct ShortNeighbors
    {
        explicit ShortNeighbors() {}

        virtual ~ShortNeighbors() {}

        Matrix<double> distances;
        Matrix<size_t> indices;
    };

    struct LongNeighbors
    {
        explicit LongNeighbors() {}

        virtual ~LongNeighbors() {}

        Matrix<double> distances;
        Matrix<size_t> indices;
        Vector< Matrix<double> > distances_matrix;
    };

    // METHODS

    // Get methods

    size_t get_k() const;

    // Set methods

    void set_dataset(DataSet*);

    void set_k(const size_t&);

    void set_weights(const Matrix<double>&);

    void set_scaling_method(const ScalingMethod&);

    void set_scaling_method(const string&);

    void set_distance_method(const DistanceMethod&);

    void set_distance_method(const string&);

    // Algorithm methods

    Matrix<double> calculate_correlation_weights(void) const;

    Vector<double> calculate_distances_weights(const Vector<double>&) const;

    ShortNeighbors calculate_k_nearest_neighbors_unsupervised(const Vector<double>&) const;
    LongNeighbors calculate_long_k_nearest_neighbors_unsupervised(const Vector<double>&) const;

    ShortNeighbors calculate_k_nearest_neighbors_supervised(const Vector<double>&) const;

    // Output methods

    Vector<double> calculate_outputs(const Vector<double>&) const;

    Vector<double> calculate_outputs(const ShortNeighbors&) const;

    // Output data methods

    Matrix<double> calculate_outputs(const Matrix<double>&) const;

    Matrix<double> calculate_selection_outputs(void) const;

    Matrix<double> calculate_testing_outputs(void) const;

    // Error methods

    Vector< Statistics<double> > calculate_testing_error_statistics(void) const;

    Vector< Statistics<double> > calculate_selection_error_statistics(void) const;

    Vector< Statistics<double> > calculate_error_statistics(const Matrix<double>&, const Matrix<double>&) const;

    // Selection methods

    void perform_k_selection(const size_t&, const size_t&);

    // Reachability distances methods (outliers)

    Matrix<double> calculate_reachability_distances(const Matrix<double>&, const Vector<double>&) const;

    Vector<double> calculate_reachability_density(const Matrix<double>&) const;

private:

    DataSet* data_set_pointer;

    size_t k;

    Matrix<double> weights;

    ScalingMethod scaling_method;
    DistanceMethod distance_method;
};

}

#endif


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
