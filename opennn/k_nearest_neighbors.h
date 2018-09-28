/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   K   N E A R E S T   N E I G H B O R S   H E A D E R                                                        */
/*                                                                                                              */
/*   Javier Sanchez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   javiersanchez@artelnics.com                                                                                */
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

#include "vector.h"
#include "matrix.h"

#include "opennn.h"

// TinyXml includes

#include "tinyxml2.h"


namespace OpenNN
{

class KNearestNeighbors
{

public:

    // DEFAULT CONSTRUCTOR

    explicit KNearestNeighbors();

    // DATASET CONSTRUCTOR

    explicit KNearestNeighbors(DataSet* dataset);


    // DESTRUCTOR

    virtual ~KNearestNeighbors();

    /// Instances  object(training, selection and testing instances).

    Instances instances;


    // METHODS

    // Get methods
    size_t get_k() const;


    // Set methods

    void set_dataset(DataSet* dataset);
    void set_k(size_t k);

    // Algorithm methods

    Matrix<double> calculate_instances_distances() const;

    Vector<double> calculate_instance_distances(const Vector<double>& input) const;

    Matrix<size_t> calculate_k_nearest_neighbors(const Matrix<double>& distances) const;

    Vector<size_t> calculate_k_nearest_neighbors(const Vector<double>& distances) const;

    Vector<double> calculate_k_distances(const Matrix<double>& distances) const;

    Matrix<double> calculate_reachability_distances(const Matrix<double>& distances, const Vector<double>& k_distances) const;

    Vector<double> calculate_reachability_density(const Matrix<double>& distances) const;

    Vector<double> calculate_output(const Vector<double>& input) const;

    Matrix<double> calculate_output_data(const Matrix<double>& input_data) const;

    Matrix<double> calculate_testing_output() const;

private:

    size_t k = 3;

    DataSet* data_set_pointer = NULL;

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
