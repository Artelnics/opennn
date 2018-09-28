/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   A S S O C I A T I O N   R U L E S   C L A S S   H E A D E R                                                */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

#ifndef __ASSOCIATIONRULES_H__
#define __ASSOCIATIONRULES_H__

// System includes

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>

// OpenNN includes

#include "vector.h"
#include "matrix.h"

// TinyXml includes

#include "tinyxml2.h"


namespace OpenNN
{

class AssociationRules
{

public:

    // DEFAULT CONSTRUCTOR

    explicit AssociationRules();

    // DESTRUCTOR

    virtual ~AssociationRules();

    // GET METHODS

    SparseMatrix<int> get_sparse_matrix() const;

    double get_minimum_support() const;

    double get_maximum_time() const;

    const bool& get_display() const;

    // SET METHODS

    void set_sparse_matrix(const SparseMatrix<int>&);

    void set_minimum_support(const double&);

    void set_maximum_time(const double&);

    void set_display(const bool&);

    // Auxiliar methods

    unsigned long long calculate_combinations_number(const size_t&, const size_t&) const;

    Matrix<size_t> calculate_combinations(const Vector<size_t>&, const size_t&) const;

    // Association rules

    Matrix<double> calculate_support(const size_t&, const Vector<size_t>& = Vector<size_t>()) const;

    Matrix<double> calculate_confidence(const size_t&, const size_t&, const Vector<size_t>& = Vector<size_t>()) const;

    Matrix<double> calculate_lift(const size_t&, const size_t&, const Vector<size_t>& = Vector<size_t>()) const;

    // Algorithms

    Vector< Matrix<double> > perform_a_priori_algorithm(const size_t& = 0);

private:

    // MEMBERS

    /// Display messages to screen.

    bool display;

    double minimum_support = 0.1;

    double maximum_time = 1000;

    SparseMatrix<int> sparse_matrix;

    unsigned long long calculate_factorial(const size_t&) const;

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

