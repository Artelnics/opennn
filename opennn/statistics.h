//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A T I S T I C S   H E A D E R
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

struct Descriptives
{
    Descriptives(const type = type(NAN), type = type(NAN), type = type(NAN), type = type(NAN));

    VectorR to_tensor() const;

    void set(const type = type(NAN), type = type(NAN), type = type(NAN), type = type(NAN));

    void save(const filesystem::path&) const;

    void print(const string& = "Descriptives:") const;

    string name = "Descriptives";

    type minimum = type(-1.0);

    type maximum = type(1);

    type mean = type(0);

    type standard_deviation = type(1);

};


struct BoxPlot
{
    BoxPlot(const type = type(NAN),
            type = type(NAN),
            type = type(NAN),
            type = type(NAN),
            type = type(NAN));

    void set(const type = type(NAN),
             type = type(NAN),
             type = type(NAN),
             type = type(NAN),
             type = type(NAN));

    type minimum = type(NAN);

    type first_quartile = type(NAN);

    type median = type(NAN);

    type third_quartile = type(NAN);

    type maximum = type(NAN);
};


struct Histogram
{
    Histogram(const Index = 0);

    Histogram(const VectorR&, const VectorR&);

    Histogram(const VectorR&, const VectorR&, const VectorR&, const VectorR&);

    Histogram(const VectorR&, Index);

    Histogram(const VectorR&);

    // Methods

    Index get_bins_number() const;

    Index count_empty_bins() const;

    Index calculate_minimum_frequency() const;

    Index calculate_maximum_frequency() const;

    Index calculate_most_populated_bin() const;

    VectorR calculate_minimal_centers() const;

    VectorR calculate_maximal_centers() const;

    Index calculate_bin(const type) const;

    Index calculate_frequency(const type) const;

    void save(const filesystem::path&) const;

    VectorR minimums;

    VectorR maximums;

    VectorR centers;

    VectorR frequencies;
};

// Minimum
type minimum(const MatrixR&);
type minimum(const VectorR&, const vector<Index>&);
//Index minimum(const VectorI&);
VectorR column_minimums(const Tensor2&, const vector<Index>& = vector<Index>(), const vector<Index>& = vector<Index>());

// Maximum
type maximum(const MatrixR&);
type maximum(const VectorR&, const vector<Index>&);
//Index maximum(const VectorI&);
VectorR column_maximums(const Tensor2&, const vector<Index>& = vector<Index>(), const vector<Index>& = vector<Index>());

// Range
type range(const VectorR&);

// Mean
type mean(const VectorR&);
type mean(const VectorR&, Index, Index);
type mean(const MatrixR&, Index);
VectorR mean(const MatrixR&);
VectorR mean(const MatrixR&, const vector<Index>&);
VectorR mean(const MatrixR&, const vector<Index>&, const vector<Index>&);

// Median
type median(const VectorR&);
type median(const MatrixR&, Index);
VectorR median(const MatrixR&);
VectorR median(const MatrixR&, const vector<Index>&);
VectorR median(const MatrixR&, const vector<Index>&, const vector<Index>&);

// Variance
type variance(const VectorR&);
type variance(const VectorR&, const VectorI&);

// Standard deviation
type standard_deviation(const VectorR&);
//type standard_deviation(const VectorR&, const VectorI&);
VectorR standard_deviation(const VectorR&, Index);

// Assymetry
type asymmetry(const VectorR&);

// Kurtosis
type kurtosis(const VectorR&);

// Quartiles
VectorR quartiles(const VectorR&);
VectorR quartiles(const VectorR&, const vector<Index>&);

// Box plot
BoxPlot box_plot(const VectorR&);
BoxPlot box_plot(const VectorR&, const vector<Index>&);

// Descriptives vector
Descriptives vector_descriptives(const VectorR&);

// Descriptives matrix
vector<Descriptives> descriptives(const MatrixR&);
vector<Descriptives> descriptives(const MatrixR&, const vector<Index>&, const vector<Index>&);

// Histograms
Histogram histogram(const VectorR&, Index  = 10);
Histogram histogram_centered(const VectorR&, type = type(0), Index  = 10);
Histogram histogram(const VectorB&);
Histogram histogram(const VectorI&, Index  = 10);
vector<Histogram> histograms(const Tensor2&, Index = 10);
VectorI total_frequencies(const vector<Histogram>&);

// Minimal indices
Index minimal_index(const VectorR&);
VectorI minimal_indices(const VectorR&, Index);
VectorI minimal_indices(const MatrixR&);

// Maximal indices
Index maximal_index(const VectorR&);
VectorI maximal_indices(const VectorR&, Index);
VectorI maximal_indices(const MatrixR&);

// Percentiles
VectorR percentiles(const VectorR&);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
