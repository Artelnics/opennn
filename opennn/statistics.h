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
type minimum(const VectorR&);
type minimum(const VectorR&, const vector<Index>&);
VectorR column_minimums(const Tensor2&, const vector<Index>& = vector<Index>(), const vector<Index>& = vector<Index>());

// Maximum
type maximum(const MatrixR&);
type maximum(const VectorR&);
type maximum(const VectorR&, const vector<Index>&);
VectorR column_maximums(const Tensor2&, const vector<Index>& = vector<Index>(), const vector<Index>& = vector<Index>());

// Range
type range(const VectorR&);

// Mean
type mean(const VectorR&);
type mean(const MatrixR&, Index);
VectorR mean(const MatrixR&);
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
VectorR standard_deviation(const VectorR&, Index);

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
vector<Histogram> histograms(const MatrixR&, Index = 10);
VectorI total_frequencies(const vector<Histogram>&);

// Minimal indices
Index minimal_index(const VectorR&);
VectorI minimal_indices(const VectorR&, Index);
VectorI minimal_indices(const MatrixR&);

// Maximal indices
Index maximal_index(const VectorR&);
VectorI maximal_indices(const VectorR&, Index);
VectorI maximal_indices(const MatrixR&);

// =====================================================================
// Eigen Matrix/Vector data-manipulation helpers.
// (Moved here from tensor_utilities.h: these operate on Eigen types, not
// TensorView, and live in the same conceptual neighbourhood as Descriptives
// and the slicing/filtering code that already lives here.)
// =====================================================================

inline bool row_finite(const VectorR& v, Index i) { return isfinite(v(i)); }
inline bool row_finite(const MatrixR& m, Index i) { return m.row(i).array().isFinite().all(); }

inline VectorR slice_rows(const VectorR& v, const vector<Index>& idx)
{
    VectorR r(idx.size());
    for (Index i = 0; i < Index(idx.size()); ++i) r(i) = v(idx[i]);
    return r;
}

inline MatrixR slice_rows(const MatrixR& m, const vector<Index>& idx)
{
    MatrixR r(idx.size(), m.cols());
    for (Index i = 0; i < Index(idx.size()); ++i) r.row(i) = m.row(idx[i]);
    return r;
}

VectorR filter_missing_values(const VectorR&);

template<typename X, typename Y>
pair<X, Y> filter_missing_values(const X& x, const Y& y)
{
    if (x.rows() != y.rows())
        throw runtime_error("filter_missing_values: row count mismatch");

    vector<Index> valid;
    valid.reserve(x.rows());

    for (Index i = 0; i < x.rows(); ++i)
        if (row_finite(x, i) && row_finite(y, i))
            valid.push_back(i);

    return { slice_rows(x, valid), slice_rows(y, valid) };
}

void shuffle_rows(MatrixR& matrix);

inline bool is_contiguous(const vector<Index>& v)
{
    return std::adjacent_find(v.begin(), v.end(),
        [](Index a, Index b) { return b != a + 1; }) == v.end();
}

template <typename T>
inline bool is_binary(const T& tensor)
{
    return all_of(tensor.data(), tensor.data() + tensor.size(),
                  [](type v) { return v == type(0) || v == type(1) || isnan(v); });
}

MatrixR append_rows(const MatrixR&, const MatrixR&);

template<typename T>
vector<T> gather_by_index(const vector<T>& data, const vector<Index>& indices)
{
    vector<T> result;
    result.reserve(indices.size());

    transform(indices.begin(), indices.end(), back_inserter(result),
              [&data](Index i) { return data[i]; });

    return result;
}

vector<Index> build_feasible_rows_mask(const MatrixR& outputs, const VectorR& minimums, const VectorR& maximums);

template <typename T>
inline bool is_constant(const T& tensor)
{
    const type* data = tensor.data();
    const type* end = data + tensor.size();

    const type* first = find_if(data, end, [](type v) { return !isnan(v); });

    if (first == end)
        return true;

    const type val = *first;

    return all_of(first + 1, end,
                  [val](type v) { return isnan(v) || abs(val - v) <= numeric_limits<float>::min(); });
}

inline vector<Index> get_true_indices(const VectorB& v)
{
    vector<Index> indices;
    indices.reserve(v.size());

    for(Index i = 0; i < v.size(); ++i)
        if (v(i))
            indices.push_back(i);

    return indices;
}

VectorI calculate_rank(const VectorR&, bool ascending = true);

vector<Index> get_elements_greater_than(const vector<Index>&, Index);

VectorI get_nearest_points(const MatrixR&, const VectorR&, int = 1);

void fill_tensor_data(const MatrixR&, const vector<Index>&, const vector<Index>&, type*, bool = true, int contiguous = -1);

VectorR perform_Householder_QR_decomposition(const MatrixR&, const VectorR&);

VectorMap vector_map(const MatrixR&, Index);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
