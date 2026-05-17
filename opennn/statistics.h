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
    Descriptives(const float = NAN, float = NAN, float = NAN, float = NAN);

    VectorR to_tensor() const;

    void set(const float = NAN, float = NAN, float = NAN, float = NAN);

    void save(const filesystem::path&) const;

    void print(const string& = "Descriptives:") const;

    string name = "Descriptives";

    float minimum = -1.0f;

    float maximum = 1.0f;

    float mean = 0.0f;

    float standard_deviation = 1.0f;

};

struct BoxPlot
{
    BoxPlot(const float = NAN,
            float = NAN,
            float = NAN,
            float = NAN,
            float = NAN);

    void set(const float = NAN,
             float = NAN,
             float = NAN,
             float = NAN,
             float = NAN);

    float minimum = NAN;

    float first_quartile = NAN;

    float median = NAN;

    float third_quartile = NAN;

    float maximum = NAN;
};

struct Histogram
{
    Histogram(const Index = 0);

    Histogram(const VectorR&, const VectorR&);

    Histogram(const VectorR&, const VectorR&, const VectorR&, const VectorR&);

    Histogram(const VectorR&, Index);

    Histogram(const VectorR&);
    Index get_bins_number() const;

    Index count_empty_bins() const;

    Index calculate_minimum_frequency() const;

    Index calculate_maximum_frequency() const;

    Index calculate_most_populated_bin() const;

    VectorR calculate_minimal_centers() const;

    VectorR calculate_maximal_centers() const;

    Index calculate_bin(const float) const;

    Index calculate_frequency(const float) const;

    void save(const filesystem::path&) const;

    VectorR minimums;

    VectorR maximums;

    VectorR centers;

    VectorR frequencies;
};
float minimum(const MatrixR&);
float minimum(const VectorR&);
float minimum(const VectorR&, const vector<Index>&);
VectorR column_minimums(const Tensor2&, const vector<Index>& = vector<Index>(), const vector<Index>& = vector<Index>());
float maximum(const MatrixR&);
float maximum(const VectorR&);
float maximum(const VectorR&, const vector<Index>&);
VectorR column_maximums(const Tensor2&, const vector<Index>& = vector<Index>(), const vector<Index>& = vector<Index>());
float range(const VectorR&);
float mean(const VectorR&);
float mean(const MatrixR&, Index);
VectorR mean(const MatrixR&);
VectorR mean(const MatrixR&, const vector<Index>&, const vector<Index>&);
float median(const VectorR&);
float median(const MatrixR&, Index);
VectorR median(const MatrixR&);
VectorR median(const MatrixR&, const vector<Index>&);
VectorR median(const MatrixR&, const vector<Index>&, const vector<Index>&);
float variance(const VectorR&);
float variance(const VectorR&, const VectorI&);
float standard_deviation(const VectorR&);
VectorR standard_deviation(const VectorR&, Index);
VectorR quartiles(const VectorR&);
VectorR quartiles(const VectorR&, const vector<Index>&);
BoxPlot box_plot(const VectorR&);
BoxPlot box_plot(const VectorR&, const vector<Index>&);
Descriptives vector_descriptives(const VectorR&);
vector<Descriptives> descriptives(const MatrixR&);
vector<Descriptives> descriptives(const MatrixR&, const vector<Index>&, const vector<Index>&);
Histogram histogram(const VectorR&, Index  = 10);
Histogram histogram_centered(const VectorR&, float = 0.0f, Index  = 10);
Histogram histogram(const VectorB&);
Histogram histogram(const VectorI&, Index  = 10);
vector<Histogram> histograms(const MatrixR&, Index = 10);
VectorI total_frequencies(const vector<Histogram>&);
Index minimal_index(const VectorR&);
VectorI minimal_indices(const VectorR&, Index);
VectorI minimal_indices(const MatrixR&);
Index maximal_index(const VectorR&);
VectorI maximal_indices(const VectorR&, Index);
VectorI maximal_indices(const MatrixR&);
inline bool row_finite(const VectorR& values, Index i) { return isfinite(values(i)); }
inline bool row_finite(const MatrixR& matrix, Index i) { return matrix.row(i).array().isFinite().all(); }

inline VectorR slice_rows(const VectorR& values, const vector<Index>& indices)
{
    VectorR result(indices.size());
    for (Index i = 0; i < Index(indices.size()); ++i) result(i) = values(indices[i]);
    return result;
}

inline MatrixR slice_rows(const MatrixR& matrix, const vector<Index>& indices)
{
    MatrixR result(indices.size(), matrix.cols());
    for (Index i = 0; i < Index(indices.size()); ++i) result.row(i) = matrix.row(indices[i]);
    return result;
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

inline bool is_contiguous(const vector<Index>& indices)
{
    return ranges::adjacent_find(indices,
        [](Index a, Index b) { return b != a + 1; }) == indices.end();
}

template <typename T>
inline bool is_binary(const T& tensor)
{
    return all_of(tensor.data(), tensor.data() + tensor.size(),
                  [](float value) { return value == 0.0f || value == 1.0f || isnan(value); });
}

MatrixR append_rows(const MatrixR&, const MatrixR&);

template<typename T>
vector<T> gather_by_index(const vector<T>& data, const vector<Index>& indices)
{
    vector<T> result;
    result.reserve(indices.size());

    ranges::transform(indices, back_inserter(result),
                      [&data](Index i) { return data[i]; });

    return result;
}

vector<Index> build_feasible_rows_mask(const MatrixR& outputs, const VectorR& minimums, const VectorR& maximums);

template <typename T>
inline bool is_constant(const T& tensor)
{
    const float* data = tensor.data();
    const float* end = data + tensor.size();

    const float* first = find_if(data, end, [](float value) { return !isnan(value); });

    if (first == end)
        return true;

    const float reference_value = *first;

    return all_of(first + 1, end,
                  [reference_value](float value) { return isnan(value) || abs(reference_value - value) <= numeric_limits<float>::min(); });
}

inline vector<Index> get_true_indices(const VectorB& flags)
{
    vector<Index> indices;
    indices.reserve(flags.size());

    for (Index i = 0; i < flags.size(); ++i)
        if (flags(i))
            indices.push_back(i);

    return indices;
}

VectorI calculate_rank(const VectorR&, bool ascending = true);

vector<Index> get_elements_greater_than(const vector<Index>&, Index);

VectorI get_nearest_points(const MatrixR&, const VectorR&, int = 1);

void fill_tensor_data(const MatrixR&, const vector<Index>&, const vector<Index>&, float*, bool = true, int contiguous = -1);

VectorR perform_Householder_QR_decomposition(const MatrixR&, const VectorR&);

VectorMap vector_map(const MatrixR&, Index);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
