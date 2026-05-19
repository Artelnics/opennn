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

/// @brief Summary statistics (minimum, maximum, mean, standard deviation) for one variable.
struct Descriptives
{
    /// @brief Constructs a descriptives record from minimum, maximum, mean, and standard deviation.
    Descriptives(const float = NAN, float = NAN, float = NAN, float = NAN);

    /// @brief Returns the four statistics as a length-4 vector [min, max, mean, std].
    [[nodiscard]] VectorR to_tensor() const;

    /// @brief Sets the four statistics in place.
    void set(const float = NAN, float = NAN, float = NAN, float = NAN);

    /// @brief Saves the descriptives to a text file at the given path.
    void save(const filesystem::path&) const;

    /// @brief Prints the descriptives to stdout under the given header.
    void print(const string& = "Descriptives:") const;

    string name = "Descriptives";

    float minimum = -1.0f;

    float maximum = 1.0f;

    float mean = 0.0f;

    float standard_deviation = 1.0f;

};

/// @brief Five-number summary (minimum, Q1, median, Q3, maximum) used to draw a box plot.
struct BoxPlot
{
    /// @brief Constructs a box plot from minimum, first quartile, median, third quartile, and maximum.
    BoxPlot(const float = NAN,
            float = NAN,
            float = NAN,
            float = NAN,
            float = NAN);

    /// @brief Sets the five statistics in place.
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

/// @brief Frequency histogram with per-bin minimums, maximums, centers, and counts.
struct Histogram
{
    /// @brief Constructs an empty histogram with the given number of bins.
    Histogram(const Index = 0);

    /// @brief Constructs a histogram from precomputed bin centers and frequencies.
    Histogram(const VectorR&, const VectorR&);

    /// @brief Constructs a histogram from bin minimums, maximums, centers, and frequencies.
    Histogram(const VectorR&, const VectorR&, const VectorR&, const VectorR&);

    /// @brief Builds a histogram of the data with the given number of equal-width bins.
    Histogram(const VectorR&, Index);

    /// @brief Builds a histogram of the data using a default bin count.
    Histogram(const VectorR&);

    /// @brief Returns the number of bins in the histogram.
    [[nodiscard]] Index get_bins_number() const;

    /// @brief Returns the number of bins with zero frequency.
    [[nodiscard]] Index count_empty_bins() const;

    /// @brief Returns the smallest bin frequency.
    [[nodiscard]] Index calculate_minimum_frequency() const;

    /// @brief Returns the largest bin frequency.
    [[nodiscard]] Index calculate_maximum_frequency() const;

    /// @brief Returns the index of the bin with the largest frequency.
    [[nodiscard]] Index calculate_most_populated_bin() const;

    /// @brief Returns the centers of all bins tied for the minimum frequency.
    [[nodiscard]] VectorR calculate_minimal_centers() const;

    /// @brief Returns the centers of all bins tied for the maximum frequency.
    [[nodiscard]] VectorR calculate_maximal_centers() const;

    /// @brief Returns the bin index that contains the given value.
    [[nodiscard]] Index calculate_bin(const float) const;

    /// @brief Returns the frequency of the bin that contains the given value.
    [[nodiscard]] Index calculate_frequency(const float) const;

    /// @brief Saves the histogram (centers and frequencies) to a text file.
    void save(const filesystem::path&) const;

    VectorR minimums;

    VectorR maximums;

    VectorR centers;

    VectorR frequencies;
};
/// @brief Returns the smallest finite element of a matrix.
[[nodiscard]] float minimum(const MatrixR&);
/// @brief Returns the smallest finite element of a vector.
[[nodiscard]] float minimum(const VectorR&);
/// @brief Returns the smallest finite element of the selected rows of a vector.
[[nodiscard]] float minimum(const VectorR&, const vector<Index>&);
/// @brief Per-column minimums of a 2D tensor.
/// @param matrix Source matrix.
/// @param row_indices Optional row subset (empty = all rows).
/// @param column_indices Optional column subset (empty = all columns).
[[nodiscard]] VectorR column_minimums(const Tensor2&, const vector<Index>& = {}, const vector<Index>& = {});
/// @brief Returns the largest finite element of a matrix.
[[nodiscard]] float maximum(const MatrixR&);
/// @brief Returns the largest finite element of a vector.
[[nodiscard]] float maximum(const VectorR&);
/// @brief Returns the largest finite element of the selected rows of a vector.
[[nodiscard]] float maximum(const VectorR&, const vector<Index>&);
/// @brief Per-column maximums of a 2D tensor, optionally restricted to a row/column subset.
[[nodiscard]] VectorR column_maximums(const Tensor2&, const vector<Index>& = {}, const vector<Index>& = {});
/// @brief Returns the maximum minus the minimum of a vector.
[[nodiscard]] float range(const VectorR&);
/// @brief Arithmetic mean of a vector, ignoring NaNs.
[[nodiscard]] float mean(const VectorR&);
/// @brief Arithmetic mean of a matrix column, ignoring NaNs.
[[nodiscard]] float mean(const MatrixR&, Index);
/// @brief Column-wise arithmetic means of a matrix.
[[nodiscard]] VectorR mean(const MatrixR&);
/// @brief Column-wise means of a matrix restricted to the given rows and columns.
[[nodiscard]] VectorR mean(const MatrixR&, const vector<Index>&, const vector<Index>&);
/// @brief Median of a vector.
[[nodiscard]] float median(const VectorR&);
/// @brief Median of a single matrix column.
[[nodiscard]] float median(const MatrixR&, Index);
/// @brief Column-wise medians of a matrix.
[[nodiscard]] VectorR median(const MatrixR&);
/// @brief Column-wise medians of the selected columns.
[[nodiscard]] VectorR median(const MatrixR&, const vector<Index>&);
/// @brief Column-wise medians restricted to the given rows and columns.
[[nodiscard]] VectorR median(const MatrixR&, const vector<Index>&, const vector<Index>&);
/// @brief Sample variance of a vector.
[[nodiscard]] float variance(const VectorR&);
/// @brief Sample variance of the selected entries of a vector.
[[nodiscard]] float variance(const VectorR&, const VectorI&);
/// @brief Sample standard deviation of a vector.
[[nodiscard]] float standard_deviation(const VectorR&);
/// @brief Rolling standard deviation of a vector over a window of the given size.
[[nodiscard]] VectorR standard_deviation(const VectorR&, Index);
/// @brief Returns [Q1, Q2, Q3] of a vector.
[[nodiscard]] VectorR quartiles(const VectorR&);
/// @brief Returns [Q1, Q2, Q3] of the selected entries of a vector.
[[nodiscard]] VectorR quartiles(const VectorR&, const vector<Index>&);
/// @brief Five-number summary (box plot) of a vector.
[[nodiscard]] BoxPlot box_plot(const VectorR&);
/// @brief Five-number summary (box plot) of the selected entries of a vector.
[[nodiscard]] BoxPlot box_plot(const VectorR&, const vector<Index>&);
/// @brief Returns the (min, max, mean, std) descriptives of a vector.
[[nodiscard]] Descriptives vector_descriptives(const VectorR&);
/// @brief Returns the per-column descriptives of a matrix.
[[nodiscard]] vector<Descriptives> descriptives(const MatrixR&);
/// @brief Returns per-column descriptives restricted to the given rows and columns.
[[nodiscard]] vector<Descriptives> descriptives(const MatrixR&, const vector<Index>&, const vector<Index>&);
/// @brief Builds an equal-width histogram of a vector.
/// @param values Input data.
/// @param bins Number of equal-width bins.
[[nodiscard]] Histogram histogram(const VectorR&, Index  = 10);
/// @brief Builds a histogram with one bin centered on the given value.
[[nodiscard]] Histogram histogram_centered(const VectorR&, float = 0.0f, Index  = 10);
/// @brief Builds a two-bin histogram counting false/true entries.
[[nodiscard]] Histogram histogram(const VectorB&);
/// @brief Builds an equal-width histogram of an integer vector.
[[nodiscard]] Histogram histogram(const VectorI&, Index  = 10);
/// @brief Builds one histogram per matrix column.
[[nodiscard]] vector<Histogram> histograms(const MatrixR&, Index = 10);
/// @brief Sums the per-bin frequencies across a collection of histograms.
[[nodiscard]] VectorI total_frequencies(const vector<Histogram>&);
/// @brief Index of the smallest element in a vector.
[[nodiscard]] Index minimal_index(const VectorR&);
/// @brief Indices of the n smallest elements of a vector.
[[nodiscard]] VectorI minimal_indices(const VectorR&, Index);
/// @brief Row/column coordinates of the smallest element of a matrix.
[[nodiscard]] VectorI minimal_indices(const MatrixR&);
/// @brief Index of the largest element in a vector.
[[nodiscard]] Index maximal_index(const VectorR&);
/// @brief Indices of the n largest elements of a vector.
[[nodiscard]] VectorI maximal_indices(const VectorR&, Index);
/// @brief Row/column coordinates of the largest element of a matrix.
[[nodiscard]] VectorI maximal_indices(const MatrixR&);
/// @brief Returns true if the i-th entry of the vector is finite.
[[nodiscard]] inline bool row_finite(const VectorR& values, Index i) { return isfinite(values(i)); }
/// @brief Returns true if every entry in row i of the matrix is finite.
[[nodiscard]] inline bool row_finite(const MatrixR& matrix, Index i) { return matrix.row(i).array().isFinite().all(); }

/// @brief Returns a copy of the vector containing only the entries at the given indices.
[[nodiscard]] inline VectorR slice_rows(const VectorR& values, const vector<Index>& indices)
{
    VectorR result(indices.size());
    for (Index i = 0; i < Index(indices.size()); ++i) result(i) = values(indices[i]);
    return result;
}

/// @brief Returns a copy of the matrix containing only the rows at the given indices.
[[nodiscard]] inline MatrixR slice_rows(const MatrixR& matrix, const vector<Index>& indices)
{
    MatrixR result(indices.size(), matrix.cols());
    for (Index i = 0; i < Index(indices.size()); ++i) result.row(i) = matrix.row(indices[i]);
    return result;
}

/// @brief Returns a copy of the vector with NaN entries removed.
[[nodiscard]] VectorR filter_missing_values(const VectorR&);

/// @brief Returns x and y restricted to rows where both are finite (row counts must match).
template<typename X, typename Y>
[[nodiscard]] pair<X, Y> filter_missing_values(const X& x, const Y& y)
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

/// @brief Returns true if the sorted indices form a contiguous run (each entry equals the previous plus one).
[[nodiscard]] inline bool is_contiguous(const vector<Index>& indices)
{
    return ranges::adjacent_find(indices,
        [](Index a, Index b) { return b != a + 1; }) == indices.end();
}

/// @brief Returns true if every non-NaN entry of the tensor is exactly 0.0 or 1.0.
template <typename T>
[[nodiscard]] inline bool is_binary(const T& tensor)
{
    return all_of(tensor.data(), tensor.data() + tensor.size(),
                  [](float value) { return value == 0.0f || value == 1.0f || isnan(value); });
}

/// @brief Returns the row-wise concatenation of two matrices with matching column counts.
[[nodiscard]] MatrixR append_rows(const MatrixR&, const MatrixR&);

/// @brief Returns the elements of data at the given indices.
template<typename T>
[[nodiscard]] vector<T> gather_by_index(const vector<T>& data, const vector<Index>& indices)
{
    vector<T> result;
    result.reserve(indices.size());

    ranges::transform(indices, back_inserter(result),
                      [&data](Index i) { return data[i]; });

    return result;
}

/// @brief Returns the indices of the rows of outputs that lie within the per-column bounds.
[[nodiscard]] vector<Index> build_feasible_rows_mask(const MatrixR& outputs, const VectorR& minimums, const VectorR& maximums);

/// @brief Returns true if every non-NaN entry of the tensor equals the first finite entry.
template <typename T>
[[nodiscard]] inline bool is_constant(const T& tensor)
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

/// @brief Returns the positions of the true entries in a boolean vector.
[[nodiscard]] inline vector<Index> get_true_indices(const VectorB& flags)
{
    vector<Index> indices;
    indices.reserve(flags.size());

    for (Index i = 0; i < flags.size(); ++i)
        if (flags(i))
            indices.push_back(i);

    return indices;
}

/// @brief Returns the rank of each element (1-based), ascending by default.
[[nodiscard]] VectorI calculate_rank(const VectorR&, bool ascending = true);

/// @brief Returns the entries of indices that are strictly greater than the given threshold.
[[nodiscard]] vector<Index> get_elements_greater_than(const vector<Index>&, Index);

/// @brief Finds the n rows of the matrix closest to the given point by Euclidean distance.
[[nodiscard]] VectorI get_nearest_points(const MatrixR&, const VectorR&, int = 1);

/// @brief Copies the selected sub-matrix into a flat float buffer.
/// @param matrix Source data matrix.
/// @param row_indices Rows to copy.
/// @param column_indices Columns to copy.
/// @param data Destination buffer (caller-owned).
/// @param row_major Lay rows contiguously when true, columns when false.
/// @param contiguous Optional fast-path hint when -1 the layout is inferred.
void fill_tensor_data(const MatrixR&, const vector<Index>&, const vector<Index>&, float*, bool = true, int contiguous = -1);

/// @brief Solves a linear least-squares problem via Householder QR decomposition.
[[nodiscard]] VectorR perform_Householder_QR_decomposition(const MatrixR&, const VectorR&);

/// @brief Returns an Eigen VectorMap that views a single column of a matrix without copying.
[[nodiscard]] VectorMap vector_map(const MatrixR&, Index);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
