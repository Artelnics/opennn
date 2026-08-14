//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A T I S T I C S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/statistics.h"
#include "opennn/core/parallel_algorithms.h"
#include "opennn/core/tensor_types.h"
#include "opennn/core/random_utilities.h"

#include <utility>

#include <Eigen/Dense>

namespace opennn
{

namespace
{

template <typename Sum>
struct MaskedMoments
{
    float minimum = POS_INFINITY;
    float maximum = NEG_INFINITY;
    Sum sum = 0;
    Sum squared_sum = 0;
    Index count = 0;
};

template <typename Sum, typename Addend, typename GetValue>
MaskedMoments<Sum> masked_moments(Index size,
                                  GetValue&& value_at,
                                  float minimum_start = POS_INFINITY,
                                  float maximum_start = NEG_INFINITY)
{
    MaskedMoments<Sum> moments;
    moments.minimum = minimum_start;
    moments.maximum = maximum_start;

    for (Index i = 0; i < size; ++i)
    {
        const float value = value_at(i);

        if (isnan(value)) continue;

        if (value < moments.minimum) moments.minimum = value;
        if (value > moments.maximum) moments.maximum = value;

        const Addend addend = static_cast<Addend>(value);
        moments.sum += addend;
        moments.squared_sum += addend * addend;

        ++moments.count;
    }

    return moments;
}

Index clamped_bin(float value, float origin, float inv_length, Index bins_number)
{
    return clamp(Index((value - origin) * inv_length), Index(0), bins_number - 1);
}

Index refined_bin(float value, float origin, float inv_length,
                  const VectorR& minimums, const VectorR& maximums)
{
    const Index bins_number = minimums.size();

    Index j = clamped_bin(value, origin, inv_length, bins_number);

    while (j > 0 && value < minimums(j)) j--;
    while (j < bins_number - 1 && value >= maximums(j)) j++;

    return j;
}

template <typename SkipValue>
void fill_frequencies(const VectorR& data, float origin, float inv_length,
                      const VectorR& minimums, const VectorR& maximums,
                      VectorR& frequencies, SkipValue&& skip)
{
    for (Index i = 0; i < data.size(); ++i)
    {
        const float value = data(i);

        if (skip(value)) continue;

        frequencies(refined_bin(value, origin, inv_length, minimums, maximums))++;
    }
}

Histogram assemble_histogram(const VectorR& centers,
                             const VectorR& minimums,
                             const VectorR& maximums,
                             const VectorR& frequencies)
{
    Histogram histogram(centers.size());
    histogram.centers = centers;
    histogram.minimums = minimums;
    histogram.maximums = maximums;
    histogram.frequencies = frequencies;

    return histogram;
}

}

Descriptives::Descriptives(float new_minimum,
                           float new_maximum,
                           float new_mean,
                           float new_standard_deviation) :
    minimum(new_minimum),
    maximum(new_maximum),
    mean(new_mean),
    standard_deviation(new_standard_deviation)
{
}

void Descriptives::set(const float new_minimum, float new_maximum,
                       float new_mean, float new_standard_deviation)
{
    minimum = new_minimum;
    maximum = new_maximum;
    mean = new_mean;
    standard_deviation = new_standard_deviation;
}

BoxPlot::BoxPlot(float new_minimum,
                 float new_first_quartile,
                 float new_median,
                 float new_third_quartile,
                 float new_maximum)
    : minimum(new_minimum),
      first_quartile(new_first_quartile),
      median(new_median),
      third_quartile(new_third_quartile),
      maximum(new_maximum)
{
}

Histogram::Histogram(const Index bins_number)
{
    centers.resize(bins_number);
    frequencies.resize(bins_number);
}

Histogram::Histogram(const VectorR& new_centers,
                     const VectorR& new_frequencies)
    : centers(new_centers),
      frequencies(new_frequencies)
{
}

Histogram::Histogram(const VectorR& data, Index bins_number)
{
    if (bins_number <= 0 || data.size() == 0) return;

    const float data_maximum = maximum(data);
    const float data_minimum = minimum(data);
    const float step = (data_maximum - data_minimum) / float(bins_number);

    centers = VectorR::LinSpaced(bins_number, data_minimum + 0.5f * step, data_maximum - 0.5f * step);
    frequencies = VectorR::Zero(bins_number);

    const float inv_step = (step < EPSILON) ? 0.0f : 1.0f / step;

    for (Index i = 0; i < data.size(); ++i)
    {
        const float value = data(i);
        if (isnan(value)) continue;

        frequencies(clamped_bin(value, data_minimum, inv_step, bins_number))++;
    }
}

float minimum(const MatrixR& matrix)
{
    return matrix.size() == 0 ? QUIET_NAN : matrix.minCoeff();
}

float maximum(const MatrixR& matrix)
{
    return matrix.size() == 0 ? QUIET_NAN : matrix.maxCoeff();
}

float minimum(const VectorR& vector)
{
    return vector.size() == 0 ? QUIET_NAN : vector.minCoeff();
}

float maximum(const VectorR& vector)
{
    return vector.size() == 0 ? QUIET_NAN : vector.maxCoeff();
}

float minimum(const VectorR& data, const vector<Index>& indices)
{
    if (indices.empty()) return QUIET_NAN;

    return masked_moments<float, float>(ssize(indices),
                                        [&](Index i) { return data(indices[i]); },
                                        MAX, -MAX).minimum;
}

float maximum(const VectorR& data, const vector<Index>& indices)
{
    if (indices.empty()) return QUIET_NAN;

    return masked_moments<float, float>(ssize(indices),
                                        [&](Index i) { return data(indices[i]); },
                                        MAX, -MAX).maximum;
}

float mean(const VectorR& vector)
{
    const auto is_finite = vector.array().isFinite();
    const Index count = is_finite.count();

    if (count == 0) return QUIET_NAN;

    return is_finite.select(vector.array(), 0.0f).sum() / static_cast<float>(count);
}

float variance(const VectorR& vector)
{
    const VectorR new_vector = filter_missing_values(vector);

    const Index count = new_vector.size();

    if (count <= 1) return 0.0f;

    const auto new_vector_d = new_vector.cast<double>();
    const double sum = new_vector_d.sum();
    const double squared_sum = new_vector_d.squaredNorm();

    return (squared_sum - (sum * sum) / count) / (count - 1);
}

float variance(const VectorR& vector, const VectorI& indices)
{
    const auto moments = masked_moments<long double, double>(indices.size(),
        [&](Index i) { return vector(indices(i)); });

    const long double sum = moments.sum;
    const long double squared_sum = moments.squared_sum;
    const Index count = moments.count;

    if (count <= 1) return 0.0f;

    return float(squared_sum/(count - 1) - (sum/count)*(sum/count)*count/(count-1));
}

float standard_deviation(const VectorR& vector)
{
    return sqrt(variance(vector));
}

float median(const VectorR& input_vector)
{
    VectorR valid = filter_missing_values(input_vector);
    const Index size = valid.size();

    if (size == 0) return QUIET_NAN;

    sort(valid.data(), valid.data() + size);

    return (size % 2 == 0)
        ? (valid(size/2 - 1) + valid(size/2)) / 2.0f
        : valid(size/2);
}

VectorR quartiles(const VectorR& data)
{
    VectorR valid_data = filter_missing_values(data);
    const Index new_size = valid_data.size();

    if (new_size == 0)
        return VectorR::Constant(3, QUIET_NAN);

    sort(valid_data.data(), valid_data.data() + new_size);

    VectorR quartiles(3);

    if (new_size == 1)
    {
        quartiles.setConstant(valid_data(0));
    }
    else if (new_size == 2)
    {
        const float v0 = valid_data(0);
        const float v1 = valid_data(1);
        quartiles(0) = v0 + 0.25f * (v1 - v0);
        quartiles(1) = v0 + 0.50f * (v1 - v0);
        quartiles(2) = v0 + 0.75f * (v1 - v0);
    }
    else if (new_size == 3)
    {
        quartiles(0) = (valid_data(0) + valid_data(1)) / 2.0f;
        quartiles(1) = valid_data(1);
        quartiles(2) = (valid_data(1) + valid_data(2)) / 2.0f;
    }
    else
    {
        const Index half_size = new_size / 2;

        quartiles(0) = median(VectorR(valid_data.head(half_size)));
        quartiles(1) = median(valid_data);
        quartiles(2) = median(VectorR(valid_data.tail(half_size)));
    }

    return quartiles;
}

VectorR quartiles(const VectorR& data, const vector<Index>& indices)
{
    VectorR valid_data(indices.size());
    Index sorted_index = 0;

    for (const Index index : indices)
        if (!isnan(data(index)))
            valid_data(sorted_index++) = data(index);

    valid_data.conservativeResize(sorted_index);

    return quartiles(valid_data);
}

BoxPlot box_plot(const VectorR& vector)
{
    BoxPlot box_plot;

    const VectorR valid = filter_missing_values(vector);

    if (valid.size() == 0) return box_plot;

    const VectorR quartiles = opennn::quartiles(valid);
    box_plot.minimum = minimum(valid);
    box_plot.first_quartile = quartiles(0);
    box_plot.median = quartiles(1);
    box_plot.third_quartile = quartiles(2);
    box_plot.maximum = maximum(valid);
    return box_plot;
}

BoxPlot box_plot(const VectorR& data, const vector<Index>& indices)
{
    BoxPlot box_plot;

    if (data.size() == 0 || indices.empty())
        return box_plot;

    const VectorR quartiles = opennn::quartiles(data, indices);

    box_plot.minimum = minimum(data, indices);
    box_plot.first_quartile = quartiles(0);
    box_plot.median = quartiles(1);
    box_plot.third_quartile = quartiles(2);
    box_plot.maximum = maximum(data, indices);

    return box_plot;
}

Histogram histogram(const VectorR& new_vector, Index bins_number)
{
    const Index size = new_vector.size();

    if (size == 0) return Histogram(bins_number);

    VectorR minimums(bins_number);
    VectorR maximums(bins_number);

    VectorR centers(bins_number);
    VectorR frequencies = VectorR::Zero(bins_number);

    const size_t unique_capacity = static_cast<size_t>(min(size, bins_number));

    vector<float> unique_values;
    unordered_set<float> unique_set;
    unique_values.reserve(unique_capacity);
    unique_set.reserve(unique_capacity);

    for (Index i = 0; i < size; ++i)
    {
        const float value = new_vector(i);

        if (!isnan(value) && !unique_set.contains(value))
        {
            unique_values.push_back(value);
            unique_set.insert(value);

            if (ssize(unique_values) > bins_number)
                break;
        }
    }

    const Index unique_values_number = ssize(unique_values);
    if (unique_values_number <= bins_number)
    {
        ranges::sort(unique_values);

        VectorR tensor_unique(unique_values.size());
        ranges::copy(unique_values, tensor_unique.data());

        centers = tensor_unique;
        minimums = tensor_unique;
        maximums = std::move(tensor_unique);

        frequencies = VectorR::Zero(unique_values_number);

        for (Index i = 0; i < size; ++i)
        {
            if (isnan(new_vector(i))) continue;

            for (Index j = 0; j < unique_values_number; ++j)
            {
                if (abs(new_vector(i) - centers(j)) < EPSILON)
                {
                    frequencies(j)++;
                    break;
                }
            }
        }
    }
    else
    {
        const float min = minimum(new_vector);
        const float max = maximum(new_vector);

        const float length = (max - min) /float(bins_number);
        const float inv_length = 1.0f / length;

        for (Index i = 0; i < bins_number; ++i)
        {
            minimums(i) = min + i * length;
            maximums(i) = min + (i + 1) * length;
            centers(i)  = min + (i + 0.5f) * length;
        }

        fill_frequencies(new_vector, min, inv_length, minimums, maximums, frequencies,
                         [&](float value) { return isnan(value) || value < minimums(0); });
    }

    return assemble_histogram(centers, minimums, maximums, frequencies);
}

Histogram histogram_centered(const VectorR& vector, float center, Index bins_number)
{
    const Index bin_center = (bins_number % 2 == 0)
        ? Index(float(bins_number) / 2.0f)
        : Index(float(bins_number) / 2.0f + 0.5f);

    VectorR minimums(bins_number);
    VectorR maximums(bins_number);

    VectorR centers(bins_number);
    VectorR frequencies = VectorR::Zero(bins_number);

    const float min = minimum(vector);
    const float max = maximum(vector);

    const float length = (max - min)/float(bins_number);
    const float inv_length = 1.0f / length;

    minimums(bin_center-1) = center - length;
    maximums(bin_center-1) = center + length;
    centers(bin_center-1) = center;

    for (Index i = bin_center; i < bins_number; ++i)
    {
        minimums(i) = minimums(i - 1) + length;
        maximums(i) = maximums(i - 1) + length;

        centers(i) = (maximums(i) + minimums(i)) /2.0f;
    }

    for (Index i = Index(bin_center)-2; i >= 0; i--)
    {
        minimums(i) = minimums(i+1) - length;
        maximums(i) = maximums(i+1) - length;

        centers(i) = (maximums(i) + minimums(i)) /2.0f;
    }

    fill_frequencies(vector, minimums(0), inv_length, minimums, maximums, frequencies,
                     [&](float value) { return !(value >= minimums(0)); });

    return assemble_histogram(centers, minimums, maximums, frequencies);
}

vector<Histogram> histograms(const MatrixR& matrix, Index bins_number)
{
    const Index columns_number = matrix.cols();

    vector<Histogram> histograms(columns_number);

    for (Index i = 0; i < columns_number; ++i)
        histograms[i] = histogram(VectorR(matrix.col(i)), bins_number);

    return histograms;
}

Descriptives vector_descriptives(const VectorR& x)
{
    if (x.size() == 0)
        return Descriptives();

    const VectorR valid = filter_missing_values(x);

    if (valid.size() == 0)
        return Descriptives(0.0f, 0.0f, 0.0f, 0.0f);

    return Descriptives(valid.minCoeff(), valid.maxCoeff(), valid.mean(), standard_deviation(valid));
}

vector<Descriptives> descriptives(const MatrixR& matrix)
{
    const Index columns_number = matrix.cols();

    vector<Descriptives> descriptives(columns_number);

    for (Index i = 0; i < columns_number; ++i)
        descriptives[i] = vector_descriptives(matrix.col(i));

    return descriptives;
}

vector<Descriptives> descriptives(const MatrixR& matrix,
                                  const vector<Index>& row_indices,
                                  const vector<Index>& column_indices)
{
    const Index row_indices_size = ssize(row_indices);
    const Index column_indices_size = ssize(column_indices);

    vector<Descriptives> descriptives_results(column_indices_size);

    VectorR minimums = VectorR::Zero(column_indices_size);
    VectorR maximums = VectorR::Zero(column_indices_size);

    VectorXd sums = VectorXd::Zero(column_indices_size);
    VectorXd squared_sums = VectorXd::Zero(column_indices_size);

    VectorI count = VectorI::Zero(column_indices_size);

#pragma omp parallel for
    for (Index j = 0; j < column_indices_size; ++j)
    {
        const Index column_index = column_indices[j];

        const auto moments = masked_moments<double, double>(row_indices_size,
            [&](Index i) { return matrix(row_indices[i], column_index); });

        minimums(j) = (moments.count == 0) ? 0 : moments.minimum;
        maximums(j) = (moments.count == 0) ? 0 : moments.maximum;
        sums(j) = moments.sum;
        squared_sums(j) = moments.squared_sum;
        count(j) = moments.count;
    }

    const VectorXd mean = sums.array() / count.cast<double>().array();
    VectorXd standard_deviation = VectorXd::Zero(column_indices_size);

    #pragma omp parallel for
    for (Index i = 0; i < column_indices_size; ++i)
    {
        if (count(i) > 1)
        {
            const double sample_count = static_cast<double>(count(i));
            const double variance = (squared_sums(i) - (sums(i) * sums(i) / sample_count)) / (sample_count - 1.0);
            standard_deviation(i) = sqrt(max(0.0, variance));
        }

        descriptives_results[i].set(minimums(i),
                                    maximums(i),
                                    static_cast<float>(mean(i)),
                                    static_cast<float>(standard_deviation(i)));
    }

    return descriptives_results;
}

VectorR mean(const MatrixR& matrix)
{
    const auto finite = matrix.array().isFinite();
    const VectorR sums   = finite.select(matrix.array(), 0.0f).colwise().sum();
    const VectorR counts = finite.cast<float>().colwise().sum();

    return (counts.array() > 0.0f).select(sums.array() / counts.array(), QUIET_NAN);
}

VectorR mean(const MatrixR& matrix, const vector<Index>& row_indices, const vector<Index>& column_indices)
{
    const Index row_indices_size = row_indices.size();
    const Index column_indices_size = column_indices.size();

    if (row_indices_size == 0 || column_indices_size == 0) return {};

    VectorR means(column_indices_size);

    for (Index j = 0; j < column_indices_size; ++j)
    {
        const Index column_index = column_indices[j];

        const auto moments = masked_moments<float, float>(row_indices_size,
            [&](Index i) { return matrix(row_indices[i], column_index); });

        means(j) = (moments.count > 0) ? moments.sum / float(moments.count) : QUIET_NAN;
    }

    return means;
}

float mean(const MatrixR& matrix, Index column_index)
{
    if (matrix.size() == 0) return QUIET_NAN;

    const VectorR col = matrix.col(column_index);
    const auto finite = col.array().isFinite();
    const Index count = finite.count();

    if (count == 0) return QUIET_NAN;

    return finite.select(col.array(), 0.0f).sum() / float(count);
}

float median(const MatrixR& matrix, Index column_index)
{
    return median(VectorR(matrix.col(column_index)));
}

VectorR median(const MatrixR& matrix,
               const vector<Index>& row_indices,
               const vector<Index>& column_indices)
{
    const Index column_indices_size = ssize(column_indices);

    VectorR medians(column_indices_size);

    for (Index j = 0; j < column_indices_size; ++j)
        medians(j) = median(VectorR(matrix(row_indices, column_indices[j])));

    return medians;
}

// PropagateNumbers so NaN entries are skipped rather than compared against. Callers use
// NaN to mark "no value here" (an unevaluated epoch, a missing sample), and plain
// minCoeff/maxCoeff give implementation-defined results once a NaN is in the range.
Index minimal_index(const VectorR& vector)
{
    Index index = 0;
    if (vector.size() > 0)
        vector.minCoeff<PropagateNumbers>(&index);
    return index;
}

Index maximal_index(const VectorR& vector)
{
    Index index = 0;
    if (vector.size() > 0)
        vector.maxCoeff<PropagateNumbers>(&index);
    return index;
}

VectorI maximal_indices(const VectorR& data, Index count)
{
    vector<Index> indices(data.size());
    iota(indices.begin(), indices.end(), 0);

    count = min(count, ssize(data));

    partial_sort(indices.begin(), indices.begin() + count, indices.end(),
                 [&data](Index i, Index j) {
                     if (data(i) == data(j)) return i < j;
                     return data(i) > data(j);
                 });

    return Map<VectorI>(indices.data(), count);
}

VectorI maximal_indices(const MatrixR& matrix)
{
    VectorI result(2);
    matrix.maxCoeff(&result(0), &result(1));
    return result;
}

VectorI calculate_rank(const VectorR& vector, bool ascending)
{
    const Index size = vector.size();

    VectorI rank(size);
    iota(rank.data(), rank.data() + rank.size(), 0);

    sort_parallel_if_large(
        rank.data(), rank.data() + rank.size(),
        [&](Index i, Index j) { return ascending ? vector[i] < vector[j] : vector[i] > vector[j]; });

    return rank;
}

VectorR filter_missing_values(const VectorR& x)
{
    vector<Index> valid;
    valid.reserve(x.size());

    for (Index i = 0; i < x.size(); ++i)
        if (isfinite(x(i))) valid.push_back(i);

    return x(valid);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
