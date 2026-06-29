//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A T I S T I C S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "statistics.h"
#include "tensor_types.h"
#include "random_utilities.h"

#include <Eigen/Dense>

namespace opennn
{

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

void Descriptives::print(const string& title) const
{
    cout << title << "\n"
         << "Minimum: " << minimum << "\n"
         << "Maximum: " << maximum << "\n"
         << "Mean: " << mean << "\n"
         << "Standard deviation: " << standard_deviation << "\n";
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

void BoxPlot::set(float new_minimum,
                  float new_first_quartile,
                  float new_median,
                  float new_third_quartile,
                  float new_maximum)
{
    minimum = new_minimum;
    first_quartile = new_first_quartile;
    median = new_median;
    third_quartile = new_third_quartile;
    maximum = new_maximum;
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

        const Index corresponding_bin = clamp(Index((value - data_minimum) * inv_step), Index(0), bins_number - 1);

        frequencies(corresponding_bin)++;
    }
}

Index Histogram::get_bins_number() const
{
    return centers.size();
}

float minimum(const MatrixR& matrix)
{
    return matrix.size() == 0 ? NAN : matrix.minCoeff();
}

float maximum(const MatrixR& matrix)
{
    return matrix.size() == 0 ? NAN : matrix.maxCoeff();
}

float minimum(const VectorR& vector)
{
    return vector.size() == 0 ? NAN : vector.minCoeff();
}

float maximum(const VectorR& vector)
{
    return vector.size() == 0 ? NAN : vector.maxCoeff();
}

float minimum(const VectorR& data, const vector<Index>& indices)
{
    if (indices.empty()) return NAN;

    float minimum = MAX;

    for (const Index index : indices)
        if (data(index) < minimum && !isnan(data(index)))
            minimum = data(index);

    return minimum;
}

float maximum(const VectorR& data, const vector<Index>& indices)
{
    if (indices.empty()) return NAN;

    float maximum = -MAX;

    for (const Index index : indices)
        if (!isnan(data(index)) && data(index) > maximum)
            maximum = data(index);

    return maximum;
}

float mean(const VectorR& vector)
{
    const auto is_finite = vector.array().isFinite();
    const Index count = is_finite.count();

    if (count == 0) return NAN;

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
    const Index size = indices.size();

    long double sum = 0.0;
    long double squared_sum = 0.0;

    Index count = 0;

    for (Index i = 0; i < size; ++i)
    {
        const float value = vector(indices(i));

        if (!isnan(value))
        {
            const double v = value;
            sum += v;
            squared_sum += v * v;

            ++count;
        }
    }

    if (count <= 1) return 0.0f;

    return float(squared_sum/(count - 1) - (sum/count)*(sum/count)*count/(count-1));
}

float standard_deviation(const VectorR& vector)
{
    return vector.size() == 0 ? 0.0f : sqrt(variance(vector));
}

float median(const VectorR& input_vector)
{
    VectorR valid = filter_missing_values(input_vector);
    const Index size = valid.size();

    if (size == 0) return NAN;

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
        quartiles(1) = median(VectorR(valid_data));
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
    if (vector.size() == 0)
        return box_plot;

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
        maximums = move(tensor_unique);

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

        minimums(0) = min;
        maximums(0) = min + length;
        centers(0) = (maximums(0) + minimums(0)) /2.0f;


        for (Index i = 1; i < bins_number; ++i)
        {
            minimums(i) = minimums(i - 1) + length;
            maximums(i) = maximums(i - 1) + length;

            centers(i) = (maximums(i) + minimums(i)) /2.0f;
        }


        const Index size = new_vector.size();

        for (Index i = 0; i < size; ++i)
        {
            const float value = new_vector(i);

            if (isnan(value) || value < minimums(0)) continue;

            Index j = clamp(Index((value - min) * inv_length), Index(0), bins_number - 1);

            while (j > 0 && value < minimums(j)) j--;
            while (j < bins_number - 1 && value >= maximums(j)) j++;

            frequencies(j)++;
        }
    }

    Histogram histogram(bins_number);
    histogram.centers = centers;
    histogram.minimums = minimums;
    histogram.maximums = maximums;
    histogram.frequencies = frequencies;

    return histogram;
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


    const Index size = vector.size();

    for (Index i = 0; i < size; ++i)
    {
        const float value = vector(i);

        if (!(value >= minimums(0))) continue;

        Index j = clamp(Index((value - minimums(0)) * inv_length), Index(0), bins_number - 1);

        while (j > 0 && value < minimums(j)) j--;
        while (j < bins_number - 1 && value >= maximums(j)) j++;

        frequencies(j)++;
    }

    Histogram histogram(bins_number);
    histogram.centers = centers;
    histogram.minimums = minimums;
    histogram.maximums = maximums;
    histogram.frequencies = frequencies;

    return histogram;
}

Histogram histogram(const VectorB& flags)
{
    VectorR minimums = VectorR::Zero(2);

    VectorR maximums = VectorR::Ones(2);

    VectorR centers(2);
    centers << 0.0f, 1.0f;

    VectorR frequencies = VectorR::Zero(2);


    const Index size = flags.size();

    for (Index i = 0; i < size; ++i)
        frequencies(flags(i) ? 1 : 0)++;

    Histogram histogram(2);
    histogram.centers = centers;
    histogram.minimums = minimums;
    histogram.maximums = maximums;
    histogram.frequencies = frequencies;

    return histogram;
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
    const Index count = valid.size();

    const float min = (count > 0) ? valid.minCoeff() : 0.0f;
    const float max = (count > 0) ? valid.maxCoeff() : 0.0f;

    const float mean = (count > 0) ? valid.mean() : 0.0f;

    return Descriptives(min, max, mean, standard_deviation(valid));
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

        float current_min = numeric_limits<float>::infinity();
        float current_max = -numeric_limits<float>::infinity();
        double current_sum = 0;
        double current_sq_sum = 0;
        Index current_count = 0;

        for (Index i = 0; i < row_indices_size; ++i)
        {
            const Index row_index = row_indices[i];
            const float value = matrix(row_index, column_index);

            if (isnan(value)) continue;

            if (value < current_min) current_min = value;
            if (value > current_max) current_max = value;

            const double v = static_cast<double>(value);
            current_sum += v;
            current_sq_sum += v * v;
            ++current_count;
        }

        if (current_count == 0) current_min = current_max = 0;

        minimums(j) = current_min;
        maximums(j) = current_max;
        sums(j) = current_sum;
        squared_sums(j) = current_sq_sum;
        count(j) = current_count;
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

    return (counts.array() > 0.0f).select(sums.array() / counts.array(), NAN);
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

        float sum = 0;
        Index count = 0;

        for (Index i = 0; i < row_indices_size; ++i)
        {
            const float value = matrix(row_indices[i], column_index);
            if (isnan(value)) continue;
            sum += value;
            ++count;
        }

        means(j) = (count > 0) ? sum / float(count) : NAN;
    }

    return means;
}

float mean(const MatrixR& matrix, Index column_index)
{
    if (matrix.size() == 0) return NAN;

    const VectorR col = matrix.col(column_index);
    const auto finite = col.array().isFinite();
    const Index count = finite.count();

    if (count == 0) return NAN;

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
    const Index row_indices_size = ssize(row_indices);
    const Index column_indices_size = ssize(column_indices);

    VectorR medians(column_indices_size);

    for (Index j = 0; j < column_indices_size; ++j)
    {
        VectorR column(row_indices_size);

        for (Index i = 0; i < row_indices_size; ++i)
            column(i) = matrix(row_indices[i], column_indices[j]);

        medians(j) = median(column);
    }

    return medians;
}

Index minimal_index(const VectorR& vector)
{
    Index index = 0;
    if (vector.size() > 0)
        vector.minCoeff(&index);
    return index;
}

Index maximal_index(const VectorR& vector)
{
    Index index = 0;
    if (vector.size() > 0)
        vector.maxCoeff(&index);
    return index;
}

VectorI minimal_indices(const VectorR& data, Index count)
{
    vector<Index> indices(data.size());
    iota(indices.begin(), indices.end(), 0);

    count = min(count, ssize(data));

    partial_sort(indices.begin(),
                 indices.begin() + count,
                 indices.end(),
                 [&data](Index i, Index j) {
                     if (data(i) == data(j)) return i < j;
                     return data(i) < data(j);
                 });

    return Map<VectorI>(indices.data(), count);
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

VectorI minimal_indices(const MatrixR& matrix)
{
    VectorI result(2);
    matrix.minCoeff(&result(0), &result(1));
    return result;
}

VectorI maximal_indices(const MatrixR& matrix)
{
    VectorI result(2);
    matrix.maxCoeff(&result(0), &result(1));
    return result;
}

VectorR local_outlier_factor(const MatrixR& points, Index neighbors_number)
{
    const Index points_number = points.rows();

    if (points_number <= 1)
        return VectorR::Ones(points_number);

    neighbors_number = min(neighbors_number, points_number - 1);

    const MatrixR distances = calculate_distances(points);

    vector<VectorI> neighbors(points_number);
    VectorR neighbor_distance(points_number);

    for (Index i = 0; i < points_number; i++)
    {
        VectorR row = distances.row(i).transpose();
        row(i) = MAX;
        neighbors[i] = maximal_indices(-row, neighbors_number);
        neighbor_distance(i) = row(neighbors[i](neighbors_number - 1));
    }

    VectorR reachability_density(points_number);

    for (Index i = 0; i < points_number; i++)
    {
        float reachability_sum = 0.0f;
        for (Index j = 0; j < neighbors_number; j++)
            reachability_sum += max(neighbor_distance(neighbors[i](j)), distances(i, neighbors[i](j)));
        reachability_density(i) = reachability_sum > EPSILON ? float(neighbors_number) / reachability_sum : MAX;
    }

    VectorR outlier_factor(points_number);

    for (Index i = 0; i < points_number; i++)
    {
        float density_sum = 0.0f;
        for (Index j = 0; j < neighbors_number; j++)
            density_sum += reachability_density(neighbors[i](j));
        outlier_factor(i) = reachability_density(i) > EPSILON
            ? density_sum / (float(neighbors_number) * reachability_density(i)) : 1.0f;
    }

    return outlier_factor;
}

vector<Index> build_feasible_rows_mask(const MatrixR& outputs, const VectorR& minimums, const VectorR& maximums)
{
    const Index rows_unfiltered = outputs.rows();
    const Index variables_to_filter = outputs.cols();

    throw_if(minimums.size() != variables_to_filter || maximums.size() != variables_to_filter,
             "build_feasible_rows_mask: Minimums/maximums size mismatch with outputs columns.\n");

    vector<Index> feasible_rows;
    feasible_rows.reserve(static_cast<size_t>(rows_unfiltered));

    const auto min_bound = minimums.transpose().array();
    const auto max_bound = maximums.transpose().array();

    for (Index i = 0; i < rows_unfiltered; ++i)
    {
        const auto row_arr = outputs.row(i).array();

        if ((row_arr >= min_bound && row_arr <= max_bound).all())
            feasible_rows.push_back(i);
    }

    return feasible_rows;
}

VectorI calculate_rank(const VectorR& vector, bool ascending)
{
    const Index size = vector.size();

    VectorI rank(size);
    iota(rank.data(), rank.data() + rank.size(), 0);

    sort(rank.data(),
         rank.data() + rank.size(),
         [&](Index i, Index j) { return ascending ? vector[i] < vector[j] : vector[i] > vector[j]; });

    return rank;
}

vector<Index> get_elements_greater_than(const vector<Index>& data, Index bound)
{
    vector<Index> indices;
    indices.reserve(data.size());

    ranges::copy_if(data, back_inserter(indices),
                    [bound](Index value) { return value > bound; });
    return indices;
}

VectorR perform_Householder_QR_decomposition(const MatrixR& A, const VectorR& b)
{
    return A.colPivHouseholderQr().solve(b);
}

VectorR filter_missing_values(const VectorR& x)
{
    vector<Index> valid;
    valid.reserve(x.size());

    for (Index i = 0; i < x.size(); ++i)
        if (isfinite(x(i))) valid.push_back(i);

    return slice_rows(x, valid);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
