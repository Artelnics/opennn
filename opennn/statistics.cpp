//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A T I S T I C S
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "statistics.h"
#include "tensor_utilities.h"
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

VectorR Descriptives::to_tensor() const
{
    VectorR descriptives_tensor(4);
    descriptives_tensor << minimum, maximum, mean, standard_deviation;

    return descriptives_tensor;
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

void Descriptives::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        throw runtime_error("Cannot open descriptives data file.\n");

    file << "Minimum: " << minimum << "\n"
         << "Maximum: " << maximum << "\n"
         << "Mean: " << mean << "\n"
         << "Standard deviation: " << standard_deviation << "\n";

    file.close();
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

Histogram::Histogram(const VectorR& new_frequencies,
                     const VectorR& new_centers,
                     const VectorR& new_minimums,
                     const VectorR& new_maximums)
    : minimums(new_minimums),
      maximums(new_maximums),
      centers(new_centers),
      frequencies(new_frequencies)
{
}

Histogram::Histogram(const VectorR& data, Index bins_number)
{
    const float data_maximum = maximum(data);
    const float data_minimum = minimum(data);
    const float step = (data_maximum - data_minimum) / float(bins_number);
    const float inv_step = 1.0f / step;

    centers = VectorR::LinSpaced(bins_number, data_minimum + 0.5f * step, data_maximum - 0.5f * step);
    frequencies = VectorR::Zero(bins_number);

    for (Index i = 0; i < data.size(); ++i)
    {
        const float value = data(i);
        if (isnan(value)) continue;

        const Index corresponding_bin = min(Index((value - data_minimum) * inv_step), bins_number - 1);

        frequencies(corresponding_bin)++;
    }
}

Histogram::Histogram(const VectorR& probability_data)
{
    const size_t bins_number = 10;
    const float data_minimum = 0.0f;
    const float data_maximum = maximum(probability_data) > 1.0f ? 100.0f : 1.0f;

    const float step = (data_maximum - data_minimum) / float(bins_number);
    const float inv_step = 1.0f / step;

    centers = VectorR::LinSpaced(bins_number, data_minimum + 0.5f * step, data_maximum - 0.5f * step);
    frequencies = VectorR::Zero(bins_number);

    for (Index i = 0; i < probability_data.size(); ++i)
    {
        const float value = probability_data(i);
        if (isnan(value)) continue;

        const Index corresponding_bin = min(Index((value - data_minimum) * inv_step), Index(bins_number) - 1);

        frequencies(corresponding_bin)++;
    }
}

Index Histogram::get_bins_number() const
{
    return centers.size();
}

Index Histogram::count_empty_bins() const
{
    return static_cast<Index>((frequencies.array() == 0.0f).count());
}

Index Histogram::calculate_minimum_frequency() const
{
    return minimum(frequencies);
}

Index Histogram::calculate_maximum_frequency() const
{
    return maximum(frequencies);
}

Index Histogram::calculate_most_populated_bin() const
{
    if (frequencies.size() == 0)
        return 0;

    Index max_index;
    frequencies.maxCoeff(&max_index);

    return max_index;
}

VectorR Histogram::calculate_minimal_centers() const
{
    if (frequencies.size() == 0)
        return VectorR::Constant(1, NAN);

    const Index minimum_frequency = calculate_minimum_frequency();
    const Index count = (frequencies.array() == minimum_frequency).count();

    VectorR minimal_centers(count);
    Index index = 0;

    for (Index i = 0; i < frequencies.size(); ++i)
        if (frequencies(i) == minimum_frequency)
            minimal_centers(index++) = centers(i);

    return minimal_centers;
}

VectorR Histogram::calculate_maximal_centers() const
{
    if (frequencies.size() == 0)
        return VectorR::Constant(1, NAN);

    const Index maximum_frequency = calculate_maximum_frequency();
    const Index count = (frequencies.array() == maximum_frequency).count();

    VectorR maximal_centers(count);
    Index index = 0;

    for (Index i = 0; i < frequencies.size(); ++i)
        if (frequencies(i) == maximum_frequency)
            maximal_centers(index++) = centers(i);

    return maximal_centers;
}

Index Histogram::calculate_bin(const float value) const
{
    const Index bins_number = get_bins_number();

    if (bins_number == 0) return 0;

    const float min_center = centers(0);
    const float max_center = centers(bins_number - 1);
    const float bin_width = (max_center - min_center) / (bins_number - 1);

    for (Index i = 0; i < bins_number; ++i) {
        if (value < centers(i) + bin_width / 2) {
            return i;
        }
    }

    return bins_number - 1;
}

Index Histogram::calculate_frequency(const float value) const
{
    if (get_bins_number() == 0) return 0;

    return frequencies[calculate_bin(value)];
}

void Histogram::save(const filesystem::path& histogram_file_name) const
{
    const Index bins_number = centers.size();
    ofstream histogram_file(histogram_file_name);

    histogram_file << "centers,frequencies" << "\n";

    for (Index i = 0; i < bins_number; ++i)
        histogram_file << centers(i) << ","
                       << frequencies(i) << "\n";

    histogram_file.close();
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

VectorR column_maximums(const MatrixR& matrix,
                        const vector<Index>& row_indices,
                        const vector<Index>& column_indices)
{
    vector<Index> used_column_indices = column_indices;
    if (used_column_indices.empty())
    {
        used_column_indices.resize(matrix.cols());
        iota(used_column_indices.begin(), used_column_indices.end(), 0);
    }

    vector<Index> used_row_indices = row_indices;
    if (used_row_indices.empty())
    {
        used_row_indices.resize(matrix.rows());
        iota(used_row_indices.begin(), used_row_indices.end(), 0);
    }

    const Index row_indices_size = used_row_indices.size();
    const Index column_indices_size = used_column_indices.size();

    VectorR maximums(column_indices_size);

    for (Index j = 0; j < column_indices_size; ++j)
    {
        const Index column_index = used_column_indices[j];

        VectorR column(row_indices_size);

        for (Index i = 0; i < row_indices_size; ++i)
            column(i) = matrix(used_row_indices[i], column_index);

        maximums(j) = maximum(column);
    }

    return maximums;
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
        const float sum = valid_data(0) + valid_data(1);
        quartiles(0) = sum / 4.0f;
        quartiles(1) = sum / 2.0f;
        quartiles(2) = sum * 0.75f;
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

    if (data.size() == 0 || indices.size() == 0)
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
    VectorR minimums(bins_number);
    VectorR maximums(bins_number);

    VectorR centers(bins_number);
    VectorR frequencies = VectorR::Zero(bins_number);

    vector<float> unique_values;
    unordered_set<float> unique_set;

    unique_values.reserve(min<Index>(size, bins_number));
    unique_values.push_back(new_vector(0));
    unique_set.insert(new_vector(0));

    for (Index i = 1; i < size; ++i)
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
                if (new_vector(i) - centers(j) < EPSILON)
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

        // Calculate bins center

        for (Index i = 1; i < bins_number; ++i)
        {
            minimums(i) = minimums(i - 1) + length;
            maximums(i) = maximums(i - 1) + length;

            centers(i) = (maximums(i) + minimums(i)) /2.0f;
        }

        // Calculate bins frequency

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

    // Calculate bins center

    for (Index i = bin_center; i < bins_number; ++i) // Upper centers
    {
        minimums(i) = minimums(i - 1) + length;
        maximums(i) = maximums(i - 1) + length;

        centers(i) = (maximums(i) + minimums(i)) /2.0f;
    }

    for (Index i = Index(bin_center)-2; i >= 0; i--) // Lower centers
    {
        minimums(i) = minimums(i+1) - length;
        maximums(i) = maximums(i+1) - length;

        centers(i) = (maximums(i) + minimums(i)) /2.0f;
    }

    // Calculate bins frequency

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

    // Calculate bins frequency

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

VectorI total_frequencies(const Tensor<Histogram, 1>& histograms)
{
    const Index histograms_number = histograms.size();

    VectorI total_frequencies(histograms_number);

    for (Index i = 0; i < histograms_number; ++i)
        total_frequencies(i) = histograms(i).frequencies(i);

    return total_frequencies;
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

    const float min = x.minCoeff();
    const float max = x.maxCoeff();

    const VectorR valid = filter_missing_values(x);
    const Index count = valid.size();

    const float mean = (count > 0) ? valid.mean() : 0.0f;

    float standard_deviation = 0;

    if (count > 1)
    {
        const auto valid_d = valid.cast<double>();
        const double sum = valid_d.sum();
        const double squared_sum = valid_d.squaredNorm();
        standard_deviation = sqrt(float((squared_sum - sum * sum / count) / (count - 1)));
    }

    return Descriptives(min, max, mean, standard_deviation);
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

VectorR column_minimums(const MatrixR& matrix,
                        const vector<Index>& row_indices,
                        const vector<Index>& column_indices)
{
    vector<Index> used_column_indices = column_indices;
    if (used_column_indices.empty())
    {
        used_column_indices.resize(matrix.cols());
        iota(used_column_indices.begin(), used_column_indices.end(), 0);
    }

    vector<Index> used_row_indices = row_indices;
    if (used_row_indices.empty())
    {
        used_row_indices.resize(matrix.rows());
        iota(used_row_indices.begin(), used_row_indices.end(), 0);
    }

    const Index row_indices_size = used_row_indices.size();
    const Index column_indices_size = used_column_indices.size();

    VectorR minimums(column_indices_size);

    for (Index j = 0; j < column_indices_size; ++j)
    {
        const Index column_index = used_column_indices[j];

        VectorR column(row_indices_size);

        for (Index i = 0; i < row_indices_size; ++i)
            column(i) = matrix(used_row_indices[i], column_index);

        minimums(j) = minimum(column);
    }

    return minimums;
}

float range(const VectorR& vector)
{
    const float min = minimum(vector);
    const float max = maximum(vector);

    return abs(max - min);
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

VectorR median(const MatrixR& matrix)
{
    const Index columns_number = matrix.cols();

    VectorR medians(columns_number);

    for (Index j = 0; j < columns_number; ++j)
    {
        const auto column = matrix.col(j);

        const Index valid_count = column.array().isFinite().count();

        if (valid_count == 0)
        {
            medians(j) = QUIET_NAN;
            continue;
        }

        VectorR valid_values(valid_count);

        Index k = 0;

        for (Index i = 0; i < column.size(); ++i)
            if (isfinite(column(i)))
                valid_values(k++) = column(i);

        sort(valid_values.data(), valid_values.data() + valid_count);

        medians(j) = (valid_count % 2 == 0)
            ? (valid_values(valid_count / 2 - 1) + valid_values(valid_count / 2)) / 2.0f
            : valid_values(valid_count / 2);
    }

    return medians;
}

float median(const MatrixR& matrix, Index column_index)
{
    vector<float> sorted_column;
    sorted_column.reserve(matrix.rows());

    for (Index i = 0; i < matrix.rows(); ++i)
        if (!isnan(matrix(i, column_index)))
            sorted_column.push_back(matrix(i, column_index));

    if (sorted_column.empty()) return NAN;

    ranges::sort(sorted_column);

    const Index valid_count = ssize(sorted_column);
    const Index median_index = valid_count / 2;

    return (valid_count % 2 == 0)
        ? (sorted_column[median_index - 1] + sorted_column[median_index]) / 2.0f
        : sorted_column[median_index];
}

VectorR median(const MatrixR& matrix, const VectorI& column_indices)
{
    const Index column_indices_size = column_indices.size();

    VectorR medians(column_indices_size);

    for (Index j = 0; j < column_indices_size; ++j)
    {
        const Index column_index = column_indices(j);
        const VectorR column = matrix.col(column_index);

        const Index valid_count = column.array().isFinite().count();

        if (valid_count == 0)
        {
            medians(j) = QUIET_NAN;
            continue;
        }

        VectorR valid_values(valid_count);
        Index k = 0;

        for (Index i = 0; i < column.size(); ++i)
            if (isfinite(column(i)))
                valid_values(k++) = column(i);

        sort(valid_values.data(), valid_values.data() + valid_count);

        medians(j) = (valid_count % 2 == 0)
            ? (valid_values(valid_count / 2 - 1) + valid_values(valid_count / 2)) / 2.0f
            : valid_values(valid_count / 2);
    }

    return medians;
}

VectorR median(const MatrixR& matrix,
               const vector<Index>& row_indices,
               const vector<Index>& column_indices)
{
    const Index row_indices_size = row_indices.size();
    const Index column_indices_size = column_indices.size();

    VectorR medians(column_indices_size);

    for (Index j = 0; j < column_indices_size; ++j)
    {
        const Index column_index = column_indices[j];

        const Index valid_count = count_if(row_indices.data(), row_indices.data() + row_indices_size,
            [&](Index r) { return isfinite(matrix(r, column_index)); });

        if (valid_count == 0)
        {
            medians(j) = QUIET_NAN;
            continue;
        }

        VectorR valid_values(valid_count);
        Index idx = 0;

        for (Index i = 0; i < row_indices_size; ++i)
        {
            const float value = matrix(row_indices[i], column_index);
            if (isfinite(value))
                valid_values(idx++) = value;
        }

        sort(valid_values.data(), valid_values.data() + valid_count);

        medians(j) = (valid_count % 2 == 0)
            ? (valid_values(valid_count / 2 - 1) + valid_values(valid_count / 2)) / 2.0f
            : valid_values(valid_count / 2);
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
MatrixR append_rows(const MatrixR& starting_matrix, const MatrixR& block)
{
    if (starting_matrix.size() == 0)
        return block;
    if (block.size() == 0)
        return starting_matrix;

    if (starting_matrix.cols() != block.cols())
        throw runtime_error(format("append_rows: Column mismatch ({} vs {})",
                                   starting_matrix.cols(), block.cols()));

    MatrixR final_matrix(starting_matrix.rows() + block.rows(), starting_matrix.cols());

    final_matrix.topRows(starting_matrix.rows()) = starting_matrix;
    final_matrix.bottomRows(block.rows()) = block;

    return final_matrix;
}

vector<Index> build_feasible_rows_mask(const MatrixR& outputs, const VectorR& minimums, const VectorR& maximums)
{
    const Index rows_unfiltered = outputs.rows();
    const Index variables_to_filter = outputs.cols();

    if (minimums.size() != variables_to_filter || maximums.size() != variables_to_filter)
        throw runtime_error("build_feasible_rows_mask: Minimums/maximums size mismatch with outputs columns.\n");

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
    ranges::copy_if(data, back_inserter(indices),
                    [bound](Index value) { return value > bound; });
    return indices;
}

VectorI get_nearest_points(const MatrixR& matrix, const VectorR& point, int neighbors_number)
{
    const Index rows = matrix.rows();

    const VectorR distances = (matrix.rowwise() - point.transpose()).rowwise().norm();

    vector<pair<float, Index>> pairs(rows);

    for (Index i = 0; i < rows; ++i)
        pairs[i] = {distances(i), i};

    if (neighbors_number > rows)
        neighbors_number = rows;

    partial_sort(pairs.begin(), pairs.begin() + neighbors_number, pairs.end());

    VectorI result(neighbors_number);
    transform(pairs.begin(), pairs.begin() + neighbors_number, result.data(),
              [](const auto& p) { return p.second; });
    return result;
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
