//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A T I S T I C S
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "statistics.h"
#include "tensors.h"

using namespace std;

namespace opennn
{

Descriptives::Descriptives(const type new_minimum,
                           type new_maximum,
                           type new_mean,
                           type new_standard_deviation) :
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


void Descriptives::set(const type new_minimum, type new_maximum,
                       type new_mean, type new_standard_deviation)
{
    minimum = new_minimum;
    maximum = new_maximum;
    mean = new_mean;
    standard_deviation = new_standard_deviation;
}


void Descriptives::print(const string& title) const
{
    cout << title << endl
         << "Minimum: " << minimum << endl
         << "Maximum: " << maximum << endl
         << "Mean: " << mean << endl
         << "Standard deviation: " << standard_deviation << endl;
}


BoxPlot::BoxPlot(const type new_minimum,
                 type new_first_quartile,
                 type new_median,
                 type new_third_quartile,
                 type new_maximum)
{
    minimum = new_minimum;
    first_quartile = new_first_quartile;
    median = new_median;
    third_quartile = new_third_quartile;
    maximum = new_maximum;
}


void BoxPlot::set(const type new_minimum,
                  type new_first_quartile,
                  type new_median,
                  type new_third_quartile,
                  type new_maximum)
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

    if(!file.is_open())
        throw runtime_error("Cannot open descriptives data file.\n");

    file << "Minimum: " << minimum << endl
         << "Maximum: " << maximum << endl
         << "Mean: " << mean << endl
         << "Standard deviation: " << standard_deviation << endl;

    file.close();
}


Histogram::Histogram(const Index bins_number)
{
    centers.resize(bins_number);
    frequencies.resize(bins_number);
}


Histogram::Histogram(const VectorR&new_centers,
                     const VectorR&new_frequencies)
{
    centers = new_centers;
    frequencies = new_frequencies;
}


Histogram::Histogram(const VectorR& new_frequencies,
                     const VectorR& new_centers,
                     const VectorR& new_minimums,
                     const VectorR& new_maximums)
{
    centers = new_centers;
    frequencies = new_frequencies;
    minimums = new_minimums;
    maximums = new_maximums;
}


Histogram::Histogram(const VectorR& data, Index bins_number)
{
    const type data_maximum = maximum(data);
    const type data_minimum = minimum(data);
    const type step = (data_maximum - data_minimum) / type(bins_number);

    VectorR new_centers(bins_number);

    for(Index i = 0; i < bins_number; i++)
        new_centers(i) = data_minimum + (type(0.5) * step) + (step * type(i));

    VectorR new_frequencies(bins_number);
    new_frequencies.setZero();

    type value;
    Index corresponding_bin;

    for(Index i = 0; i < data.size(); i++)
    {
        value = data(i);
        if(isnan(value)) continue;

        corresponding_bin = int((value - data_minimum) / step);

        if(corresponding_bin >= bins_number)
            corresponding_bin = bins_number - 1;

        new_frequencies(corresponding_bin)++;
    }

    centers = new_centers;
    frequencies = new_frequencies;
}


Histogram::Histogram(const VectorR& probability_data)
{
    const size_t bins_number = 10;
    type data_maximum = maximum(probability_data);
    const type data_minimum = type(0);

    data_maximum = (data_maximum > type(1)) ? type(100.0) : type(1);

    const type step = (data_maximum - data_minimum) / type(bins_number);

    VectorR new_centers(bins_number);

    for(size_t i = 0; i < bins_number; i++)
        new_centers(i) = data_minimum + (type(0.5) * step) + (step * type(i));

    VectorR new_frequencies(bins_number);
    new_frequencies.setZero();

    type value;
    Index corresponding_bin;

    for(Index i = 0; i < probability_data.size(); i++)
    {
        value = probability_data(i);
        corresponding_bin = int((value - data_minimum) / step);

        new_frequencies(corresponding_bin)++;
    }

    centers = new_centers;
    frequencies = new_frequencies;
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
    {
        VectorR nan(1);
        nan << type(NAN);
        return nan;
    }

    Index minimum_frequency = frequencies(0);
    for(Index j = 1; j < frequencies.size(); j++) {
        if (frequencies(j - 1) > frequencies(j)) {
            minimum_frequency = frequencies(j);
        }
    }

    Index minimal_indices_size = 0;

    for(Index i = 0; i < frequencies.size(); i++)
        if(frequencies(i) == minimum_frequency)
            minimal_indices_size++;

    Index index = 0;

    VectorR minimal_centers(minimal_indices_size);

    for(Index i = 0; i < frequencies.size(); i++)
        if(frequencies(i) == minimum_frequency)
            minimal_centers(index++) = type(centers(i));

    return minimal_centers;
}


VectorR Histogram::calculate_maximal_centers() const
{
    const Index maximum_frequency = calculate_maximum_frequency();

    Index maximal_indices_size = 0;

    if (frequencies.size() == 0) {
        VectorR nan(1);
        nan << type(NAN);
        return nan;
    }

    for(Index i = 0; i < frequencies.size(); i++) {
        if (frequencies(i) == maximum_frequency) {
            maximal_indices_size++;
        }
    }

    VectorR maximal_centers(maximal_indices_size);
    Index index = 0;

    for(Index i = 0; i < frequencies.size(); i++)
        if (frequencies(i) == maximum_frequency)
            maximal_centers(index++) = type(centers(i));            

    return maximal_centers;
}


Index Histogram::calculate_bin(const type value) const
{
    const Index bins_number = get_bins_number();

    if (bins_number == 0) return 0;

    const type min_center = centers(0);
    const type max_center = centers(bins_number - 1);
    const type bin_width = (max_center - min_center) / (bins_number - 1);

    for(Index i = 0; i < bins_number; ++i) {
        if (value < centers(i) + bin_width / 2) {
            return i;
        }
    }

    return bins_number - 1; 
}


Index Histogram::calculate_frequency(const type value) const
{
    const Index bins_number = get_bins_number();
    
    if(bins_number == 0) return 0;

    const Index bin_number = calculate_bin(value);

    const Index frequency = frequencies[bin_number];
    
    return frequency;
}


void Histogram::save(const filesystem::path& histogram_file_name) const
{
    const Index bins_number = centers.size();
    ofstream histogram_file(histogram_file_name);

    histogram_file << "centers,frequencies" << endl;

    for(Index i = 0; i < bins_number; i++)
        histogram_file << centers(i) << ","
                       << frequencies(i) << endl;

    histogram_file.close();
}


type minimum(const VectorR& data, const vector<Index>& indices)
{
    const Index size = indices.size();

    if(size == 0) return type(NAN);

    type minimum = numeric_limits<type>::max();

    Index index;

    for(Index i = 0; i < size; i++)
    {
        index = indices[i];

        if(data(index) < minimum && !isnan(data(index)))
            minimum = data(index);
    }

    return minimum;
}


type maximum(const VectorR& data, const vector<Index>& indices)
{
    const Index size = indices.size();

    if(size == 0) return type(NAN);

    type maximum = -numeric_limits<type>::max();

    Index index;

    for(Index i = 0; i < size; i++)
    {
        index = indices[i];

        if(!isnan(data(index)) && data(index) > maximum)
            maximum = data(index);
    }

    return maximum;
}


// Index maximum(const VectorI& vector)
// {
//     if(vector.size() == 0) return 0;

//     const Tensor<Index, 0> m = vector.maximum();

//     return m(0);
// }


VectorR column_maximums(const MatrixR& matrix,
                        const vector<Index>& row_indices,
                        const vector<Index>& column_indices)
{
    const Index rows_number = matrix.rows();
    const Index columns_number = matrix.cols();

    vector<Index> used_column_indices;

    if(column_indices.size() == 0)
    {
        used_column_indices.resize(columns_number);

        iota(used_column_indices.begin(), used_column_indices.end(), 0);
    }
    else
    {
        used_column_indices = column_indices;
    }

    vector<Index> used_row_indices;

    if(row_indices.size() == 0)
    {
        used_row_indices.resize(rows_number);

        iota(used_row_indices.begin(), used_row_indices.end(), 0);
    }
    else
    {
        used_row_indices = row_indices;
    }

    const Index row_indices_size = used_row_indices.size();
    const Index column_indices_size = used_column_indices.size();

    VectorR maximums(column_indices_size);

    Index row_index;
    Index column_index;

    VectorR column(row_indices_size);

    for(Index j = 0; j < column_indices_size; j++)
    {
        column_index = used_column_indices[j];

        for(Index i = 0; i < row_indices_size; i++)
        {
            row_index = used_row_indices[i];

            column(i) = matrix(row_index,column_index);
        }

        maximums(j) = maximum(column);
    }

    return maximums;
}


type mean(const VectorR& v, Index begin, Index end)
{
    if(end == begin) return NAN;

    return v.segment(begin, end - begin + 1).mean();
}


type mean(const VectorR& vector)
{
    auto is_finite = vector.array().isFinite();

    const Index count = is_finite.count();

    if (count == 0) return type(0);

    return is_finite.select(vector.array(), 0.0f).sum() / static_cast<type>(count);
}


type variance(const VectorR& vector)
{
    const Index size = vector.size();

    long double sum = 0.0;
    long double squared_sum = 0.0;

    Index count = 0;

    for(Index i = 0; i < size; i++)
    {
        if (isnan(vector(i))) continue;

        sum += vector(i);
        squared_sum += double(vector(i)) * double(vector(i));

        count++;
    }

    if(count <= 1) return type(0);

    const type variance
        = type(squared_sum/(count - 1) - (sum/count)*(sum/count)*count/(count-1));

    return variance;
}


type variance(const VectorR& vector, const VectorI& indices)
{
    const Index size = indices.size();

    long double sum = 0.0;
    long double squared_sum = 0.0;

    Index count = 0;

    Index index = 0;

    for(Index i = 0; i < size; i++)
    {
        index = indices(i);

        if(!isnan(vector(index)))
        {
            sum += vector(index);
            squared_sum += double(vector(index)) * double(vector(index));

            count++;
        }
    }

    if(count <= 1) return type(0);

    const type variance
        = type(squared_sum/(count - 1) - (sum/count)*(sum/count)*count/(count-1));

    return variance;
}


type standard_deviation(const VectorR& vector)
{
    if(vector.size() == 0) return type(0);

    return sqrt(variance(vector));
}


type median(const VectorR& input_vector)
{
    const Index size = input_vector.size();

    // Fix missing values

    Index new_size = 0;

    for(Index i = 0; i < size; i++)
        if(!isnan(input_vector(i)))
            new_size++;

    vector<Index> sorted_vector;

    for(Index i = 0; i < size; i++)
        if(!isnan(input_vector(i))) {
            sorted_vector.push_back(input_vector(i));
        }

    // Calculate median

    Index median_index;

    if (new_size % 2 == 0)
    {
        median_index = Index(new_size / 2);

        type median = (sorted_vector[median_index - 1] + sorted_vector[median_index]) / type(2.0);

        return median;
    }
    else
    {
        median_index = Index(new_size / 2);

        return sorted_vector[median_index];
    }
}


VectorR quartiles(const VectorR& input_vector)
{
    const Index size = input_vector.size();

    // Fix missing values

    Index new_size = 0;

    for(Index i = 0; i < size; i++)
        if(!isnan(input_vector(i)))
            new_size++;

    vector<type> sorted_vector;

    for(Index i = 0; i < size; i++)
        if(!isnan(input_vector(i)))
            sorted_vector.push_back(input_vector(i));        

    std::sort(sorted_vector.begin(), sorted_vector.end());
    
    // Calculate quartiles

    vector<type> first_sorted_vector;
    vector<type> last_sorted_vector;

    if (new_size % 2 == 0)
    {
        for(Index i = 0; i < new_size / 2; i++)
        {
            first_sorted_vector.push_back(sorted_vector[i]);
            last_sorted_vector.push_back(sorted_vector[i+new_size/2]);
        }
    }
    else
    {
        for(Index i = 0; i < new_size / 2; i++)
        {
            first_sorted_vector.push_back(sorted_vector[i]);
            last_sorted_vector.push_back(sorted_vector[i + new_size / 2+1]);
        }
    }

    VectorR quartiles(3);

    if (new_size == 1)
    {
        quartiles(0) = sorted_vector[0];
        quartiles(1) = sorted_vector[0];
        quartiles(2) = sorted_vector[0];
    }
    else if (new_size == 2)
    {
        quartiles(0) = (sorted_vector[0] + sorted_vector[1]) / type(4);
        quartiles(1) = (sorted_vector[0] + sorted_vector[1]) / type(2);
        quartiles(2) = (sorted_vector[0] + sorted_vector[1]) / type(3.0/4.0);
    }
    else if (new_size == 3)
    {
        quartiles(0) = (sorted_vector[0] + sorted_vector[1]) / type(2);
        quartiles(1) = sorted_vector[1];
        quartiles(2) = (sorted_vector[2] + sorted_vector[1]) / type(2);
    }
    else
    {
        Index median_index;

        if (new_size % 2 == 0)
        {
            median_index = Index(new_size / 2);

            quartiles(1) = (sorted_vector[median_index - 1] + sorted_vector[median_index]) / type(2.0);

        }
        else
        {
            median_index = Index(new_size / 2);

            quartiles(1) = sorted_vector[median_index];
        }

        const Index first_vector_size = first_sorted_vector.size();
        const Index last_vector_size = first_sorted_vector.size();

        if (first_vector_size % 2 == 0 && last_vector_size % 2 == 0)
        {
            median_index = Index(first_vector_size / 2);

            quartiles(0) = (first_sorted_vector[median_index - 1] + first_sorted_vector[median_index]) / type(2.0);
            quartiles(2) = (last_sorted_vector[median_index - 1] + last_sorted_vector[median_index]) / type(2.0);

        }
        else if (first_vector_size % 2 == 0 && last_vector_size % 2 != 0)
        {
            const Index median_index_first = Index(first_vector_size / 2);

            quartiles(0) = (first_sorted_vector[median_index_first - 1] + first_sorted_vector[median_index_first]) / type(2.0);

            const Index median_index_last = Index(last_vector_size / 2);

            quartiles(2) = last_sorted_vector[median_index_last];
        }
        else if (first_vector_size % 2 != 0 && last_vector_size % 2 == 0)
        {
            const Index median_index_first = Index(first_vector_size / 2);

            quartiles(0) = first_sorted_vector[median_index_first];

            const Index median_index_last = Index(last_vector_size / 2);

            quartiles(2) = (last_sorted_vector[median_index_last - 1] + last_sorted_vector[median_index_last]) / type(2.0);

        }
        else
        {
            median_index = Index(first_vector_size / 2);

            quartiles(0) = first_sorted_vector[median_index];
            quartiles(2) = last_sorted_vector[median_index];
        }
    }

    return quartiles;
}


VectorR quartiles(const VectorR& data, const vector<Index>& indices)
{
    const Index indices_size = indices.size();

    // Fix missing values

    Index index;
    Index new_size = 0;

    for(Index i = 0; i < indices_size; i++)
        if(!isnan(data(indices[i])))
            new_size++;

    VectorR sorted_vector(new_size);

    Index sorted_index = 0;

    for(Index i = 0; i < indices_size; i++)
    {
        index = indices[i];

        if(!isnan(data(index)))
            sorted_vector(sorted_index++) = data(index);
    }

    sort(sorted_vector.data(), sorted_vector.data() + sorted_vector.size(), less<type>());

    // Calculate quartiles

    VectorR first_sorted_vector(new_size/2);
    VectorR last_sorted_vector(new_size/2);

    for(Index i = 0; i < new_size/2 ; i++)
        first_sorted_vector(i) = sorted_vector(i);

    for(Index i = 0; i < new_size/2; i++)
        last_sorted_vector(i) = sorted_vector(i + new_size - new_size/2);

    VectorR quartiles(3);

    if(new_size == 1)
    {
        quartiles(0) = sorted_vector(0);
        quartiles(1) = sorted_vector(0);
        quartiles(2) = sorted_vector(0);
    }
    else if(new_size == 2)
    {
        quartiles(0) = (sorted_vector(0)+sorted_vector(1))/ type(4);
        quartiles(1) = (sorted_vector(0)+sorted_vector(1))/ type(2);
        quartiles(2) = (sorted_vector(0)+sorted_vector(1))* type(3/4);
    }
    else if(new_size == 3)
    {
        quartiles(0) = (sorted_vector(0)+sorted_vector(1))/ type(2);
        quartiles(1) = sorted_vector(1);
        quartiles(2) = (sorted_vector(2)+sorted_vector(1))/ type(2);
    }
    else if(new_size % 2 == 0)
    {
        Index median_index = Index(first_sorted_vector.size() / 2);
        quartiles(0) = (first_sorted_vector(median_index-1) + first_sorted_vector(median_index)) / type(2.0);

        median_index = Index(new_size / 2);
        quartiles(1) = (sorted_vector(median_index-1) + sorted_vector(median_index)) / type(2.0);

        median_index = Index(last_sorted_vector.size() / 2);
        quartiles(2) = (last_sorted_vector(median_index-1) + last_sorted_vector(median_index)) / type(2.0);
    }
    else
    {
        quartiles(0) = sorted_vector(new_size/4);
        quartiles(1) = sorted_vector(new_size/2);
        quartiles(2) = sorted_vector(new_size*3/4);
    }

    return quartiles;
}


BoxPlot box_plot(const VectorR& vector)
{
    BoxPlot box_plot;

    if(vector.size() == 0)
        return box_plot;
 
    const VectorR quartiles = opennn::quartiles(vector);

    box_plot.minimum = minimum(vector);
    box_plot.first_quartile = quartiles(0);
    box_plot.median = quartiles(1);
    box_plot.third_quartile = quartiles(2);
    box_plot.maximum = maximum(vector);

    return box_plot;
}


BoxPlot box_plot(const VectorR& data, const vector<Index>& indices)
{
    BoxPlot box_plot;

    if(data.size() == 0 || indices.size() == 0)
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
    VectorR frequencies(bins_number);
    frequencies.setZero();

    vector<type> unique_values;

    unique_values.reserve(min<Index>(size, bins_number));
    unique_values.push_back(new_vector(0));

    for(Index i = 1; i < size; i++)
    {
        const type value = new_vector(i);

        if(!isnan(value))
            if (find(unique_values.begin(), unique_values.end(), value) == unique_values.end())
            {
                unique_values.push_back(value);

                if (static_cast<Index>(unique_values.size()) > bins_number)
                    break;
            }
    }

    const Index unique_values_number = static_cast<Index>(unique_values.size());
    if(unique_values_number <= bins_number)
    {
        sort(unique_values.data(), unique_values.data() + unique_values.size(), less<type>());

        VectorR tensor_unique(unique_values.size());

        for(Index i = 0; i < Index(unique_values.size()); ++i)
            tensor_unique(i) = unique_values[i];

        centers = tensor_unique;
        minimums = tensor_unique;
        maximums = tensor_unique;

        frequencies.resize(unique_values_number);
        frequencies.setZero();

        for(Index i = 0; i < size; i++)
        {
            if(isnan(new_vector(i))) continue;

            for(Index j = 0; j < unique_values_number; j++)
            {
                if(new_vector(i) - centers(j) < NUMERIC_LIMITS_MIN)
                {
                    frequencies(j)++;
                    break;
                }
            }
        }
    }
    else
    {
        const type min = minimum(new_vector);
        const type max = maximum(new_vector);

        const type length = (max - min) /type(bins_number);

        minimums(0) = min;
        maximums(0) = min + length;
        centers(0) = (maximums(0) + minimums(0)) /type(2.0);

        // Calculate bins center

        for(Index i = 1; i < bins_number; i++)
        {
            minimums(i) = minimums(i - 1) + length;
            maximums(i) = maximums(i - 1) + length;

            centers(i) = (maximums(i) + minimums(i)) /type(2.0);
        }

        // Calculate bins frequency

        const Index size = new_vector.size();

        for(Index i = 0; i < size; i++)
        {
            if(isnan(new_vector(i))) continue;

            for(Index j = 0; j < bins_number - 1; j++)
            {
                if(new_vector(i) >= minimums(j) && new_vector(i) < maximums(j))
                {
                    frequencies(j)++;
                    break;
                }
            }

            if(new_vector(i) >= minimums(bins_number - 1))
                frequencies(bins_number - 1)++;
        }
    }

    Histogram histogram(bins_number);
    histogram.centers = centers;
    histogram.minimums = minimums;
    histogram.maximums = maximums;
    histogram.frequencies = frequencies;

    return histogram;
}


Histogram histogram_centered(const VectorR& vector, type center, Index bins_number)
{
    const Index bin_center = (bins_number % 2 == 0) 
        ? Index(type(bins_number) / type(2.0)) 
        : Index(type(bins_number) / type(2.0) + type(0.5));

    VectorR minimums(bins_number);
    VectorR maximums(bins_number);

    VectorR centers(bins_number);
    VectorR frequencies(bins_number);
    frequencies.setZero();

    const type min = minimum(vector);
    const type max = maximum(vector);

    const type length = (max - min)/type(bins_number);

    minimums(bin_center-1) = center - length;
    maximums(bin_center-1) = center + length;
    centers(bin_center-1) = center;

    // Calculate bins center

    for(Index i = bin_center; i < bins_number; i++) // Upper centers
    {
        minimums(i) = minimums(i - 1) + length;
        maximums(i) = maximums(i - 1) + length;

        centers(i) = (maximums(i) + minimums(i)) /type(2.0);
    }

    for(Index i = Index(bin_center)-2; i >= 0; i--) // Lower centers
    {
        minimums(i) = minimums(i+1) - length;
        maximums(i) = maximums(i+1) - length;

        centers(i) = (maximums(i) + minimums(i)) /type(2.0);
    }

    // Calculate bins frequency

    const Index size = vector.size();

    for(Index i = 0; i < size; i++)
    {
        for(Index j = 0; j < bins_number - 1; j++)
            if(vector(i) >= minimums(j) && vector(i) < maximums(j))
                frequencies(j)++;

        if(vector(i) >= minimums(bins_number - 1))
            frequencies(bins_number - 1)++;
    }

    Histogram histogram(bins_number);
    histogram.centers = centers;
    histogram.minimums = minimums;
    histogram.maximums = maximums;
    histogram.frequencies = frequencies;

    return histogram;
}


Histogram histogram(const VectorB& v)
{
    VectorR minimums(2);
    minimums.setZero();

    VectorR maximums(2);
    maximums.setConstant(type(1));

    VectorR centers(2);
    centers << type(0), type(1);

    VectorR frequencies(2);
    frequencies.setZero();

    // Calculate bins frequency

    const Index size = v.size();

    for(Index i = 0; i < size; i++)
        for(Index j = 0; j < 2; j++)
            if(Index(v(i)) == Index(minimums(j)))
                frequencies(j)++;

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

    for(Index i = 0; i < histograms_number; i++)
        total_frequencies(i) = histograms(i).frequencies(i);

    return total_frequencies;
}


vector<Histogram> histograms(const MatrixR& matrix, Index bins_number)
{
    const Index columns_number = matrix.cols();

    vector<Histogram> histograms(columns_number);

    for(Index i = 0; i < columns_number; i++)
    {
        const VectorR column = VectorR(vector_map(matrix, i));
        histograms[i] = histogram(column, bins_number);
    }

    return histograms;
}


Descriptives vector_descriptives(const VectorR& x)
{
    Descriptives my_descriptives;

    const Index size = x.size();

    if (size <= 0)
        return my_descriptives;

    const type minimum = x.minCoeff();
    const type maximum = x.maxCoeff();

    long double sum = 0.0;
    long double squared_sum = 0;
    Index count = 0;

    for(Index i = 0; i < size; i++)
    {
        if (isnan(x(i))) continue;

        if (i >= 0 && i < x.size())
            sum += x(i);
        else
            cerr << "Index out of range: " << i << endl;

        squared_sum += double(x(i)) * double(x(i));
        count++;
    }

    type mean = 0;

    if (count > 0)
        mean = type(sum / count);

    type standard_deviation = 0;

    if (count > 1)
    {
        const type numerator = type(squared_sum - sum * sum / count);
        const type denominator = type(count - 1);
        standard_deviation = sqrt(numerator / denominator);
    }

    my_descriptives.set(minimum, maximum, mean, standard_deviation);

    return my_descriptives;
}


vector<Descriptives> descriptives(const MatrixR& matrix)
{
    const Index rows_number = matrix.rows();
    const Index columns_number = matrix.cols();

    vector<Descriptives> descriptives(columns_number);
    VectorR column(rows_number);

    for(Index i = 0; i < columns_number; i++)
    {
        column = matrix.col(i);

        if (i >= 0 && i < Index(descriptives.size()))
            descriptives[i] = vector_descriptives(column);   
        else
            cerr << "Index out of range: " << i << endl;
    }

    return descriptives;
}


vector<Descriptives> descriptives(const MatrixR& matrix,
                                  const vector<Index>& row_indices,
                                  const vector<Index>& column_indices)
{
    const Index row_indices_size = static_cast<Index>(row_indices.size());
    const Index column_indices_size = static_cast<Index>(column_indices.size());

    vector<Descriptives> descriptives_results(column_indices_size);

    // Using VectorR (Matrix API) instead of VectorR
    VectorR minimums = VectorR::Zero(column_indices_size);
    VectorR maximums = VectorR::Zero(column_indices_size);

    // Use double precision for intermediate accumulation
    VectorXd sums = VectorXd::Zero(column_indices_size);
    VectorXd squared_sums = VectorXd::Zero(column_indices_size);

    // Count remains VectorI (assuming Matrix<Index, Dynamic, 1>)
    VectorI count = VectorI::Zero(column_indices_size);

#pragma omp parallel for
    for(Index j = 0; j < column_indices_size; j++)
    {
        const Index column_index = column_indices[j];

        type current_min = 0;
        type current_max = 0;
        double current_sum = 0;
        double current_sq_sum = 0;
        Index current_count = 0;
        bool first_iteration = true;

        for(Index i = 0; i < row_indices_size; i++)
        {
            const Index row_index = row_indices[i];
            const type value = matrix(row_index, column_index);

            if (std::isnan(value)) continue;

            if (first_iteration)
            {
                current_min = value;
                current_max = value;
                first_iteration = false;
            }
            else {
                if (value < current_min) current_min = value;
                if (value > current_max) current_max = value;
            }

            current_sum += static_cast<double>(value);
            current_sq_sum += static_cast<double>(value) * static_cast<double>(value);
            current_count++;
        }

        minimums(j) = current_min;
        maximums(j) = current_max;
        sums(j) = current_sum;
        squared_sums(j) = current_sq_sum;
        count(j) = current_count;
    }

    const VectorXd mean = sums.array() / count.cast<double>().array();
    VectorXd standard_deviation = VectorXd::Zero(column_indices_size);

    #pragma omp parallel for
    for(Index i = 0; i < column_indices_size; i++)
    {
        if (count(i) > 1)
        {
            const double n = static_cast<double>(count(i));
            const double variance = (squared_sums(i) - (sums(i) * sums(i) / n)) / (n - 1.0);
            standard_deviation(i) = sqrt(max(0.0, variance));
        }

        // Populate the results vector
        descriptives_results[i].set(minimums(i),
                                    maximums(i),
                                    static_cast<type>(mean(i)),
                                    static_cast<type>(standard_deviation(i)));
    }

    return descriptives_results;
}


VectorR column_minimums(const MatrixR& matrix,
                        const vector<Index>& row_indices,
                        const vector<Index>& column_indices)
{
    const Index rows_number = matrix.rows();
    const Index columns_number = matrix.cols();

    vector<Index> used_column_indices;

    if(column_indices.size() == 0)
    {
        used_column_indices.resize(columns_number);

        iota(used_column_indices.begin(), used_column_indices.end(), 0);
    }
    else
    {
        used_column_indices = column_indices;
    }

    vector<Index> used_row_indices;

    if(row_indices.size() == 0)
    {
        used_row_indices.resize(rows_number);

        iota(used_row_indices.begin(), used_row_indices.end(), 0);
    }
    else
    {
        used_row_indices = row_indices;
    }

    const Index row_indices_size = used_row_indices.size();
    const Index column_indices_size = used_column_indices.size();

    VectorR minimums(column_indices_size);

    Index row_index;
    Index column_index;

    for(Index j = 0; j < column_indices_size; j++)
    {
        column_index = used_column_indices[j];

        VectorR column(row_indices_size);

        for(Index i = 0; i < row_indices_size; i++)
        {
            row_index = used_row_indices[i];

            column(i) = matrix(row_index,column_index);
        }

        minimums(j) = minimum(column);
    }

    return minimums;
}


VectorR column_maximums(const MatrixR& matrix, const vector<Index>& column_indices)
{
    const Index rows_number = matrix.rows();
    const Index columns_number = matrix.cols();

    vector<Index> used_column_indices;

    if(column_indices.size() == 0)
        used_column_indices.resize(columns_number);
    else
        used_column_indices = column_indices;

    const Index column_indices_size = used_column_indices.size();

    VectorR maximums(column_indices_size);

    Index column_index;
    VectorR column(rows_number);

    for(Index i = 0; i < column_indices_size; i++)
    {
        column_index = used_column_indices[i];

        column = matrix.col(column_index);

        maximums(i) = maximum(column);
    }

    return maximums;
}


type range(const VectorR& vector)
{
    const type min = minimum(vector);
    const type max = maximum(vector);

    return abs(max - min);
}


VectorR mean(const MatrixR& matrix)
{
    const Index rows_number = matrix.rows();
    const Index columns_number = matrix.cols();

    // Mean

    VectorR mean(columns_number);
    mean.setZero();

    for(Index j = 0; j < columns_number; j++)
    {
        for(Index i = 0; i < rows_number; i++)
            if(!isnan(matrix(i, j)))
                mean(j) += matrix(i, j);

        mean(j) /= type(rows_number);
    }

    return mean;
}


VectorR mean(const MatrixR& matrix, const VectorI& column_indices)
{
    const Index rows_number = matrix.rows();

    const Index column_indices_size = column_indices.size();

    Index column_index;

    // Mean

    VectorR mean(column_indices_size);
    mean.setZero();

    for(Index j = 0; j < column_indices_size; j++)
    {
        column_index = column_indices(j);

        for(Index i = 0; i < rows_number; i++)
            mean(j) += matrix(i, column_index);

        mean(j) /= type(rows_number);
    }

    return mean;
}


VectorR mean(const MatrixR& matrix, const vector<Index>& row_indices, const vector<Index>& column_indices)
{
    const Index row_indices_size = row_indices.size();
    const Index column_indices_size = column_indices.size();

    if(row_indices_size == 0 || column_indices_size == 0) 
        return VectorR();

    Index row_index;
    Index column_index;

    Index count = 0;

    // Mean

    VectorR mean(column_indices_size);
    mean.setZero();
    
    for(Index j = 0; j < column_indices_size; j++)
    {
        column_index = column_indices[j];

        count = 0;

        for(Index i = 0; i < row_indices_size; i++)
        {
            row_index = row_indices[i];

            if (isnan(matrix(row_index, column_index))) continue;

            mean(j) += matrix(row_index,column_index);
            count++;            
        }

        mean(j) /= type(count);
    }
    
    return mean;
}


type mean(const MatrixR& matrix, Index column_index)
{
    const Index rows_number = matrix.rows();
    const Index columns_number = matrix.cols();

    if(rows_number == 0 && columns_number == 0) return type(NAN);

    if(rows_number == 0 && columns_number == 0) return type(NAN);

    // Mean

    type mean = type(0);

    Index count = 0;

    for(Index i = 0; i < rows_number; i++)
    {
        if(!isnan(matrix(i,column_index)))
        {
            mean += matrix(i,column_index);
            count++;
        }
    }

    mean /= type(count);

    return mean;
}


VectorR median(const MatrixR& matrix)
{
    const Index columns_number = matrix.cols();

    VectorR medians(columns_number);

    for(Index j = 0; j < columns_number; j++)
    {
        const auto column = matrix.col(j);

        const Index n = column.array().isFinite().count();

        if (n == 0)
        {
            medians(j) = numeric_limits<type>::quiet_NaN();
            continue;
        }

        VectorR valid_values(n);

        Index k = 0;

        for(Index i = 0; i < column.size(); ++i)
            if(isfinite(column(i)))
                valid_values(k++) = column(i);

        sort(valid_values.data(), valid_values.data() + n);

        (n % 2 == 0)
            ? medians(j) = (valid_values(n / 2 - 1) + valid_values(n / 2)) / 2.0f
            : medians(j) = valid_values(n / 2);

    }

    return medians;
}


type median(const MatrixR& matrix, Index column_index)
{
    type median = type(0);

    vector<type> sorted_column;

    Index rows_number = 0;

    for(Index i = 0; i < matrix.rows(); i++)
    {
        if(isnan(matrix(i,column_index)))
            continue;

        sorted_column.push_back(matrix(i, column_index));
        rows_number++;
    }

    Index median_index;

    if (rows_number % 2 == 0)
    {
        median_index = type(rows_number / 2);

        return (sorted_column[median_index - 1] + sorted_column[median_index]) / type(2.0);
    }
    else
    {
        median_index = Index(rows_number / 2);

        return sorted_column[median_index];
    }

    return median;
}


VectorR median(const MatrixR& matrix, const VectorI& column_indices)
{
    const Index rows_number = matrix.rows();
    const Index column_indices_size = column_indices.size();

    VectorR medians(column_indices_size);

    for(Index j = 0; j < column_indices_size; j++)
    {
        const Index column_index = column_indices(j);
        const VectorR column = matrix.col(column_index);

        const Index n = column.array().isFinite().count();

        if (n == 0)
        {
            medians(j) = numeric_limits<type>::quiet_NaN();
            continue;
        }

        VectorR valid_values(n);
        Index k = 0;

        for(Index i = 0; i < column.size(); i++)
            if(isfinite(column(i)))
                valid_values(k++) = column(i);

        sort(valid_values.data(), valid_values.data() + n);

        medians(j) = (n % 2 == 0)
                         ? (valid_values(n / 2 - 1) + valid_values(n / 2)) / 2.0f
                         : valid_values(n / 2);
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
/*
    for(Index j = 0; j < column_indices_size; j++)
    {
        const Index column_index = column_indices[j];
        Index n = 0;

        for(Index k = 0; k < row_indices_size; k++)
            if(isfinite(matrix(row_indices, column_index)))
                n++;

        if (n == 0)
        {
            medians(j) = numeric_limits<type>::quiet_NaN();
            continue;
        }

        VectorR valid_values(n);
        Index idx = 0;

        for(Index row_index = 0; row_index < row_indices_size; row_index++)
            if(isfinite(matrix(row_indices, column_index)))
                valid_values(idx++) = matrix(row_index, column_index);

        sort(valid_values.data(), valid_values.data() + n);

        medians(j) = (n % 2 == 0)
            ? (valid_values(n / 2 - 1) + valid_values(n / 2)) / 2.0f
            : valid_values(n / 2);
    }
*/
    return medians;
}


Index minimal_index(const VectorR& vector)
{
    if(vector.size() == 0) return 0;

    Index index;
    vector.minCoeff(&index);

    return index;
}


Index maximal_index(const VectorR& vector)
{
    if(vector.size() == 0) return 0;

    Index index;
    vector.minCoeff(&index);

    return index;
}


VectorI minimal_indices(const VectorR& input_vector, Index number)
{
    VectorR copy_vector = input_vector;

    const Index size = copy_vector.size();
    VectorI minimal_indices(number);

    Index val_max=0;

    for(Index i = 0; i < size; i++)
        if (input_vector(i) > val_max)
            val_max = input_vector(i);

    for(Index j = 0; j < number; j++)
    {
        Index minimal_index = 0;
        type minimum = copy_vector[0];

        for(Index i = 0; i < size; i++)
        {
            if (copy_vector[i] < minimum)
            {
                minimal_index = i;
                minimum = copy_vector[i];
            }
        }

        copy_vector[minimal_index] = val_max + type(1);
        minimal_indices(j) = minimal_index;
    }

    return minimal_indices;
}


VectorI maximal_indices(const VectorR& input_vector, Index number)
{
    VectorR copy_vector = input_vector;

    const Index size = copy_vector.size();

    Index val_min = 0;
    for(Index i = 0; i < size; i++)
        if (input_vector(i) < val_min)
            val_min = input_vector(i);            

    VectorI maximal_indices(number);

    for(Index j = 0; j < number; j++)
    {
        Index maximal_index = 0;
        type maximal = copy_vector[0];

        for(Index i = 0; i < size; i++)
        {
            if (copy_vector[i] > maximal)
            {
                maximal_index = i;
                maximal = copy_vector[i];
            }
        }

        copy_vector[maximal_index] = val_min - type(1);
        maximal_indices(j) = maximal_index;
    }

    return maximal_indices;
}


VectorI minimal_indices(const MatrixR& matrix)
{
    VectorI minimal_indices(2);

    Index minRow, minCol;

    matrix.minCoeff(&minRow, &minCol);

    minimal_indices << minRow, minCol;

    return minimal_indices;
}


VectorI maximal_indices(const MatrixR& matrix)
{
    VectorI maximal_indices(2);

    Index maxRow, maxCol;

    matrix.maxCoeff(&maxRow, &maxCol);

    maximal_indices << maxRow, maxCol;

    return maximal_indices;
}


VectorR percentiles(const VectorI& input_vector)
{
    const Index n = input_vector.array().isFinite().count();

    if (n == 0)
        return VectorR::Constant(1, numeric_limits<type>::quiet_NaN());

    VectorR sorted(n);

    Index j = 0;

    for (Index i = 0; i < input_vector.size(); ++i)
        if (isfinite(input_vector(i)))
            sorted(j++) = input_vector(i);

    sort(sorted.data(), sorted.data() + sorted.size());

    VectorR result(10);

    for (Index i = 0; i < 9; i++)
    {
        const Index pos_scaled = n * (i + 1);

        if (pos_scaled % 10 == 0)
        {
            const Index k = pos_scaled / 10;
            result(i) = (sorted(k - 1) + sorted(k)) / 2.0f;
        }
        else
        {
            result(i) = sorted(pos_scaled / 10);
        }
    }

    result(9) = sorted.maxCoeff();

    return result;
}

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
