//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A T I S T I C S
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "statistics.h"
#include "tensors.h"
#include "descriptives.h"
#include "box_plot.h"
#include "histogram.h"

using namespace std;

namespace opennn
{

Descriptives::Descriptives(const type& new_minimum,
                           const type& new_maximum,
                           const type& new_mean,
                           const type& new_standard_deviation)
{
    minimum = new_minimum;
    maximum = new_maximum;
    mean = new_mean;
    standard_deviation = new_standard_deviation;

}


Tensor<type, 1> Descriptives::to_tensor() const
{
    Tensor<type, 1> descriptives_tensor(4);
    descriptives_tensor.setValues({minimum, maximum, mean, standard_deviation});

    return descriptives_tensor;
}


void Descriptives::set(const type& new_minimum, const type& new_maximum,
                       const type& new_mean, const type& new_standard_deviation)
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


BoxPlot::BoxPlot(const type& new_minimum,
                 const type& new_first_cuartile,
                 const type& new_median,
                 const type& new_third_quartile,
                 const type& new_maximum)
{
    minimum = new_minimum;
    first_quartile = new_first_cuartile;
    median = new_median;
    third_quartile = new_third_quartile;
    maximum = new_maximum;
}


void BoxPlot::set(const type& new_minimum,
                  const type& new_first_cuartile,
                  const type& new_median,
                  const type& new_third_quartile,
                  const type& new_maximum)
{
    minimum = new_minimum;
    first_quartile = new_first_cuartile;
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


Histogram::Histogram(const Index& bins_number)
{
    centers.resize(bins_number);
    frequencies.resize(bins_number);
}


Histogram::Histogram(const Tensor<type, 1>&new_centers,
                     const Tensor<Index, 1>&new_frequencies)
{
    centers = new_centers;
    frequencies = new_frequencies;
}


Histogram::Histogram(const Tensor<Index, 1>& new_frequencies,
                     const Tensor<type, 1>& new_centers,
                     const Tensor<type, 1>& new_minimums,
                     const Tensor<type, 1>& new_maximums)
{
    centers = new_centers;
    frequencies = new_frequencies;
    minimums = new_minimums;
    maximums = new_maximums;
}


Histogram::Histogram(const Tensor<type, 1>& data,
                     const Index& bins_number)
{
    const type data_maximum = maximum(data);
    const type data_minimum = minimum(data);
    const type step = (data_maximum - data_minimum) / type(bins_number);

    Tensor<type, 1> new_centers(bins_number);

    for(Index i = 0; i < bins_number; i++)
        new_centers(i) = data_minimum + (type(0.5) * step) + (step * type(i));

    Tensor<Index, 1> new_frequencies(bins_number);
    new_frequencies.setZero();

    type value;
    Index corresponding_bin;

    for(Index i = 0; i < data.dimension(0); i++)
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


Histogram::Histogram(const Tensor<type, 1>& probability_data)
{
    const size_t bins_number = 10;
    type data_maximum = maximum(probability_data);
    const type data_minimum = type(0);

    data_maximum = (data_maximum > type(1)) ? type(100.0) : type(1);

    const type step = (data_maximum - data_minimum) / type(bins_number);

    Tensor<type, 1> new_centers(bins_number);

    for(size_t i = 0; i < bins_number; i++)
        new_centers(i) = data_minimum + (type(0.5) * step) + (step * type(i));

    Tensor<Index, 1> new_frequencies(bins_number);
    new_frequencies.setZero();

    type value;
    Index corresponding_bin;

    for(Index i = 0; i < probability_data.dimension(0); i++)
    {
        value = probability_data(i);
        corresponding_bin = int((value - data_minimum) / step);

        new_frequencies(corresponding_bin)++;
    }

    centers = new_centers;
    frequencies = new_frequencies;
    /*
    cout << "Tamano de frequencies: " << frequencies.size() << "\n";
    cout << "Valores de frequencies: ";
    for (Index i = 0; i < frequencies.size(); ++i) {
        cout << frequencies(i) << " ";
    }
    cout << "\n";
    */
}


Index Histogram::get_bins_number() const
{
    return centers.size();
}


Index Histogram::count_empty_bins() const
{
    const auto size = frequencies.dimension(0);

    Index count = 0;

    #pragma omp parallel for reduction(+: count)

    for(Index i = 0; i < size; i++)
        if(frequencies(i) == 0) 
            count++;

    return count;
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

    Index max_element=frequencies(0);
    for (Index j = 1; j < frequencies.size(); j++) {
        if (frequencies(j - 1) < frequencies(j)) {
            max_element = frequencies(j);
        }
    }
    
    for (Index i = 0; i < frequencies.size(); i++) {
        if (frequencies(i) == max_element) {
            return i;
        }
    }

    return 0;
} 


Tensor<type, 1> Histogram::calculate_minimal_centers() const
{
    if (frequencies.size() == 0)
    {
        Tensor<type, 1> nan(1);
        nan.setValues({ type(NAN) });
        return nan;
    }

    Index minimum_frequency = frequencies(0);
    for (Index j = 1; j < frequencies.size(); j++) {
        if (frequencies(j - 1) > frequencies(j)) {
            minimum_frequency = frequencies(j);
        }
    }

    Index minimal_indices_size = 0;

    for(Index i = 0; i < frequencies.size(); i++)
        if(frequencies(i) == minimum_frequency)
            minimal_indices_size++;

    Index index = 0;

    Tensor<type, 1> minimal_centers(minimal_indices_size);

    for(Index i = 0; i < frequencies.size(); i++)
        if(frequencies(i) == minimum_frequency)
            minimal_centers(index++) = type(centers(i));

    return minimal_centers;
}


Tensor<type, 1> Histogram::calculate_maximal_centers() const
{
    const Index maximum_frequency = calculate_maximum_frequency();

    Index maximal_indices_size = 0;

    if (frequencies.size() == 0) {
        Tensor<type, 1> nan(1);
        nan.setValues({ type(NAN) });
        return nan;
    }

    for (Index i = 0; i < frequencies.size(); i++) {
        if (frequencies(i) == maximum_frequency) {
            maximal_indices_size++;
        }
    }

    Tensor<type, 1> maximal_centers(maximal_indices_size);
    Index index = 0;

    for (Index i = 0; i < frequencies.size(); i++) {
        if (frequencies(i) == maximum_frequency) {
            maximal_centers(index++) = type(centers(i));
        }
    }

    return maximal_centers;
}


Index Histogram::calculate_bin(const type& value) const
{
    const Index bins_number = get_bins_number();

    if (bins_number == 0) return 0;

    const type min_center = centers(0);
    const type max_center = centers(bins_number - 1);
    const type bin_width = (max_center - min_center) / (bins_number - 1);

    for (Index i = 0; i < bins_number; ++i) {
        if (value < centers(i) + bin_width / 2) {
            return i;
        }
    }

    return bins_number - 1; 
}


Index Histogram::calculate_frequency(const type&value) const
{
    const Index bins_number = get_bins_number();
    
    if(bins_number == 0) return 0;

    const Index bin_number = calculate_bin(value);

    const Index frequency = frequencies[bin_number];
    
    return frequency;
}


void Histogram::save(const filesystem::path& histogram_file_name) const
{
    const Index bins_number = centers.dimension(0);
    ofstream histogram_file(histogram_file_name);

    histogram_file << "centers,frequencies" << endl;

    for(Index i = 0; i < bins_number; i++)
        histogram_file << centers(i) << ","
                       << frequencies(i) << endl;

    histogram_file.close();
}


type minimum(const Tensor<type, 1>& vector)
{
    const Index size = vector.dimension(0);

    if(size == 0) return type(NAN);

    type minimum = numeric_limits<type>::max();

    for(Index i = 0; i < size; i++)
        if(vector(i) < minimum && !isnan(vector(i)))
            minimum = vector(i);

    return minimum;
}


Index minimum(const Tensor<Index, 1>& vector)
{
    const Index size = vector.size();

    if(size == 0) return Index(NAN);

    Index minimum = numeric_limits<Index>::max();

    for(Index i = 0; i < size; i++)
        if(vector(i) < minimum)
            minimum = vector(i);

    return minimum;
}


type minimum(const Tensor<type, 1>& data, const vector<Index>& indices)
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


type maximum(const Tensor<type, 1>& vector)
{
    const Index size = vector.dimension(0);

    if(size == 0) return type(NAN);

    type maximum = -numeric_limits<type>::max();

    for(Index i = 0; i < size; i++)
        if(!isnan(vector(i)) && vector(i) > maximum)
            maximum = vector(i);

    return maximum;
}


type maximum(const Tensor<type, 1>& data, const vector<Index>& indices)
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


Index maximum(const Tensor<Index, 1>& vector)
{
    const Index size = vector.size();

    if(size == 0) return Index(NAN);

    Index maximum = -numeric_limits<Index>::max();

    for(Index i = 0; i < size; i++)
        if(vector(i) > maximum)
            maximum = vector(i);

    return maximum;
}


Tensor<type, 1> column_maximums(const Tensor<type, 2>& matrix,
                                const vector<Index>& row_indices,
                                const vector<Index>& column_indices)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

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

    Tensor<type, 1> maximums(column_indices_size);

    Index row_index;
    Index column_index;

    Tensor<type, 1> column(row_indices_size);

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


type mean(const Tensor<type, 1>& vector, const Index& begin, const Index& end)
{
    if(end == begin) return vector[begin];

    long double sum = 0.0;

    for(Index i = begin; i <= end; i++)
        sum += vector(i);

    return type(sum/(end-begin+1));
}


type mean(const Tensor<type, 1>& vector)
{
    const Index size = vector.dimension(0);

    if(size == 0) return type(0);

    long double sum = 0.0;

    Index count = 0;

    for(Index i = 0; i < size; i++)
    {
        if (isnan(vector(i))) continue;

        sum += vector(i);
        count++;
    }

    const type mean = type(sum/count);

    return mean;
}


type variance(const Tensor<type, 1>& vector)
{
    const Index size = vector.dimension(0);

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


type variance(const Tensor<type, 1>& vector, const Tensor<Index, 1>& indices)
{
    const Index size = indices.dimension(0);

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


type standard_deviation(const Tensor<type, 1>& vector)
{
    if(vector.size() == 0) return type(0);

    return sqrt(variance(vector));
}

type median(const Tensor<type, 1>& vector) {
    const Index size = vector.dimension(0);

    // Fix missing values

    Index new_size = 0;

    for (Index i = 0; i < size; i++)
        if (!isnan(vector(i)))
            new_size++;

    std::vector<Index> sorted_vector;

    for (Index i = 0; i < size; i++)
        if (!isnan(vector(i))) {
            sorted_vector.push_back(vector(i));
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

Tensor<type, 1> quartiles(const Tensor<type, 1>& vector)
{
    const Index size = vector.dimension(0);

    // Fix missing values

    Index new_size = 0;

    for (Index i = 0; i < size; i++)
        if (!isnan(vector(i)))
            new_size++;

    std::vector<type> sorted_vector;

    for (Index i = 0; i < size; i++)
        if (!isnan(vector(i))) {
            sorted_vector.push_back(vector(i));
        }
    std::sort(sorted_vector.begin(), sorted_vector.end());
    
    // Calculate quartiles

    std::vector<type> first_sorted_vector;
    std::vector<type> last_sorted_vector;

    if (new_size % 2 == 0)
    {
        for (Index i = 0; i < new_size / 2; i++)
        {
            first_sorted_vector.push_back(sorted_vector[i]);
            last_sorted_vector.push_back(sorted_vector[i+new_size/2]);
        }
    }
    else
    {
        for (Index i = 0; i < new_size / 2; i++)
        {
            first_sorted_vector.push_back(sorted_vector[i]);
            last_sorted_vector.push_back(sorted_vector[i + new_size / 2+1]);
        }
    }

    Tensor<type, 1> quartiles(3);

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
        quartiles(2) = (sorted_vector[0] + sorted_vector[1]) / type(3/4);
 
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

        Index first_vector_size = first_sorted_vector.size();
        Index last_vector_size = first_sorted_vector.size();
        if (first_vector_size % 2 == 0 && last_vector_size % 2 == 0)
        {
            median_index = Index(first_vector_size / 2);

            quartiles(0) = (first_sorted_vector[median_index - 1] + first_sorted_vector[median_index]) / type(2.0);
            quartiles(2) = (last_sorted_vector[median_index - 1] + last_sorted_vector[median_index]) / type(2.0);

        }
        else if (first_vector_size % 2 == 0 && last_vector_size % 2 != 0)
        {
            Index median_index_first = Index(first_vector_size / 2);

            quartiles(0) = (first_sorted_vector[median_index_first - 1] + first_sorted_vector[median_index_first]) / type(2.0);

            Index median_index_last = Index(last_vector_size / 2);

            quartiles(2) = last_sorted_vector[median_index_last];
        }
        else if (first_vector_size % 2 != 0 && last_vector_size % 2 == 0)
        {
            Index median_index_first = Index(first_vector_size / 2);

            quartiles(0) = first_sorted_vector[median_index_first];

            Index median_index_last = Index(last_vector_size / 2);

            quartiles(2) = (last_sorted_vector[median_index_last - 1] + last_sorted_vector[median_index_last]) / type(2.0);

        }
        else {
            median_index = Index(first_vector_size / 2);

            quartiles(0) = first_sorted_vector[median_index];
            quartiles(2) = last_sorted_vector[median_index];
        }
    }
    return quartiles;
}

Tensor<type, 1> quartiles(const Tensor<type, 1>& data, const vector<Index>& indices)
{
    const Index indices_size = indices.size();

    // Fix missing values

    Index index;
    Index new_size = 0;

    for(Index i = 0; i < indices_size; i++)
        if(!isnan(data(indices[i]))) 
            new_size++;

    Tensor<type, 1> sorted_vector(new_size);

    Index sorted_index = 0;

    for(Index i = 0; i < indices_size; i++)
    {
        index = indices[i];

        if(!isnan(data(index)))
            sorted_vector(sorted_index++) = data(index);
    }

    sort(sorted_vector.data(), sorted_vector.data() + sorted_vector.size(), less<type>());

    // Calculate quartiles

    Tensor<type, 1> first_sorted_vector(new_size/2);
    Tensor<type, 1> last_sorted_vector(new_size/2);

    for(Index i = 0; i < new_size/2 ; i++)
        first_sorted_vector(i) = sorted_vector(i);

    for(Index i = 0; i < new_size/2; i++)
        last_sorted_vector(i) = sorted_vector(i + new_size - new_size/2);

    Tensor<type, 1> quartiles(3);

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


BoxPlot box_plot(const Tensor<type, 1>& vector)
{
    BoxPlot box_plot;

    if(vector.dimension(0) == 0) 
        return box_plot;
 
    //const Tensor<type, 1> quartiles = opennn::quartiles(vector);

    box_plot.minimum = minimum(vector);
    //box_plot.first_quartile = quartiles(0);
    //box_plot.median = quartiles(1);
    //box_plot.third_quartile = quartiles(2);
    box_plot.maximum = maximum(vector);

    return box_plot;
}


BoxPlot box_plot(const Tensor<type, 1>& data, const vector<Index>& indices)
{
    BoxPlot box_plot;

    if(data.dimension(0) == 0 || indices.size() == 0) 
        return box_plot;

    const Tensor<type, 1> quartiles = opennn::quartiles(data, indices);

    box_plot.minimum = minimum(data, indices);
    box_plot.first_quartile = quartiles(0);
    box_plot.median = quartiles(1);
    box_plot.third_quartile = quartiles(2);
    box_plot.maximum = maximum(data, indices);

    return box_plot;
}


Histogram histogram(const Tensor<type, 1>& vector, const Index& bins_number) 
{
    /*
    const Index size = vector.dimension(0);

    if (size == 0 || bins_number <= 0) {
        throw invalid_argument("Vector size or number of bins must be positive.");

    std::vector<type> value_keys;
    std::vector<Index> value_frequencies;

    for (Index i = 0; i < size; i++) {
        const type val = vector(i);
        if (isnan(val)) continue;

        auto it = find(value_keys.begin(), value_keys.end(), val);

        if (it != value_keys.end()) {
            Index index = distance(value_keys.begin(), it);
            value_frequencies[index]++;
        }
        else {
            value_keys.push_back(val);
            value_frequencies.push_back(1);
        }
    }

    std::vector<pair<type, Index>> sorted_values;
    for (size_t i = 0; i < value_keys.size(); i++) {
        sorted_values.emplace_back(value_keys[i], value_frequencies[i]);
    }

    sort(sorted_values.begin(), sorted_values.end());

    std::vector<type> unique_values;
    std::vector<Index> unique_frequencies;
    for (const auto& entry : sorted_values) {
        unique_values.push_back(entry.first);
        unique_frequencies.push_back(entry.second);
    }

    const Index unique_values_number = static_cast<Index>(unique_values.size());
    //const Index effective_bins = std::min(bins_number, unique_values_number);
    const Index effective_bins = bins_number;

    Tensor<type, 1> minimums(effective_bins);
    minimums.setZero();

    Tensor<type, 1> maximums(effective_bins);
    maximums.setZero();

    Tensor<type, 1> centers(effective_bins);
    centers.setZero();

    Tensor<Index, 1> frequencies(effective_bins);
    frequencies.setZero();

    Index values_per_bin = unique_values_number / effective_bins;
    Index remaining_values = unique_values_number % effective_bins;

    Index current_bin = 0;
    Index current_value_index = 0;

    while (current_bin < effective_bins) {
        Index bin_size = values_per_bin;
        if (remaining_values > 0) {
            bin_size++;
            remaining_values--;
        }

        std::vector<type> bin_values(unique_values.begin() + current_value_index,
            unique_values.begin() + current_value_index + bin_size);
        std::vector<Index> bin_freqs(unique_frequencies.begin() + current_value_index,
            unique_frequencies.begin() + current_value_index + bin_size);

        minimums(current_bin) = *std::min_element(bin_values.begin(), bin_values.end());
        maximums(current_bin) = *std::max_element(bin_values.begin(), bin_values.end());
        centers(current_bin) = (minimums(current_bin) + maximums(current_bin)) / type(2.0);
        frequencies(current_bin) = std::accumulate(bin_freqs.begin(), bin_freqs.end(), Index(0));

        current_value_index += bin_size;
        current_bin++;
    }

    Histogram hist(bins_number);
    hist.centers = centers;
    hist.minimums = minimums;
    hist.maximums = maximums;
    hist.frequencies = frequencies;
    */
    return 0;
}

Histogram histogram(const Tensor<type, 1>& vector, const Index bins_number)
{
    /*
    const Index size = vector.dimension(0);

    Tensor<type, 1> minimums(bins_number);
    Tensor<type, 1> maximums(bins_number);

    Tensor<type, 1> centers(bins_number);
    Tensor<Index, 1> frequencies(bins_number);
    frequencies.setZero();

    std::vector<type> unique_values;
    unique_values.reserve(min<Index>(size, bins_number));
    unique_values.push_back(vector(0));
    
    for(Index i = 1; i < size; i++)
    {
        const type value = vector(i);

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

        centers = unique_values;
        minimums = unique_values;
        maximums = unique_values;

        frequencies.resize(unique_values_number);
        frequencies.setZero();

        for(Index i = 0; i < size; i++)
        {
            if(isnan(vector(i))) continue;

            for(Index j = 0; j < unique_values_number; j++)
            {
                if(vector(i) - centers(j) < NUMERIC_LIMITS_MIN)
                {
                    frequencies(j)++;
                    break;
                }
            }
        }
    }
    else
    {
        const type min = minimum(vector);
        const type max = maximum(vector);

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

        const Index size = vector.dimension(0);

        for(Index i = 0; i < size; i++)
        {
            if(isnan(vector(i))) continue;

            for(Index j = 0; j < bins_number - 1; j++)
            {
                if(vector(i) >= minimums(j) && vector(i) < maximums(j))
                {
                    frequencies(j)++;
                    break;
                }
            }

            if(vector(i) >= minimums(bins_number - 1))
                frequencies(bins_number - 1)++;
        }
    }

    Histogram histogram(bins_number);
    histogram.centers = centers;
    histogram.minimums = minimums;
    histogram.maximums = maximums;
    histogram.frequencies = frequencies;

    */
    
    Histogram histogram(bins_number);
    return histogram;
}


Histogram histogram_centered(const Tensor<type, 1>& vector, const type& center, const Index& bins_number)
{
    const Index bin_center = (bins_number % 2 == 0) 
        ? Index(type(bins_number) / type(2.0)) 
        : Index(type(bins_number) / type(2.0) + type(0.5));

    Tensor<type, 1> minimums(bins_number);
    Tensor<type, 1> maximums(bins_number);

    Tensor<type, 1> centers(bins_number);
    Tensor<Index, 1> frequencies(bins_number);
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

    const Index size = vector.dimension(0);

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

Histogram histogram(const Tensor<bool, 1>& v)
{
    Tensor<type, 1> minimums(2);
    minimums.setZero();

    Tensor<type, 1> maximums(2);
    maximums.setConstant(type(1));

    Tensor<type, 1> centers(2);
    centers.setValues({type(0), type(1)});

    Tensor<Index, 1> frequencies(2);
    frequencies.setZero();

    // Calculate bins frequency

    const Index size = v.dimension(0);

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

/*
Tensor<Index, 1> total_frequencies(const Tensor<Histogram, 1>& histograms)
{
    const Index histograms_number = histograms.size();

    Tensor<Index, 1> total_frequencies(histograms_number);

    for(Index i = 0; i < histograms_number; i++)
        total_frequencies(i) = histograms(i).frequencies(i);

    return total_frequencies;
}
*/

vector<Histogram> histograms(const Tensor<type, 2>& matrix, const Index& bins_number)
{
    const Index columns_number = matrix.dimension(1);

    vector<Histogram> histograms(columns_number);
/*
    for(Index i = 0; i < columns_number; i++)
        histograms(i) = histogram(tensor_map(matrix, i), bins_number);
*/
    return histograms;
}

Descriptives vector_descriptives(const Tensor<type, 1>& x)
{
    Descriptives my_descriptives;

    const Index size = x.size();

    if (size <= 0)
    {
        std::cerr << "Error: vector_descriptives() recibi� un tensor vac�o." << std::endl;
        return my_descriptives;
    }

    const Tensor<type, 0> minimum = x.minimum();
    const Tensor<type, 0> maximum = x.maximum();

    long double sum = 0.0;
    long double squared_sum = 0;
    Index count = 0;

    for (Index i = 0; i < size; i++)
    {
        if (isnan(x(i))) continue;

        if (i >= 0 && i < x.size()) {
            sum += x(i);
        }
        else {
            std::cerr << "�ndice fuera de rango: " << i << std::endl;
        }
        squared_sum += double(x(i)) * double(x(i));
        count++;
    }

    type mean = 0;
    if (count > 0)
    {
        mean = type(sum / count);
    }

    type standard_deviation = 0;
    if (count > 1)
    {
        const type numerator = type(squared_sum - sum * sum / count);
        const type denominator = type(count - 1);
        standard_deviation = sqrt(numerator / denominator);
    }

    my_descriptives.set(minimum(0), maximum(0), mean, standard_deviation);

    return my_descriptives;
}

vector<Descriptives> descriptives(const Tensor<type, 2>& matrix)
{
    const Index columns_number = matrix.dimension(1);

    vector<Descriptives> descriptives(columns_number);

    for (Index i = 0; i < columns_number; i++) {
        // Acceder directamente a la columna sin usar TensorMap
        const Tensor<type, 1> column = matrix.chip(i, 1); // Chip devuelve un tensor con la columna

        if (i >= 0 && i < static_cast<Index>(descriptives.size())) {
            descriptives[i] = vector_descriptives(column);
        }
        else {
            std::cerr << "�ndice fuera de rango: " << i << std::endl;
        }

        std::cout << "Tama�o del vector descriptives: " << descriptives.size() << std::endl;
    }

    return descriptives;
}


vector<Descriptives> descriptives(const Tensor<type, 2>& matrix,
                                  const vector<Index>& row_indices,
                                  const vector<Index>& column_indices)
{
    const Index row_indices_size = row_indices.size();
    const Index column_indices_size = column_indices.size();

    vector<Descriptives> descriptives(column_indices_size);

    Index row_index;
    Index column_index;

    Tensor<type, 1> minimums(column_indices_size);
    minimums.setConstant(numeric_limits<type>::max());

    Tensor<type, 1> maximums(column_indices_size);
    maximums.setConstant(NUMERIC_LIMITS_MIN);

    Tensor<double, 1> sums(column_indices_size);
    Tensor<double, 1> squared_sums(column_indices_size);
    Tensor<Index, 1> count(column_indices_size);

    sums.setZero();
    squared_sums.setZero();
    count.setZero();

    // @todo optimize this loop
    for(Index i = 0; i < row_indices_size; i++)
    {
        row_index = row_indices[i];

        //        #pragma omp parallel for private(column_index)

        for(Index j = 0; j < column_indices_size; j++)
        {
            column_index = column_indices[j];

            const type value = matrix(row_index, column_index);

            if(isnan(value)) continue;

            minimums(j) = min(minimums(j), value);
            maximums(j) = max(maximums(j), value);

            sums(j) += double(value);
            squared_sums(j) += double(value)*double(value);
            count(j)++;
        }
    }

    const Tensor<double, 1> mean = sums/count.cast<double>();

    Tensor<double, 1> standard_deviation(column_indices_size);

    if(row_indices_size > 1)
    {
        //        #pragma omp parallel for

        for(Index i = 0; i < column_indices_size; i++)
        {
            const double variance = squared_sums(i)/double(count(i)-1)
                    - (sums(i)/double(count(i)))*(sums(i)/double(count(i)))*double(count(i))/double(count(i)-1);

            standard_deviation(i) = sqrt(variance);
        }
    }
    else
    {
        for(Index i = 0; i < column_indices_size; i++)
            standard_deviation(i) = type(0);
    }

    for(Index i = 0; i < column_indices_size; i++)
        descriptives[i].set(type(minimums(i)), 
                            type(maximums(i)), 
                            type(mean(i)), 
                            type(standard_deviation(i)));

    return descriptives;
}


Tensor<type, 1> column_minimums(const Tensor<type, 2>& matrix,
                                const vector<Index>& row_indices,
                                const vector<Index>& column_indices)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

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

    Tensor<type, 1> minimums(column_indices_size);

    Index row_index;
    Index column_index;

    for(Index j = 0; j < column_indices_size; j++)
    {
        column_index = used_column_indices[j];

        Tensor<type, 1> column(row_indices_size);

        for(Index i = 0; i < row_indices_size; i++)
        {
            row_index = used_row_indices[i];

            column(i) = matrix(row_index,column_index);
        }

        minimums(j) = minimum(column);
    }

    return minimums;
}


Tensor<type, 1> column_maximums(const Tensor<type, 2>& matrix, const vector<Index>& column_indices)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    vector<Index> used_column_indices;

    if(column_indices.size() == 0)
        used_column_indices.resize(columns_number);
    else
        used_column_indices = column_indices;

    const Index column_indices_size = used_column_indices.size();

    Tensor<type, 1> maximums(column_indices_size);

    Index column_index;
    Tensor<type, 1> column(rows_number);

    for(Index i = 0; i < column_indices_size; i++)
    {
        column_index = used_column_indices[i];

        column = matrix.chip(column_index,1);

        maximums(i) = maximum(column);
    }

    return maximums;
}


type range(const Tensor<type, 1>& vector)
{
    const type min = minimum(vector);
    const type max = maximum(vector);

    return abs(max - min);
}


Tensor<type, 1> mean(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    // Mean

    Tensor<type, 1> mean(columns_number);
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


Tensor<type, 1> mean(const Tensor<type, 2>& matrix, const Tensor<Index, 1>& column_indices)
{
    const Index rows_number = matrix.dimension(0);

    const Index column_indices_size = column_indices.size();

    Index column_index;

    // Mean

    Tensor<type, 1> mean(column_indices_size);
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


Tensor<type, 1> mean(const Tensor<type, 2>& matrix, const vector<Index>& row_indices, const vector<Index>& column_indices)
{
    const Index row_indices_size = row_indices.size();
    const Index column_indices_size = column_indices.size();

    if(row_indices_size == 0 && column_indices_size == 0) 
        return Tensor<type, 1>();

    Index row_index;
    Index column_index;

    Index count = 0;

    // Mean

    Tensor<type, 1> mean(column_indices_size); // @todo check test crash here
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


type mean(const Tensor<type, 2>& matrix, const Index& column_index)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

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


Tensor<type, 1> median(const Tensor<type, 2>& matrix)
{
    const Index columns_number = matrix.dimension(1);

    // median

    Tensor<type, 1> median(columns_number);

    for(Index j = 0; j < columns_number; j++)
    {
        Tensor<type, 1> column(matrix.chip(j,1));
        Tensor<type, 1> sorted_column;
        Index median_index;
        Index rows_number = 0;

        for(Index i = 0; i < column.size(); i++)
        {
            if(!isnan(column(i)))
            {
                push_back(sorted_column, column(i));
                rows_number++;
            }
        }

        sort(sorted_column.data(), sorted_column.data() + sorted_column.size(), less<type>());

        median_index = Index(rows_number/2);

        median(j) = (rows_number % 2 == 0)
            ? (sorted_column[median_index - 1] + sorted_column[median_index]) / type(2)
            : sorted_column[median_index - 1 / 2];
    }

    return median;
}


type median(const Tensor<type, 2>& matrix, const Index& column_index)
{
    // median

    type median = type(0);

    std::vector<type> sorted_column;

    Index rows_number = 0;

    for (Index i = 0; i < matrix.dimension(0); i++) {
        if (!isnan(matrix(i,column_index)))
        {
            sorted_column.push_back(matrix(i, column_index));
            rows_number++;
        }
    }

    Index median_index;

    if (rows_number % 2 == 0)
    {
        median_index = type(rows_number / 2);

        type median = (sorted_column[median_index - 1] + sorted_column[median_index]) / type(2.0);

        return median;
    }
    else
    {
        median_index = Index(rows_number / 2);

        return sorted_column[median_index];
    }

    return median;
}


Tensor<type, 1> median(const Tensor<type, 2>& matrix, const Tensor<Index, 1>& column_indices)
{
    const Index rows_number = matrix.dimension(0);

    const Index column_indices_size = column_indices.size();

    Index column_index;

    // median

    Tensor<type, 1> median(column_indices_size);

    for(Index j = 0; j < column_indices_size; j++)
    {
        column_index = column_indices(j);

        Tensor<type, 1> sorted_column(0);

        const Tensor<type, 1> column = matrix.chip(column_index, 1);

        for(Index i = 0; i < column.size(); i++)
            if(!isnan(column(i)))
                push_back(sorted_column,column(i));

        sort(sorted_column.data(), sorted_column.data() + sorted_column.size(), less<type>());

        median(j) = (rows_number % 2 == 0)
            ? (sorted_column[sorted_column.size() * 2 / 4] + sorted_column[sorted_column.size() * 2 / 4 + 1]) / type(2)
            : sorted_column[sorted_column.size() * 2 / 4];
    }

    return median;
}


Tensor<type, 1> median(const Tensor<type, 2>& matrix, 
                       const vector<Index>& row_indices,  
                       const vector<Index>& column_indices)
{
    const Index row_indices_size = row_indices.size();
    const Index column_indices_size = column_indices.size();

    Index column_index;

    // median

    Tensor<type, 1> median(column_indices_size);

    for(Index j = 0; j < column_indices_size; j++)
    {
        column_index = column_indices[j];

        Tensor<type, 1> sorted_column;

        for(Index k = 0; k < row_indices_size; k++)
        {
            const Index row_index = row_indices[k];

            if(!isnan(matrix(row_index, column_index)))
                push_back(sorted_column, matrix(row_index, column_index));
        }

        sort(sorted_column.data(), sorted_column.data() + sorted_column.size(), less<type>());

        const Index sorted_list_size = sorted_column.size();

        median(j) = (sorted_list_size % 2 == 0)
            ? (sorted_column[sorted_list_size * 2 / 4] + sorted_column[sorted_list_size * 2 / 4 + 1]) / type(2)
            : sorted_column[sorted_list_size * 2 / 4];
    }

    return median;
}


Index minimal_index(const Tensor<type, 1>& vector)
{
    const Index size = vector.dimension(0);

    if(size == 0) return Index();

    Index minimal_index = 0;
    type minimum = vector[0];

    for(Index i = 1; i < size; i++)
    {
        if(vector(i) < minimum)
        {
            minimal_index = i;
            minimum = vector(i);
        }
    }

    return minimal_index;
}


Index maximal_index(const Tensor<type, 1>& vector)
{
    const Index size = vector.dimension(0);

    if(size == 0) return Index();

    Index maximal_index = 0;
    type maximum = vector[0];

    for(Index i = 1; i < size; i++)
    {
        if(vector(i) > maximum)
        {
            maximal_index = i;
            maximum = vector(i);
        }
    }

    return maximal_index;
}

Tensor<Index, 1> minimal_indices(const Tensor<type, 1>& vector, const Index& number)
{
    std::vector<type> vector_(vector.dimension(0));
    for (Index i = 0; i < vector.dimension(0); i++) {
        vector_[i] = vector(i);
    }

    const Index size = vector_.size();
    Tensor<Index, 1> minimal_indices(number);

    Index val_max=0;
    for (Index i = 0; i < size; i++) {
        if (vector(i) > val_max) {
            val_max = vector(i);
        }
    }

    for (Index j = 0; j < number; j++)
    {
        Index minimal_index = 0;
        type minimum = vector_[0];

        for (Index i = 0; i < size; i++)
        {
            if (vector_[i] < minimum)
            {
                minimal_index = i;
                minimum = vector_[i];
            }
        }

        vector_[minimal_index] = val_max + type(1);
        minimal_indices(j) = minimal_index;
    }

    return minimal_indices;
}

Tensor<Index, 1> maximal_indices(const Tensor<type, 1>& vector, const Index& number)
{
    std::vector<type> vector_(vector.dimension(0));
    for (Index i = 0; i < vector.dimension(0); i++) {
        vector_[i] = vector(i);
    }

    const Index size = vector_.size();

    Index val_min = 0;
    for (Index i = 0; i < size; i++) {
        if (vector(i) < val_min) {
            val_min = vector(i);
        }
    }

    Tensor<Index, 1> maximal_indices(number);

    for (Index j = 0; j < number; j++)
    {
        Index maximal_index = 0;
        type maximal = vector_[0];

        for (Index i = 0; i < size; i++)
        {
            if (vector_[i] > maximal)
            {
                maximal_index = i;
                maximal = vector_[i];
            }
        }

        vector_[maximal_index] = val_min - type(1);
        maximal_indices(j) = maximal_index;
    }

    return maximal_indices;
}

Tensor<Index, 1> minimal_indices(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    type minimum = matrix(0,0);

    Tensor<Index, 1> minimal_indices(2);
    minimal_indices.setZero();

    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < columns_number; j++)
        {
            if(!isnan(matrix(i, j)) && matrix(i, j) < minimum)
            {
                minimum = matrix(i, j);
                minimal_indices(0) = i;
                minimal_indices(1) = j;
            }
        }
    }

    return minimal_indices;
}


Tensor<Index, 1> maximal_indices(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    type maximum = matrix(0,0);

    Tensor<Index, 1> maximal_indices(2);
    maximal_indices.setZero();

    for(Index i = 0; i < rows_number; i++)
    {
        for(Index j = 0; j < columns_number; j++)
        {
            if(!isnan(matrix(i, j)) && matrix(i, j) > maximum)
            {
                maximum = matrix(i, j);
                maximal_indices(0) = i;
                maximal_indices(1) = j;
            }
        }
    }

    return maximal_indices;
}

Tensor<type, 1> percentiles(const Tensor<type, 1>& vector)
{
    const Index size = vector.dimension(0);

    Index new_size = 0;

    for (Index i = 0; i < size; i++)
        if (!isnan(vector(i)))
            new_size++;

    if (new_size == 0)
    {
        Tensor<type, 1> nan(1);
        nan.setValues({ type(NAN) });
        return nan;
    }

    Index index = 0;
    
    std::vector<type> new_vector(new_size);

    for (Index i = 0; i < size; i++)
        if (!isnan(vector(i)))
            new_vector[index++] = vector(i);

    std::vector<type> sorted_vector(new_size);
    sorted_vector = new_vector;
    sort(sorted_vector.begin(), sorted_vector.end());

    Tensor<type, 1> percentiles(10);

    for (Index i = 0; i < 9; i++)
        percentiles[i] = (new_size * (i + 1) % 10 == 0)
        ? (sorted_vector[new_size * (i + 1) / 10 - 1] + sorted_vector[new_size * (i + 1) / 10]) / type(2.0)
        : type(sorted_vector[new_size * (i + 1) / 10]);

    Index val_max = 0;
    for (Index i = 0; i < new_size; i++) {
        if (new_vector[i] > val_max) {
            val_max = new_vector[i];
        }
    }
    percentiles[9] = val_max;

    return percentiles;
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
