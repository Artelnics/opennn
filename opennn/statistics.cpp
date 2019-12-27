//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A T I S T I C S   S O U R C E
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "statistics.h"

namespace OpenNN {

Descriptives::Descriptives()
{
  name = "Descriptives";
  minimum = -1.0;
  maximum = 1.0;
  mean = 0.0;
  standard_deviation = 1.0;
}


/// Values constructor.

Descriptives::Descriptives(const double &new_minimum, const double &new_maximum,
                           const double &new_mean, const double &new_standard_deviation)
{
  minimum = new_minimum;
  maximum = new_maximum;
  mean = new_mean;
  standard_deviation = new_standard_deviation;
}


/// Destructor.

Descriptives::~Descriptives()
{}


/// Sets a new minimum value in the descriptives structure.
/// @param new_minimum Minimum value.

void Descriptives::set_minimum(const double &new_minimum) {
  minimum = new_minimum;
}


/// Sets a new maximum value in the descriptives structure.
/// @param new_maximum Maximum value.

void Descriptives::set_maximum(const double &new_maximum) {
  maximum = new_maximum;
}


/// Sets a new mean value in the descriptives structure.
/// @param new_mean Mean value.

void Descriptives::set_mean(const double &new_mean) {
  mean = new_mean;
}


/// Sets a new standard deviation value in the descriptives structure.
/// @param new_standard_deviation Standard deviation value.

void Descriptives::set_standard_deviation(const double &new_standard_deviation) {
  standard_deviation = new_standard_deviation;
}


/// Returns all the statistical parameters contained in a single vector.
/// The size of that vector is seven.
/// The elements correspond to the minimum, maximum, mean and standard deviation
/// values respectively.

Vector<double> Descriptives::to_vector() const
{
  Vector<double> statistics_vector(4);
  statistics_vector[0] = minimum;
  statistics_vector[1] = maximum;
  statistics_vector[2] = mean;
  statistics_vector[3] = standard_deviation;

  return statistics_vector;
}


/// Returns true if the minimum value is -1 and the maximum value is +1,
/// and false otherwise.

bool Descriptives::has_minimum_minus_one_maximum_one() {
  if(-1.000001 < minimum && minimum < -0.999999 && 0.999999 < maximum &&
      maximum < 1.000001) {
    return true;
  } else {
    return false;
  }
}


/// Returns true if the mean value is 0 and the standard deviation value is 1,
/// and false otherwise.

bool Descriptives::has_mean_zero_standard_deviation_one() {
  if(-0.000001 < mean && mean < 0.000001 && 0.999999 < standard_deviation &&
      standard_deviation < 1.000001) {
    return true;
  } else {
    return false;
  }
}


/// Print the tittle of descriptives structure

void Descriptives::print(const string& title) const
{
  cout << title << endl
       << "Minimum: " << minimum << endl
       << "Maximum: " << maximum << endl
       << "Mean: " << mean << endl
       << "Standard deviation: " << standard_deviation << endl;
}


BoxPlot::BoxPlot(const double & new_minimum, const double & new_first_cuartile, const double & new_median, const double & new_third_quartile, const double & new_maximum)
{
    minimum = new_minimum;
    first_quartile = new_first_cuartile;
    median = new_median;
    third_quartile = new_third_quartile;
    maximum = new_maximum;
}


/// Saves to a file the minimum, maximum, mean and standard deviation
/// of the descriptives structure.
/// @param file_name Name of descriptives data file.

void Descriptives::save(const string &file_name) const {
  ofstream file(file_name.c_str());

  if(!file.is_open()) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "void save(const string&) const method.\n"
           << "Cannot open descriptives data file.\n";

    throw logic_error(buffer.str());
  }

  // Write file

  file << "Minimum: " << minimum << endl
       << "Maximum: " << maximum << endl
       << "Mean: " << mean << endl
       << "Standard deviation: " << standard_deviation << endl;

  // Close file

  file.close();
}


Histogram::Histogram() {}


/// Destructor.

Histogram::~Histogram() {}


/// Bins number constructor.
/// @param bins_number Number of bins in the histogram.

Histogram::Histogram(const size_t &bins_number)
{
  centers.resize(bins_number);
  frequencies.resize(bins_number);
}


/// Values constructor.
/// @param new_centers Center values for the bins.
/// @param new_frequencies Number of variates in each bin.

Histogram::Histogram(const Vector<double>&new_centers,
                        const Vector<size_t>&new_frequencies) {
  centers = new_centers;
  frequencies = new_frequencies;
}


/// Returns the number of bins in the histogram.

size_t Histogram::get_bins_number() const {
  return centers.size();
}


/// Returns the number of bins with zero variates.

size_t Histogram::count_empty_bins() const {
  return frequencies.count_equal_to(0);
}


/// Returns the number of variates in the less populated bin.

size_t Histogram::calculate_minimum_frequency() const
{ 
 return minimum(frequencies);
}


/// Returns the number of variates in the most populated bin.

size_t Histogram::calculate_maximum_frequency() const
{
  return maximum(frequencies);

}


/// Retuns the index of the most populated bin.

size_t Histogram::calculate_most_populated_bin() const
{
  return maximal_index(frequencies.to_double_vector());
}


/// Returns a vector with the centers of the less populated bins.

Vector<double> Histogram::calculate_minimal_centers() const
{
  const size_t minimum_frequency = calculate_minimum_frequency();
  const Vector<size_t> minimal_indices = frequencies.get_indices_equal_to(minimum_frequency);

  return(centers.get_subvector(minimal_indices));
}


/// Returns a vector with the centers of the most populated bins.

Vector<double> Histogram::calculate_maximal_centers() const {
  const size_t maximum_frequency = calculate_maximum_frequency();

  const Vector<size_t> maximal_indices = frequencies.get_indices_equal_to(maximum_frequency);

  return(centers.get_subvector(maximal_indices));
}


/// Returns the number of the bin to which a given value belongs to.
/// @param value Value for which we want to get the bin.

size_t Histogram::calculate_bin(const double &value) const
{
  const size_t bins_number = get_bins_number();

  const double minimum_center = centers[0];
  const double maximum_center = centers[bins_number - 1];

  const double length = static_cast<double>(maximum_center - minimum_center)/static_cast<double>(bins_number - 1.0);

  double minimum_value = centers[0] - length / 2;
  double maximum_value = minimum_value + length;

  if(value < maximum_value) {
    return 0;
  }

  for(size_t j = 1; j < bins_number - 1; j++) {
    minimum_value = minimum_value + length;
    maximum_value = maximum_value + length;

    if(value >= minimum_value && value < maximum_value) {
      return(j);
    }
  }

  if(value >= maximum_value) {
    return(bins_number - 1);
  } else {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "size_t Histogram::calculate_bin(const double&) const.\n"
           << "Unknown return value.\n";

    throw logic_error(buffer.str());
  }
}


/// Returns the frequency of the bin to which a given value bolongs to.
/// @param value Value for which we want to get the frequency.

size_t Histogram::calculate_frequency(const double &value) const
{
  const size_t bin_number = calculate_bin(value);

  const size_t frequency = frequencies[bin_number];

  return frequency;
}


/// Returns the smallest element of a double vector.
/// @param vector

double minimum(const Vector<double>& vector)
{
    const double min = *min_element(vector.begin(), vector.end());

    return min;
}


/// Returns the smallest element of a size_t vector.
/// @param vector

size_t minimum(const Vector<size_t>& vector)
{
    const size_t min = *min_element(vector.begin(), vector.end());

    return min;
}


time_t minimum(const Vector<time_t>& vector)
{
    const time_t min = *min_element(vector.begin(), vector.end());

    return min;
}


/// Returns the largest element in the vector.
/// @param vector

double maximum(const Vector<double>& vector)
{
    const double max = *max_element(vector.begin(), vector.end());

    return max;
}


size_t maximum(const Vector<size_t>& vector)
{
    const size_t max = *max_element(vector.begin(), vector.end());

    return max;
}


time_t maximum(const Vector<time_t>& vector)
{
    const time_t max = *max_element(vector.begin(), vector.end());

    return max;
}


/// Returns the smallest element in the vector.

double minimum_missing_values(const Vector<double>& vector)
{
  const size_t this_size = vector.size();

  double minimum = numeric_limits<double>::max();

  for(size_t i = 0; i < this_size; i++)
  {
    if(vector[i] < minimum && !::isnan(vector[i]))
    {
      minimum = vector[i];
    }
  }

  return minimum;
}


/// Returns the largest element in the vector.

double maximum_missing_values(const Vector<double>& vector)
{
  const size_t this_size = vector.size();

  double maximum = -numeric_limits<double>::max();

  for(size_t i = 0; i < this_size; i++) {
    if(!::isnan(vector[i]) && vector[i] > maximum) {
      maximum = vector[i];
    }
  }

  return maximum;
}


/// Returns the mean of the elements in the vector.
/// @param vector

double mean(const Vector<double>& vector)
{
  const size_t size = vector.size();

#ifdef __OPENNN_DEBUG__

  if(size == 0)
  {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics class.\n"
           << "double mean(const Vector<double>&) const method.\n"
           << "Size of vector must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  const double sum = vector.calculate_sum();

  const double mean = sum /static_cast<double>(size);

  return mean;
}


/// Returns the mean of the subvector defined by a start and end elements.
/// @param vector
/// @param begin Start element.
/// @param end End element.

double mean(const Vector<double>& vector, const size_t& begin, const size_t& end)
{
  #ifdef __OPENNN_DEBUG__

    if(begin > end) {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "double mean(const Vector<double>& vector, const size_t& begin, const size_t& end) \n"
             << "Begin must be less or equal than end.\n";

      throw logic_error(buffer.str());
    }

  #endif

  if(end == begin) return vector[begin];

  double sum = 0.0;

  for(size_t i = begin; i <= end; i++)
  {
      sum += vector[i];
  }

  return(sum /static_cast<double>(end-begin+1));
}


/// Returns the mean of the elements in the vector.
/// @param vector

double mean_missing_values(const Vector<double>& vector)
{
  const size_t this_size = vector.size();

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double mean_missing_values(const Vector<double>& vector, const size_t& begin, const size_t& end) "
              "const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  double sum = 0;

  size_t count = 0;

  for(size_t i = 0; i < this_size; i++) {
    if(!::isnan(vector[i]))
    {
      sum += vector[i];
      count++;
    }
  }

  const double mean = sum /static_cast<double>(count);

  return mean;
}


/// Returns the variance of the elements in the vector.
/// @param vector

double variance(const Vector<double>& vector)
{
  const size_t this_size = vector.size();

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double variance(const Vector<double>&) const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  if(this_size == 1)
  {
    return(0.0);
  }

  double sum = 0.0;
  double squared_sum = 0.0;

  for(size_t i = 0; i < this_size; i++)
  {
    sum += vector[i];
    squared_sum += vector[i] * vector[i];
  }

  const double numerator = squared_sum -(sum * sum) /static_cast<double>(this_size);
  const double denominator = this_size - 1.0;

  if(denominator == 0.0)
  {
      return 0.0;
  }
  else
  {
      return numerator/denominator;
  }
}


/// Returns the variance of the elements in the vector.
/// @param vector

double variance_missing_values(const Vector<double>& vector)
{
  const size_t this_size = vector.size();

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double variance_missing_values(const Vector<double>& vector) "
              "const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  double sum = 0.0;
  double squared_sum = 0.0;

  size_t count = 0;

  for(size_t i = 0; i < this_size; i++) {
    if(!::isnan(vector[i])) {
      sum += vector[i];
      squared_sum += vector[i] * vector[i];

      count++;
    }
  }

  if(count <= 1) {
    return(0.0);
  }

  const double numerator = squared_sum -(sum * sum) /static_cast<double>(count);
  const double denominator = count - 1.0;

  return numerator/denominator;
}


/// Returns the standard deviation of the elements in the vector.
/// @param vector

double standard_deviation(const Vector<double>& vector)
{
#ifdef __OPENNN_DEBUG__

  const size_t this_size = vector.size();

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double standard_deviation(const Vector<double>&) const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  return(sqrt(variance(vector)));
}


Vector<double> standard_deviation(const Vector<double>& vector, const size_t& period)
{
  const size_t this_size = vector.size();

  Vector<double> standard_deviation(this_size, 0.0);

  double mean_value = 0.0;
  double sum = 0.0;

  for(size_t i = 0; i < this_size; i++)
  {
      const size_t begin = i < period ? 0 : i - period + 1;
      const size_t end = i;

      mean_value = mean(vector, begin,end);

      for(size_t j = begin; j < end+1; j++)
      {
          sum += (vector[j] - mean_value) *(vector[j] - mean_value);
      }

      standard_deviation[i] = sqrt(sum / double(period));

      mean_value = 0.0;
      sum = 0.0;
  }

  standard_deviation[0] = standard_deviation[1];

  return standard_deviation;
}


/// Returns the standard deviation of the elements in the vector.
/// @param vector

double standard_deviation_missing_values(const Vector<double>& vector)
{
#ifdef __OPENNN_DEBUG__

  const size_t this_size = vector.size();

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double standard_deviation_missing_values(const Vector<double>& vector) const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  return sqrt(variance_missing_values(vector));
}


/// Returns the asymmetry of the elements in the vector
/// @param vector

double asymmetry(const Vector<double>& vector)
{
  const size_t this_size = vector.size();

#ifdef __OPENNN_DEBUG__

  if(this_size == 0)
  {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double asymmetry(const Vector<double>& vector) const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  if(this_size == 1)
  {
    return 0.0;
  }

  const double standard_deviation_value = standard_deviation(vector);

  const double mean_value = mean(vector);

  double sum = 0.0;

  for(size_t i = 0; i < this_size; i++)
  {
    sum += (vector[i] - mean_value)*(vector[i] - mean_value)*(vector[i] - mean_value);
  }
  const double numerator = sum /static_cast<double>(this_size);
  const double denominator = standard_deviation_value * standard_deviation_value * standard_deviation_value;

  return numerator/denominator;
}


/// Returns the kurtosis value of the elements in the vector.
/// @param vector

double kurtosis(const Vector<double>& vector)
{
  const size_t this_size = vector.size();

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistic Class.\n"
           << "double kurtosis(const Vector<double>& vector) const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  if(this_size == 1) {
    return 0.0;
  }

  const double standard_deviation_value = standard_deviation(vector);

  const double mean_value = mean(vector);

  double sum = 0.0;

  for(size_t i = 0; i < this_size; i++)
  {
    sum += (vector[i] - mean_value)*(vector[i] - mean_value)*(vector[i] - mean_value)*(vector[i] - mean_value);
  }

  const double numerator = sum/static_cast<double>(this_size);
  const double denominator = standard_deviation_value*standard_deviation_value*standard_deviation_value*standard_deviation_value;

  return numerator/denominator - 3.0;
}


/// Returns the asymmetry of the elements in the vector.
/// @param vector


double asymmetry_missing_values(const Vector<double>& vector)
{
  const size_t this_size = vector.size();
#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double asymmetry_missing_values(const Vector<double>& vector) const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  if(this_size == 1) {
    return 0.0;
  }

  const double standard_deviation_value = standard_deviation_missing_values(vector);

  const double mean_value = mean_missing_values(vector);

  double sum = 0.0;

  for(size_t i = 0; i < this_size; i++)
  {
    if(!::isnan(vector[i]))
    {
      sum += (vector[i] - mean_value) *(vector[i] - mean_value) *(vector[i] - mean_value);
    }
  }

  const double numerator = sum /vector.count_not_NAN();
  const double denominator = standard_deviation_value * standard_deviation_value * standard_deviation_value;

  return numerator/denominator;
}


/// Returns the kurtosis of the elements in the vector.
/// @param vector


double kurtosis_missing_values(const Vector<double>& vector)
{
  const size_t this_size = vector.size();
#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double kurtosis_missing_values(const Vector<double>& vector) const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  if(this_size == 1)
  {
    return 0.0;
  }

  const double standard_deviation_value = standard_deviation_missing_values(vector);

  const double mean_value = mean_missing_values(vector);

  double sum = 0.0;

  for(size_t i = 0; i < this_size; i++)
  {
      if(!::isnan(vector[i]))
    {
      sum += (vector[i] - mean_value)*(vector[i] - mean_value)*(vector[i] - mean_value)*(vector[i] - mean_value);
    }
  }

  const double numerator = sum /vector.count_not_NAN();
  const double denominator = standard_deviation_value*standard_deviation_value*standard_deviation_value*standard_deviation_value;

  return numerator/denominator - 3.0;
}


/// Returns the median of the elements in the vector

double median(const Vector<double>& vector)
{
  const size_t this_size = vector.size();

  Vector<double> sorted_vector(vector);

  sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

  size_t median_index;

  if(this_size % 2 == 0) {
    median_index = static_cast<size_t>(this_size / 2);

    return (sorted_vector[median_index-1] + sorted_vector[median_index]) / 2.0;
  } else {
    median_index = static_cast<size_t>(this_size / 2);

    return sorted_vector[median_index];
  }
}


/// Returns the quarters of the elements in the vector.

Vector<double> quartiles(const Vector<double>& vector)
{
  const size_t this_size = vector.size();

  Vector<double> sorted_vector(vector);

  sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

  Vector<double> quartiles(3);

  if(this_size == 1)
  {
      quartiles[0] = sorted_vector[0];
      quartiles[1] = sorted_vector[0];
      quartiles[2] = sorted_vector[0];
  }
  else if(this_size == 2)
  {
      quartiles[0] = (sorted_vector[0]+sorted_vector[1])/4;
      quartiles[1] = (sorted_vector[0]+sorted_vector[1])/2;
      quartiles[2] = (sorted_vector[0]+sorted_vector[1])*3/4;
  }
  else if(this_size == 3)
  {
      quartiles[0] = (sorted_vector[0]+sorted_vector[1])/2;
      quartiles[1] = sorted_vector[1];
      quartiles[2] = (sorted_vector[2]+sorted_vector[1])/2;
  }
  else if(this_size % 2 == 0)
  {
      quartiles[0] = median(sorted_vector.get_first(this_size/2));
      quartiles[1] = median(sorted_vector);
      quartiles[2] = median(sorted_vector.get_last(this_size/2));

  }
  else
  {
      quartiles[0] = sorted_vector[this_size/4];
      quartiles[1] = sorted_vector[this_size/2];
      quartiles[2] = sorted_vector[this_size*3/4];
  }

  return(quartiles);
}


/// Returns the quartiles of the elements in the vector when there are missing values.

Vector<double> quartiles_missing_values(const Vector<double>& vector)
{
    const size_t this_size = vector.size();

    const size_t new_size = vector.count_not_NAN();

    Vector<double> new_vector(new_size);

    size_t index = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        if(!isnan(vector[i]))
        {
             new_vector[index] = vector[i];

             index++;
        }
    }

    return quartiles(new_vector);
}


/// Returns the box and whispers for a vector.

BoxPlot box_plot(const Vector<double>& vector)
{
    BoxPlot boxplot;

    if(vector.empty()) return boxplot;

    const Vector<double> quartiles = OpenNN::quartiles(vector);

    boxplot.minimum = minimum(vector);
    boxplot.first_quartile = quartiles[0];
    boxplot.median = quartiles[1];
    boxplot.third_quartile = quartiles[2];
    boxplot.maximum = maximum(vector);

    return boxplot;
}


/// Returns the box and whispers for a vector when there are missing values.

BoxPlot box_plot_missing_values(const Vector<double>& vector)
{
    BoxPlot boxplot;

    if(vector.empty()) return boxplot;

    const Vector<double> quartiles = OpenNN::quartiles_missing_values(vector);

    boxplot.minimum = minimum_missing_values(vector);
    boxplot.first_quartile = quartiles[0];
    boxplot.median = quartiles[1];
    boxplot.third_quartile = quartiles[2];
    boxplot.maximum = maximum_missing_values(vector);

    return boxplot;
}


/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.
/// @param vector
/// @param bins_number

Histogram histogram(const Vector<double>& vector, const size_t &bins_number)
{
#ifdef __OPENNN_DEBUG__

  if(bins_number < 1) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "Histogram histogram(const Vector<double>&, "
              "const size_t&) const method.\n"
           << "Number of bins is less than one.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<double> minimums(bins_number);
  Vector<double> maximums(bins_number);

  Vector<double> centers(bins_number);
  Vector<size_t> frequencies(bins_number, 0);

  const double min = minimum(vector);
  const double max = maximum(vector);

  const double length = (max - min) /static_cast<double>(bins_number);

  minimums[0] = min;
  maximums[0] = min + length;
  centers[0] = (maximums[0] + minimums[0]) / 2.0;

  // Calculate bins center

  for(size_t i = 1; i < bins_number; i++)
  {
    minimums[i] = minimums[i - 1] + length;
    maximums[i] = maximums[i - 1] + length;

    centers[i] = (maximums[i] + minimums[i]) / 2.0;
  }

  // Calculate bins frequency

  const size_t this_size = vector.size();

  for(size_t i = 0; i < this_size; i++) {
    for(size_t j = 0; j < bins_number - 1; j++) {
      if(vector[i] >= minimums[j] && vector[i] < maximums[j]) {
        frequencies[j]++;
      }
    }

    if(vector[i] >= minimums[bins_number - 1]) {
      frequencies[bins_number - 1]++;
    }
  }

  Histogram histogram(bins_number);
  histogram.centers = centers;
  histogram.minimums = minimums;
  histogram.maximums = maximums;
  histogram.frequencies = frequencies;

  return histogram;
}


/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.
/// @param vector
/// @param center
/// @param bins_number


Histogram histogram_centered(const Vector<double>& vector, const double& center, const size_t & bins_number)
{
    #ifdef __OPENNN_DEBUG__

      if(bins_number < 1) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Statistics Class.\n"
               << "Histogram histogram_centered(const Vector<double>&, "
                  "const double&, const size_t&) const method.\n"
               << "Number of bins is less than one.\n";

        throw logic_error(buffer.str());
      }

    #endif

      size_t bin_center;

      if(bins_number%2 == 0)
      {
          bin_center = static_cast<size_t>(static_cast<double>(bins_number)/2.0);
      }
      else
      {
          bin_center = static_cast<size_t>(static_cast<double>(bins_number)/2.0+1.0/2.0);
      }

      Vector<double> minimums(bins_number);
      Vector<double> maximums(bins_number);

      Vector<double> centers(bins_number);
      Vector<size_t> frequencies(bins_number, 0);

      const double min = minimum(vector);
      const double max = maximum(vector);

      const double length = (max - min)/static_cast<double>(bins_number);

      minimums[bin_center-1] = center - length;
      maximums[bin_center-1] = center + length;
      centers[bin_center-1] = center;

      // Calculate bins center

      for(size_t i = bin_center; i < bins_number; i++) // Upper centers
      {
        minimums[i] = minimums[i - 1] + length;
        maximums[i] = maximums[i - 1] + length;

        centers[i] = (maximums[i] + minimums[i]) / 2.0;
      }

      for(int i = static_cast<int>(bin_center)-2; i >= 0; i--) // Lower centers
      {
        minimums[static_cast<size_t>(i)] = minimums[static_cast<size_t>(i) + 1] - length;
        maximums[static_cast<size_t>(i)] = maximums[static_cast<size_t>(i) + 1] - length;

        centers[static_cast<size_t>(i)] = (maximums[static_cast<size_t>(i)] + minimums[static_cast<size_t>(i)]) / 2.0;
      }

      // Calculate bins frequency

      const size_t this_size = vector.size();

      for(size_t i = 0; i < this_size; i++) {
        for(size_t j = 0; j < bins_number - 1; j++) {
          if(vector[i] >= minimums[j] && vector[i] < maximums[j]) {
            frequencies[j]++;
          }
        }

        if(vector[i] >= minimums[bins_number - 1]) {
          frequencies[bins_number - 1]++;
        }
      }

      Histogram histogram(bins_number);
      histogram.centers = centers;
      histogram.minimums = minimums;
      histogram.maximums = maximums;
      histogram.frequencies = frequencies;

      return histogram;
}


/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.

Histogram histogram(const Vector<bool>& vector)
{
  const Vector<size_t> minimums(2, 0);
  const Vector<size_t> maximums(2, 1);

  const Vector<size_t> centers({0,1});
  Vector<size_t> frequencies(2, 0);

  // Calculate bins frequency

  const size_t this_size = vector.size();

  for(size_t i = 0; i < this_size; i++)
  {
    for(size_t j = 0; j < 2; j++)
    {
      if(static_cast<size_t>(vector[i]) == minimums[j])
      {
        frequencies[j]++;
      }
    }
  }

  Histogram histogram(2);
  histogram.centers = centers.to_double_vector();
  histogram.minimums = minimums.to_double_vector();
  histogram.maximums = maximums.to_double_vector();
  histogram.frequencies = frequencies;

  return histogram;
}


/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.
/// @param vector
/// @param bins_number

Histogram histogram(const Vector<int>& vector, const size_t& bins_number)
{
    #ifdef __OPENNN_DEBUG__

      if(bins_number < 1) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Statistics Class.\n"
               << "Histogram calculate_histogram_integers(const Vector<int>&, "
                  "const size_t&) const method.\n"
               << "Number of bins is less than one.\n";

        throw logic_error(buffer.str());
      }

    #endif

    Vector<int> centers = vector.get_integer_elements(bins_number);
    const size_t centers_number = centers.size();

    sort(centers.begin(), centers.end(), less<int>());

    Vector<double> minimums(centers_number);
    Vector<double> maximums(centers_number);
    Vector<size_t> frequencies(centers_number);

    for(size_t i = 0; i < centers_number; i++)
    {
        minimums[i] = centers[i];
        maximums[i] = centers[i];
        frequencies[i] = vector.count_equal_to(centers[i]);
    }

    Histogram histogram(centers_number);
    histogram.centers = centers.to_double_vector();
    histogram.minimums = minimums;
    histogram.maximums = maximums;
    histogram.frequencies = frequencies;

    return histogram;
}


/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.
/// @param vector
/// @param bins_number

Histogram histogram_missing_values(const Vector<double>& vector, const size_t &bins_number)
{
#ifdef __OPENNN_DEBUG__

  if(bins_number < 1) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistic Class.\n"
           << "Histogram histogram_missing_values(const Vector<double>&, const Vector<size_t>&) const method.\n"
           << "Number of bins is less than one.\n";

    throw logic_error(buffer.str());
  }

#endif


  const size_t this_size = vector.size();

  const size_t new_size = vector.count_not_NAN();

  Vector<double> new_vector(new_size);

  size_t index = 0;

  for(size_t i = 0; i < this_size; i++)
  {
      if(!::isnan(vector[i]))
      {
           new_vector[index] = vector[i];

           index++;
      }
   }

  return histogram(new_vector, bins_number);
}


/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.
/// @param vector
/// @todo missing values
/*
Histogram histogram_missing_values(const Vector<bool>& vector)
{
  Vector<size_t> minimums(2);
  Vector<size_t> maximums(2);

  Vector<size_t> centers(2);
  Vector<size_t> frequencies(2, 0);

  minimums[0] = 0;
  maximums[0] = 0;
  centers[0] = 0;

  minimums[1] = 1;
  maximums[1] = 1;
  centers[1] = 1;

  // Calculate bins frequency

  const size_t this_size = vector.size();

  for(size_t i = 0; i < this_size; i++) {
    if(!missing_values.contains(i)) {
    for(size_t j = 0; j < 2; j++) {
      if(static_cast<size_t>(vector[i]) == minimums[j]) {
        frequencies[j]++;
      }
    }
    }
  }

  Histogram histogram(2);
  histogram.centers = centers.to_double_vector();
  histogram.minimums = minimums.to_double_vector();
  histogram.maximums = maximums.to_double_vector();
  histogram.frequencies = frequencies;

  return histogram;
}
*/

/// Returns a vector containing the sum of the frequencies of the bins to which
/// this vector belongs.
/// @param histograms Used histograms.

Vector<size_t> total_frequencies(const Vector<Histogram>&histograms)
{
  const size_t histograms_number = histograms.size();

  Vector<size_t> total_frequencies(histograms_number);

  for(size_t i = 0; i < histograms_number; i++)
  {
    total_frequencies[i] = histograms[i].frequencies[i];
  }

  return total_frequencies;
}


/// Calculates a histogram for each column, each having a given number of bins.
/// It returns a vector of vectors of vectors.
/// The size of the main vector is the number of columns.
/// Each subvector contains the frequencies and centers of that colums.
/// @param bins_number Number of bins for each histogram.

Vector<Histogram> histograms(const Matrix<double>& matrix, const size_t& bins_number)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   Vector<Histogram> histograms(columns_number);

   Vector<double> column(rows_number);

   for(size_t i = 0; i < columns_number; i++)
   {
      column = matrix.get_column(i);

      if(column.is_binary())
      {
          histograms[i] = histogram(column.to_bool_vector());
      }
      else
      {
          histograms[i] = histogram(column, bins_number);
      }
   }

   return histograms;
}


/// Calculates a histogram for each column, each having a given number of bins, when the data has missing values.
/// It returns a vector of vectors.
/// The size of the main vector is the number of columns.
/// Each subvector contains the frequencies and centers of that colums.
/// @param bins_number Number of bins for each histogram.

Vector<Histogram> histograms_missing_values(const Matrix<double>& matrix, const size_t& bins_number)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   Vector<Histogram> histograms(columns_number);

   Vector<double> column(rows_number);

   for(size_t i = 0; i < columns_number; i++)
   {
      column = matrix.get_column(i);

    histograms[i] = histogram_missing_values(column, bins_number);
   }

   return histograms;
}


/// Returns the basic descriptives of the columns.
/// The format is a vector of descriptives structures.
/// The size of that vector is equal to the number of columns in this matrix.
/// @param matrix Used matrix.

Vector<Descriptives> descriptives(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics Class.\n"
             << "Vector<Descriptives> descriptives(const Matrix<double>&) "
                "const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   #endif

   Vector<Descriptives> descriptives(columns_number);

   Vector<double> column(rows_number);

    #pragma omp parallel for private(column)

   for(size_t i = 0; i < columns_number; i++)
   {
      column = matrix.get_column(i);

      descriptives[i] = OpenNN::descriptives(column);

      descriptives[i].name = matrix.get_header(i);
   }

   return descriptives;
}


/// Returns the basic descriptives of the columns when the matrix has missing values.
/// The format is a vector of descriptives structures.
/// The size of that vector is equal to the number of columns in this matrix.
/// @param matrix Used matrix.

Vector<Descriptives> descriptives_missing_values(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics Class.\n"
             << "Vector<Descriptives> descriptives_missing_values(const Matrix<double>&) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   #endif

   Vector<Descriptives> descriptives(columns_number);

   Vector<double> column(rows_number);

   for(size_t i = 0; i < columns_number; i++)
   {
      column = matrix.get_column(i);

      descriptives[i] = descriptives_missing_values(column);
   }

   return descriptives;
}


Vector<Descriptives> descriptives_missing_values(const Matrix<double>& matrix,
                                                 const Vector<size_t>& rows_indices,
                                                 const Vector<size_t>& columns_indices)
{
    const size_t rows_size = rows_indices.size();
    const size_t columns_size = columns_indices.size();

   Vector<Descriptives> descriptives(columns_size);

   Vector<double> column(rows_size);

   for(size_t i = 0; i < columns_size; i++)
   {
      column = matrix.get_column(columns_indices[i], rows_indices);

      descriptives[i] = descriptives_missing_values(column);
   }

   return descriptives;
}


/// Returns the basic descriptives of given columns for given rows.
/// The format is a vector of descriptives structures.
/// The size of that vector is equal to the number of given columns.
/// @param row_indices Indices of the rows for which the descriptives are to be computed.
/// @param columns_indices Indices of the columns for which the descriptives are to be computed.

Vector<Descriptives> descriptives(const Matrix<double>& matrix, const Vector<size_t>& row_indices, const Vector<size_t>& columns_indices)
{
    const size_t row_indices_size = row_indices.size();
    const size_t columns_indices_size = columns_indices.size();

    Vector<Descriptives> descriptives(columns_indices_size);

    size_t row_index, column_index;

    Vector<double> minimums(columns_indices_size, numeric_limits<double>::max());
    Vector<double> maximums;

    maximums.set(columns_indices_size, -numeric_limits<double>::max());

    Vector<double> sums(columns_indices_size, 0.0);
    Vector<double> squared_sums(columns_indices_size, 0.0);

    for(size_t i = 0; i < row_indices_size; i++)
    {
        row_index = row_indices[i];

 #pragma omp parallel for private(column_index)

        for(size_t j = 0; j < columns_indices_size; j++)
        {
            column_index = columns_indices[j];

            if(matrix(row_index,column_index) < minimums[j])
            {
                minimums[j] = matrix(row_index,column_index);
            }

            if(matrix(row_index,column_index) > maximums[j])
            {
                maximums[j] = matrix(row_index,column_index);
            }

            sums[j] += matrix(row_index,column_index);
            squared_sums[j] += matrix(row_index,column_index)*matrix(row_index,column_index);
        }
    }

    const Vector<double> mean = sums/static_cast<double>(row_indices_size);

    Vector<double> standard_deviation(columns_indices_size, 0.0);

    if(row_indices_size > 1)
    {
        for(size_t i = 0; i < columns_indices_size; i++)
        {
            const double numerator = squared_sums[i] -(sums[i] * sums[i]) / row_indices_size;
            const double denominator = row_indices_size - 1.0;

            standard_deviation[i] = numerator / denominator;

            standard_deviation[i] = sqrt(standard_deviation[i]);
        }
    }

    for(size_t i = 0; i < columns_indices_size; i++)
    {
        descriptives[i].minimum = minimums[i];
        descriptives[i].maximum = maximums[i];
        descriptives[i].mean = mean[i];
        descriptives[i].standard_deviation = standard_deviation[i];
    }

    return descriptives;
}


/// Returns the basic descriptives of all the columns for given rows when the matrix has missing values.
/// The format is a vector of descriptives structures.
/// The size of that vector is equal to the number of columns in this matrix.
/// @param matrix Used matrix.
/// @param row_indices Indices of the rows for which the descriptives are to be computed.

Vector<Descriptives> rows_descriptives_missing_values(const Matrix<double>& matrix, const Vector<size_t>& row_indices)
{
    const size_t columns_number = matrix.get_columns_number();

    const size_t row_indices_size = row_indices.size();

    Vector<Descriptives> descriptives(columns_number);

    Vector<double> column(row_indices_size);

    for(size_t i = 0; i < columns_number; i++)
    {
        column = matrix.get_column(i, row_indices);

        descriptives[i] = descriptives_missing_values(column);
    }

    return descriptives;
}


/// Returns the means of given rows.
/// The format is a vector of double values.
/// The size of that vector is equal to the number of given rows.
/// @param matrix Used matrix.
/// @param columns_indices Indices of the columns for which the descriptives are to be computed.

Vector<double> rows_means(const Matrix<double>& matrix, const Vector<size_t>& row_indices)
{
    const size_t columns_number = matrix.get_columns_number();

    Vector<size_t> used_row_indices;

    if(row_indices.empty())
    {
        used_row_indices.set(matrix.get_rows_number());
        used_row_indices.initialize_sequential();
    }
    else
    {
        used_row_indices = row_indices;
    }

    const size_t row_indices_size = used_row_indices.size();

    Vector<double> means(columns_number);

    Vector<double> column(row_indices_size);

    for(size_t i = 0; i < columns_number; i++)
    {
        column = matrix.get_column(i, used_row_indices);

        means[i] = mean_missing_values(column);
    }

    return means;
}


/// Returns the minimums values of given columns.
/// The format is a vector of double values.
/// The size of that vector is equal to the number of given columns.
/// @param matrix Used matrix.
/// @param columns_indices Indices of the columns for which the descriptives are to be computed.

Vector<double> columns_minimums(const Matrix<double>& matrix, const Vector<size_t>& columns_indices)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    Vector<size_t> used_columns_indices;

    if(columns_indices.empty())
    {
        used_columns_indices.set(columns_number);
        used_columns_indices.initialize_sequential();
    }
    else
    {
        used_columns_indices = columns_indices;
    }

    const size_t columns_indices_size = used_columns_indices.size();

    Vector<double> minimums(columns_indices_size);

    size_t index;
    Vector<double> column(rows_number);

    for(size_t i = 0; i < columns_indices_size; i++)
    {
        index = used_columns_indices[i];

        column = matrix.get_column(index);

        minimums[i] = minimum(column);
    }

    return minimums;
}


/// Returns the maximums values of given columns.
/// The format is a vector of double values.
/// The size of that vector is equal to the number of given columns.
/// @param matrix Used matrix.
/// @param columns_indices Indices of the columns for which the descriptives are to be computed.

Vector<double> columns_maximums(const Matrix<double>& matrix, const Vector<size_t>& columns_indices)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    Vector<size_t> used_columns_indices;

    if(columns_indices.empty())
    {
        used_columns_indices.set(columns_number);
        used_columns_indices.initialize_sequential();
    }
    else
    {
        used_columns_indices = columns_indices;
    }

    const size_t columns_indices_size = used_columns_indices.size();

    Vector<double> maximums(columns_indices_size);

    size_t index;
    Vector<double> column(rows_number);

    for(size_t i = 0; i < columns_indices_size; i++)
    {
        index = used_columns_indices[i];

        column = matrix.get_column(index);

        maximums[i] = maximum(column);
    }

    return maximums;
}


double range(const Vector<double>& vector)
{
    const double min = minimum(vector);
    const double max = maximum(vector);

    return abs(max - min);
}


/// Calculates the box plots for a set of rows of each of the given columns of this matrix.
/// @param matrix Used matrix.
/// @param rows_indices Rows to be used for the box plot.
/// @param columns_indices Indices of the columns for which box plots are going to be calculated.
/// @todo

Vector<BoxPlot> box_plots(const Matrix<double>& matrix, const Vector<Vector<size_t>>& rows_indices, const Vector<size_t>& columns_indices)
{
    const size_t columns_number = columns_indices.size();

    #ifdef __OPENNN_DEBUG__

    if(columns_number == rows_indices.size())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Statistics class."
              << "void box_plots(const Matrix<double>&, "
                 "const Vector<Vector<size_t>>&, const Vector<size_t>&) const method.\n"
              << "Size of row indices must be equal to the number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    Vector<BoxPlot> box_plots(columns_number);

    for(size_t i = 0; i < matrix.get_columns_number(); i++)
    {


    }
/*
    Vector<double> column;

     #pragma omp parallel for private(column)

    for(size_t i = 0; i < columns_number; i++)
    {
        box_plots[i].resize(5);

        const size_t rows_number = rows_indices[i].size();

        column = matrix.get_column(columns_indices[i]).get_subvector(rows_indices[i]);

        sort(column.begin(), column.end(), less<double>());

        // Minimum

        box_plots[static_cast<size_t>(i)][0] = column[0];

        if(rows_number % 2 == 0)
        {
            // First quartile

            box_plots[static_cast<size_t>(i)][1] = (column[rows_number / 4] + column[rows_number / 4 + 1]) / 2.0;

            // Second quartile

            box_plots[static_cast<size_t>(i)][2] = (column[rows_number * 2 / 4] +
                           column[rows_number * 2 / 4 + 1]) /
                          2.0;

            // Third quartile

            box_plots[static_cast<size_t>(i)][3] = (column[rows_number * 3 / 4] +
                           column[rows_number * 3 / 4 + 1]) /
                          2.0;
        }
        else
        {
            // First quartile

            box_plots[static_cast<size_t>(i)][1] = column[rows_number / 4];

            // Second quartile

            box_plots[static_cast<size_t>(i)][2] = column[rows_number * 2 / 4];

            //Third quartile

            box_plots[static_cast<size_t>(i)][3] = column[rows_number * 3 / 4];
        }

        // Maximum

        box_plots[static_cast<size_t>(i)][4] = column[rows_number-1];
    }
*/
    return box_plots;
}


/// Returns the minimum, maximum, mean and standard deviation of the elements in
/// the vector.
/// @param Used vector.

Descriptives descriptives(const Vector<double>& vector)  {

    const size_t this_size = vector.size();

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double descriptives(const Vector<double>&).\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  Descriptives descriptives;
  double minimum = numeric_limits<double>::max();
  double maximum;
  double sum = 0;
  double squared_sum = 0;
  size_t count = 0;

  maximum = -1.0*numeric_limits<double>::max();

  for(size_t i = 0; i < this_size; i++)
  {
      if(vector[i] < minimum)
      {
          minimum = vector[i];
      }

      if(vector[i] > maximum)
      {
          maximum = vector[i];
      }

      sum += vector[i];
      squared_sum += vector[i]*vector[i];

      count++;
  }

  const double mean = sum/static_cast<double>(count);

  double standard_deviation;

  if(count <= 1)
  {
    standard_deviation = 0.0;
  }
  else
  {
      const double numerator = squared_sum -(sum * sum) / count;
      const double denominator = this_size - 1.0;

      standard_deviation = numerator / denominator;
  }

  standard_deviation = sqrt(standard_deviation);

  descriptives.minimum = minimum;
  descriptives.maximum = maximum;
  descriptives.mean = mean;
  descriptives.standard_deviation = standard_deviation;

  return descriptives;
}


/// Returns the minimum, maximum, mean and standard deviation of the elements in the vector.
/// @param vector Used vector.

Descriptives descriptives_missing_values(const Vector<double>& vector)
{


    const size_t this_size = vector.size();

#ifdef __OPENNN_DEBUG__

  if(this_size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double descriptives_missing_values(const Vector<double>&, "
              "const Vector<size_t>&).\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  Descriptives descriptives;

  double minimum = numeric_limits<double>::max();
  double maximum;

  double sum = 0;
  double squared_sum = 0;
  size_t count = 0;

  maximum = -numeric_limits<double>::max();

  for(size_t i = 0; i < this_size; i++) {
      if(!::isnan(vector[i]))
      {
          if(vector[i] < minimum)
          {
            minimum = vector[i];
          }

          if(vector[i] > maximum) {
            maximum = vector[i];
          }

          sum += vector[i];
          squared_sum += vector[i] *vector[i];

          count++;
      }
  }

  const double mean = sum/static_cast<double>(count);

  double standard_deviation;

  if(count <= 1)
  {
    standard_deviation = 0.0;
  }
  else
  {
      const double numerator = squared_sum -(sum * sum) / count;
      const double denominator = this_size - 1.0;

      standard_deviation = numerator / denominator;
  }

  standard_deviation = sqrt(standard_deviation);

  descriptives.minimum = minimum;
  descriptives.maximum = maximum;
  descriptives.mean = mean;
  descriptives.standard_deviation = standard_deviation;

  return descriptives;
}


/// Calculates the distance between the empirical distribution of the vector and
/// the normal, half-normal and uniform cumulative distribution. It returns 0, 1
/// or 2 if the closest distribution is the normal, half-normal or the uniform,
/// respectively.
/// @todo review.

size_t perform_distribution_distance_analysis(const Vector<double>& vector)
{
    Vector<double> distances(2, 0.0);

    const size_t n = vector.size();

    Vector<double> sorted_vector(vector);
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

    const Descriptives descriptives = OpenNN::descriptives(vector);

    const double mean = descriptives.mean;
    const double standard_deviation = descriptives.standard_deviation;
    const double minimum = sorted_vector[0];
    const double maximum = sorted_vector[n-1];

 #pragma omp parallel for schedule(dynamic)

    for(size_t i = 0; i < n; i++)
    {
        const double normal_distribution = 0.5 * erfc((mean - sorted_vector[i])/(standard_deviation*sqrt(2)));
/*        const double half_normal_distribution = erf((sorted_vector[i])/(standard_deviation * sqrt(2))); */
        const double uniform_distribution = (sorted_vector[i]-minimum)/(maximum-minimum);

        double empirical_distribution;

        size_t counter = 0;

        if(vector[i] < sorted_vector[0])
        {
            empirical_distribution = 0.0;
        }
        else if(vector[i] >= sorted_vector[n-1])
        {
            empirical_distribution = 1.0;
        }
        else
        {
            counter = static_cast<size_t>(i + 1);

            for(size_t j = i+1; j < n; j++)
            {
                if(sorted_vector[j] <= sorted_vector[i])
                {
                    counter++;
                }
                else
                {
                    break;
                }
            }

            empirical_distribution = static_cast<double>(counter)/static_cast<double>(n);
        }

          #pragma omp critical
        {
            distances[0] += abs(normal_distribution - empirical_distribution);
/*            distances[1] += abs(half_normal_distribution - empirical_distribution); */
            distances[1] += abs(uniform_distribution - empirical_distribution);
        }
    }

    return minimal_index(distances);
}


/// Calculates the distance between the empirical distribution of the vector and
/// the normal, half-normal and uniform cumulative distribution. It returns 0, 1
/// or 2 if the closest distribution is the normal, half-normal or the uniform,
/// respectively.

size_t perform_distribution_distance_analysis_missing_values(const Vector<double>& vector, const Vector<size_t>& missing_indices)
{
    Vector<double> distances(3, 0.0);

    double normal_distribution; // Normal distribution
    double half_normal_distribution; // Half-normal distribution
    double uniform_distribution; // Uniform distribution
    double empirical_distribution; // Empirical distribution

    Vector<size_t> used_indices(1,1, vector.size());
    used_indices = used_indices.get_difference(missing_indices);

    const Vector<double> used_values = vector.get_subvector(used_indices);
    const size_t n = used_values.size();

    Vector<double> sorted_vector(used_values);
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

    const Descriptives descriptives = OpenNN::descriptives(used_values);

    const double mean = descriptives.mean;
    const double standard_deviation = descriptives.standard_deviation;
    const double minimum = sorted_vector[0];
    const double maximum = sorted_vector[n-1];

    if(abs(minimum - maximum) < numeric_limits<double>::epsilon() || standard_deviation < 1.0e-09)
    {
        return 2;
    }

    size_t counter = 0;

 #pragma omp parallel for private(empirical_distribution, normal_distribution, half_normal_distribution, uniform_distribution, counter)

    for(size_t i = 0; i < n; i++)
    {
        normal_distribution = 0.5 * erfc((mean - sorted_vector[i])/(standard_deviation*sqrt(2)));
        half_normal_distribution = erf((sorted_vector[i])/(standard_deviation * sqrt(2)));
        uniform_distribution = (sorted_vector[i]-minimum)/(maximum-minimum);
        counter = 0;

        for(size_t j = 0; j < n; j++)
        {
            if(sorted_vector[j] <= sorted_vector[i])
            {
                counter++;
            }
            else
            {
                break;
            }
        }

        empirical_distribution = static_cast<double>(counter)/static_cast<double>(n);

        #pragma omp critical
        {
            distances[0] += abs(normal_distribution - empirical_distribution);
            distances[1] += abs(half_normal_distribution - empirical_distribution);
            distances[2] += abs(uniform_distribution - empirical_distribution);
        }
    }

    return minimal_index(distances);
}


Vector<double> columns_mean(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();

   return matrix.calculate_columns_sum()/static_cast<double>(rows_number);
}


/// Returns a vector with the mean values of all the matrix columns.
/// The size is equal to the number of columns in the matrix.
/// @param matrix Matrix used.

Vector<double> mean(const Tensor<double>& matrix)
{
    const size_t rows_number = matrix.get_dimension(0);
    const size_t columns_number = matrix.get_dimension(1);

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "Vector<double> mean(const Matrix<double>&) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Mean

   Vector<double> mean(columns_number, 0.0);

   for(size_t j = 0; j < columns_number; j++)
   {
      for(size_t i = 0; i < rows_number; i++)
      {
         mean[j] += matrix(i,j);
      }

      mean[j] /= static_cast<double>(rows_number);
   }

   return mean;
}


/// Returns a vector with the mean values of given columns.
/// The size of the vector is equal to the size of the column indices vector.
/// @param columns_indices Indices of columns.

Vector<double> mean(const Matrix<double>& matrix, const Vector<size_t>& columns_indices)
{
    const size_t rows_number = matrix.get_rows_number();

   const size_t columns_indices_size = columns_indices.size();

   size_t column_index;

   // Mean

   Vector<double> mean(columns_indices_size, 0.0);

   for(size_t j = 0; j < columns_indices_size; j++)
   {
      column_index = columns_indices[j];

      for(size_t i = 0; i < rows_number; i++)
      {
         mean[j] += matrix(i, column_index);
      }

      mean[j] /= static_cast<double>(rows_number);
   }

   return mean;
}


/// Returns a vector with the mean values of given columns for given rows.
/// The size of the vector is equal to the size of the column indices vector.
/// @param matrix Matrix used.
/// @param row_indices Indices of rows.
/// @param columns_indices Indices of columns.

Vector<double> mean(const Matrix<double>& matrix, const Vector<size_t>& row_indices, const Vector<size_t>& columns_indices)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();


   const size_t row_indices_size = row_indices.size();
   const size_t columns_indices_size = columns_indices.size();



   #ifdef __OPENNN_DEBUG__

   // Rows check

   if(row_indices_size > rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "Vector<double> mean(const Matrix<double>& matrix, "
                "const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Size of row indices(" << row_indices_size << ") is greater than number of rows(" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < row_indices_size; i++)
   {
      if(row_indices[i] >= rows_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Statistics class.\n"
                << "Vector<double> mean(const Matrix<double>& matrix, "
                   "const Vector<size_t>&, const Vector<size_t>&) const method.\n"
                << "Row index " << i << " must be less than rows number.\n";

         throw logic_error(buffer.str());
      }
   }

   if(row_indices_size == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "Vector<double> mean(const Matrix<double>& matrix, "
                "const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Size of row indices must be greater than zero.\n";

      throw logic_error(buffer.str());
   }

   // Columns check

   if(columns_indices_size > columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "Vector<double> mean(const Matrix<double>& matrix, "
                "const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Column indices size must be equal or less than columns number.\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < columns_indices_size; i++)
   {
      if(columns_indices[i] >= columns_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Statistics class.\n"
                << "Vector<double> mean(const Matrix<double>& matrix, "
                   "const Vector<size_t>&, const Vector<size_t>&) const method.\n"
                << "Column index " << i << " must be less than columns number.\n";

         throw logic_error(buffer.str());
      }
   }

   #endif

   size_t row_index;
   size_t column_index;

   // Mean

   Vector<double> mean(columns_indices_size, 0.0);

   for(size_t j = 0; j < columns_indices_size; j++)
   {
      column_index = columns_indices[j];

      for(size_t i = 0; i < row_indices_size; i++)
      {
         row_index = row_indices[i];

         mean[j] += matrix(row_index,column_index);
      }

      mean[j] /= static_cast<double>(rows_number);
   }

   return mean;
}


/// Returns a vector with the mean values of all the matrix columns.
/// The size is equal to the number of columns in the matrix.
double mean(const Matrix<double>& matrix, const size_t& column_index)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "double mean(const Matrix<double>&, const size_t&) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   if(column_index >= columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "double mean(const Matrix<double>&, const size_t&) const method.\n"
             << "Index of column must be less than number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Mean

   double mean = 0.0;

    for(size_t i = 0; i < rows_number; i++)
    {
        mean += matrix(i,column_index);
    }

   mean /= static_cast<double>(rows_number);

   return mean;
}


/// Returns a vector with the mean values of given columns for given rows when the matrix has missing values.
/// The size of the vector is equal to the size of the column indices vector.

Vector<double> mean_missing_values(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    const Vector<size_t> row_indices(0, 1, rows_number-1);
    const Vector<size_t> columns_indices(0, 1, columns_number-1);

    return mean_missing_values(matrix, row_indices, columns_indices);
}


/// Returns a vector with the mean values of given columns for given rows when the matrix has missing values.
/// The size of the vector is equal to the size of the column indices vector.
/// @param row_indices Indices of rows.
/// @param columns_indices Indices of columns.

Vector<double> mean_missing_values(const Matrix<double>& matrix, const Vector<size_t>& row_indices, const Vector<size_t>& columns_indices)
{

   const size_t columns_indices_size = columns_indices.size();

   #ifdef __OPENNN_DEBUG__

   const size_t rows_number = matrix.get_rows_number();
   const size_t columns_number = matrix.get_columns_number();

   const size_t row_indices_size = row_indices.size();

   // Rows check

   if(row_indices_size > rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "Vector<double> mean_missing_values(const Matrix<double>&, "
                "const Vector<size_t>&, const Vector<size_t>&, const Vector<Vector<size_t>>&) const method.\n"
             << "Size of row indices(" << row_indices_size << ") is greater than number of rows(" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < row_indices_size; i++)
   {
      if(row_indices[i] >= rows_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Statistics class.\n"
                << "Vector<double> mean_missing_values(const Matrix<double>&, "
                   "const Vector<size_t>&, const Vector<size_t>&, const Vector<Vector<size_t>>&) const method.\n"
                << "Row index " << i << " must be less than rows number.\n";

         throw logic_error(buffer.str());
      }
   }

   if(row_indices_size == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "Vector<double> mean_missing_values(const Matrix<double>&, "
                "const Vector<size_t>&, const Vector<size_t>&, const Vector<Vector<size_t>>&) const method.\n"
             << "Size of row indices must be greater than zero.\n";

      throw logic_error(buffer.str());
   }

   // Columns check

   if(columns_indices_size > columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector<Vector<size_t>>&) const method.\n"
             << "Column indices size must be equal or less than columns number.\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < columns_indices_size; i++)
   {
      if(columns_indices[i] >= columns_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> mean_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector<Vector<size_t>>&) const method.\n"
                << "Column index " << i << " must be less than columns number.\n";

         throw logic_error(buffer.str());
      }
   }

   #endif

   Vector<double> mean(columns_indices_size, 0.0);

   for(size_t j = 0; j < columns_indices_size; j++)
   {
       const size_t column_index = columns_indices[j];

       Vector<double> column_missing_values(matrix.get_column(column_index, row_indices));

       mean[j] = mean_missing_values(column_missing_values);
   }
   return mean;
}


/// Returns a vector with the median values of all the matrix columns.
/// The size is equal to the number of columns in the matrix.

Vector<double> median(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> median() const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // median

   Vector<double> median(columns_number, 0.0);

   for(size_t j = 0; j < columns_number; j++)
   {
       Vector<double> sorted_column(matrix.get_column(j));

       sort(sorted_column.begin(), sorted_column.end(), less<double>());

       if(rows_number % 2 == 0)
       {
         median[j] = (sorted_column[rows_number*2/4] + sorted_column[rows_number*2/4+1])/2;
       }
       else
       {
         median[j] = sorted_column[rows_number*2/4];
       }
   }

   return median;
}


/// Returns a vector with the median values of all the matrix columns.
/// The size is equal to the number of columns in the matrix.

double median(const Matrix<double>& matrix, const size_t& column_index)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "double median(const size_t&) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   if(column_index >= columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "double median(const size_t&) const method.\n"
             << "Index of column must be less than number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // median

   double median = 0.0;

   Vector<double> sorted_column(matrix.get_column(column_index));

   sort(sorted_column.begin(), sorted_column.end(), less<double>());

   if(rows_number % 2 == 0)
   {
     median = (sorted_column[rows_number*2/4] + sorted_column[rows_number*2/4+1])/2;
   }
   else
   {
     median = sorted_column[rows_number*2/4];
   }

   return median;
}


/// Returns a vector with the median values of given columns.
/// The size of the vector is equal to the size of the column indices vector.
/// @param columns_indices Indices of columns.


Vector<double> median(const Matrix<double>& matrix, const Vector<size_t>& columns_indices)
{
    const size_t rows_number = matrix.get_rows_number();

   const size_t columns_indices_size = columns_indices.size();

   size_t column_index;

   // median

   Vector<double> median(columns_indices_size, 0.0);

   for(size_t j = 0; j < columns_indices_size; j++)
   {
      column_index = columns_indices[j];

      Vector<double> sorted_column(matrix.get_column(column_index));

      sort(sorted_column.begin(), sorted_column.end(), less<double>());

      if(rows_number % 2 == 0)
      {
        median[j] = (sorted_column[rows_number*2/4] + sorted_column[rows_number*2/4+1])/2;
      }
      else
      {
        median[j] = sorted_column[rows_number*2/4];
      }
   }

   return median;
}


/// Returns a vector with the median values of given columns for given rows.
/// The size of the vector is equal to the size of the column indices vector.
/// @param row_indices Indices of rows.
/// @param columns_indices Indices of columns.

Vector<double> median(const Matrix<double>& matrix, const Vector<size_t>& row_indices, const Vector<size_t>& columns_indices)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   const size_t row_indices_size = row_indices.size();
   const size_t columns_indices_size = columns_indices.size();

   #ifdef __OPENNN_DEBUG__

   // Rows check

   if(row_indices_size > rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> median(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Size of row indices(" << row_indices_size << ") is greater than number of rows(" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < row_indices_size; i++)
   {
      if(row_indices[i] >= rows_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> median(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
                << "Row index " << i << " must be less than rows number.\n";

         throw logic_error(buffer.str());
      }
   }

   if(row_indices_size == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> median(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Size of row indices must be greater than zero.\n";

      throw logic_error(buffer.str());
   }

   // Columns check

   if(columns_indices_size > columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> median(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
             << "Column indices size must be equal or less than columns number.\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < columns_indices_size; i++)
   {
      if(columns_indices[i] >= columns_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> median(const Vector<size_t>&, const Vector<size_t>&) const method.\n"
                << "Column index " << i << " must be less than columns number.\n";

         throw logic_error(buffer.str());
      }
   }

   #endif

   size_t column_index;

   // median

   Vector<double> median(columns_indices_size, 0.0);

   for(size_t j = 0; j < columns_indices_size; j++)
   {
      column_index = columns_indices[j];

      Vector<double> sorted_column(matrix.get_column(column_index, row_indices));

      sort(sorted_column.begin(), sorted_column.end(), less<double>());

      if(row_indices_size % 2 == 0)
      {
        median[j] = (sorted_column[row_indices_size*2/4] + sorted_column[row_indices_size*2/4 + 1])/2;
      }
      else
      {
        median[j] = sorted_column[row_indices_size * 2 / 4];
      }
   }
   return median;
}

/// Returns the median of the elements of a vector with missing values.

double median_missing_values(const Vector<double>& vector)
{

  const size_t this_size = vector.size();

  const size_t nan = vector.count_NAN();

  const size_t new_size = this_size - nan;

  Vector<double> new_vector(new_size);

  size_t index = 0;

  for(size_t i = 0; i < this_size; i++){

      if(!::isnan(vector[i]))
      {
           new_vector[index] = vector[i];

           index++;

       }
     }

  return(median(new_vector));
}


/// Returns a vector with the median values of given columns for given rows when the matrix has missing values.
/// The size of the vector is equal to the size of the column indices vector.

Vector<double> median_missing_values(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    Vector<size_t> row_indices(0, 1, rows_number-1);
    Vector<size_t> columns_indices(0, 1, columns_number-1);

    return median_missing_values(matrix, row_indices, columns_indices);
}


/// Returns a vector with the median values of given columns for given rows when the matrix has missing values.
/// The size of the vector is equal to the size of the column indices vector.
/// @param row_indices Indices of rows.
/// @param columns_indices Indices of columns.

Vector<double> median_missing_values(const Matrix<double>& matrix,
                                     const Vector<size_t>& row_indices,
                                     const Vector<size_t>& columns_indices)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();
    const size_t columns_indices_size = columns_indices.size();

   #ifdef __OPENNN_DEBUG__

   const size_t row_indices_size = row_indices.size();

   // Rows check

   if(row_indices_size > rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> median_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector<Vector<size_t>>&) const method.\n"
             << "Size of row indices(" << row_indices_size << ") is greater than number of rows(" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < row_indices_size; i++)
   {
      if(row_indices[i] >= rows_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> median_missing_values(const Vector<size_t>&, const Vector<size_t>&, Vector<Vector<size_t>>&) const method.\n"
                << "Row index " << i << " must be less than rows number.\n";

         throw logic_error(buffer.str());
      }
   }

   if(row_indices_size == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> median_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector<Vector<size_t>>&) const method.\n"
             << "Size of row indices must be greater than zero.\n";

      throw logic_error(buffer.str());
   }

   // Columns check

   if(columns_indices_size > columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Vector<double> median_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector<Vector<size_t>>&) const method.\n"
             << "Column indices size must be equal or less than columns number.\n";

      throw logic_error(buffer.str());
   }

   for(size_t i = 0; i < columns_indices_size; i++)
   {
      if(columns_indices[i] >= columns_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Vector<double> median_missing_values(const Vector<size_t>&, const Vector<size_t>&, const Vector<Vector<size_t>>&) const method.\n"
                << "Column index " << i << " must be less than columns number.\n";

         throw logic_error(buffer.str());
      }
   }

   #endif

   // median

   Vector<double> median(columns_indices_size, 0.0);

   for(size_t j = 0; j < columns_indices_size; j++)
   {
      const size_t column_index = columns_indices[j];

      Vector<double> column_missing_values(matrix.get_column(column_index, row_indices));

      median[j] = median_missing_values(column_missing_values);
   }

   return median;
}


/// Returns true if the elements in the vector have a normal distribution with a given critical value.
/// @param critical_value Critical value to be used in the test.

bool perform_Lilliefors_normality_test(const Vector<double>& vector, const double& critical_value)
{
#ifndef Cpp11__

    const size_t n = vector.size();

    const double mean = OpenNN::mean(vector);
    const double standard_deviation = OpenNN::standard_deviation(vector);

    Vector<double> sorted_vector(vector);

    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

    double Fx;
    double Snx;

    double D = -1;

    for(size_t i = 0; i < n; i++)
    {
        Fx = 0.5 * erfc((mean - vector[i])/(standard_deviation*sqrt(2)));

        if(vector[i] < sorted_vector[0])
        {
            Snx = 0.0;
        }
        else if(vector[i] >= sorted_vector[n-1])
        {
            Snx = 1.0;
        }
        else
        {
            for(size_t j = 0; j < n-1; j++)
            {
                if(vector[i] >= sorted_vector[j] && vector[i] < sorted_vector[j+1])
                {
                    Snx = static_cast<double>(j+1)/static_cast<double>(n);
                }
            }
        }

        if(D < abs(Fx - Snx))
        {
            D = abs(Fx - Snx);
        }
    }

    if(D < critical_value)
    {
        return true;
    }
    else
    {
        return false;
    }

#else
    return false;
#endif
}


/// Returns true if the elements in the vector have a normal distribution with a given set of critical values.
/// @param critical_values Critical values to be used in the test.

Vector<bool> perform_Lilliefors_normality_test(const Vector<double>& vector, const Vector<double>& critical_values)
{
    const size_t size = critical_values.size();

    Vector<bool> normality_tests(size);

    for(size_t i = 0; i < size; i++)
    {
        normality_tests[i] = perform_Lilliefors_normality_test(vector, critical_values[i]);
    }

    return normality_tests;
}


/// Calculates the distance between the empirical distribution of the vector and the
/// normal distribution.

double normal_distribution_distance(const Vector<double>& vector)
{
    double normal_distribution_distance = 0.0;

    const size_t n = vector.size();

    const double mean_value = mean(vector);
    const double standard_deviation = OpenNN::standard_deviation(vector);

    double normal_distribution; // Normal distribution
    double empirical_distribution; // Empirical distribution

    Vector<double> sorted_vector(vector);
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

    size_t counter = 0;

    for(size_t i = 0; i < n; i++)
    {
        normal_distribution = 0.5 * erfc((mean_value - sorted_vector[i])/(standard_deviation*sqrt(2.0)));
        counter = 0;

        for(size_t j = 0; j < n; j++)
        {
            if(sorted_vector[j] <= sorted_vector[i])
            {
                counter++;
            }
            else
            {
                break;
            }
        }

        empirical_distribution = static_cast<double>(counter)/static_cast<double>(n);

        normal_distribution_distance += abs(normal_distribution - empirical_distribution);
    }

    return normal_distribution_distance;
}


/// Calculates the distance between the empirical distribution of the vector and the
/// half normal distribution.

double half_normal_distribution_distance(const Vector<double>& vector)
{
    double half_normal_distribution_distance = 0.0;

    const size_t n = vector.size();

    const double standard_deviation = OpenNN::standard_deviation(vector);

    double half_normal_distribution; // Half normal distribution
    double empirical_distribution; // Empirical distribution

    Vector<double> sorted_vector(vector);
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

    size_t counter = 0;

    for(size_t i = 0; i < n; i++)
    {
        half_normal_distribution = erf((sorted_vector[i])/(standard_deviation * sqrt(2)));
        counter = 0;

        for(size_t j = 0; j < n; j++)
        {
            if(sorted_vector[j] <= sorted_vector[i])
            {
                counter++;
            }
            else
            {
                break;
            }
        }

        empirical_distribution = static_cast<double>(counter)/static_cast<double>(n);

        half_normal_distribution_distance += abs(half_normal_distribution - empirical_distribution);
    }

    return half_normal_distribution_distance;
}


/// Calculates the distance between the empirical distribution of the vector and the
/// uniform distribution.

double uniform_distribution_distance(const Vector<double>& vector)
{
    double uniform_distribution_distance = 0.0;

    const size_t n = vector.size();

    double uniform_distribution; // Uniform distribution
    double empirical_distribution; // Empirical distribution

    Vector<double> sorted_vector(vector);
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

    const double minimum = sorted_vector[0];
    const double maximum = sorted_vector[n-1];

    size_t counter = 0;

    for(size_t i = 0; i < n; i++)
    {
        uniform_distribution = (sorted_vector[i]-minimum)/(maximum-minimum);
        counter = 0;

        for(size_t j = 0; j < n; j++)
        {
            if(sorted_vector[j] <= sorted_vector[i])
            {
                counter++;
            }
            else
            {
                break;
            }
        }

        empirical_distribution = static_cast<double>(counter)/static_cast<double>(n);

        uniform_distribution_distance += abs(uniform_distribution - empirical_distribution);
    }

    return uniform_distribution_distance;
}


/// Performs the Lilliefors normality tests varying the confindence level from 0.05 to 0.5.
/// It returns a vector containing the results of the tests.
/// @todo review.

Vector<bool> perform_normality_analysis(const Vector<double>& vector)
{
    const size_t size = vector.size();

    double significance_level = 0.05;

    double A_significance_level;
    double B_significance_level;
    Vector<double> critical_values(9);

    for(size_t i = 0; i < 9; i++)
    {
        A_significance_level = 6.32207539843126
                               - 17.1398870006148*(1 - significance_level)
                               + 38.42812675101057*pow((1 - significance_level),2)
                               - 45.93241384693391*pow((1 - significance_level),3)
                               + 7.88697700041829*pow((1 - significance_level),4)
                               + 29.79317711037858*pow((1 - significance_level),5)
                               - 18.48090137098585*pow((1 - significance_level),6);

        B_significance_level = 12.940399038404
                               - 53.458334259532*(1 - significance_level)
                               + 186.923866119699*pow((1 - significance_level),2)
                               - 410.582178349305*pow((1 - significance_level),3)
                               + 517.377862566267*pow((1 - significance_level),4)
                               - 343.581476222384*pow((1 - significance_level),5)
                               + 92.123451358715*pow((1 - significance_level),6);

        critical_values[i] = sqrt(1/(A_significance_level*size+B_significance_level));

        significance_level += 0.05;
    }

    //return vector.Lilliefors_normality_test(critical_values);
    return perform_Lilliefors_normality_test(vector,critical_values);

}


///@todo

double normality_parameter(const Vector<double>& vector)
{
    const double max = maximum(vector);
    const double min = minimum(vector);

    const size_t n = vector.size();

    const double mean_value = mean(vector);
    const double standard_deviation = OpenNN::standard_deviation(vector);

    double normal_distribution;
    double empirical_distribution;
    double previous_normal_distribution = 0.0;
    double previous_empirical_distribution = 0.0;

    Vector<double> sorted_vector(vector);
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

    double empirical_area = 0.0;
    double normal_area = 0.0;

    size_t counter = 0;

    for(size_t i = 0; i < n; i++)
    {
        normal_distribution = 0.5 * erfc((mean_value - sorted_vector[i])/(standard_deviation*sqrt(2.0)));
        counter = 0;

        for(size_t j = 0; j < n; j++)
        {
            if(sorted_vector[j] <= sorted_vector[i])
            {
                counter++;
            }
            else
            {
                break;
            }
        }

        empirical_distribution = static_cast<double>(counter)/static_cast<double>(n);

        if(i == 0)
        {
            previous_normal_distribution = normal_distribution;
            previous_empirical_distribution = empirical_distribution;
        }
        else
        {
            normal_area += 0.5*(sorted_vector[i]-sorted_vector[i-1])*(normal_distribution+previous_normal_distribution);
            empirical_area += 0.5*(sorted_vector[i]-sorted_vector[i-1])*(empirical_distribution+previous_empirical_distribution);

            previous_normal_distribution = normal_distribution;
            previous_empirical_distribution = empirical_distribution;
        }
    }

    const double uniform_area = (max - min)/2.0;

    return uniform_area;
}


Vector<double> variation_percentage(const Vector<double>& vector)
{
    const size_t this_size = vector.size();

    Vector<double> new_vector(this_size, 0);

    for(size_t i = 1; i < this_size; i++)
    {
        if(abs(vector[i-1]) < numeric_limits<double>::min())
        {
            new_vector[i] = (vector[i] - vector[i-1])*100.0/vector[i-1];
        }
    }

    return new_vector;
}


/// Returns the index of the smallest element in the vector.

size_t minimal_index(const Vector<double>& vector)
{
    const size_t size = vector.size();

    if(size == 0) return size_t();

    size_t minimal_index = 0;
    double minimum = vector[0];

    for(size_t i = 1; i < size; i++)
    {
        if(vector[i] < minimum)
        {
            minimal_index = i;
            minimum = vector[i];
        }
    }

    return minimal_index;
}


/// Returns the index of the largest element in the vector.

size_t maximal_index(const Vector<double>& vector)
{
    const size_t size = vector.size();

    if(size == 0) return size_t();

    size_t maximal_index = 0;
    double maximum = vector[0];

    for(size_t i = 1; i < size; i++)
    {
        if(vector[i] > maximum)
        {
            maximal_index = i;
            maximum = vector[i];
        }
    }

    return maximal_index;
}


/// Returns the indices of the smallest elements in the vector.
/// @param number Number of minimal indices to be computed.

Vector<size_t> minimal_indices(const Vector<double>& vector, const size_t &number)
{
  const size_t this_size = vector.size();

  const Vector<size_t> rank = vector.calculate_less_rank();

  Vector<size_t> minimal_indices(number);

   #pragma omp parallel for

  for(int i = 0; i < static_cast<int>(this_size); i++)
  {
    for(size_t j = 0; j < number; j++)
    {
      if(rank[static_cast<size_t>(i)] == j)
      {
        minimal_indices[j] = static_cast<size_t>(i);
      }
    }
  }

  return minimal_indices;
}


/// Returns the indices of the largest elements in the vector.
/// @param number Number of maximal indices to be computed.

Vector<size_t> maximal_indices(const Vector<double>& vector, const size_t& number)
{
  const size_t this_size = vector.size();

  const Vector<size_t> rank = vector.calculate_greater_rank();

  Vector<size_t> maximal_indices(number);

  for(size_t i = 0; i < this_size; i++)
  {
    for(size_t j = 0; j < number; j++)
    {
      if(rank[i] == j)
      {
        maximal_indices[j] = i;
      }
    }
  }

  return maximal_indices;
}


/// Returns the row and column indices corresponding to the entry with minimum value.

Vector<size_t> minimal_indices(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   double minimum = matrix(0,0);
   Vector<size_t> minimal_indices(2, 0);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         if(matrix(i,j) < minimum)
         {
            minimum = matrix(i,j);
            minimal_indices[0] = i;
            minimal_indices[1] = j;
         }
      }
   }

   return minimal_indices;
}


Vector<size_t> minimal_indices_omit(const Matrix<double>& matrix, const double& value_to_omit)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   double minimum = numeric_limits<double>::max();

   Vector<size_t> minimal_indices(2, 0);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         if(abs(matrix(i,j) - value_to_omit) < numeric_limits<double>::min()
         && matrix(i,j) < minimum)
         {
            minimum = matrix(i,j);
            minimal_indices[0] = i;
            minimal_indices[1] = j;
         }
      }
   }

   return minimal_indices;
}


/// Returns the row and column indices corresponding to the entry with maximum value.

Vector<size_t> maximal_indices(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   double maximum = matrix(0,0);

   Vector<size_t> maximal_indices(2, 0);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         if(matrix(i,j) > maximum)
         {
            maximum = matrix(i,j);
            maximal_indices[0] = i;
            maximal_indices[1] = j;
         }
      }
   }

   return maximal_indices;
}


Vector<size_t> maximal_indices_omit(const Matrix<double>& matrix, const double& value_to_omit)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   double maximum = numeric_limits<double>::min();

   Vector<size_t> maximum_indices(2, 0);

   for(size_t i = 0; i < rows_number; i++)
   {
      for(size_t j = 0; j < columns_number; j++)
      {
         if(abs(matrix(i,j) - value_to_omit) < numeric_limits<double>::min()
         && matrix(i,j) > maximum)
         {
            maximum = matrix(i,j);
            maximum_indices[0] = i;
            maximum_indices[1] = j;
         }
      }
   }

   return maximum_indices;
}


/// Returns the minimum value from all elements in the matrix.

double minimum_matrix(const Matrix<double>& matrix)
{
   double minimum = static_cast<double>(numeric_limits<double>::max());

   for(size_t i = 0; i < matrix.size(); i++)
   {
         if(matrix[i] < minimum)
         {
            minimum = matrix[i];
         }
   }

   return minimum;
}


/// Returns the maximum value from all elements in the matrix.

double maximum_matrix(const Matrix<double>& matrix)
{
    double maximum = static_cast<double>(-numeric_limits<double>::max());

    for(size_t i = 0; i < matrix.size(); i++)
    {
          if(matrix[i] > maximum)
          {
             maximum = matrix[i];
          }
    }

   return maximum;
}


double strongest(const Vector<double>& vector)
{
    const size_t size = vector.size();

    if(size == 0) return 0.0;

    double strongest = vector[0];

    for(size_t i = 0; i < size; i++)
    {
        if(fabs(vector[i]) > fabs(strongest))
        {
            strongest = vector[i];
        }
    }

    return strongest;
}


/// Returns a vector containing the means of the subsets which correspond
/// to each of the given integers. The matrix must have 2 columns, the first
/// one containing the integers and the second one the corresponding values.

Vector<double> means_by_categories(const Matrix<double>& matrix)
{
    const size_t integers_number = matrix.size();
    Vector<double> elements_uniques = matrix.get_column(0).get_unique_elements();
    Vector<double> values = matrix.get_column(1);

    #ifdef __OPENNN_DEBUG__

    if(integers_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "Vector<T> calculate_means_integers(const Matrix<double>& \n"
              << "Number of integers must be greater than 0.\n";

       throw logic_error(buffer.str());
    }

    #endif

    const size_t rows_number = matrix.get_rows_number();

    Vector<double> means(elements_uniques);

    double sum = 0.0;
    size_t count = 0;

    for(size_t i = 0; i < integers_number; i++)
    {
        sum = 0.0;
        count = 0;

        for(unsigned j = 0; j < rows_number; j++)
        {
            if(matrix(j,0) == elements_uniques[i] && !::isnan(values[j]))
            {
                sum += matrix(j,1);
                count++;
            }
        }

        if(count != 0)
        {
            means[i] = static_cast<double>(sum)/static_cast<double>(count);

        }
        else
        {
            means[i] = 0.0;
        }
    }

    return means;
}


/// Returns a vector containing the means of the subsets which correspond
/// to each of the given integers. The matrix must have 2 columns, the first
/// one containing the integers and the second one the corresponding values.
/// Both columns can contain NAN.

Vector<double> means_by_categories_missing_values(const Matrix<double>& matrix)
{

    const size_t integers_number = matrix.size();
    Vector<double> elements_uniques = matrix.get_column(0).get_unique_elements();
    Vector<double> values = matrix.get_column(1);

    #ifdef __OPENNN_DEBUG__

    if(integers_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "Vector<T> calculate_means_integers(const Matrix<double>& \n"
              << "Number of integers must be greater than 0.\n";

       throw logic_error(buffer.str());
    }

    #endif

    const size_t rows_number = matrix.get_rows_number();

    Vector<double> means(elements_uniques);

    double sum = 0.0;
    size_t count = 0;

    for(size_t i = 0; i < integers_number; i++)
    {
        sum = 0.0;
        count = 0;

        for(unsigned j = 0; j < rows_number; j++)
        {
            if(matrix(j,0) == elements_uniques[i] && !::isnan(values[j]))
            {
                sum += matrix(j,1);
                count++;
            }
        }

        if(count != 0)
        {
            means[i] = static_cast<double>(sum)/static_cast<double>(count);

        }
        else
        {
            means[i] = 0.0;
        }
    }

    return means;

}


/// Returns a vector containing the values of the means for the 0s and 1s of a
/// binary column.
/// The matrix must have 2 columns, the first one has to be binary.

Vector<double> means_binary_column(const Matrix<double>& matrix)
{
    Vector<double> means(2,0.0);

    size_t count = 0;

    for(size_t i = 0; i < matrix.get_rows_number(); i++)
    {
        if(matrix(i,0) == 0.0)
        {
            means[0] += matrix(i,1);
            count++;
        }
        else if(matrix(i,0) == 1.0)
        {
            means[1] += matrix(i,1);
            count++;
        }
    }

    if(count != 0)
    {
        means[0] = static_cast<double>(means[0])/static_cast<double>(count);
        means[1] = static_cast<double>(means[1])/static_cast<double>(count);
    }
    else
    {
        means[0] = 0.0;
        means[1] = 0.0;
    }

    return means;
}


/// Returns a vector containing the values of the means for the 1s of each
/// of all binary columns.
/// All the columns except the last one must be binary.

Vector<double> means_binary_columns(const Matrix<double>& matrix)
{
    Vector<double> means(matrix.get_columns_number()-1);

    double sum = 0.0;
    size_t count = 0;

    for(size_t i = 0; i < matrix.get_columns_number()-1; i++)
    {
        sum = 0.0;
        count = 0;

        for(size_t j = 0; j < matrix.get_rows_number(); j++)
        {
            if(matrix(j,i) == 1.0)
            {
                sum += matrix(j,matrix.get_columns_number()-1);

                count++;
            }
        }

        if(count != 0)
        {
            means[i] = static_cast<double>(sum)/static_cast<double>(count);

        }
        else
        {
            means[i] = 0.0;
        }
    }
    return means;
}


/// Returns a vector containing the values of the means for the 1s of each
/// of the binary columns.
/// All the columns except the last one must be binary.

Vector<double>means_binary_columns_missing_values(const Matrix<double>& matrix)
{
   return means_binary_columns(matrix.delete_rows_with_value(static_cast<double>(NAN)));
}


///Returns a vector with the percentiles of a vector given.

Vector<double> percentiles(const Vector<double>& vector)
{
  const size_t this_size = vector.size();

  const Vector<size_t> sorted_vector = vector.sort_ascending_indices();

  Vector<double> percentiles(10);

  if(this_size % 2 == 0)
  {
    percentiles[0] = (sorted_vector[this_size * 1 / 10] + sorted_vector[this_size * 1 / 10 + 1]) / 2.0;
    percentiles[1] = (sorted_vector[this_size * 2 / 10] + sorted_vector[this_size * 2 / 10 + 1]) / 2.0;
    percentiles[2] = (sorted_vector[this_size * 3 / 10] + sorted_vector[this_size * 3 / 10 + 1]) / 2.0;
    percentiles[3] = (sorted_vector[this_size * 4 / 10] + sorted_vector[this_size * 4 / 10 + 1]) / 2.0;
    percentiles[4] = (sorted_vector[this_size * 5 / 10] + sorted_vector[this_size * 5 / 10 + 1]) / 2.0;
    percentiles[5] = (sorted_vector[this_size * 6 / 10] + sorted_vector[this_size * 6 / 10 + 1]) / 2.0;
    percentiles[6] = (sorted_vector[this_size * 7 / 10] + sorted_vector[this_size * 7 / 10 + 1]) / 2.0;
    percentiles[7] = (sorted_vector[this_size * 8 / 10] + sorted_vector[this_size * 8 / 10 + 1]) / 2.0;
    percentiles[8] = (sorted_vector[this_size * 9 / 10] + sorted_vector[this_size * 9 / 10 + 1]) / 2.0;
    percentiles[9] = maximum(vector);
  }
  else
  {
    percentiles[0] = static_cast<double>(sorted_vector[this_size * 1 / 10]);
    percentiles[1] = static_cast<double>(sorted_vector[this_size * 2 / 10]);
    percentiles[2] = static_cast<double>(sorted_vector[this_size * 3 / 10]);
    percentiles[3] = static_cast<double>(sorted_vector[this_size * 4 / 10]);
    percentiles[4] = static_cast<double>(sorted_vector[this_size * 5 / 10]);
    percentiles[5] = static_cast<double>(sorted_vector[this_size * 6 / 10]);
    percentiles[6] = static_cast<double>(sorted_vector[this_size * 7 / 10]);
    percentiles[7] = static_cast<double>(sorted_vector[this_size * 8 / 10]);
    percentiles[8] = static_cast<double>(sorted_vector[this_size * 9 / 10]);
    percentiles[9] = maximum(vector);
  }
  return percentiles;
}


Vector<double> percentiles_missing_values(const Vector<double>& x)
{
    const size_t this_size = x.size();

    size_t new_size;

    Vector<double> new_x(new_size);

    size_t index = 0;

    for(size_t i = 0; i < this_size ; i++)
    {
        if(!isnan(x[i]))
        {
            new_x[index] = x[i];

            index++;
        }
    }
    return percentiles(new_x);
}


/// Returns the weighted mean of the vector.
/// @param weights Weights of the elements of the vector in the mean.

double weighted_mean(const Vector<double>& vector, const Vector<double>& weights)
{
    const size_t this_size = vector.size();

  #ifdef __OPENNN_DEBUG__

    if(this_size == 0) {
      ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "double calculate_weighted_mean(const Vector<double>&) const method.\n"
             << "Size must be greater than zero.\n";

      throw logic_error(buffer.str());
    }

    const size_t weights_size = weights.size();

    if(this_size != weights_size) {
      ostringstream buffer;

      buffer << "OpenNN Exception: Vector Template.\n"
             << "double calculate_weighted_mean(const Vector<double>&) "
                "const method.\n"
             << "Size of weights must be equal to vector size.\n";

      throw logic_error(buffer.str());
    }
  #endif

    double weights_sum = 0;

    double sum = 0;

    for(size_t i = 0; i < this_size; i++)
    {
        sum += weights[i]*vector[i];
        weights_sum += weights[i];
    }

    const double mean = sum / weights_sum;

    return mean;
}


/// Calculates the explained variance for a given vector(principal components analysis).
/// This method returns a vector whose size is the same as the size of the given vector.

Vector<double> explained_variance(const Vector<double>& vector)
{
    const size_t this_size = vector.size();

    #ifdef __OPENNN_DEBUG__

      if(this_size == 0) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "Vector<T> explained_variance() const method.\n"
               << "Size of vector must be greater than zero.\n";

        throw logic_error(buffer.str());
      }

    #endif

    const double this_sum = absolute_value(vector).calculate_sum();

    #ifdef __OPENNN_DEBUG__

      if(abs(this_sum) < numeric_limits<double>::min())
      {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "Vector<T> explained_variance() const method.\n"
               << "Sum of the members of the vector (" << abs(this_sum) << ") must be greater than zero.\n";

        throw logic_error(buffer.str());
      }

    #endif

    #ifdef __OPENNN_DEBUG__

      if(this_sum < 0) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "Vector<T> explained_variance() const method.\n"
               << "Sum of the members of the vector cannot be negative.\n";

        throw logic_error(buffer.str());
      }

    #endif

    Vector<double> explained_variance(this_size);

    for(size_t i = 0; i < this_size; i++)
    {
        explained_variance[i] = vector[i]*100.0/this_sum;

        if(explained_variance[i] - 0.0 < 1.0e-16)
        {
            explained_variance[i] = 0.0;
        }
    }

    #ifdef __OPENNN_DEBUG__

      if(explained_variance.calculate_sum() != 1.0) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Vector Template.\n"
               << "Vector<T> explained_variance() const method.\n"
               << "Sum of explained variance must be 1.\n";

        throw logic_error(buffer.str());
      }

    #endif

    return explained_variance;
}

}
