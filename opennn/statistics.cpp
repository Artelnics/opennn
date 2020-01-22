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

Tensor<type, 1> Descriptives::to_vector() const
{
  Tensor<type, 1> statistics_vector(4);
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

void Descriptives::save(const string &file_name) const
{
/*
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
*/
}


Histogram::Histogram() {}


/// Destructor.

Histogram::~Histogram() {}


/// Bins number constructor.
/// @param bins_number Number of bins in the histogram.

Histogram::Histogram(const int &bins_number)
{
  centers.resize(bins_number);
  frequencies.resize(bins_number);
}


/// Values constructor.
/// @param new_centers Center values for the bins.
/// @param new_frequencies Number of variates in each bin.

Histogram::Histogram(const Tensor<type, 1>&new_centers,
                        const Tensor<int, 1>&new_frequencies) {
  centers = new_centers;
  frequencies = new_frequencies;
}


/// Returns the number of bins in the histogram.

int Histogram::get_bins_number() const {
  return centers.size();
}


/// Returns the number of bins with zero variates.

int Histogram::count_empty_bins() const
{
/*
  return frequencies.count_equal_to(0);
*/
    return 0;
}


/// Returns the number of variates in the less populated bin.

int Histogram::calculate_minimum_frequency() const
{ 
 return minimum(frequencies);
}


/// Returns the number of variates in the most populated bin.

int Histogram::calculate_maximum_frequency() const
{
  return maximum(frequencies);

}


/// Retuns the index of the most populated bin.

int Histogram::calculate_most_populated_bin() const
{
/*
  return maximal_index(frequencies.to_double_vector());
*/
    return 0;
}


/// Returns a vector with the centers of the less populated bins.

Tensor<type, 1> Histogram::calculate_minimal_centers() const
{
/*
  const int minimum_frequency = calculate_minimum_frequency();
  const Tensor<int, 1> minimal_indices = frequencies.get_indices_equal_to(minimum_frequency);

  return(centers.get_subvector(minimal_indices));
*/

    return Tensor<type, 1>();
}


/// Returns a vector with the centers of the most populated bins.

Tensor<type, 1> Histogram::calculate_maximal_centers() const
{
/*
  const int maximum_frequency = calculate_maximum_frequency();

  const Tensor<int, 1> maximal_indices = frequencies.get_indices_equal_to(maximum_frequency);

  return(centers.get_subvector(maximal_indices));
*/
    return Tensor<type, 1>();

}


/// Returns the number of the bin to which a given value belongs to.
/// @param value Value for which we want to get the bin.

int Histogram::calculate_bin(const double &value) const
{
  const int bins_number = get_bins_number();

  const double minimum_center = centers[0];
  const double maximum_center = centers[bins_number - 1];

  const double length = static_cast<double>(maximum_center - minimum_center)/static_cast<double>(bins_number - 1.0);

  double minimum_value = centers[0] - length / 2;
  double maximum_value = minimum_value + length;

  if(value < maximum_value) {
    return 0;
  }

  for(int j = 1; j < bins_number - 1; j++) {
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
           << "int Histogram::calculate_bin(const double&) const.\n"
           << "Unknown return value.\n";

    throw logic_error(buffer.str());
  }
}


/// Returns the frequency of the bin to which a given value bolongs to.
/// @param value Value for which we want to get the frequency.

int Histogram::calculate_frequency(const double &value) const
{
  const int bin_number = calculate_bin(value);

  const int frequency = frequencies[bin_number];

  return frequency;
}


/// Returns the smallest element of a double vector.
/// @param vector

double minimum(const Tensor<type, 1>& vector)
{
/*
    const double min = *min_element(vector.begin(), vector.end());

    return min;
*/
    return 0.0;
}


/// Returns the smallest element of a int vector.
/// @param vector

int minimum(const Tensor<int, 1>& vector)
{
/*
    const int min = *min_element(vector.begin(), vector.end());

    return min;
*/
    return 0.0;
}


time_t minimum(const vector<time_t>& vector)
{
    /*
    const time_t min = *min_element(vector.begin(), vector.end());

    return min;
*/
    return 0.0;

}


/// Returns the largest element in the vector.
/// @param vector

double maximum(const Tensor<type, 1>& vector)
{
/*
    const double max = *max_element(vector.begin(), vector.end());

    return max;
*/
    return 0.0;

}


int maximum(const Tensor<int, 1>& vector)
{
/*
    const int max = *max_element(vector.begin(), vector.end());

    return max;
*/
    return 0;
}


time_t maximum(const vector<time_t>& vector)
{
    const time_t max = *max_element(vector.begin(), vector.end());

    return max;
}


/// Returns the smallest element in the vector.

double minimum_missing_values(const Tensor<type, 1>& vector)
{
  const Index size = vector.dimension(0);

  double minimum = 999999;

  for(int i = 0; i < size; i++)
  {
    if(vector[i] < minimum && !::isnan(vector[i]))
    {
      minimum = vector[i];
    }
  }

  return minimum;
}


/// Returns the largest element in the vector.

double maximum_missing_values(const Tensor<type, 1>& vector)
{
  const Index size = vector.dimension(0);

  double maximum = -999999;

  for(int i = 0; i < size; i++) {
    if(!::isnan(vector[i]) && vector[i] > maximum) {
      maximum = vector[i];
    }
  }

  return maximum;
}


/// Returns the mean of the elements in the vector.
/// @param vector

double mean(const Tensor<type, 1>& vector)
{
  const Index size = vector.dimension(0);

#ifdef __OPENNN_DEBUG__

  if(size == 0)
  {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics class.\n"
           << "double mean(const Tensor<type, 1>&) const method.\n"
           << "Size of vector must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif
/*
  const double sum = vector.calculate_sum();

  const double mean = sum /static_cast<double>(size);

  return mean;
*/
    return 0.0;
}


/// Returns the mean of the subvector defined by a start and end elements.
/// @param vector
/// @param begin Start element.
/// @param end End element.

double mean(const Tensor<type, 1>& vector, const int& begin, const int& end)
{
  #ifdef __OPENNN_DEBUG__

    if(begin > end) {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "double mean(const Tensor<type, 1>& vector, const int& begin, const int& end) \n"
             << "Begin must be less or equal than end.\n";

      throw logic_error(buffer.str());
    }

  #endif

  if(end == begin) return vector[begin];

  double sum = 0.0;

  for(int i = begin; i <= end; i++)
  {
      sum += vector[i];
  }

  return(sum /static_cast<double>(end-begin+1));
}


/// Returns the mean of the elements in the vector.
/// @param vector

double mean_missing_values(const Tensor<type, 1>& vector)
{
  const Index size = vector.dimension(0);

#ifdef __OPENNN_DEBUG__

  if(size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double mean_missing_values(const Tensor<type, 1>& vector, const int& begin, const int& end) "
              "const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  double sum = 0;

  int count = 0;

  for(int i = 0; i < size; i++) {
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

double variance(const Tensor<type, 1>& vector)
{
  const Index size = vector.dimension(0);

#ifdef __OPENNN_DEBUG__

  if(size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double variance(const Tensor<type, 1>&) const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  if(size == 1)
  {
    return(0.0);
  }

  double sum = 0.0;
  double squared_sum = 0.0;

  for(int i = 0; i < size; i++)
  {
    sum += vector[i];
    squared_sum += vector[i] * vector[i];
  }

  const double numerator = squared_sum -(sum * sum) /static_cast<double>(size);
  const double denominator = size - 1.0;

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

double variance_missing_values(const Tensor<type, 1>& vector)
{
  const Index size = vector.dimension(0);

#ifdef __OPENNN_DEBUG__

  if(size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double variance_missing_values(const Tensor<type, 1>& vector) "
              "const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  double sum = 0.0;
  double squared_sum = 0.0;

  int count = 0;

  for(int i = 0; i < size; i++) {
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

double standard_deviation(const Tensor<type, 1>& vector)
{
#ifdef __OPENNN_DEBUG__

  const Index size = vector.dimension(0);

  if(size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double standard_deviation(const Tensor<type, 1>&) const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  return(sqrt(variance(vector)));
}


/// @todo check

Tensor<type, 1> standard_deviation(const Tensor<type, 1>& vector, const int& period)
{
  const Index size = vector.dimension(0);

  Tensor<type, 1> std(size);

  double mean_value = 0.0;
  double sum = 0.0;

  for(int i = 0; i < size; i++)
  {
      const int begin = i < period ? 0 : i - period + 1;
      const int end = i;

      mean_value = mean(vector, begin,end);

      for(int j = begin; j < end+1; j++)
      {
          sum += (vector[j] - mean_value) *(vector[j] - mean_value);
      }

      std[i] = sqrt(sum / double(period));

      mean_value = 0.0;
      sum = 0.0;
  }


  return std;
}


/// Returns the standard deviation of the elements in the vector.
/// @param vector

double standard_deviation_missing_values(const Tensor<type, 1>& vector)
{
#ifdef __OPENNN_DEBUG__

  const Index size = vector.dimension(0);

  if(size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double standard_deviation_missing_values(const Tensor<type, 1>& vector) const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  return sqrt(variance_missing_values(vector));
}


/// Returns the asymmetry of the elements in the vector
/// @param vector

double asymmetry(const Tensor<type, 1>& vector)
{
  const Index size = vector.dimension(0);

#ifdef __OPENNN_DEBUG__

  if(size == 0)
  {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double asymmetry(const Tensor<type, 1>& vector) const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  if(size == 1)
  {
    return 0.0;
  }

  const double standard_deviation_value = standard_deviation(vector);

  const double mean_value = mean(vector);

  double sum = 0.0;

  for(int i = 0; i < size; i++)
  {
    sum += (vector[i] - mean_value)*(vector[i] - mean_value)*(vector[i] - mean_value);
  }
  const double numerator = sum /static_cast<double>(size);
  const double denominator = standard_deviation_value * standard_deviation_value * standard_deviation_value;

  return numerator/denominator;
}


/// Returns the kurtosis value of the elements in the vector.
/// @param vector

double kurtosis(const Tensor<type, 1>& vector)
{
  const Index size = vector.dimension(0);

#ifdef __OPENNN_DEBUG__

  if(size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistic Class.\n"
           << "double kurtosis(const Tensor<type, 1>& vector) const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  if(size == 1) {
    return 0.0;
  }

  const double standard_deviation_value = standard_deviation(vector);

  const double mean_value = mean(vector);

  double sum = 0.0;

  for(int i = 0; i < size; i++)
  {
    sum += (vector[i] - mean_value)*(vector[i] - mean_value)*(vector[i] - mean_value)*(vector[i] - mean_value);
  }

  const double numerator = sum/static_cast<double>(size);
  const double denominator = standard_deviation_value*standard_deviation_value*standard_deviation_value*standard_deviation_value;

  return numerator/denominator - 3.0;
}


/// Returns the asymmetry of the elements in the vector.
/// @param vector


double asymmetry_missing_values(const Tensor<type, 1>& vector)
{
  const Index size = vector.dimension(0);
#ifdef __OPENNN_DEBUG__

  if(size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double asymmetry_missing_values(const Tensor<type, 1>& vector) const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  if(size == 1) {
    return 0.0;
  }

  const double standard_deviation_value = standard_deviation_missing_values(vector);

  const double mean_value = mean_missing_values(vector);

  double sum = 0.0;

  for(int i = 0; i < size; i++)
  {
    if(!::isnan(vector[i]))
    {
      sum += (vector[i] - mean_value) *(vector[i] - mean_value) *(vector[i] - mean_value);
    }
  }
/*
  const double numerator = sum /vector.count_not_NAN();
  const double denominator = standard_deviation_value * standard_deviation_value * standard_deviation_value;

  return numerator/denominator;
*/
    return 0.0;
}


/// Returns the kurtosis of the elements in the vector.
/// @param vector


double kurtosis_missing_values(const Tensor<type, 1>& vector)
{
  const Index size = vector.dimension(0);
#ifdef __OPENNN_DEBUG__

  if(size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double kurtosis_missing_values(const Tensor<type, 1>& vector) const method.\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  if(size == 1)
  {
    return 0.0;
  }

  const double standard_deviation_value = standard_deviation_missing_values(vector);

  const double mean_value = mean_missing_values(vector);

  double sum = 0.0;

  for(int i = 0; i < size; i++)
  {
      if(!::isnan(vector[i]))
    {
      sum += (vector[i] - mean_value)*(vector[i] - mean_value)*(vector[i] - mean_value)*(vector[i] - mean_value);
    }
  }
/*
  const double numerator = sum /vector.count_not_NAN();
  const double denominator = standard_deviation_value*standard_deviation_value*standard_deviation_value*standard_deviation_value;

  return numerator/denominator - 3.0;
*/
    return 0.0;
}


/// Returns the median of the elements in the vector

double median(const Tensor<type, 1>& vector)
{
  const Index size = vector.dimension(0);

  Tensor<type, 1> sorted_vector(vector);
/*
  sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

  int median_index;

  if(size % 2 == 0) {
    median_index = static_cast<int>(size / 2);

    return (sorted_vector[median_index-1] + sorted_vector[median_index]) / 2.0;
  } else {
    median_index = static_cast<int>(size / 2);

    return sorted_vector[median_index];
  }
*/
    return 0.0;
}


/// Returns the quarters of the elements in the vector.

Tensor<type, 1> quartiles(const Tensor<type, 1>& vector)
{
  const Index size = vector.dimension(0);

  Tensor<type, 1> sorted_vector(vector);
/*
  sort(sorted_vector.begin(), sorted_vector.end(), less<double>());

  Tensor<type, 1> quartiles(3);

  if(size == 1)
  {
      quartiles[0] = sorted_vector[0];
      quartiles[1] = sorted_vector[0];
      quartiles[2] = sorted_vector[0];
  }
  else if(size == 2)
  {
      quartiles[0] = (sorted_vector[0]+sorted_vector[1])/4;
      quartiles[1] = (sorted_vector[0]+sorted_vector[1])/2;
      quartiles[2] = (sorted_vector[0]+sorted_vector[1])*3/4;
  }
  else if(size == 3)
  {
      quartiles[0] = (sorted_vector[0]+sorted_vector[1])/2;
      quartiles[1] = sorted_vector[1];
      quartiles[2] = (sorted_vector[2]+sorted_vector[1])/2;
  }
  else if(size % 2 == 0)
  {
      quartiles[0] = median(sorted_vector.get_first(size/2));
      quartiles[1] = median(sorted_vector);
      quartiles[2] = median(sorted_vector.get_last(size/2));

  }
  else
  {
      quartiles[0] = sorted_vector[size/4];
      quartiles[1] = sorted_vector[size/2];
      quartiles[2] = sorted_vector[size*3/4];
  }

  return(quartiles);
*/
  return Tensor<type, 1>();
}


/// Returns the quartiles of the elements in the vector when there are missing values.

Tensor<type, 1> quartiles_missing_values(const Tensor<type, 1>& vector)
{
/*
    const Index size = vector.dimension(0);

    const int new_size = vector.count_not_NAN();

    Tensor<type, 1> new_vector(new_size);

    int index = 0;

    for(int i = 0; i < size; i++)
    {
        if(!isnan(vector[i]))
        {
             new_vector[index] = vector[i];

             index++;
        }
    }

    return quartiles(new_vector);
*/
    return Tensor<type, 1>();
}


/// Returns the box and whispers for a vector.

BoxPlot box_plot(const Tensor<type, 1>& vector)
{
    BoxPlot boxplot;
/*
    if(vector.empty()) return boxplot;

    const Tensor<type, 1> quartiles = OpenNN::quartiles(vector);

    boxplot.minimum = minimum(vector);
    boxplot.first_quartile = quartiles[0];
    boxplot.median = quartiles[1];
    boxplot.third_quartile = quartiles[2];
    boxplot.maximum = maximum(vector);
*/
    return boxplot;
}


/// Returns the box and whispers for a vector when there are missing values.

BoxPlot box_plot_missing_values(const Tensor<type, 1>& vector)
{
    BoxPlot boxplot;
/*
    if(vector.empty()) return boxplot;

    const Tensor<type, 1> quartiles = OpenNN::quartiles_missing_values(vector);

    boxplot.minimum = minimum_missing_values(vector);
    boxplot.first_quartile = quartiles[0];
    boxplot.median = quartiles[1];
    boxplot.third_quartile = quartiles[2];
    boxplot.maximum = maximum_missing_values(vector);
*/
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

Histogram histogram(const Tensor<type, 1>& vector, const int &bins_number)
{
#ifdef __OPENNN_DEBUG__

  if(bins_number < 1) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "Histogram histogram(const Tensor<type, 1>&, "
              "const int&) const method.\n"
           << "Number of bins is less than one.\n";

    throw logic_error(buffer.str());
  }

#endif

  Tensor<type, 1> minimums(bins_number);
  Tensor<type, 1> maximums(bins_number);
/*
  Tensor<type, 1> centers(bins_number);
  Tensor<int, 1> frequencies(bins_number, 0);

  const double min = minimum(vector);
  const double max = maximum(vector);

  const double length = (max - min) /static_cast<double>(bins_number);

  minimums[0] = min;
  maximums[0] = min + length;
  centers[0] = (maximums[0] + minimums[0]) / 2.0;

  // Calculate bins center

  for(int i = 1; i < bins_number; i++)
  {
    minimums[i] = minimums[i - 1] + length;
    maximums[i] = maximums[i - 1] + length;

    centers[i] = (maximums[i] + minimums[i]) / 2.0;
  }

  // Calculate bins frequency

  const Index size = vector.dimension(0);

  for(int i = 0; i < size; i++) {
    for(int j = 0; j < bins_number - 1; j++) {
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
*/
    return Histogram();
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


Histogram histogram_centered(const Tensor<type, 1>& vector, const double& center, const int & bins_number)
{
    #ifdef __OPENNN_DEBUG__

      if(bins_number < 1) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Statistics Class.\n"
               << "Histogram histogram_centered(const Tensor<type, 1>&, "
                  "const double&, const int&) const method.\n"
               << "Number of bins is less than one.\n";

        throw logic_error(buffer.str());
      }

    #endif

      int bin_center;

      if(bins_number%2 == 0)
      {
          bin_center = static_cast<int>(static_cast<double>(bins_number)/2.0);
      }
      else
      {
          bin_center = static_cast<int>(static_cast<double>(bins_number)/2.0+1.0/2.0);
      }
/*
      Tensor<type, 1> minimums(bins_number);
      Tensor<type, 1> maximums(bins_number);

      Tensor<type, 1> centers(bins_number);
      Tensor<int, 1> frequencies(bins_number, 0);

      const double min = minimum(vector);
      const double max = maximum(vector);

      const double length = (max - min)/static_cast<double>(bins_number);

      minimums[bin_center-1] = center - length;
      maximums[bin_center-1] = center + length;
      centers[bin_center-1] = center;

      // Calculate bins center

      for(int i = bin_center; i < bins_number; i++) // Upper centers
      {
        minimums[i] = minimums[i - 1] + length;
        maximums[i] = maximums[i - 1] + length;

        centers[i] = (maximums[i] + minimums[i]) / 2.0;
      }

      for(int i = static_cast<int>(bin_center)-2; i >= 0; i--) // Lower centers
      {
        minimums[static_cast<int>(i)] = minimums[static_cast<int>(i) + 1] - length;
        maximums[static_cast<int>(i)] = maximums[static_cast<int>(i) + 1] - length;

        centers[static_cast<int>(i)] = (maximums[static_cast<int>(i)] + minimums[static_cast<int>(i)]) / 2.0;
      }

      // Calculate bins frequency

      const Index size = vector.dimension(0);

      for(int i = 0; i < size; i++) {
        for(int j = 0; j < bins_number - 1; j++) {
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
*/
    return Histogram();
}


/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.

Histogram histogram(const Tensor<bool, 1>& vector)
{
/*
  const Tensor<int, 1> minimums(2, 0);
  const Tensor<int, 1> maximums(2, 1);

  const Tensor<int, 1> centers({0,1});
  Tensor<int, 1> frequencies(2, 0);

  // Calculate bins frequency

  const Index size = vector.dimension(0);

  for(int i = 0; i < size; i++)
  {
    for(int j = 0; j < 2; j++)
    {
      if(static_cast<int>(vector[i]) == minimums[j])
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
*/
    return Histogram();
}


/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.
/// @param vector
/// @param bins_number

Histogram histogram(const Tensor<int, 1>& vector, const int& bins_number)
{
    #ifdef __OPENNN_DEBUG__

      if(bins_number < 1) {
        ostringstream buffer;

        buffer << "OpenNN Exception: Statistics Class.\n"
               << "Histogram calculate_histogram_integers(const Tensor<int, 1>&, "
                  "const int&) const method.\n"
               << "Number of bins is less than one.\n";

        throw logic_error(buffer.str());
      }

    #endif
/*
    Tensor<int, 1> centers = vector.get_integer_elements(bins_number);
    const int centers_number = centers.size();

    sort(centers.begin(), centers.end(), less<int>());

    Tensor<type, 1> minimums(centers_number);
    Tensor<type, 1> maximums(centers_number);
    Tensor<int, 1> frequencies(centers_number);

    for(int i = 0; i < centers_number; i++)
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
*/
      return Histogram();
}


/// This method bins the elements of the vector into a given number of equally
/// spaced containers.
/// It returns a vector of two vectors.
/// The size of both subvectors is the number of bins.
/// The first subvector contains the frequency of the bins.
/// The second subvector contains the center of the bins.
/// @param vector
/// @param bins_number

Histogram histogram_missing_values(const Tensor<type, 1>& vector, const int &bins_number)
{
#ifdef __OPENNN_DEBUG__

  if(bins_number < 1) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistic Class.\n"
           << "Histogram histogram_missing_values(const Tensor<type, 1>&, const Tensor<int, 1>&) const method.\n"
           << "Number of bins is less than one.\n";

    throw logic_error(buffer.str());
  }

#endif


  const Index size = vector.dimension(0);
/*
  const int new_size = vector.count_not_NAN();

  Tensor<type, 1> new_vector(new_size);

  int index = 0;

  for(int i = 0; i < size; i++)
  {
      if(!::isnan(vector[i]))
      {
           new_vector[index] = vector[i];

           index++;
      }
   }

  return histogram(new_vector, bins_number);
*/
    return Histogram();
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
Histogram histogram_missing_values(const Tensor<bool, 1>& vector)
{
  Tensor<int, 1> minimums(2);
  Tensor<int, 1> maximums(2);

  Tensor<int, 1> centers(2);
  Tensor<int, 1> frequencies(2, 0);

  minimums[0] = 0;
  maximums[0] = 0;
  centers[0] = 0;

  minimums[1] = 1;
  maximums[1] = 1;
  centers[1] = 1;

  // Calculate bins frequency

  const Index size = vector.dimension(0);

  for(int i = 0; i < size; i++) {
    if(!missing_values.contains(i)) {
    for(int j = 0; j < 2; j++) {
      if(static_cast<int>(vector[i]) == minimums[j]) {
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

Tensor<int, 1> total_frequencies(const vector<Histogram>&histograms)
{
  const int histograms_number = histograms.size();

  Tensor<int, 1> total_frequencies(histograms_number);

  for(int i = 0; i < histograms_number; i++)
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

vector<Histogram> histograms(const Tensor<type, 2>& matrix, const int& bins_number)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

   vector<Histogram> histograms(columns_number);

   Tensor<type, 1> column(rows_number);
/*
   for(int i = 0; i < columns_number; i++)
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
*/
   return histograms;
}


/// Calculates a histogram for each column, each having a given number of bins, when the data has missing values.
/// It returns a vector of vectors.
/// The size of the main vector is the number of columns.
/// Each subvector contains the frequencies and centers of that colums.
/// @param bins_number Number of bins for each histogram.

vector<Histogram> histograms_missing_values(const Tensor<type, 2>& matrix, const int& bins_number)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

   vector<Histogram> histograms(columns_number);

   Tensor<type, 1> column(rows_number);
/*
   for(int i = 0; i < columns_number; i++)
   {
      column = matrix.get_column(i);

    histograms[i] = histogram_missing_values(column, bins_number);
   }
*/
   return histograms;
}


/// Returns the basic descriptives of the columns.
/// The format is a vector of descriptives structures.
/// The size of that vector is equal to the number of columns in this matrix.
/// @param matrix Used matrix.

vector<Descriptives> descriptives(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics Class.\n"
             << "vector<Descriptives> descriptives(const Tensor<type, 2>&) "
                "const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   #endif

   vector<Descriptives> descriptives(columns_number);

   Tensor<type, 1> column(rows_number);

    #pragma omp parallel for private(column)

   for(int i = 0; i < columns_number; i++)
   {
/*
      column = matrix.get_column(i);

      descriptives[i] = OpenNN::descriptives(column);

//      descriptives[i].name = matrix.get_header(i);
*/
   }

   return descriptives;
}


/// Returns the basic descriptives of the columns when the matrix has missing values.
/// The format is a vector of descriptives structures.
/// The size of that vector is equal to the number of columns in this matrix.
/// @param matrix Used matrix.

vector<Descriptives> descriptives_missing_values(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics Class.\n"
             << "vector<Descriptives> descriptives_missing_values(const Tensor<type, 2>&) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   #endif

   vector<Descriptives> descriptives(columns_number);

   Tensor<type, 1> column(rows_number);
/*
   for(int i = 0; i < columns_number; i++)
   {
      column = matrix.get_column(i);

      descriptives[i] = descriptives_missing_values(column);
   }
*/
   return descriptives;
}


vector<Descriptives> descriptives_missing_values(const Tensor<type, 2>& matrix,
                                                 const Tensor<int, 1>& rows_indices,
                                                 const Tensor<int, 1>& columns_indices)
{
    const int rows_size = rows_indices.size();
    const int columns_size = columns_indices.size();

   vector<Descriptives> descriptives(columns_size);

   Tensor<type, 1> column(rows_size);
/*
   for(int i = 0; i < columns_size; i++)
   {
      column = matrix.get_column(columns_indices[i], rows_indices);

      descriptives[i] = descriptives_missing_values(column);
   }
*/
   return descriptives;
}


/// Returns the basic descriptives of given columns for given rows.
/// The format is a vector of descriptives structures.
/// The size of that vector is equal to the number of given columns.
/// @param row_indices Indices of the rows for which the descriptives are to be computed.
/// @param columns_indices Indices of the columns for which the descriptives are to be computed.

vector<Descriptives> descriptives(const Tensor<type, 2>& matrix, const Tensor<int, 1>& row_indices, const Tensor<int, 1>& columns_indices)
{

    const int row_indices_size = row_indices.size();
    const int columns_indices_size = columns_indices.size();

    vector<Descriptives> descriptives(columns_indices_size);
/*
    int row_index, column_index;

    Tensor<type, 1> minimums(columns_indices_size);
    minimums.setConstant(999999);

    Tensor<type, 1> maximums;

    maximums.resize(columns_indices_size);
    maximums.setConstant(-999999);

    Tensor<type, 1> sums(columns_indices_size);
    Tensor<type, 1> squared_sums(columns_indices_size);

    for(int i = 0; i < row_indices_size; i++)
    {
        row_index = row_indices[i];

 #pragma omp parallel for private(column_index)

        for(int j = 0; j < columns_indices_size; j++)
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

    const Tensor<type, 1> mean = sums/static_cast<double>(row_indices_size);

    Tensor<type, 1> standard_deviation(columns_indices_size);

    if(row_indices_size > 1)
    {
        for(int i = 0; i < columns_indices_size; i++)
        {
            const double numerator = squared_sums[i] -(sums[i] * sums[i]) / row_indices_size;
            const double denominator = row_indices_size - 1.0;

            standard_deviation[i] = numerator / denominator;

            standard_deviation[i] = sqrt(standard_deviation[i]);
        }
    }

    for(int i = 0; i < columns_indices_size; i++)
    {
        descriptives[i].minimum = minimums[i];
        descriptives[i].maximum = maximums[i];
        descriptives[i].mean = mean[i];
        descriptives[i].standard_deviation = standard_deviation[i];
    }
*/
    return descriptives;
}


/// Returns the basic descriptives of all the columns for given rows when the matrix has missing values.
/// The format is a vector of descriptives structures.
/// The size of that vector is equal to the number of columns in this matrix.
/// @param matrix Used matrix.
/// @param row_indices Indices of the rows for which the descriptives are to be computed.

vector<Descriptives> rows_descriptives_missing_values(const Tensor<type, 2>& matrix, const Tensor<int, 1>& row_indices)
{
    const Index columns_number = matrix.dimension(1);

    const int row_indices_size = row_indices.size();

    vector<Descriptives> descriptives(columns_number);

    Tensor<type, 1> column(row_indices_size);
/*
    for(int i = 0; i < columns_number; i++)
    {
        column = matrix.get_column(i, row_indices);

        descriptives[i] = descriptives_missing_values(column);
    }
*/
    return descriptives;
}


/// Returns the means of given rows.
/// The format is a vector of double values.
/// The size of that vector is equal to the number of given rows.
/// @param matrix Used matrix.
/// @param columns_indices Indices of the columns for which the descriptives are to be computed.

Tensor<type, 1> rows_means(const Tensor<type, 2>& matrix, const Tensor<int, 1>& row_indices)
{
    const Index columns_number = matrix.dimension(1);

    Tensor<int, 1> used_row_indices;
/*
    if(row_indices.empty())
    {
        used_row_indices.resize(matrix.dimension(0));
        used_row_indices.initialize_sequential();
    }
    else
    {
        used_row_indices = row_indices;
    }
*/
    const int row_indices_size = used_row_indices.size();

    Tensor<type, 1> means(columns_number);

    Tensor<type, 1> column(row_indices_size);
/*
    for(int i = 0; i < columns_number; i++)
    {
        column = matrix.get_column(i, used_row_indices);

        means[i] = mean_missing_values(column);
    }
*/
    return means;
}


/// Returns the minimums values of given columns.
/// The format is a vector of double values.
/// The size of that vector is equal to the number of given columns.
/// @param matrix Used matrix.
/// @param columns_indices Indices of the columns for which the descriptives are to be computed.

Tensor<type, 1> columns_minimums(const Tensor<type, 2>& matrix, const Tensor<int, 1>& columns_indices)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    Tensor<int, 1> used_columns_indices;
/*
    if(columns_indices.empty())
    {
        used_columns_indices.resize(columns_number);
        used_columns_indices.initialize_sequential();
    }
    else
    {
        used_columns_indices = columns_indices;
    }
*/
    const int columns_indices_size = used_columns_indices.size();

    Tensor<type, 1> minimums(columns_indices_size);

    int index;
    Tensor<type, 1> column(rows_number);
/*
    for(int i = 0; i < columns_indices_size; i++)
    {
        index = used_columns_indices[i];

        column = matrix.get_column(index);

        minimums[i] = minimum(column);
    }
*/
    return minimums;
}


/// Returns the maximums values of given columns.
/// The format is a vector of double values.
/// The size of that vector is equal to the number of given columns.
/// @param matrix Used matrix.
/// @param columns_indices Indices of the columns for which the descriptives are to be computed.

Tensor<type, 1> columns_maximums(const Tensor<type, 2>& matrix, const Tensor<int, 1>& columns_indices)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    Tensor<int, 1> used_columns_indices;
/*
    if(columns_indices.empty())
    {
        used_columns_indices.resize(columns_number);
        used_columns_indices.initialize_sequential();
    }
    else
    {
        used_columns_indices = columns_indices;
    }
*/
    const int columns_indices_size = used_columns_indices.size();

    Tensor<type, 1> maximums(columns_indices_size);

    int index;
    Tensor<type, 1> column(rows_number);
/*
    for(int i = 0; i < columns_indices_size; i++)
    {
        index = used_columns_indices[i];

        column = matrix.get_column(index);

        maximums[i] = maximum(column);
    }
*/
    return maximums;
}


double range(const Tensor<type, 1>& vector)
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

vector<BoxPlot> box_plots(const Tensor<type, 2>& matrix, const vector<Tensor<int, 1>>& rows_indices, const Tensor<int, 1>& columns_indices)
{
    const int columns_number = columns_indices.size();

    #ifdef __OPENNN_DEBUG__

    if(columns_number == rows_indices.size())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Statistics class."
              << "void box_plots(const Tensor<type, 2>&, "
                 "const vector<Tensor<int, 1>>&, const Tensor<int, 1>&) const method.\n"
              << "Size of row indices must be equal to the number of columns.\n";

       throw logic_error(buffer.str());
    }

    #endif

    vector<BoxPlot> box_plots(columns_number);

    for(int i = 0; i < matrix.dimension(1); i++)
    {


    }
/*
    Tensor<type, 1> column;

     #pragma omp parallel for private(column)

    for(int i = 0; i < columns_number; i++)
    {
        box_plots[i].resize(5);

        const int rows_number = rows_indices[i].size();

        column = matrix.get_column(columns_indices[i]).get_subvector(rows_indices[i]);

        sort(column.begin(), column.end(), less<double>());

        // Minimum

        box_plots[static_cast<int>(i)][0] = column[0];

        if(rows_number % 2 == 0)
        {
            // First quartile

            box_plots[static_cast<int>(i)][1] = (column[rows_number / 4] + column[rows_number / 4 + 1]) / 2.0;

            // Second quartile

            box_plots[static_cast<int>(i)][2] = (column[rows_number * 2 / 4] +
                           column[rows_number * 2 / 4 + 1]) /
                          2.0;

            // Third quartile

            box_plots[static_cast<int>(i)][3] = (column[rows_number * 3 / 4] +
                           column[rows_number * 3 / 4 + 1]) /
                          2.0;
        }
        else
        {
            // First quartile

            box_plots[static_cast<int>(i)][1] = column[rows_number / 4];

            // Second quartile

            box_plots[static_cast<int>(i)][2] = column[rows_number * 2 / 4];

            //Third quartile

            box_plots[static_cast<int>(i)][3] = column[rows_number * 3 / 4];
        }

        // Maximum

        box_plots[static_cast<int>(i)][4] = column[rows_number-1];
    }
*/
    return box_plots;
}


/// Returns the minimum, maximum, mean and standard deviation of the elements in
/// the vector.
/// @param Used vector.

Descriptives descriptives(const Tensor<type, 1>& vector)
{
    const Index size = vector.dimension(0);

#ifdef __OPENNN_DEBUG__

  if(size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double descriptives(const Tensor<type, 1>&).\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  Descriptives descriptives;
  double minimum = 999999;
  double maximum;
  double sum = 0;
  double squared_sum = 0;
  int count = 0;

  maximum = -1.0*999999;

  for(int i = 0; i < size; i++)
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
      const double denominator = size - 1.0;

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

Descriptives descriptives_missing_values(const Tensor<type, 1>& vector)
{
    const Index size = vector.dimension(0);

#ifdef __OPENNN_DEBUG__

  if(size == 0) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Statistics Class.\n"
           << "double descriptives_missing_values(const Tensor<type, 1>&, "
              "const Tensor<int, 1>&).\n"
           << "Size must be greater than zero.\n";

    throw logic_error(buffer.str());
  }

#endif

  Descriptives descriptives;

  double minimum = 999999;
  double maximum;

  double sum = 0;
  double squared_sum = 0;
  int count = 0;

  maximum = -999999;

  for(int i = 0; i < size; i++) {
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
      const double denominator = size - 1.0;

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

int perform_distribution_distance_analysis(const Tensor<type, 1>& vector)
{
    Tensor<type, 1> distances(2);

    const int n = vector.dimension(0);

    Tensor<type, 1> sorted_vector(vector);
/*
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());
*/
    const Descriptives descriptives = OpenNN::descriptives(vector);

    const double mean = descriptives.mean;
    const double standard_deviation = descriptives.standard_deviation;
    const double minimum = sorted_vector[0];
    const double maximum = sorted_vector[n-1];

 #pragma omp parallel for schedule(dynamic)

    for(int i = 0; i < n; i++)
    {
        const double normal_distribution = 0.5 * erfc((mean - sorted_vector[i])/(standard_deviation*sqrt(2)));
/*        const double half_normal_distribution = erf((sorted_vector[i])/(standard_deviation * sqrt(2))); */
        const double uniform_distribution = (sorted_vector[i]-minimum)/(maximum-minimum);

        double empirical_distribution;

        int counter = 0;

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
            counter = static_cast<int>(i + 1);

            for(int j = i+1; j < n; j++)
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

int perform_distribution_distance_analysis_missing_values(const Tensor<type, 1>& vector, const Tensor<int, 1>& missing_indices)
{
/*
    Tensor<type, 1> distances(3, 0.0);

    double normal_distribution; // Normal distribution
    double half_normal_distribution; // Half-normal distribution
    double uniform_distribution; // Uniform distribution
    double empirical_distribution; // Empirical distribution

    Tensor<int, 1> used_indices(1,1, vector.size());
    used_indices = used_indices.get_difference(missing_indices);

    const Tensor<type, 1> used_values = vector.get_subvector(used_indices);
    const int n = used_values.size();

    Tensor<type, 1> sorted_vector(used_values);
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

    int counter = 0;

 #pragma omp parallel for private(empirical_distribution, normal_distribution, half_normal_distribution, uniform_distribution, counter)

    for(int i = 0; i < n; i++)
    {
        normal_distribution = 0.5 * erfc((mean - sorted_vector[i])/(standard_deviation*sqrt(2)));
        half_normal_distribution = erf((sorted_vector[i])/(standard_deviation * sqrt(2)));
        uniform_distribution = (sorted_vector[i]-minimum)/(maximum-minimum);
        counter = 0;

        for(int j = 0; j < n; j++)
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
*/
    return 0;
}


Tensor<type, 1> columns_mean(const Tensor<type, 2>& matrix)
{
/*
    const Index rows_number = matrix.dimension(0);

   return matrix.calculate_columns_sum()/static_cast<double>(rows_number);
*/
    return Tensor<type, 1>();
}


/// Returns a vector with the mean values of all the matrix columns.
/// The size is equal to the number of columns in the matrix.
/// @param matrix Matrix used.

Tensor<type, 1> mean(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "Tensor<type, 1> mean(const Tensor<type, 2>&) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Mean

   Tensor<type, 1> mean(columns_number);

   for(int j = 0; j < columns_number; j++)
   {
      for(int i = 0; i < rows_number; i++)
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

Tensor<type, 1> mean(const Tensor<type, 2>& matrix, const Tensor<int, 1>& columns_indices)
{
    const Index rows_number = matrix.dimension(0);

   const int columns_indices_size = columns_indices.size();

   int column_index;

   // Mean

   Tensor<type, 1> mean(columns_indices_size);
   mean.setZero();

   for(int j = 0; j < columns_indices_size; j++)
   {
      column_index = columns_indices[j];

      for(int i = 0; i < rows_number; i++)
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

Tensor<type, 1> mean(const Tensor<type, 2>& matrix, const Tensor<int, 1>& row_indices, const Tensor<int, 1>& columns_indices)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);


   const int row_indices_size = row_indices.size();
   const int columns_indices_size = columns_indices.size();



   #ifdef __OPENNN_DEBUG__

   // Rows check

   if(row_indices_size > rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "Tensor<type, 1> mean(const Tensor<type, 2>& matrix, "
                "const Tensor<int, 1>&, const Tensor<int, 1>&) const method.\n"
             << "Size of row indices(" << row_indices_size << ") is greater than number of rows(" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   for(int i = 0; i < row_indices_size; i++)
   {
      if(row_indices[i] >= rows_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Statistics class.\n"
                << "Tensor<type, 1> mean(const Tensor<type, 2>& matrix, "
                   "const Tensor<int, 1>&, const Tensor<int, 1>&) const method.\n"
                << "Row index " << i << " must be less than rows number.\n";

         throw logic_error(buffer.str());
      }
   }

   if(row_indices_size == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "Tensor<type, 1> mean(const Tensor<type, 2>& matrix, "
                "const Tensor<int, 1>&, const Tensor<int, 1>&) const method.\n"
             << "Size of row indices must be greater than zero.\n";

      throw logic_error(buffer.str());
   }

   // Columns check

   if(columns_indices_size > columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "Tensor<type, 1> mean(const Tensor<type, 2>& matrix, "
                "const Tensor<int, 1>&, const Tensor<int, 1>&) const method.\n"
             << "Column indices size must be equal or less than columns number.\n";

      throw logic_error(buffer.str());
   }

   for(int i = 0; i < columns_indices_size; i++)
   {
      if(columns_indices[i] >= columns_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Statistics class.\n"
                << "Tensor<type, 1> mean(const Tensor<type, 2>& matrix, "
                   "const Tensor<int, 1>&, const Tensor<int, 1>&) const method.\n"
                << "Column index " << i << " must be less than columns number.\n";

         throw logic_error(buffer.str());
      }
   }

   #endif

   int row_index;
   int column_index;

   // Mean

   Tensor<type, 1> mean(columns_indices_size);
   mean.setZero();

   for(int j = 0; j < columns_indices_size; j++)
   {
      column_index = columns_indices[j];

      for(int i = 0; i < row_indices_size; i++)
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
double mean(const Tensor<type, 2>& matrix, const int& column_index)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "double mean(const Tensor<type, 2>&, const int&) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   if(column_index >= columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "double mean(const Tensor<type, 2>&, const int&) const method.\n"
             << "Index of column must be less than number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Mean

   double mean = 0.0;

    for(int i = 0; i < rows_number; i++)
    {
        mean += matrix(i,column_index);
    }

   mean /= static_cast<double>(rows_number);

   return mean;
}


/// Returns a vector with the mean values of given columns for given rows when the matrix has missing values.
/// The size of the vector is equal to the size of the column indices vector.

Tensor<type, 1> mean_missing_values(const Tensor<type, 2>& matrix)
{
/*
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    const Tensor<int, 1> row_indices(0, 1, rows_number-1);
    const Tensor<int, 1> columns_indices(0, 1, columns_number-1);

    return mean_missing_values(matrix, row_indices, columns_indices);
*/
    return Tensor<type, 1>();
}


/// Returns a vector with the mean values of given columns for given rows when the matrix has missing values.
/// The size of the vector is equal to the size of the column indices vector.
/// @param row_indices Indices of rows.
/// @param columns_indices Indices of columns.

Tensor<type, 1> mean_missing_values(const Tensor<type, 2>& matrix, const Tensor<int, 1>& row_indices, const Tensor<int, 1>& columns_indices)
{

   const int columns_indices_size = columns_indices.size();

   #ifdef __OPENNN_DEBUG__

   const Index rows_number = matrix.dimension(0);
   const Index columns_number = matrix.dimension(1);

   const int row_indices_size = row_indices.size();

   // Rows check

   if(row_indices_size > rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "Tensor<type, 1> mean_missing_values(const Tensor<type, 2>&, "
                "const Tensor<int, 1>&, const Tensor<int, 1>&, const vector<Tensor<int, 1>>&) const method.\n"
             << "Size of row indices(" << row_indices_size << ") is greater than number of rows(" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   for(int i = 0; i < row_indices_size; i++)
   {
      if(row_indices[i] >= rows_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Statistics class.\n"
                << "Tensor<type, 1> mean_missing_values(const Tensor<type, 2>&, "
                   "const Tensor<int, 1>&, const Tensor<int, 1>&, const vector<Tensor<int, 1>>&) const method.\n"
                << "Row index " << i << " must be less than rows number.\n";

         throw logic_error(buffer.str());
      }
   }

   if(row_indices_size == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Statistics class.\n"
             << "Tensor<type, 1> mean_missing_values(const Tensor<type, 2>&, "
                "const Tensor<int, 1>&, const Tensor<int, 1>&, const vector<Tensor<int, 1>>&) const method.\n"
             << "Size of row indices must be greater than zero.\n";

      throw logic_error(buffer.str());
   }

   // Columns check

   if(columns_indices_size > columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Tensor<type, 1> mean_missing_values(const Tensor<int, 1>&, const Tensor<int, 1>&, const vector<Tensor<int, 1>>&) const method.\n"
             << "Column indices size must be equal or less than columns number.\n";

      throw logic_error(buffer.str());
   }

   for(int i = 0; i < columns_indices_size; i++)
   {
      if(columns_indices[i] >= columns_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Tensor<type, 1> mean_missing_values(const Tensor<int, 1>&, const Tensor<int, 1>&, const vector<Tensor<int, 1>>&) const method.\n"
                << "Column index " << i << " must be less than columns number.\n";

         throw logic_error(buffer.str());
      }
   }

   #endif

   Tensor<type, 1> mean(columns_indices_size);
   mean.setZero();
/*
   for(int j = 0; j < columns_indices_size; j++)
   {
       const int column_index = columns_indices[j];

       Tensor<type, 1> column_missing_values(matrix.get_column(column_index, row_indices));

       mean[j] = mean_missing_values(column_missing_values);
   }
*/
   return mean;
}


/// Returns a vector with the median values of all the matrix columns.
/// The size is equal to the number of columns in the matrix.

Tensor<type, 1> median(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Tensor<type, 1> median() const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // median

   Tensor<type, 1> median(columns_number);
/*
   for(int j = 0; j < columns_number; j++)
   {
       Tensor<type, 1> sorted_column(matrix.get_column(j));

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
*/
   return median;
}


/// Returns a vector with the median values of all the matrix columns.
/// The size is equal to the number of columns in the matrix.

double median(const Tensor<type, 2>& matrix, const int& column_index)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

   #ifdef __OPENNN_DEBUG__

   if(rows_number == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "double median(const int&) const method.\n"
             << "Number of rows must be greater than one.\n";

      throw logic_error(buffer.str());
   }

   if(column_index >= columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "double median(const int&) const method.\n"
             << "Index of column must be less than number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   // median

   double median = 0.0;
/*
   Tensor<type, 1> sorted_column(matrix.get_column(column_index));

   sort(sorted_column.begin(), sorted_column.end(), less<double>());

   if(rows_number % 2 == 0)
   {
     median = (sorted_column[rows_number*2/4] + sorted_column[rows_number*2/4+1])/2;
   }
   else
   {
     median = sorted_column[rows_number*2/4];
   }
*/
   return median;
}


/// Returns a vector with the median values of given columns.
/// The size of the vector is equal to the size of the column indices vector.
/// @param columns_indices Indices of columns.


Tensor<type, 1> median(const Tensor<type, 2>& matrix, const Tensor<int, 1>& columns_indices)
{
    const Index rows_number = matrix.dimension(0);

   const int columns_indices_size = columns_indices.size();

   int column_index;

   // median

   Tensor<type, 1> median(columns_indices_size);
/*
   for(int j = 0; j < columns_indices_size; j++)
   {
      column_index = columns_indices[j];

      Tensor<type, 1> sorted_column(matrix.get_column(column_index));

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
*/
   return median;
}


/// Returns a vector with the median values of given columns for given rows.
/// The size of the vector is equal to the size of the column indices vector.
/// @param row_indices Indices of rows.
/// @param columns_indices Indices of columns.

Tensor<type, 1> median(const Tensor<type, 2>& matrix, const Tensor<int, 1>& row_indices, const Tensor<int, 1>& columns_indices)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

   const int row_indices_size = row_indices.size();
   const int columns_indices_size = columns_indices.size();

   #ifdef __OPENNN_DEBUG__

   // Rows check

   if(row_indices_size > rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Tensor<type, 1> median(const Tensor<int, 1>&, const Tensor<int, 1>&) const method.\n"
             << "Size of row indices(" << row_indices_size << ") is greater than number of rows(" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   for(int i = 0; i < row_indices_size; i++)
   {
      if(row_indices[i] >= rows_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Tensor<type, 1> median(const Tensor<int, 1>&, const Tensor<int, 1>&) const method.\n"
                << "Row index " << i << " must be less than rows number.\n";

         throw logic_error(buffer.str());
      }
   }

   if(row_indices_size == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Tensor<type, 1> median(const Tensor<int, 1>&, const Tensor<int, 1>&) const method.\n"
             << "Size of row indices must be greater than zero.\n";

      throw logic_error(buffer.str());
   }

   // Columns check

   if(columns_indices_size > columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Tensor<type, 1> median(const Tensor<int, 1>&, const Tensor<int, 1>&) const method.\n"
             << "Column indices size must be equal or less than columns number.\n";

      throw logic_error(buffer.str());
   }

   for(int i = 0; i < columns_indices_size; i++)
   {
      if(columns_indices[i] >= columns_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Tensor<type, 1> median(const Tensor<int, 1>&, const Tensor<int, 1>&) const method.\n"
                << "Column index " << i << " must be less than columns number.\n";

         throw logic_error(buffer.str());
      }
   }

   #endif

   int column_index;

   // median

   Tensor<type, 1> median(columns_indices_size);
/*
   for(int j = 0; j < columns_indices_size; j++)
   {
      column_index = columns_indices[j];

      Tensor<type, 1> sorted_column(matrix.get_column(column_index, row_indices));

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
*/
   return median;
}

/// Returns the median of the elements of a vector with missing values.

double median_missing_values(const Tensor<type, 1>& vector)
{
/*
  const Index size = vector.dimension(0);

  const int nan = vector.count_NAN();

  const int new_size = size - nan;

  Tensor<type, 1> new_vector(new_size);

  int index = 0;

  for(int i = 0; i < size; i++){

      if(!::isnan(vector[i]))
      {
           new_vector[index] = vector[i];

           index++;

       }
     }

  return median(new_vector);
*/
    return 0.0;
}


/// Returns a vector with the median values of given columns for given rows when the matrix has missing values.
/// The size of the vector is equal to the size of the column indices vector.

Tensor<type, 1> median_missing_values(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);
/*
    Tensor<int, 1> row_indices(0, 1, rows_number-1);
    Tensor<int, 1> columns_indices(0, 1, columns_number-1);

    return median_missing_values(matrix, row_indices, columns_indices);
*/

    return Tensor<type, 1>();
}


/// Returns a vector with the median values of given columns for given rows when the matrix has missing values.
/// The size of the vector is equal to the size of the column indices vector.
/// @param row_indices Indices of rows.
/// @param columns_indices Indices of columns.

Tensor<type, 1> median_missing_values(const Tensor<type, 2>& matrix,
                                     const Tensor<int, 1>& row_indices,
                                     const Tensor<int, 1>& columns_indices)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);
    const int columns_indices_size = columns_indices.size();

   #ifdef __OPENNN_DEBUG__

   const int row_indices_size = row_indices.size();

   // Rows check

   if(row_indices_size > rows_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Tensor<type, 1> median_missing_values(const Tensor<int, 1>&, const Tensor<int, 1>&, const vector<Tensor<int, 1>>&) const method.\n"
             << "Size of row indices(" << row_indices_size << ") is greater than number of rows(" << rows_number << ").\n";

      throw logic_error(buffer.str());
   }

   for(int i = 0; i < row_indices_size; i++)
   {
      if(row_indices[i] >= rows_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Tensor<type, 1> median_missing_values(const Tensor<int, 1>&, const Tensor<int, 1>&, vector<Tensor<int, 1>>&) const method.\n"
                << "Row index " << i << " must be less than rows number.\n";

         throw logic_error(buffer.str());
      }
   }

   if(row_indices_size == 0)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Tensor<type, 1> median_missing_values(const Tensor<int, 1>&, const Tensor<int, 1>&, const vector<Tensor<int, 1>>&) const method.\n"
             << "Size of row indices must be greater than zero.\n";

      throw logic_error(buffer.str());
   }

   // Columns check

   if(columns_indices_size > columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Matrix template.\n"
             << "Tensor<type, 1> median_missing_values(const Tensor<int, 1>&, const Tensor<int, 1>&, const vector<Tensor<int, 1>>&) const method.\n"
             << "Column indices size must be equal or less than columns number.\n";

      throw logic_error(buffer.str());
   }

   for(int i = 0; i < columns_indices_size; i++)
   {
      if(columns_indices[i] >= columns_number)
      {
         ostringstream buffer;

         buffer << "OpenNN Exception: Matrix template.\n"
                << "Tensor<type, 1> median_missing_values(const Tensor<int, 1>&, const Tensor<int, 1>&, const vector<Tensor<int, 1>>&) const method.\n"
                << "Column index " << i << " must be less than columns number.\n";

         throw logic_error(buffer.str());
      }
   }

   #endif

   // median

   Tensor<type, 1> median(columns_indices_size);
/*
   for(int j = 0; j < columns_indices_size; j++)
   {
      const int column_index = columns_indices[j];

      Tensor<type, 1> column_missing_values(matrix.get_column(column_index, row_indices));

      median[j] = median_missing_values(column_missing_values);
   }
*/
   return median;
}


/// Returns true if the elements in the vector have a normal distribution with a given critical value.
/// @param critical_value Critical value to be used in the test.

bool perform_Lilliefors_normality_test(const Tensor<type, 1>& vector, const double& critical_value)
{
#ifndef Cpp11__

    const int n = vector.dimension(0);

    const double mean = OpenNN::mean(vector);
    const double standard_deviation = OpenNN::standard_deviation(vector);

    Tensor<type, 1> sorted_vector(vector);
/*
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());
*/
    double Fx;
    double Snx;

    double D = -1;

    for(int i = 0; i < n; i++)
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
            for(int j = 0; j < n-1; j++)
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

Tensor<bool, 1> perform_Lilliefors_normality_test(const Tensor<type, 1>& vector, const Tensor<type, 1>& critical_values)
{
    const int size = critical_values.size();

    Tensor<bool, 1> normality_tests(size);

    for(int i = 0; i < size; i++)
    {
        normality_tests[i] = perform_Lilliefors_normality_test(vector, critical_values[i]);
    }

    return normality_tests;
}


/// Calculates the distance between the empirical distribution of the vector and the
/// normal distribution.

double normal_distribution_distance(const Tensor<type, 1>& vector)
{
    double normal_distribution_distance = 0.0;

    const int n = vector.dimension(0);

    const double mean_value = mean(vector);
    const double standard_deviation = OpenNN::standard_deviation(vector);

    double normal_distribution; // Normal distribution
    double empirical_distribution; // Empirical distribution

    Tensor<type, 1> sorted_vector(vector);
/*
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());
*/
    int counter = 0;

    for(int i = 0; i < n; i++)
    {
        normal_distribution = 0.5 * erfc((mean_value - sorted_vector[i])/(standard_deviation*sqrt(2.0)));
        counter = 0;

        for(int j = 0; j < n; j++)
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

double half_normal_distribution_distance(const Tensor<type, 1>& vector)
{
    double half_normal_distribution_distance = 0.0;

    const int n = vector.dimension(0);

    const double standard_deviation = OpenNN::standard_deviation(vector);

    double half_normal_distribution; // Half normal distribution
    double empirical_distribution; // Empirical distribution

    Tensor<type, 1> sorted_vector(vector);
/*
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());
*/
    int counter = 0;

    for(int i = 0; i < n; i++)
    {
        half_normal_distribution = erf((sorted_vector[i])/(standard_deviation * sqrt(2)));
        counter = 0;

        for(int j = 0; j < n; j++)
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

double uniform_distribution_distance(const Tensor<type, 1>& vector)
{
    double uniform_distribution_distance = 0.0;

    const int n = vector.dimension(0);

    double uniform_distribution; // Uniform distribution
    double empirical_distribution; // Empirical distribution

    Tensor<type, 1> sorted_vector(vector);
/*
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());
*/
    const double minimum = sorted_vector[0];
    const double maximum = sorted_vector[n-1];

    int counter = 0;

    for(int i = 0; i < n; i++)
    {
        uniform_distribution = (sorted_vector[i]-minimum)/(maximum-minimum);
        counter = 0;

        for(int j = 0; j < n; j++)
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

Tensor<bool, 1> perform_normality_analysis(const Tensor<type, 1>& vector)
{
    const Index size = vector.dimension(0);

    double significance_level = 0.05;

    double A_significance_level;
    double B_significance_level;
    Tensor<type, 1> critical_values(9);

    for(int i = 0; i < 9; i++)
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

double normality_parameter(const Tensor<type, 1>& vector)
{
    const double max = maximum(vector);
    const double min = minimum(vector);

    const int n = vector.dimension(0);

    const double mean_value = mean(vector);
    const double standard_deviation = OpenNN::standard_deviation(vector);

    double normal_distribution;
    double empirical_distribution;
    double previous_normal_distribution = 0.0;
    double previous_empirical_distribution = 0.0;

    Tensor<type, 1> sorted_vector(vector);
/*
    sort(sorted_vector.begin(), sorted_vector.end(), less<double>());
*/
    double empirical_area = 0.0;
    double normal_area = 0.0;

    int counter = 0;

    for(int i = 0; i < n; i++)
    {
        normal_distribution = 0.5 * erfc((mean_value - sorted_vector[i])/(standard_deviation*sqrt(2.0)));
        counter = 0;

        for(int j = 0; j < n; j++)
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


Tensor<type, 1> variation_percentage(const Tensor<type, 1>& vector)
{
    const Index size = vector.dimension(0);

    Tensor<type, 1> new_vector(size);

    for(int i = 1; i < size; i++)
    {
        if(abs(vector[i-1]) < 1.0e-99)
        {
            new_vector[i] = (vector[i] - vector[i-1])*100.0/vector[i-1];
        }
    }

    return new_vector;
}


/// Returns the index of the smallest element in the vector.

int minimal_index(const Tensor<type, 1>& vector)
{
    const Index size = vector.dimension(0);

    if(size == 0) return int();

    int minimal_index = 0;
    double minimum = vector[0];

    for(int i = 1; i < size; i++)
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

int maximal_index(const Tensor<type, 1>& vector)
{
    const Index size = vector.dimension(0);

    if(size == 0) return int();

    int maximal_index = 0;
    double maximum = vector[0];

    for(int i = 1; i < size; i++)
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

Tensor<int, 1> minimal_indices(const Tensor<type, 1>& vector, const int &number)
{
/*
  const Index size = vector.dimension(0);

  const std::Tensor<int, 1> rank = vector.calculate_less_rank();

  std::Tensor<int, 1> minimal_indices(number);

   #pragma omp parallel for

  for(int i = 0; i < static_cast<int>(size); i++)
  {
    for(int j = 0; j < number; j++)
    {
      if(rank[static_cast<int>(i)] == j)
      {
        minimal_indices[j] = static_cast<int>(i);
      }
    }
  }

  return minimal_indices;
*/
  return Tensor<int, 1>();
}


/// Returns the indices of the largest elements in the vector.
/// @param number Number of maximal indices to be computed.

Tensor<int, 1> maximal_indices(const Tensor<type, 1>& vector, const int& number)
{
/*
  const Index size = vector.dimension(0);

  const Tensor<int, 1> rank = vector.calculate_greater_rank();

  Tensor<int, 1> maximal_indices(number);

  for(int i = 0; i < size; i++)
  {
    for(int j = 0; j < number; j++)
    {
      if(rank[i] == j)
      {
        maximal_indices[j] = i;
      }
    }
  }

  return maximal_indices;
*/

    return Tensor<int, 1>();
}


/// Returns the row and column indices corresponding to the entry with minimum value.

Tensor<int, 1> minimal_indices(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

   double minimum = matrix(0,0);
   Tensor<int, 1> minimal_indices(2);

   for(int i = 0; i < rows_number; i++)
   {
      for(int j = 0; j < columns_number; j++)
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


Tensor<int, 1> minimal_indices_omit(const Tensor<type, 2>& matrix, const double& value_to_omit)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

   double minimum = 999999;

   Tensor<int, 1> minimal_indices(2);

   for(int i = 0; i < rows_number; i++)
   {
      for(int j = 0; j < columns_number; j++)
      {
         if(abs(matrix(i,j) - value_to_omit) < 1.0e-99 && matrix(i,j) < minimum)
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

Tensor<int, 1> maximal_indices(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

   double maximum = matrix(0,0);

   Tensor<int, 1> maximal_indices(2);

   for(int i = 0; i < rows_number; i++)
   {
      for(int j = 0; j < columns_number; j++)
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


Tensor<int, 1> maximal_indices_omit(const Tensor<type, 2>& matrix, const double& value_to_omit)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

   double maximum = 1.0e-99;

   Tensor<int, 1> maximum_indices(2);

   for(int i = 0; i < rows_number; i++)
   {
      for(int j = 0; j < columns_number; j++)
      {
         if(abs(matrix(i,j) - value_to_omit) < 1.0e-99 && matrix(i,j) > maximum)
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

double minimum_matrix(const Tensor<type, 2>& matrix)
{
   double minimum = static_cast<double>(999999);
/*
   for(int i = 0; i < matrix.size(); i++)
   {
         if(matrix[i] < minimum)
         {
            minimum = matrix[i];
         }
   }
*/
   return minimum;
}


/// Returns the maximum value from all elements in the matrix.

double maximum_matrix(const Tensor<type, 2>& matrix)
{
    double maximum = static_cast<double>(-999999);
/*
    for(int i = 0; i < matrix.size(); i++)
    {
          if(matrix[i] > maximum)
          {
             maximum = matrix[i];
          }
    }
*/
   return maximum;
}


double strongest(const Tensor<type, 1>& vector)
{
    const Index size = vector.dimension(0);

    if(size == 0) return 0.0;

    double strongest = vector[0];

    for(int i = 0; i < size; i++)
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

Tensor<type, 1> means_by_categories(const Tensor<type, 2>& matrix)
{
/*
    const int integers_number = matrix.size();
    Tensor<type, 1> elements_uniques = matrix.get_column(0).get_unique_elements();
    Tensor<type, 1> values = matrix.get_column(1);

    #ifdef __OPENNN_DEBUG__

    if(integers_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "vector<T> calculate_means_integers(const Tensor<type, 2>& \n"
              << "Number of integers must be greater than 0.\n";

       throw logic_error(buffer.str());
    }

    #endif

    const Index rows_number = matrix.dimension(0);

    Tensor<type, 1> means(elements_uniques);

    double sum = 0.0;
    int count = 0;

    for(int i = 0; i < integers_number; i++)
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
*/
    return Tensor<type, 1>();
}


/// Returns a vector containing the means of the subsets which correspond
/// to each of the given integers. The matrix must have 2 columns, the first
/// one containing the integers and the second one the corresponding values.
/// Both columns can contain NAN.

Tensor<type, 1> means_by_categories_missing_values(const Tensor<type, 2>& matrix)
{
/*
    const int integers_number = matrix.size();

    Tensor<type, 1> elements_uniques = matrix.get_column(0).get_unique_elements();
    Tensor<type, 1> values = matrix.get_column(1);

    #ifdef __OPENNN_DEBUG__

    if(integers_number == 0)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Matrix template.\n"
              << "vector<T> calculate_means_integers(const Tensor<type, 2>& \n"
              << "Number of integers must be greater than 0.\n";

       throw logic_error(buffer.str());
    }

    #endif

    const Index rows_number = matrix.dimension(0);

    Tensor<type, 1> means(elements_uniques);

    double sum = 0.0;
    int count = 0;

    for(int i = 0; i < integers_number; i++)
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
*/

    return Tensor<type, 1>();
}


/// Returns a vector containing the values of the means for the 0s and 1s of a
/// binary column.
/// The matrix must have 2 columns, the first one has to be binary.

Tensor<type, 1> means_binary_column(const Tensor<type, 2>& matrix)
{
    Tensor<type, 1> means(2);
    means.setZero();

    int count = 0;

    for(int i = 0; i < matrix.dimension(0); i++)
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

Tensor<type, 1> means_binary_columns(const Tensor<type, 2>& matrix)
{
    Tensor<type, 1> means(matrix.dimension(1)-1);

    double sum = 0.0;
    int count = 0;

    for(int i = 0; i < matrix.dimension(1)-1; i++)
    {
        sum = 0.0;
        count = 0;

        for(int j = 0; j < matrix.dimension(0); j++)
        {
            if(matrix(j,i) == 1.0)
            {
                sum += matrix(j,matrix.dimension(1)-1);

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

Tensor<type, 1> means_binary_columns_missing_values(const Tensor<type, 2>& matrix)
{
/*
   return means_binary_columns(matrix.delete_rows_with_value(static_cast<double>(NAN)));
*/
    return Tensor<type, 1>();
}


///Returns a vector with the percentiles of a vector given.

Tensor<type, 1> percentiles(const Tensor<type, 1>& vector)
{
/*
  const Index size = vector.dimension(0);

  const Tensor<int, 1> sorted_vector = vector.sort_ascending_indices();

  Tensor<type, 1> percentiles(10);

  if(size % 2 == 0)
  {
    percentiles[0] = (sorted_vector[size * 1 / 10] + sorted_vector[size * 1 / 10 + 1]) / 2.0;
    percentiles[1] = (sorted_vector[size * 2 / 10] + sorted_vector[size * 2 / 10 + 1]) / 2.0;
    percentiles[2] = (sorted_vector[size * 3 / 10] + sorted_vector[size * 3 / 10 + 1]) / 2.0;
    percentiles[3] = (sorted_vector[size * 4 / 10] + sorted_vector[size * 4 / 10 + 1]) / 2.0;
    percentiles[4] = (sorted_vector[size * 5 / 10] + sorted_vector[size * 5 / 10 + 1]) / 2.0;
    percentiles[5] = (sorted_vector[size * 6 / 10] + sorted_vector[size * 6 / 10 + 1]) / 2.0;
    percentiles[6] = (sorted_vector[size * 7 / 10] + sorted_vector[size * 7 / 10 + 1]) / 2.0;
    percentiles[7] = (sorted_vector[size * 8 / 10] + sorted_vector[size * 8 / 10 + 1]) / 2.0;
    percentiles[8] = (sorted_vector[size * 9 / 10] + sorted_vector[size * 9 / 10 + 1]) / 2.0;
    percentiles[9] = maximum(vector);
  }
  else
  {
    percentiles[0] = static_cast<double>(sorted_vector[size * 1 / 10]);
    percentiles[1] = static_cast<double>(sorted_vector[size * 2 / 10]);
    percentiles[2] = static_cast<double>(sorted_vector[size * 3 / 10]);
    percentiles[3] = static_cast<double>(sorted_vector[size * 4 / 10]);
    percentiles[4] = static_cast<double>(sorted_vector[size * 5 / 10]);
    percentiles[5] = static_cast<double>(sorted_vector[size * 6 / 10]);
    percentiles[6] = static_cast<double>(sorted_vector[size * 7 / 10]);
    percentiles[7] = static_cast<double>(sorted_vector[size * 8 / 10]);
    percentiles[8] = static_cast<double>(sorted_vector[size * 9 / 10]);
    percentiles[9] = maximum(vector);
  }

  return percentiles;
*/
    return Tensor<type, 1>();
}


Tensor<type, 1> percentiles_missing_values(const Tensor<type, 1>& x)
{
    const int size = x.size();

    int new_size;

    Tensor<type, 1> new_x(new_size);

    int index = 0;

    for(int i = 0; i < size ; i++)
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

double weighted_mean(const Tensor<type, 1>& vector, const Tensor<type, 1>& weights)
{
    const Index size = vector.dimension(0);

  #ifdef __OPENNN_DEBUG__

    if(size == 0) {
      ostringstream buffer;

      buffer << "OpenNN Exception: vector Template.\n"
             << "double calculate_weighted_mean(const Tensor<type, 1>&) const method.\n"
             << "Size must be greater than zero.\n";

      throw logic_error(buffer.str());
    }

    const int weights_size = weights.size();

    if(size != weights_size) {
      ostringstream buffer;

      buffer << "OpenNN Exception: vector Template.\n"
             << "double calculate_weighted_mean(const Tensor<type, 1>&) "
                "const method.\n"
             << "Size of weights must be equal to vector size.\n";

      throw logic_error(buffer.str());
    }
  #endif

    double weights_sum = 0;

    double sum = 0;

    for(int i = 0; i < size; i++)
    {
        sum += weights[i]*vector[i];
        weights_sum += weights[i];
    }

    const double mean = sum / weights_sum;

    return mean;
}


/// Calculates the explained variance for a given vector(principal components analysis).
/// This method returns a vector whose size is the same as the size of the given vector.

Tensor<type, 1> explained_variance(const Tensor<type, 1>& vector)
{
    const Index size = vector.dimension(0);

    #ifdef __OPENNN_DEBUG__

      if(size == 0) {
        ostringstream buffer;

        buffer << "OpenNN Exception: vector Template.\n"
               << "vector<T> explained_variance() const method.\n"
               << "Size of vector must be greater than zero.\n";

        throw logic_error(buffer.str());
      }

    #endif
/*
    const double this_sum = absolute_value(vector).calculate_sum();

    #ifdef __OPENNN_DEBUG__

      if(abs(this_sum) < 1.0e-99)
      {
        ostringstream buffer;

        buffer << "OpenNN Exception: vector Template.\n"
               << "vector<T> explained_variance() const method.\n"
               << "Sum of the members of the vector (" << abs(this_sum) << ") must be greater than zero.\n";

        throw logic_error(buffer.str());
      }

    #endif

    #ifdef __OPENNN_DEBUG__

      if(this_sum < 0) {
        ostringstream buffer;

        buffer << "OpenNN Exception: vector Template.\n"
               << "vector<T> explained_variance() const method.\n"
               << "Sum of the members of the vector cannot be negative.\n";

        throw logic_error(buffer.str());
      }

    #endif
*/
    Tensor<type, 1> explained_variance(size);
/*
    for(int i = 0; i < size; i++)
    {
        explained_variance[i] = vector[i]*100.0/this_sum;

        if(explained_variance[i] - 0.0 < 1.0e-16)
        {
            explained_variance[i] = 0.0;
        }
    }
*/
/*
    #ifdef __OPENNN_DEBUG__

      if(explained_variance.calculate_sum() != 1.0) {
        ostringstream buffer;

        buffer << "OpenNN Exception: vector Template.\n"
               << "vector<T> explained_variance() const method.\n"
               << "Sum of explained variance must be 1.\n";

        throw logic_error(buffer.str());
      }

    #endif
*/
    return explained_variance;
}

}
