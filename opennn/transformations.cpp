//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A N S F O R M A T I O N S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "transformations.h"

namespace OpenNN {


/// Normalizes the elements of this vector using the minimum and maximum method.
/// @param vector Vector to be scaled.
/// @param descriptives Descriptives structure, which contains the minimum and
/// maximum values for the scaling.

void scale_minimum_maximum(Vector<double>& vector, const Descriptives &descriptives)
{
  scale_minimum_maximum(vector, descriptives.minimum, descriptives.maximum);
}


void scale_minimum_maximum(Vector<double>& vector, const double& minimum, const double& maximum)
{
    const size_t size = vector.size();

    for(size_t i = 0; i < size; i++)
    {
        if(maximum - minimum <= 0.0)
        {
            vector[i] = 0.0;
        }
        else
        {
            vector[i] = (vector[i] - minimum)/(maximum - minimum);
        }
    }
}


/// Normalizes the elements of the vector with the minimum and maximum method.
/// The minimum and maximum values used are those calculated from the vector.
/// It also returns the descriptives from the vector.
/// @param vector Vector with values to be scaled(inputs,targets).

 Descriptives scale_minimum_maximum(Vector<double>& vector)
 {
  const Descriptives descriptives = OpenNN::descriptives(vector);

  scale_minimum_maximum(vector, descriptives);

  return descriptives;
}


/// Normalizes the elements of this vector using the mean and standard deviation
/// method.
/// @param vector Vector with values to be scaled(inputs,targets).
/// @param mean Mean value for the scaling.
/// @param standard_deviation Standard deviation value for the scaling.

void scale_mean_standard_deviation(Vector<double>& vector, const double &mean, const double &standard_deviation) {

  if(standard_deviation < numeric_limits<double>::min()) return;

  const size_t this_size = vector.size();

  for(size_t i = 0; i < this_size; i++)
  {
    vector[i] = (vector[i] - mean) / standard_deviation;
  }
}


/// Normalizes the elements of this vector using the mean and standard deviation
/// method.
/// @param vector Vector with values to be scaled(inputs,targets).
/// @param descriptives Descriptives structure,
/// which contains the mean and standard deviation values for the scaling.

void scale_mean_standard_deviation(Vector<double>& vector, const Descriptives &descriptives)
{
  scale_mean_standard_deviation(vector, descriptives.mean, descriptives.standard_deviation);
}


/// Normalizes the elements of the vector with the mean and standard deviation
/// method.
/// @param vector Vector with values to be scaled(inputs,targets).
/// The values used are those calculated from the vector.
/// It also returns the descriptives from the vector.

Descriptives scale_mean_standard_deviation(Vector<double>& vector)
{
  const Descriptives _descriptives = descriptives(vector);

  scale_mean_standard_deviation(vector, _descriptives);

  return _descriptives;
}


/// Normalizes the elements of this vector using standard deviationmethod.
/// @param vector Vector with values to be scaled(inputs,targets).
/// @param standard_deviation Standard deviation value for the scaling.

void scale_standard_deviation(Vector<double>& vector, const double &standard_deviation)
{
  if(standard_deviation < numeric_limits<double>::min()) {
    return;
  }

  const size_t this_size = vector.size();

  for(size_t i = 0; i < this_size; i++) {
   vector[i] = vector[i] / standard_deviation;
  }
}


/// Normalizes the elements of this vector using standard deviation method.
/// @param vector Vector to be scaled.
/// @param descriptives Descriptives structure,
/// which contains standard deviation value for the scaling.

void scale_standard_deviation(Vector<double>& vector, const Descriptives &descriptives)
{
  scale_standard_deviation(vector, descriptives.standard_deviation);
}


/// Normalizes the elements of the vector with the standard deviation method.
/// @param vector Vector with values to be scaled(inputs,targets,...).
/// The values used are those calculated from the vector.
/// It also returns the descriptives from the vector.

Descriptives scale_standard_deviation(Vector<double>& vector)
{
  const Descriptives _descriptives = descriptives(vector);

  scale_standard_deviation(vector, _descriptives);

  return _descriptives;
}


/// Scales the vector elements with given standard deviation values.
/// It updates the data in the vector.
/// The size of the standard deviation vector must be equal to the
/// size of the vector.
/// @param vector Vector with values to be scaled(inputs,targets,...).
/// @param standard_deviation Standard deviation values.

void scale_standard_deviation(Vector<double>& vector, const Vector<double>&standard_deviation)
{
  const size_t this_size = vector.size();

#ifdef __OPENNN_DEBUG__

  const size_t standard_deviation_size = standard_deviation.size();

  if(standard_deviation_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Transformations."
           << "void scale_standard_deviation(const Vector<double>&, const "
              "Vector<double>&) method.\n"
           << "Size of standard deviation vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

#endif

  // Rescale data

  for(size_t i = 0; i < this_size; i++) {
    if(standard_deviation[i] < numeric_limits<double>::min()) {
//      cout << "OpenNN Warning: Vector class.\n"
//                << "void scale_mean_standard_deviation(const Vector<double>&, const "
//                   "Vector<double>&) method.\n"
//                << "Standard deviation of variable " << i << " is zero.\n"
//                << "Those elements won't be scaled.\n";

      // Do nothing
    } else {
       vector[i] = vector[i] / standard_deviation[i];
    }
  }
}


/// Unscales the vector elements with given minimum and maximum values.
/// It updates the vector elements.
/// The size of the minimum and maximum vectors must be equal to the size of the
/// vector.
/// @param vector Vector with values to be unscaled(inputs,targets,...).
/// @param minimum Minimum values.
/// @param maximum Maximum deviation values.

void unscale_minimum_maximum(Vector<double>& vector, const Vector<double>&minimum, const Vector<double>&maximum)
{
  const size_t this_size = vector.size();

#ifdef __OPENNN_DEBUG__

  const size_t minimum_size = minimum.size();

  if(minimum_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Transformations."
           << "void unscale_minimum_maximum(const Vector<double>&, const "
              "Vector<double>&) method.\n"
           << "Size of minimum vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

  const size_t maximum_size = maximum.size();

  if(maximum_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Transformations."
           << "void unscale_minimum_maximum(const Vector<double>&, const "
              "Vector<double>&) method.\n"
           << "Size of maximum vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

#endif

  for(size_t i = 0; i < this_size; i++) {
    if(maximum[i] - minimum[i] < numeric_limits<double>::min()) {
      cout << "OpenNN Warning: Transformations.\n"
                << "void unscale_minimum_maximum(const Vector<double>&, const "
                   "Vector<double>&) method.\n"
                << "Minimum and maximum values of variable " << i
                << " are equal.\n"
                << "Those elements won't be unscaled.\n";

      // Do nothing
    } else {
     vector[i] = 0.5 *(vector[i] + 1.0) *(maximum[i] - minimum[i]) + minimum[i];
    }
  }
}


/// Unscales the vector elements with given mean and standard deviation values.
/// It updates the vector elements.
/// The size of the mean and standard deviation vectors must be equal to the
/// size of the vector.
/// @param vector Vector with values to be unscaled(inputs,targets,...).
/// @param mean Mean values.
/// @param standard_deviation Standard deviation values.

void unscale_mean_standard_deviation(Vector<double>& vector,
                                                const Vector<double>&mean, const Vector<double>&standard_deviation)
{
  const size_t this_size = vector.size();

#ifdef __OPENNN_DEBUG__

  const size_t mean_size = mean.size();

  if(mean_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Transformations."
           << "void unscale_mean_standard_deviation(const Vector<double>&, const "
              "Vector<double>&) method.\n"
           << "Size of mean vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

  const size_t standard_deviation_size = standard_deviation.size();

  if(standard_deviation_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Transformations.\n"
           << "void unscale_mean_standard_deviation(const Vector<double>&, const "
              "Vector<double>&) method.\n"
           << "Size of standard deviation vector must be equal to size.\n";

    throw logic_error(buffer.str());
  }

#endif

  for(size_t i = 0; i < this_size; i++) {
    if(standard_deviation[i] < numeric_limits<double>::min()) {
      cout << "OpenNN Warning: Transformations.\n"
                << "void unscale_mean_standard_deviation(const Vector<double>&, "
                   "const Vector<double>&) method.\n"
                << "Standard deviation of variable " << i << " is zero.\n"
                << "Those elements won't be scaled.\n";

      // Do nothing
    } else {
     vector[i] = vector[i] * standard_deviation[i] + mean[i];
    }
  }
}


/// Scales the matrix elements with the mean and standard deviation method.
/// It updates the data in the matrix.
/// @param matrix Matrix with values to be scaled(inputs,targets,...).
/// @param descriptives Vector of descriptives structures conatining the mean and standard deviation values for the scaling.
/// The size of that vector must be equal to the number of columns in this matrix.

void scale_mean_standard_deviation(Matrix<double>& matrix, const Vector<Descriptives>& descriptives)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   #ifdef __OPENNN_DEBUG__

   const size_t size = descriptives.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Transformations."
             << "void scale_mean_standard_deviation(const Vector<Descriptives>&) const method.\n"
             << "Size of descriptives (" << size << ") must be equal to number of columns (" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Rescale data

   for(size_t j = 0; j < columns_number; j++)
   {
      if(descriptives[j].standard_deviation < numeric_limits<double>::min())
      {
         // Do nothing
      }
      else
      {
         for(size_t i = 0; i < rows_number; i++)
         {
           matrix(i,j) = (matrix(i,j) - descriptives[j].mean)/descriptives[j].standard_deviation;
         }
      }
   }
}


/// Unscales given rows using the minimum and maximum method.
/// @param matrix Matrix with values to be unscaled(inputs,targets,...).
/// @param descriptives Vector of descriptives structures for all the columns.
/// The size of this vector must be equal to the number of columns.
/// @param row_indices Indices of rows to be unscaled.

void unscale_rows_minimum_maximum(Matrix<double>& matrix, const Vector<Descriptives>& descriptives, const Vector<size_t>& row_indices)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    size_t row_index;

    // Unscale rows

    for(size_t j = 0; j < columns_number; j++)
    {
       if(descriptives[j].maximum - descriptives[j].minimum < numeric_limits<double>::min())
       {
          // Do nothing
       }
       else
       {
          for(size_t i = 0; i < rows_number; i++)
          {
              row_index = row_indices[i];

            matrix(row_index,j) = 0.5*(matrix(row_index,j) + 1.0)*(descriptives[j].maximum-descriptives[j].minimum)
             + descriptives[j].minimum;
          }
       }
    }
}


/// Unscales given columns in the matrix with the minimum and maximum method.
/// @param matrix Matrix with values to be unscaled(inputs,targets,...).
/// @param descriptives Vector of descriptives structures containing the minimum and maximum values for the unscaling.
/// The size of that vector must be less or equal to the number of columns in the matrix.
/// @param columns_indices Vector of indices of the columns to be unscaled.
/// The size of that vector must be equal to the number of columns to be unscaled.

void unscale_columns_minimum_maximum(Matrix<double>& matrix,
                                     const Vector<Descriptives>& descriptives,
                                     const Vector<size_t>& columns_indices)
{
    const size_t rows_number = matrix.get_rows_number();

    #ifdef __OPENNN_DEBUG__

    const size_t size = descriptives.size();

    const size_t columns_number = columns_indices.size();

    if(size != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Transformations.\n"
              << "void unscale_columns_minimum_maximum(const Matrix<double>&, const Vector<Descriptives>&, const Vector<size_t>&) const method.\n"
              << "Size of descriptives (" << size << ") must be equal to number of columns (" << columns_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    // Unscale columns

    for(size_t j = 0; j < columns_indices.size(); j++)
    {
       const size_t column_index = columns_indices[j];

       if(descriptives[j].maximum - descriptives[j].minimum > 0.0)
       {
          for(size_t i = 0; i < rows_number; i++)
          {
            matrix(i,column_index) = 0.5*(matrix(i,column_index) + 1.0)*(descriptives[j].maximum-descriptives[j].minimum)
                                     + descriptives[j].minimum;
          }
       }
    }
}


/// Unscales the matrix columns with the logarithmic method.
/// @param matrix Matrix with values to be unscaled(inputs,targets,...).
/// @param descriptives Vector of descriptives which contains the minimum and maximum scaling values.
/// The size of that vector must be equal to the number of columns in this matrix.

void unscale_logarithmic(Matrix<double>& matrix, const Vector<Descriptives>& descriptives)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   #ifdef __OPENNN_DEBUG__

   const size_t size = descriptives.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Transformations."
             << "void unscale_logarithmic(const Vector<Descriptives>&) method.\n"
             << "Size of minimum vector must be equal to number of columns.\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t j = 0; j < columns_number; j++)
   {
      if(descriptives[j].maximum - descriptives[j].minimum < numeric_limits<double>::min())
      {
         cout << "OpenNN Warning: Transformations.\n"
                   << "void unscale_minimum_maximum(const Vector<Descriptives>&) const method.\n"
                   << "Minimum and maximum values of column " << j << " are equal.\n"
                   << "Those columns won't be unscaled.\n";

         // Do nothing
      }
      else
      {
         for(size_t i = 0; i < rows_number; i++)
         {
           matrix(i,j) = 0.5*(exp(matrix(i,j)))*(descriptives[j].maximum-descriptives[j].minimum) + descriptives[j].minimum;
         }
      }
   }
}


/// Unscales given rows using the logarithimic method.
/// @param matrix Matrix with values to be unscaled(inputs,targets,...).
/// @param descriptives Vector of descriptives structures for all the columns.
/// The size of this vector must be equal to the number of columns.
/// @param row_indices Indices of rows to be unscaled.

void unscale_rows_logarithmic(Matrix<double>& matrix,
                                               const Vector<Descriptives>& descriptives,
                                               const Vector<size_t>& row_indices)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    size_t row_index;

    // Unscale rows

    for(size_t j = 0; j < columns_number; j++)
    {
       if(descriptives[j].maximum - descriptives[j].minimum < numeric_limits<double>::min())
       {
          // Do nothing
       }
       else
       {
          for(size_t i = 0; i < rows_number; i++)
          {
              row_index = row_indices[i];

            matrix(row_index,j) = 0.5*(exp(matrix(row_index,j)))*(descriptives[j].maximum-descriptives[j].minimum)
             + descriptives[j].minimum;
          }
       }
    }
}


/// Unscales given columns in the matrix with the logarithmic method.
/// @param matrix Matrix with values to be unscaled(inputs,targets,...).
/// @param descriptives Vector of descriptives structures containing the minimum and maximum values for the unscaling.
/// The size of that vector must be equal to the number of columns in the matrix.
/// @param columns_indices Vector of indices of the columns to be unscaled.
/// The size of that vector must be equal to the number of columns to be unscaled.

void unscale_columns_logarithmic(Matrix<double>& matrix, const Vector<Descriptives>& descriptives, const Vector<size_t>& columns_indices)
{
    const size_t columns_number = matrix.get_columns_number();
    const size_t rows_number = matrix.get_rows_number();

    #ifdef __OPENNN_DEBUG__

    const size_t size = descriptives.size();

    if(size != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Transformations.\n"
              << "void unscale_columns_logarithmic(const Vector<Descriptives>&, const Vector<size_t>&) const method.\n"
              << "Size of descriptives (" << size << ") must be equal to number of columns (" << columns_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    size_t column_index;

    // Unscale columns

    for(size_t j = 0; j < columns_indices.size(); j++)
    {
        column_index = columns_indices[j];

       if(descriptives[column_index].maximum - descriptives[column_index].minimum < numeric_limits<double>::min())
       {
          // Do nothing
       }
       else
       {
          for(size_t i = 0; i < rows_number; i++)
          {
            matrix(i,column_index) = 0.5*(exp(matrix(i,column_index)))*(descriptives[column_index].maximum-descriptives[column_index].minimum)
             + descriptives[column_index].minimum;
          }
       }
    }
}


/// Scales the data using the mean and standard deviation method and
/// the mean and standard deviation values calculated from the matrix.
/// It also returns the descriptives of all the columns.
/// @param matrix Matrix with values to be scaled(inputs,targets,...).

Vector<Descriptives> scale_mean_standard_deviation(Matrix<double>& matrix)
{
    const Vector<Descriptives> _descriptives = descriptives(matrix);

    scale_mean_standard_deviation(matrix, _descriptives);

    return _descriptives;
}


/// Scales given rows from the matrix using the mean and standard deviation method.
/// @param matrix Matrix with values to be scaled(inputs,targets,...).
/// @param descriptives Vector of descriptives for all the columns.
/// @param row_indices Indices of rows to be scaled.

void scale_rows_mean_standard_deviation(Matrix<double>& matrix,
                                                         const Vector<Descriptives>& descriptives,
                                                         const Vector<size_t>& row_indices)
{
    const size_t columns_number = matrix.get_columns_number();

    #ifdef __OPENNN_DEBUG__

    const size_t size = descriptives.size();

    if(size != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Transformations.\n"
              << "void scale_rows_mean_standard_deviation(const Vector<Descriptives>&, const Vector<size_t>&) method.\n"
              << "Size of descriptives (" << size << ") must be equal to number of columns (" << columns_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    size_t row_index;

    // Scale columns

    for(size_t j = 0; j < columns_number; j++)
    {
       if(descriptives[j].standard_deviation < numeric_limits<double>::min())
       {
          // Do nothing
       }
       else
       {
          for(size_t i = 0; i < row_indices.size(); i++)
          {
             row_index = row_indices[i];

            matrix(row_index,j) = (matrix(row_index,j) - descriptives[j].mean)/descriptives[j].standard_deviation;
          }
       }
    }
}


/// Scales given columns of this matrix with the mean and standard deviation method.
/// @param matrix Matrix with values to be scaled(inputs,targets,...).
/// @param descriptives Vector of descriptives structure containing the mean and standard deviation values for the scaling.
/// The size of that vector must be equal to the number of columns to be scaled.
/// @param columns_indices Vector of indices with the columns to be scaled.
/// The size of that vector must be equal to the number of columns to be scaled.

void scale_columns_mean_standard_deviation(Matrix<double>& matrix,
                                           const Vector<Descriptives>& descriptives,
                                           const Vector<size_t>& columns_indices)
{
    const size_t rows_number = matrix.get_rows_number();

   const size_t columns_indices_size = columns_indices.size();

   #ifdef __OPENNN_DEBUG__

   const size_t descriptives_size = descriptives.size();

   if(descriptives_size != columns_indices_size)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Transformations.\n"
             << "void scale_columns_mean_standard_deviation(const Vector<Descriptives>&, const Vector<size_t>&) method.\n"
             << "Size of descriptives must be equal to size of columns indices.\n";

      throw logic_error(buffer.str());
   }

   #endif

   size_t column_index;

   // Scale columns

   for(size_t j = 0; j < columns_indices_size; j++)
   {
      if(descriptives[j].standard_deviation < numeric_limits<double>::min())
      {
         // Do nothing
      }
      else
      {
         column_index = columns_indices[j];

         for(size_t i = 0; i < rows_number; i++)
         {
           matrix(i,column_index) = (matrix(i,column_index) - descriptives[j].mean)/descriptives[j].standard_deviation;
         }
      }
   }
}


/// Scales the matrix columns with the minimum and maximum method.
/// It updates the data in the matrix.
/// @param matrix Matrix with values to be scaled(inputs,targets,...).
/// @param descriptives Vector of descriptives structures containing the minimum and maximum values for the scaling.
/// The size of that vector must be equal to the number of columns in this matrix.

void scale_minimum_maximum(Matrix<double>& matrix, const Vector<Descriptives>& descriptives)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   #ifdef __OPENNN_DEBUG__

   const size_t size = descriptives.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Transformations."
             << "void scale_minimum_maximum(const Vector<Descriptives>&) method.\n"
             << "Size of descriptives (" << size << ") must be equal to number of columns (" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Rescale data

   for(size_t j = 0; j < columns_number; j++)
   {
      if(descriptives[j].maximum - descriptives[j].minimum < numeric_limits<double>::min())
      {
            // Do nothing
      }
      else
      {
         for(size_t i = 0; i < rows_number; i++)
         {
           matrix(i,j) = 2.0*(matrix(i,j) - descriptives[j].minimum)/(descriptives[j].maximum-descriptives[j].minimum)-1.0;
         }
      }
   }
}


/// Scales the matrix columns with the range method.
/// It updates the data in the matrix.
/// @param matrix Matrix with values to be scaled(inputs,targets,...).
/// @param descriptives Vector of descriptives structures containing the minimum and maximum values for the scaling.
/// The size of that vector must be equal to the number of columns in this matrix.
/// @param minimum Minimum values.
/// @param maximum Maximum deviation values.

void scale_range(Matrix<double>& matrix,
                                  const Vector<Descriptives>& descriptives,
                                  const double& minimum, const double& maximum)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   #ifdef __OPENNN_DEBUG__

   const size_t size = descriptives.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Transformations."
             << "void scale_minimum_maximum(const Vector<Descriptives>&) method.\n"
             << "Size of descriptives (" << size << ") must be equal to number of columns (" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Rescale data

   for(size_t j = 0; j < columns_number; j++)
   {
      if(descriptives[j].maximum - descriptives[j].minimum < numeric_limits<double>::min())
      {
          for(size_t i = 0; i < rows_number; i++)
          {
               matrix(i,j) = 0.0;
          }
      }
      else
      {
         for(size_t i = 0; i < rows_number; i++)
         {
           matrix(i,j) = (maximum-minimum)*(matrix(i,j) - descriptives[j].minimum)/(descriptives[j].maximum-descriptives[j].minimum)+minimum;
         }
      }
   }
}


/// Scales the data using the minimum and maximum method and
/// the minimum and maximum values calculated from the matrix.
/// It also returns the descriptives of all the columns.
/// @param matrix Matrix with values to be scaled(inputs,targets,...).

Vector<Descriptives> scale_minimum_maximum(Matrix<double>& matrix)
{
    const Vector<Descriptives> _descriptives = OpenNN::descriptives(matrix);

    scale_minimum_maximum(matrix, _descriptives);

    return _descriptives;
}

/// Scales the data using the range method.
/// It also returns the descriptives of all the columns.
/// @param matrix Matrix with values to be scaled(inputs,targets,...).
/// @param minimum Minimum values.
/// @param maximum Maximum deviation values.

Vector<Descriptives> scale_range(Matrix<double>& matrix, const double& minimum, const double& maximum)
{
    const Vector<Descriptives> _descriptives = descriptives(matrix);

    scale_range(matrix, _descriptives, minimum, maximum);

    return _descriptives;
}


/// Scales given rows from the matrix using the minimum and maximum method.
/// @param matrix Matrix with values to be scaled(inputs,targets,...).
/// @param descriptives Vector of descriptives for all the columns.
/// @param row_indices Indices of rows to be scaled.

void scale_rows_minimum_maximum(Matrix<double>& matrix, const Vector<Descriptives>& descriptives, const Vector<size_t>& row_indices)
{
    const size_t columns_number = matrix.get_columns_number();

    

    const size_t row_indices_size = row_indices.size();

    #ifdef __OPENNN_DEBUG__

    const size_t size = descriptives.size();

    if(size != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Transformations.\n"
              << "void scale_rows_minimum_maximum(const Vector<Descriptives>&, const Vector<size_t>&) method.\n"
              << "Size of descriptives (" << size << ") must be equal to number of columns (" << columns_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    // Rescale data

    size_t row_index;

    for(size_t j = 0; j < columns_number; j++)
    {
       if(descriptives[j].maximum - descriptives[j].minimum < numeric_limits<double>::min())
       {
          // Do nothing
       }
       else
       {
          for(size_t i = 0; i < row_indices_size; i++)
          {
             row_index = row_indices[i];

            matrix(row_index,j) = 2.0*(matrix(row_index,j) - descriptives[j].minimum)/(descriptives[j].maximum-descriptives[j].minimum) - 1.0;
          }
       }
    }
}


/// Scales given columns of this matrix with the minimum and maximum method.
/// @param matrix Matrix with values to be scaled(inputs,targets,...).
/// @param descriptives Vector of descriptives structure containing the minimum and maximum values for the scaling.
/// The size of that vector must be equal to the number of columns to be scaled.
/// @param columns_indices Vector of indices with the columns to be scaled.
/// The size of that vector must be equal to the number of columns to be scaled.

void scale_columns_minimum_maximum(Matrix<double>& matrix,
                                   const Vector<Descriptives>& descriptives,
                                   const Vector<size_t>& columns_indices)
{
    const size_t rows_number = matrix.get_rows_number();

    const size_t columns_indices_size = columns_indices.size();

    #ifdef __OPENNN_DEBUG__

    const size_t descriptives_size = descriptives.size();

    if(descriptives_size != columns_indices_size)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Transformations.\n"
              << "void scale_columns_minimum_maximum(Matrix<double>&, const Vector<Descriptives>&, const Vector<size_t>&) method.\n"
              << "Size of descriptives (" << descriptives_size << ") must be equal to size of columns indices (" << columns_indices_size << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    size_t column_index;

    // Rescale data

    for(size_t j = 0; j < columns_indices_size; j++)
    {
       column_index = columns_indices[j];

       if(descriptives[j].maximum - descriptives[j].minimum > 0.0)
       {

      for(int i = 0; i < static_cast<int>(rows_number); i++)
      {
        matrix(static_cast<size_t>(i),column_index) =
                2.0*(matrix(static_cast<size_t>(i),column_index) - descriptives[j].minimum)/(descriptives[j].maximum-descriptives[j].minimum) - 1.0;
      }
       }
    }
}


/// Scales the matrix columns with the logarithimic method.
/// It updates the data in the matrix.
/// @param matrix Matrix with values to be scaled(inputs,targets,...).
/// @param descriptives Vector of descriptives structures containing the minimum and maximum values for the scaling.
/// The size of that vector must be equal to the number of columns in this matrix.

void scale_logarithmic(Matrix<double>& matrix, const Vector<Descriptives>& descriptives)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   #ifdef __OPENNN_DEBUG__

   const size_t size = descriptives.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Transformations."
             << "void scale_logarithmic(const Vector<Descriptives>&) method.\n"
             << "Size of descriptives (" << size << ") must be equal to number of columns (" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   // Rescale data

   for(size_t j = 0; j < columns_number; j++)
   {
      if(descriptives[j].maximum - descriptives[j].minimum < numeric_limits<double>::min())
      {
            // Do nothing
      }
      else
      {
         for(size_t i = 0; i < rows_number; i++)
         {
           matrix(i,j) = log(1.0+ (2.0*(matrix(i,j) - descriptives[j].minimum)/(descriptives[j].maximum-descriptives[j].minimum)));
         }
      }
   }
}


/// Scales the data using the logarithmic method and
/// the minimum and maximum values calculated from the matrix.
/// It also returns the descriptives of all the columns.
/// @param matrix Matrix with values to be scaled(inputs,targets,...).

Vector<Descriptives> scale_logarithmic(Matrix<double>& matrix)
{
    const Vector<Descriptives> descriptives = OpenNN::descriptives(matrix);

    scale_logarithmic(matrix, descriptives);

    return descriptives;
}


/// Scales given rows from the matrix using the logarithmic method.
/// @param matrix Matrix with values to be scaled(inputs,targets,...).
/// @param descriptives Vector of descriptives for all the columns.
/// @param row_indices Indices of rows to be scaled.

void scale_rows_logarithmic(Matrix<double>& matrix, const Vector<Descriptives>& descriptives, const Vector<size_t>& row_indices)
{
    const size_t columns_number = matrix.get_columns_number();

    const size_t row_indices_size = row_indices.size();

    #ifdef __OPENNN_DEBUG__

    const size_t size = descriptives.size();

    if(size != columns_number)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Transformations.\n"
              << "void scale_rows_logarithmic(const Vector<Descriptives>&, const Vector<size_t>&) method.\n"
              << "Size of descriptives (" << size << ") must be equal to number of columns (" << columns_number << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    // Rescale data

    size_t row_index;

    for(size_t j = 0; j < columns_number; j++)
    {
       if(descriptives[j].maximum - descriptives[j].minimum < numeric_limits<double>::min())
       {
          // Do nothing
       }
       else
       {
          for(size_t i = 0; i < row_indices_size; i++)
          {
             row_index = row_indices[i];

            matrix(row_index,j) = log(1.0+ (2.0*(matrix(row_index,j) - descriptives[j].minimum)/(descriptives[j].maximum-descriptives[j].minimum)));
          }
       }
    }
}


/// Scales given columns of this matrix with the logarithmic method.
/// @param matrix Matrix with values to be scaled(inputs,targets,...).
/// @param descriptives Vector of descriptives structure containing the minimum and maximum values for the scaling.
/// The size of that vector must be equal to the number of columns to be scaled.
/// @param columns_indices Vector of indices with the columns to be scaled.
/// The size of that vector must be equal to the number of columns to be scaled.

void scale_columns_logarithmic(Matrix<double>& matrix,
                                                const Vector<Descriptives>& descriptives,
                                                const Vector<size_t>& columns_indices)
{

    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    

    const size_t columns_indices_size = columns_indices.size();

    #ifdef __OPENNN_DEBUG__

    const size_t descriptives_size = descriptives.size();

    if(descriptives_size != columns_indices_size)
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Transformations.\n"
              << "void scale_columns_logarithmic(const Vector<Descriptives>&, const Vector<size_t>&) method.\n"
              << "Size of descriptives must be equal to size of columns indices.\n";

       throw logic_error(buffer.str());
    }

    #endif

    size_t column_index;

    // Rescale data

    for(size_t j = 0; j < columns_indices_size; j++)
    {
       column_index = columns_indices[j];

       if(descriptives[j].maximum - descriptives[j].minimum < numeric_limits<double>::min())
       {
          // Do nothing
       }
       else
       {

#pragma omp parallel for
          for(int i = 0; i < static_cast<int>(rows_number); i++)
          {
            matrix(i,column_index) = log(1.0+ (2.0*(matrix(i,column_index) - descriptives[j].minimum)/(descriptives[j].maximum-descriptives[j].minimum)));
          }
       }
    }
}


/// Unscales the matrix columns with the mean and standard deviation method.
/// It updates the matrix elements.
/// @param matrix Matrix with values to be unscaled(inputs,targets,...).
/// @param descriptives Vector of descriptives structures containing the mean and standard deviations for the unscaling.
/// The size of that vector must be equal to the number of columns in this matrix.

void unscale_mean_standard_deviation(Matrix<double>& matrix, const Vector<Descriptives>& descriptives)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   #ifdef __OPENNN_DEBUG__

   const size_t size = descriptives.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Transformations."
             << "void unscale_mean_standard_deviation(const Vector<Descriptives>&) const method.\n"
             << "Size of descriptives (" << size << ") must be equal to number of columns (" << columns_number << ").\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t j = 0; j < columns_number; j++)
   {
      if(descriptives[j].standard_deviation < numeric_limits<double>::min())
      {
         // Do nothing
      }
      else
      {
         for(size_t i = 0; i < rows_number; i++)
         {
           matrix(i,j) = matrix(i,j)*descriptives[j].standard_deviation + descriptives[j].mean;
         }
      }
   }
}


/// Unscales given rows using the mean and standard deviation method.
/// @param matrix Matrix with values to be unscaled(inputs,targets,...).
/// @param descriptives Vector of descriptives structures for all the columns.
/// The size of this vector must be equal to the number of columns.
/// @param row_indices Indices of rows to be unscaled.

void unscale_rows_mean_standard_deviation(Matrix<double>& matrix, const Vector<Descriptives>& descriptives, const Vector<size_t>& row_indices)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    size_t row_index;

    // Unscale columns

    for(size_t j = 0;  j < columns_number; j++)
    {
       if(descriptives[j].standard_deviation < numeric_limits<double>::min())
       {
          // Do nothing
       }
       else
       {
          for(size_t i = 0; i < rows_number; i++)
          {
             row_index = row_indices[i];

            matrix(row_index,j) = matrix(row_index,j)*descriptives[j].standard_deviation + descriptives[j].mean;
          }
       }
    }
}


/// Unscales given columns of this matrix with the mean and standard deviation method.
/// @param matrix Matrix with values to be unscaled(inputs,targets,...).
/// @param descriptives Vector of descriptives structure containing the mean and standard deviation values for the scaling.
/// The size of that vector must be equal to the number of columns in the matrix.
/// @param columns_indices Vector of indices with the columns to be unscaled.
/// The size of that vector must be equal to the number of columns to be scaled.

void unscale_columns_mean_standard_deviation(Matrix<double>& matrix, const Vector<Descriptives>& descriptives, const Vector<size_t>& columns_indices)
{
    #ifdef __OPENNN_DEBUG__

    if(descriptives.size() != matrix.get_columns_number())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Transformations.\n"
              << "void unscale_columns_mean_standard_deviation(const Vector<Descriptives>&, const Vector<size_t>&) const method.\n"
              << "Size of descriptives vector (" << descriptives.size() << ") must be equal to number of columns (" << columns_indices.size() << ").\n";

       throw logic_error(buffer.str());
    }

    #endif

    const size_t rows_number = matrix.get_rows_number();

   size_t column_index;

   // Unscale columns

   for(size_t j = 0;  j < columns_indices.size(); j++)
   {
      column_index = columns_indices[j];

      if(descriptives[j].standard_deviation > numeric_limits<double>::min())
      {
         for(size_t i = 0; i < rows_number; i++)
         {
           matrix(i,column_index) = matrix(i,column_index)*descriptives[j].standard_deviation + descriptives[j].mean;
         }
      }
   }
}


/// Unscales the matrix columns with the minimum and maximum method.
/// @param matrix Matrix with values to be unscaled(inputs,targets,...).
/// @param descriptives Vector of descriptives which contains the minimum and maximum scaling values.
/// The size of that vector must be equal to the number of columns in this matrix.

void unscale_minimum_maximum(Matrix<double>& matrix, const Vector<Descriptives>& descriptives)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

   #ifdef __OPENNN_DEBUG__

   const size_t size = descriptives.size();

   if(size != columns_number)
   {
      ostringstream buffer;

      buffer << "OpenNN Exception: Transformations."
             << "void unscale_minimum_maximum(Matrix<double>&, const Vector<Descriptives>&) method.\n"
             << "Size of descriptives vector(" << size << ") must be equal to number of columns(" << columns_number <<").\n";

      throw logic_error(buffer.str());
   }

   #endif

   for(size_t j = 0; j < columns_number; j++)
   {
      if(descriptives[j].maximum - descriptives[j].minimum < numeric_limits<double>::min())
      {
         cout << "OpenNN Warning: Transformations.\n"
                   << "void unscale_minimum_maximum(const Vector<Descriptives>&) const method.\n"
                   << "Minimum and maximum values of column " << j << " are equal.\n"
                   << "Those columns won't be unscaled.\n";

         // Do nothing
      }
      else
      {
         for(size_t i = 0; i < rows_number; i++)
         {
           matrix(i,j) = 0.5*(matrix(i,j) + 1.0)*(descriptives[j].maximum-descriptives[j].minimum) + descriptives[j].minimum;
         }
      }
   }
}


/// Arranges the matrix in a proper format for association.
/// @todo describe what it does.
/// Note that this method sets new numbers of columns in the matrix.

void transform_association(Matrix<double>& matrix)
{
    Matrix<double> copy(matrix);

    matrix.set(copy.assemble_columns(copy));
}


/// Sets the elements of the vector to a given value if they fall below that
/// value.
/// @param vector Vector to be processed.
/// @param lower_bound Lower bound value.

void apply_lower_bound(Vector<double>& vector, const double &lower_bound)
{
  const size_t this_size = vector.size();

  for(size_t i = 0; i < this_size; i++) {
    if(vector[i] < lower_bound) {
     vector[i] = lower_bound;
    }
  }
}


/// Sets the elements of the vector to given values if they fall below that
/// values.
/// @param vector Vector to be processed.
/// @param lower_bound Lower bound values.

void apply_lower_bound(Vector<double>& vector, const Vector<double>&lower_bound)
{
  const size_t this_size = vector.size();

  for(size_t i = 0; i < this_size; i++) {
    if(vector[i] < lower_bound[i]) {
     vector[i] = lower_bound[i];
    }
  }
}


/// Sets the elements of the vector to a given value if they fall above that
/// value.
/// @param vector Vector to be processed.
/// @param upper_bound Upper bound value.

void apply_upper_bound(Vector<double>& vector, const double&upper_bound)
{
  const size_t this_size = vector.size();

  for(size_t i = 0; i < this_size; i++) {
    if(vector[i] > upper_bound) {
     vector[i] = upper_bound;
    }
  }
}


/// Sets the elements of the vector to given values if they fall above that
/// values.
/// @param vector Vector to be processed.
/// @param upper_bound Upper bound values.

void apply_upper_bound(Vector<double>& vector, const Vector<double>&upper_bound)
{
  const size_t this_size = vector.size();

  for(size_t i = 0; i < this_size; i++) {
    if(vector[i] > upper_bound[i]) {
     vector[i] = upper_bound[i];
    }
  }
}


/// Sets the elements of the vector to a given lower bound value if they fall
/// below that value,
/// or to a given upper bound value if they fall above that value.
/// @param vector Vector to be processed.
/// @param lower_bound Lower bound value.
/// @param upper_bound Upper bound value.

void apply_lower_upper_bounds(Vector<double>& vector, const double &lower_bound,
                                         const double &upper_bound)
{
  const size_t this_size = vector.size();

  for(size_t i = 0; i < this_size; i++) {
    if(vector[i] < lower_bound) {
     vector[i] = lower_bound;
    } else if(vector[i] > upper_bound) {
     vector[i] = upper_bound;
    }
  }
}


/// Sets the elements of the vector to given lower bound values if they fall
/// below that values,
/// or to given upper bound values if they fall above that values.
/// @param vector Vector to be processed.
/// @param lower_bound Lower bound values.
/// @param upper_bound Upper bound values.

void apply_lower_upper_bounds(Vector<double>& vector, const Vector<double>&lower_bound,
                                         const Vector<double>&upper_bound)
{
  const size_t this_size = vector.size();

  for(size_t i = 0; i < this_size; i++) {
    if(vector[i] < lower_bound[i]) {
     vector[i] = lower_bound[i];
    } else if(vector[i] > upper_bound[i]) {
     vector[i] = upper_bound[i];
    }
  }
}


/// Arranges a time series data matrix in a proper format for forecasting.
/// Note that this method sets new numbers of rows and columns in the matrix.
/// @param lags_number Number of lags for the prediction.
/// @param steps_ahead_number Number of steps ahead for the prediction.
/// @param time_index Index of the time column.

void transform_time_series(Matrix<double>& matrix,
                           const size_t& lags_number,
                           const size_t& steps_ahead_number,
                           const size_t& time_index)
{
    const size_t rows_number = matrix.get_rows_number();

    const size_t columns_number = matrix.get_columns_number();

    const Vector<double> time = matrix.get_column(time_index);

    matrix = matrix.delete_column(time_index);
    const size_t new_rows_number = rows_number - lags_number - steps_ahead_number + 1;
    const size_t new_columns_number =(columns_number-1) *(lags_number + steps_ahead_number);

    const Vector<size_t> indices(0, 1, new_rows_number-1);

    const Vector<double> new_time = time.get_subvector(indices);

    Matrix<double> new_matrix(new_rows_number, new_columns_number);

    Vector<double> new_row(new_columns_number);

    for(size_t i = 0; i < new_rows_number; i++)
    {
        new_row = matrix.get_rows(i+1, i + lags_number + steps_ahead_number);

        new_matrix.set_row(i, new_row);
    }

    new_matrix = new_matrix.insert_column(0, new_time, "Time");

    matrix.set(new_matrix);
}


/// Arranges a time series data matrix in a proper format for forecasting.
/// Note that this method sets new numbers of rows and columns in the matrix.
/// This method is used when there is not a time column in the matrix.
/// @param lags_number Number of lags for the prediction.
/// @param steps_ahead_number Number of steps ahead for the prediction.

void transform_time_series(Matrix<double>& matrix,
                          const size_t& lags_number,
                          const size_t& steps_ahead_number)
{
    const size_t rows_number = matrix.get_rows_number();

    const size_t columns_number = matrix.get_columns_number();

    const size_t new_rows_number = rows_number - lags_number - steps_ahead_number + 1;
    const size_t new_columns_number = columns_number *(lags_number + steps_ahead_number);

    Matrix<double> new_matrix(new_rows_number, new_columns_number);

    Vector<double> new_row(new_columns_number);

    for(size_t i = 0; i < new_rows_number; i++)
    {
        new_row = matrix.get_rows(i+1, i + lags_number + steps_ahead_number);

        new_matrix.set_row(i, new_row);
    }

    matrix.set(new_matrix);
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
