//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F U N C T I O N S   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "functions.h"

namespace OpenNN
{


Vector<double> sine(const Vector<double>& x)
{
const size_t n = x.size();

Vector<double> y(n);

for(size_t i = 0; i < n; i ++)
   y[i] = sin(x[i]);

return y;
}


Matrix<double> sine(const Matrix<double>& x)
{
const size_t rows_number = x.get_rows_number();
const size_t columns_number = x.get_columns_number();
const size_t n = rows_number * columns_number;

Matrix<double> y(rows_number, columns_number, 0.0);

for(size_t i = 0; i < n; i++)
{
   y[i] = sin(x[i]);
}

return y;
}


Vector<double> cosine(const Vector<double>& x)
{
const size_t n = x.size();

Vector<double> y(n);

for(size_t i = 0; i < n; i ++)
    y[i] = cos(x[i]);

return y;
}


Matrix<double> cosine(const Matrix<double>& x)
{
const size_t rows_number = x.get_rows_number();
const size_t columns_number = x.get_columns_number();
const size_t n = rows_number * columns_number;

Matrix<double> y(rows_number, columns_number, 0.0);

for(size_t i = 0; i < n; i++)
{
   y[i] = cos(x[i]);
}

return y;
}


Vector<double> exponential(const Vector<double>& x)
{
    const size_t size = x.size();

    Vector<double> y(size);

    for(size_t i = 0; i < size; i++)
        y[i] = exp(x[i]);

    return y;
}


Vector<double> logarithm(const Vector<double>& x)
{
    const size_t size = x.size();

    Vector<double> y(size);

    for(size_t i = 0; i < size; i++)
        y[i] = log(x[i]);

    return y;
}


Vector<double> power(const Vector<double>& x, const double& a)
{
    const size_t size = x.size();

    Vector<double> y(size);

    for(size_t i = 0; i < size; i++)
        y[i] = pow(x[i],a);

    return y;
}


// LINEAR

Tensor<double> linear(const Tensor<double>& x)
{
    return x;
}

Vector<double> linear(const Vector<double>& x)
{
    return x;
}


Tensor<double> hyperbolic_tangent(const Tensor<double>& x)
{
    const size_t size = x.size();

    const Vector<size_t> dimensions = x.get_dimensions();

    Tensor<double> y(dimensions);

    for(size_t i = 0; i < size; i++)
    {
        y[i] = tanh(x[i]);
    }

    return y;
}


Vector<double> hyperbolic_tangent(const Vector<double>& x)
{
    const size_t size = x.size();

    Vector<double> y(size);

    for(size_t i = 0; i < size; i++)
    {
        y[i] = tanh(x[i]);
    }

    return y;
}


Tensor<double> logistic(const Tensor<double>& x)
{
    const Vector<size_t> dimensions = x.get_dimensions();

    Tensor<double> y(dimensions);

    for(size_t i = 0; i < x.size(); i++)
    {
        y[i] = 1.0 / (1.0 + exp(-x[i]));
    }

    return y;
}

Vector<double> logistic(const Vector<double>& x)
{
    const size_t size = x.size();

    Tensor<double> y(size);

    for(size_t i = 0; i < size; i++)
    {
        y[i] = 1.0 / (1.0 + exp(-x[i]));
    }

    return y;
}


Vector<double> logistic_function(const Vector<double>& x, const double& a, const double& b)
{
    const size_t size = x.size();

    Tensor<double> y(size);

    for(size_t i = 0; i < size; i++)
    {
        y[i] = 1.0 / (1.0 + exp(-a-b*x[i]));
    }

    return y;
}


// THRESHOLD

Tensor<double> threshold(const Tensor<double>& x)
{
    const Vector<size_t> dimensions = x.get_dimensions();

    Tensor<double> y(dimensions);

    for(size_t i = 0; i < x.size(); i++)

      y[i] = x[i] < 0 ? 0.0 : 1.0;

    return y;
}


Vector<double> threshold(const Vector<double>& x)
{

    Vector<double> y(x.size());

    for(size_t i = 0; i < x.size(); i++)

      y[i] = x[i] < 0 ? 0.0 : 1.0;

    return y;
}


// SYMMETRIC THRESHOLD

Tensor<double> symmetric_threshold(const Tensor<double>& x)
{
    const size_t n = x.size();

    const Vector<size_t> dimensions = x.get_dimensions();

    Tensor<double> y(dimensions);

     for(size_t i = 0; i < n; i++)
         y[i] = x[i] < 0 ? -1.0 : 1.0;

    return y;
}


Vector<double> symmetric_threshold(const Vector<double>& x)
{
    const size_t n = x.size();

    Vector<double> y(n);

     for(size_t i = 0; i < n; i++)
         y[i] = x[i] < 0 ? -1.0 : 1.0;

    return y;
}

// RECTIFIED LINEAR

Tensor<double> rectified_linear(const Tensor<double>& x)
{
        const size_t n = x.size();

        const Vector<size_t> dimensions = x.get_dimensions();

        Tensor<double> y(dimensions);

        for(size_t i = 0; i < n; i++)
        {
            y[i] = x[i] < 0.0 ? 0.0 : x[i];
        }

        return y;
}


Vector<double> rectified_linear(const Vector<double>& x)
{
        const size_t n = x.size();

        Vector<double> y(n);

        for(size_t i = 0; i < n; i++)
        {
            y[i] = x[i] < 0.0 ? 0.0 : x[i];
        }

        return y;
}

// SCALED EXPONENTIAL LINEAR

Tensor<double> scaled_exponential_linear(const Tensor<double>& x)
{
    const size_t n = x.size();

    const Vector<size_t> dimensions = x.get_dimensions();

    double lambda =1.0507;
    double alpha =1.67326;


    Tensor<double> y(dimensions);


    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = lambda * alpha * (exp(x[i]) - 1) : y[i] = lambda * x[i];
    }

    return y;
}


Vector<double> scaled_exponential_linear(const Vector<double>& x)
{
    const size_t n = x.size();

    double lambda =1.0507;
    double alpha =1.67326;


    Vector<double> y(n);


    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = lambda * alpha * (exp(x[i]) - 1) : y[i] = lambda * x[i];
    }

    return y;
}
// SOFT PLUS

Tensor<double> soft_plus(const Tensor<double>& x)
{
    const size_t n = x.size();

    const Vector<size_t> dimensions = x.get_dimensions();

    Tensor<double> y(dimensions);

    for(size_t i = 0; i < n; i++)
    {
        y[i] = log(1 + exp(x[i]));
    }

    return y;
}


Vector<double> soft_plus(const Vector<double>& x)
{
    const size_t n = x.size();

    Vector<double> y(n);

    for(size_t i = 0; i < n; i++)
    {
        y[i] = log(1 + exp(x[i]));
    }

    return y;
}

// SOFT SIGN

Tensor<double> soft_sign(const Tensor<double>& x)
{
    const size_t n = x.size();

    const Vector<size_t> dimensions = x.get_dimensions();

    Tensor<double> y(dimensions);

    for(size_t i = 0; i < n; i++)
    {
       x[i] < 0.0 ? y[i] = x[i] / (1 - x[i]) : y[i] = x[i] / (1 + x[i]);
    }

    return y;
}


Vector<double> soft_sign(const Vector<double>& x)
{
    const size_t n = x.size();

    Vector<double> y(n);

    for(size_t i = 0; i < n; i++)
    {
       x[i] < 0.0 ? y[i] = x[i] / (1 - x[i]) : y[i] = x[i] / (1 + x[i]);
    }

    return y;
}

// HARD SIGMOID

Tensor<double> hard_sigmoid(const Tensor<double>& x)
{
   const size_t n = x.size();

   const Vector<size_t> dimensions = x.get_dimensions();

   Tensor<double> y(dimensions);

    for(size_t i = 0; i < n; i++)
    {
        if(x[i] < -2.5)
        {
           y[i] = 0;
        }
        else if(x[i] > 2.5)
        {
            y[i] = 1;
        }
        else
        {
            y[i] = 0.2 * x[i] + 0.5;
        }
    }

    return y;
}


Tensor<double> exponential_linear(const Tensor<double>& x)
{
    const size_t n = x.size();

    const Vector<size_t> dimensions = x.get_dimensions();

    Tensor<double> y(dimensions, 0.0);

    const double alpha = 1.0;

    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = alpha * (exp(x[i])- 1) : y[i] = x[i];
    }

    return y;
}


Vector<double> exponential_linear(const Vector<double>& x)
{
    const size_t n = x.size();

    Vector<double> y(n, 0.0);

    const double alpha = 1.0;

    for(size_t i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = alpha * (exp(x[i])- 1) : y[i] = x[i];
    }

    return y;
}


// SOFTMAX

Tensor<double> softmax(const Tensor<double>& x)
{
    const size_t rows_number = x.get_dimension(0);
    const size_t columns_number = x.get_dimension(1);

  Tensor<double> softmax(rows_number, columns_number);

  for(size_t j = 0; j < rows_number; j++)
  {
      double sum = 0.0;

      for(size_t i = 0; i < columns_number; i++)
      {
        sum += exp(x(j,i));
      }

      for(size_t i = 0; i < columns_number; i++)
      {
        softmax(j,i) = exp(x(j,i)) / sum;
      }
  }

  return softmax;
}

Tensor<double> softmax_rows(const Tensor<double>&)
{
    return Tensor<double>();
}


Matrix<double> hyperbolic_tangent(const Matrix<double>& x)
{
    const size_t rows_number = x.get_rows_number();
    const size_t columns_number = x.get_columns_number();

    const size_t n = x.size();

    Matrix<double> y(rows_number, columns_number);

    for(size_t i = 0; i < n; i ++)
       y[i] = tanh(x[i]);

    return  y;
}


Matrix<double> logistic(const Matrix<double>& x)
{
    Matrix<double> y(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < x.size(); i++)
    {
        y[i] = 1.0/(1.0 + exp(-x[i]));
    }

    return y;
}


Matrix<double> threshold(const Matrix<double>& x)
{
    Matrix<double> y(x.get_rows_number(), x.get_columns_number());

    for(size_t i = 0; i < x.size(); i++)

      y[i] = x[i] < 0 ? 0.0 : 1.0;

    return y;
}


Matrix<double> symmetric_threshold(const Matrix<double>& x)
{
 const size_t n = x.size();

 Matrix<double> y(x.get_rows_number(), x.get_columns_number());

  for(size_t i = 0; i < n; i++)
      y[i] = x[i] < 0 ? -1.0 : 1.0;

 return y;
}


Matrix<double> rectified_linear(const Matrix<double>& x)
{
 const size_t n = x.size();

 Matrix<double> y(x.get_rows_number(), x.get_columns_number());

 for(size_t i = 0; i < n; i++)
 {
     y[i] = x[i] < 0.0 ? 0.0 : x[i];
 }

 return y;

}


// SCALED EXPONENTIAL LINEAR

Matrix<double> scaled_exponential_linear(const Matrix<double>& x)
{
 const size_t n = x.size();

 double lambda =1.0507;
 double alpha =1.67326;


 Matrix<double> y(x.get_rows_number(), x.get_columns_number());


 for(size_t i = 0; i < n; i++)
 {
     x[i] < 0.0 ? y[i] = lambda * alpha * (exp(x[i]) - 1) : y[i] = lambda * x[i];
 }

 return y;
}


Matrix<double> soft_plus(const Matrix<double>& x)
{
 const size_t n = x.size();

 Matrix<double> y(x.get_rows_number(), x.get_columns_number());

 for(size_t i = 0; i < n; i++)
 {
     y[i] = log(1 + exp(x[i]));
 }

 return y;
}


Matrix<double> soft_sign(const Matrix<double>& x)
{
 const size_t n = x.size();

 Matrix<double> y(x.get_rows_number(), x.get_columns_number());

 for(size_t i = 0; i < n; i++)
 {
    x[i] < 0.0 ? y[i] = x[i] / (1 - x[i]) : y[i] = x[i] / (1 + x[i]);
 }

 return y;
}


Matrix<double> hard_sigmoid(const Matrix<double>& x)
{
 const size_t n = x.size();

 Matrix<double> y(x.get_rows_number(), x.get_columns_number());

 for(size_t i = 0; i < n; i++)
 {
     if(x[i] < -2.5)
     {
        y[i] = 0;
     }
     else if(x[i] > 2.5)
     {
         y[i] = 1;
     }
     else
     {
         y[i] = 0.2 * x[i] + 0.5;
     }
 }

 return y;
}


Vector<double> hard_sigmoid(const Vector<double>& x)
{
 const size_t n = x.size();

 Vector<double> y(x.size());

 for(size_t i = 0; i < n; i++)
 {
     if(x[i] < -2.5)
     {
        y[i] = 0;
     }
     else if(x[i] > 2.5)
     {
         y[i] = 1;
     }
     else
     {
         y[i] = 0.2 * x[i] + 0.5;
     }
 }

 return y;
}

Matrix<double> exponential_linear(const Matrix<double>& x)
{

 const size_t n = x.size();

 Matrix<double> y(x.get_rows_number(), x.get_columns_number(), 0.0);

 const double alpha = 1.0;

 for(size_t i = 0; i < n; i++)
 {
     x[i] < 0.0 ? y[i] = alpha * (exp(x[i])- 1) : y[i] = x[i];
 }

 return y;
}


Tensor<double> linear_derivatives(const Tensor<double>& x)
{
     /*const size_t n = x.get_dimension(0);

     const size_t columns_number = x.get_dimension(1);

     Tensor<double> y(Vector<size_t>({columns_number, columns_number, n}));

     for(size_t i = 0; i < n; i++)
     {
         for(size_t j = 0; j < columns_number; j++)
         {
             for(size_t k = 0; k < columns_number; k ++)
             {
                 if(j == k) y(j, k, i) = 1.0;
             }
         }
     }*/

     Tensor<double> y(x.get_dimensions(), 1.0);

     return y;
}


Tensor<double> hyperbolic_tangent_derivatives(const Tensor<double>& x)
{
    const size_t n = x.size();

    Tensor<double> y(x.get_dimensions());

    for(size_t i = 0; i < n; i++)
    {
        const double hyperbolic_tangent = tanh(x[i]);

        y[i] = 1.0 - hyperbolic_tangent*hyperbolic_tangent;
    }

    return y;
}


Vector<double> linear_derivatives(const Vector<double>& x)
{
    const size_t n = x.size();

    Tensor<double> y(n);

    for(size_t i = 0; i < n; i++)
    {
        y[i] = 1.0;
    }

    return y;

}


Vector<double> hyperbolic_tangent_derivatives(const Vector<double>& x)
{
    const size_t n = x.size();

    Tensor<double> y(n);

    for(size_t i = 0; i < n; i++)
    {
        const double hyperbolic_tangent = tanh(x[i]);

        y[i] = 1.0 - hyperbolic_tangent*hyperbolic_tangent;
    }

    return y;
}


Tensor<double> logistic_derivatives(const Tensor<double>& x)
{
    Tensor<double> y(x.get_dimensions(), 1);

    for(size_t i = 0; i < x.size(); i++)
    {
        const double exponential = exp(-x[i]);

        y[i] = exponential/((1.0 + exponential)*(1.0 + exponential));
    }

    return y;
}


Vector<double> logistic_derivatives(const Vector<double>& x)
{
    Vector<double> y(x.size());

    for(size_t i = 0; i < x.size(); i++)
    {
        const double exponential = exp(-x[i]);

        y[i] = exponential/((1.0 + exponential)*(1.0 + exponential));
    }

    return y;
}


Tensor<double> logistic_second_derivatives(const Tensor<double>& x)
{
    Tensor<double> y(x.get_dimensions(), 1);

    for(size_t i = 0; i < x.size(); i++)
    {
        Tensor<double> logistic_normal = logistic(y);
        Tensor<double> logistic_deriv = logistic_derivatives(y);

       y[i] = logistic_deriv[i]*(1-2*logistic_normal[i]);
    }

    return y;
}


Tensor<double> binary(const Tensor<double>& x)
{
    Tensor<double> y(x.get_dimensions(), 1);

    for(size_t i = 0; i < x.size(); i++)
    {
        x[i] < 0.5 ? y[i] = false : y [i] = true;
    }

    return y;
}


Tensor<double> threshold_derivatives(const Tensor<double>& x)
{
    const Tensor<double> y(x.get_dimensions(), 0.0);

  return y;
}


Vector<double> threshold_derivatives(const Vector<double>& x)
{
    return Vector<double>(x.size(), 0.0);
}


Tensor<double> symmetric_threshold_derivatives(const Tensor<double>& x)
{
 const Tensor<double> y(x.get_dimensions(), 0.0);

 return y;
}


Vector<double> symmetric_threshold_derivatives(const Vector<double>& x)
{
 const Vector<double> y(x.size(), 0.0);

 return y;
}


Tensor<double> rectified_linear_derivatives(const Tensor<double>& x)
{
     const size_t n = x.size();

     Tensor<double> derivatives(x.get_dimensions());

     for(size_t i = 0; i < n; i++)
     {
         x[i] < 0.0 ? derivatives[i] = 0.0 : derivatives[i] = 1.0;
     }

     return derivatives;
}


Vector<double> rectified_linear_derivatives(const Vector<double>& x)
{
     const size_t n = x.size();

     Tensor<double> derivatives(n);

     for(size_t i = 0; i < n; i++)
     {
         x[i] < 0.0 ? derivatives[i] = 0.0 : derivatives[i] = 1.0;
     }

     return derivatives;
}

Tensor<double> scaled_exponential_linear_derivatives(const Tensor<double>& x)
{
 const size_t n = x.size();

 const double lambda =1.0507;
 const double alpha =1.67326;

 Tensor<double> derivatives(x.get_dimensions());

 for(size_t i = 0; i < n; i++)
 {
     x[i] < 0.0 ? derivatives[i] = lambda * alpha * exp(x[i]) : derivatives[i] = lambda;
 }

 return derivatives;
}

Vector<double> scaled_exponential_linear_derivatives(const Vector<double>& x)
{
 const size_t n = x.size();

 const double lambda =1.0507;
 const double alpha =1.67326;

 Tensor<double> derivatives(n);

 for(size_t i = 0; i < n; i++)
 {
     x[i] < 0.0 ? derivatives[i] = lambda * alpha * exp(x[i]) : derivatives[i] = lambda;
 }

 return derivatives;
}


Tensor<double> soft_plus_derivatives(const Tensor<double>& x)
{
 const size_t n = x.size();

 Tensor<double> derivatives(x.get_dimensions());

 for(size_t i = 0; i < n; i++)
 {
     derivatives[i] = 1/(1 + exp(-x[i]));
 }

 return derivatives;
}


Vector<double> soft_plus_derivatives(const Vector<double>& x)
{
 const size_t n = x.size();

 Tensor<double> derivatives(n);

 for(size_t i = 0; i < n; i++)
 {
     derivatives[i] = 1/(1 + exp(-x[i]));
 }

 return derivatives;
}

Tensor<double> soft_sign_derivatives(const Tensor<double>& x)
{
 const size_t n = x.size();

 Tensor<double> derivatives(x.get_dimensions());

 for(size_t i = 0; i < n; i++)
 {
    x[i] < 0.0 ? derivatives[i] = 1 / pow((1 - x[i]), 2) : derivatives[i] = 1 / pow((1 + x[i]), 2);

 }

 return derivatives;
}

Vector<double> soft_sign_derivatives(const Vector<double>& x)
{
 const size_t n = x.size();

 Vector<double> derivatives(n);

 for(size_t i = 0; i < n; i++)
 {
    x[i] < 0.0 ? derivatives[i] = 1 / pow((1 - x[i]), 2) : derivatives[i] = 1 / pow((1 + x[i]), 2);

 }

 return derivatives;
}


Tensor<double> hard_sigmoid_derivatives(const Tensor<double>& x)
{
const size_t n = x.size();

Tensor<double> derivatives(x.get_dimensions());

 for(size_t i = 0; i < n; i++)
 {
     x[i] < -2.5 || x[i] > 2.5 ? derivatives[i] = 0.0 : derivatives[i] = 0.2;
 }

 return derivatives;
}


Vector<double> hard_sigmoid_derivatives(const Vector<double>& x)
{
const size_t n = x.size();

Tensor<double> derivatives(n);

 for(size_t i = 0; i < n; i++)
 {
     x[i] < -2.5 || x[i] > 2.5 ? derivatives[i] = 0.0 : derivatives[i] = 0.2;
 }

 return derivatives;
}


Tensor<double> exponential_linear_derivatives(const Tensor<double>& x)
{
 const size_t n = x.size();

 Tensor<double> derivatives(x.get_dimensions());

 const double alpha = 1.0;

 for(size_t i = 0; i < n; i++)
 {
     x[i] < 0.0 ? derivatives[i] = alpha * exp(x[i]) : derivatives[i] = 1.0;
 }

 return derivatives;
}


Vector<double> exponential_linear_derivatives(const Vector<double>& x)
{
 const size_t n = x.size();

 Vector<double> derivatives(n);

 const double alpha = 1.0;

 for(size_t i = 0; i < n; i++)
 {
     x[i] < 0.0 ? derivatives[i] = alpha * exp(x[i]) : derivatives[i] = 1.0;
 }

 return derivatives;
}

Vector<double> softmax(const Vector<double>& x)
{
    const size_t this_size = x.size();

    Vector<double> softmax(this_size);

    double sum = 0;

    for(size_t i = 0; i < this_size; i++) {
        sum += exp(x[i]);
    }

    for(size_t i = 0; i < this_size; i++) {
        softmax[i] = exp(x[i]) / sum;
    }

    return softmax;
}


Tensor<double> softmax_derivatives(const Tensor<double>& x)
{
 const size_t n = x.get_dimension(0);

 const size_t columns_number = x.get_dimension(1);

 Vector<size_t> dimensions = {columns_number, columns_number, n};

 Tensor<double> y(dimensions);

 for(size_t i = 0; i < n; i ++)
 {
     const Vector<double> softmax_values = softmax(x.get_matrix(0).get_row(i));

     for(size_t j = 0; j < columns_number; j++)
     {
         for(size_t k = 0; k < columns_number; k++)
         {
             if(j == k)
             {
                 y(j,k,i) = softmax_values[j]*(1.0 - softmax_values[j]);
             }
             else
             {
                 y(j,k,i) = -softmax_values[j] * softmax_values[k];
             }
         }
     }
 }

 return y;
}


Matrix<double> competitive(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    Matrix<double> competitive(rows_number, columns_number, 0);

    for(size_t i = 0; i < rows_number; i++)
    {
        const size_t maximal_index = OpenNN::maximal_index(matrix.get_row(i));

        competitive(i, maximal_index) = 1;
    }

    return(competitive);
}


Tensor<double> competitive(const Tensor<double>& x)
{
    const size_t rows_number = x.get_dimension(0);
    const size_t columns_number = x.get_dimension(1);

    Tensor<double> competitive({rows_number, columns_number}, 0.0);

    for(size_t i = 0; i < rows_number; i++)
    {
        const size_t maximal_index = OpenNN::maximal_index(x.get_matrix(0).get_row(i));

        competitive(i, maximal_index) = 1;
    }

    return(competitive);

}


Matrix<double> softmax_rows(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    Matrix<double> softmax(rows_number,columns_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        softmax.set_row(i, OpenNN::softmax(matrix.get_row(i)));
    }

    return softmax;
}


Matrix<double> softmax_columns(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    Matrix<double> softmax(rows_number,columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        softmax.set_column(i, OpenNN::softmax(matrix.get_column(i)));
    }

    return softmax;
}


/// This method converts the values of the vector to be binary.
/// The threshold value used is 0.5.

Vector<bool> binary(const Vector<double>& vector)  {
  const size_t this_size = vector.size();

  Vector<bool> binary(this_size);

  for(size_t i = 0; i < this_size; i++) {
    if(vector[i] < 0.5) {
      binary[i] = false;
    } else {
      binary[i] = true;
    }
  }

  return(binary);
}


///Calculate the square_root of a vector

Vector<double> square_root(const Vector<double>& vector)
{
    const size_t this_size = vector.size();

    Vector<double>square(this_size);

    for(size_t i = 0; i < this_size; i++) {

        square[i]=sqrt(vector[i]);
    }


    return square;
}


/// Return the cumulative vector of this vector,
/// where each element is summed up with all the previous ones.

Vector<double> cumulative(const Vector<double>& vector)
{
  const size_t this_size = vector.size();

  Vector<double> cumulative(this_size);

  if(this_size > 0) {
    cumulative[0] = vector[0];

    for(size_t i = 1; i < this_size; i++) {
      cumulative[i] = cumulative[i - 1] + vector[i];
    }
  }

  return(cumulative);
}


/// Returns the softmax vector of this matrix,
/// whose elements sum one, and can be interpreted as probabilities.

Matrix<double> softmax(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

  Matrix<double> softmax(rows_number, columns_number);

  for(size_t j = 0; j < rows_number; j++)
  {
      double sum = 0;

      for(size_t i = 0; i < columns_number; i++)
      {
        sum += exp(matrix(j,i));
      }

      for(size_t i = 0; i < columns_number; i++)
      {
        softmax(j,i) = exp(matrix(j,i)) / sum;
      }
  }

  return softmax;
}


/// Returns a random number chosen from a normal distribution.
/// @param mean Mean value of normal distribution.
/// @param standard_deviation Standard deviation value of normal distribution.

double random_normal(const double &mean, const double &standard_deviation)
{
  const double pi = 4.0 * atan(1.0);

  double random_uniform_1;

  do {
    random_uniform_1 = static_cast<double>(rand()) /(RAND_MAX + 1.0);

  } while(random_uniform_1 == 0.0);

  const double random_uniform_2 = static_cast<double>(rand()) /(RAND_MAX + 1.0);

  // Box-Muller transformation

  const double random_normal = mean +
                          sqrt(-2.0 * log(random_uniform_1)) *
                              sin(2.0 * pi * random_uniform_2) *
                              standard_deviation;

  return(random_normal);
}


size_t factorial(const size_t& number)
{
    size_t fact = 1;

    if(number == 0)
    {
        fact = 1;
    }
    else
    {
        for(size_t i = 1; i <= number; i++)
        {
            fact = fact*i;
        }
    }

    return fact;
}


/// Returns a vector with the bounded elements from below of the current vector.
/// @param lower_bound Lower bound values.

Vector<double> lower_bounded(const Vector<double>& vector, const double &lower_bound)
{
  const size_t this_size = vector.size();

  Vector<double> bounded_vector(this_size);

  for(size_t i = 0; i < this_size; i++) {
    if(vector[i] < lower_bound) {
      bounded_vector[i] = lower_bound;
    } else {
      bounded_vector[i] = vector[i];
    }
  }

  return bounded_vector;
}


/// Returns a vector with the bounded elements from above of the current vector.
/// @param lower_bound Lower bound values.

Vector<double> lower_bounded(const Vector<double>& vector, const Vector<double>& lower_bound)
{
  const size_t this_size = vector.size();

#ifdef __OPENNN_DEBUG__

  const size_t lower_bound_size = lower_bound.size();

  if(lower_bound_size != this_size) {
    ostringstream buffer;

    buffer
        << "OpenNN Exception: Vector Template.\n"
        << "Vector<T> calculate_lower_bounded(const Vector<T>&) const method.\n"
        << "Lower bound size must be equal to vector size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<double> bounded_vector(this_size);

  // Apply lower bound

  for(size_t i = 0; i < this_size; i++) {
    if(vector[i] < lower_bound[i]) {
      bounded_vector[i] = lower_bound[i];
    } else {
      bounded_vector[i] = vector[i];
    }
  }

  return bounded_vector;
}


/// This method bounds the elements of the vector if they fall above an upper
/// bound value.
/// @param upper_bound Upper bound value.

Vector<double> upper_bounded(const Vector<double>& vector, const double &upper_bound)
{
  const size_t this_size = vector.size();

  Vector<double> bounded_vector(this_size);

  for(size_t i = 0; i < this_size; i++) {
    if(vector[i] > upper_bound) {
      bounded_vector[i] = upper_bound;
    } else {
      bounded_vector[i] = vector[i];
    }
  }

  return bounded_vector;
}


/// This method bounds the elements of the vector if they fall above their
/// corresponding upper bound values.
/// @param upper_bound Upper bound values.

Vector<double> upper_bounded(const Vector<double>& vector, const Vector<double>&upper_bound)
{
  const size_t this_size = vector.size();

  

#ifdef __OPENNN_DEBUG__

  const size_t upper_bound_size = upper_bound.size();

  if(upper_bound_size != this_size) {
    ostringstream buffer;

    buffer
        << "OpenNN Exception: Vector Template.\n"
        << "Vector<T> calculate_upper_bounded(const Vector<T>&) const method.\n"
        << "Upper bound size must be equal to vector size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<double> bounded_vector(this_size);

  // Apply upper bound

  for(size_t i = 0; i < this_size; i++) {
    if(vector[i] > upper_bound[i]) {
      bounded_vector[i] = upper_bound[i];
    } else {
      bounded_vector[i] = vector[i];
    }
  }

  return bounded_vector;
}


/// This method bounds the elements of the vector if they fall above or below
/// their lower or upper
/// bound values, respectively.
/// @param lower_bound Lower bound value.
/// @param upper_bound Upper bound value.

Vector<double> lower_upper_bounded(const Vector<double>& vector, const double &lower_bound, const double &upper_bound)
{
  const size_t this_size = vector.size();

  Vector<double> bounded_vector(this_size);

  for(size_t i = 0; i < this_size; i++) {
    if(vector[i] < lower_bound) {
      bounded_vector[i] = lower_bound;
    } else if(vector[i] > upper_bound) {
      bounded_vector[i] = upper_bound;
    } else {
      bounded_vector[i] = vector[i];
    }
  }

  return bounded_vector;
}


/// This method bounds the elements of the vector if they fall above or below
/// their corresponding lower or upper
/// bound values, respectively.
/// @param lower_bound Lower bound values.
/// @param upper_bound Upper bound values.


Vector<double> lower_upper_bounded(const Vector<double>& vector, const Vector<double>&lower_bound,
                                         const Vector<double>&upper_bound)
{
  const size_t this_size = vector.size();



#ifdef __OPENNN_DEBUG__

  const size_t lower_bound_size = lower_bound.size();
  const size_t upper_bound_size = upper_bound.size();

  if(lower_bound_size != this_size || upper_bound_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: Vector Template.\n"
           << "Vector<T> calculate_lower_upper_bounded(const Vector<T>&, const "
              "Vector<T>&) const method.\n"
           << "Lower and upper bound sizes must be equal to vector size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Vector<double> bounded_vector(this_size);

  // Apply lower and upper bounds

  for(size_t i = 0; i < this_size; i++) {
    if(vector[i] < lower_bound[i]) {
      bounded_vector[i] = lower_bound[i];
    } else if(vector[i] > upper_bound[i]) {
      bounded_vector[i] = upper_bound[i];
    } else {
      bounded_vector[i] = vector[i];
    }
  }

  return bounded_vector;
}


Matrix<double> lower_bounded(const Matrix<double>& matrix, const double & lower_bound)
{
    const size_t this_size = matrix.size();

    Matrix<double> bounded_matrix(matrix);

    for(size_t i = 0; i < this_size; i++)
    {
      if(matrix[i] < lower_bound)
      {
        bounded_matrix[i] = lower_bound;
      }
    }

    return(bounded_matrix);
}


Matrix<double> upper_bounded(const Matrix<double>& matrix, const double & upper_bound)
{
    const size_t this_size = matrix.size();

    Matrix<double> bounded_matrix(matrix);

    for(size_t i = 0; i < this_size; i++)
    {
      if(matrix[i] > upper_bound)
      {
        bounded_matrix[i] = upper_bound;
      }
    }

    return(bounded_matrix);
}


Matrix<double> lower_upper_bounded(const Matrix<double>& matrix, const Vector<double>& lower_bounds, const Vector<double>& upper_bounds)
{
    Matrix<double> bounded_matrix(matrix);

    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if(bounded_matrix(i,j) < lower_bounds[j]) bounded_matrix(i,j) = lower_bounds[j];
            else if(bounded_matrix(i,j) > upper_bounds[j]) bounded_matrix(i,j) = upper_bounds[j];
        }
    }

    return bounded_matrix;
}


Tensor<double>lower_upper_bounded(const Tensor<double>& tensor, const Vector<double>& lower_bounds, const Vector<double>& upper_bounds)
{
    Tensor<double> bounded_tensor(tensor);

    const size_t rows_number = tensor.get_dimension(0);
    const size_t columns_number = tensor.get_dimension(1);

    for(size_t i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < columns_number; j++)
        {
            if(bounded_tensor(i,j) < lower_bounds[j]) bounded_tensor(i,j) = lower_bounds[j];
            else if(bounded_tensor(i,j) > upper_bounds[j]) bounded_tensor(i,j) = upper_bounds[j];
        }
    }

    return bounded_tensor;
}


/// Returns the gradient of the vector norm.

Vector<double> sign(const Vector<double>& vector)
{
  const size_t this_size = vector.size();  

  Vector<double> sign_vector(this_size);

  for(size_t i = 0; i < this_size; i++)
  {
    if(vector[i] < 0)
    {
        sign_vector[i] = -1.0;
    }
    else if(vector[i] > 0)
    {
        sign_vector[i] = 1.0;
    }
    else
    {
        throw logic_error("Error: Parameter " + to_string(i) + " is equal to zero: Non-derivative function");
    }
  }

  return(sign_vector);
}


/// Returns this vector divided by its norm.

Vector<double> normalized(const Vector<double>& vector)
{
  const size_t this_size = vector.size();

  Vector<double> normalized(this_size);

  const double norm = l2_norm(vector);

  if(norm == 0.0) {
    normalized.initialize(0.0);
  } else {
    normalized = vector / norm;
  }

  return normalized;
}


Vector<double> absolute_value(const Vector<double>& vector)
{
  const size_t this_size = vector.size();

  Vector<double> absolute_value(this_size);

  for(size_t i = 0; i < this_size; i++) {
    if(vector[i] > 0) {
      absolute_value[i] = vector[i];
    } else {
      absolute_value[i] = -vector[i];
    }
  }

  return(absolute_value);
}


Matrix<double> normalized_columns(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    Matrix<double> softmax(rows_number,columns_number);

    for(size_t i = 0; i < columns_number; i++)
    {
        softmax.set_column(i, normalized(matrix.get_column(i)));
    }

    return softmax;
}


/// Returns a matrix with the absolute values of this matrix.

Matrix<double> absolute_value(const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    Matrix<double> absolute_value(rows_number, columns_number);

    for(size_t i = 0; i < matrix.size(); i++)
    {
          if(matrix[i] > 0)
          {
             absolute_value[i] = matrix[i];
          }
          else
          {
             absolute_value[i] = -matrix[i];
          }
    }

    return absolute_value;
}

}
