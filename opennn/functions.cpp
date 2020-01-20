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

Tensor<type, 1> sine(const Tensor<type, 1>& x)
{
const int n = x.size();

Tensor<type, 1> y(n);

for(int i = 0; i < n; i ++)
   y[i] = sin(x[i]);

return y;
}


Tensor<type, 2> sine(const Tensor<type, 2>& x)
{
const int rows_number = x.dimension(0);
const int columns_number = x.dimension(1);
const int n = rows_number * columns_number;

Tensor<type, 2> y(rows_number, columns_number);
/*
for(int i = 0; i < n; i++)
{
   y[i] = sin(x[i]);
}
*/
return y;
}


Tensor<type, 1> cosine(const Tensor<type, 1>& x)
{      
const int n = x.size();

Tensor<type, 1> y(n);

for(int i = 0; i < n; i ++)
    y[i] = cos(x[i]);

return y;
}


Tensor<type, 2> cosine(const Tensor<type, 2>& x)
{
const int rows_number = x.dimension(0);
const int columns_number = x.dimension(1);
const int n = rows_number * columns_number;

Tensor<type, 2> y(rows_number, columns_number);
/*
for(int i = 0; i < n; i++)
{
   y[i] = cos(x[i]);
}
*/
return y;
}


Tensor<type, 1> exponential(const Tensor<type, 1>& x)
{
    const int size = x.size();

    Tensor<type, 1> y(size);

    for(int i = 0; i < size; i++)
        y[i] = exp(x[i]);

    return y;
}


Tensor<type, 1> logarithm(const Tensor<type, 1>& x)
{
    const int size = x.size();

    Tensor<type, 1> y(size);

    for(int i = 0; i < size; i++)
        y[i] = log(x[i]);

    return y;
}


Tensor<type, 1> power(const Tensor<type, 1>& x, const double& a)
{
    const int size = x.size();

    Tensor<type, 1> y(size);

    for(int i = 0; i < size; i++)
        y[i] = pow(x[i],a);

    return y;
}


// LINEAR

Tensor<type, 2> linear(const Tensor<type, 2>& x)
{
    return x;
}

Tensor<type, 1> linear(const Tensor<type, 1>& x)
{
    return x;
}


Tensor<type, 2> hyperbolic_tangent(const Tensor<type, 2>& x)
{

    const int size = x.size();

    const auto& dimensions = x.dimensions();

    Tensor<type, 2> y(dimensions);
/*
    for(int i = 0; i < size; i++)
    {
        y[i] = tanh(x[i]);
    }
*/
    return y;
}

Tensor<type, 1> hyperbolic_tangent(const Tensor<type, 1>& x)
{
    const int size = x.size();

    Tensor<type, 1> y(size);

    for(int i = 0; i < size; i++)
    {
        y[i] = tanh(x[i]);
    }

    return y;
}


Tensor<type, 2> logistic(const Tensor<type, 2>& x)
{
    const auto& dimensions = x.dimensions();

    Tensor<type, 2> y(dimensions);
/*
    for(int i = 0; i < x.size(); i++)
    {
        y[i] = 1.0 / (1.0 + exp(-x[i]));
    }
*/
    return y;
}

Tensor<type, 1> logistic(const Tensor<type, 1>& x)
{
    const int size = x.size();

    Tensor<type, 1> y(size);

    for(int i = 0; i < size; i++)
    {
        y[i] = 1.0 / (1.0 + exp(-x[i]));
    }

    return y;
}


Tensor<type, 1> logistic_function(const Tensor<type, 1>& x, const double& a, const double& b)
{
    const int size = x.size();

    Tensor<type, 1> y(size);

    for(int i = 0; i < size; i++)
    {
        y[i] = 1.0 / (1.0 + exp(-a-b*x[i]));
    }

    return y;
}


// THRESHOLD

Tensor<type, 2> threshold(const Tensor<type, 2>& x)
{
    const auto& dimensions = x.dimensions();

    Tensor<type, 2> y(dimensions);
/*
    for(int i = 0; i < x.size(); i++)

      y[i] = x[i] < 0 ? 0.0 : 1.0;
*/
    return y;
}


Tensor<type, 1> threshold(const Tensor<type, 1>& x)
{

    Tensor<type, 1> y(x.size());

    for(int i = 0; i < x.size(); i++)

      y[i] = x[i] < 0 ? 0.0 : 1.0;

    return y;
}


// SYMMETRIC THRESHOLD

Tensor<type, 2> symmetric_threshold(const Tensor<type, 2>& x)
{
    const int n = x.size();

    const auto& dimensions = x.dimensions();

    Tensor<type, 2> y(dimensions);
/*

     for(int i = 0; i < n; i++)
         y[i] = x[i] < 0 ? -1.0 : 1.0;
*/
    return y;
}


Tensor<type, 1> symmetric_threshold(const Tensor<type, 1>& x)
{
    const int n = x.size();

    Tensor<type, 1> y(n);

     for(int i = 0; i < n; i++)
         y[i] = x[i] < 0 ? -1.0 : 1.0;

    return y;
}

// RECTIFIED LINEAR

Tensor<type, 2> rectified_linear(const Tensor<type, 2>& x)
{
    const int n = x.size();

    const auto& dimensions = x.dimensions();

    Tensor<type, 2> y(dimensions);
/*
    for(int i = 0; i < n; i++)
    {
        y[i] = x[i] < 0.0 ? 0.0 : x[i];
    }
*/
    return y;
}


Tensor<type, 1> rectified_linear(const Tensor<type, 1>& x)
{
        const int n = x.size();

        Tensor<type, 1> y(n);

        for(int i = 0; i < n; i++)
        {
            y[i] = x[i] < 0.0 ? 0.0 : x[i];
        }

        return y;
}

// SCALED EXPONENTIAL LINEAR

Tensor<type, 2> scaled_exponential_linear(const Tensor<type, 2>& x)
{
    const int n = x.size();

    const auto& dimensions = x.dimensions();

    const double lambda =1.0507;
    const double alpha =1.67326;

    Tensor<type, 2> y(dimensions);
/*
    for(int i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = lambda * alpha * (exp(x[i]) - 1) : y[i] = lambda * x[i];
    }
*/
    return y;
}


Tensor<type, 1> scaled_exponential_linear(const Tensor<type, 1>& x)
{
    const int n = x.size();

    const double lambda =1.0507;
    const double alpha =1.67326;

    Tensor<type, 1> y(n);

    for(int i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = lambda * alpha * (exp(x[i]) - 1) : y[i] = lambda * x[i];
    }

    return y;
}


// SOFT PLUS

Tensor<type, 2> soft_plus(const Tensor<type, 2>& x)
{
    const int n = x.size();

    const auto& dimensions = x.dimensions();

    Tensor<type, 2> y(dimensions);
/*
    for(int i = 0; i < n; i++)
    {
        y[i] = log(1 + exp(x[i]));
    }
*/
    return y;
}


Tensor<type, 1> soft_plus(const Tensor<type, 1>& x)
{
    const int n = x.size();

    Tensor<type, 1> y(n);

    for(int i = 0; i < n; i++)
    {
        y[i] = log(1 + exp(x[i]));
    }

    return y;
}


Tensor<type, 2> soft_sign(const Tensor<type, 2>& x)
{
    const int n = x.size();

    const auto& dimensions = x.dimensions();

    Tensor<type, 2> y(dimensions);
/*
    for(int i = 0; i < n; i++)
    {
       x[i] < 0.0 ? y[i] = x[i] / (1 - x[i]) : y[i] = x[i] / (1 + x[i]);
    }
*/
    return y;
}


Tensor<type, 1> soft_sign(const Tensor<type, 1>& x)
{
    const int n = x.size();

    Tensor<type, 1> y(n);

    for(int i = 0; i < n; i++)
    {
       x[i] < 0.0 ? y[i] = x[i] / (1 - x[i]) : y[i] = x[i] / (1 + x[i]);
    }

    return y;
}

// HARD SIGMOID

Tensor<type, 2> hard_sigmoid(const Tensor<type, 2>& x)
{
   const int n = x.size();

   const auto& dimensions = x.dimensions();

   Tensor<type, 2> y(dimensions);
/*
    for(int i = 0; i < n; i++)
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
*/
    return y;
}


Tensor<type, 2> exponential_linear(const Tensor<type, 2>& x)
{
    const int n = x.size();

    const auto& dimensions = x.dimensions();

    Tensor<type, 2> y(dimensions);

    const double alpha = 1.0;
/*
    for(int i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = alpha * (exp(x[i])- 1) : y[i] = x[i];
    }
*/
    return y;
}


Tensor<type, 1> exponential_linear(const Tensor<type, 1>& x)
{
    const int n = x.size();

    Tensor<type, 1> y(n);
/*
    const double alpha = 1.0;

    for(int i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = alpha * (exp(x[i])- 1) : y[i] = x[i];
    }
*/
    return y;
}


// SOFTMAX

Tensor<type, 2> softmax(const Tensor<type, 2>& x)
{
    const int rows_number = x.dimension(0);
    const int columns_number = x.dimension(1);

  Tensor<type, 2> softmax(rows_number, columns_number);

  for(int j = 0; j < rows_number; j++)
  {
      double sum = 0.0;

      for(int i = 0; i < columns_number; i++)
      {
        sum += exp(x(j,i));
      }

      for(int i = 0; i < columns_number; i++)
      {
        softmax(j,i) = exp(x(j,i)) / sum;
      }
  }

  return softmax;
}

Tensor<type, 2> softmax_rows(const Tensor<type, 2>&)
{
    return Tensor<type, 2>();
}

/*
Tensor<type, 2> hyperbolic_tangent(const Tensor<type, 2>& x)
{
    const int rows_number = x.dimension(0);
    const int columns_number = x.dimension(1);

    const int n = x.size();

    Tensor<type, 2> y(rows_number, columns_number);

    for(int i = 0; i < n; i ++)
       y[i] = tanh(x[i]);

    return  y;
}


Tensor<type, 2> logistic(const Tensor<type, 2>& x)
{
    Tensor<type, 2> y(x.dimension(0), x.dimension(1));

    for(int i = 0; i < x.size(); i++)
    {
        y[i] = 1.0/(1.0 + exp(-x[i]));
    }

    return y;
}


Tensor<type, 2> threshold(const Tensor<type, 2>& x)
{
    Tensor<type, 2> y(x.dimension(0), x.dimension(1));

    for(int i = 0; i < x.size(); i++)

      y[i] = x[i] < 0 ? 0.0 : 1.0;

    return y;
}


Tensor<type, 2> symmetric_threshold(const Tensor<type, 2>& x)
{
 const int n = x.size();

 Tensor<type, 2> y(x.dimension(0), x.dimension(1));

  for(int i = 0; i < n; i++)
      y[i] = x[i] < 0 ? -1.0 : 1.0;

 return y;
}


Tensor<type, 2> rectified_linear(const Tensor<type, 2>& x)
{
 const int n = x.size();

 Tensor<type, 2> y(x.dimension(0), x.dimension(1));

 for(int i = 0; i < n; i++)
 {
     y[i] = x[i] < 0.0 ? 0.0 : x[i];
 }

 return y;

}


// SCALED EXPONENTIAL LINEAR

Tensor<type, 2> scaled_exponential_linear(const Tensor<type, 2>& x)
{
 const int n = x.size();

 const double lambda =1.0507;
 const double alpha =1.67326;

 Tensor<type, 2> y(x.dimension(0), x.dimension(1));


 for(int i = 0; i < n; i++)
 {
     x[i] < 0.0 ? y[i] = lambda * alpha * (exp(x[i]) - 1) : y[i] = lambda * x[i];
 }

 return y;
}


Tensor<type, 2> soft_plus(const Tensor<type, 2>& x)
{
 const int n = x.size();

 Tensor<type, 2> y(x.dimension(0), x.dimension(1));

 for(int i = 0; i < n; i++)
 {
     y[i] = log(1 + exp(x[i]));
 }

 return y;
}


Tensor<type, 2> soft_sign(const Tensor<type, 2>& x)
{
 const int n = x.size();

 Tensor<type, 2> y(x.dimension(0), x.dimension(1));

 for(int i = 0; i < n; i++)
 {
    x[i] < 0.0 ? y[i] = x[i] / (1 - x[i]) : y[i] = x[i] / (1 + x[i]);
 }

 return y;
}


Tensor<type, 2> hard_sigmoid(const Tensor<type, 2>& x)
{
 const int n = x.size();

 Tensor<type, 2> y(x.dimension(0), x.dimension(1));

 for(int i = 0; i < n; i++)
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


Tensor<type, 1> hard_sigmoid(const Tensor<type, 1>& x)
{
 const int n = x.size();

 Tensor<type, 1> y(x.size());

 for(int i = 0; i < n; i++)
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

Tensor<type, 2> exponential_linear(const Tensor<type, 2>& x)
{

 const int n = x.size();

 Tensor<type, 2> y(x.dimension(0), x.dimension(1), 0.0);

 const double alpha = 1.0;

 for(int i = 0; i < n; i++)
 {
     x[i] < 0.0 ? y[i] = alpha * (exp(x[i])- 1) : y[i] = x[i];
 }

 return y;
}


Tensor<type, 2> linear_derivatives(const Tensor<type, 2>& x)
{
     const int n = x.dimension(0);

     const int columns_number = x.dimension(1);

     Tensor<type, 2> y(vector<int>({columns_number, columns_number, n}));

     for(int i = 0; i < n; i++)
     {
         for(int j = 0; j < columns_number; j++)
         {
             for(int k = 0; k < columns_number; k ++)
             {
                 if(j == k) y(j, k, i) = 1.0;
             }
         }
     }



     Tensor<type, 2> y(x.dimensions());
     y.setConstant(1.0);

     return y;
}


Tensor<type, 2> hyperbolic_tangent_derivatives(const Tensor<type, 2>& x)
{
    const int n = x.size();

    Tensor<type, 2> y(x.dimensions());

    for(int i = 0; i < n; i++)
    {
        const double hyperbolic_tangent = tanh(x[i]);

        y[i] = 1.0 - hyperbolic_tangent*hyperbolic_tangent;
    }

    return y;
}
*/

Tensor<type, 1> linear_derivatives(const Tensor<type, 1>& x)
{
    const int n = x.size();

    Tensor<type, 1> y(n);

    for(int i = 0; i < n; i++)
    {
        y[i] = 1.0;
    }

    return y;

}


Tensor<type, 1> hyperbolic_tangent_derivatives(const Tensor<type, 1>& x)
{
    const int n = x.size();

    Tensor<type, 1> y(n);

    for(int i = 0; i < n; i++)
    {
        const double hyperbolic_tangent = tanh(x[i]);

        y[i] = 1.0 - hyperbolic_tangent*hyperbolic_tangent;
    }

    return y;
}


Tensor<type, 2> logistic_derivatives(const Tensor<type, 2>& x)
{
    Tensor<type, 2> y(x.dimensions());
/*
    for(int i = 0; i < x.size(); i++)
    {
        const double exponential = exp(-x[i]);

        y[i] = exponential/((1.0 + exponential)*(1.0 + exponential));
    }
*/
    return y;
}


Tensor<type, 1> logistic_derivatives(const Tensor<type, 1>& x)
{
    Tensor<type, 1> y(x.size());

    for(int i = 0; i < x.size(); i++)
    {
        const double exponential = exp(-x[i]);

        y[i] = exponential/((1.0 + exponential)*(1.0 + exponential));
    }

    return y;
}


Tensor<type, 2> logistic_second_derivatives(const Tensor<type, 2>& x)
{
    Tensor<type, 2> y(x.dimensions());
/*
    for(int i = 0; i < x.size(); i++)
    {
        Tensor<type, 2> logistic_normal = logistic(y);
        Tensor<type, 2> logistic_deriv = logistic_derivatives(y);

       y[i] = logistic_deriv[i]*(1-2*logistic_normal[i]);
    }
*/
    return y;
}


Tensor<type, 2> binary(const Tensor<type, 2>& x)
{
    Tensor<type, 2> y(x.dimensions());
/*
    for(int i = 0; i < x.size(); i++)
    {
        x[i] < 0.5 ? y[i] = false : y [i] = true;
    }
*/
    return y;
}


Tensor<type, 2> threshold_derivatives(const Tensor<type, 2>& x)
{
    Tensor<type, 2> y(x.dimensions());
    y.setZero();

  return y;
}


Tensor<type, 1> threshold_derivatives(const Tensor<type, 1>& x)
{
    Tensor<type, 1> y(x.size());
    y.setZero();

    return y;
}


Tensor<type, 2> symmetric_threshold_derivatives(const Tensor<type, 2>& x)
{
    Tensor<type, 2> y(x.dimensions());
    y.setZero();

 return y;
}


Tensor<type, 1> symmetric_threshold_derivatives(const Tensor<type, 1>& x)
{
 Tensor<type, 1> y(x.size());
 y.setZero();

 return y;
}


Tensor<type, 2> rectified_linear_derivatives(const Tensor<type, 2>& x)
{
     const int n = x.size();

     Tensor<type, 2> derivatives(x.dimensions());
/*
     for(int i = 0; i < n; i++)
     {
         x[i] < 0.0 ? derivatives[i] = 0.0 : derivatives[i] = 1.0;
     }
*/
     return derivatives;
}


Tensor<type, 1> rectified_linear_derivatives(const Tensor<type, 1>& x)
{
     const int n = x.size();

     Tensor<type, 1> derivatives(n);

     for(int i = 0; i < n; i++)
     {
         x[i] < 0.0 ? derivatives[i] = 0.0 : derivatives[i] = 1.0;
     }

     return derivatives;
}

Tensor<type, 2> scaled_exponential_linear_derivatives(const Tensor<type, 2>& x)
{
 const int n = x.size();

 const double lambda =1.0507;
 const double alpha =1.67326;

 Tensor<type, 2> derivatives(x.dimensions());
/*
 for(int i = 0; i < n; i++)
 {
     x[i] < 0.0 ? derivatives[i] = lambda * alpha * exp(x[i]) : derivatives[i] = lambda;
 }
*/
 return derivatives;
}

Tensor<type, 1> scaled_exponential_linear_derivatives(const Tensor<type, 1>& x)
{
 const int n = x.size();

 const double lambda =1.0507;
 const double alpha =1.67326;

 Tensor<type, 1> derivatives(n);

 for(int i = 0; i < n; i++)
 {
     x[i] < 0.0 ? derivatives[i] = lambda * alpha * exp(x[i]) : derivatives[i] = lambda;
 }

 return derivatives;
}


Tensor<type, 2> soft_plus_derivatives(const Tensor<type, 2>& x)
{
 const int n = x.size();

 Tensor<type, 2> derivatives(x.dimensions());
/*
 for(int i = 0; i < n; i++)
 {
     derivatives[i] = 1/(1 + exp(-x[i]));
 }
*/
 return derivatives;
}


Tensor<type, 1> soft_plus_derivatives(const Tensor<type, 1>& x)
{
 const int n = x.size();

 Tensor<type, 1> derivatives(n);

 for(int i = 0; i < n; i++)
 {
     derivatives[i] = 1/(1 + exp(-x[i]));
 }

 return derivatives;
}


Tensor<type, 2> soft_sign_derivatives(const Tensor<type, 2>& x)
{
 const int n = x.size();

 Tensor<type, 2> derivatives(x.dimensions());
/*
 for(int i = 0; i < n; i++)
 {
    x[i] < 0.0 ? derivatives[i] = 1.0 / pow(1.0 - x[i], 2) : derivatives[i] = 1.0 / pow(1.0 + x[i], 2);
 }
*/
 return derivatives;
}


Tensor<type, 1> soft_sign_derivatives(const Tensor<type, 1>& x)
{
 const int n = x.size();

 Tensor<type, 1> derivatives(n);

 for(int i = 0; i < n; i++)
 {
    x[i] < 0.0 ? derivatives[i] = 1 / pow((1 - x[i]), 2) : derivatives[i] = 1 / pow((1 + x[i]), 2);

 }

 return derivatives;
}


Tensor<type, 2> hard_sigmoid_derivatives(const Tensor<type, 2>& x)
{
    const int n = x.size();

    Tensor<type, 2> derivatives(x.dimensions());
/*
    for(int i = 0; i < n; i++)
    {
        x[i] < -2.5 || x[i] > 2.5 ? derivatives[i] = 0.0 : derivatives[i] = 0.2;
    }
*/
    return derivatives;
}


Tensor<type, 1> hard_sigmoid_derivatives(const Tensor<type, 1>& x)
{
const int n = x.size();

    Tensor<type, 1> derivatives(n);

 for(int i = 0; i < n; i++)
 {
     x[i] < -2.5 || x[i] > 2.5 ? derivatives[i] = 0.0 : derivatives[i] = 0.2;
 }

 return derivatives;
}


Tensor<type, 2> exponential_linear_derivatives(const Tensor<type, 2>& x)
{
 const int n = x.size();

 Tensor<type, 2> derivatives(x.dimensions());

 const double alpha = 1.0;
/*
 for(int i = 0; i < n; i++)
 {
     x[i] < 0.0 ? derivatives[i] = alpha * exp(x[i]) : derivatives[i] = 1.0;
 }
*/
 return derivatives;
}


Tensor<type, 1> exponential_linear_derivatives(const Tensor<type, 1>& x)
{
 const int n = x.size();

 Tensor<type, 1> derivatives(n);

 const double alpha = 1.0;

 for(int i = 0; i < n; i++)
 {
     x[i] < 0.0 ? derivatives[i] = alpha * exp(x[i]) : derivatives[i] = 1.0;
 }

 return derivatives;
}

Tensor<type, 1> softmax(const Tensor<type, 1>& x)
{
    const int this_size = x.size();

    Tensor<type, 1> softmax(this_size);

    double sum = 0;

    for(int i = 0; i < this_size; i++) {
        sum += exp(x[i]);
    }

    for(int i = 0; i < this_size; i++) {
        softmax[i] = exp(x[i]) / sum;
    }

    return softmax;
}


Tensor<type, 2> softmax_derivatives(const Tensor<type, 2>& x)
{
 const int n = x.dimension(0);

 const int columns_number = x.dimension(1);

// vector<int> dimensions = {columns_number, columns_number, n};

// Tensor<type, 2> y(columns_number, columns_number, n);

 for(int i = 0; i < n; i ++)
 {
/*
     const Tensor<type, 1> softmax_values = softmax(x.get_matrix(0).get_row(i));

     for(int j = 0; j < columns_number; j++)
     {
         for(int k = 0; k < columns_number; k++)
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
*/
 }

// return y;

 return Tensor<type, 2>();

}


Tensor<type, 2> competitive(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    Tensor<type, 2> competitive(rows_number, columns_number);
/*
    for(int i = 0; i < rows_number; i++)
    {
        const int maximal_index = OpenNN::maximal_index(matrix.get_row(i));

        competitive(i, maximal_index) = 1;
    }
*/
    return(competitive);
}

/*
Tensor<type, 2> competitive(const Tensor<type, 2>& x)
{
    const int rows_number = x.dimension(0);
    const int columns_number = x.dimension(1);

    Tensor<type, 2> competitive(rows_number, columns_number);

    for(int i = 0; i < rows_number; i++)
    {
        const int maximal_index = OpenNN::maximal_index(x.get_matrix(0).get_row(i));

        competitive(i, maximal_index) = 1;
    }

    return competitive;
}
*/
/*
Tensor<type, 2> softmax_rows(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    Tensor<type, 2> softmax(rows_number,columns_number);

    for(int i = 0; i < rows_number; i++)
    {
        softmax.set_row(i, OpenNN::softmax(matrix.get_row(i)));
    }

    return softmax;
}
*/

Tensor<type, 2> softmax_columns(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    Tensor<type, 2> softmax(rows_number,columns_number);
/*
    for(int i = 0; i < columns_number; i++)
    {
        softmax.set_column(i, OpenNN::softmax(matrix.get_column(i)));
    }
*/
    return softmax;
}


/// This method converts the values of the vector to be binary.
/// The threshold value used is 0.5.

vector<bool> binary(const Tensor<type, 1>& vector)
{
  const int this_size = vector.size();

  std::vector<bool> result(this_size);

  for(int i = 0; i < this_size; i++) {
    if(result[i] < 0.5) {
      result[i] = false;
    } else {
      result[i] = true;
    }
  }

  return(result);
}


///Calculate the square_root of a vector

Tensor<type, 1> square_root(const Tensor<type, 1>& vector)
{
    const int this_size = vector.size();

    Tensor<type, 1>square(this_size);

    for(int i = 0; i < this_size; i++) {

        square[i]=sqrt(vector[i]);
    }


    return square;
}


/// Return the cumulative vector of this vector,
/// where each element is summed up with all the previous ones.

Tensor<type, 1> cumulative(const Tensor<type, 1>& vector)
{
  const int this_size = vector.size();

  Tensor<type, 1> cumulative(this_size);

  if(this_size > 0) {
    cumulative[0] = vector[0];

    for(int i = 1; i < this_size; i++) {
      cumulative[i] = cumulative[i - 1] + vector[i];
    }
  }

  return(cumulative);
}


/// Returns the softmax vector of this matrix,
/// whose elements sum one, and can be interpreted as probabilities.
/*
Tensor<type, 2> softmax(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

  Tensor<type, 2> softmax(rows_number, columns_number);

  for(int j = 0; j < rows_number; j++)
  {
      double sum = 0;

      for(int i = 0; i < columns_number; i++)
      {
        sum += exp(matrix(j,i));
      }

      for(int i = 0; i < columns_number; i++)
      {
        softmax(j,i) = exp(matrix(j,i)) / sum;
      }
  }

  return softmax;
}
*/

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


int factorial(const int& number)
{
    int fact = 1;

    if(number == 0)
    {
        fact = 1;
    }
    else
    {
        for(int i = 1; i <= number; i++)
        {
            fact = fact*i;
        }
    }

    return fact;
}


/// Returns a vector with the bounded elements from below of the current vector.
/// @param lower_bound Lower bound values.

Tensor<type, 1> lower_bounded(const Tensor<type, 1>& vector, const double &lower_bound)
{
  const int this_size = vector.size();

  Tensor<type, 1> bounded_vector(this_size);

  for(int i = 0; i < this_size; i++) {
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

Tensor<type, 1> lower_bounded(const Tensor<type, 1>& vector, const Tensor<type, 1>& lower_bound)
{
  const int this_size = vector.size();

#ifdef __OPENNN_DEBUG__

  const int lower_bound_size = lower_bound.size();

  if(lower_bound_size != this_size) {
    ostringstream buffer;

    buffer
        << "OpenNN Exception: vector Template.\n"
        << "vector<T> calculate_lower_bounded(const vector<T>&) const method.\n"
        << "Lower bound size must be equal to vector size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Tensor<type, 1> bounded_vector(this_size);

  // Apply lower bound

  for(int i = 0; i < this_size; i++) {
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

Tensor<type, 1> upper_bounded(const Tensor<type, 1>& vector, const double &upper_bound)
{
  const int this_size = vector.size();

  Tensor<type, 1> bounded_vector(this_size);

  for(int i = 0; i < this_size; i++) {
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

Tensor<type, 1> upper_bounded(const Tensor<type, 1>& vector, const Tensor<type, 1>&upper_bound)
{
  const int this_size = vector.size();

  

#ifdef __OPENNN_DEBUG__

  const int upper_bound_size = upper_bound.size();

  if(upper_bound_size != this_size) {
    ostringstream buffer;

    buffer
        << "OpenNN Exception: vector Template.\n"
        << "vector<T> calculate_upper_bounded(const vector<T>&) const method.\n"
        << "Upper bound size must be equal to vector size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Tensor<type, 1> bounded_vector(this_size);

  // Apply upper bound

  for(int i = 0; i < this_size; i++) {
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

Tensor<type, 1> lower_upper_bounded(const Tensor<type, 1>& vector, const double &lower_bound, const double &upper_bound)
{
  const int this_size = vector.size();

  Tensor<type, 1> bounded_vector(this_size);

  for(int i = 0; i < this_size; i++) {
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


Tensor<type, 1> lower_upper_bounded(const Tensor<type, 1>& vector, const Tensor<type, 1>&lower_bound,
                                         const Tensor<type, 1>&upper_bound)
{
  const int this_size = vector.size();



#ifdef __OPENNN_DEBUG__

  const int lower_bound_size = lower_bound.size();
  const int upper_bound_size = upper_bound.size();

  if(lower_bound_size != this_size || upper_bound_size != this_size) {
    ostringstream buffer;

    buffer << "OpenNN Exception: vector Template.\n"
           << "vector<T> calculate_lower_upper_bounded(const vector<T>&, const "
              "vector<T>&) const method.\n"
           << "Lower and upper bound sizes must be equal to vector size.\n";

    throw logic_error(buffer.str());
  }

#endif

  Tensor<type, 1> bounded_vector(this_size);

  // Apply lower and upper bounds

  for(int i = 0; i < this_size; i++) {
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


Tensor<type, 2> lower_bounded(const Tensor<type, 2>& matrix, const double & lower_bound)
{
    const int this_size = matrix.size();

    Tensor<type, 2> bounded_matrix(matrix);
/*
    for(int i = 0; i < this_size; i++)
    {
      if(matrix[i] < lower_bound)
      {
        bounded_matrix[i] = lower_bound;
      }
    }
*/
    return(bounded_matrix);
}


Tensor<type, 2> upper_bounded(const Tensor<type, 2>& matrix, const double & upper_bound)
{
    const int this_size = matrix.size();

    Tensor<type, 2> bounded_matrix(matrix);
/*
    for(int i = 0; i < this_size; i++)
    {
      if(matrix[i] > upper_bound)
      {
        bounded_matrix[i] = upper_bound;
      }
    }
*/
    return(bounded_matrix);
}


Tensor<type, 2> lower_upper_bounded(const Tensor<type, 2>& matrix, const Tensor<type, 1>& lower_bounds, const Tensor<type, 1>& upper_bounds)
{
    Tensor<type, 2> bounded_matrix(matrix);

    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    for(int i = 0; i < rows_number; i++)
    {
        for(int j = 0; j < columns_number; j++)
        {
            if(bounded_matrix(i,j) < lower_bounds[j]) bounded_matrix(i,j) = lower_bounds[j];
            else if(bounded_matrix(i,j) > upper_bounds[j]) bounded_matrix(i,j) = upper_bounds[j];
        }
    }

    return bounded_matrix;
}

/*
Tensor<type, 2>lower_upper_bounded(const Tensor<type, 2>& tensor, const Tensor<type, 1>& lower_bounds, const Tensor<type, 1>& upper_bounds)
{
    Tensor<type, 2> bounded_tensor(tensor);

    const int rows_number = tensor.dimension(0);
    const int columns_number = tensor.dimension(1);

    for(int i = 0; i < rows_number; i++)
    {
        for(int j = 0; j < columns_number; j++)
        {
            if(bounded_tensor(i,j) < lower_bounds[j]) bounded_tensor(i,j) = lower_bounds[j];
            else if(bounded_tensor(i,j) > upper_bounds[j]) bounded_tensor(i,j) = upper_bounds[j];
        }
    }

    return bounded_tensor;
}
*/

/// Returns the gradient of the vector norm.

Tensor<type, 1> sign(const Tensor<type, 1>& vector)
{
  const int this_size = vector.size();  

  Tensor<type, 1> sign_vector(this_size);

  for(int i = 0; i < this_size; i++)
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

Tensor<type, 1> normalized(const Tensor<type, 1>& vector)
{
  const int this_size = vector.size();

  Tensor<type, 1> normalized(this_size);
/*
  const double norm = l2_norm(vector);

  if(norm == 0.0) {
    normalized.setZero();
  } else {
    normalized = vector / norm;
  }
*/
  return normalized;
}


Tensor<type, 1> absolute_value(const Tensor<type, 1>& vector)
{
  const int this_size = vector.size();

  Tensor<type, 1> absolute_value(this_size);

  for(int i = 0; i < this_size; i++) {
    if(vector[i] > 0) {
      absolute_value[i] = vector[i];
    } else {
      absolute_value[i] = -vector[i];
    }
  }

  return(absolute_value);
}


Tensor<type, 2> normalized_columns(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    Tensor<type, 2> softmax(rows_number,columns_number);
/*
    for(int i = 0; i < columns_number; i++)
    {
        softmax.set_column(i, normalized(matrix.get_column(i)));
    }
*/
    return softmax;
}


/// Returns a matrix with the absolute values of this matrix.

Tensor<type, 2> absolute_value(const Tensor<type, 2>& matrix)
{
    const Index rows_number = matrix.dimension(0);
    const Index columns_number = matrix.dimension(1);

    Tensor<type, 2> absolute_value(rows_number, columns_number);
/*
    for(int i = 0; i < matrix.size(); i++)
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
*/
    return absolute_value;
}


void hard_sigmoid(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
    const int n = x.size();
/*
    #pragma omp parallel for

     for(int i = 0; i < n; i++)
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
*/
}


void hyperbolic_tangent(const ThreadPoolDevice& thread_pool_device, const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
    y.device(thread_pool_device) = x.tanh();
}


void logistic(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
    const int n = x.size();
/*
    #pragma omp parallel for

    for(int i = 0; i < n; i++)
    {
        y[i] = 1.0 / (1.0 + exp(-x[i]));
    }
*/
}


void linear(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
    y = x;
}


void threshold(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
    const int n = x.size();
/*
    #pragma omp parallel for

    for(int i = 0; i < n; i++)
    {
         y[i] = x[i] < 0 ? -1.0 : 1.0;
    }
*/
}


void symmetric_threshold(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
    const int n = x.size();
/*
    #pragma omp parallel for

     for(int i = 0; i < n; i++)
     {
         y[i] = x[i] < 0 ? -1.0 : 1.0;
     }
*/
}


void rectified_linear(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
    const int n = x.size();
/*
    #pragma omp parallel for

    for(int i = 0; i < n; i++)
    {
        y[i] = x[i] < 0.0 ? 0.0 : x[i];
    }
*/
}


void scaled_exponential_linear(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
    const int n = x.size();

    const double lambda = 1.0507;
    const double alpha = 1.67326;
/*
    #pragma omp parallel for

    for(int i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = lambda * alpha * (exp(x[i]) - 1) : y[i] = lambda * x[i];
    }
*/
}


void soft_plus(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
    const int n = x.size();
/*
    #pragma omp parallel for

    for(int i = 0; i < n; i++)
    {
        y[i] = log(1.0 + exp(x[i]));
    }
*/
}


void soft_sign(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
    const int n = x.size();
/*
    #pragma omp parallel for

    for(int i = 0; i < n; i++)
    {
       x[i] < 0.0 ? y[i] = x[i] / (1.0 - x[i]) : y[i] = x[i] / (1.0 + x[i]);
    }
*/
}


void exponential_linear(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
    const int n = x.size();

    const double alpha = 1.0;
/*
    #pragma omp parallel for

    for(int i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = alpha * (exp(x[i]) - 1) : y[i] = x[i];
    }
*/
}


void logistic_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
    const int n = x.size();
/*
    #pragma omp parallel for

    for(int i = 0; i < n; i++)
    {
        const double exponential = exp(-x[i]);

        y[i] = exponential/((1.0 + exponential)*(1.0 + exponential));
    }
*/
}


void threshold_derivatives(const Tensor<type, 2>&, Tensor<type, 2>& y)
{
    y.setZero();
}


void symmetric_threshold_derivatives(const Tensor<type, 2>&, Tensor<type, 2>& y)
{
    y.setZero();
}


void linear_derivatives(const Tensor<type, 2>&, Tensor<type, 2>& y)
{
    y.setConstant(1.0);
}


void hyperbolic_tangent_derivatives(const ThreadPoolDevice& thread_pool_device, const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
    y.device(thread_pool_device) = x.constant(1.0) - x.tanh()*x.tanh();
}


void rectified_linear_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
/*
    const int n = x.size();

    #pragma omp parallel for

    for(int i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = 0.0 : y[i] = 1.0;
    }
*/
}


void scaled_exponential_linear_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
/*
    const int n = x.size();

    const double lambda =1.0507;
    const double alpha =1.67326;

    #pragma omp parallel for

    for(int i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = lambda * alpha * exp(x[i]) : y[i] = lambda;
    }
*/
}


void soft_plus_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
/*
    const int n = x.size();

    #pragma omp parallel for

    for(int i = 0; i < n; i++)
    {
        y[i] = 1.0/(1.0 + exp(-x[i]));
    }
*/
}


void soft_sign_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
/*
    const int n = x.size();

    #pragma omp parallel for

    for(int i = 0; i < n; i++)
    {
       x[i] < 0.0 ? y[i] = 1.0 / pow(1.0 - x[i], 2) : y[i] = 1.0 / pow(1.0 + x[i], 2);
    }
*/
}


void hard_sigmoid_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
/*
    const int n = x.size();

    #pragma omp parallel for

    for(int i = 0; i < n; i++)
    {
        x[i] < -2.5 || x[i] > 2.5 ? y[i] = 0.0 : y[i] = 0.2;
    }
*/
}


void exponential_linear_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
/*
    const int n = x.size();

    const double alpha = 1.0;

    #pragma omp parallel for

    for(int i = 0; i < n; i++)
    {
        x[i] < 0.0 ? y[i] = alpha * exp(x[i]) : y[i] = 1.0;
    }
*/
}


/// @todo Fails

void softmax_derivatives(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
#ifdef __OPENNN_DEBUG__

    if(x.dimension(0) != y.dimension(0))
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: Functions.\n"
              << "void softmax_derivatives(const Tensor<type, 2>&, Tensor<type, 2>&) method.\n"
              << "Number of rows in x must be equal to number of rows in d.\n";

       throw logic_error(buffer.str());
    }

#endif

    const int n = x.dimension(0);

    const int columns_number = x.dimension(1);

    #pragma omp parallel for

    for(int i = 0; i < n; i ++)
    {
/*
        const Tensor<type, 1> softmax_values = softmax(x.get_matrix(0).get_row(i));

        for(int j = 0; j < columns_number; j++)
        {
            for(int k = 0; k < columns_number; k++)
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
*/
    }
}


void binary(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
/*
    const int n = x.size();

    #pragma omp parallel for

    for(int i = 0; i < n; i++)
    {
        x[i] < 0.5 ? y[i] = false : y [i] = true;
    }
*/
}


void competitive(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
    const int rows_number = x.dimension(0);
/*
    #pragma omp parallel for

    for(int i = 0; i < rows_number; i++)
    {
        const int maximal_index = OpenNN::maximal_index(x.get_matrix(0).get_row(i));

        y(i, maximal_index) = 1;
    }
*/
}


void softmax(const Tensor<type, 2>& x, Tensor<type, 2>& y)
{
    const int rows_number = x.dimension(0);
    const int columns_number = x.dimension(1);

    #pragma omp parallel for

  for(int j = 0; j < rows_number; j++)
  {
      double sum = 0.0;

      for(int i = 0; i < columns_number; i++)
      {
        sum += exp(x(j,i));
      }

      for(int i = 0; i < columns_number; i++)
      {
        y(j,i) = exp(x(j,i)) / sum;
      }
  }
}

}
