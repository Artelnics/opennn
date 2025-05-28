//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C O R R E L A T I O N S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef CORRELATIONS_H
#define CORRELATIONS_H

namespace opennn
{


struct Correlation
{
    enum class Method{Pearson, Spearman};

    enum class Form{Linear, Logistic, Logarithmic, Exponential, Power};

    Correlation() {}

    void set_perfect()
    {
        r = type(1);
        a = type(0);
        b = type(1);

        upper_confidence = type(1);
        lower_confidence = type(1);
        form = Correlation::Form::Linear;
    }

    string write_type() const
    {
        switch(form)
        {
        case Form::Linear: return "linear";
        case Form::Logistic: return "logistic";
        case Form::Logarithmic: return "logarithmic";
        case Form::Exponential: return "exponential";
        case Form::Power: return "power";
        default:
            return string();
        }
    }

    void print() const
    {
        cout << "Correlation" << endl
             << "Type: " << write_type() << endl
             << "a: " << a << endl
             << "b: " << b << endl
             << "r: " << r << endl
             << "Lower confidence: " << lower_confidence << endl
             << "Upper confidence: " << upper_confidence << endl;
    }

    type a = type(NAN);
    type b = type(NAN);
    type r = type(NAN);

    type lower_confidence = type(NAN);
    type upper_confidence = type(NAN);

    Method method = Method::Pearson;
    Form form = Form::Linear;
};


Correlation linear_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

Correlation logarithmic_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

Correlation exponential_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

Correlation power_correlation(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

Correlation logistic_correlation_vector_vector(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

Correlation logistic_correlation_vector_matrix(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 2>&);

Correlation logistic_correlation_matrix_vector(const ThreadPoolDevice*, const Tensor<type, 2>&, const Tensor<type, 1>&);

Correlation logistic_correlation_matrix_matrix(const ThreadPoolDevice*, const Tensor<type, 2>&, const Tensor<type, 2>&);

Correlation correlation(const ThreadPoolDevice*, const Tensor<type, 2>&, const Tensor<type, 2>&);

// Spearman correlation

Correlation linear_correlation_spearman(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

Tensor<type, 1> calculate_spearman_ranks(const Tensor<type, 1>&);

Correlation logistic_correlation_vector_vector_spearman(const ThreadPoolDevice*, const Tensor<type, 1>&, const Tensor<type, 1>&);

Correlation correlation_spearman(const ThreadPoolDevice*, const Tensor<type, 2>&, const Tensor<type, 2>&);

// Confidence interval

type r_correlation_to_z_correlation(const type&);
type z_correlation_to_r_correlation(const type&);

Tensor<type, 1> confidence_interval_z_correlation(const type&, const Index&);


// Time series correlation

Tensor<type, 1> autocorrelations(const ThreadPoolDevice*,
                                 const Tensor<type, 1>&,
                                 const Index&  = 10);

Tensor<type, 1> cross_correlations(const ThreadPoolDevice*,
                                   const Tensor<type, 1>&,
                                   const Tensor<type, 1>&,
                                   const Index&);

Tensor<type, 2> get_correlation_values(const Tensor<Correlation, 2>&);

// Missing values

pair<Tensor<type, 1>, Tensor<type, 1>> filter_missing_values_vector_vector(const Tensor<type, 1>&, const Tensor<type, 1>&);
pair<Tensor<type, 1>, Tensor<type, 2>> filter_missing_values_vector_matrix(const Tensor<type, 1>&, const Tensor<type, 2>&);
pair<Tensor<type, 1>, Tensor<type, 2>> filter_missing_values_matrix_vector(const Tensor<type, 2>&, const Tensor<type, 1>&);
pair<Tensor<type, 2>, Tensor<type, 2>> filter_missing_values_matrix_matrix(const Tensor<type, 2>&, const Tensor<type, 2>&);

}

#endif


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
