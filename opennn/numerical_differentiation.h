//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   N U M E R I C A L   D I F F E R E N T I A T I O N   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef NUMERICALDIFFERENTIATION_H
#define NUMERICALDIFFERENTIATION_H

// System includes

#include<iostream>
#include<vector>
#include<vector>
#include<limits>
#include<cstddef>

// OpenNN includes

#include "config.h"

namespace opennn
{

/// This class contains methods for numerical differentiation of functions.
/// In particular it implements the forward and central differences methods for derivatives, Jacobians, hessians or hessian forms.

class NumericalDifferentiation
{

public:

    // Constructors

    explicit NumericalDifferentiation();

    /// Enumeration of the available methods for numerical differentiation.

    const Index& get_precision_digits() const;

    const bool& get_display() const;

    void set(const NumericalDifferentiation&);

    void set_precision_digits(const Index&);

    void set_display(const bool&);

    void set_default();

    type calculate_eta() const;

    type calculate_h(const type&) const;
    Tensor<type, 1> calculate_h(const Tensor<type, 1>&) const;
    Tensor<type, 2> calculate_h(const Tensor<type, 2>&) const;
    Tensor<type, 4> calculate_h(const Tensor<type, 4>&) const;

    template<class T>
    type calculate_derivatives(const T& t, type(T::*f)(const type&) const , const type& x) const
    {
        const type h = calculate_h(x);

        const type y_forward = (t.*f)(x+h);

        const type y_backward = (t.*f)(x-h);

        const type d = (y_forward - y_backward)/(static_cast<type>(2.0)*h);

        return d;
    }


    /// Returns the derivatives of a vector function using the central differences method.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 1> calculate_derivatives(const T& t, Tensor<type, 1>(T::*f)(const Tensor<type, 1>&) const, const Tensor<type, 1>& x) const
    {

        const Tensor<type, 1> h = calculate_h(x);

        const Tensor<type, 1> x_forward = x + h;
        const Tensor<type, 1> x_backward = x - h;

        const Tensor<type, 1> y_forward = (t.*f)(x_forward);
        const Tensor<type, 1> y_backward = (t.*f)(x_backward);

        const Tensor<type, 1> y = (t.*f)(x);

        const Tensor<type, 1> d = (y_forward - y_backward)/(static_cast<type>(2.0)*h);

        return d;
    }


    template<class T>
    Tensor<type, 2> calculate_derivatives(const T& t, Tensor<type, 2>(T::*f)(const Tensor<type, 2>&) const, const Tensor<type, 2>& x) const
    {
        const Tensor<type, 2> h = calculate_h(x);

        const Tensor<type, 2> x_forward = x + h;
        const Tensor<type, 2> x_backward = x - h;

        const Tensor<type, 2> y_forward = (t.*f)(x_forward);
        const Tensor<type, 2> y_backward = (t.*f)(x_backward);

        const Tensor<type, 2> y = (t.*f)(x);

        const Tensor<type, 2> d = (y_forward - y_backward)/(static_cast<type>(2.0)*h);

        return d;
    }


    template<class T>
    Tensor<type, 4> calculate_derivatives(const T& t, Tensor<type, 4>(T::*f)(const Tensor<type, 4>&) const, const Tensor<type, 4>& x) const
    {
        const Tensor<type, 4> h = calculate_h(x);

        const Tensor<type, 4> x_forward = x + h;
        const Tensor<type, 4> x_backward = x - h;

        const Tensor<type, 4> y_forward = (t.*f)(x_forward);
        const Tensor<type, 4> y_backward = (t.*f)(x_backward);

        const Tensor<type, 4> y = (t.*f)(x);

        const Tensor<type, 4> d = (y_forward - y_backward)/(static_cast<type>(2.0)*h);

        return d;
    }


    /// Returns the derivatives of a vector function using the central differences method.
    /// The function to be differentiated is of the following form: Tensor<type, 1> f(const Index&, const Tensor<type, 1>&) const.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param dummy: Dummy integer for the method prototype.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 1> calculate_derivatives(const T& t, Tensor<type, 1>(T::*f)(const Index&, const Tensor<type, 1>&) const, const Index& dummy, const Tensor<type, 1>& x) const
    {
        const Tensor<type, 1> h = calculate_h(x);

        const Tensor<type, 1> x_forward = x + h;
        const Tensor<type, 1> x_backward = x - h;

        const Tensor<type, 1> y_forward = (t.*f)(dummy, x_forward);
        const Tensor<type, 1> y_backward = (t.*f)(dummy, x_backward);

        const Tensor<type, 1> d = (y_forward - y_backward)/(static_cast<type>(2.0)*h);

        return d;
    }


    template<class T>
    Tensor<type, 2> calculate_derivatives(const T& t, void(T::*f)(const Tensor<type, 2>&, Tensor<type, 2>&) const, const Index& dummy, const Tensor<type, 2>& x) const
    {
        const Index rn = x.dimension(0);
        const Index cn = x.dimension(1);

        const Tensor<type, 2> h = calculate_h(x);

        const Tensor<type, 2> x_forward = x + h;
        const Tensor<type, 2> x_backward = x - h;

        Tensor<type, 2> y_forward(rn,cn);
        (t.*f)(x_forward, y_forward);
        Tensor<type, 2> y_backward(rn,cn);
        (t.*f)(x_backward, y_backward);

        const Tensor<type, 2> d = (y_forward - y_backward)/(static_cast<type>(2.0)*h);

        return d;
    }


    template<class T>
    Tensor<type, 4> calculate_derivatives(const T& t, void(T::*f)(const Tensor<type, 4>&, Tensor<type, 4>&) const, const Index& dummy, const Tensor<type, 4>& x) const
    {
        const Index rn = x.dimension(0);
        const Index cn = x.dimension(1);
        const Index kn = x.dimension(2);
        const Index in = x.dimension(3);

        const Tensor<type, 4> h = calculate_h(x);

        const Tensor<type, 4> x_forward = x + h;
        const Tensor<type, 4> x_backward = x - h;

        Tensor<type, 4> y_forward(rn,cn, kn, in);
        (t.*f)(x_forward, y_forward);
        Tensor<type, 4> y_backward(rn,cn, kn, in);
        (t.*f)(x_backward, y_backward);

        const Tensor<type, 4> d = (y_forward - y_backward)/(static_cast<type>(2.0)*h);

        return d;
    }


    /// Returns the second derivative of a function using the central differences method.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param x: Differentiation point.

    template<class T>
    type calculate_second_derivatives(const T& t, type(T::*f)(const type&) const , const type& x) const
    {
        const type h = calculate_h(x);

        const type x_forward_2 = x + static_cast<type>(2.0)*h;

        const type y_forward_2 = (t.*f)(x_forward_2);

        const type x_forward = x + h;

        const type y_forward = (t.*f)(x_forward);

        const type y = (t.*f)(x);

        const type x_backward = x - h;

        const type y_backward = (t.*f)(x_backward);

        const type x_backward_2 = x - static_cast<type>(2.0)*h;

        const type y_backward_2 = (t.*f)(x_backward_2);

        const type d2 = (-y_forward_2 + type(16.0)*y_forward - type(30.0)*y + type(16.0)*y_backward - y_backward_2)/(type(12.0)*pow(h, type(2)));

        return d2;
    }


    /// Returns the second derivative of a vector function using the central differences method.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 1> calculate_second_derivatives(const T& t, Tensor<type, 1>(T::*f)(const Tensor<type, 1>&) const, const Tensor<type, 1>& x) const
    {
        const Tensor<type, 1> h = calculate_h(x);

        const Tensor<type, 1> x_forward = x + h;
        const Tensor<type, 1> x_forward_2 = x + h*static_cast<type>(2.0);

        const Tensor<type, 1> x_backward = x - h;
        const Tensor<type, 1> x_backward_2 = x - h*static_cast<type>(2.0);

        const Tensor<type, 1> y = (t.*f)(x);

        const Tensor<type, 1> y_forward = (t.*f)(x_forward);
        const Tensor<type, 1> y_forward_2 = (t.*f)(x_forward_2);

        const Tensor<type, 1> y_backward = (t.*f)(x_backward);
        const Tensor<type, 1> y_backward_2 = (t.*f)(x_backward_2);

        return ((y_forward_2*type(-1) + y_forward*type(16) + y*type(-30) + y_backward*type(16) + y_backward_2*type(-1))/(h*h*type(12)));
    }


    /// Returns the second derivatives of a vector function using the central differences method.
    /// The function to be differentiated is of the following form: Tensor<type, 1> f(const Index&, const Tensor<type, 1>&) const.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param dummy: Dummy integer for the method prototype.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 1> calculate_second_derivatives(const T& t,
                                                 Tensor<type, 1>(T::*f)(const Index&, const Tensor<type, 1>&) const,
                                                 const Index& dummy, const Tensor<type, 1>& x) const
    {
        const Tensor<type, 1> h = calculate_h(x);

        const Tensor<type, 1> x_forward = x + h;
        const Tensor<type, 1> x_forward_2 = x + h*static_cast<type>(2.0);

        const Tensor<type, 1> x_backward = x - h;
        const Tensor<type, 1> x_backward_2 = x - h*static_cast<type>(2.0);

        const Tensor<type, 1> y = (t.*f)(dummy, x);

        const Tensor<type, 1> y_forward = (t.*f)(dummy, x_forward);
        const Tensor<type, 1> y_forward_2 = (t.*f)(dummy, x_forward_2);

        const Tensor<type, 1> y_backward = (t.*f)(dummy, x_backward);
        const Tensor<type, 1> y_backward_2 = (t.*f)(dummy, x_backward_2);

        return (y_forward_2*type(-1) + y_forward*type(16) + y*type(-30) + y_backward*type(16) + y_backward_2*type(-1))/(h*h*type(12));
    }


    /// Returns the gradient of a function of several dimensions using the central differences method.
    /// The function to be differentiated is of the following form: type f(const Tensor<type, 1>&) const.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 1> calculate_gradient(const T& t, type (T::*f)(const Tensor<type, 1>&) const, const Tensor<type, 1>& x) const
    {

        const Index n = x.size();

        type h;

        Tensor<type, 1> x_forward = x;
        Tensor<type, 1> x_backward(x);

        type y_forward;
        type y_backward;

        Tensor<type, 1> g(n);

        for(Index i = 0; i < n; i++)
        {
            h = calculate_h(x(i));

            x_forward(i) += h;

            y_forward = (t.*f)(x_forward);

            x_forward(i) -= h;
            x_backward(i) -= h;

            y_backward = (t.*f)(x_backward);
            x_backward(i) += h;

            g(i) = (y_forward - y_backward)/(type(2.0)*h);

        }

        return g;
    }

    /// Returns the gradient of a function of several dimensions using the central differences method.
    /// The function to be differentiated is of the following form: type f(const Tensor<type, 1>&) const.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 1> calculate_gradient(const T& t, type(T::*f)(const Tensor<type, 1>&), const Tensor<type, 1>& x) const
    {
        const Index n = x.size();

        type h;

        Tensor<type, 1> x_forward(x);
        Tensor<type, 1> x_backward(x);

        type y_forward;
        type y_backward;

        Tensor<type, 1> g(n);

        for(Index i = 0; i < n; i++)
        {
            h = calculate_h(x(i));

            x_forward(i) += h;
            y_forward = (t.*f)(x_forward);
            x_forward(i) -= h;

            x_backward(i) -= h;
            y_backward = (t.*f)(x_backward);
            x_backward(i) += h;

            g(i) = (y_forward - y_backward)/(static_cast<type>(2.0)*h);
        }

        return g;
    }

    /// Returns the gradient of a function of several dimensions using the central differences method.
    /// The function to be differentiated is of the following form: type f(const Tensor<type, 1>&, const Tensor<type, 1>&) const.
    /// The first vector argument is dummy, differentiation is performed with respect to the second vector argument.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param dummy: Dummy vector for the method prototype.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 1> calculate_gradient(const T& t, type(T::*f)(const Tensor<type, 1>&, const Tensor<type, 1>&) const, const Tensor<type, 1>& dummy, const Tensor<type, 1>& x) const
    {
        const Index n = x.size();

        type h;

        Tensor<type, 1> x_forward(x);
        Tensor<type, 1> x_backward(x);

        type y_forward;
        type y_backward;

        Tensor<type, 1> g(n);

        for(Index i = 0; i < n; i++)
        {
            h = calculate_h(x(i));

            x_forward(i) += h;
            y_forward = (t.*f)(dummy, x_forward);
            x_forward(i) -= h;

            x_backward(i) -= h;
            y_backward = (t.*f)(dummy, x_backward);
            x_backward(i) += h;

            g(i) = (y_forward - y_backward)/(static_cast<type>(2.0)*h);
        }

        return g;
    }

    /// Returns the gradient of a function of several dimensions using the central differences method.
    /// The function to be differentiated is of the following form: type f(const Index&, const Tensor<type, 1>&) const.
    /// The first integer argument is used for the function definition, differentiation is performed with respect to the second vector argument.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param dummy: Dummy integer for the method prototype.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 1> calculate_gradient(const T& t, type(T::*f)(const Index&, const Tensor<type, 1>&) const, const Index& dummy, const Tensor<type, 1>& x) const
    {
        const Index n = x.size();

        type h;

        Tensor<type, 1> x_forward(x);
        Tensor<type, 1> x_backward(x);

        type y_forward;
        type y_backward;

        Tensor<type, 1> g(n);

        for(Index i = 0; i < n; i++)
        {
            h = calculate_h(x(i));

            x_forward(i) += h;
            y_forward = (t.*f)(dummy, x_forward);
            x_forward(i) -= h;

            x_backward(i) -= h;
            y_backward = (t.*f)(dummy, x_backward);
            x_backward(i) += h;

            g(i) = (y_forward - y_backward)/(static_cast<type>(2.0)*h);
        }

        return g;
    }


    /// Returns the gradient of a function of several dimensions using the central differences method.
    /// The function to be differentiated is of the following form: type f(const Index&, const Tensor<type, 1>&) const.
    /// The first integer argument is used for the function definition, differentiation is performed with respect to the second vector argument.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param dummy: Dummy integer for the method prototype.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 1> calculate_gradient(const T& t, Tensor<type, 1>(T::*f)(const Index&, const Tensor<type, 2>&) const, const Index& dummy, const Tensor<type, 2>& x) const
    {
        const Index n = x.size();

        type h;

        Tensor<type, 1> x_forward(x);
        Tensor<type, 1> x_backward(x);

        type y_forward;
        type y_backward;

        Tensor<type, 1> g(n);

        for(Index i = 0; i < n; i++)
        {
            h = calculate_h(x(i));

            x_forward(i) += h;
            y_forward = (t.*f)(dummy, x_forward);
            x_forward(i) -= h;

            x_backward(i) -= h;
            y_backward = (t.*f)(dummy, x_backward);
            x_backward(i) += h;

            g(i) = (y_forward - y_backward)/(static_cast<type>(2.0)*h);
        }

        return g;
    }


    /// Returns the gradient of a function of several dimensions using the central differences method.
    /// The function to be differentiated is of the following form: type f(const Index&, const Tensor<type, 1>&) const.
    /// The first integer argument is used for the function definition, differentiation is performed with respect to the second vector argument.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param dummy: Dummy integer for the method prototype.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 1> calculate_gradient(const T& t, type(T::*f)(const Tensor<Index, 1>&, const Tensor<type, 1>&) const, const Tensor<Index, 1>& dummy, const Tensor<type, 1>& x) const
    {
        const Index n = x.size();

        type h;
        Tensor<type, 1> x_forward(x);
        Tensor<type, 1> x_backward(x);

        type y_forward;
        type y_backward;

        Tensor<type, 1> g(n);

        for(Index i = 0; i < n; i++)
        {
            h = calculate_h(x(i));

            x_forward(i) += h;
            y_forward = (t.*f)(dummy, x_forward);
            x_forward(i) -= h;

            x_backward(i) -= h;
            y_backward = (t.*f)(dummy, x_backward);
            x_backward(i) += h;

            g(i) = (y_forward - y_backward)/(h*type(2));
        }

        return g;
    }


    template<class T>
    Tensor<type, 2> calculate_gradient_matrix(const T& t, Tensor<type, 1>(T::*f)(const Index&, const Tensor<type, 2>&) const, const Index& integer, const Tensor<type, 2>& x) const
    {
        const Index rows_number = x.dimension(0);
        const Index columns_number = x.dimension(1);

        Tensor<type, 2> gradient(rows_number, columns_number);

        type h;
        Tensor<type, 2> x_forward(x);
        Tensor<type, 2> x_backward(x);

        type y_forward;
        type y_backward;

        for(Index i = 0; i < rows_number; i++)
        {
            for(Index j = 0; j < columns_number; j++)
            {
                h = calculate_h(x(i,j));

                x_forward(i,j) += h;
                y_forward = (t.*f)(integer, x_forward)(i);
                x_forward(i,j) -= h;

                x_backward(i,j) -= h;
                y_backward = (t.*f)(integer, x_backward)(i);
                x_backward(i,j) += h;

                gradient(i,j) = (y_forward - y_backward)/(static_cast<type>(2.0)*h);
            }
        }

        return gradient;
    }


    /// Returns the hessian matrix of a function of several dimensions using the central differences method.
    /// The function to be differentiated is of the following form: type f(const Tensor<type, 1>&) const.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 2> calculate_hessian(const T& t, type(T::*f)(const Tensor<type, 1>&) const, const Tensor<type, 1>& x) const
    {
        const Index n = x.size();

        type y = (t.*f)(x);

        Tensor<type, 2> H(n, n);

        type h_i;
        type h_j;

        Tensor<type, 1> x_backward_2i(x);
        Tensor<type, 1> x_backward_i(x);

        Tensor<type, 1> x_forward_i(x);
        Tensor<type, 1> x_forward_2i(x);

        Tensor<type, 1> x_backward_ij(x);
        Tensor<type, 1> x_forward_ij(x);

        Tensor<type, 1> x_backward_i_forward_j(x);
        Tensor<type, 1> x_forward_i_backward_j(x);

        type y_backward_2i;
        type y_backward_i;

        type y_forward_i;
        type y_forward_2i;

        type y_backward_ij;
        type y_forward_ij;

        type y_backward_i_forward_j;
        type y_forward_i_backward_j;

        for(Index i = 0; i < n; i++)
        {
            h_i = calculate_h(x(i));

            x_backward_2i(i) -= static_cast<type>(2.0)*h_i;
            y_backward_2i = (t.*f)(x_backward_2i);
            x_backward_2i(i) += static_cast<type>(2.0)*h_i;

            x_backward_i(i) -= h_i;
            y_backward_i = (t.*f)(x_backward_i);
            x_backward_i(i) += h_i;

            x_forward_i(i) += h_i;
            y_forward_i = (t.*f)(x_forward_i);
            x_forward_i(i) -= h_i;

            x_forward_2i(i) += static_cast<type>(2.0)*h_i;
            y_forward_2i = (t.*f)(x_forward_2i);
            x_forward_2i(i) -= static_cast<type>(2.0)*h_i;

            H(i,i) = (-y_forward_2i + type(16.0)*y_forward_i - type(30.0)*y + type(16.0)*y_backward_i - y_backward_2i)/(type(12.0)*pow(h_i, type(2)));

            for(Index j = i; j < n; j++)
            {
                h_j = calculate_h(x(j));

                x_backward_ij(i) -= h_i;
                x_backward_ij(j) -= h_j;
                y_backward_ij = (t.*f)(x_backward_ij);
                x_backward_ij(i) += h_i;
                x_backward_ij(j) += h_j;

                x_forward_ij(i) += h_i;
                x_forward_ij(j) += h_j;
                y_forward_ij = (t.*f)(x_forward_ij);
                x_forward_ij(i) -= h_i;
                x_forward_ij(j) -= h_j;

                x_backward_i_forward_j(i) -= h_i;
                x_backward_i_forward_j(j) += h_j;
                y_backward_i_forward_j = (t.*f)(x_backward_i_forward_j);
                x_backward_i_forward_j(i) += h_i;
                x_backward_i_forward_j(j) -= h_j;

                x_forward_i_backward_j(i) += h_i;
                x_forward_i_backward_j(j) -= h_j;
                y_forward_i_backward_j = (t.*f)(x_forward_i_backward_j);
                x_forward_i_backward_j(i) -= h_i;
                x_forward_i_backward_j(j) += h_j;

                H(i,j) = (y_forward_ij - y_forward_i_backward_j - y_backward_i_forward_j + y_backward_ij)/(type(4.0)*h_i*h_j);
            }
        }

        for(Index i = 0; i < n; i++)
        {
            for(Index j = 0; j < i; j++)
            {
                H(i,j) = H(j,i);
            }
        }

        return H;
    }


    /// Returns the hessian matrix of a function of several dimensions using the central differences method.
    /// The function to be differentiated is of the following form: type f(const Tensor<type, 1>&, const Tensor<type, 1>&) const.
    /// The first vector argument is dummy, differentiation is performed with respect to the second vector argument.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param dummy: Dummy vector for the method prototype.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 2> calculate_hessian(const T& t, type(T::*f)(const Tensor<type, 1>&, const Tensor<type, 1>&) const, const Tensor<type, 1>& dummy, const Tensor<type, 1>& x) const
    {
        const Index n = x.size();

        type y = (t.*f)(dummy, x);

        Tensor<type, 2> H(n, n);

        type h_i;
        type h_j;

        Tensor<type, 1> x_backward_2i(x);
        Tensor<type, 1> x_backward_i(x);

        Tensor<type, 1> x_forward_i(x);
        Tensor<type, 1> x_forward_2i(x);

        Tensor<type, 1> x_backward_ij(x);
        Tensor<type, 1> x_forward_ij(x);

        Tensor<type, 1> x_backward_i_forward_j(x);
        Tensor<type, 1> x_forward_i_backward_j(x);

        type y_backward_2i;
        type y_backward_i;

        type y_forward_i;
        type y_forward_2i;

        type y_backward_ij;
        type y_forward_ij;

        type y_backward_i_forward_j;
        type y_forward_i_backward_j;

        for(Index i = 0; i < n; i++)
        {
            h_i = calculate_h(x(i));

            x_backward_2i(i) -= static_cast<type>(2.0)*h_i;
            y_backward_2i = (t.*f)(dummy, x_backward_2i);
            x_backward_2i(i) += static_cast<type>(2.0)*h_i;

            x_backward_i(i) -= h_i;
            y_backward_i = (t.*f)(dummy, x_backward_i);
            x_backward_i(i) += h_i;

            x_forward_i(i) += h_i;
            y_forward_i = (t.*f)(dummy, x_forward_i);
            x_forward_i(i) -= h_i;

            x_forward_2i(i) += static_cast<type>(2.0)*h_i;
            y_forward_2i = (t.*f)(dummy, x_forward_2i);
            x_forward_2i(i) -= static_cast<type>(2.0)*h_i;

            H(i,i) = (-y_forward_2i + type(16.0)*y_forward_i - type(30.0)*y + type(16.0)*y_backward_i - y_backward_2i)/(type(12.0)*pow(h_i, type(2)));

            for(Index j = i; j < n; j++)
            {
                h_j = calculate_h(x(j));

                x_backward_ij(i) -= h_i;
                x_backward_ij(j) -= h_j;
                y_backward_ij = (t.*f)(dummy, x_backward_ij);
                x_backward_ij(i) += h_i;
                x_backward_ij(j) += h_j;

                x_forward_ij(i) += h_i;
                x_forward_ij(j) += h_j;
                y_forward_ij = (t.*f)(dummy, x_forward_ij);
                x_forward_ij(i) -= h_i;
                x_forward_ij(j) -= h_j;

                x_backward_i_forward_j(i) -= h_i;
                x_backward_i_forward_j(j) += h_j;
                y_backward_i_forward_j = (t.*f)(dummy, x_backward_i_forward_j);
                x_backward_i_forward_j(i) += h_i;
                x_backward_i_forward_j(j) -= h_j;

                x_forward_i_backward_j(i) += h_i;
                x_forward_i_backward_j(j) -= h_j;
                y_forward_i_backward_j = (t.*f)(dummy, x_forward_i_backward_j);
                x_forward_i_backward_j(i) -= h_i;
                x_forward_i_backward_j(j) += h_j;

                H(i,j) = (y_forward_ij - y_forward_i_backward_j - y_backward_i_forward_j + y_backward_ij)/(type(4.0)*h_i*h_j);
            }
        }

        for(Index i = 0; i < n; i++)
        {
            for(Index j = 0; j < i; j++)
            {
                H(i,j) = H(j,i);
            }
        }

        return H;
    }

    /// Returns the hessian matrix of a function of several dimensions using the central differences method.
    /// The function to be differentiated is of the following form: type f(const Index&, const Tensor<type, 1>&) const.
    /// The first integer argument is dummy, differentiation is performed with respect to the second vector argument.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param dummy: Dummy integer for the method prototype.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 2> calculate_hessian(const T& t, type(T::*f)(const Index&, const Tensor<type, 1>&) const, const Index& dummy, const Tensor<type, 1>& x) const
    {
        const Index n = x.size();

        type y = (t.*f)(dummy, x);

        Tensor<type, 2> H(n, n);

        type h_i;
        type h_j;

        Tensor<type, 1> x_backward_2i(x);
        Tensor<type, 1> x_backward_i(x);

        Tensor<type, 1> x_forward_i(x);
        Tensor<type, 1> x_forward_2i(x);

        Tensor<type, 1> x_backward_ij(x);
        Tensor<type, 1> x_forward_ij(x);

        Tensor<type, 1> x_backward_i_forward_j(x);
        Tensor<type, 1> x_forward_i_backward_j(x);

        type y_backward_2i;
        type y_backward_i;

        type y_forward_i;
        type y_forward_2i;

        type y_backward_ij;
        type y_forward_ij;

        type y_backward_i_forward_j;
        type y_forward_i_backward_j;

        for(Index i = 0; i < n; i++)
        {
            h_i = calculate_h(x(i));

            x_backward_2i(i) -= static_cast<type>(2.0)*h_i;
            y_backward_2i = (t.*f)(dummy, x_backward_2i);
            x_backward_2i(i) += static_cast<type>(2.0)*h_i;

            x_backward_i(i) -= h_i;
            y_backward_i = (t.*f)(dummy, x_backward_i);
            x_backward_i(i) += h_i;

            x_forward_i(i) += h_i;
            y_forward_i = (t.*f)(dummy, x_forward_i);
            x_forward_i(i) -= h_i;

            x_forward_2i(i) += static_cast<type>(2.0)*h_i;
            y_forward_2i = (t.*f)(dummy, x_forward_2i);
            x_forward_2i(i) -= static_cast<type>(2.0)*h_i;

            H(i,i) = (-y_forward_2i + type(16.0)*y_forward_i - type(30.0)*y + type(16.0)*y_backward_i - y_backward_2i)/(type(12.0)*pow(h_i, type(2)));

            for(Index j = i; j < n; j++)
            {
                h_j = calculate_h(x(j));

                x_backward_ij(i) -= h_i;
                x_backward_ij(j) -= h_j;
                y_backward_ij = (t.*f)(dummy, x_backward_ij);
                x_backward_ij(i) += h_i;
                x_backward_ij(j) += h_j;

                x_forward_ij(i) += h_i;
                x_forward_ij(j) += h_j;
                y_forward_ij = (t.*f)(dummy, x_forward_ij);
                x_forward_ij(i) -= h_i;
                x_forward_ij(j) -= h_j;

                x_backward_i_forward_j(i) -= h_i;
                x_backward_i_forward_j(j) += h_j;
                y_backward_i_forward_j = (t.*f)(dummy, x_backward_i_forward_j);
                x_backward_i_forward_j(i) += h_i;
                x_backward_i_forward_j(j) -= h_j;

                x_forward_i_backward_j(i) += h_i;
                x_forward_i_backward_j(j) -= h_j;
                y_forward_i_backward_j = (t.*f)(dummy, x_forward_i_backward_j);
                x_forward_i_backward_j(i) -= h_i;
                x_forward_i_backward_j(j) += h_j;

                H(i,j) = (y_forward_ij - y_forward_i_backward_j - y_backward_i_forward_j + y_backward_ij)/(type(4.0)*h_i*h_j);
            }
        }

        for(Index i = 0; i < n; i++)
        {
            for(Index j = 0; j < i; j++)
            {
                H(i,j) = H(j,i);
            }
        }

        return H;
    }

    /// Returns the Jacobian matrix of a function of many inputs and many outputs using the central differences method.
    /// The function to be differentiated is of the following form: Tensor<type, 1> f(const Tensor<type, 1>&, const Tensor<type, 1>&) const.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 2> calculate_Jacobian(const T& t, Tensor<type, 1>(T::*f)(const Tensor<type, 1>&) const, const Tensor<type, 1>& x) const
    {
        Tensor<type, 1> y = (t.*f)(x);

        type h;

        const Index n = x.size();
        Index m = y.size();

        Tensor<type, 1> x_forward(x);
        Tensor<type, 1> x_backward(x);

        Tensor<type, 1> y_forward(n);
        Tensor<type, 1> y_backward(n);

        Tensor<type, 2> J(m,n);

        for(Index j = 0; j < n; j++)
        {
            h = calculate_h(x(j));

            x_backward(j) -= h;
            y_backward = (t.*f)(x_backward);
            x_backward(j) += h;

            x_forward(j) += h;
            y_forward = (t.*f)(x_forward);
            x_forward(j) -= h;

            for(Index i = 0; i < m; i++)
            {
                J(i,j) = (y_forward(i) - y_backward(i))/(static_cast<type>(2.0)*h);
            }
        }

        return J;
    }


    /// Returns the Jacobian matrix of a function of many inputs and many outputs using the central differences method.
    /// The function to be differentiated is of the following form: Tensor<type, 1> f(const Tensor<type, 1>&, const Tensor<type, 1>&) const.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param dummy: Dummy vector for the method prototype.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 2> calculate_Jacobian(const T& t, Tensor<type, 1>(T::*f)(const Tensor<type, 1>&, const Tensor<type, 1>&) const, const Tensor<type, 1>& dummy, const Tensor<type, 1>& x) const
    {
        Tensor<type, 1> y = (t.*f)(dummy, x);

        type h;

        const Index n = x.size();
        Index m = y.size();

        Tensor<type, 1> x_forward(x);
        Tensor<type, 1> x_backward(x);

        Tensor<type, 1> y_forward(n);
        Tensor<type, 1> y_backward(n);

        Tensor<type, 2> J(m,n);

        for(Index j = 0; j < n; j++)
        {
            h = calculate_h(x(j));

            x_backward(j) -= h;
            y_backward = (t.*f)(dummy, x_backward);
            x_backward(j) += h;

            x_forward(j) += h;
            y_forward = (t.*f)(dummy, x_forward);
            x_forward(j) -= h;

            for(Index i = 0; i < m; i++)
            {
                J(i,j) = (y_forward(i) - y_backward(i))/(static_cast<type>(2.0)*h);
            }
        }

        return J;
    }

    /// Returns the Jacobian matrix of a function of many inputs and many outputs using the central differences method.
    /// The function to be differentiated is of the following form: type f(const Index&, const Tensor<type, 1>&) const.
    /// The first integer argument is dummy, differentiation is performed with respect to the second vector argument.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param dummy: Dummy integer for the method prototype.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 2> calculate_Jacobian(const T& t, Tensor<type, 1>(T::*f)(const Index&, const Tensor<type, 1>&) const, const Index& dummy, const Tensor<type, 1>& x) const
    {
        Tensor<type, 1> y = (t.*f)(dummy, x);

        type h;

        const Index n = x.size();
        Index m = y.size();

        Tensor<type, 1> x_forward(x);
        Tensor<type, 1> x_backward(x);

        Tensor<type, 1> y_forward(n);
        Tensor<type, 1> y_backward(n);

        Tensor<type, 2> J(m,n);

        for(Index j = 0; j < n; j++)
        {
            h = calculate_h(x(j));

            x_backward(j) -= h;
            y_backward = (t.*f)(dummy, x_backward);
            x_backward(j) += h;

            x_forward(j) += h;
            y_forward = (t.*f)(dummy, x_forward);
            x_forward(j) -= h;

            for(Index i = 0; i < m; i++)
            {
                J(i,j) = (y_forward(i) - y_backward(i))/(static_cast<type>(2.0)*h);
            }
        }

        return J;
    }

    /// Returns the Jacobian matrix of a function of many inputs and many outputs using the central differences method.
    /// The function to be differentiated is of the following form: type f(const Index&, const Tensor<type, 1>&, const Tensor<type, 1>&) const.
    /// The first and second arguments are dummy, differentiation is performed with respect to the third argument.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param dummy_int: Dummy integer for the method prototype.
    /// @param dummy_vector: Dummy vector for the method prototype.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 2> calculate_Jacobian
    (const T& t, Tensor<type, 1>(T::*f)(const Index&, const Tensor<type, 1>&, const Tensor<type, 1>&) const, const Index& dummy_int, const Tensor<type, 1>& dummy_vector, const Tensor<type, 1>& x) const
    {
        const Index n = x.size();

        const Tensor<type, 1> y = (t.*f)(dummy_int, dummy_vector, x);
        const Index m = y.size();

        type h;

        Tensor<type, 1> x_forward(x);
        Tensor<type, 1> x_backward(x);

        Tensor<type, 1> y_forward(n);
        Tensor<type, 1> y_backward(n);

        Tensor<type, 2> J(m,n);

        for(Index j = 0; j < n; j++)
        {
            h = calculate_h(x(j));

            x_backward(j) -= h;
            y_backward = (t.*f)(dummy_int, dummy_vector, x_backward);
            x_backward(j) += h;

            x_forward(j) += h;
            y_forward = (t.*f)(dummy_int, dummy_vector, x_forward);
            x_forward(j) -= h;

            for(Index i = 0; i < m; i++)
            {
                J(i,j) = (y_forward(i) - y_backward(i))/(static_cast<type>(2.0)*h);
            }
        }

        return J;
    }

    /// Returns the Jacobian matrix of a function of many inputs and many outputs using the central differences method.
    /// The function to be differentiated is of the following form: type f(const Index&, const Tensor<type, 1>&, const Tensor<type, 1>&) const.
    /// The first and second arguments are dummy, differentiation is performed with respect to the third argument.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param dummy_int_1: Dummy integer for the method prototype.
    /// @param dummy_int_2: Dummy integer for the method prototype.
    /// @param x: Input vector.

    template<class T>
    Tensor<type, 2> calculate_Jacobian
    (const T& t, Tensor<type, 1>(T::*f)(const Index&, const Index&, const Tensor<type, 1>&) const, const Index& dummy_int_1, const Index& dummy_int_2, const Tensor<type, 1>& x) const
    {
        const Tensor<type, 1> y = (t.*f)(dummy_int_1, dummy_int_2, x);

        const Index n = x.size();
        const Index m = y.size();

        type h;

        Tensor<type, 1> x_forward(x);
        Tensor<type, 1> x_backward(x);

        Tensor<type, 1> y_forward(n);
        Tensor<type, 1> y_backward(n);

        Tensor<type, 2> J(m,n);

        for(Index j = 0; j < n; j++)
        {
            h = calculate_h(x(j));

            x_backward(j) -= h;
            y_backward = (t.*f)(dummy_int_1, dummy_int_2, x_backward);
            x_backward(j) += h;

            x_forward(j) += h;
            y_forward = (t.*f)(dummy_int_1, dummy_int_2, x_forward);
            x_forward(j) -= h;

            for(Index i = 0; i < m; i++)
            {
                J(i,j) = (y_forward(i) - y_backward(i))/(static_cast<type>(2.0)*h);
            }
        }

        return J;
    }

    /// Returns the hessian form, as a vector of matrices, of a function of many inputs and many outputs using the central differences method.
    /// The function to be differentiated is of the following form: Tensor<type, 1> f(const Tensor<type, 1>&) const.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param x: Input vector.

    template<class T>
    Tensor<Tensor<type, 2>, 1> calculate_hessian(const T& t, Tensor<type, 1>(T::*f)(const Tensor<type, 1>&) const, const Tensor<type, 1>& x) const
    {
        Tensor<type, 1> y = (t.*f)(x);

        Index s = y.size();
        const Index n = x.size();

        type h_j;
        type h_k;

        Tensor<type, 1> x_backward_2j(x);
        Tensor<type, 1> x_backward_j(x);

        Tensor<type, 1> x_forward_j(x);
        Tensor<type, 1> x_forward_2j(x);

        Tensor<type, 1> x_backward_jk(x);
        Tensor<type, 1> x_forward_jk(x);

        Tensor<type, 1> x_backward_j_forward_k(x);
        Tensor<type, 1> x_forward_j_backward_k(x);

        Tensor<type, 1> y_backward_2j;
        Tensor<type, 1> y_backward_j;

        Tensor<type, 1> y_forward_j;
        Tensor<type, 1> y_forward_2j;

        Tensor<type, 1> y_backward_jk;
        Tensor<type, 1> y_forward_jk;

        Tensor<type, 1> y_backward_j_forward_k;
        Tensor<type, 1> y_forward_j_backward_k;

        Tensor<Tensor<type, 2>, 1> H(s);

        for(Index i = 0; i < s; i++)
        {
            H(i).resize(n,n);

            for(Index j = 0; j < n; j++)
            {
                h_j = calculate_h(x(j));

                x_backward_2j(j) -= static_cast<type>(2.0)*h_j;
                y_backward_2j = (t.*f)(x_backward_2j);
                x_backward_2j(j) += static_cast<type>(2.0)*h_j;

                x_backward_j(j) -= h_j;
                y_backward_j = (t.*f)(x_backward_j);
                x_backward_j(j) += h_j;

                x_forward_j(j) += h_j;
                y_forward_j = (t.*f)(x_forward_j);
                x_forward_j(j) -= h_j;

                x_forward_2j(j) += static_cast<type>(2.0)*h_j;
                y_forward_2j = (t.*f)(x_forward_2j);
                x_forward_2j(j) -= static_cast<type>(2.0)*h_j;

                H(i)(j,j) = (-y_forward_2j(i) + type(16.0)*y_forward_j(i) - type(30.0)*y(i) + type(16.0)*y_backward_j(i) - y_backward_2j(i))/(type(12.0)*pow(h_j, type(2)));

                for(Index k = j; k < n; k++)
                {
                    h_k = calculate_h(x[k]);

                    x_backward_jk(j) -= h_j;
                    x_backward_jk[k] -= h_k;
                    y_backward_jk = (t.*f)(x_backward_jk);
                    x_backward_jk(j) += h_j;
                    x_backward_jk[k] += h_k;

                    x_forward_jk(j) += h_j;
                    x_forward_jk[k] += h_k;
                    y_forward_jk = (t.*f)(x_forward_jk);
                    x_forward_jk(j) -= h_j;
                    x_forward_jk[k] -= h_k;

                    x_backward_j_forward_k(j) -= h_j;
                    x_backward_j_forward_k[k] += h_k;
                    y_backward_j_forward_k = (t.*f)(x_backward_j_forward_k);
                    x_backward_j_forward_k(j) += h_j;
                    x_backward_j_forward_k[k] -= h_k;

                    x_forward_j_backward_k(j) += h_j;
                    x_forward_j_backward_k[k] -= h_k;
                    y_forward_j_backward_k = (t.*f)(x_forward_j_backward_k);
                    x_forward_j_backward_k(j) -= h_j;
                    x_forward_j_backward_k[k] += h_k;

                    H(i)(j,k) = (y_forward_jk(i) - y_forward_j_backward_k(i) - y_backward_j_forward_k(i) + y_backward_jk(i))/(type(4.0)*h_j*h_k);
                }
            }

            for(Index j = 0; j < n; j++)
            {
                for(Index k = 0; k < j; k++)
                {
                    H(i)(j,k) = H(i)(k,j);
                }
            }
        }

        return H;
    }


    /// Returns the hessian form, as a vector of matrices, of a function of many inputs and many outputs using the central differences method.
    /// The function to be differentiated is of the following form: Tensor<type, 1> f(const Index&, const Tensor<type, 1>&, const Tensor<type, 1>&) const.
    /// The first and second arguments are dummy, differentiation is performed with respect to the third argument.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param dummy_vector: Dummy vector for the method prototype.
    /// @param x: Input vector.

    template<class T>
    Tensor<Tensor<type, 2>, 1> calculate_hessian
    (const T& t, Tensor<type, 1>(T::*f)(const Tensor<type, 1>&, const Tensor<type, 1>&) const, const Tensor<type, 1>& dummy_vector, const Tensor<type, 1>& x) const
    {
        Tensor<type, 1> y = (t.*f)(dummy_vector, x);

        Index s = y.size();
        const Index n = x.size();

        type h_j;
        type h_k;

        Tensor<type, 1> x_backward_2j(x);
        Tensor<type, 1> x_backward_j(x);

        Tensor<type, 1> x_forward_j(x);
        Tensor<type, 1> x_forward_2j(x);

        Tensor<type, 1> x_backward_jk(x);
        Tensor<type, 1> x_forward_jk(x);

        Tensor<type, 1> x_backward_j_forward_k(x);
        Tensor<type, 1> x_forward_j_backward_k(x);

        Tensor<type, 1> y_backward_2j;
        Tensor<type, 1> y_backward_j;

        Tensor<type, 1> y_forward_j;
        Tensor<type, 1> y_forward_2j;

        Tensor<type, 1> y_backward_jk;
        Tensor<type, 1> y_forward_jk;

        Tensor<type, 1> y_backward_j_forward_k;
        Tensor<type, 1> y_forward_j_backward_k;

        Tensor<Tensor<type, 2>, 1> H(s);

        for(Index i = 0; i < s; i++)
        {
            //         H(i).set(n,n);
            H(i).resize(n,n);

            for(Index j = 0; j < n; j++)
            {
                h_j = calculate_h(x(j));

                x_backward_2j(j) -= static_cast<type>(2.0)*h_j;
                y_backward_2j = (t.*f)(dummy_vector, x_backward_2j);
                x_backward_2j(j) += static_cast<type>(2.0)*h_j;

                x_backward_j(j) -= h_j;
                y_backward_j = (t.*f)(dummy_vector, x_backward_j);
                x_backward_j(j) += h_j;

                x_forward_j(j) += h_j;
                y_forward_j = (t.*f)(dummy_vector, x_forward_j);
                x_forward_j(j) -= h_j;

                x_forward_2j(j) += static_cast<type>(2.0)*h_j;
                y_forward_2j = (t.*f)(dummy_vector, x_forward_2j);
                x_forward_2j(j) -= static_cast<type>(2.0)*h_j;

                H(i)(j,j) = (-y_forward_2j(i) + type(16.0)*y_forward_j(i) - type(30.0)*y(i) + type(16.0)*y_backward_j(i) - y_backward_2j(i))/(type(12.0)*pow(h_j, type(2)));

                for(Index k = j; k < n; k++)
                {
                    h_k = calculate_h(x[k]);

                    x_backward_jk(j) -= h_j;
                    x_backward_jk[k] -= h_k;
                    y_backward_jk = (t.*f)(dummy_vector, x_backward_jk);
                    x_backward_jk(j) += h_j;
                    x_backward_jk[k] += h_k;

                    x_forward_jk(j) += h_j;
                    x_forward_jk[k] += h_k;
                    y_forward_jk = (t.*f)(dummy_vector, x_forward_jk);
                    x_forward_jk(j) -= h_j;
                    x_forward_jk[k] -= h_k;

                    x_backward_j_forward_k(j) -= h_j;
                    x_backward_j_forward_k[k] += h_k;
                    y_backward_j_forward_k = (t.*f)(dummy_vector, x_backward_j_forward_k);
                    x_backward_j_forward_k(j) += h_j;
                    x_backward_j_forward_k[k] -= h_k;

                    x_forward_j_backward_k(j) += h_j;
                    x_forward_j_backward_k[k] -= h_k;
                    y_forward_j_backward_k = (t.*f)(dummy_vector, x_forward_j_backward_k);
                    x_forward_j_backward_k(j) -= h_j;
                    x_forward_j_backward_k[k] += h_k;

                    H(i)(j,k) = (y_forward_jk(i) - y_forward_j_backward_k(i) - y_backward_j_forward_k(i) + y_backward_jk(i))/(type(4.0)*h_j*h_k);
                }
            }

            for(Index j = 0; j < n; j++)
            {
                for(Index k = 0; k < j; k++)
                {
                    H(i)(j,k) = H(i)(k,j);
                }
            }
        }

        return H;
    }

    /// Returns the hessian form, as a vector of matrices, of a function of many inputs and many outputs using the central differences method.
    /// The function to be differentiated is of the following form: Tensor<type, 1> f(const Index&, const Tensor<type, 1>&) const.
    /// The first argument is dummy, differentiation is performed with respect to the second argument.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param dummy: Dummy integer for the method prototype.
    /// @param x: Input vector.

    template<class T>
    Tensor<Tensor<type, 2>, 1> calculate_hessian(const T& t, Tensor<type, 1>(T::*f)(const Index&, const Tensor<type, 1>&) const, const Index& dummy, const Tensor<type, 1>& x) const
    {
        Tensor<type, 1> y = (t.*f)(dummy, x);

        Index s = y.size();
        const Index n = x.size();

        type h_j;
        type h_k;

        Tensor<type, 1> x_backward_2j(x);
        Tensor<type, 1> x_backward_j(x);

        Tensor<type, 1> x_forward_j(x);
        Tensor<type, 1> x_forward_2j(x);

        Tensor<type, 1> x_backward_jk(x);
        Tensor<type, 1> x_forward_jk(x);

        Tensor<type, 1> x_backward_j_forward_k(x);
        Tensor<type, 1> x_forward_j_backward_k(x);

        Tensor<type, 1> y_backward_2j;
        Tensor<type, 1> y_backward_j;

        Tensor<type, 1> y_forward_j;
        Tensor<type, 1> y_forward_2j;

        Tensor<type, 1> y_backward_jk;
        Tensor<type, 1> y_forward_jk;

        Tensor<type, 1> y_backward_j_forward_k;
        Tensor<type, 1> y_forward_j_backward_k;

        Tensor<Tensor<type, 2>, 1> H(s);

        for(Index i = 0; i < s; i++)
        {
            H(i).resize(n,n);

            for(Index j = 0; j < n; j++)
            {
                h_j = calculate_h(x(j));

                x_backward_2j(j) -= static_cast<type>(2.0)*h_j;
                y_backward_2j = (t.*f)(dummy, x_backward_2j);
                x_backward_2j(j) += static_cast<type>(2.0)*h_j;

                x_backward_j(j) -= h_j;
                y_backward_j = (t.*f)(dummy, x_backward_j);
                x_backward_j(j) += h_j;

                x_forward_j(j) += h_j;
                y_forward_j = (t.*f)(dummy, x_forward_j);
                x_forward_j(j) -= h_j;

                x_forward_2j(j) += static_cast<type>(2.0)*h_j;
                y_forward_2j = (t.*f)(dummy, x_forward_2j);
                x_forward_2j(j) -= static_cast<type>(2.0)*h_j;

                H(i)(j,j) = (-y_forward_2j(i) + type(16.0)*y_forward_j(i) - type(30.0)*y(i) + type(16.0)*y_backward_j(i) - y_backward_2j(i))/(type(12.0)*pow(h_j, type(2)));

                for(Index k = j; k < n; k++)
                {
                    h_k = calculate_h(x[k]);

                    x_backward_jk(j) -= h_j;
                    x_backward_jk[k] -= h_k;
                    y_backward_jk = (t.*f)(dummy, x_backward_jk);
                    x_backward_jk(j) += h_j;
                    x_backward_jk[k] += h_k;

                    x_forward_jk(j) += h_j;
                    x_forward_jk[k] += h_k;
                    y_forward_jk = (t.*f)(dummy, x_forward_jk);
                    x_forward_jk(j) -= h_j;
                    x_forward_jk[k] -= h_k;

                    x_backward_j_forward_k(j) -= h_j;
                    x_backward_j_forward_k[k] += h_k;
                    y_backward_j_forward_k = (t.*f)(dummy, x_backward_j_forward_k);
                    x_backward_j_forward_k(j) += h_j;
                    x_backward_j_forward_k[k] -= h_k;

                    x_forward_j_backward_k(j) += h_j;
                    x_forward_j_backward_k[k] -= h_k;
                    y_forward_j_backward_k = (t.*f)(dummy, x_forward_j_backward_k);
                    x_forward_j_backward_k(j) -= h_j;
                    x_forward_j_backward_k[k] += h_k;

                    H(i)(j,k) = (y_forward_jk(i) - y_forward_j_backward_k(i) - y_backward_j_forward_k(i) + y_backward_jk(i))/(type(4.0)*h_j*h_k);
                }
            }

            for(Index j = 0; j < n; j++)
            {
                for(Index k = 0; k < j; k++)
                {
                    H(i)(j,k) = H(i)(k,j);
                }
            }
        }

        return H;
    }

    /// Returns the hessian form, as a vector of matrices, of a function of many inputs and many outputs using the central differences method.
    /// The function to be differentiated is of the following form: Tensor<type, 1> f(const Index&, const Tensor<type, 1>&, const Tensor<type, 1>&) const.
    /// The first and second arguments are dummy, differentiation is performed with respect to the third argument.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param dummy_int: Dummy integer for the method prototype.
    /// @param dummy_vector: Dummy vector for the method prototype.
    /// @param x: Input vector.

    template<class T>
    Tensor<Tensor<type, 2>, 1> calculate_hessian
    (const T& t, Tensor<type, 1>(T::*f)(const Index&, const Tensor<type, 1>&, const Tensor<type, 1>&) const, const Index& dummy_int, const Tensor<type, 1>& dummy_vector, const Tensor<type, 1>& x) const
    {
        const Tensor<type, 1> y = (t.*f)(dummy_int, dummy_vector, x);

        Index s = y.size();
        const Index n = x.size();

        type h_j;
        type h_k;

        Tensor<type, 1> x_backward_2j(x);
        Tensor<type, 1> x_backward_j(x);

        Tensor<type, 1> x_forward_j(x);
        Tensor<type, 1> x_forward_2j(x);

        Tensor<type, 1> x_backward_jk(x);
        Tensor<type, 1> x_forward_jk(x);

        Tensor<type, 1> x_backward_j_forward_k(x);
        Tensor<type, 1> x_forward_j_backward_k(x);

        Tensor<type, 1> y_backward_2j;
        Tensor<type, 1> y_backward_j;

        Tensor<type, 1> y_forward_j;
        Tensor<type, 1> y_forward_2j;

        Tensor<type, 1> y_backward_jk;
        Tensor<type, 1> y_forward_jk;

        Tensor<type, 1> y_backward_j_forward_k;
        Tensor<type, 1> y_forward_j_backward_k;

        Tensor<Tensor<type, 2>, 1> H(s);

        for(Index i = 0; i < s; i++)
        {
            H(i).resize(n,n);

            for(Index j = 0; j < n; j++)
            {
                h_j = calculate_h(x(j));

                x_backward_2j(j) -= static_cast<type>(2.0)*h_j;
                y_backward_2j = (t.*f)(dummy_int, dummy_vector, x_backward_2j);
                x_backward_2j(j) += static_cast<type>(2.0)*h_j;

                x_backward_j(j) -= h_j;
                y_backward_j = (t.*f)(dummy_int, dummy_vector, x_backward_j);
                x_backward_j(j) += h_j;

                x_forward_j(j) += h_j;
                y_forward_j = (t.*f)(dummy_int, dummy_vector, x_forward_j);
                x_forward_j(j) -= h_j;

                x_forward_2j(j) += static_cast<type>(2.0)*h_j;
                y_forward_2j = (t.*f)(dummy_int, dummy_vector, x_forward_2j);
                x_forward_2j(j) -= static_cast<type>(2.0)*h_j;

                H(i)(j,j) = (-y_forward_2j(i) + type(16.0)*y_forward_j(i) - type(30.0)*y(i) + type(16.0)*y_backward_j(i) - y_backward_2j(i))/(type(12.0)*pow(h_j, type(2)));

                for(Index k = j; k < n; k++)
                {
                    h_k = calculate_h(x[k]);

                    x_backward_jk(j) -= h_j;
                    x_backward_jk[k] -= h_k;
                    y_backward_jk = (t.*f)(dummy_int, dummy_vector, x_backward_jk);
                    x_backward_jk(j) += h_j;
                    x_backward_jk[k] += h_k;

                    x_forward_jk(j) += h_j;
                    x_forward_jk[k] += h_k;
                    y_forward_jk = (t.*f)(dummy_int, dummy_vector, x_forward_jk);
                    x_forward_jk(j) -= h_j;
                    x_forward_jk[k] -= h_k;

                    x_backward_j_forward_k(j) -= h_j;
                    x_backward_j_forward_k[k] += h_k;
                    y_backward_j_forward_k = (t.*f)(dummy_int, dummy_vector, x_backward_j_forward_k);
                    x_backward_j_forward_k(j) += h_j;
                    x_backward_j_forward_k[k] -= h_k;

                    x_forward_j_backward_k(j) += h_j;
                    x_forward_j_backward_k[k] -= h_k;
                    y_forward_j_backward_k = (t.*f)(dummy_int, dummy_vector, x_forward_j_backward_k);
                    x_forward_j_backward_k(j) -= h_j;
                    x_forward_j_backward_k[k] += h_k;

                    H(i)(j,k) = (y_forward_jk(i) - y_forward_j_backward_k(i) - y_backward_j_forward_k(i) + y_backward_jk(i))/(type(4.0)*h_j*h_k);
                }
            }

            for(Index j = 0; j < n; j++)
            {
                for(Index k = 0; k < j; k++)
                {
                    H(i)(j,k) = H(i)(k,j);
                }
            }
        }

        return H;
    }

    /// Returns the hessian matrices, as a matrix of matrices, of a function of many inputs and many outputs using the central differences method.
    /// The function to be differentiated is of the following form: Tensor<type, 1> f(const Index&, const Tensor<type, 1>&, const Tensor<type, 1>&) const.
    /// The first and second arguments are dummy, differentiation is performed with respect to the third argument.
    /// @param t : Object constructor containing the member method to differentiate.
    /// @param f: Pointer to the member method.
    /// @param dummy_int: Dummy integer for the method prototype.
    /// @param dummy_vector: Dummy vector for the method prototype.
    /// @param x: Input vector.

    template<class T>
    Tensor< Tensor<type, 2>, 2> calculate_hessian_matrices
    (const T& t, Tensor<type, 1>(T::*f)(const Index&, const Tensor<type, 1>&, const Tensor<type, 1>&) const, const Index& dummy_int, const Tensor<type, 1>& dummy_vector, const Tensor<type, 1>& x) const
    {
        const Tensor<type, 1> y = (t.*f)(dummy_int, dummy_vector, x);

        Index s = y.size();
        const Index n = x.size();

        type h_j;
        type h_k;

        Tensor<type, 1> x_backward_2j(x);
        Tensor<type, 1> x_backward_j(x);

        Tensor<type, 1> x_forward_j(x);
        Tensor<type, 1> x_forward_2j(x);

        Tensor<type, 1> x_backward_jk(x);
        Tensor<type, 1> x_forward_jk(x);

        Tensor<type, 1> x_backward_j_forward_k(x);
        Tensor<type, 1> x_forward_j_backward_k(x);

        Tensor<type, 1> y_backward_2j;
        Tensor<type, 1> y_backward_j;

        Tensor<type, 1> y_forward_j;
        Tensor<type, 1> y_forward_2j;

        Tensor<type, 1> y_backward_jk;
        Tensor<type, 1> y_forward_jk;

        Tensor<type, 1> y_backward_j_forward_k;
        Tensor<type, 1> y_forward_j_backward_k;

        Tensor< Tensor<type, 2>, 2> H;

        H.resize(n,s);

        for(Index i = 0; i < s; i++)
        {
            for(Index t = 0; t < n; t++)
            {
                H(i,t).resize(n,s);

                for(Index j = 0; j < n; j++)
                {
                    h_j = calculate_h(x(j));

                    x_backward_2j(j) -= static_cast<type>(2.0)*h_j;
                    y_backward_2j = (t.*f)(dummy_int, dummy_vector, x_backward_2j);
                    x_backward_2j(j) += static_cast<type>(2.0)*h_j;

                    x_backward_j(j) -= h_j;
                    y_backward_j = (t.*f)(dummy_int, dummy_vector, x_backward_j);
                    x_backward_j(j) += h_j;

                    x_forward_j(j) += h_j;
                    y_forward_j = (t.*f)(dummy_int, dummy_vector, x_forward_j);
                    x_forward_j(j) -= h_j;

                    x_forward_2j(j) += static_cast<type>(2.0)*h_j;
                    y_forward_2j = (t.*f)(dummy_int, dummy_vector, x_forward_2j);
                    x_forward_2j(j) -= static_cast<type>(2.0)*h_j;

                    H(i)(j,j) = (-y_forward_2j(i) + type(16.0)*y_forward_j(i) - type(30.0)*y(i) + type(16.0)*y_backward_j(i) - y_backward_2j(i))/(type(12.0)*pow(h_j, type(2)));

                    for(Index k = j; k < s; k++)
                    {
                        h_k = calculate_h(x[k]);

                        x_backward_jk(j) -= h_j;
                        x_backward_jk[k] -= h_k;
                        y_backward_jk = (t.*f)(dummy_int, dummy_vector, x_backward_jk);
                        x_backward_jk(j) += h_j;
                        x_backward_jk[k] += h_k;

                        x_forward_jk(j) += h_j;
                        x_forward_jk[k] += h_k;
                        y_forward_jk = (t.*f)(dummy_int, dummy_vector, x_forward_jk);
                        x_forward_jk(j) -= h_j;
                        x_forward_jk[k] -= h_k;

                        x_backward_j_forward_k(j) -= h_j;
                        x_backward_j_forward_k[k] += h_k;
                        y_backward_j_forward_k = (t.*f)(dummy_int, dummy_vector, x_backward_j_forward_k);
                        x_backward_j_forward_k(j) += h_j;
                        x_backward_j_forward_k[k] -= h_k;

                        x_forward_j_backward_k(j) += h_j;
                        x_forward_j_backward_k[k] -= h_k;
                        y_forward_j_backward_k = (t.*f)(dummy_int, dummy_vector, x_forward_j_backward_k);
                        x_forward_j_backward_k(j) -= h_j;
                        x_forward_j_backward_k[k] += h_k;

                        H(i,t)(j,k) = (y_forward_jk(i) - y_forward_j_backward_k(i) - y_backward_j_forward_k(i) + y_backward_jk(i))/(type(4.0)*h_j*h_k);
                    }
                }

                for(Index j = 0; j < n; j++)
                {
                    for(Index k = 0; k < j; k++)
                    {
                        H(i,t)(j,k) = H(i)(k,j);
                    }
                }
            }
        }

        return H;
    }

private:

    /// Number of precision digits.

    Index precision_digits;

    /// Flag for displaying warning messages from this class.

    bool display = true;

};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2022 Artificial Intelligence Techniques, SL.
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
