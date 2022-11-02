//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

// OpenNN includes

#include "../opennn/opennn.h"
using namespace opennn;

// Program to find correlation
// coefficient
#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

// Utility Function to print
// a Vector
void printVector(const Tensor<type,1> &X)
{
    cout << X << endl;

    /*
    for (auto i: X)
        cout << i << " ";

    cout << endl;
    */
}

// Function returns the rank vector
// of the set of observations
Tensor<type,1> rankify(Tensor<type,1> & X) {

    int N = X.size();

    // Rank Vector
    Tensor<type,1> Rank_X(N);

    for(int i = 0; i < N; i++)
    {
        int r = 1, s = 1;

        // Count no of smaller elements
        // in 0 to i-1
        for(int j = 0; j < i; j++) {
            if (X[j] < X[i] ) r++;
            if (X[j] == X[i] ) s++;
        }

        // Count no of smaller elements
        // in i+1 to N-1
        for (int j = i+1; j < N; j++) {
            if (X[j] < X[i] ) r++;
            if (X[j] == X[i] ) s++;
        }

        // Use Fractional Rank formula
        // fractional_rank = r + (n-1)/2
        Rank_X[i] = r + (s-1) * 0.5;
    }

    // Return Rank Vector
    return Rank_X;
}

// function that returns
// Pearson correlation coefficient.
float correlationCoefficient
        (Tensor<type,1> &X, Tensor<type,1> &Y)
{
    int n = X.size();
    float sum_X = 0, sum_Y = 0,
                    sum_XY = 0;
    float squareSum_X = 0,
        squareSum_Y = 0;

    for (int i = 0; i < n; i++)
    {
        // sum of elements of array X.
        sum_X = sum_X + X[i];

        // sum of elements of array Y.
        sum_Y = sum_Y + Y[i];

        // sum of X[i] * Y[i].
        sum_XY = sum_XY + X[i] * Y[i];

        // sum of square of array elements.
        squareSum_X = squareSum_X +
                    X[i] * X[i];
        squareSum_Y = squareSum_Y +
                    Y[i] * Y[i];
    }

    // use formula for calculating
    // correlation coefficient.
    float corr = (float)(n * sum_XY -
                sum_X * sum_Y) /
                sqrt((n * squareSum_X -
                    sum_X * sum_X) *
                    (n * squareSum_Y -
                    sum_Y * sum_Y));

    return corr;
}

// Driver function
int main()
{
    Tensor<type,1> X(5);// = {15,18,21, 15, 21};
    X(0) = 15;
    X(1) = 18;
    X(2) = 21;
    X(3) = 15;
    X(4) = 21;

    Tensor<type,1> Y(5); //= {25,25,27,27,27};
    Y(0) = 25;
    Y(1) = 25;
    Y(2) = 27;
    Y(3) = 27;
    Y(4) = 27;

    // Get ranks of vector X
    const Tensor<type,1> rank_x = rankify(X);

    // Get ranks of vector y
    const Tensor<type,1> rank_y = rankify(Y);

    cout << "Vector X" << endl;
    printVector(X);

    // Print rank vector of X
    cout << "Rankings of X" << endl;
    printVector(rank_x);

    // Print Vector Y
    cout << "Vector Y" << endl;
    printVector(Y);

    // Print rank vector of Y
    cout << "Rankings of Y" << endl;
    printVector(rank_y);

    // Print Spearmans coefficient
    cout << "Spearman's Rank correlation: " << endl;

    const ThreadPoolDevice* thread_pool_device;
    cout << linear_correlation_spearman(thread_pool_device, rank_x, rank_y).r << endl;

    return 0;
}
// problema derivado de const delante de las definiciones, no vamos a definir Tensor< a secas nunca.

/*
Tensor<type, 1> rankify(Tensor<type, 1> & x) {

    int n = x.size();

    Tensor<type, 1> rank_x(n);

    for(int i = 0; i < n; i++)
    {
        int r = 1, s = 1;

        // Count no of smaller elements in 0 to i-1
        for(int j = 0; j < i; j++) {
            if (x[j] < x[i] ) r++;
            if (x[j] == x[i] ) s++;
        }

        // Count no of smaller elements in i+1 to N-1
        for (int j = i+1; j < n; j++) {
            if (x[j] < x[i] ) r++;
            if (x[j] == x[i] ) s++;
        }

        // Use Fractional Rank formula fractional_rank = r + (n-1)/2
        rank_x[i] = r + (s-1) * 0.5;
    }

    return rank_x;
}


// function that returns
// Pearson correlation coefficient.
float correlationCoefficient(vector<type> &X, vector<type> &Y)
{
    int n = X.size();
    float sum_X = 0, sum_Y = 0,
                    sum_XY = 0;
    float squareSum_X = 0,
        squareSum_Y = 0;

    for (int i = 0; i < n; i++)
    {
        // sum of elements of array X.
        sum_X = sum_X + X[i];

        // sum of elements of array Y.
        sum_Y = sum_Y + Y[i];

        // sum of X[i] * Y[i].
        sum_XY = sum_XY + X[i] * Y[i];

        // sum of square of array elements.
        squareSum_X = squareSum_X +
                      X[i] * X[i];
        squareSum_Y = squareSum_Y +
                      Y[i] * Y[i];
    }

    // use formula for calculating
    // correlation coefficient.
    float corr = (float)(n * sum_XY -
                  sum_X * sum_Y) /
                  sqrt((n * squareSum_X -
                       sum_X * sum_X) *
                       (n * squareSum_Y -
                       sum_Y * sum_Y));

    return corr;
}


int main(int argc, char *argv[])
{
    try
    {
        cout << "Hello OpenNN!" << endl;

        Tensor<type, 1> v(4);
        v(0) = 20;
        v(1) = 30;
        v(2) = 10;
        v(2) = 10;

        cout << rankify(v) << endl;
/*
        Tensor<Index, 1> rank = calculate_rank_less(v);

        cout << rank << endl;

        Tensor<type, 1> rank_values(4);

        for(Index i = 0; i < rank.size(); i++)
        {
            rank_values(rank(i)) = (type)i;
        }

        cout << endl;

        cout << rank_values << endl;


//        v(0) = 2;
//        v(1) = 3;
//        v(2) = 1;



        cout << "Bye OpenNN!" << endl;
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}
*/
// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
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

