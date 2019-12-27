//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M E T R I C S   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "metrics_test.h"


MetricsTest::MetricsTest() : UnitTesting()
{
}


MetricsTest::~MetricsTest()
{
}


void MetricsTest::test_constructor()
{
   cout << "test_constructor\n"; 

}


void MetricsTest::test_destructor()
{
   cout << "test_destructor\n"; 
}


void MetricsTest::test_l1_norm()
{
   cout << "test_l1_norm\n";

   Vector<double> vector;
   vector.set(3);
   vector[0]=1;
   vector[1]=2;
   vector[2]=3;

   assert_true(abs(l1_norm(vector) - 6.0) < numeric_limits<double>::min(), LOG);

}

void MetricsTest::test_l1_norm_gradient()
{
   cout << "test_l1_norm_gradient\n";

   Vector<double> vector(3);
   Vector<double> gradient(3);
   vector[0]=1;
   vector[1]=-2.3;
   vector[2]=3.3;
   gradient=l1_norm_gradient(vector);
   assert_true(abs(gradient[0] - 1.0) < numeric_limits<double>::min(), LOG);
   assert_true(abs(gradient[1] + 1.0) < numeric_limits<double>::min(), LOG);
   assert_true(abs(gradient[2] - 1.0) < numeric_limits<double>::min(), LOG);

}


void MetricsTest::test_l1_norm_hessian()
{
   cout << "test_l1_norm_hessian\n";
   Vector<double> vector(3);
   Matrix<double> hessian(3,3);
   vector[0]=1;
   vector[1]=-2.3;
   vector[2]=3.3;
   hessian=l1_norm_hessian(vector);

   assert_true(hessian==0, LOG);

}

void MetricsTest::test_l2_norm()
{
   cout << "test_l2_norm\n";
   Vector<double> vector(3);
   vector[0]=3;
   vector[1]=4;
   vector[2]=5;

   assert_true(abs(l2_norm(vector) - sqrt(50)) < numeric_limits<double>::min(), LOG);
}


void MetricsTest::test_l2_norm_gradient()
{
   cout << "test_l2_norm_gradient\n";

   Vector<double> vector(3);
   Vector<double> gradient;
   vector[0]=3;
   vector[1]=4;
   vector[2]=5;

   gradient = l2_norm_gradient(vector);

   assert_true(gradient.size() == 3, LOG);
   assert_true(abs(gradient[0] - 3/sqrt(50)) < numeric_limits<double>::min(), LOG);
   assert_true(abs(gradient[1] - 4/sqrt(50)) < numeric_limits<double>::min(), LOG);
   assert_true(abs(gradient[2] - 5/sqrt(50)) < numeric_limits<double>::min(), LOG);
}


void MetricsTest::test_l2_norm_hessian()
{
   cout << "test_l2_norm_hessian\n";

}


void MetricsTest::test_Lp_norm()
{
   cout << "test_Lp_norm\n";

   Vector<double> vector(4);

   vector[0]=0;
   vector[1]=1;
   vector[2]=1;
   vector[3]=2;

   assert_true(abs(lp_norm(vector,4) - pow(18,0.25)) < numeric_limits<double>::min(), LOG);
}


/// @todo

void MetricsTest::test_Lp_norm_gradient()
{
   cout << "test_Lp_gradient\n";

   Vector<double> vector({1,2});
   Vector<double> gradient;
   const double p = 8*pow(pow(17,0.75), -1.0);

//   gradient = lp_norm_gradient(vector,4);

   assert_true(gradient.size() == 3, LOG);

   assert_true(abs(gradient[1] - p) < numeric_limits<double>::min(), LOG);
}


void MetricsTest::test_determinant()
{
    cout << "test_Ldeterminat\n";

    Matrix<double> matrix;

    matrix = {{1,2,3},{1,2,0},{2,2,0}};

    assert_true(abs(determinant(matrix) + 6) < numeric_limits<double>::min(), LOG);
}


void MetricsTest::test_cofactor()
{
    cout << "test_cofactor\n";

    Matrix<double> matrix;
    Matrix<double> cofact;

    matrix = {{1,2,3},{1,2,0},{2,2,0}};
    cofact=cofactor(matrix);

    assert_true(abs(cofact(0,1) - 6) < numeric_limits<double>::min(), LOG);
    assert_true(abs(cofact(1,2) - 3) < numeric_limits<double>::min(), LOG);
    assert_true(abs(cofact(2,2) - 0) < numeric_limits<double>::min(), LOG);
    assert_true(abs(cofact(1,0) - 0) < numeric_limits<double>::min(), LOG);
}


void MetricsTest::test_inverse()
{
    cout << "test_inverse\n";

    Matrix<double> matrix;
    Matrix<double> inver;

    matrix = {{1,2,3},{1,2,0},{2,2,0}};
    inver=inverse(matrix);

    assert_true(inver(0,1) == 0, LOG);
    assert_true(inver(2,1) == -0.5, LOG);
    assert_true(inver(2,2) == 0, LOG);
    assert_true(inver(1,0) == -1, LOG);

}

void MetricsTest::test_eigenvalues()
{
    cout << "test_eigenvalues\n";

    Matrix<double> matrix;
    Matrix<double> eig;

    matrix={{1,2},{1,2}};
    eig=eigenvalues(matrix);

    assert_true(eig(0,0) == 0, LOG);
}

void MetricsTest::test_eigenvectors()
{
    cout << "test_eigenvectors\n";

    Matrix<double> matrix;
    Matrix<double> eig;

    matrix = {{1,2},{1,2}};

    eig=eigenvectors(matrix);

    assert_true(eig(0,0) == 0, LOG);
}


void MetricsTest::test_direct()
{
    cout << "test_direct\n";

    Matrix<double> matrix_1;
    Matrix<double> matrix_2;
    Matrix<double> result;

    //Test

    matrix_1 = {{1,2}};
    matrix_2 = {{1,3}};
    result = direct(matrix_1,matrix_2);

    //assert_true(result.get_columns_number() == )
    assert_true(result(0,1) == 3, LOG);

}


void MetricsTest::test_linear_combinations()
{
   cout << "test_linear_combinations\n";

   Tensor<double> matrix_1;
   Matrix<double> matrix_2;

   Vector<double> vector;
   Tensor<double> result;

   // Test

   matrix_1.set({1,1}, 1.0);
   matrix_2.set(1,1, 2.0);
   vector.set(1, 3.0);

   result = linear_combinations(matrix_1, matrix_2, vector);

   assert_true(result.get_dimensions_number() == 2, LOG);
   assert_true(result.get_dimension(0) == 1, LOG);
   assert_true(result.get_dimension(1) == 1, LOG);
   assert_true(result(0,0) == 5.0, LOG);

   // Test

   matrix_1.set({2,3}, 1.0);
   matrix_2.set(3, 4, 2.0);
   vector.set(4, 3.0);

   //result =

}


void MetricsTest::test_euclidean_distance()
{
     cout << "test_euclidean_distance\n";

}

void MetricsTest::test_euclidean_weighted_distance()
{
     cout << "test_euclidean_weighted_distance\n";

     Vector<double> vector_1;
     Vector<double> vector_2;
     double dis;

     vector_1={{1,2,3}};
     vector_2={{2,3,4}};

     dis=euclidean_weighted_distance(vector_1,vector_2,{2,1,0});

     assert_true(dis == sqrt(3), LOG);

}


void MetricsTest::test_euclidean_weighted_distance_vector()
{
     cout << "test_euclidean_weighted__distance_vector\n";

     Vector<double> vector_1;
     Vector<double> vector_2;
     double dis;

     vector_1={{1,2,5}};
     vector_2={{2,3,4}};

     dis=euclidean_weighted_distance(vector_1,vector_2,{2,1,3});

     assert_true(dis == sqrt(6), LOG);

}


void MetricsTest::test_manhattan_distance()
{
     cout << "test_manhattan_distance\n";

     Matrix <double> M;
     double dis_1;
     double dis_2;
     double dis_3;

     M = {{1,2,5},{2,3,4},{6,8,9}};
     dis_1 = manhattan_distance(M,0,1);
     dis_2 = manhattan_distance(M,0,2);
     dis_3 = manhattan_distance(M,1,2);

     assert_true(dis_1 == 4, LOG);
     assert_true(dis_2 == 9, LOG);
     assert_true(dis_3 == 5, LOG);

}

void MetricsTest::test_manhattan_weighted_distance()
{
     cout << "test_manhattan__weighted_distance\n";

     Matrix <double> M;
     double dis_1;
     double dis_2;
     double dis_3;

     M = {{1,2,5},{2,3,4},{6,8,9}};
     dis_1 = manhattan_distance(M,0,1);
     dis_2 = manhattan_distance(M,0,2);
     dis_3 = manhattan_distance(M,1,2);

     assert_true(dis_1 == 4, LOG);
     assert_true(dis_2 == 9, LOG);
     assert_true(dis_3 == 5, LOG);

}





void MetricsTest::run_test_case()
{
   cout << "Running linear algebra test case...\n";

   // Constructor and destructor methods

   test_linear_combinations();

   test_constructor();

   test_destructor();

   test_l1_norm();

   test_l1_norm_gradient();

   test_l1_norm_hessian();

   test_l2_norm();

   test_l2_norm_gradient();

   test_l2_norm_hessian();

   test_Lp_norm();

   test_Lp_norm_gradient();

   test_determinant();

   test_cofactor();

   test_inverse();

   test_eigenvalues();

   test_eigenvectors();

   test_direct();



   test_euclidean_distance();

   test_euclidean_weighted_distance();

   test_euclidean_weighted_distance_vector();

   test_manhattan_distance();

   test_manhattan_weighted_distance();

 //  test_manhattan_weighted_distance_vector();

   cout << "End of linear algebra test case.\n";
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques, SL.
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
