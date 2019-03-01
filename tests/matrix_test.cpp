/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   M A T R I X   T E S T   C L A S S                                                                          */
/*                                                                                                              */ 
 
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "matrix_test.h"

// GENERAL CONSTRUCTOR

MatrixTest::MatrixTest() : UnitTesting()
{   
}


// DESTRUCTOR

MatrixTest::~MatrixTest()
{
}


// METHODS

void MatrixTest::test_constructor()
{
   message += "test_constructor\n";

   string file_name = "../data/matrix.dat";

   // Default

   Matrix<size_t> m1;

   assert_true(m1.get_rows_number() == 0, LOG);
   assert_true(m1.get_columns_number() == 0, LOG);

   // Rows and columns numbers

   Matrix<size_t> m2(0, 0);

   assert_true(m2.get_rows_number() == 0, LOG);
   assert_true(m2.get_columns_number() == 0, LOG);
  
   Matrix<double> m3(1, 1, 1.0);
   assert_true(m3.get_rows_number() == 1, LOG);
   assert_true(m3.get_columns_number() == 1, LOG);

   // Rows and columns numbers and initialization

   Matrix<size_t> m4(0, 0, 1);

   assert_true(m4.get_rows_number() == 0, LOG);
   assert_true(m4.get_columns_number() == 0, LOG);

   Matrix<size_t> m5(1, 1, 1);

   assert_true(m5.get_rows_number() == 1, LOG);
   assert_true(m5.get_columns_number() == 1, LOG);
   assert_true(m5 == true, LOG);

   // File constructor

   m1.save(file_name);

   Matrix<size_t> m6(file_name);
   assert_true(m6.get_rows_number() == 0, LOG);
   assert_true(m6.get_columns_number() == 0, LOG);

   m2.save(file_name);
   Matrix<size_t> m7(file_name);
   assert_true(m7.get_rows_number() == 0, LOG);
   assert_true(m7.get_columns_number() == 0, LOG);

   m3.save(file_name);

   Matrix<double> m8(file_name);
   assert_true(m8.get_rows_number() == 1, LOG);
   assert_true(m8.get_columns_number() == 1, LOG);

   m4.save(file_name);
   Matrix<size_t> m9(file_name);
   assert_true(m9.get_rows_number() == 0, LOG);
   assert_true(m9.get_columns_number() == 0, LOG);

   m5.save(file_name);

   Matrix<size_t> m10(file_name);
   assert_true(m10.get_rows_number() == 1, LOG);
   assert_true(m10.get_columns_number() == 1, LOG);
   assert_true(m10 == true, LOG); 

   // Copy constructor

   Matrix<double> a5;
   Matrix<double> b5(a5);

   assert_true(b5.get_rows_number() == 0, LOG);
   assert_true(b5.get_columns_number() == 0, LOG);

   Matrix<size_t> a6(1, 1, true);

   Matrix<size_t> b6(a6);

   assert_true(b6.get_rows_number() == 1, LOG);
   assert_true(b6.get_columns_number() == 1, LOG);
   assert_true(b6 == true, LOG);

   // Operator ++

   Matrix<size_t> m11(2, 2, 0);
   m11(0,0)++;
   m11(1,1)++;

   assert_true(m11(0,0) == 1, LOG);
   assert_true(m11(0,1) == 0, LOG);
   assert_true(m11(1,0) == 0, LOG);
   assert_true(m11(1,1) == 1, LOG);
}


void MatrixTest::test_destructor()
{  
   message += "test_destructor\n";
}


void MatrixTest::test_assignment_operator()
{
   message += "test_assignment_operator\n";

   Matrix<int> a(1, 1, 0);

   Matrix<int> b = a;

   for(size_t i = 0; i < 2; i++)
   {
      b = a;
   }

   assert_true(b.get_rows_number() == 1, LOG);
   assert_true(b.get_columns_number() == 1, LOG);
   assert_true(b == 0, LOG);
}


void MatrixTest::test_reference_operator()
{
   message += "test_reference_operator\n";
}


void MatrixTest::test_sum_operator()
{
   message += "test_sum_operator\n";

   Matrix<int> a(1, 1, 1);
   Matrix<int> b(1, 1, 1);
   Matrix<int> c(1, 1);

   // Test
   
   c = a + 1;

   assert_true(c.get_rows_number() == 1, LOG);
   assert_true(c.get_columns_number() == 1, LOG);
   assert_true(c == 2, LOG);

   // Test

   c = a + b;

   assert_true(c.get_rows_number() == 1, LOG);
   assert_true(c.get_columns_number() == 1, LOG);
   assert_true(c == 2, LOG);
}


void MatrixTest::test_rest_operator()
{
   message += "test_rest_operator\n";

   Matrix<int> a(1, 1, 1);
   Matrix<int> b(1, 1, 1);
   Matrix<int> c(1, 1);
   Matrix<int> d;

   // Test

   c = a - 1;

   assert_true(c.get_rows_number() == 1, LOG);
   assert_true(c.get_columns_number() == 1, LOG);
   assert_true(c == 0, LOG);

   // Test

   c = a - b;

   assert_true(c.get_rows_number() == 1, LOG);
   assert_true(c.get_columns_number() == 1, LOG);
   assert_true(c == 0, LOG);

   // Test

   a.set(3, 3, 1);
   b.set(3, 3, 1);
   c.set(3, 3, 1);

   d = a + b - c;

   assert_true(d.get_rows_number() == 3, LOG);
   assert_true(d.get_columns_number() == 3, LOG);
   assert_true(d == 1, LOG);

}


void MatrixTest::test_multiplication_operator()
{
   message += "test_multiplication_operator\n";

   Matrix<int> a;
   Matrix<int> b;
   Matrix<int> c;
   
   Vector<int> v;

   // Scalar

   a.set(1, 1, 2);

   c = a*2;

   assert_true(c.get_rows_number() == 1, LOG);
   assert_true(c.get_columns_number() == 1, LOG);
   assert_true(c == 4, LOG);

   // Vector

   a.set(1, 1, 1);
   v.set(1, 1);
  
   // Matrix

   a.set(1, 1, 2);
   b.set(1, 1, 2);

   c = a*b;

   assert_true(c.get_rows_number() == 1, LOG);
   assert_true(c.get_columns_number() == 1, LOG);
   assert_true(c == 4, LOG);

}


void MatrixTest::test_division_operator()
{
   message += "test_division_operator\n";

   Matrix<int> a(1, 1, 2);
   Matrix<int> b(1, 1, 2);
   Matrix<int> c(1, 1);
   
   c = a/2;

   assert_true(c.get_rows_number() == 1, LOG);
   assert_true(c.get_columns_number() == 1, LOG);
   assert_true(c == 1, LOG);

   c = a/b;

   assert_true(c.get_rows_number() == 1, LOG);
   assert_true(c.get_columns_number() == 1, LOG);
   assert_true(c == 1, LOG);
}


void MatrixTest::test_sum_assignment_operator()
{
   message += "test_sum_assignment_operator\n";
}


void MatrixTest::test_rest_assignment_operator()
{
   message += "test_rest_assignment_operator\n";
}


void MatrixTest::test_multiplication_assignment_operator()
{
   message += "test_multiplication_assignment_operator\n";
}


void MatrixTest::test_division_assignment_operator()
{
   message += "test_division_assignment_operator\n";
}


void MatrixTest::test_equal_to_operator()
{
	message += "test_equal_to_operator\n";

   Matrix<int> a(1,1,0);
   Matrix<int> b(1,1,0);
   Matrix<int> c(1,1,1);

   assert_true(a == b, LOG);
   assert_false(a == c, LOG);
}


void MatrixTest::test_not_equal_to_operator()
{
   message += "test_not_equal_to_operator\n";

   Matrix<int> a(1,1,0);
   Matrix<int> b(1,1,0);
   Matrix<int> c(1,1,1);

   assert_false(a != b, LOG);
   assert_true(a != c, LOG);
}


void MatrixTest::test_greater_than_operator()
{
   message += "test_greater_than_operator\n";

   Matrix<double> a(1,1,1.0);
   Matrix<double> b(1,1,0.0);

   assert_true(a > 0.0, LOG);
   assert_true(a > b, LOG);
}


void MatrixTest::test_less_than_operator()
{
   message += "test_less_than_operator\n";

   Matrix<double> a(1,1,0.0);
   Matrix<double> b(1,1,1.0);

   assert_true(a < 1.0, LOG);
   assert_true(a < b, LOG);
}


void MatrixTest::test_greater_than_or_equal_to_operator()
{
   message += "test_greater_than_or_equal_to_operator\n";

   Matrix<double> a(1,1,1.0);
   Matrix<double> b(1,1,1.0);

   assert_true(a >= 1.0, LOG);
   assert_true(a >= b, LOG);
}


void MatrixTest::test_less_than_or_equal_to_operator()
{
   message += "test_less_than_or_equal_to_operator\n";

   Matrix<double> a(1,1,1.0);
   Matrix<double> b(1,1,1.0);

   assert_true(a <= 1.0, LOG);
   assert_true(a <= b, LOG);
}


void MatrixTest::test_output_operator()
{
   message += "test_output_operator\n";

   Matrix<double> m1;
   Matrix< Vector<double> > m2;
   Matrix< Matrix<size_t> > m3;

   // Test

   m1.set(2, 3, 0.0);

   // Test

   m2.set(2, 2);
   m2(0,0).set(1, 0.0);
   m2(0,1).set(1, 1.0);
   m2(1,0).set(1, 0.0);
   m2(1,1).set(1, 1.0);

   // Test

   m3.set(2, 2);
   m3(0,0).set(1, 1, 0);
   m3(0,1).set(1, 1, 1);
   m3(1,0).set(1, 1, 0);
   m3(1,1).set(1, 1, 1);
}


void MatrixTest::test_get_rows_number()
{
   message += "test_get_rows_number\n";

   Matrix<size_t> m(2,3);

   size_t rows_number = m.get_rows_number();

   assert_true(rows_number == 2, LOG);
}


void MatrixTest::test_get_columns_number()
{
   message += "test_get_columns_number\n";

   Matrix<size_t> m(2,3);

   size_t columns_number = m.get_columns_number();

   assert_true(columns_number == 3, LOG);
}


void MatrixTest::test_get_row()
{
   message += "test_get_row\n";

   Matrix<int> m(1, 1, 0);

   Vector<int> row = m.get_row(0);

   assert_true(row == 0, LOG);
}


void MatrixTest::test_get_column()
{
   message += "test_get_column\n";

   Matrix<int> m(1, 1, 0);

   Vector<int> column = m.get_column(0);

   assert_true(column == 0, LOG);
}


void MatrixTest::test_get_submatrix()
{
   message += "test_get_submatrix\n";
}


void MatrixTest::test_set()
{
   message += "test_set\n";

   string file_name = "../data/matrix.dat";

   Matrix<double> m;

   // Default

   m.set();

   assert_true(m.get_rows_number() == 0, LOG);
   assert_true(m.get_columns_number() == 0, LOG);

   // Numbers of rows and columns

   m.set(0, 0);

   assert_true(m.get_rows_number() == 0, LOG);
   assert_true(m.get_columns_number() == 0, LOG);

   m.set(2, 3);

   assert_true(m.get_rows_number() == 2, LOG);
   assert_true(m.get_columns_number() == 3, LOG);

   m.set(0, 0);

   assert_true(m.get_rows_number() == 0, LOG);
   assert_true(m.get_columns_number() == 0, LOG);

   // Initialization 

   m.set(3, 2, 1.0);

   assert_true(m.get_rows_number() == 3, LOG);
   assert_true(m.get_columns_number() == 2, LOG);
   assert_true(m == 1.0, LOG);

   // File 

   m.save(file_name);
   m.set(file_name);

   assert_true(m.get_rows_number() == 3, LOG);
   assert_true(m.get_columns_number() == 2, LOG);
   assert_true(m == 1.0, LOG);
}


void MatrixTest::test_set_rows_number()
{
   message += "test_set_rows_number\n";
}


void MatrixTest::test_set_columns_number()
{
   message += "test_set_columns_number\n";
}


void MatrixTest::test_set_row()
{
   message += "test_set_row\n";

   Matrix<double> m(1,1);

   Vector<double> row(1, 1.0);

   m.set_row(0, row);

   assert_true(m.get_row(0) == row, LOG);
}


void MatrixTest::test_set_column()
{
   message += "test_set_column\n";

   Matrix<double> m(1,1);

   Vector<double> column(1, 1.0);

   m.set_column(0, column);

   assert_true(m.get_column(0) == column, LOG);
}


void MatrixTest::test_get_diagonal()
{
   message += "test_get_diagonal\n";

   Matrix<size_t> m(2, 2, 1);

   Vector<size_t> diagonal = m.get_diagonal();

   assert_true(diagonal.size() == 2, LOG);
   assert_true(diagonal == 1, LOG);
}


void MatrixTest::test_set_diagonal()
{
   message += "test_set_diagonal\n";

   Matrix<size_t> m;
   Vector<size_t> diagonal;

   // Test

   m.set(2, 2, 1);

   m.set_diagonal(0);

   diagonal = m.get_diagonal();

   assert_true(diagonal.size() == 2, LOG);
   assert_true(diagonal == 0, LOG);

   // Test

   diagonal.set(2);
   diagonal[0] = 1;
   diagonal[1] = 0;

   m.set_diagonal(diagonal);

   diagonal = m.get_diagonal();

   assert_true(diagonal.size() == 2, LOG);
   assert_true(diagonal[0] == 1, LOG);
   assert_true(diagonal[1] == 0, LOG);
}


void MatrixTest::test_sum_diagonal()
{
   message += "test_sum_diagonal\n";

   Matrix<int> m;
   Matrix<int> sum;  
   Vector<int> diagonal;

   // Test

   m.set(2, 2, 1);

   m.sum_diagonal(1);

   diagonal = m.get_diagonal();

   assert_true(diagonal.size() == 2, LOG);
   assert_true(diagonal == 2, LOG);
}


void MatrixTest::test_append_row()
{
   message += "test_append_row\n";

   Matrix<size_t> m(1, 1, 0);

   Vector<size_t> v(1, 1);

   m.append_row(v);

   assert_true(m.get_rows_number() == 2, LOG);
   assert_true(m(1,0) == 1, LOG);
}


void MatrixTest::test_append_column()
{
   message += "test_append_column\n";

   Matrix<size_t> m(1, 1, 0);

   Vector<size_t> v(1, 1);

   m = m.append_column(v);

   assert_true(m.get_columns_number() == 2, LOG);
   assert_true(m(0,1) == 1, LOG);
}


void MatrixTest::test_insert_row()
{
   message += "test_insert_row\n";

   Matrix<size_t> m(2, 1, 0);

   Vector<size_t> v(1, 1);

   m = m.insert_row(1, v);

   assert_true(m.get_rows_number() == 3, LOG);
   assert_true(m(1,0) == 1, LOG);
}


void MatrixTest::test_insert_column()
{
   message += "test_insert_column\n";

   Matrix<size_t> m(1, 2, 0);

   Vector<size_t> v(1, 1);

   m = m.insert_column(1, v);

   assert_true(m.get_columns_number() == 3, LOG);
   assert_true(m(0,1) == 1, LOG);
}


// @todo

void MatrixTest::test_subtract_row()
{
//   message += "test_subtract_row\n";

//   Matrix<size_t> m(2, 1);
//   m(0,0) = true;
//   m(1,0) = false;

//   m.subtract_row(0);

//   assert_true(m.get_rows_number() == 1, LOG);
//   assert_true(m(0,0) == false, LOG);
}


// @todo

void MatrixTest::test_subtract_column()
{
//   message += "test_subtract_column\n";

//   Matrix<size_t> m(1, 2, false);
//   m(0,0) = true;
//   m(0,1) = false;

//   m.subtract_column(0);

//   assert_true(m.get_columns_number() == 1, LOG);
//   assert_true(m(0,0) == false, LOG);
}


// @todo

void MatrixTest::test_sort_less_rows()
{
//    message += "test_sort_less_rows";

//    Matrix<double> m;

//    Matrix<double> sorted_m;

//    //Test

//    m.set(3, 3);
//    sorted_m.set(3, 3);

//    m(0, 0) =  5;   m(0, 1) = 0.9;   m(0, 2) =  0.8;
//    m(1, 0) =  9;   m(1, 1) =   7;   m(1, 2) =    5;
//    m(2, 0) = -2;   m(2, 1) =   8;   m(2, 2) = -0.9;

//    sorted_m = m.sort_less_rows(0);

//    assert_true(sorted_m(0, 0) == -2, LOG);
//    assert_true(sorted_m(0, 1) == 8, LOG);
//    assert_true(sorted_m(0, 2) == -0.9, LOG);
//    assert_true(sorted_m(1, 0) == 5, LOG);
//    assert_true(sorted_m(1, 1) == 0.9, LOG);
//    assert_true(sorted_m(1, 2) == 0.8, LOG);
//    assert_true(sorted_m(2, 0) == 9, LOG);
//    assert_true(sorted_m(2, 1) == 7, LOG);
//    assert_true(sorted_m(2, 2) == 5, LOG);

//    //Test

//    m.set(6, 2);
//    sorted_m.set(6, 2);

//    m(0, 0) =  0.33;   m(0, 1) = 0.9;
//    m(1, 0) =  0.33;   m(1, 1) =   7;
//    m(2, 0) =  0.33;   m(2, 1) =   8;
//    m(3, 0) =  0.33;   m(3, 1) = 0.9;
//    m(4, 0) =  0.9;   m(4, 1) =   7;
//    m(5, 0) =  0.2;   m(5, 1) =   8;

//    sorted_m = m.sort_less_rows(0);

//    assert_true(sorted_m(0, 0) == 0.2, LOG);
//    assert_true(sorted_m(0, 1) == 8, LOG);
//    assert_true(sorted_m(1, 0) == 0.33, LOG);
//    assert_true(sorted_m(1, 1) == 0.9, LOG);
//    assert_true(sorted_m(2, 0) == 0.33, LOG);
//    assert_true(sorted_m(2, 1) == 7, LOG);
//    assert_true(sorted_m(3, 0) == 0.33, LOG);
//    assert_true(sorted_m(3, 1) == 8, LOG);
//    assert_true(sorted_m(4, 0) == 0.33, LOG);
//    assert_true(sorted_m(4, 1) == 0.9, LOG);
//    assert_true(sorted_m(5, 0) == 0.9, LOG);
//    assert_true(sorted_m(5, 1) == 7, LOG);
}


// @todo

void MatrixTest::test_sort_greater_rows()
{
//    message += "test_sort_greater_rows";

//    Matrix<double> m;

//    Matrix<double> sorted_m;

//    //Test

//    m.set(3, 3);
//    sorted_m.set(3, 3);

//    m(0, 0) =  5;   m(0, 1) = 0.9;   m(0, 2) =  0.8;
//    m(1, 0) =  9;   m(1, 1) =   7;   m(1, 2) =    5;
//    m(2, 0) = -2;   m(2, 1) =   8;   m(2, 2) = -0.9;

//    sorted_m = m.sort_greater_rows(2);

//    assert_true(sorted_m(0, 0) == 9, LOG);
//    assert_true(sorted_m(0, 1) == 7, LOG);
//    assert_true(sorted_m(0, 2) == 5, LOG);
//    assert_true(sorted_m(1, 0) == 5, LOG);
//    assert_true(sorted_m(1, 1) == 0.9, LOG);
//    assert_true(sorted_m(1, 2) == 0.8, LOG);
//    assert_true(sorted_m(2, 0) == -2, LOG);
//    assert_true(sorted_m(2, 1) == 8, LOG);
//    assert_true(sorted_m(2, 2) == -0.9, LOG);

//    //Test

//    m.set(6, 2);
//    sorted_m.set(6, 2);

//    m(0, 0) =  0.33;   m(0, 1) = 0.9;
//    m(1, 0) =  0.33;   m(1, 1) =   7;
//    m(2, 0) =  0.33;   m(2, 1) =   8;
//    m(3, 0) =  0.33;   m(3, 1) = 0.9;
//    m(4, 0) =  0.9;   m(4, 1) =   7;
//    m(5, 0) =  0.2;   m(5, 1) =   8;

//    sorted_m = m.sort_greater_rows(0);

//    assert_true(sorted_m(0, 0) == 0.9, LOG);
//    assert_true(sorted_m(0, 1) == 7, LOG);
//    assert_true(sorted_m(1, 0) == 0.33, LOG);
//    assert_true(sorted_m(1, 1) == 0.9, LOG);
//    assert_true(sorted_m(2, 0) == 0.33, LOG);
//    assert_true(sorted_m(2, 1) == 7, LOG);
//    assert_true(sorted_m(3, 0) == 0.33, LOG);
//    assert_true(sorted_m(3, 1) == 8, LOG);
//    assert_true(sorted_m(4, 0) == 0.33, LOG);
//    assert_true(sorted_m(4, 1) == 0.9, LOG);
//    assert_true(sorted_m(5, 0) == 0.2, LOG);
//    assert_true(sorted_m(5, 1) == 8, LOG);
}


void MatrixTest::test_initialize()
{
   message += "test_initialize\n";
}


void MatrixTest::test_randomize_uniform()
{
   message += "test_randomize_uniform\n";

   Matrix<double> m(1, 1);

   m.randomize_uniform();

   assert_true(m >= -1.0, LOG);
   assert_true(m <=  1.0, LOG);

   m.randomize_uniform(-1.0, 0.0);

   assert_true(m >= -1.0, LOG);
   assert_true(m <=  0.0, LOG);
}


void MatrixTest::test_randomize_normal()
{
   message += "test_randomize_normal\n";
}


void MatrixTest::test_set_to_identity()
{
   message += "test_set_to_identity\n";

   Matrix<int> a(2, 2);
   a.initialize_identity();

   Matrix<int> b(2, 2);
   b(0,0) = 1;
   b(0,1) = 0;
   b(1,0) = 0;
   b(1,1) = 1;

   assert_true(a == b, LOG);
}


void MatrixTest::test_calculate_sum()
{
    message += "test_calculate_sum\n";
}


void MatrixTest::test_calculate_rows_sum()
{
    message += "test_calculate_rows_sum\n";
}


void MatrixTest::test_dot_vector()
{
   message += "test_dot_vector\n";

   Matrix<double> a;
   Vector<double> b;

   Vector<double> c;

   // Test

   a.set(2, 2, 0.0);
   b.set(2, 0.0);

   c = a.dot(b);

   assert_true(c == 0.0, LOG);

   // Test

   a.set(2, 2, 1.0);
   b.set(2, 1.0);

   c = a.dot(b);

   assert_true(c == 2.0, LOG);

   // Test

   a.set(2, 5);
   a.randomize_normal();

   b.set(5);
   b.randomize_normal();

   c = a.dot(b);

   assert_true((c - dot(a, b)).calculate_absolute_value() < 1.0e-3, LOG);

   // Test

   a.set(2, 2);
   a(0,0) = 1.0;
   a(0,1) = 2.0;
   a(1,0) = 3.0;
   a(1,1) = 4.0;

   b.set(2);
   b[0] = -1.0;
   b[1] =  1.0;

   c = a.dot(b);

   assert_true(c == 1.0, LOG);
}


void MatrixTest::test_dot_matrix()
{
   message += "test_dot_matrix\n";

   Matrix<double> a;
   Matrix<double> b;

   Matrix<double> c;

   // Test

   a.set(2, 2, 0.0);
   b.set(2, 2, 0.0);

   c = a.dot(b);

   assert_true(c == 0.0, LOG);

   // Test

   a.set(2, 2, 1.0);
   b.set(2, 2, 1.0);

   c = a.dot(b);

   assert_true(c == 2.0, LOG);

   // Test

   a.set(2, 2);
   a(0,0) = 1.0;
   a(0,1) = 2.0;
   a(1,0) = 3.0;
   a(1,1) = 4.0;

   b = a;

   c = a.dot(b);

   assert_true(c(0,0) ==  7.0, LOG);
   assert_true(c(0,1) == 10.0, LOG);
   assert_true(c(1,0) == 15.0, LOG);
   assert_true(c(1,1) == 22.0, LOG);

   // Test

   a.set(3, 2);
   a.randomize_normal();

   b.set(2, 3);
   b.randomize_normal();

   c = a.dot(b);

   assert_true((c - dot(a, b)).calculate_absolute_value() < 1.0e-3, LOG);
}


void MatrixTest::test_calculate_eigenvalues()
{
    message += "test_calculate_eigenvalues\n";

    Matrix<double> eigenvalues;

    Matrix<double> m;

    // Test

    m.set(10,10);

    m.randomize_normal();

    eigenvalues = m.calculate_eigenvalues();

    assert_true(eigenvalues.size() == 10, LOG);

    // Test

    m.set_identity(20);

    eigenvalues = m.calculate_eigenvalues();

    assert_true(eigenvalues.size() == 20, LOG);
    assert_true(eigenvalues.get_column(0).is_constant(1.0), LOG);
}


void MatrixTest::test_calculate_eigenvectors()
{
    message += "test_calculate_eigenvectors\n";

    Matrix<double> eigenvectors;

    Matrix<double> m;

    // Test

    m.set(10,10);

    m.randomize_normal();

    eigenvectors = m.calculate_eigenvectors();

    assert_true(eigenvectors.get_rows_number() == 10, LOG);
    assert_true(eigenvectors.get_columns_number() == 10, LOG);
}


void MatrixTest::test_direct()
{
   message += "test_direct\n";

   Matrix<int> a;
   Matrix<int> b;
   Matrix<int> direct;

   // Test

   a.set(2,2);
   a(0,0) = 1;
   a(0,1) = 2;
   a(1,0) = 3;
   a(1,1) = 4;

   b.set(2,2);
   b(0,0) = 0;
   b(0,1) = 5;
   b(1,0) = 6;
   b(1,1) = 7;

   direct = a.direct(b);

   assert_true(direct.get_rows_number() == 4, LOG);
   assert_true(direct.get_columns_number() == 4, LOG);
   assert_true(direct(0,0) == 0, LOG);
   assert_true(direct(3,3) == 28, LOG);
}


void MatrixTest::test_calculate_mean_standard_deviation()
{
   message += "test_calculate_mean_standard_deviation\n";
}


void MatrixTest::test_calculate_statistics()
{
   message += "test_calculate_statistics\n";
}


void MatrixTest::test_calculate_histogram()
{
   message += "test_calculate_histogram\n";

   Matrix<double> m;

   Vector< Histogram<double> >  histograms;
   Histogram<double> histogram;

   size_t bins_number;

   // Test

   m.set(2, 3);
   m.randomize_normal();

   bins_number = 1;

   histograms = m.calculate_histograms(bins_number);

   assert_true(histograms.size() == m.get_columns_number(), LOG);

   if (!m.get_column(0).is_binary())
   {
       assert_true(histograms[0].get_bins_number() == bins_number, LOG);
   }

   // Test

   m.set(4, 3);
   m.randomize_normal();

   bins_number = 4;

   histograms = m.calculate_histograms(bins_number);

   assert_true(histograms.size() == m.get_columns_number(), LOG);
   if (!m.get_column(0).is_binary())
   {
       assert_true(histograms[0].get_bins_number() == bins_number, LOG);
   }
}


void MatrixTest::test_calculate_covariance_matrix()
{
    message += "test_calculate_covariance_matrix\n";

    Matrix<double> covariance_matrix;

    Matrix<double> data;

    // Test

    data.set(10,5);
    data.randomize_normal();

    covariance_matrix = data.calculate_covariance_matrix();

    assert_true(covariance_matrix.get_rows_number() == 5, LOG);
    assert_true(covariance_matrix.get_columns_number() == 5, LOG);
    assert_true(covariance_matrix.is_symmetric(), LOG);

    // Test

    data.set(10,20);
    data.randomize_normal();

    covariance_matrix = data.calculate_covariance_matrix();

    assert_true(covariance_matrix.get_rows_number() == 20, LOG);
    assert_true(covariance_matrix.get_columns_number() == 20, LOG);
    assert_true(covariance_matrix.is_symmetric(), LOG);
}


void MatrixTest::test_calculate_minimal_indices()
{
   message += "test_calculate_minimal_indices\n";
}


void MatrixTest::test_calculate_maximal_indices()
{
   message += "test_calculate_maximal_indices\n";
}


void MatrixTest::test_calculate_minimal_maximal_indices()
{
   message += "test_calculate_minimal_maximal_indices\n";
}


void MatrixTest::test_calculate_sum_squared_error()
{
   message += "test_calculate_sum_squared_error\n";
}


void MatrixTest::test_calculate_mean_squared_error()
{
   message += "test_calculate_mean_squared_error\n";
}


void MatrixTest::test_calculate_root_mean_squared_error()
{
   message += "test_calculate_root_mean_squared_error\n";
}


void MatrixTest::test_calculate_minimum_maximum()
{
   message += "test_calculate_minimum_maximum\n";
}


void MatrixTest::test_calculate_determinant()
{
   message += "test_calculate_determinant\n";

   Matrix<int> m(1, 1, 1);

   assert_true(m.calculate_determinant() == 1, LOG);

   m.set(2, 2);

   m(0,0) = 1;
   m(0,1) = 2;

   m(1,0) = 3;
   m(1,1) = 4;

   assert_true(m.calculate_determinant() == -2, LOG);

   m.set(3, 3);

   m(0,0) = 1;
   m(0,1) = 2;
   m(0,2) = 3;

   m(1,0) = 4;
   m(1,1) = 5;
   m(1,2) = 6;

   m(2,0) = 7;
   m(2,1) = 8;
   m(2,2) = 9;

   assert_true(m.calculate_determinant() == 0, LOG);

   m.set(4, 4);

   m(0,0) = 1;
   m(0,1) = 2;
   m(0,2) = 3;
   m(0,3) = 4;

   m(1,0) = 5;
   m(1,1) = 6;
   m(1,2) = 7;
   m(1,3) = 8;

   m(2,0) = 9;
   m(2,1) = 10;
   m(2,2) = 11;
   m(2,3) = 12;

   m(3,0) = 13;
   m(3,1) = 14;
   m(3,2) = 15;
   m(3,3) = 16;

   assert_true(m.calculate_determinant() == 0, LOG);
}


void MatrixTest::test_calculate_transpose()
{
   message += "test_calculate_transpose\n";

   Matrix<int> m(1, 1, 0);

   Matrix<int> transpose = m.calculate_transpose();

   assert_true(transpose == m, LOG);
}


void MatrixTest::test_calculate_cofactor()
{
   message += "test_calculate_cofactor\n";
}


void MatrixTest::test_calculate_inverse()
{
   message += "test_calculate_inverse\n";

   Matrix<double> m;
   Matrix<double> inverse;

   // Test

   m.set(1, 1, 1.0);

   assert_true(m.calculate_inverse() == 1.0, LOG);

   // Test

   m.set(2, 2);

   m(0,0) = 1.0;
   m(0,1) = 2.0;

   m(1,0) = 3.0;
   m(1,1) = 4.0;

   inverse = m.calculate_inverse();

   assert_true(inverse.get_rows_number() == 2, LOG);
   assert_true(inverse(0,0) == -2.0, LOG);
   assert_true(inverse(0,1) ==  1.0, LOG);
   assert_true(inverse(1,0) ==  3.0/2.0, LOG);
   assert_true(inverse(1,1) == -1.0/2.0, LOG);

   // Test

   m.set(3, 3);

   m(0,0) =  24.0;
   m(0,1) = -12.0;
   m(0,2) =  -2.0;

   m(1,0) =  5.0;
   m(1,1) =  3.0;
   m(1,2) = -5.0;

   m(2,0) = -4.0;
   m(2,1) =  2.0;
   m(2,2) =  4.0;

   inverse = m.calculate_inverse();

   assert_true(inverse.get_rows_number() == 3, LOG);

   m.set(4, 4);

   m(0,0) = 1.0;
   m(0,1) = -2.0;
   m(0,2) = 3.0;
   m(0,3) = -4.0;

   m(1,0) = 5.0;
   m(1,1) = 6.0;
   m(1,2) = 7.0;
   m(1,3) = 8.0;

   m(2,0) = 9.0;
   m(2,1) = 10.0;
   m(2,2) = 11.0;
   m(2,3) = 12.0;

   m(3,0) = -13.0;
   m(3,1) = 14.0;
   m(3,2) = -15.0;
   m(3,3) = 16.0;

   inverse = m.calculate_inverse();

   assert_true(inverse.get_rows_number() == 4, LOG);



}


void MatrixTest::test_is_symmetric()
{
   message += "test_is_symmetric\n";

   Matrix<int> m(1, 1, 1);

   assert_true(m.is_symmetric(), LOG);

   m.set(2, 2);

   m.initialize_identity();

   assert_true(m.is_symmetric(), LOG);
}


void MatrixTest::test_is_antisymmetric()
{
   message += "test_is_antisymmetric\n";

   Matrix<int> m;

   // Test

   m.set(1, 1, 0);

   assert_true(m.is_antisymmetric() == true, LOG);

   // Test

   m.set(2, 2, 1);

   assert_true(m.is_antisymmetric() == false, LOG);

   // Test

   m.set(2, 2, 1);

   m(0,0) = 0;
   m(0,1) = -2;
   m(1,0) = 2;
   m(1,1) = 0;

   assert_true(m.is_antisymmetric() == true, LOG);
}


void MatrixTest::test_calculate_k_means()
{
    Matrix<double> m;

    // Test

//    m.set(3,1);

//    m(0,0) = 1;
//    m(1,0) = 10;
//    m(2,0) = 11;

//    Vector< Vector<size_t> > groups = m.calculate_k_means(2);

//    assert_true(groups[0][0] == 0, LOG);
//    assert_true(groups[1].size() == 2, LOG);
}


void MatrixTest::test_threshold()
{
    Matrix<double> m;
    Matrix<double> l;

    m.set(2,2);
    l.set(2,2);

    m(0,0) = -1;
    m(0,1) = 1;
    m(1,0) = 2;
    m(1,1) = -2;

    l = threshold(m);

    assert_true(fabs(l(0,0) - 0) < 0.001, LOG);
    assert_true(fabs(l(0,1) - 1) < 0.001, LOG);
    assert_true(fabs(l(1,0) - 1) < 0.001, LOG);
    assert_true(fabs(l(1,1) - 0) < 0.001, LOG);
}


void MatrixTest::test_symmetric_threshold()
{
    Matrix<double> m;
    Matrix<double> l;

    m.set(2,2);
    l.set(2,2);

    m(0,0) = -1;
    m(0,1) = 1;
    m(1,0) = 2;
    m(1,1) = -2;

    l = symmetric_threshold(m);

    assert_true(fabs(l(0,0) - -1) < 0.001, LOG);
    assert_true(fabs(l(0,1) - 1) < 0.001, LOG);
    assert_true(fabs(l(1,0) - 1) < 0.001, LOG);
    assert_true(fabs(l(1,1) - -1) < 0.001, LOG);
}


void MatrixTest::test_logistic()
{
    Matrix<double> m;
    Matrix<double> l;

    m.set(2,2);
    l.set(2,2);

    m(0,0) = -1;
    m(0,1) = 1;
    m(1,0) = 2;
    m(1,1) = -2;

    l = logistic(m);

    assert_true(fabs(l(0,0) - 0.268941) < 0.000001, LOG);
    assert_true(fabs(l(0,1) - 0.731059) < 0.000001, LOG);
    assert_true(fabs(l(1,0) - 0.880797) < 0.000001, LOG);
    assert_true(fabs(l(1,1) - 0.119203) < 0.000001, LOG);
}


void MatrixTest::test_hyperbolic_tangent()
{
    Matrix<double> m;
    Matrix<double> l;

    m.set(2,2);
    l.set(2,2);

    m(0,0) = -1;
    m(0,1) = 1;
    m(1,0) = 2;
    m(1,1) = -2;

    l = hyperbolic_tangent(m);

    assert_true(fabs(l(0,0) - -0.761594) < 0.000001, LOG);
    assert_true(fabs(l(0,1) - 0.761594) < 0.000001, LOG);
    assert_true(fabs(l(1,0) - 0.964028) < 0.000001, LOG);
    assert_true(fabs(l(1,1) - -0.964028) < 0.000001, LOG);
}


void MatrixTest::test_hyperbolic_tangent_derivatives()
{
    Matrix<double> m;
    Matrix<double> l;

    m.set(2,2);
    l.set(2,2);

    m(0,0) = -1;
    m(0,1) = 1;
    m(1,0) = 2;
    m(1,1) = -2;

    l = hyperbolic_tangent_derivatives(m);

    assert_true(fabs(l(0,0) - 0.419974) < 0.000001, LOG);
    assert_true(fabs(l(0,1) - 0.419974) < 0.000001, LOG);
    assert_true(fabs(l(1,0) - 0.070651) < 0.000001, LOG);
    assert_true(fabs(l(1,1) - 0.070651) < 0.000001, LOG);
}


void MatrixTest::test_hyperbolic_tangent_second_derivatives()
{
    Matrix<double> m;
    Matrix<double> l;

    m.set(2,2);
    l.set(2,2);

    m(0,0) = -1;
    m(0,1) = 1;
    m(1,0) = 2;
    m(1,1) = -2;

    l = hyperbolic_tangent_second_derivatives(m);

    assert_true(fabs(l(0,0) - 0.639700) < 0.000001, LOG);
    assert_true(fabs(l(0,1) - -0.639700) < 0.000001, LOG);
    assert_true(fabs(l(1,0) - -0.136219) < 0.000001, LOG);
    assert_true(fabs(l(1,1) - 0.136219) < 0.000001, LOG);
}


void MatrixTest::test_logistic_derivatives()
{
    Matrix<double> m;
    Matrix<double> l;

    m.set(2,2);
    l.set(2,2);

    m(0,0) = -1;
    m(0,1) = 1;
    m(1,0) = 2;
    m(1,1) = -2;

    l = logistic_derivatives(m);

    assert_true(fabs(l(0,0) - 0.196612) < 0.000001, LOG);
    assert_true(fabs(l(0,1) - 0.196612) < 0.000001, LOG);
    assert_true(fabs(l(1,0) - 0.104994) < 0.000001, LOG);
    assert_true(fabs(l(1,1) - 0.104994) < 0.000001, LOG);
}


void MatrixTest::test_logistic_second_derivatives()
{
    Matrix<double> m;
    Matrix<double> l;

    m.set(2,2);
    l.set(2,2);

    m(0,0) = -1;
    m(0,1) = 1;
    m(1,0) = 2;
    m(1,1) = -2;

    l = logistic_second_derivatives(m);

    assert_true(fabs(l(0,0) - 0.090858) < 0.000001, LOG);
    assert_true(fabs(l(0,1) - -0.090858) < 0.000001, LOG);
    assert_true(fabs(l(1,0) - -0.079963) < 0.000001, LOG);
    assert_true(fabs(l(1,1) - 0.079963) < 0.000001, LOG);
}


void MatrixTest::test_threshold_derivatives()
{
    Matrix<double> m;
    Matrix<double> l;

    m.set(1,2);
    l.set(1,2);

    m(0,0) = 2;
    m(0,1) = -2;

    l = threshold_derivatives(m);

    assert_true(fabs(l(0,0) - 0) < 0.000001, LOG);
    assert_true(fabs(l(0,1) - 0) < 0.000001, LOG);
}


void MatrixTest::test_threshold_second_derivatives()
{
    Matrix<double> m;
    Matrix<double> l;

    m.set(1,2);
    l.set(1,2);

    m(0,0) = 2;
    m(0,1) = -2;

    l = threshold_derivatives(m);

    assert_true(fabs(l(0,0) - 0) < 0.000001, LOG);
    assert_true(fabs(l(0,1) - 0) < 0.000001, LOG);
}


void MatrixTest::test_symmetric_threshold_derivatives()
{
    Matrix<double> m;
    Matrix<double> l;

    m.set(1,2);
    l.set(1,2);

    m(0,0) = 2;
    m(0,1) = -2;

    l = threshold_derivatives(m);

    assert_true(fabs(l(0,0) - 0) < 0.000001, LOG);
    assert_true(fabs(l(0,1) - 0) < 0.000001, LOG);
}


void MatrixTest::test_symmetric_threshold_second_derivatives()
{
    Matrix<double> m;
    Matrix<double> l;

    m.set(1,2);
    l.set(1,2);

    m(0,0) = 2;
    m(0,1) = -2;

    l = threshold_derivatives(m);

    assert_true(fabs(l(0,0) - 0) < 0.000001, LOG);
    assert_true(fabs(l(0,1) - 0) < 0.000001, LOG);
}


void MatrixTest::test_scale_mean_standard_deviation()
{
   message += "test_scale_mean_standard_deviation\n";

   Matrix<double> m;

   Vector< Statistics<double> > statistics;

   // Test

   m.set(2, 2);
   m.randomize_uniform();

   m.scale_mean_standard_deviation();

   statistics = m.calculate_statistics();

   assert_true(statistics[0].has_mean_zero_standard_deviation_one(), LOG);
   assert_true(statistics[1].has_mean_zero_standard_deviation_one(), LOG);
}


void MatrixTest::test_scale_rows_mean_standard_deviation()
{
   message += "test_scale_rows_mean_standard_deviation\n";
}


void MatrixTest::test_scale_columns_mean_standard_deviation()
{
   message += "test_scale_columns_mean_standard_deviation\n";
}


void MatrixTest::test_scale_rows_columns_mean_standard_deviation()
{
   message += "test_scale_rows_columns_mean_standard_deviation\n";
}


void MatrixTest::test_scale_minimum_maximum()
{
   message += "test_scale_minimum_maximum\n";
}


void MatrixTest::test_scale_rows_minimum_maximum()
{
   message += "test_scale_rows_minimum_maximum\n";
}


void MatrixTest::test_scale_columns_minimum_maximum()
{
   message += "test_scale_columns_minimum_maximum\n";
}


void MatrixTest::test_scale_rows_columns_minimum_maximum()
{
   message += "test_scale_rows_columns_minimum_maximum\n";
}


void MatrixTest::test_unscale_mean_standard_deviation()
{
   message += "test_unscale_mean_standard_deviation\n";
}


void MatrixTest::test_unscale_rows_mean_standard_deviation()
{
   message += "test_unscale_rows_mean_standard_deviation\n";
}


void MatrixTest::test_unscale_columns_mean_standard_deviation()
{
   message += "test_unscale_columns_mean_standard_deviation\n";
}


void MatrixTest::test_unscale_rows_columns_mean_standard_deviation()
{
   message += "test_unscale_rows_columns_mean_standard_deviation\n";
}


void MatrixTest::test_unscale_minimum_maximum()
{
   message += "test_unscale_minimum_maximum\n";
}


void MatrixTest::test_unscale_rows_minimum_maximum()
{
   message += "test_unscale_rows_minimum_maximum\n";
}


void MatrixTest::test_unscale_columns_minimum_maximum()
{
   message += "test_unscale_columns_minimum_maximum\n";
}


void MatrixTest::test_unscale_rows_columns_minimum_maximum()
{
   message += "test_unscale_rows_columns_minimum_maximum\n";
}


void MatrixTest::test_convert_angular_variables_degrees()
{
   message += "test_convert_angular_variables_degrees\n";

   Matrix<double> m;

   // Test

   m.set(1, 1, 90.0);

   m.convert_angular_variables_degrees(0);

   assert_true(m.get_rows_number() == 1, LOG);
   assert_true(m.get_columns_number() == 2, LOG);

   assert_true(fabs(m(0,0) - 1.0) < 1.0e-6, LOG);
   assert_true(fabs(m(0,1) - 0.0) < 1.0e-6, LOG);
}


void MatrixTest::test_convert_angular_variables_radians()
{
   message += "test_convert_angular_variables_radians\n";

   Matrix<double> m;

   // Test

   m.set(1, 1, 3.1415927/2.0);

   m.convert_angular_variables_radians(0);

   assert_true(m.get_rows_number() == 1, LOG);
   assert_true(m.get_columns_number() == 2, LOG);

   assert_true(fabs(m(0,0) - 1.0) < 1.0e-3, LOG);
   assert_true(fabs(m(0,1) - 0.0) < 1.0e-3, LOG);
}



void MatrixTest::test_to_time_t()
{
    message += "test_to_time_t\n";

    time_t timestamp;

    // Test

    timestamp = Matrix<int>::to_time_t(1,8,2018);

    assert_true(timestamp == 1533081600, LOG);

    cout << timestamp << endl;

    // Test

    timestamp = Matrix<int>::to_time_t(1,1,1970);

    assert_true(timestamp == 0, LOG);

    cout << timestamp << endl;
}


void MatrixTest::test_print()
{
   message += "test_print\n";

   Matrix<size_t> m(6, 1, true);
   //m.print();
}


void MatrixTest::test_save()
{
   message += "test_save\n";

   string file_name = "../data/matrix.dat";

   Matrix<int> m;

   m.save(file_name);
}


void MatrixTest::test_load()
{
   message += "test_load\n";

   string file_name = "../data/matrix.dat";

   Matrix<int> m;

   // Test

   m.set();

   m.save(file_name);
   m.load(file_name);

   assert_true(m.get_rows_number() == 0, LOG);
   assert_true(m.get_columns_number() == 0, LOG);

   // Test

   m.set(1, 2, 3);

   m.save(file_name);
   m.load(file_name);

   assert_true(m.get_rows_number() == 1, LOG);
   assert_true(m.get_columns_number() == 2, LOG);
   assert_true(m == 3, LOG);   

   // Test

   m.set(2, 1, 1);

   m.save(file_name);
   m.load(file_name);

   assert_true(m.get_rows_number() == 2, LOG);
   assert_true(m.get_columns_number() == 1, LOG);

   // Test

   m.set(4, 4, 0);

   m.save(file_name);
   m.load(file_name);

   assert_true(m.get_rows_number() == 4, LOG);
   assert_true(m.get_columns_number() == 4, LOG);
   assert_true(m == 0, LOG);

   // Test

   m.set(1, 1, -99);

   m.save(file_name);
   m.load(file_name);

   assert_true(m.get_rows_number() == 1, LOG);
   assert_true(m.get_columns_number() == 1, LOG);
   assert_true(m == -99, LOG);

   // Test

   m.set(3, 2);

   m(0,0) = 3; m(0,1) = 5;
   m(1,0) = 7; m(1,1) = 9;
   m(2,0) = 2; m(2,1) = 4;

   m.save(file_name);
   m.load(file_name);

   assert_true(m(0,0) == 3, LOG); assert_true(m(0,1) == 5, LOG);
   assert_true(m(1,0) == 7, LOG); assert_true(m(1,1) == 9, LOG);
   assert_true(m(2,0) == 2, LOG); assert_true(m(2,1) == 4, LOG);
}


void MatrixTest::test_parse()
{
    message += "test_parse\n";

    Matrix<int> m;
    string str;

    // Test

    str = "";

    m.parse(str);

    assert_true(m.get_rows_number() == 0, LOG);
    assert_true(m.get_columns_number() == 0, LOG);

    // Test

    str =
    "1 2 3\n"
    "4 5 6\n";

    m.parse(str);

    assert_true(m.get_rows_number() == 2, LOG);
    assert_true(m.get_columns_number() == 3, LOG);

    // Test

    str =
    "1 2\n"
    "3 4\n"
    "5 6\n";

    m.parse(str);

    assert_true(m.get_rows_number() == 3, LOG);
    assert_true(m.get_columns_number() == 2, LOG);
}


void MatrixTest::run_test_case()
{
   message += "Running matrix test case...\n";  

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();   

   // Reference operator methods

   test_reference_operator();   

   // Arithmetic operators

   test_sum_operator();
   test_rest_operator();
   test_multiplication_operator();
   test_division_operator();

   // Arithmetic and assignment operators

   test_sum_assignment_operator();
   test_rest_assignment_operator();
   test_multiplication_assignment_operator();
   test_division_assignment_operator();

   // Equality and relational operators

   test_equal_to_operator();
   test_not_equal_to_operator();
   test_greater_than_operator();
   test_less_than_operator();
   test_greater_than_or_equal_to_operator();
   test_less_than_or_equal_to_operator();

   // Output operators

   test_output_operator();

   // Get methods

   test_get_rows_number();
   test_get_columns_number();  

   test_get_row();
   test_get_column();

   test_get_submatrix();

   // Set methods

   test_set();
   
   test_set_rows_number();
   test_set_columns_number();

   test_set_row();
   test_set_column();

   // Diagonal methods

   test_get_diagonal();
   test_set_diagonal();
   test_sum_diagonal();

   // Resize methods

   test_append_row();
   test_append_column();

   test_insert_row();
   test_insert_column();

   test_subtract_row();
   test_subtract_column();

   test_sort_less_rows();
   test_sort_greater_rows();

   // Initialization methods

   test_initialize();
   test_randomize_uniform();
   test_randomize_normal();

   test_set_to_identity();

   // Mathematical methods

   test_calculate_sum();
   test_calculate_rows_sum();

   test_dot_vector();
   test_dot_matrix();

   test_calculate_eigenvalues();
   test_calculate_eigenvectors();

   test_direct();

   test_calculate_minimum_maximum();
   test_calculate_mean_standard_deviation();

   test_calculate_statistics();

   test_calculate_histogram();

   test_calculate_covariance_matrix();

   test_calculate_minimal_indices();
   test_calculate_maximal_indices();
   
   test_calculate_minimal_maximal_indices();

   test_calculate_sum_squared_error();
   test_calculate_mean_squared_error();
   test_calculate_root_mean_squared_error();

   test_calculate_determinant();
   test_calculate_transpose();
   test_calculate_cofactor();
   test_calculate_inverse();

   test_is_symmetric();
   test_is_antisymmetric();

   test_calculate_k_means();

   test_threshold();
   test_symmetric_threshold();
   test_logistic();
   test_hyperbolic_tangent();

   test_hyperbolic_tangent_derivatives();
   test_hyperbolic_tangent_second_derivatives();
   test_logistic_derivatives();
   test_logistic_second_derivatives();
   test_threshold_derivatives();
   test_threshold_second_derivatives();
   test_symmetric_threshold_derivatives();
   test_symmetric_threshold_second_derivatives();

   // Scaling methods
 
   test_scale_mean_standard_deviation();
   test_scale_rows_mean_standard_deviation();
   test_scale_columns_mean_standard_deviation();
   test_scale_rows_columns_mean_standard_deviation();

   test_scale_minimum_maximum();
   test_scale_rows_minimum_maximum();
   test_scale_columns_minimum_maximum();
   test_scale_rows_columns_minimum_maximum();

   // Unscaling methods

   test_unscale_mean_standard_deviation();
   test_unscale_rows_mean_standard_deviation();
   test_unscale_columns_mean_standard_deviation();
   test_unscale_rows_columns_mean_standard_deviation();

   test_unscale_minimum_maximum();
   test_unscale_rows_minimum_maximum();
   test_unscale_columns_minimum_maximum();
   test_unscale_rows_columns_minimum_maximum();

   test_convert_angular_variables_degrees();
   test_convert_angular_variables_radians();

   test_to_time_t();

   // Serialization methods

   test_print();

   test_load();

   test_save();

   test_parse();

   message += "End of matrix test case.\n";
}


Vector<double> MatrixTest::dot(const Matrix<double>& matrix, const Vector<double>& vector)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    Vector<double> product(rows_number);

    for(size_t i = 0; i < rows_number; i++)
    {
        product[i] = 0;

       for(size_t j = 0; j < columns_number; j++)
       {
          product[i] += vector[j]*matrix(i,j);
       }
    }

    return(product);
}


Matrix<double> MatrixTest::dot(const Matrix<double>& matrix, const Matrix<double>& other_matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    const size_t other_columns_number = other_matrix.get_columns_number();

    Matrix<double> product(rows_number, other_columns_number);

    for(size_t i = 0; i < rows_number; i++) {
        for(size_t j = 0; j < other_columns_number; j++) {
            for(size_t k = 0; k < columns_number; k++) {
                product(i,j) += matrix(i,k)*other_matrix(k,j);
            }
        }
    }

    return(product);
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2018 Artificial Intelligence Techniques, SL.
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
