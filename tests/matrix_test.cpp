//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   M A T R I X   T E S T   C L A S S                                     
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "matrix_test.h"


MatrixTest::MatrixTest() : UnitTesting()
{   
}


MatrixTest::~MatrixTest()
{
}


void MatrixTest::test_constructor()
{
   cout << "test_constructor\n";

   string file_name = "../data/matrix.csv";

   // Default

   Matrix<size_t> matrix1;

   assert_true(matrix1.get_rows_number() == 0, LOG);
   assert_true(matrix1.get_columns_number() == 0, LOG);

   // Rows and columns numbers

   Matrix<size_t> matrix2(0, 0);

   assert_true(matrix2.get_rows_number() == 0, LOG);
   assert_true(matrix2.get_columns_number() == 0, LOG);
  
   Matrix<double> matrix3(1, 1, 1.0);
   assert_true(matrix3.get_rows_number() == 1, LOG);
   assert_true(matrix3.get_columns_number() == 1, LOG);

   // Rows and columns numbers and initialization

   Matrix<size_t> matrix4(0, 0, 1);

   assert_true(matrix4.get_rows_number() == 0, LOG);
   assert_true(matrix4.get_columns_number() == 0, LOG);

   Matrix<size_t> matrix5(1, 1, 1);

   assert_true(matrix5.get_rows_number() == 1, LOG);
   assert_true(matrix5.get_columns_number() == 1, LOG);
   assert_true(matrix5 == true, LOG);

   // File constructor

   matrix1.save_csv(file_name);

   Matrix<size_t> matrix6(file_name,',',false);
   assert_true(matrix6.get_rows_number() == 0, LOG);
   assert_true(matrix6.get_columns_number() == 0, LOG);

   matrix2.save_csv(file_name);

   Matrix<size_t> matrix7(file_name,',',false);
   assert_true(matrix7.get_rows_number() == 0, LOG);
   assert_true(matrix7.get_columns_number() == 0, LOG);

   matrix3.save_csv(file_name);

   Matrix<double> matrix8(file_name,',',false);
   assert_true(matrix8.get_rows_number() == 1, LOG);
   assert_true(matrix8.get_columns_number() == 1, LOG);

   matrix4.save_csv(file_name);
   Matrix<size_t> matrix9(file_name,',',false);
   assert_true(matrix9.get_rows_number() == 0, LOG);
   assert_true(matrix9.get_columns_number() == 0, LOG);

   matrix5.save_csv(file_name);

   Matrix<double> matrix10(file_name,',',false);
   assert_true(matrix10.get_rows_number() == 1, LOG);
   assert_true(matrix10.get_columns_number() == 1, LOG);
   assert_true(matrix10 == true, LOG);

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

   Matrix<size_t> matrix11(2, 2, 0);
   matrix11(0,0)++;
   matrix11(1,1)++;

   assert_true(matrix11(0,0) == 1, LOG);
   assert_true(matrix11(0,1) == 0, LOG);
   assert_true(matrix11(1,0) == 0, LOG);
   assert_true(matrix11(1,1) == 1, LOG);
}


void MatrixTest::test_destructor()
{  
   cout << "test_destructor\n";
}


void MatrixTest::test_assignment_operator()
{
   cout << "test_assignment_operator\n";

   Matrix<int> matrix1(1, 1, 0);

   Matrix<int> matrix2 = matrix1;

   for(size_t i = 0; i < 2; i++)
   {
      matrix2 = matrix1;
   }

   assert_true(matrix2.get_rows_number() == 1, LOG);
   assert_true(matrix2.get_columns_number() == 1, LOG);
   assert_true(matrix2 == 0, LOG);
}


void MatrixTest::test_sum_operator()
{
   cout << "test_sum_operator\n";

   Matrix<int> matrix1(1, 1, 1);
   Matrix<int> matrix2(1, 1, 1);
   Matrix<int> matrix3(1, 1);

   // Test
   
   matrix3 = matrix1 + 1;

   assert_true(matrix3.get_rows_number() == 1, LOG);
   assert_true(matrix3.get_columns_number() == 1, LOG);
   assert_true(matrix3 == 2, LOG);

   // Test

   matrix3 = matrix1 + matrix2;

   assert_true(matrix3.get_rows_number() == 1, LOG);
   assert_true(matrix3.get_columns_number() == 1, LOG);
   assert_true(matrix3 == 2, LOG);
}


void MatrixTest::test_rest_operator()
{
   cout << "test_rest_operator\n";

   Matrix<int> matrix1(1, 1, 1);
   Matrix<int> matrix2(1, 1, 1);
   Matrix<int> matrix3(1, 1);
   Matrix<int> matrix4;

   // Test

   matrix3 = matrix1 - 1;

   assert_true(matrix3.get_rows_number() == 1, LOG);
   assert_true(matrix3.get_columns_number() == 1, LOG);
   assert_true(matrix3 == 0, LOG);

   // Test

   matrix3 = matrix1 - matrix2;

   assert_true(matrix3.get_rows_number() == 1, LOG);
   assert_true(matrix3.get_columns_number() == 1, LOG);
   assert_true(matrix3 == 0, LOG);

   // Test

   matrix1.set(3, 3, 1);
   matrix2.set(3, 3, 1);
   matrix3.set(3, 3, 1);

   matrix4 = matrix1 + matrix2 - matrix3;

   assert_true(matrix4.get_rows_number() == 3, LOG);
   assert_true(matrix4.get_columns_number() == 3, LOG);
   assert_true(matrix4 == 1, LOG);
}


void MatrixTest::test_multiplication_operator()
{
   cout << "test_multiplication_operator\n";

   Matrix<int> matrix1;
   Matrix<int> matrix2;
   Matrix<int> matrix3;
   
   Vector<int> vector;

   // Scalar

   matrix1.set(1, 1, 2);

   matrix3 = matrix1*2;

   assert_true(matrix3.get_rows_number() == 1, LOG);
   assert_true(matrix3.get_columns_number() == 1, LOG);
   assert_true(matrix3 == 4, LOG);

   // Vector

   matrix1.set(1, 1, 1);
   vector.set(1, 1);
  
   // Matrix

   matrix1.set(1, 1, 2);
   matrix2.set(1, 1, 2);

   matrix3 = matrix1*matrix2;

   assert_true(matrix3.get_rows_number() == 1, LOG);
   assert_true(matrix3.get_columns_number() == 1, LOG);
   assert_true(matrix3 == 4, LOG);
}


void MatrixTest::test_division_operator()
{
   cout << "test_division_operator\n";

   Matrix<int> matrix1(1, 1, 2);
   Matrix<int> matrix2(1, 1, 2);
   Matrix<int> matrix3(1, 1);
   
   matrix3 = matrix1/2;

   assert_true(matrix3.get_rows_number() == 1, LOG);
   assert_true(matrix3.get_columns_number() == 1, LOG);
   assert_true(matrix3 == 1, LOG);

   matrix3 = matrix1/matrix2;

   assert_true(matrix3.get_rows_number() == 1, LOG);
   assert_true(matrix3.get_columns_number() == 1, LOG);
   assert_true(matrix3 == 1, LOG);
}


void MatrixTest::test_equal_to_operator()
{
    cout << "test_equal_to_operator\n";

   Matrix<int> matrix1(1,1,0);
   Matrix<int> matrix2(1,1,0);
   Matrix<int> matrix3(1,1,1);

   assert_true(matrix1 == matrix2, LOG);
   assert_false(matrix1 == matrix3, LOG);
}


void MatrixTest::test_not_equal_to_operator()
{
   cout << "test_not_equal_to_operator\n";

   Matrix<int> matrix1(1,1,0);
   Matrix<int> matrix2(1,1,0);
   Matrix<int> matrix3(1,1,1);

   assert_false(matrix1 != matrix2, LOG);
   assert_true(matrix1 != matrix3, LOG);
}


void MatrixTest::test_greater_than_operator()
{
   cout << "test_greater_than_operator\n";

   Matrix<double> matrix1(1,1,1.0);
   Matrix<double> matrix2(1,1,0.0);

   assert_true(matrix1 > 0.0, LOG);
   assert_true(matrix1 > matrix2, LOG);
}


void MatrixTest::test_less_than_operator()
{
   cout << "test_less_than_operator\n";

   Matrix<double> matrix1(1,1,0.0);
   Matrix<double> matrix2(1,1,1.0);

   assert_true(matrix1 < 1.0, LOG);
   assert_true(matrix1 < matrix2, LOG);
}


void MatrixTest::test_greater_than_or_equal_to_operator()
{
   cout << "test_greater_than_or_equal_to_operator\n";

   Matrix<double> matrix1(1,1,1.0);
   Matrix<double> matrix2(1,1,1.0);

   assert_true(matrix1 >= 1.0, LOG);
   assert_true(matrix1 >= matrix2, LOG);
}


void MatrixTest::test_less_than_or_equal_to_operator()
{
   cout << "test_less_than_or_equal_to_operator\n";

   Matrix<double> matrix1(1,1,1.0);
   Matrix<double> matrix2(1,1,1.0);

   assert_true(matrix1 <= 1.0, LOG);
   assert_true(matrix1 <= matrix2, LOG);
}


void MatrixTest::test_output_operator()
{
   cout << "test_output_operator\n";

   Matrix<double> matrix1;
   Matrix<Vector<double>> matrix2;
   Matrix< Matrix<size_t> > matrix3;

   // Test

   matrix1.set(2, 3, 0.0);

   // Test

   matrix2.set(2, 2);
   matrix2(0,0).set(1, 0.0);
   matrix2(0,1).set(1, 1.0);
   matrix2(1,0).set(1, 0.0);
   matrix2(1,1).set(1, 1.0);

   // Test

   matrix3.set(2, 2);
   matrix3(0,0).set(1, 1, 0);
   matrix3(0,1).set(1, 1, 1);
   matrix3(1,0).set(1, 1, 0);
   matrix3(1,1).set(1, 1, 1);
}


void MatrixTest::test_get_rows_number()
{
   cout << "test_get_rows_number\n";

   Matrix<size_t> m(2,3);

   size_t rows_number = m.get_rows_number();

   assert_true(rows_number == 2, LOG);
}


void MatrixTest::test_get_columns_number()
{
   cout << "test_get_columns_number\n";

   Matrix<size_t> matrix(2,3);

   size_t columns_number = matrix.get_columns_number();

   assert_true(columns_number == 3, LOG);
}


void MatrixTest::test_get_row()
{
   cout << "test_get_row\n";

   Matrix<int> matrix(1, 1, 0);

   Vector<int> row = matrix.get_row(0);

   assert_true(row == 0, LOG);
}


void MatrixTest::test_get_rows()
{
    cout << "test_get_rows\n";

    Matrix<double> matrix(4,4);
    Vector<double> check_matrix;
    Vector<double> solution({1, 2, 3, 4, 5, 6, 7, 8});

    matrix.set_row(0, {1, 2, 3, 4});
    matrix.set_row(1, {5, 6, 7, 8});
    matrix.set_row(2, {9, 10, 11, 12});
    matrix.set_row(3, {13, 14, 15, 16});

    check_matrix = matrix.get_rows(1,2);

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_get_column()
{
   cout << "test_get_column\n";

   //One specific column demanded

   Matrix<size_t> matrix(2, 2);

   matrix.set_column(0, {1,5});
   matrix.set_column(1, {4,7});

   Vector<size_t> column = matrix.get_column(0);
   Vector<size_t> solution({1,5});

   assert_true(column == solution, LOG);

   // Column and row indices demanded

   Vector<size_t> column1;
   Vector<size_t> solution1({5});

   column1 = matrix.get_column(0, {1});

   assert_true(column1 == solution1, LOG);

   //Header demanded

   Vector<size_t> check_matrix;

   matrix.set_header({"col1", "col2"});

   check_matrix = matrix.get_column({"col1"});

   assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_get_columns()
{
    cout << "test_get_columns\n";

    Matrix<double> matrix(4,4);
    Matrix<double> check_matrix;
    Matrix<double> solution(4,2);

    matrix.set_row(0, {1, 2, 3, 4});
    matrix.set_row(1, {5, 6, 7, 8});
    matrix.set_row(2, {9, 10, 11, 12});
    matrix.set_row(3, {13, 14, 15, 16});

    solution.set_column(0, {1, 5, 9, 13});
    solution.set_column(1, {2, 6, 10, 14});

    solution.set_header({"col1", "col2"});

    matrix.set_header({"col1", "col2", "col3", "col4"});

    check_matrix = matrix.get_columns({"col1", "col2"});

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_get_header()
{
   cout << "test_get_header\n";

   // All headers

   Matrix<int> matrix(1, 1, 0);

   Vector<string> header("number");

   matrix.set_header(header);

   const Vector<string> solution("number");

   Vector<string> check_matrix  = matrix.get_header();

   assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_get_column_index()
{
    cout << "test_get_column_index\n";

    Matrix<size_t> matrix1(2, 2);

    matrix1.set_column(0, {4, 5});
    matrix1.set_column(1, {4, 6});

    matrix1.set_header({"number1","number2"});

    size_t solution = 1;
    size_t check_matrix;

    check_matrix = matrix1.get_column_index("number2");

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_get_columns_indices()
{
    cout << "test_get_columns_indices\n";

      Matrix<size_t> matrix1(2,2);

      matrix1.set_column(0, {4, 5});
      matrix1.set_column(1, {4, 6});

      matrix1.set_header({"number1", "number2"});

      Vector<size_t> solution({0, 1});
      Vector<size_t> check_matrix;

      Vector<string> header({"number1", "number2"});

      check_matrix = matrix1.get_columns_indices(header);

      assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_get_binary_columns_indices()
{
    cout << "test_get_binary_columns_indices\n";

    //Normal case

    Matrix<size_t> matrix1(3,3);
    Vector<size_t> solution({1, 2});

    matrix1.set_column(0, {1, 2, 3});
    matrix1.set_column(1, {1, 0, 1});
    matrix1.set_column(2, {0, 0, 1});

    assert_true(matrix1.get_binary_columns_indices() == solution, LOG);

    //Non binary columns

    Matrix<size_t> matrix(3,3);
    Vector<size_t> solution2({});

    matrix.set_column(0, {1, 2, 3});
    matrix.set_column(1, {1, 3, 1});
    matrix.set_column(2, {0, 6, 1});

    assert_true(matrix1.get_binary_columns_indices() == solution, LOG);
}


void MatrixTest::test_get_submatrix()
{
   cout << "test_get_submatrix\n";

   Matrix<size_t> matrix(4,4);
   Matrix<size_t> sub_matrix(4,4);
   Matrix<size_t> solution(2,4);

   matrix.set_row(0, {1, 2, 3, 4});
   matrix.set_row(1, {5, 6, 7, 8});
   matrix.set_row(2, {9, 10, 11, 12});
   matrix.set_row(3, {13, 14, 15, 16});

   solution.set_row(0, {5, 6, 7, 8});
   solution.set_row(1, {9, 10, 11, 12});

   sub_matrix = matrix.get_submatrix({1,2}, {0, 1, 2, 3});

   assert_true(sub_matrix == solution, LOG);
}


void MatrixTest::test_get_submatrix_rows()
{
   cout << "test_get_submatrix_rows\n";

   Matrix<size_t> matrix(4,4);
   Matrix<size_t> sub_matrix(4,4);
   Matrix<size_t> solution(2,4);

   matrix.set_row(0, {1, 2, 3, 4});
   matrix.set_row(1, {5, 6, 7, 8});
   matrix.set_row(2, {9, 10, 11, 12});
   matrix.set_row(3, {13, 14, 15, 16});

   solution.set_row(0, {5, 6, 7, 8});
   solution.set_row(1, {9, 10, 11, 12});

   sub_matrix = matrix.get_submatrix_rows({1,2});

   assert_true(sub_matrix == solution, LOG);
}


void MatrixTest::test_get_submatrix_columns()
{
   cout << "test_get_submatrix_columns\n";

   Matrix<size_t> matrix(4,4);
   Matrix<size_t> sub_matrix(4,4);
   Matrix<size_t> solution(4,2);

   matrix.set_row(0, {1, 2, 3, 4});
   matrix.set_row(1, {5, 6, 7, 8});
   matrix.set_row(2, {9, 10, 11, 12});
   matrix.set_row(3, {13, 14, 15, 16});

   solution.set_column(0, {1, 5, 9, 13});
   solution.set_column(1, {2, 6, 10, 14});

   sub_matrix = matrix.get_submatrix_columns({0, 1});

   assert_true(sub_matrix == solution, LOG);
}


void MatrixTest::test_get_first()
{
    cout << "test_get_first\n";

    //Column number case

    Matrix<size_t> matrix (3, 3);

    const size_t solution = 4;
    size_t check_matrix;

    matrix.set_row(0, {5, 4, 3});
    matrix.set_row(1, {5, 5, 6});
    matrix.set_row(2, {5, 1, 9});

    check_matrix = matrix.get_first(1);

    assert_true(check_matrix == solution, LOG);

    //Header case

    matrix.set_header({"col1", "col2", "col3"});

    const size_t solution1 = 3;
    size_t check_matrix1;

    check_matrix1 = matrix.get_first("col3");

    assert_true(check_matrix1 == solution1, LOG);
}


void MatrixTest::test_get_last()
{
    cout << "test_get_last\n";

    //Column number case

    Matrix<size_t> matrix (3, 3);

    const size_t solution = 1;
    size_t check_matrix;

    matrix.set_row(0, {5, 4, 3});
    matrix.set_row(1, {5, 5, 6});
    matrix.set_row(2, {5, 1, 9});

    check_matrix = matrix.get_last(1);

    assert_true(check_matrix == solution, LOG);

    //Header case

    matrix.set_header({"col1", "col2", "col3"});

    const size_t solution1 = 9;
    size_t check_matrix1;

    check_matrix1 = matrix.get_last("col3");

    assert_true(check_matrix1 == solution1, LOG);
}


void MatrixTest::test_get_constant_columns_indices()
{
    cout << "test_get_constant_columns_indices\n";

    Matrix<size_t> matrix (3, 3);

    const Vector<size_t> solution({0});
    Vector<size_t> check_matrix;

    matrix.set_row(0, {5, 4, 3});
    matrix.set_row(1, {5, 5, 6});
    matrix.set_row(2, {5, 1, 9});

    check_matrix = matrix.get_constant_columns_indices();

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_get_first_rows()
{
    cout << "test_get_first_rows\n";

    Matrix<size_t> matrix(3, 3);

    Matrix<size_t> solution(2,3);
    Matrix<size_t> check_matrix;

    matrix.set_row(0, {5, 4, 3});
    matrix.set_row(1, {5, 5, 6});
    matrix.set_row(2, {5, 1, 9});

    solution.set_row(0, {5, 4, 3});
    solution.set_row(1, {5, 5, 6});

    check_matrix = solution.get_first_rows(2);

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_get_last_rows()
{
    cout << "test_get_last_rows\n";

    Matrix<size_t> matrix (3, 3);

    Matrix<size_t> solution(2,3);
    Matrix<size_t> check_matrix;

    matrix.set_row(0, {5, 4, 3});
    matrix.set_row(1, {5, 5, 6});
    matrix.set_row(2, {5, 1, 9});

    solution.set_row(1, {5, 1, 9});
    solution.set_row(0, {5, 5, 6});

    check_matrix = solution.get_last_rows(2);

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_get_first_columns()
{
    cout << "test_get_first_columns\n";

    Matrix<size_t> matrix (3, 3);

    Matrix<size_t> solution(3,2);
    Matrix<size_t> check_matrix;

    matrix.set_row(0, {5, 4, 3});
    matrix.set_row(1, {5, 5, 6});
    matrix.set_row(2, {5, 1, 9});

    solution.set_column(0, {5, 5, 5});
    solution.set_column(1, {4, 5, 1});

    check_matrix = matrix.get_first_columns(2);

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_get_last_columns()
{
    cout << "test_get_last_columns\n";

    Matrix<size_t> matrix (3, 3);

    Matrix<size_t> solution(3,2);
    Matrix<size_t> check_matrix;

    matrix.set_row(0, {5, 4, 3});
    matrix.set_row(1, {5, 5, 6});
    matrix.set_row(2, {5, 1, 9});

    solution.set_column(1, {3, 6, 9});
    solution.set_column(0, {4, 5, 1});

    check_matrix = matrix.get_last_columns(2);

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_set()
{
   cout << "test_set\n";

   string file_name = "../data/matrix.dat";

   Matrix<double> matrix;

   // Default

   matrix.set();

   assert_true(matrix.get_rows_number() == 0, LOG);
   assert_true(matrix.get_columns_number() == 0, LOG);

   // Numbers of rows and columns

   matrix.set(0, 0);

   assert_true(matrix.get_rows_number() == 0, LOG);
   assert_true(matrix.get_columns_number() == 0, LOG);

   matrix.set(2, 3);

   assert_true(matrix.get_rows_number() == 2, LOG);
   assert_true(matrix.get_columns_number() == 3, LOG);

   matrix.set(0, 0);

   assert_true(matrix.get_rows_number() == 0, LOG);
   assert_true(matrix.get_columns_number() == 0, LOG);

   // Initialization 

   matrix.set(3, 2, 1.0);

   assert_true(matrix.get_rows_number() == 3, LOG);
   assert_true(matrix.get_columns_number() == 2, LOG);
   assert_true(matrix == 1.0, LOG);

   // File 

   matrix.save_csv(file_name);
   matrix.set(file_name);

   assert_true(matrix.get_rows_number() == 3, LOG);
   assert_true(matrix.get_columns_number() == 2, LOG);
   assert_true(matrix == 1.0, LOG);
}


void MatrixTest::test_set_identity()
{
    cout << "test_set_identity\n";

    Matrix<size_t> matrix (3, 3);

    Matrix<size_t> solution(2,2);
    Matrix<size_t> check_matrix(2,2);

    matrix.set_row(0, {5, 4, 3});
    matrix.set_row(1, {5, 5, 6});
    matrix.set_row(2, {5, 1, 9});

    solution.set_column(0, {1, 0});
    solution.set_column(1, {0, 1});

    matrix.set_identity(2);

    assert_true(matrix == solution, LOG);
}


void MatrixTest::test_set_rows_number()
{
   cout << "test_set_rows_number\n";

   Matrix<size_t> matrix (3, 3);

   const size_t solution = 4;

   matrix.set_rows_number(4);

   assert_true(matrix.get_rows_number() == solution, LOG);
}


void MatrixTest::test_set_columns_number()
{
   cout << "test_set_columns_number\n";

   Matrix<size_t> matrix (3, 3);

   const size_t solution = 4;

   matrix.set_columns_number(4);

   assert_true(matrix.get_columns_number() == solution, LOG);
}


void MatrixTest::test_set_header()
{
    cout << "test_set_header\n";

    //Normal case

    Matrix<size_t> matrix (3, 3);

    Vector<string> solution({"col1", "col2", "col3"});
    Vector<string> solution1({"col5", "col2", "col3"});
    Vector<string> check_matrix;

    matrix.set_row(0, {5, 4, 3});
    matrix.set_row(1, {5, 5, 6});
    matrix.set_row(2, {5, 1, 9});

    matrix.set_header({"col1", "col2", "col3"});

    assert_true(matrix.get_header() == solution, LOG);

    //Replace case

    matrix.set_header(0, "col5");

    assert_true(matrix.get_header() == solution1, LOG);
}


void MatrixTest::test_set_row()
{
   cout << "test_set_row\n";

   Matrix<double> matrix(1,1);

   Vector<double> row(1, 1.0);

   matrix.set_row(0, row);

   assert_true(matrix.get_row(0) == row, LOG);
}


void MatrixTest::test_set_column()
{
   cout << "test_set_column\n";

   Matrix<double> matrix(1,1);

   Vector<double> column(1, 1.0);

   matrix.set_column(0, column);

   assert_true(matrix.get_column(0) == column, LOG);
}


void MatrixTest::test_set_submatrix_rows()
{
    cout << "test_set_submatrix_rows\n";

    Matrix<size_t> matrix (3, 3);
    Matrix<size_t> solution (3, 3);
    Matrix<size_t> replace(1,3);


    matrix.set_row(0, {5, 4, 3});
    matrix.set_row(1, {5, 5, 6});
    matrix.set_row(2, {5, 1, 9});

    solution.set_row(0, {5, 4, 3});
    solution.set_row(1, {5, 5, 6});
    solution.set_row(2, {6, 7, 8});

    replace.set_row(0, {6, 7, 8});

    matrix.set_submatrix_rows(2, replace);

    assert_true(matrix == solution, LOG);
}


void MatrixTest::test_empty()
{
    cout << "test_empty\n";

    Matrix<size_t> matrix (3, 3);
    Matrix<size_t> matrix1 (0,0);

    assert_true(matrix1.empty(), LOG);
    assert_true(!matrix.empty(), LOG);
}


void MatrixTest::test_is_square()
{
    cout << "test_is_square\n";

    Matrix<size_t> matrix (3, 4);
    Matrix<size_t> matrix1 (3,3);

    assert_true(matrix1.is_square(), LOG);
    assert_true(!matrix.is_square(), LOG);
}


void MatrixTest::test_is_symmetric()
{
    cout << "test_is_symmetric\n";

     Matrix<size_t> matrix (3,3);

     matrix.set_row(0, {5, 3, 1});
     matrix.set_row(1, {3, 5, 1});
     matrix.set_row(2, {1, 1, 9});

     assert_true(matrix.is_symmetric(), LOG);
}


void MatrixTest::test_is_diagonal()
{
    cout << "test_is_diagonal\n";

    Matrix<size_t> matrix (3,3);

    matrix.set_row(0, {5, 3, 1});
    matrix.set_row(1, {3, 5, 1});
    matrix.set_row(2, {1, 1, 9});

    Matrix<size_t> matrix1 (3,3);

    matrix1.set_row(0, {5, 0, 0});
    matrix1.set_row(1, {0, 5, 0});
    matrix1.set_row(2, {0, 0, 5});

    assert_true(!matrix.is_diagonal(), LOG);
    assert_true(matrix1.is_diagonal(), LOG);
}


void MatrixTest::test_is_scalar()
{
    cout << "test_is_scalar\n";

    Matrix<double> matrix (3,3);

    matrix.set_row(0, {5, 3, 1});
    matrix.set_row(1, {3, 5, 1});
    matrix.set_row(2, {1, 1, 9});

    Matrix<double> matrix1 (3,3);

    matrix1.set_row(0, {5, 0, 0});
    matrix1.set_row(1, {0, 5, 0});
    matrix1.set_row(2, {0, 0, 5});

    assert_true(!matrix.is_scalar(), LOG);
    assert_true(matrix1.is_scalar(), LOG);
}


void MatrixTest::test_is_identity()
{
    cout << "test_is_identity\n";

    Matrix<double> matrix (3,3);

    matrix.set_row(0, {5, 3, 1});
    matrix.set_row(1, {3, 5, 1});
    matrix.set_row(2, {1, 1, 9});

    Matrix<double> matrix1 (3,3);

    matrix1.set_row(0, {1, 0, 0});
    matrix1.set_row(1, {0, 1, 0});
    matrix1.set_row(2, {0, 0, 1});

    assert_true(!matrix.is_identity(), LOG);
    assert_true(matrix1.is_identity(), LOG);
}


void MatrixTest::test_is_binary()
{
    cout << "test_is_binary\n";

    Matrix<double> matrix (3,3);

    matrix.set_row(0, {5, 3, 1});
    matrix.set_row(1, {3, 5, 1});
    matrix.set_row(2, {1, 1, 9});

    Matrix<double> matrix1 (3,3);

    matrix1.set_row(0, {1, 0, 0});
    matrix1.set_row(1, {0, 1, 0});
    matrix1.set_row(2, {1, 0, 1});

    assert_true(!matrix.is_binary(), LOG);
    assert_true(matrix1.is_binary(), LOG);
}


void MatrixTest::test_is_column_binary()
{
    cout << "test_is_column_binary\n";

    Matrix<double> matrix1 (3,3);

    matrix1.set_row(0, {1, 0, 0});
    matrix1.set_row(1, {0, 1, 0});
    matrix1.set_row(2, {1, 5, 1});

    assert_true(matrix1.is_column_binary(0), LOG);
    assert_true(!matrix1.is_column_binary(1), LOG);
}


void MatrixTest::test_is_column_constant()
{
    cout << "test_is_column_constant\n";

    Matrix<double> matrix1 (3,3);

    matrix1.set_row(0, {1, 0, 0});
    matrix1.set_row(1, {0, 1, 0});
    matrix1.set_row(2, {1, 5, 0});

    assert_true(matrix1.is_column_binary(2), LOG);
    assert_true(!matrix1.is_column_binary(1), LOG);
}


void MatrixTest::test_is_positive()
{
    cout << "test_is_positive\n";

    Matrix<double> matrix (3,3);

    matrix.set_row(0, {5, 3, 1});
    matrix.set_row(1, {3, 5, 1});
    matrix.set_row(2, {1, 1, 9});

    Matrix<double> matrix1 (3,3);

    matrix1.set_row(0, {1, 0, 0});
    matrix1.set_row(1, {0, -1, 0});
    matrix1.set_row(2, {-1, 0, 1});

    assert_true(matrix.is_positive(), LOG);
    assert_true(!matrix1.is_positive(), LOG);
}


void MatrixTest::test_is_row_equal_to()
{
    cout << "test_is_row_equal_to\n";

    Matrix<double> matrix (3,3);

    matrix.set_row(0, {5, 3, 1});
    matrix.set_row(1, {3, 5, 5});
    matrix.set_row(2, {1, 1, 9});

    assert_true(matrix.is_row_equal_to(1, {1, 2}, 5), LOG);
    assert_true(!matrix.is_row_equal_to(1, {0}, 5), LOG);
}


void MatrixTest::test_has_column_value()
{
    cout << "test_has_column_value\n";

    Matrix<double> matrix (3,3);

    matrix.set_row(0, {5, 3, 1});
    matrix.set_row(1, {3, 5, 5});
    matrix.set_row(2, {1, 1, 9});

    assert_true(matrix.has_column_value(1,3), LOG);
    assert_true(!matrix.has_column_value(1,9), LOG);
}


void MatrixTest::test_count_diagonal_elements()
{
    cout << "test_count_diagonal_elements\n";

    Matrix<double> matrix (3,3);

    const size_t solution = 3;
    size_t check_matrix;

    matrix.set_row(0, {5, 3, 1});
    matrix.set_row(1, {3, 5, 5});
    matrix.set_row(2, {1, 1, 9});

    check_matrix = matrix.count_diagonal_elements();

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_count_off_diagonal_elements()
{
    cout << "test_count_off_diagonal_elements\n";

    Matrix<double> matrix (3,3);

    const size_t solution = 3;
    size_t check_matrix;

    matrix.set_row(0, {5, 3, 0});
    matrix.set_row(1, {3, 5, 5});
    matrix.set_row(2, {0, 0, 9});

    check_matrix = matrix.count_off_diagonal_elements();

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_count_equal_to()
{
    cout << "test_count_equal_to\n";

    //Normal case

    Matrix<double> matrix (3,3);

    const size_t solution = 3;
    size_t check_matrix;


    matrix.set_row(0, {5, 3, 0});
    matrix.set_row(1, {3, 5, 5});
    matrix.set_row(2, {0, 0, 9});

    check_matrix = matrix.count_equal_to(5);

    assert_true(check_matrix == solution, LOG);

    // Column case

    Matrix<double> matrix1 (3,3);

    const size_t solution1 = 2;
    size_t check_matrix1;


    matrix1.set_row(0, {5, 3, 0});
    matrix1.set_row(1, {3, 5, 9});
    matrix1.set_row(2, {0, 0, 9});

    check_matrix1 = matrix1.count_equal_to(2,9);

    assert_true(check_matrix1 == solution1, LOG);

    // Set values case

    Matrix<double> matrix2 (3,3);

    const size_t solution2 = 2;
    size_t check_matrix2;


    matrix2.set_row(0, {5, 3, 0});
    matrix2.set_row(1, {3, 5, 9});
    matrix2.set_row(2, {0, 0, 9});

    check_matrix2 = matrix2.count_equal_to(2, {9});

    assert_true(check_matrix2 == solution2, LOG);

    // Header case

    matrix2.set_header({"col1","col2","col3"});

    assert_true(matrix2.count_equal_to("col3",9) == 2, LOG);
}


void MatrixTest::test_count_not_equal_to()
{
    cout << "test_count_not_equal_to\n";

    //Normal case

    Matrix<double> matrix (3,3);

    const size_t solution = 6;
    size_t check_matrix;


    matrix.set_row(0, {5, 3, 0});
    matrix.set_row(1, {3, 5, 5});
    matrix.set_row(2, {0, 0, 9});

    check_matrix = matrix.count_not_equal_to(5);

    assert_true(check_matrix == solution, LOG);

    // Column case

    Matrix<double> matrix1 (3,3);

    const size_t solution1 = 2;
    size_t check_matrix1;


    matrix1.set_row(0, {5, 3, 0});
    matrix1.set_row(1, {3, 5, 9});
    matrix1.set_row(2, {0, 0, 9});

    check_matrix1 = matrix1.count_not_equal_to(2,0);

    assert_true(check_matrix1 == solution1, LOG);

    // Set values case

    Matrix<double> matrix2 (3,3);

    const size_t solution2 = 1;
    size_t check_matrix2;


    matrix2.set_row(0, {5, 3, 0});
    matrix2.set_row(1, {3, 5, 9});
    matrix2.set_row(2, {0, 0, 9});

    check_matrix2 = matrix2.count_not_equal_to(2, {9});

    assert_true(check_matrix2 == solution2, LOG);

    //Two columns case

    const size_t solution3 = 1;
    size_t check_matrix3;

    check_matrix3 = matrix2.count_not_equal_to(0, 3, 2, 9);

    assert_true(check_matrix3 == solution3, LOG);


    // Header case

    matrix2.set_header({"col1","col2","col3"});

    assert_true(matrix2.count_not_equal_to("col3",9) == 1, LOG);

    //Two columns - header case

    assert_true(matrix2.count_not_equal_to("col1", 5, "col3",0 ) == 2,  LOG);
}


void MatrixTest::test_count_equal_to_by_rows()
{
    cout << "test_count_equal_to_by_rows\n";

    Matrix<double> matrix (3,3);

    const Vector<size_t> solution({1, 2, 0});
    Vector<size_t> check_matrix;

    matrix.set_row(0, {5, 3, 0});
    matrix.set_row(1, {3, 5, 5});
    matrix.set_row(2, {0, 0, 9});

    assert_true(matrix.count_equal_to_by_rows(5) == solution, LOG);
}


void MatrixTest::test_count_rows_equal_to()
{
    cout << "test_count_rows_equal_to\n";

    Matrix<double> matrix (3,3);

    const size_t solution = 1;

    matrix.set_row(0, {5, 3, 0});
    matrix.set_row(1, {5, 5, 5});
    matrix.set_row(2, {0, 0, 9});

    assert_true(matrix.count_rows_equal_to(5) == solution, LOG);
}


void MatrixTest::test_count_rows_not_equal_to()
{
    cout << "test_count_rows_not_equal_to\n";

    Matrix<double> matrix (3,3);

    const size_t solution = 2;

    matrix.set_row(0, {5, 3, 0});
    matrix.set_row(1, {5, 5, 5});
    matrix.set_row(2, {0, 0, 9});

    assert_true(matrix.count_rows_not_equal_to(5) == solution, LOG);
}


void MatrixTest::test_has_nan()
{
    cout << "test_has_nan\n";

    Matrix<double> matrix (3,3);

    matrix.set_row(0, {5, 3, 0});
    matrix.set_row(1, {5, 5, 5});
    matrix.set_row(2, {0, 0, 9});

    assert_true(!matrix.has_nan(), LOG);
}


void MatrixTest::test_count_nan()
{
    cout << "test_count_nan\n";

    Matrix<double> matrix (3,3);

    matrix.set_row(0, {5, 3, 0});
    matrix.set_row(1, {5, 5, 5});
    matrix.set_row(2, {0, 0, 9});

    assert_true(matrix.count_nan() == 0, LOG);
}


void MatrixTest::test_count_rows_with_nan()
{
    cout << "test_count_rows_with_nan\n";

    Matrix<double> matrix (3,3);

    matrix.set_row(0, {5, 3, 0});
    matrix.set_row(1, {5, 5, 5});
    matrix.set_row(2, {NAN, 0, 9});

    assert_true(matrix.count_rows_with_nan() == 1, LOG);
}


void MatrixTest::test_count_columns_with_nan()
{
    cout << "test_count_columns_with_nan\n";

    Matrix<double> matrix (3,3);

    matrix.set_row(0, {5, 3, 0});
    matrix.set_row(1, {5, 5, 5});
    matrix.set_row(2, {0, 0, 9});

    assert_true(matrix.count_columns_with_nan() == 0, LOG);
}


void MatrixTest::test_count_nan_rows()
{
    cout << "test_count_nan_rows\n";

    Matrix<double> matrix (3,3);

    Vector<size_t> solution({1, 0, 2});

    matrix.set_row(0, {5, NAN, 0});
    matrix.set_row(1, {5, 5, 5});
    matrix.set_row(2, {NAN, NAN, 9});

    assert_true(matrix.count_nan_rows() == solution, LOG);
}


void MatrixTest::test_count_nan_columns()
{
    cout << "test_count_nan_columns\n";

    Matrix<double> matrix (3,3);

    Vector<size_t> solution({1, 0, 2});

    matrix.set_column(0, {5, NAN, 0});
    matrix.set_column(1, {5, 5, 5});
    matrix.set_column(2, {NAN, NAN, 9});

    assert_true(matrix.count_nan_columns() == solution, LOG);
}


void MatrixTest::test_filter_column_equal_to()
{
    cout << "test_filter_column_equal_to\n";

    //Normal case

    Matrix<double> matrix (3,3);
    Matrix<double> solution (2, 3);

    matrix.set_row(0, {5, 3, 0});
    matrix.set_row(1, {5, 5, 2});
    matrix.set_row(2, {0, 0, 9});

    solution.set_row(0, {5, 3, 0});
    solution.set_row(1, {5, 5, 2});

    assert_true(matrix.filter_column_equal_to(0,5) == solution, LOG);

    //Header case

    matrix.set_header({"col1", "col2", "col3"});
    assert_true(matrix.filter_column_equal_to("col1", 5) == solution, LOG);

    //Set values case

    assert_true(matrix.filter_column_equal_to(0, {5,5}) == solution, LOG);
}


void MatrixTest::test_filter_column_not_equal_to()
{
    cout << "test_filter_column_not_equal_to\n";

    //Normal case

    Matrix<double> matrix (3,3);
    Matrix<double> solution (2, 3);

    matrix.set_row(0, {5, 3, 0});
    matrix.set_row(1, {5, 5, 2});
    matrix.set_row(2, {0, 0, 9});

    solution.set_row(0, {5, 3, 0});
    solution.set_row(1, {5, 5, 2});

    assert_true(matrix.filter_column_not_equal_to(0,0) == solution, LOG);

    //Header case

    matrix.set_header({"col1", "col2", "col3"});
    assert_true(matrix.filter_column_not_equal_to("col1", 0) == solution, LOG);

    //Set values case

    assert_true(matrix.filter_column_not_equal_to(0, {0}) == solution, LOG);
}


void MatrixTest::test_filter_column_less_than()
{
    cout << "test_filter_column_less_than\n";

    //Normal case

    Matrix<double> matrix (3,3);
    Matrix<double> solution (1,3);

    matrix.set_row(0, {5, 3, 0});
    matrix.set_row(1, {5, 5, 2});
    matrix.set_row(2, {0, 0, 9});

    solution.set_row(0, {0, 0, 9});

    assert_true(matrix.filter_column_less_than(0,5) == solution, LOG);

    //Header case

    matrix.set_header({"col1", "col2", "col3"});
    assert_true(matrix.filter_column_less_than("col1", 5) == solution, LOG);

    //Set values case

    assert_true(matrix.filter_column_less_than(0, {5}) == solution, LOG);
}


void MatrixTest::test_filter_column_greater_than()
{
    cout << "test_filter_column_greater_than\n";

    //Normal case

    Matrix<double> matrix (3,3);
    Matrix<double> solution (2, 3);

    matrix.set_row(0, {5, 3, 0});
    matrix.set_row(1, {5, 5, 2});
    matrix.set_row(2, {0, 0, 9});

    solution.set_row(0, {5, 3, 0});
    solution.set_row(1, {5, 5, 2});

    assert_true(matrix.filter_column_greater_than(0,1) == solution, LOG);

    //Header case

    matrix.set_header({"col1", "col2", "col3"});
    assert_true(matrix.filter_column_greater_than("col1", 1) == solution, LOG);

    //Set values case

    assert_true(matrix.filter_column_greater_than(0, {1}) == solution, LOG);
}


void MatrixTest::test_filter_column_minimum_maximum()
{
    cout << "test_filter_column_minimum_maximum\n";

    //Normal case

    Matrix<double> matrix (3,3);
    Matrix<double> solution (2, 3);

    matrix.set_row(0, {7, 3, 0});
    matrix.set_row(1, {5, 5, 2});
    matrix.set_row(2, {0, 0, 9});

    solution.set_row(0, {7, 3, 0});
    solution.set_row(1, {5, 5, 2});

    assert_true(matrix.filter_column_minimum_maximum(0,4,8) == solution, LOG);

    //Header case

    matrix.set_header({"col1", "col2", "col3"});
    assert_true(matrix.filter_column_minimum_maximum("col1", 1,8) == solution, LOG);
}


void MatrixTest::test_initialize()
{
   cout << "test_initialize\n";

   Matrix<double> matrix (3,3);
   Matrix<double> solution (3, 3);

   solution.set_row(0, {3, 3, 3});
   solution.set_row(1, {3, 3, 3});
   solution.set_row(2, {3, 3, 3});

   matrix.initialize(3);

   assert_true(matrix == solution, LOG);
}


void MatrixTest::test_get_diagonal()
{
   cout << "test_get_diagonal\n";

   Matrix<size_t> matrix(2, 2, 1);

   Vector<size_t> diagonal = matrix.get_diagonal();

   assert_true(diagonal.size() == 2, LOG);
   assert_true(diagonal == 1, LOG);
}


void MatrixTest::test_set_diagonal()
{
   cout << "test_set_diagonal\n";

   Matrix<size_t> matrix;
   Vector<size_t> diagonal;

   // Test

   matrix.set(2, 2, 1);

   matrix.set_diagonal(0);

   diagonal = matrix.get_diagonal();

   assert_true(diagonal.size() == 2, LOG);
   assert_true(diagonal == 0, LOG);

   // Test

   diagonal.set(2);
   diagonal[0] = 1;
   diagonal[1] = 0;

   matrix.set_diagonal(diagonal);

   diagonal = matrix.get_diagonal();

   assert_true(diagonal.size() == 2, LOG);
   assert_true(diagonal[0] == 1, LOG);
   assert_true(diagonal[1] == 0, LOG);
}


void MatrixTest::test_sum_diagonal()
{
   cout << "test_sum_diagonal\n";

   Matrix<int> matrix;
   Matrix<int> sum;  
   Vector<int> diagonal;

   // Test

   matrix.set(2, 2, 1);

   matrix.sum_diagonal(1);

   diagonal = matrix.get_diagonal();

   assert_true(diagonal.size() == 2, LOG);
   assert_true(diagonal == 2, LOG);
}


void MatrixTest::test_append_row()
{
   cout << "test_append_row\n";

   Matrix<size_t> matrix(1, 1, 0);

   Vector<size_t> vector(1, 1);

   matrix = matrix.append_row(vector);

   assert_true(matrix.get_rows_number() == 2, LOG);
   assert_true(matrix(1,0) == 1, LOG);
}


void MatrixTest::test_append_column()
{
   cout << "test_append_column\n";

   Matrix<size_t> matrix(1, 1, 0);

   Vector<size_t> vector(1, 1);

   matrix = matrix.append_column(vector);

   assert_true(matrix.get_columns_number() == 2, LOG);
   assert_true(matrix(0,1) == 1, LOG);
}


void MatrixTest::test_insert_row()
{
   cout << "test_insert_row\n";

   Matrix<size_t> matrix(2, 1, 0);

   Vector<size_t> vector(1, 1);

   matrix = matrix.insert_row(1, vector);

   assert_true(matrix.get_rows_number() == 3, LOG);
   assert_true(matrix(1,0) == 1, LOG);
}


void MatrixTest::test_insert_column()
{
   cout << "test_insert_column\n";

   Matrix<size_t> matrix(1, 2, 0);

   Vector<size_t> vector(1, 1);

   matrix = matrix.insert_column(1, vector);

   assert_true(matrix.get_columns_number() == 3, LOG);
   assert_true(matrix(0,1) == 1, LOG);
}


void MatrixTest::test_insert_row_values()
{
    cout << "test_insert_row_values\n";

    Matrix<size_t> matrix(3,3);
    Matrix<size_t> solution(3,3);

    solution.set_row(0, {1, 2, 3});
    solution.set_row(1, {4, 5, 6});
    solution.set_row(2, {11, 12, 13});

    matrix.set_row(0, {1, 2, 3});
    matrix.set_row(1, {4, 5, 6});
    matrix.set_row(2, {7, 8, 9});

    Vector<size_t> vector({11, 12, 13});

    matrix.insert_row_values(2, 0, vector);

    assert_true(matrix == solution, LOG);
}


void MatrixTest::test_add_column()
{
    cout << "test_add_column\n";


    Matrix<size_t> matrix(3,3);
    Matrix<size_t> solution(3,4);
    Matrix<size_t> check_matrix;

    check_matrix = matrix.add_columns(1);

    assert_true(check_matrix.get_columns_number() == solution.get_columns_number(), LOG);
}


void MatrixTest::test_add_column_first()
{
    cout << "test_add_column_first\n";

    Matrix<size_t> matrix(3,3);
    Matrix<size_t> solution(3,4);
    Matrix<size_t> check_matrix;

    check_matrix = matrix.add_columns_first(1);

    assert_true(check_matrix.get_columns_number() == solution.get_columns_number(), LOG);
}


void MatrixTest::test_swap_columns()
{
    cout << "test_swap_columns\n";

    //Normal case

    Matrix<size_t> matrix(3,3);
    Matrix<size_t> solution(3,3);

    solution.set_column(0, {1, 2, 3});
    solution.set_column(1, {4, 5, 6});
    solution.set_column(2, {7, 8, 9});

    matrix.set_column(0, {1, 2, 3});
    matrix.set_column(1, {7, 8, 9});
    matrix.set_column(2, {4, 5, 6});

    matrix.swap_columns(1,2);

    assert_true(matrix == solution, LOG);

    // Header case

    Matrix<size_t> matrix1(3,3);
    Matrix<size_t> solution1(3,3);

    solution1.set_column(0, {1, 2, 3});
    solution1.set_column(1, {4, 5, 6});
    solution1.set_column(2, {7, 8, 9});

    matrix1.set_column(0, {1, 2, 3});
    matrix1.set_column(1, {7, 8, 9});
    matrix1.set_column(2, {4, 5, 6});

    solution1.set_header({"col1", "col2", "col3"});
    matrix1.set_header({"col1", "col2", "col3"});

    matrix1.swap_columns("col2", "col3");

    assert_true(matrix1 == solution1, LOG);
}


void MatrixTest::test_delete_row()
{
    cout << "test_delete_row\n";

    //Normal case

    Matrix<size_t> matrix(3,3);
    Matrix<size_t> solution(2,3);
    Matrix<size_t> check_matrix;

    solution.set_row(0, {1, 2, 3});
    solution.set_row(1, {4, 5, 6});

    matrix.set_row(0, {1, 2, 3});
    matrix.set_row(1, {4, 5, 6});
    matrix.set_row(2, {7, 8, 9});

    check_matrix = matrix.delete_row(2);

    assert_true(check_matrix == solution, LOG);

    // More than one row case

    check_matrix = matrix.delete_row({2});

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_delete_column()
{
    cout << "test_delete_column\n";

    //Normal case

    Matrix<size_t> matrix(3,3);
    Matrix<size_t> solution(3,2);
    Matrix<size_t> check_matrix;

    solution.set_column(0, {1, 2, 3});
    solution.set_column(1, {4, 5, 6});

    matrix.set_column(0, {1, 2, 3});
    matrix.set_column(1, {4, 5, 6});
    matrix.set_column(2, {7, 8, 9});

    check_matrix = matrix.delete_column(2);

    assert_true(check_matrix == solution, LOG);

    // More than one row case

    check_matrix = matrix.delete_column({2});

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_delete_rows_wiht_value()
{
    cout << "test_delete_row_with_value\n";

    //Normal case

    Matrix<size_t> matrix(3,3);
    Matrix<size_t> solution(2,3);
    Matrix<size_t> check_matrix;

    solution.set_row(0, {1, 2, 3});
    solution.set_row(1, {4, 5, 6});

    matrix.set_row(0, {1, 2, 3});
    matrix.set_row(1, {4, 5, 6});
    matrix.set_row(2, {7, 8, 9});

    check_matrix = matrix.delete_rows_with_value(7);

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_delete_columns_with_value()
{
    cout << "test_delete_columns_with_value\n";

    //Normal case

    Matrix<size_t> matrix(3,3);
    Matrix<size_t> solution(3,2);
    Matrix<size_t> check_matrix;

    solution.set_column(0, {1, 2, 3});
    solution.set_column(1, {4, 5, 6});

    matrix.set_column(0, {1, 2, 3});
    matrix.set_column(1, {4, 5, 6});
    matrix.set_column(2, {7, 8, 9});

    check_matrix = matrix.delete_columns_with_value(7);

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_delete_first_rows()
{
    cout << "test_delete_first_rows\n";

    //Normal case

    Matrix<size_t> matrix(3,3);
    Matrix<size_t> solution(1,3);
    Matrix<size_t> check_matrix;

    solution.set_row(0, {7, 8, 9});

    matrix.set_row(0, {1, 2, 3});
    matrix.set_row(1, {4, 5, 6});
    matrix.set_row(2, {7, 8, 9});

    check_matrix = matrix.delete_first_rows(2);

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_delete_last_rows()
{
    cout << "test_delete_last_rows\n";

    //Normal case

    Matrix<size_t> matrix(3,3);
    Matrix<size_t> solution(1,3);
    Matrix<size_t> check_matrix;

    solution.set_row(0, {1, 2, 3});

    matrix.set_row(0, {1, 2, 3});
    matrix.set_row(1, {4, 5, 6});
    matrix.set_row(2, {7, 8, 9});

    check_matrix = matrix.delete_last_rows(2);

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_delete_first_columns()
{
    cout << "test_delete_first_columns\n";

    //Normal case

    Matrix<size_t> matrix(3,3);
    Matrix<size_t> solution(3,1);
    Matrix<size_t> check_matrix;

    solution.set_column(0, {7, 8, 9});

    matrix.set_column(0, {1, 2, 3});
    matrix.set_column(1, {4, 5, 6});
    matrix.set_column(2, {7, 8, 9});

    check_matrix = matrix.delete_first_columns(2);

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_delete_last_columns()
{
    cout << "test_delete_last_columns\n";

    //Normal case

    Matrix<size_t> matrix(3,3);
    Matrix<size_t> solution(3,1);
    Matrix<size_t> check_matrix;

    solution.set_column(0, {1, 2, 3});

    matrix.set_column(0, {1, 2, 3});
    matrix.set_column(1, {4, 5, 6});
    matrix.set_column(2, {7, 8, 9});

    check_matrix = matrix.delete_last_columns(2);

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_delete_columns_name_contain()
{
    cout << "test_delete_columnd_name_contain\n";

    //Normal case

    Matrix<size_t> matrix(3,3);
    Matrix<size_t> solution(3,1);
    Matrix<size_t> check_matrix;

    solution.set_column(0, {1, 2, 3});

    matrix.set_column(0, {1, 2, 3});
    matrix.set_column(1, {4, 5, 6});
    matrix.set_column(2, {7, 8, 9});

    matrix.set_header({"col1", "columna1", "columna2"});

    check_matrix = matrix.delete_columns_name_contains({"columna"});

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_delete_constant_rows()
{
    cout << "test_delete_constant_rows\n";

    //Normal case

    Matrix<double> matrix(3,3);
    Matrix<double> solution(1,3);
    Matrix<double> check_matrix;

    solution.set_row(0, {1, 2, 3});

    matrix.set_row(0, {1, 2, 3});
    matrix.set_row(1, {5, 5, 5});
    matrix.set_row(2, {5, 5, 5});

    check_matrix = matrix.delete_constant_rows();

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_delete_constant_columns()
{
    cout << "test_delete_constant_columns\n";

    //Normal case

    Matrix<double> matrix(3,3);
    Matrix<double> solution(3,1);
    Matrix<double> check_matrix;

    solution.set_column(0, {1, 2, 3});

    matrix.set_column(0, {1, 2, 3});
    matrix.set_column(1, {5, 5, 5});
    matrix.set_column(2, {5, 5, 5});

    check_matrix = matrix.delete_constant_columns();

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_delete_binary_columns()
{
    cout << "test_delete_binary_columns\n";

    //Normal case

    Matrix<double> matrix(3,3);
    Matrix<double> solution(3,1);
    Matrix<double> check_matrix;

    solution.set_column(0, {1, 2, 3});

    matrix.set_column(0, {1, 2, 3});
    matrix.set_column(1, {1, 0, 1});
    matrix.set_column(2, {1, 0, 1});

    check_matrix = matrix.delete_binary_columns();

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_assembla_rows()
{
    cout << "test_assembla_rows\n";

    Matrix<size_t> matrix(2,2);
    Matrix<size_t> matrix1(2,2);
    Matrix<size_t> solution(4,2);
    Matrix<size_t> check_matrix;

    matrix.set_row(0, {1, 2});
    matrix.set_row(1, {3, 4});

    matrix1.set_row(0, {5, 6});
    matrix1.set_row(1, {7, 8});

    solution.set_row(0, {1, 2});
    solution.set_row(1, {3, 4});
    solution.set_row(2, {5, 6});
    solution.set_row(3, {7, 8});

    check_matrix = matrix.assemble_rows(matrix1);

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_assembla_columns()
{
    cout << "test_assembla_columns\n";

    Matrix<size_t> matrix(2,2);
    Matrix<size_t> matrix1(2,2);
    Matrix<size_t> solution(2,4);
    Matrix<size_t> check_matrix;

    matrix.set_column(0, {1, 2});
    matrix.set_column(1, {3, 4});

    matrix1.set_column(0, {5, 6});
    matrix1.set_column(1, {7, 8});

    solution.set_row(0, {1, 3, 5, 7});
    solution.set_row(1, {2, 4, 6, 8});

    check_matrix = matrix.assemble_columns(matrix1);

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_subtract_row()
{
   cout << "test_subtract_row\n";

   Matrix<size_t> matrix(2, 2);
   Matrix<size_t> solution(2, 2);

   matrix.set_row(0, {1, 2});
   matrix.set_row(1, {3, 4});

   solution.set_row(0, {0, 1});
   solution.set_row(1, {2, 3});

   assert_true(matrix.subtract_rows({1, 1}) == solution, LOG);
}


void MatrixTest::test_sort_ascending()
{
    cout << "test_sort_ascending\n";

    Matrix<double> matrix(3,3);
    Matrix<double> solution(3,3);
    Matrix<double> check_matrix;

    matrix.set_column(0, {1, 7, 5});
    matrix.set_column(1, {3, 7, 5});
    matrix.set_column(2, {2, 2, 5});

    solution.set_row(0, {1, 3, 2});
    solution.set_row(1, {5, 5, 5});
    solution.set_row(2, {7, 7, 2});

    check_matrix = matrix.sort_ascending(1);

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_sort_descending()
{
    cout << "test_sort_descending\n";

    Matrix<double> matrix(3,3);
    Matrix<double> solution(3,3);
    Matrix<double> check_matrix;

    matrix.set_column(0, {1, 7, 5});
    matrix.set_column(1, {3, 7, 5});
    matrix.set_column(2, {2, 2, 5});

    solution.set_row(2, {1, 3, 2});
    solution.set_row(1, {5, 5, 5});
    solution.set_row(0, {7, 7, 2});

    check_matrix = matrix.sort_descending(1);

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_replace()
{
    cout <<"test_replace\n";

    //Normal case

    Matrix<double> matrix(3,3);
    Matrix<double> solution(3,3);
    Matrix<double> check_matrix;

    matrix.set_column(0, {1, 7, 5});
    matrix.set_column(1, {3, 7, 5});
    matrix.set_column(2, {2, 2, 5});

    solution.set_column(0, {0, 7, 5});
    solution.set_column(1, {3, 7, 5});
    solution.set_column(2, {2, 2, 5});

    matrix.replace(1, 0);

    assert_true(matrix == solution, LOG);

    //More than one equal value

    solution.set_column(0, {0, 7, 8});
    solution.set_column(1, {3, 7, 8});
    solution.set_column(2, {2, 2, 8});

    matrix.replace(5, 8);

    assert_true(matrix == solution, LOG);
}


void MatrixTest::test_replace_header()
{
    cout << "test_replace_header\n";

    Matrix<double> matrix(3,3);
    Matrix<double> solution(3,3);
    Matrix<double> check_matrix;

    matrix.set_column(0, {1, 7, 5});
    matrix.set_column(1, {3, 7, 5});
    matrix.set_column(2, {2, 2, 5});

    matrix.set_header({"col1", "col2", "col3"});

    solution.set_column(0, {1, 7, 5});
    solution.set_column(1, {3, 7, 5});
    solution.set_column(2, {2, 2, 5});

    solution.set_header({"col1", "col2", "columna"});

    matrix.replace_header("col3", "columna");

    assert_true(matrix == solution, LOG);
}


void MatrixTest::test_replace_in_row()
{
    cout << "test_replace_in_row\n";

    Matrix<double> matrix(3,3);
    Matrix<double> solution(3,3);
    Matrix<double> check_matrix;

    matrix.set_row(0, {1, 7, 5});
    matrix.set_row(1, {3, 7, 5});
    matrix.set_row(2, {2, 2, 5});

    solution.set_row(0, {1, 7, 5});
    solution.set_row(1, {3, 9, 5});
    solution.set_row(2, {2, 2, 5});

    matrix.replace_in_row(1, 7, 9);

    assert_true(matrix == solution, LOG);
}


void MatrixTest::test_replace_in_column()
{
    cout << "test_replace_in_column\n";

    // Normal case

    Matrix<double> matrix(3,3);
    Matrix<double> solution(3,3);
    Matrix<double> check_matrix;

    matrix.set_column(0, {1, 7, 5});
    matrix.set_column(1, {3, 7, 5});
    matrix.set_column(2, {2, 2, 5});

    solution.set_column(0, {1, 7, 5});
    solution.set_column(1, {3, 9, 5});
    solution.set_column(2, {2, 2, 5});

    matrix.replace_in_column(1, 7, 9);

    assert_true(matrix == solution, LOG);

    //Header case

    matrix.set_header({"col1", "col2", "col3"});
    solution.set_header({"col1", "col2", "col3"});

    solution.set_column(0, {3, 7, 5});

    matrix.replace_in_column("col1", 1, 3);

    assert_true(matrix == solution, LOG);
}


void MatrixTest::test_replace_substring()
{
    cout << "test_replace_substring\n";

    //Normal case

    Matrix<string> matrix(3,3);
    Matrix<string> solution(3,3);
    Matrix<string> check_matrix;

    matrix.set_row(0, {"hello", "bye", "home"});
    matrix.set_row(1, {"hello", "byes", "homes"});
    matrix.set_row(2, {"hellod", "byed", "homed"});

    solution.set_row(0, {"columna", "bye", "home"});
    solution.set_row(1, {"columna", "byes", "homes"});
    solution.set_row(2, {"columnad", "byed", "homed"});

    matrix.replace_substring("hello", "columna");

    assert_true(matrix == solution, LOG);

    //Column case

    solution.set_row(2, {"columnad", "dog", "homed"});

    matrix.replace_substring(1, "byed", "dog");

    assert_true(matrix == solution, LOG);

    // Header case

    matrix.set_header({"col1", "col2", "col3"});
    solution.set_header({"col1", "col2", "col3"});

    solution.set_row(2, {"columnad", "dog", "cat"});

    matrix.replace_substring("col3", "homed", "cat");

    assert_true(matrix == solution, LOG);
}


void MatrixTest::test_replace_contains()
{
    cout << "test_replace_contains\n";

    //Normal case

    Matrix<string> matrix(3,3);
    Matrix<string> solution(3,3);
    Matrix<string> check_matrix;

    matrix.set_row(0, {"hello", "bye", "home"});
    matrix.set_row(1, {"hello", "byes", "homes"});
    matrix.set_row(2, {"hellod", "byed", "homed"});

    solution.set_row(0, {"columna", "bye", "home"});
    solution.set_row(1, {"columna", "byes", "homes"});
    solution.set_row(2, {"columna", "byed", "homed"});

    matrix.replace_contains("hello", "columna");

    assert_true(matrix == solution, LOG);
}


void MatrixTest::test_replace_contains_in_row()
{
    cout << "test_replace_contains\n";

    //Normal case

    Matrix<string> matrix(3,3);
    Matrix<string> solution(3,3);
    Matrix<string> check_matrix;

    matrix.set_row(0, {"hello", "bye", "home"});
    matrix.set_row(1, {"hello", "byes", "homes"});
    matrix.set_row(2, {"hellod", "byed", "homed"});

    solution.set_row(0, {"hello", "bye", "home"});
    solution.set_row(1, {"column", "byes", "homes"});
    solution.set_row(2, {"hellod", "byed", "homed"});

    matrix.replace_contains_in_row(1, "hello", "column");

    assert_true(matrix == solution, LOG);
}


void MatrixTest::test_replace_column_equal_to()
{
    cout << "test_replace_column_equal_to\n";

    Matrix<double> matrix(3,3);
    Matrix<double> solution(3,3);
    Matrix<double> check_matrix;

    matrix.set_column(0, {1, 7, 5});
    matrix.set_column(1, {3, 7, 5});
    matrix.set_column(2, {2, 2, 5});

    solution.set_column(0, {1, 7, 5});
    solution.set_column(1, {3, 9, 5});
    solution.set_column(2, {2, 2, 5});

    matrix.set_header({"col1", "col2", "col3"});
    solution.set_header({"col1", "col2", "col3"});

    matrix.replace_column_equal_to("col2", 7, 9);

    assert_true(matrix == solution, LOG);
}


void MatrixTest::test_replace_column_not_equal_to()
{
    cout << "test_replace_column_equal_to \n";

    Matrix<double> matrix(3,3);
    Matrix<double> solution(3,3);
    Matrix<double> check_matrix;

    matrix.set_column(0, {1, 7, 5});
    matrix.set_column(1, {3, 7, 5});
    matrix.set_column(2, {2, 2, 5});

    solution.set_column(0, {1, 7, 5});
    solution.set_column(1, {3, 7, 5});
    solution.set_column(2, {2, 2, 0});

    matrix.set_header({"col1", "col2", "col3"});
    solution.set_header({"col1", "col2", "col3"});

    matrix.replace_column_not_equal_to("col3", 2, 0);

    assert_true(matrix == solution, LOG);
}


void MatrixTest::test_replace_column_contain()
{
    cout << "test_replace_column_contain\n";

    Matrix<string> matrix(3,3);
    Matrix<string> solution(3,3);
    Matrix<string> check_matrix;

    matrix.set_column(0, {"hello", "bye", "home"});
    matrix.set_column(1, {"hello", "bye", "home"});
    matrix.set_column(2, {"hello", "bye", "home"});

    solution.set_column(0, {"hello", "bye", "home"});
    solution.set_column(1, {"hello", "bye", "home"});
    solution.set_column(2, {"hello", "bye", "dog"});

    matrix.set_header({"col1", "col2", "col3"});
    solution.set_header({"col1", "col2", "col3"});

    matrix.replace_column_contains("col3", "home", "dog");

    assert_true(matrix == solution, LOG);
}


void MatrixTest::test_randomize_uniform()
{
   cout << "test_randomize_uniform\n";

   Matrix<double> matrix(1, 1);

   matrix.randomize_uniform();

   assert_true(matrix >= -1.0, LOG);
   assert_true(matrix <= 1.0, LOG);

   matrix.randomize_uniform(-1.0, 0.0);

   assert_true(matrix >= -1.0, LOG);
   assert_true(matrix <= 0.0, LOG);
}


void MatrixTest::test_randomize_normal()
{
   cout << "test_randomize_normal\n";
}


void MatrixTest::test_initialize_identity()
{
    cout << "test_initialize_identity\n";

    Matrix<size_t> matrix(3,3);
    Matrix<size_t> solution(3,3);

    solution.set_column(0, {1, 0, 0});
    solution.set_column(1, {0, 1, 0});
    solution.set_column(2, {0, 0, 1});

    matrix.initialize_identity();

    assert_true(matrix == solution, LOG);
}


void MatrixTest::test_initialize_diagonal()
{
    cout << "test_initialize_diagonal\n";

    Matrix<size_t> matrix(3,3);
    Matrix<size_t> solution(3,3);

    solution.set_column(0, {3, 0, 0});
    solution.set_column(1, {0, 3, 0});
    solution.set_column(2, {0, 0, 3});

    matrix.initialize_diagonal(3);

    assert_true(matrix == solution, LOG);

    Matrix<size_t> solution1(4,4);
    Matrix<size_t> matrix1;

    solution1.set_column(0, {5, 0, 0, 0});
    solution1.set_column(1, {0, 5, 0, 0});
    solution1.set_column(2, {0, 0, 5, 0});
    solution1.set_column(3, {0, 0, 0, 5});

    matrix1.initialize_diagonal(4, 5);

    assert_true(matrix1 == solution1, LOG);
}


void MatrixTest::test_append_header()
{
    cout << "test_append_header\n";

    Matrix<size_t> matrix1(4,4);

    const Vector<string> header({"col1a", "col2a", "col3a", "col4a"});

    matrix1.set_column(0, {5, 0, 0, 0});
    matrix1.set_column(1, {0, 5, 0, 0});
    matrix1.set_column(2, {0, 0, 5, 0});
    matrix1.set_column(3, {0, 0, 0, 5});

    matrix1.set_header({"col1", "col2", "col3", "col4"});

    matrix1.append_header("a");

    assert_true(matrix1.get_header() == header, LOG);
}


void MatrixTest::test_tuck_in()
{
    cout << "tets_tuck_in\n";

    //Matrix case

    Matrix<size_t> matrix1(4,4);
    Matrix<size_t> solution(4,4);
    Matrix<size_t> other_matrix(2,2);

    matrix1.set_column(0, {5, 0, 0, 0});
    matrix1.set_column(1, {0, 5, 0, 0});
    matrix1.set_column(2, {0, 0, 5, 0});
    matrix1.set_column(3, {0, 0, 0, 5});

    solution.set_column(0, {5, 0, 0, 0});
    solution.set_column(1, {0, 5, 0, 0});
    solution.set_column(2, {0, 0, 7, 8});
    solution.set_column(3, {0, 0, 6, 9});

    other_matrix.set_column(0, {7, 8});
    other_matrix.set_column(1, {6, 9});

    matrix1.embed(2, 2, other_matrix);

    assert_true(matrix1 == solution, LOG);

    // Vector case

    const Vector<size_t> vector({1});

    Matrix<size_t> solution1(4,4);
    Matrix<size_t> matrix(4,4);

    matrix.set_column(0, {5, 0, 0, 0});
    matrix.set_column(1, {0, 5, 0, 0});
    matrix.set_column(2, {0, 0, 5, 0});
    matrix.set_column(3, {0, 0, 0, 5});

    solution1.set_column(0, {5, 0, 0, 0});
    solution1.set_column(1, {0, 5, 0, 0});
    solution1.set_column(2, {0, 0, 5, 0});
    solution1.set_column(3, {1, 0, 0, 5});

    matrix.embed(0, 3, vector);

    assert_true(matrix == solution1, LOG);
}


void MatrixTest::test_set_to_identity()
{
   cout << "test_set_to_identity\n";

   Matrix<int> matrix1(2, 2);
   matrix1.initialize_identity();

   Matrix<int> matrix2(2, 2);
   matrix2(0,0) = 1;
   matrix2(0,1) = 0;
   matrix2(1,0) = 0;
   matrix2(1,1) = 1;

   assert_true(matrix1 == matrix2, LOG);
}


void MatrixTest::test_calculate_sum()
{
    cout << "test_calculate_sum\n";

    Matrix<size_t> matrix(3,3);

    const size_t solution = 45;
    size_t check_matrix;

    matrix.set_column(0, {1, 2, 3});
    matrix.set_column(1, {4, 5, 6});
    matrix.set_column(2, {7, 8, 9});

    check_matrix = matrix.calculate_sum();

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_calculate_rows_sum()
{
    cout << "test_calculate_rows_sum\n";
    Matrix<size_t> matrix(3,3);

    const Vector<size_t> solution({12, 15, 18});
    Vector<size_t> check_matrix;

    matrix.set_column(0, {1, 2, 3});
    matrix.set_column(1, {4, 5, 6});
    matrix.set_column(2, {7, 8, 9});

    check_matrix = matrix.calculate_rows_sum();

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_calculate_columns_sum()
{
    cout << "test_calculate_columns_sum\n";

    Matrix<size_t> matrix(3,3);

    const Vector<size_t> solution({6, 15, 24});
    Vector<size_t> check_matrix;

    matrix.set_column(0, {1, 2, 3});
    matrix.set_column(1, {4, 5, 6});
    matrix.set_column(2, {7, 8, 9});

    check_matrix = matrix.calculate_columns_sum();

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_sum_row()
{
    cout << "test_sum_row\n";

    Matrix<size_t> matrix(3,3);

    Matrix<size_t> solution(3,3);
    Matrix<size_t> check_matrix;

    matrix.set_row(0, {1, 2, 3});
    matrix.set_row(1, {4, 5, 6});
    matrix.set_row(2, {7, 8, 9});

    solution.set_row(0, {2, 3, 4});
    solution.set_row(1, {4, 5, 6});
    solution.set_row(2, {7, 8, 9});

    matrix.sum_row(0, {1, 1, 1});

    assert_true(matrix == solution, LOG);
}


void MatrixTest::test_sum_rows()
{
    cout << "test_sum_rows\n";

    Matrix<size_t> matrix(3,3);

    Matrix<size_t> solution(3,3);
    Matrix<size_t> check_matrix(3,3);

    matrix.set_row(0, {1, 2, 3});
    matrix.set_row(1, {4, 5, 6});
    matrix.set_row(2, {7, 8, 9});

    solution.set_row(0, {2, 3, 4});
    solution.set_row(1, {5, 6, 7});
    solution.set_row(2, {8, 9, 10});

    check_matrix = matrix.sum_rows({1, 1, 1});

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_substract_rows()
{
    cout << "test_substract_rows\n";

    Matrix<size_t> matrix(3,3);

    Matrix<size_t> solution(3,3);
    Matrix<size_t> check_matrix(3,3);

    matrix.set_row(0, {1, 2, 3});
    matrix.set_row(1, {4, 5, 6});
    matrix.set_row(2, {7, 8, 9});

    solution.set_row(0, {0, 1, 2});
    solution.set_row(1, {3, 4, 5});
    solution.set_row(2, {6, 7, 8});

    check_matrix = matrix.subtract_rows({1, 1, 1});

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_multiply_rows()
{
    cout << "test_multiply_rows\n";

    //Vector case

    Matrix<double> matrix(3,3);

    Matrix<double> solution(3,3);
    Matrix<double> check_matrix(3,3);

    Vector<double> vector({2, 2, 2});

    matrix.set_row(0, {1, 2, 3});
    matrix.set_row(1, {4, 5, 6});
    matrix.set_row(2, {7, 8, 9});

    solution.set_row(0, {2, 4, 6});
    solution.set_row(1, {8, 10, 12});
    solution.set_row(2, {14, 16, 18});

    check_matrix = matrix.multiply_rows(vector);

    assert_true(check_matrix == solution, LOG);

    //Matrix case

    Matrix<double> matrix1(3,3);

    Matrix<double> solution1(3,3);
    Matrix<double> check_matrix1(3,3);
    Matrix<double> matrix2(3,3);

    matrix1.set_row(0, {1, 2, 3});
    matrix1.set_row(1, {4, 5, 6});
    matrix1.set_row(2, {7, 8, 9});

    matrix2.set_row(0, {1, 1, 1});
    matrix2.set_row(1, {1, 1, 1});
    matrix2.set_row(2, {1, 1, 1});

    solution1.set_row(0, {1, 2, 3});
    solution1.set_row(1, {4, 5, 6});
    solution1.set_row(2, {7, 8, 9});

    check_matrix = matrix.multiply_rows(vector);

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_divide_rows()
{
    cout << "test_divide_rows\n";

    Matrix<double> matrix(3,3);

    Matrix<double> solution(3,3);

    Vector<double> vector({2, 2, 2});

    solution.set_row(0, {1, 2, 3});
    solution.set_row(1, {4, 5, 6});
    solution.set_row(2, {7, 8, 9});

    matrix.set_row(0, {2, 4, 6});
    matrix.set_row(1, {8, 10, 12});
    matrix.set_row(2, {14, 16, 18});

    matrix.divide_rows(vector);

    assert_true(matrix == solution, LOG);
}


void MatrixTest::test_calculate_trace()
{
    cout << "test_calculate_trace\n";

    Matrix<double> matrix1(3,3);

    double solution = 15;

    matrix1.set_row(0, {1, 2, 3});
    matrix1.set_row(1, {4, 5, 6});
    matrix1.set_row(2, {7, 8, 9});

    assert_true(matrix1.calculate_trace() == solution, LOG);
}


void MatrixTest::test_calculate_reverse_columns()
{
    cout << "test_calculate_reverse_columns\n";

    Matrix<double> matrix(3,3);

    Matrix<double> solution(3,3);
    Matrix<double> check_matrix(3,3);

    matrix.set_column(0, {1, 2, 3});
    matrix.set_column(1, {4, 5, 6});
    matrix.set_column(2, {7, 8, 9});

    solution.set_column(0, {7, 8, 9});
    solution.set_column(1, {4, 5, 6});
    solution.set_column(2, {1, 2, 3});

    check_matrix = matrix.calculate_reverse_columns();

    assert_true(check_matrix == solution, LOG);
}


void MatrixTest::test_compare_rows()
{
    cout << "test_compare_rows\n";

    Matrix<double> matrix(3,3);

    Matrix<double> matrix2(3,3);
    Matrix<double> matrix3(3,3);

    matrix.set_column(0, {1, 2, 3});
    matrix.set_column(1, {4, 5, 6});
    matrix.set_column(2, {7, 8, 9});

    matrix2.set_row(0, {7, 8, 9});
    matrix2.set_row(1, {4, 5, 6});
    matrix2.set_row(2, {1, 2, 3});

    matrix3.set_column(0, {1, 2, 3});
    matrix3.set_column(1, {4, 5, 6});
    matrix3.set_column(2, {7, 8, 9});

    assert_false(matrix.compare_rows(0, matrix2, 0), LOG);
    assert_true(matrix.compare_rows(1, matrix3, 1), LOG);
}


void MatrixTest::test_dot_vector()
{
   cout << "test_dot_vector\n";

   Matrix<double> matrix;
   Vector<double> vector1;

   Vector<double> vector2;

   // Test

   matrix.set(2, 2, 0.0);
   vector1.set(2, 0.0);

   vector2 = dot(matrix, vector1);

   assert_true(vector2 == 0.0, LOG);

   // Test

   matrix.set(2, 2, 1.0);
   vector1.set(2, 1.0);

   vector2 = dot(matrix, vector1);

   assert_true(vector2 == 2.0, LOG);

   // Test

   matrix.set(2, 5);
   matrix.randomize_normal();

   vector1.set(5);
   vector1.randomize_normal();

   vector2 = dot(matrix, vector1);

   assert_true(absolute_value(vector2 - dot(matrix, vector1)) < 1.0e-3, LOG);

   // Test

   matrix.set(2, 2);
   matrix(0,0) = 1.0;
   matrix(0,1) = 2.0;
   matrix(1,0) = 3.0;
   matrix(1,1) = 4.0;

   vector1.set(2);
   vector1[0] = -1.0;
   vector1[1] = 1.0;

   vector2 = dot(matrix, vector1);

   assert_true(vector2 == 1.0, LOG);
}


void MatrixTest::test_dot_matrix()
{
   cout << "test_dot_matrix\n";

   Matrix<double> matrix1;
   Matrix<double> matrix2;
   Matrix<double> matrix3;

   // Test

   matrix1.set(2, 2, 0.0);
   matrix2.set(2, 2, 0.0);

   matrix3 = dot(matrix1,matrix2);

   assert_true(matrix3 == 0.0, LOG);

   // Test

   matrix1.set(2, 2, 1.0);
   matrix2.set(2, 2, 1.0);

   matrix3 = dot(matrix1, matrix2);

   assert_true(matrix3 == 2.0, LOG);

   // Test

   matrix1.set(2, 2);
   matrix1(0,0) = 1.0;
   matrix1(0,1) = 2.0;
   matrix1(1,0) = 3.0;
   matrix1(1,1) = 4.0;

   matrix2 = matrix1;

   matrix3 = dot(matrix1, matrix2);

   assert_true(matrix3(0,0) == 7.0, LOG);
   assert_true(matrix3(0,1) == 10.0, LOG);
   assert_true(matrix3(1,0) == 15.0, LOG);
   assert_true(matrix3(1,1) == 22.0, LOG);

   // Test

   matrix1.set(3, 2);
   matrix1.randomize_normal();

   matrix2.set(2, 3);
   matrix2.randomize_normal();

   matrix3 = dot(matrix1, matrix2);

   assert_true(absolute_value(matrix3 - dot(matrix1, matrix2)) < 1.0e-3, LOG);
}


void MatrixTest::test_eigenvalues()
{
    cout << "test_eigenvalues\n";

    Matrix<double> matrix1;

    Matrix<double> matrix2;

    // Test

    matrix2.set(10,10);

    matrix2.randomize_normal();

    matrix1 = OpenNN::eigenvalues(matrix2);

    assert_true(matrix1.size() == 10, LOG);

    // Test

    matrix2.set_identity(20);

    matrix1 = OpenNN::eigenvalues(matrix2);

    assert_true(matrix1.size() == 20, LOG);
    assert_true(matrix1.get_column(0).is_constant(1.0), LOG);
}


void MatrixTest::test_eigenvectors()
{
    cout << "test_eigenvectors\n";

    Matrix<double> matrix1;

    Matrix<double> matrix2;

    // Test

    matrix2.set(10,10);

    matrix2.randomize_normal();

    matrix1 = OpenNN::eigenvectors(matrix2);

    assert_true(matrix1.get_rows_number() == 10, LOG);
    assert_true(matrix1.get_columns_number() == 10, LOG);
}


void MatrixTest::test_direct()
{
   cout << "test_direct\n";

   Matrix<double> matrix1;
   Matrix<double> matrix2;
   Matrix<double> matrix3;

   // Test

   matrix1.set(2,2);
   matrix1(0,0) = 1;
   matrix1(0,1) = 2;
   matrix1(1,0) = 3;
   matrix1(1,1) = 4;

   matrix2.set(2,2);
   matrix2(0,0) = 0;
   matrix2(0,1) = 5;
   matrix2(1,0) = 6;
   matrix2(1,1) = 7;

   matrix3 = OpenNN::direct(matrix1, matrix2);

   assert_true(matrix3.get_rows_number() == 4, LOG);
   assert_true(matrix3.get_columns_number() == 4, LOG);
   assert_true(abs(matrix3(0,0) - 0) <= 10e-3, LOG);
   assert_true(abs(matrix3(3,3) - 28) <= 10e-3, LOG);
}


void MatrixTest::test_determinant()
{
   cout << "test_determinant\n";

   Matrix<double> matrix(1, 1, 1);

   assert_true(abs(determinant(matrix) - 1) <= 10e-3, LOG);

   matrix.set(2, 2);

   matrix(0,0) = 1;
   matrix(0,1) = 2;

   matrix(1,0) = 3;
   matrix(1,1) = 4;

   assert_true(abs(determinant(matrix) + 2) <= 10e-3, LOG);

   matrix.set(3, 3);

   matrix(0,0) = 1;
   matrix(0,1) = 4;
   matrix(0,2) = 7;

   matrix(1,0) = 2;
   matrix(1,1) = 5;
   matrix(1,2) = 8;

   matrix(2,0) = 3;
   matrix(2,1) = 6;
   matrix(2,2) = 9;

   assert_true(determinant(matrix) == 0.0, LOG);

   matrix.set(4, 4);

   matrix(0,0) = 1;
   matrix(0,1) = 2;
   matrix(0,2) = 3;
   matrix(0,3) = 4;

   matrix(1,0) = 5;
   matrix(1,1) = 6;
   matrix(1,2) = 7;
   matrix(1,3) = 8;

   matrix(2,0) = 9;
   matrix(2,1) = 10;
   matrix(2,2) = 11;
   matrix(2,3) = 12;

   matrix(3,0) = 13;
   matrix(3,1) = 14;
   matrix(3,2) = 15;
   matrix(3,3) = 16;

   assert_true(abs(determinant(matrix)) <= 10e-3, LOG);

}


void MatrixTest::test_calculate_transpose()
{
   cout << "test_calculate_transpose\n";

   Matrix<size_t> matrix(2,2);
   Matrix<size_t> solution(2,2);


   matrix.set_column(0, {1, 2});
   matrix.set_column(1, {3, 4});

   solution.set_column(0, {1, 3});
   solution.set_column(1, {2, 4});

   Matrix<size_t> transpose = matrix.calculate_transpose();

   assert_true(transpose == solution, LOG);
}


void MatrixTest::test_cofactor()
{
   cout << "test_cofactor\n";
}


void MatrixTest::test_calculate_inverse()
{
   cout << "test_calculate_inverse\n";

   Matrix<double> matrix1;
   Matrix<double> matrix2;

   // Test

   matrix1.set(1, 1, 1.0);

   assert_true(OpenNN::inverse(matrix1) == 1.0, LOG);

   // Test

   matrix1.set(2, 2);

   matrix1(0,0) = 1.0;
   matrix1(0,1) = 2.0;

   matrix1(1,0) = 3.0;
   matrix1(1,1) = 4.0;

   matrix2 = OpenNN::inverse(matrix1);

   assert_true(matrix2.get_rows_number() == 2, LOG);
   assert_true(abs(matrix2(0,0) + 2.0) <= 10e-3, LOG);
   assert_true(abs(matrix2(0,1) - 1.0) <= 10e-3, LOG);
   assert_true(abs(matrix2(1,0) - 3.0/2.0) <= 10e-3, LOG);
   assert_true(abs(matrix2(1,1) + 1.0/2.0) <= 10e-3, LOG);

   // Test

   matrix1.set(3, 3);

   matrix1(0,0) = 24.0;
   matrix1(0,1) = -12.0;
   matrix1(0,2) = -2.0;

   matrix1(1,0) = 5.0;
   matrix1(1,1) = 3.0;
   matrix1(1,2) = -5.0;

   matrix1(2,0) = -4.0;
   matrix1(2,1) = 2.0;
   matrix1(2,2) = 4.0;

   matrix2 = OpenNN::inverse(matrix1);

   assert_true(matrix2.get_rows_number() == 3, LOG);

   matrix1.set(4, 4);

   matrix1(0,0) = 1.0;
   matrix1(0,1) = -2.0;
   matrix1(0,2) = 3.0;
   matrix1(0,3) = -4.0;

   matrix1(1,0) = 5.0;
   matrix1(1,1) = 6.0;
   matrix1(1,2) = 7.0;
   matrix1(1,3) = 8.0;

   matrix1(2,0) = 9.0;
   matrix1(2,1) = 10.0;
   matrix1(2,2) = 11.0;
   matrix1(2,3) = 12.0;

   matrix1(3,0) = -13.0;
   matrix1(3,1) = 14.0;
   matrix1(3,2) = -15.0;
   matrix1(3,3) = 16.0;

   matrix2 = OpenNN::inverse(matrix1);

   assert_true(matrix2.get_rows_number() == 4, LOG);
}


void MatrixTest::test_matrix_to_string()
{
    cout << "test_matrix_to_string\n";


    Matrix<size_t> matrix(2,2);
    Matrix<string> solution(2,2);

    matrix.set_row(0, {1, 2});
    matrix.set_row(1, {3, 4});

    matrix.matrix_to_string(',');


}


void MatrixTest::test_is_antisymmetric()
{
   cout << "test_is_antisymmetric\n";

   Matrix<int> matrix;

   // Test

   matrix.set(1, 1, 0);

   assert_true(matrix.is_antisymmetric() == true, LOG);

   // Test

   matrix.set(2, 2, 1);

   assert_true(matrix.is_antisymmetric() == false, LOG);

   // Test

   matrix.set(2, 2, 1);

   matrix(0,0) = 0;
   matrix(0,1) = -2;
   matrix(1,0) = 2;
   matrix(1,1) = 0;

   assert_true(matrix.is_antisymmetric() == true, LOG);
}


void MatrixTest::test_get_tensor()
{
    cout << "test_get_tensor\n";

    Matrix<int> matrix;

    Tensor<int> tensor;

    Vector<size_t> rows_indices;
    Vector<size_t> columns_indices;
    Vector<size_t> columns_dimensions;

    // Test

    tensor = matrix.get_tensor(rows_indices, columns_indices, columns_dimensions);

    assert_true(tensor.empty(), LOG);

    // Test

    matrix.set(2,3);
    matrix.initialize_sequential();

    rows_indices.set(Vector<size_t>({0,1}));
    columns_indices.set(Vector<size_t>({0,1,2}));
    columns_dimensions.set(1, 3);

    tensor = matrix.get_tensor(rows_indices, columns_indices, columns_dimensions);

    assert_true(tensor.get_dimension(0) == 2, LOG);
    assert_true(tensor.get_dimension(1) == 3, LOG);
}


void MatrixTest::test_save()
{
   cout << "test_save\n";

   string file_name = "../data/matrix.dat";

   Matrix<int> matrix;

   matrix.save_csv(file_name);
}


void MatrixTest::test_load()
{
   cout << "test_load\n";

   string file_name = "../data/matrix.dat";

   Matrix<int> matrix;

   // Test

   matrix.set();

   matrix.save_csv(file_name);
   matrix.load_csv(file_name);

   assert_true(matrix.get_rows_number() == 0, LOG);
   assert_true(matrix.get_columns_number() == 0, LOG);

   // Test

   matrix.set(1, 2, 3);

   matrix.save_csv(file_name);
   matrix.load_csv(file_name);

   assert_true(matrix.get_rows_number() == 1, LOG);
   assert_true(matrix.get_columns_number() == 2, LOG);
   assert_true(matrix == 3, LOG);

   // Test

   matrix.set(2, 1, 1);

   matrix.save_csv(file_name);
   matrix.load_csv(file_name);

   assert_true(matrix.get_rows_number() == 2, LOG);
   assert_true(matrix.get_columns_number() == 1, LOG);

   // Test

   matrix.set(4, 4, 0);

   matrix.save_csv(file_name);
   matrix.load_csv(file_name);

   assert_true(matrix.get_rows_number() == 4, LOG);
   assert_true(matrix.get_columns_number() == 4, LOG);
   assert_true(matrix == 0, LOG);

   // Test

   matrix.set(1, 1, -99);

   matrix.save_csv(file_name);
   matrix.load_csv(file_name);

   assert_true(matrix.get_rows_number() == 1, LOG);
   assert_true(matrix.get_columns_number() == 1, LOG);
   assert_true(matrix == -99, LOG);

   // Test

   matrix.set(3, 2);

   matrix(0,0) = 3; matrix(0,1) = 5;
   matrix(1,0) = 7; matrix(1,1) = 9;
   matrix(2,0) = 2; matrix(2,1) = 4;

   matrix.save_csv(file_name);
   matrix.load_csv(file_name);

   assert_true(matrix(0,0) == 3, LOG); assert_true(matrix(0,1) == 5, LOG);
   assert_true(matrix(1,0) == 7, LOG); assert_true(matrix(1,1) == 9, LOG);
   assert_true(matrix(2,0) == 2, LOG); assert_true(matrix(2,1) == 4, LOG);
}


void MatrixTest::test_parse()
{
    cout << "test_parse\n";

    Matrix<int> matrix;
    string string;

    // Test

    string = "";

    matrix.parse(string);

    assert_true(matrix.get_rows_number() == 0, LOG);
    assert_true(matrix.get_columns_number() == 0, LOG);

    // Test

    string =
    "1 2 3\n"
    "4 5 6\n";

    matrix.parse(string);

    assert_true(matrix.get_rows_number() == 2, LOG);
    assert_true(matrix.get_columns_number() == 3, LOG);

    // Test

    string =
    "1 2\n"
    "3 4\n"
    "5 6\n";

    matrix.parse(string);

    assert_true(matrix.get_rows_number() == 3, LOG);
    assert_true(matrix.get_columns_number() == 2, LOG);
}


void MatrixTest::run_test_case()
{
   cout << "Running matrix test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Assignment operators methods

   test_assignment_operator();   

   // Arithmetic operators

   test_sum_operator();
   test_rest_operator();
   test_multiplication_operator();
   test_division_operator();

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

   test_get_header();

   test_get_column_index();
   test_get_columns_indices();
   test_get_binary_columns_indices();

   test_get_row();
   test_get_rows();
   test_get_column();
   test_get_columns();

   test_get_submatrix();
   test_get_submatrix_rows();
   test_get_submatrix_columns();

   test_get_first();
   test_get_last();

   test_get_constant_columns_indices();

   test_get_first_rows();
   test_get_last_rows();
   test_get_first_columns();
   test_get_last_columns();

   // Set methods

   test_set();

   test_set_identity();

   test_set_rows_number();
   test_set_columns_number();

   test_set_row();
   test_set_column();

   test_set_header();

   test_set_submatrix_rows();

   //Check methods

   test_empty();

   test_is_square();

   test_is_symmetric();
   test_is_antisymmetric();

   test_is_diagonal();

   test_is_scalar();

   test_is_identity();

   test_is_binary();
   test_is_column_binary();

   test_is_column_constant();

   test_is_positive();

   test_is_row_equal_to();

   test_has_column_value();

   //Count methods

   test_count_diagonal_elements();
   test_count_off_diagonal_elements();

   test_count_equal_to();
   test_count_not_equal_to();

   test_count_equal_to_by_rows();

   test_count_rows_equal_to();
   test_count_rows_not_equal_to();

   test_count_nan_rows();
   test_count_nan_columns();

   //Not a number

   test_has_nan();

   test_count_nan();

   test_count_rows_with_nan();
   test_count_columns_with_nan();

   //Filter

   test_filter_column_equal_to();
   test_filter_column_not_equal_to();

   test_filter_column_less_than();
   test_filter_column_greater_than();

   test_filter_column_minimum_maximum();

   // Initialize

   test_initialize();

   // Diagonal methods

   test_get_diagonal();
   test_set_diagonal();
   test_sum_diagonal();

   // Resize methods

   test_append_row();
   test_append_column();

   test_insert_row();
   test_insert_column();

   test_insert_row_values();

   test_add_column();
   test_add_column_first();

   test_swap_columns();

   test_delete_row();
   test_delete_column();

   test_delete_rows_wiht_value();
   test_delete_columns_with_value();

   test_delete_first_rows();
   test_delete_last_rows();

   test_delete_first_columns();
   test_delete_last_columns();

   test_delete_columns_name_contain();

   test_delete_constant_rows();
   test_delete_constant_columns();

   test_delete_binary_columns();

   test_assembla_rows();
   test_assembla_columns();

   test_subtract_row();

   //Sorting case

   test_sort_ascending();
   test_sort_descending();

   //Replace

   test_replace();

   test_replace_header();

   test_replace_in_row();
   test_replace_in_column();

   test_replace_substring();

   test_replace_contains();
   test_replace_contains_in_row();

   test_replace_column_equal_to();
   test_replace_column_not_equal_to();

   test_replace_column_contain();

   // Initialization methods

   test_initialize();

   test_randomize_uniform();
   test_randomize_normal();

   test_initialize_identity();
   test_initialize_diagonal();

   test_append_header();

   test_tuck_in();

   test_set_to_identity();

   // Mathematical methods

   test_calculate_sum();
   test_calculate_rows_sum();
   test_calculate_columns_sum();

   test_sum_row();
   test_sum_rows();

   test_substract_rows();
   test_multiply_rows();
   test_divide_rows();

   test_calculate_trace();

   test_calculate_reverse_columns();

   test_compare_rows();

   test_dot_vector();
   test_dot_matrix();

   test_eigenvalues();
   test_eigenvectors();

   test_direct();

   test_determinant();
   test_calculate_transpose();
   test_cofactor();
   test_calculate_inverse();

   //CONVERSIONS

   test_matrix_to_string();
 
   // Unscaling methods

   test_get_tensor();

   // Serialization methods

   test_load();

   test_save();

   test_parse();

   cout << "End of matrix test case.\n";
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

    return product;
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

    return product;
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
