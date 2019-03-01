/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   V E C T O R   T E S T   C L A S S                                                                          */
/*                                                                                                              */ 
 
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "vector_test.h"

// GENERAL CONSTRUCTOR

VectorTest::VectorTest() : UnitTesting() 
{   
}


// DESTRUCTOR

VectorTest::~VectorTest()
{
}


// METHODS

void VectorTest::test_constructor()
{
       message += "test_constructor\n";

       string file_name = "../data/vector.dat";

       // Default

       Vector<bool> v1;

       assert_true(v1.size() == 0, LOG);

       // Size

       Vector<bool> v2(1);

       assert_true(v2.size() == 1, LOG);

       // Size initialization

       Vector<bool> v3(1, false);

       assert_true(v3.size() == 1, LOG);
       assert_true(v3[0] == false, LOG);

       // File

       Vector<int> v4(3, 0);
       v4.save(file_name);

       Vector<int> w4(file_name);

       assert_true(w4.size() == 3, LOG);
       assert_true(w4 == 0, LOG);

       // Sequential

       Vector<int> v6(10, 5, 50);

       assert_true(v6.size() == 9, LOG);
       assert_true(v6[0] == 10, LOG);
       assert_true(v6[8] == 50, LOG);

       Vector<double> v7(3.0, 0.2, 3.8);

       assert_true(v7.size() == 5, LOG);
       assert_true(v7[0] == 3.0, LOG);
       assert_true(fabs(v7[4] - 3.8) < numeric_limits<double>::epsilon(), LOG);

       Vector<int> v8(9, -1, 1);

       assert_true(v8.size() == 9, LOG);
       assert_true(v8[0] == 9, LOG);
       assert_true(v8[8] == 1, LOG);

       // Copy

       Vector<string> v5(1, "hello");

       Vector<string> w5(v5);

       assert_true(w5.size() == 1, LOG);
       assert_true(w5[0] == "hello", LOG);
}


void VectorTest::test_destructor()
{
}


void VectorTest::test_sum_operator()
{
   message += "test_sum_operator\n";

   Vector<int> a, b, c, d;

   // Scalar

   a.set(1, 1);
   b =  a + 1;

   c.set(1, 2);
   
   assert_true(b == c, LOG);

   // Sum

   a.set(1, 1);
   b.set(1, 1);

   c = a + b;

   d.set(1, 2);

   assert_true(c == d, LOG);
}


void VectorTest::test_rest_operator()
{
   message += "test_rest_operator\n";

   Vector<double> a, b, c, d;

   // Scalar

   a.set(1, 1.0);
   b =  a - 1.0;

   c.set(1, 0.0);
   
   assert_true(b == c, LOG);

   // Vector

   a.set(1, 1.0);
   b.set(1, 1.0);

   c = a - b;

   d.set(1, 0.0);

   assert_true(c == d, LOG);
}


void VectorTest::test_multiplication_operator()
{
   message += "test_multiplication_operator\n";

   Vector<double> a, b, c, d;

   // Scalar

   a.set(1, 1.0);
   b =  a*2.0;

   c.set(1, 2.0);
   
   assert_true(b == c, LOG);

   // Vector

   a.set(1, 1.0);
   b.set(1, 1.0);

   c = a*b;

   d.set(1, 1.0);

   assert_true(c == d, LOG);

   // Matrix

   Matrix<double> m(1, 1, 0.0);

   a.set(1, 0.0);

   Matrix<double> p = a*m;

   assert_true(p.get_rows_number() == 1, LOG);
   assert_true(p.get_columns_number() == 1, LOG);
   assert_true(p == 0.0, LOG);

   m.set(3, 2, 1.0);
   a.set(3, 1.0);

   p = a*m;

   assert_true(p.get_rows_number() == 3, LOG);
   assert_true(p.get_columns_number() == 2, LOG);
   assert_true(p == 1.0, LOG);
}


void VectorTest::test_division_operator()
{
   message += "test_division_operator\n";

   Vector<double> a, b, c, d;

   // Scalar

   a.set(1, 1.0);
   b =  a/2.0;

   c.set(1, 0.5);
   
   assert_true(b == c, LOG);

   // Vector

   a.set(1, 2.0);
   b.set(1, 2.0);

   c = a/b;

   d.set(1, 1.0);

   assert_true(c == d, LOG);
}


void VectorTest::test_sum_assignment_operator()
{
   message += "test_sum_assignment_operator\n";

   Vector<int> a, b;

   // Scalar

   a.set(2, 1);

   a += 1;

   assert_true(a == 2, LOG);

   // Vector

   a.set(2, 1);
   b.set(2, 1);

   a += b;

   assert_true(a == 2, LOG);
}


void VectorTest::test_rest_assignment_operator()
{
   message += "test_rest_assignment_operator\n";

   Vector<int> a, b;

   // Scalar

   a.set(2, 1);

   a -= 1;

   assert_true(a == 0, LOG);

   // Vector

   a.set(2, 1);
   b.set(2, 1);

   a -= b;

   assert_true(a == 0, LOG);
}


void VectorTest::test_multiplication_assignment_operator()
{
   message += "test_multiplication_assignment_operator\n";

   Vector<int> a, b;

   // Scalar

   a.set(2, 2);

   a *= 1;

   assert_true(a == 2, LOG);

   // Vector

   a.set(2, 2);
   b.set(2, 1);

   a *= b;

   assert_true(a == 2, LOG);
}


void VectorTest::test_division_assignment_operator()
{
   message += "test_division_assignment_operator\n";

   Vector<int> a, b;

   // Scalar

   a.set(2, 2);

   a /= 2;

   assert_true(a == 1, LOG);

   // Vector

   a.set(2, 2);
   b.set(2, 2);

   a /= b;

   assert_true(a == 1, LOG);
}


void VectorTest::test_equal_to_operator()
{
   message += "test_equal_to_operator\n";

   Vector<int> a(2);
   a[0] = 0;
   a[1] = 1;

   Vector<int> b(2);
   b[0] = 0;
   b[1] = 1;

   Vector<int> c(2, -1);

   assert_true(a == b, LOG);
   assert_true(c == -1, LOG);
}


void VectorTest::test_not_equal_to_operator()
{
   message += "test_not_equal_to_operator\n";

   Vector<double> a(2, -1.0);
   Vector<double> b(2, 1.0);

   assert_true(a != b, LOG);
   assert_true(a != 0.0, LOG);
   assert_true(b != 0.0, LOG);
}


void VectorTest::test_greater_than_operator()
{
   message += "test_greater_than_operator\n";

   Vector<int> a(2);
   a[0] = 1;
   a[1] = 2;

   Vector<int> b(2);
   b[0] = 0;
   b[1] = 1;

   assert_true(a > b, LOG);

   assert_true(a > 0, LOG);
   assert_false(a > 1, LOG);

   assert_true(b > -1, LOG);
   assert_false(b > 0, LOG);
}


void VectorTest::test_less_than_operator()
{
   message += "test_less_than_operator\n";

   Vector<double> a(2);
   a[0] = 0.0;
   a[1] = 1.0;

   Vector<double> b(2);
   b[0] = 1.0;
   b[1] = 2.0;

   assert_true(a < b, LOG);

   assert_true(a < 2.0, LOG);
   assert_false(a < 1.0, LOG);

   assert_true(b < 3.0, LOG);
   assert_false(b < 1.0, LOG);
}


void VectorTest::test_greater_than_or_equal_to_operator()
{
   message += "test_greater_than_or_equal_to_operator\n";

   Vector<int> a(2);
   a[0] = 1;
   a[1] = 2;

   Vector<int> b(2);
   b[0] = 1;
   b[1] = 1;

   assert_true(a >= b, LOG);

   assert_true(a >= 1, LOG);
   assert_false(a >= 2, LOG);

   assert_true(b >= 1, LOG);
   assert_false(b >= 2, LOG);
}


void VectorTest::test_less_than_or_equal_to_operator()
{
   message += "test_less_than_or_equal_to_operator\n";

   Vector<double> a(2);
   a[0] = 1.0;
   a[1] = 1.0;

   Vector<double> b(2);
   b[0] = 1.0;
   b[1] = 2.0;

   assert_true(a <= b, LOG);

   assert_true(a <= 1.0, LOG);
   assert_false(a <= 0.0, LOG);

   assert_true(b <= 2.0, LOG);
   assert_false(b <= 1.0, LOG);
}


void VectorTest::test_output_operator()
{
   message += "test_output_operator\n";

   Vector< Vector<double> > w;
   Vector< Matrix<size_t> > x;
   Vector<double> v;
   Matrix<size_t> m;

   // Test

   w.set(2);
   w[0].set(2, 0.0);
   w[1].set(2, 1.0);

   v.set(2, 1.0);

   assert_true(w[1] - v == w[0], LOG);

   // Test

   x.set(2);
   x[0].set(2, 3, false);
   x[1].set(3, 4, true);

   m.set(2, 3, false);

   assert_true(x[0] == m, LOG);
}


void VectorTest::test_get_size()
{
   message += "test_get_size\n";

   Vector<int> v;

   assert_true(v.size() == 0, LOG);

   v.set(1);

   assert_true(v.size() == 1, LOG);

   v.set(0);

   assert_true(v.size() == 0, LOG);
}


void VectorTest::test_get_display()
{
   message += "test_get_display\n";
}


void VectorTest::test_get_subvector_random()
{
    message += "test_get_subvector_random\n";

    Vector<double> a(2, 0.0);

    assert_true(a.get_subvector_random(2) == a, LOG);
    assert_true(a.get_subvector_random(1).size() == 1, LOG);
    assert_true(a.get_subvector_random(1)[0] == 0.0, LOG);

    Vector<int> b(1,1,10);

    assert_true(b.get_subvector_random(5) >= 1, LOG);
    assert_true(b.get_subvector_random(5) <= 10, LOG);
}


void VectorTest::test_set()
{
   message += "test_set\n";

   string file_name = "../data/vector.dat";

   Vector<int> v(3, 0);

   // Default

   v.set();

   assert_true(v.size() == 0, LOG);

   // Size

   v.set(1);

   assert_true(v.size() == 1, LOG);

   // Size initialization

   v.set(1, 0);

   assert_true(v.size() == 1, LOG);
   assert_true(v == 0, LOG);

   // File

   v.save(file_name);
   v.set(file_name);

   assert_true(v.size() == 1, LOG);
   assert_true(v == 0, LOG);

   // Sequential

   v.set(10, 5, 50);

   assert_true(v.size() == 9, LOG);
   assert_true(v[0] == 10, LOG);
   assert_true(v[8] == 50, LOG);

   v.set(9, -1, 1);

   assert_true(v.size() == 9, LOG);
   assert_true(v[0] == 9, LOG);
   assert_true(v[8] == 1, LOG);

   // Copy

   v.set(1, 0);
   v.set(v);

   assert_true(v.size() == 1, LOG);
   assert_true(v == 0, LOG);
}


void VectorTest::test_set_display()
{
   message += "test_set_display\n";
}


void VectorTest::test_resize()
{
   message += "test_resize\n";

   Vector<int> a(1, 0);

   // Decrease size

   a.resize(2);

   assert_true(a.size() == 2, LOG);

   // Increase size

   a.resize(0);

   assert_true(a.size() == 0, LOG);
}


void VectorTest::test_initialize()
{
   message += "test_initialize\n";

   Vector<int> v(2);

   v.initialize(0);

   Vector<int> w(2, 0);
   
   assert_true(v == w, LOG);
}


void VectorTest::test_initialize_sequential()
{
   message += "test_initialize_sequential\n";

   Vector<double> v(2);

   v.initialize_sequential();

   Vector<double> w(2);
   w[0] = 0.0;
   w[1] = 1.0;
   
   assert_true(v == w, LOG);
}


void VectorTest::test_randomize_uniform()
{
   message += "test_randomize_uniform\n";

   Vector<double> v(3);

   v.randomize_uniform();

   assert_true(v >= -1.0, LOG);
   assert_true(v <=  1.0, LOG);
  
   v.randomize_uniform(0.0, 2.0);
   
   assert_true(v >= 0.0, LOG);
   assert_true(v <= 2.0, LOG);
}


void VectorTest::test_randomize_normal()
{
   message += "test_randomize_normal\n";

   Vector<double> v(2);

   v.randomize_normal();

   v.randomize_normal(0.0, 0.0);

   assert_true(v == 0.0, LOG);
}


void VectorTest::test_contains()
{
   message += "test_contains\n";

   Vector<int> v;
   Vector<int> v1;

   // Test

   assert_true(v.contains(0) == false, LOG);

   //Test

   v.set(5, -1);

   assert_true(v.contains(0) == false, LOG);

   //Test

   v.set(5, -1);
   v[3] = 1;
   v1.set(3, 1);

   assert_true(v.contains(v1) == true, LOG);

   //Test

   v.set(10, -1);

   assert_true(v.contains(0) == false, LOG);
}


void VectorTest::test_is_in()
{
   message += "test_is_in\n";

   Vector<size_t> v(5, 0);

   assert_true(v.is_in(0, 0), LOG);
}


void VectorTest::test_is_constant()
{
   message += "test_is_constant\n";

   Vector<int> v(3,2);
   Vector<double> w(1,0.5,2.5);

   assert_true(v.is_constant(1), LOG);
   assert_true(w.is_constant(2), LOG);
   assert_false(w.is_constant(1), LOG);
}


void VectorTest::test_is_crescent()
{
   message += "test_is_crescent\n";

   Vector<double> v(1,0.5,2.5);
   Vector<double> w(3);

   w[0]=1;
   w[1]=2;
   w[2]=2;

   assert_true(v.is_crescent(), LOG);
   assert_false(w.is_crescent(), LOG);
}


void VectorTest::test_is_decrescent()
{
   message += "test_is_decrescent\n";

   Vector<double> v(2.5,-0.5,1);
   Vector<double> w(3);

   w[0]=2;
   w[1]=1;
   w[2]=1;

   assert_true(v.is_decrescent(), LOG);
   assert_false(w.is_decrescent(), LOG);
}


void VectorTest::test_impute_time_series_missing_values_mean()
{
    message += "test_impute_time_series_missing_values_mean\n";

    Vector<double> v;

    Vector<double> w;

    // Test

    v.set(1, 3.141592);
    w = v.impute_time_series_missing_values_mean(999.0);

    assert_true(w == v, LOG);

    // Test

    v.set(1, 999.0);
    w = v.impute_time_series_missing_values_mean(999.0);

    assert_true(w == v, LOG);

    // Test

    v.set(3, 1.0);
    v[0] = 0.0;
    v[1] = 999.0;
    v[2] = 2.0;

    w = v.impute_time_series_missing_values_mean(999.0);

    assert_true(w[1] == 1.0, LOG);

    // Test

    v.set(3);
    v[0] = 999.0;
    v[1] = 999.0;
    v[2] = 2.0;

    w = v.impute_time_series_missing_values_mean(999.0);

    assert_true(w[0] == 2.0, LOG);
    assert_true(w[1] == 2.0, LOG);

    // Test

    v.set(3);
    v[0] = 2.0;
    v[1] = 999.0;
    v[2] = 999.0;

    w = v.impute_time_series_missing_values_mean(999.0);

    assert_true(w[1] == 2.0, LOG);
    assert_true(w[2] == 2.0, LOG);

    v.set(3);
    v[0] = 999.0;
    v[1] = 999.0;
    v[2] = 999.0;

    w = v.impute_time_series_missing_values_mean(999.0);

    assert_true(w == 999.0, LOG);
}


void VectorTest::test_calculate_sum()
{
   message += "test_calculate_sum\n";

   Vector<int> v;

   assert_true(v.calculate_sum() == 0, LOG);

   v.set(2);
   v.initialize(1);

   assert_true(v.calculate_sum() == 2, LOG);
}


void VectorTest::test_calculate_partial_sum()
{
    message += "test_calculate_partial_sum\n";

    Vector<size_t> v(5, 1);

    // Test

    Vector<size_t> indices(1, 0);

    assert_true(v.calculate_partial_sum(indices) == 1, LOG);

    // Test

    indices.set(2);

    v[4] = 8;

    indices[0] = 0;
    indices[1] = 4;

    assert_true(v.calculate_partial_sum(indices) == 9, LOG);
}


void VectorTest::test_calculate_product()
{
   message += "test_calculate_product\n";

   Vector<double> v;
   Vector<int> w;

   assert_true(v.calculate_product() == 1.0, LOG);

   v.set(2);
   v[0] = 0.5;
   v[1] = 1.5;

   w.set(3,1,6);


   assert_true(v.calculate_product() == 0.75, LOG);
   assert_true(w.calculate_product() == 360, LOG);
}


void VectorTest::test_calculate_mean()
{
   message += "test_calculate_mean\n";
   
   Vector<double> v(1, 1.0);

   assert_true(v.calculate_mean() == 1.0, LOG);

   v.set(2);
   v[0] = -1.0;
   v[1] =  1.0;

   assert_true(v.calculate_mean() == 0.0, LOG);
}


void VectorTest::test_calculate_standard_deviation()
{
   message += "test_calculate_standard_deviation\n";
   
   Vector<double> v;

   double standard_deviation;

   // Test

   v.set(1, 1.0);

   standard_deviation = v.calculate_standard_deviation();

   assert_true(standard_deviation == 0.0, LOG);

   // Test

   v.set(2);
   v[0] = -1.0;
   v[1] =  1.0;

   standard_deviation = v.calculate_standard_deviation();

   assert_true(fabs(standard_deviation-1.4142) < 1.0e-3, LOG);
}

void VectorTest::test_calculate_covariance()
{
    message += "test_calculate_covariance\n";

    Vector<double> v1;
    Vector<double> v2;
    Vector<double> v3;

    // Test

    v1.set(10);
    v2.set(10);
    v3.set(10);

    v1.randomize_normal();
    v2.randomize_normal();
    v3.randomize_normal();

    assert_true(fabs(v1.calculate_covariance(v1)-v1.calculate_variance()) < 1.0e-3, LOG);
    assert_true(fabs(v2.calculate_covariance(v2)-v2.calculate_variance()) < 1.0e-3, LOG);
    assert_true(fabs(v3.calculate_covariance(v3)-v3.calculate_variance()) < 1.0e-3, LOG);
}

   
void VectorTest::test_calculate_mean_standard_deviation()
{
   message += "test_calculate_mean_standard_deviation\n";

   Vector<double> v;
   Vector<double> mean_standard_deviation;

   // Test

   v.set(2);
   v[0] = -1.0;
   v[1] =  1.0;

   mean_standard_deviation = v.calculate_mean_standard_deviation();

   assert_true(mean_standard_deviation[0] == 0.0, LOG);
   assert_true(fabs(mean_standard_deviation[1]-1.4142) < 1.0e-3, LOG);
}


void VectorTest::test_calculate_minimum()
{
   message += "test_calculate_minimum\n";
   
   Vector<int> v(1, 1);

   assert_true(v.calculate_minimum() == 1, LOG);

   v.set(3);
   v[0] = -1;
   v[1] =  0;
   v[2] =  1;

   assert_true(v.calculate_minimum() == -1, LOG);
}


void VectorTest::test_calculate_maximum()
{
   message += "test_calculate_maximum\n";
   
   Vector<double> v(1, 1.0);

   assert_true(v.calculate_maximum() == 1.0, LOG);

   //Test

   v.set(3);
   v[0] = -1.0;
   v[1] =  0.0;
   v[2] =  1.0;

   assert_true(v.calculate_maximum() == 1.0, LOG);
}


void VectorTest::test_calculate_minimum_maximum()
{
   message += "test_calculate_minimum_maximum\n";
   
   Vector<int> v(3);
   v[0] = -1;
   v[1] =  0;
   v[2] =  1;

   Vector<int> minimum_maximum = v.calculate_minimum_maximum();

   assert_true(minimum_maximum[0] == -1, LOG);
   assert_true(minimum_maximum[1] == 1, LOG);
}


void VectorTest::test_calculate_minimum_missing_values()
{
   message += "test_calculate_minimum_missing_values\n";

   Vector<int> v;
   Vector<size_t> missing_values;

   int minimum;

   // Test

   v.set(1, 1);
   missing_values.set();

   minimum = v.calculate_minimum_missing_values(missing_values);

   assert_true(minimum == 1, LOG);

   // test

   v.set(3);
   v[0] = -1;
   v[1] =  0;
   v[2] =  1;

   missing_values.set();

   minimum = v.calculate_minimum_missing_values(missing_values);

   assert_true(minimum == -1, LOG);
}


void VectorTest::test_calculate_maximum_missing_values()
{
   message += "test_calculate_maximum_missing_values\n";

   Vector<int> v;
   Vector<size_t> missing_values;

   int maximum;

   // Test

   v.set(1, 1);
   missing_values.set();

   maximum = v.calculate_maximum_missing_values(missing_values);

   assert_true(maximum == 1, LOG);

   // test

   v.set(3);
   v[0] = -1;
   v[1] =  0;
   v[2] =  1;

   missing_values.set();

   maximum = v.calculate_maximum_missing_values(missing_values);

   assert_true(maximum == 1, LOG);
}


void VectorTest::test_calculate_minimum_maximum_missing_values()
{
   message += "test_calculate_minimum_maximum_missing_values\n";

   Vector<int> v(3);
   v[0] = -1;
   v[1] =  0;
   v[2] =  1;

   Vector<size_t> missing_values;

   Vector<int> minimum_maximum = v.calculate_minimum_maximum_missing_values(missing_values);

   assert_true(minimum_maximum[0] == -1, LOG);
   assert_true(minimum_maximum[1] == 1, LOG);
}


void VectorTest::test_calculate_explained_variance()
{
    message += "test_calculate_explained_variance\n";

    Vector<double> v;

    Vector<double> explained_variance;

    // Test

    v.set(3);

    v[0] = 7.0;
    v[1] = 2.0;
    v[2] = 1.0;

    explained_variance = v.calculate_explained_variance();

    assert_true(explained_variance.size() == 3, LOG);
    assert_true(explained_variance[0] == 70.0, LOG);
    assert_true(explained_variance[1] == 20.0, LOG);
    assert_true(explained_variance[2] == 10.0, LOG);

    // Test

    v.set(100);
    v.randomize_normal();

    explained_variance = v.calculate_explained_variance();

    assert_true(explained_variance.size() == 100, LOG);
    assert_true(explained_variance.calculate_sum() - 100.0 < 1.0e-12, LOG);
}


void VectorTest::test_calculate_statistics()
{
    message += "test_calculate_statistics\n";

    Vector<double> v;
    Statistics<double> statistics;

    // Test

    v.set(2);
    v[0] = -1.0;
    v[1] =  1.0;

    statistics = v.calculate_statistics();

    assert_true(fabs(statistics.minimum - -1.0) < numeric_limits<double>::epsilon(), LOG);
    assert_true(statistics.maximum == 1.0, LOG);
    assert_true(statistics.mean == 0.0, LOG);
    assert_true(fabs(statistics.standard_deviation-1.4142135624) < 1.0e-6 , LOG);
}


void VectorTest::test_calculate_quartiles()
{
   message += "test_calculate_quartiles\n";

   Vector<double> v1(2);
   v1[0] =  0.0;

   Vector<double> quartiles1 = v1.calculate_quartiles();

   assert_true(quartiles1[0] == 0.0, LOG);
   assert_true(quartiles1[1] == 0.0, LOG);
   assert_true(quartiles1[2] == 0.0, LOG);

   Vector<double> v2(2);
   v2[0] =  0.0;
   v2[1] =  1.0;

   Vector<double> quartiles2 = v2.calculate_quartiles();

   assert_true(quartiles2[0] == 0.25, LOG);
   assert_true(quartiles2[1] == 0.5, LOG);
   assert_true(quartiles2[2] == 0.75, LOG);

   Vector<double> v3(3);
   v3[0] =  0.0;
   v3[1] =  1.0;
   v3[2] =  2.0;

   Vector<double> quartiles3 = v3.calculate_quartiles();

   assert_true(quartiles3[0] == 0.5, LOG);
   assert_true(quartiles3[1] == 1.0, LOG);
   assert_true(quartiles3[2] == 1.5, LOG);

   Vector<double> v4(4);
   v4[0] =  0.0;
   v4[1] =  1.0;
   v4[2] =  2.0;
   v4[3] =  3.0;

   Vector<double> quartiles4 = v4.calculate_quartiles();

   assert_true(v4.count_less_equal_to(quartiles4[0])*100.0/v4.size() == 25.0, LOG);
   assert_true(v4.count_less_equal_to(quartiles4[1])*100.0/v4.size() == 50.0, LOG);
   assert_true(v4.count_less_equal_to(quartiles4[2])*100.0/v4.size() == 75.0, LOG);

   Vector<double> v5(5);
   v5[0] =  0.0;
   v5[1] =  1.0;
   v5[2] =  2.0;
   v5[3] =  3.0;
   v5[4] =  4.0;

   Vector<double> quartiles5 = v5.calculate_quartiles();

   assert_true(quartiles5[0] == 1.0, LOG);
   assert_true(quartiles5[1] == 2.0, LOG);
   assert_true(quartiles5[2] == 3.0, LOG);

   Vector<double> v6(6);
   v6[0] =  0.0;
   v6[1] =  1.0;
   v6[2] =  2.0;
   v6[3] =  3.0;
   v6[4] =  4.0;
   v6[5] =  5.0;

   Vector<double> quartiles6 = v6.calculate_quartiles();

   assert_true(quartiles6[0] == 1.0, LOG);
   assert_true(quartiles6[1] == 2.5, LOG);
   assert_true(quartiles6[2] == 4.0, LOG);
}


void VectorTest::test_calculate_histogram()
{
   message += "test_calculate_histogram\n";

   Vector<double> v;

   Histogram<double> histogram;

   Vector<double> centers;
   Vector<size_t> frequencies;

   // Test

   v.set(0.0, 1.0, 9.0);

   histogram = v.calculate_histogram(10);

   assert_true(histogram.get_bins_number() == 10, LOG);

   centers = histogram.centers;
   frequencies = histogram.frequencies;
                                        
   assert_true(fabs(centers[0] - 0.45) < 1.0e-12, LOG);
   assert_true(fabs(centers[1] - 1.35) < 1.0e-12, LOG);
   assert_true(fabs(centers[2] - 2.25) < 1.0e-12, LOG);
   assert_true(fabs(centers[3] - 3.15) < 1.0e-12, LOG);
   assert_true(fabs(centers[4] - 4.05) < 1.0e-12, LOG);
   assert_true(fabs(centers[5] - 4.95) < 1.0e-12, LOG);
   assert_true(fabs(centers[6] - 5.85) < 1.0e-12, LOG);
   assert_true(fabs(centers[7] - 6.75) < 1.0e-12, LOG);
   assert_true(fabs(centers[8] - 7.65) < 1.0e-12, LOG);
   assert_true(fabs(centers[9] - 8.55) < 1.0e-12, LOG);

   assert_true(frequencies[0] == 1, LOG);
   assert_true(frequencies[1] == 1, LOG);
   assert_true(frequencies[2] == 1, LOG);
   assert_true(frequencies[3] == 1, LOG);
   assert_true(frequencies[4] == 1, LOG);
   assert_true(frequencies[5] == 1, LOG);
   assert_true(frequencies[6] == 1, LOG);
   assert_true(frequencies[7] == 1, LOG);
   assert_true(frequencies[8] == 1, LOG);
   assert_true(frequencies[9] == 1, LOG);
   assert_true(histogram.frequencies.calculate_sum() == 10, LOG);

   // Test

   v.set(20);
   v.randomize_normal();

   histogram = v.calculate_histogram(10);

   assert_true(histogram.frequencies.calculate_sum() == 20, LOG);
}


void VectorTest::test_calculate_bin()
{
    message += "test_calculate_bin\n";

    Vector<double> v;

    size_t bin;

    Histogram<double> histogram;

    v.set(0.0, 1.0, 9.0);

    histogram = v.calculate_histogram(10);

    // Test

    bin = histogram.calculate_bin(v[0]);

    assert_true(bin == 0, LOG);

    // Test

    bin = histogram.calculate_bin(v[1]);

    assert_true(bin == 1, LOG);

    // Test

    bin = histogram.calculate_bin(v[2]);

    assert_true(bin == 2, LOG);
}


void VectorTest::test_calculate_frequency()
{
    message += "test_calculate_frequency\n";

    Vector<double> v;

    size_t frequency;

    Histogram<double> histogram;

    // Test

    v.set(0.0, 1.0, 9.0);

    histogram = v.calculate_histogram(10);

    frequency = histogram.calculate_frequency(v[9]);

    assert_true(frequency == 1, LOG);
}


void VectorTest::test_calculate_total_frequencies()
{
    message += "test_calculate_total_frequencies\n";

    Vector<double> v1;
    Vector<double> v2;
    Vector<double> v3;

    Vector<size_t> total_frequencies;

    Vector < Histogram<double> > histograms(2);

    // Test

    v1.set(0.0, 1, 9.0);

    v2.set(5);

    v2[0] = 0.0;
    v2[1] = 2.0;
    v2[2] = 6.0;
    v2[3] = 6.0;
    v2[4] = 9.0;

    v3.set(2);

    v3[0] = 8.0;
    v3[1] = 6.0;

    histograms[0] = v1.calculate_histogram(10);
    histograms[1] = v2.calculate_histogram(10);

    total_frequencies = v3.calculate_total_frequencies(histograms);

    assert_true(total_frequencies[0] == 1, LOG);
    assert_true(total_frequencies[1] == 2, LOG);
}


void VectorTest::test_calculate_minimal_index()
{
   message += "test_calculate_minimal_index\n";
   
   Vector<double> v(1, 1.0);

   assert_true(v.calculate_minimal_index() == 0, LOG);

   v.set(3);
   v[0] =  1.0;
   v[1] =  0.0;
   v[2] = -1.0;

   assert_true(v.calculate_minimal_index() == 2, LOG);
}


void VectorTest::test_calculate_maximal_index()
{
   message += "test_calculate_maximal_index\n";
   
   Vector<int> v(1);

   assert_true(v.calculate_maximal_index() == 0, LOG);

   v.set(3);
   v[0] = -1;
   v[1] =  0;
   v[2] =  1;

   assert_true(v.calculate_maximal_index() == 2, LOG);
}


void VectorTest::test_calculate_minimal_indices()
{
    message += "test_calculate_minimal_indices\n";

    Vector<double> v;
    Vector<size_t> minimal_indices;

    // Test

    v.set();

    minimal_indices = v.calculate_minimal_indices(0);

    assert_true(minimal_indices.empty(), LOG);

    // Test

    v.set(4, 0.0);

    minimal_indices = v.calculate_minimal_indices(2);

    assert_true(minimal_indices[0] == 0 || minimal_indices[0] == 1, LOG);
    assert_true(minimal_indices[1] == 0 || minimal_indices[1] == 1, LOG);

    //Test

    v.set(5);

    v[0] = 0;
    v[1] = 1;
    v[2] = 0;
    v[3] = 2;
    v[4] = 0;

    minimal_indices = v.calculate_minimal_indices(5);

    assert_true(minimal_indices[0] == 0 || minimal_indices[0] == 2 || minimal_indices[0] == 4, LOG);
    assert_true(minimal_indices[1] == 0 || minimal_indices[1] == 2 || minimal_indices[1] == 4, LOG);
    assert_true(minimal_indices[2] == 0 || minimal_indices[2] == 2 || minimal_indices[2] == 4, LOG);;
    assert_true(minimal_indices[3] == 1, LOG);
    assert_true(minimal_indices[4] == 3, LOG);

    // Test

    v.set(4);
    v[0] = -1.0;
    v[1] =  2.0;
    v[2] = -3.0;
    v[3] =  4.0;

    minimal_indices = v.calculate_minimal_indices(2);

    assert_true(minimal_indices[0] == 2, LOG);
    assert_true(minimal_indices[1] == 0, LOG);
}


void VectorTest::test_calculate_maximal_indices()
{
    message += "test_calculate_maximal_indices\n";

    Vector<double> v;
    Vector<size_t> maximal_indices;

    // Test

    v.set(4);
    v[0] = -1.0;
    v[1] =  2.0;
    v[2] = -3.0;
    v[3] =  4.0;

    maximal_indices = v.calculate_maximal_indices(2);

    assert_true(maximal_indices[0] == 3, LOG);
    assert_true(maximal_indices[1] == 1, LOG);

    // Test

    v.set(10);

    v.randomize_normal();

    maximal_indices = v.calculate_maximal_indices(10);

    assert_true(v[maximal_indices[0]] >= v[maximal_indices[1]], LOG);
    assert_true(v[maximal_indices[1]] >= v[maximal_indices[2]], LOG);
    assert_true(v[maximal_indices[2]] >= v[maximal_indices[3]], LOG);
    assert_true(v[maximal_indices[3]] >= v[maximal_indices[4]], LOG);
    assert_true(v[maximal_indices[4]] >= v[maximal_indices[5]], LOG);
    assert_true(v[maximal_indices[5]] >= v[maximal_indices[6]], LOG);
    assert_true(v[maximal_indices[6]] >= v[maximal_indices[7]], LOG);
    assert_true(v[maximal_indices[7]] >= v[maximal_indices[8]], LOG);
    assert_true(v[maximal_indices[8]] >= v[maximal_indices[9]], LOG);

    assert_true(v.get_subvector(maximal_indices).is_decrescent(), LOG);

    //Test

    v.set(5);

    v[0] = 0;
    v[1] = 1;
    v[2] = 0;
    v[3] = 2;
    v[4] = 0;

    maximal_indices = v.calculate_maximal_indices(5);

    assert_true(maximal_indices[0] == 3, LOG);
    assert_true(maximal_indices[1] == 1, LOG);
    assert_true(maximal_indices[2] == 0, LOG);
    assert_true(maximal_indices[3] == 2, LOG);
    assert_true(maximal_indices[4] == 4, LOG);
}


void VectorTest::test_calculate_minimal_maximal_index()
{
   message += "test_calculate_minimal_maximal_index\n";
   
   Vector<int> v(0, 1, 1);

   Vector<size_t> minimal_maximal_index = v.calculate_minimal_maximal_index();

   assert_true(minimal_maximal_index[0] == 0, LOG);
   assert_true(minimal_maximal_index[1] == 1, LOG);
}


void VectorTest::test_calculate_cumulative_index()
{
   message += "test_calculate_cumulative_index\n";

   Vector<double> v;
   double value;
   size_t index;

   // Test

   v.set(0.0, 1.0, 1.0);
   value = 0.0;
   index = v.calculate_cumulative_index(value);

   assert_true(index == 0, LOG);

   // Test

   v.set(0.0, 1.0, 1.0);
   value = 0.5;
   index = v.calculate_cumulative_index(value);

   assert_true(index == 1, LOG);

   // Test

   v.set(0.0, 1.0, 1.0);
   value = 1.0;
   index = v.calculate_cumulative_index(value);

   assert_true(index == 1, LOG);
}


void VectorTest::test_calculate_closest_index()
{
   message += "test_calculate_closest_index\n";

   Vector<double> v;
   double value = 2.15;

   //Test

   v.set(1,0.25,3);

   assert_true(v.calculate_closest_index(value) == 5, LOG);

   //Test

   double value2 = 2.125;
   assert_true(v.calculate_closest_index(value2) == 4, LOG);
   assert_false(v.calculate_closest_index(value2) == 5, LOG);
}


void VectorTest::test_calculate_sum_squared_error()
{
   message += "test_calculate_sum_squared_error\n";

   Vector<double> v;
   Vector<double> w;

   v.set(1,1,3);
   w.set(3,-1,1);

   assert_true(fabs(8 - v.calculate_sum_squared_error(w)) < numeric_limits<double>::epsilon(), LOG);
}


void VectorTest::test_calculate_mean_squared_error()
{
   message += "test_calculate_mean_squared_error\n";
}


void VectorTest::test_calculate_root_mean_squared_error()
{
   message += "test_calculate_root_mean_squared_error\n";
}


void VectorTest::test_calculate_norm()
{
   message += "test_calculate_norm\n";

   Vector<double> v;

   assert_true(v.calculate_L2_norm() == 0.0, LOG);

   v.set(2);
   v.initialize(1);

   assert_true(fabs(v.calculate_L2_norm() - sqrt(2.0)) < 1.0e-6, LOG);
}


void VectorTest::test_calculate_normalized()
{
   message += "test_calculate_normalized\n";

   Vector<double> v;
   Vector<double> normalized;

   // Test

   v.set(2, 3.1415927);

   normalized = v.calculate_normalized();

   assert_true(fabs(normalized.calculate_L2_norm() - 1.0) < 1.0e-6, LOG);
}


void VectorTest::test_apply_absolute_value()
{
   message += "test_apply_absolute_value\n";
}


void VectorTest::test_calculate_lower_bounded()
{
   message += "test_calculate_lower_bounded\n";

   Vector<double> v(1, -1.0);
   Vector<double> lower_bound(1, 0.0);

   assert_true(v.calculate_lower_bounded(lower_bound) == 0.0, LOG);
}


void VectorTest::test_calculate_upper_bounded()
{
   message += "test_calculate_upper_bounded\n";
}


void VectorTest::test_calculate_lower_upper_bounded()
{
   message += "test_calculate_lower_upper_bounded\n";
}


void VectorTest::test_dot_vector()
{
   message += "test_dot_vector\n";

   Vector<double> a;
   Vector<double> b;

   double c;

   // Test

   a.set(1, 2.0);
   b.set(1, 2.0);

   c = a.dot(b);

   assert_true(c == 4.0, LOG);

   // Test

   a.set(2, 0.0);
   b.set(2, 0.0);

   c = a.dot(b);

   assert_true(c == 0.0, LOG);

   // Test

   a.set(3);
   a.randomize_normal();

   b.set(3);
   b.randomize_normal();

   c = a.dot(b);

   assert_true(fabs(c - dot(a, b)) < numeric_limits<double>::epsilon(), LOG);
}


void VectorTest::test_dot_matrix()
{
   message += "test_dot_matrix\n";

   Vector<double> a;
   Matrix<double> b;

   Vector<double> c;

   // Test

   a.set(2, 0.0);
   b.set(2, 2, 0.0);

   c = a.dot(b);

   assert_true(c == 0.0, LOG);

   // Test

   a.set(2, 1.0);
   b.set(2, 2, 1.0);

   c = a.dot(b);

   assert_true(c == 2.0, LOG);

   // Test

   a.set(2);
   a[0] = -1.0;
   a[1] =  1.0;

   b.set(2, 2);
   b(0,0) = 1.0;
   b(0,1) = 2.0;
   b(1,0) = 3.0;
   b(1,1) = 4.0;

   c = a.dot(b);
   assert_true(c == 2, LOG);

   a.set(3);
   a.randomize_normal();

   b.set(3, 2);
   b.randomize_normal();

   c = a.dot(b);

   assert_true(c == dot(a, b), LOG);
}


void VectorTest::test_insert()
{
   message += "test_insert\n";

   Vector<int> a(4, 0);
   Vector<int> b(2, 1);

   a.tuck_in(1, b);

   Vector<int> c(4);
   c[0] = 0;
   c[1] = 1;
   c[2] = 1;
   c[3] = 0;

   assert_true(a == c, LOG);
}


void VectorTest::test_take_out()
{
   message += "test_take_out\n";

   Vector<int> a(4);
   a[0] = 0;
   a[1] = 1;
   a[2] = 1;
   a[3] = 0;

   Vector<int> b = a.get_subvector(1, 2);

   Vector<int> c(2, 1);

   assert_true(b == c, LOG);
}


void VectorTest::test_insert_element()
{
    message += "test_insert_element\n";
}


// @todo

void VectorTest::test_split_element()
{
//    message += "test_split_element\n";

//    Vector<string> v;

//    // Test

//    v.set(1, "hello,bye");

//    v.split_element(0, ',');

//    assert_true(v.size() == 2, LOG);
//    assert_true(v[0] == "hello", LOG);
//    assert_true(v[1] == "bye", LOG);
}


void VectorTest::test_remove_element()
{
    message += "test_remove_element\n";

    Vector<int> v;
    Vector<int> w;

    // Test

    v.set(3);
    v[0] = 2;
    v[1] = -1;
    v[2] = 3;

    w = v.delete_index(0);

    assert_true(w.size() == 2, LOG);
    assert_true(w[0] == -1, LOG);
//    assert_true(w[1] == 3, LOG);

    // Test

    v.set(3);
    v[0] = 2;
    v[1] = -1;
    v[2] = 3;

    w = v.delete_index(1);

    assert_true(w.size() == 2, LOG);
    assert_true(w[0] == 2, LOG);
    assert_true(w[1] == 3, LOG);

    // Test

    v.set(3);
    v[0] = 2;
    v[1] = -1;
    v[2] = 3;

    w = v.delete_index(2);

    assert_true(w.size() == 2, LOG);
    assert_true(w[0] == 2, LOG);
    assert_true(w[1] == -1, LOG);
}


void VectorTest::test_get_assembly()
{
   message += "test_get_assembly\n";

   Vector<int> a;
   Vector<int> b;
   Vector<int> c;
   Vector<int> d;
	   
   c = a.assemble(b);

   assert_true(c.size() == 0, LOG);

   a.set(1, 0);
   b.set(0, 0);
   c = a.assemble(b);

   assert_true(c.size() == 1, LOG);

   a.set(0, 0);
   b.set(1, 0);
   c = a.assemble(b);

   assert_true(c.size() == 1, LOG);

   a.set(1, 0);
   b.set(1, 1);
  
   c = a.assemble(b);

   d.resize(2);
   d[0] = 0;
   d[1] = 1;

   assert_true(c == d, LOG);
}


void VectorTest::test_difference()
{
    message += "test_difference\n";

    Vector<int> a;
    Vector<int> b;
    Vector<int> c;
    Vector<int> d;

    c = a.get_difference(b);

    assert_true(c.size() == 0, LOG);

    a.set(1, 0);
    b.set(0, 0);
    c = a.get_difference(b);

    assert_true(c.size() == 1, LOG);

    a.set(0, 0);
    b.set(1, 0);
    c = a.get_difference(b);

    assert_true(c.size() == 1, LOG);

    a.set(2);
    a[0] = 0;
    a[1] = 1;

    b.set(1, 1);

    c = a.get_difference(b);

    d.resize(1);
    d[0] = 0;

    assert_true(c == d, LOG);

    a.set(3);
    a[0] = 1;
    a[1] = 2;
    a[2] = 3;

    b.set(4);
    b[0] = 1;
    b[1] = 4;
    b[2] = 3;
    b[3] = 3;

    c = a.get_difference(b);

    assert_true(c.size() == 1, LOG);
    assert_true(c[0] == 2, LOG);

    // Test

//    a.set(3);
//    a.initialize_sequential();

//    b.set(3);
//    b.randomize_normal(-1000, 10000);

//    c = a.difference(b);
//    Assert?
}


void VectorTest::test_intersection()
{
    message += "test_intersection\n";

    Vector<int> a;
    Vector<int> b;
    Vector<int> c;

    a.set(3,1);
    b.set(3,1);

    c = a.get_intersection(b);

    assert_true(c.size() == 3, LOG);
    assert_true(c[0] == 1, LOG);

    a.set(3);
    a.initialize_sequential();

    b.set(5);
    b.initialize_sequential();

    c = a.get_intersection(b);

    assert_true(c.size() == 3, LOG);

    a.set(2);
    a[0] = 1;
    a[1] = 2;

    b.set(2);
    a[0] = 3;
    a[1] = 4;

    c = a.get_intersection(b);

    assert_true(c.size() == 0, LOG);
}


void VectorTest::test_get_unique()
{
    message += "test_unique\n";

    Vector<int> a;
    Vector<int> b;

    a.set(3,1);

    b = a.get_unique_elements();

    assert_true(b.size() == 1, LOG);
    assert_true(b[0] == 1, LOG);

    a.set(3);
    a[0] = 3;
    a[1] = 2;
    a[2] = 3;

    b = a.get_unique_elements();

    assert_true(b.size() == 2, LOG);

}


void VectorTest::test_apply_lower_bound()
{
   message += "test_apply_lower_bound\n";
}


void VectorTest::test_apply_upper_bound()
{
   message += "test_apply_upper_bound\n";
}


void VectorTest::test_apply_lower_upper_bounds()
{
   message += "test_apply_lower_upper_bounds\n";
}


void VectorTest::test_calculate_less_rank()
{
    message += "test_calculate_less_rank\n";

    Vector<double> v;

    Vector<size_t> rank;

    // Test

    v.set(3);
    v[0] =  0.0;
    v[1] = -1.0;
    v[2] =  1.0;

    rank = v.calculate_less_rank();

    assert_true(v.size() == 3, LOG);

    assert_true(rank[0] == 1, LOG);
    assert_true(rank[1] == 0, LOG);
    assert_true(rank[2] == 2, LOG);

    // Test

    v.set(10);
    v.randomize_normal();

    rank = v.calculate_less_rank();

    assert_true(v.calculate_minimal_index() == rank.calculate_minimal_index(), LOG);
    assert_true(v.calculate_maximal_index() == rank.calculate_maximal_index(), LOG);

    //Test

    v.set(2, 0.0);

    rank = v.calculate_less_rank();

    assert_true(rank[0] == 0 || rank[0] == 1, LOG);
}


void VectorTest::test_calculate_greater_rank()
{
   message += "test_calculate_greater_rank\n";

   Vector<double> v;

   Vector<size_t> rank;

   // Test

   v.set(3);
   v[0] =  0.0;
   v[1] = -1.0;
   v[2] =  1.0;

   rank = v.calculate_greater_rank();

   assert_true(v.size() == 3, LOG);

   assert_true(rank[0] == 1, LOG);
   assert_true(rank[1] == 2, LOG);
   assert_true(rank[2] == 0, LOG);

   // Test

   v.set(10);
   v.randomize_normal();

   rank = v.calculate_greater_rank();

   assert_true(v.calculate_minimal_index() == rank.calculate_maximal_index(), LOG);
   assert_true(v.calculate_maximal_index() == rank.calculate_minimal_index(), LOG);

   //Test

   v.set(6);

   v[0] =  0.0;
   v[1] =  0.0;
   v[2] =  0.0;
   v[3] =  0.0;
   v[4] =  0.0;
   v[5] =  0.0;

   rank = v.calculate_greater_rank();

   assert_true(rank[0] == 0, LOG);
   assert_true(rank[1] == 1, LOG);
   assert_true(rank[2] == 2, LOG);
   assert_true(rank[3] == 3, LOG);
   assert_true(rank[4] == 4, LOG);
   assert_true(rank[5] == 5, LOG);
}


void VectorTest::test_calculate_linear_regression_parameters()
{
    message += "test_calculate_linear_regression_parameters\n";

    Vector<double> x;
    Vector<double> y;

    LinearRegressionParameters<double> linear_regression_parameters;

    // Test

    x.set(5);
    x.randomize_normal();

    y.set(x);

    linear_regression_parameters = y.calculate_linear_regression_parameters(x);

    assert_true(fabs(linear_regression_parameters.intercept) < 1.0e-6, LOG);
    assert_true(fabs(linear_regression_parameters.slope - 1.0) < 1.0e-6, LOG);
    assert_true(fabs(linear_regression_parameters.correlation - 1.0) < 1.0e-6, LOG);

    // Test

    x.set(15);
    y.set(15);

    x[0]  = 1.47; y[0]  = 52.21;
    x[1]  = 1.50; y[1]  = 53.12;
    x[2]  = 1.52; y[2]  = 54.48;
    x[3]  = 1.55; y[3]  = 55.84;
    x[4]  = 1.57; y[4]  = 57.20;
    x[5]  = 1.60; y[5]  = 58.57;
    x[6]  = 1.63; y[6]  = 59.93;
    x[7]  = 1.65; y[7]  = 61.29;
    x[8]  = 1.68; y[8]  = 63.11;
    x[9]  = 1.70; y[9]  = 64.47;
    x[10] = 1.73; y[10] = 66.28;
    x[11] = 1.75; y[11] = 68.10;
    x[12] = 1.78; y[12] = 69.92;
    x[13] = 1.80; y[13] = 72.19;
    x[14] = 1.83; y[14] = 74.46;

    linear_regression_parameters = y.calculate_linear_regression_parameters(x);

    assert_true(fabs(fabs(linear_regression_parameters.intercept) - fabs(-39.1468)) < 1.0, LOG);
    assert_true(fabs(linear_regression_parameters.slope - 61.6746) < 1.0, LOG);
    assert_true(fabs(linear_regression_parameters.correlation - 0.9945) < 1.0e-3, LOG);
}


void VectorTest::test_threshold()
{
    Vector<double> m;
    Vector<double> l;

    m.set(4);
    l.set(4);

    m[0] = -1;
    m[1] = 1;
    m[2] = 2;
    m[3] = -2;

    l = threshold(m);

    assert_true(fabs(l[0] - 0) < 0.001, LOG);
    assert_true(fabs(l[1] - 1) < 0.001, LOG);
    assert_true(fabs(l[2] - 1) < 0.001, LOG);
    assert_true(fabs(l[3] - 0) < 0.001, LOG);
}


void VectorTest::test_symmetric_threshold()
{
    Vector<double> m;
    Vector<double> l;

    m.set(4);
    l.set(4);

    m[0] = -1;
    m[1] = 1;
    m[2] = 2;
    m[3] = -2;

    l = symmetric_threshold(m);

    assert_true(fabs(l[0] - -1) < 0.001, LOG);
    assert_true(fabs(l[1] - 1) < 0.001, LOG);
    assert_true(fabs(l[2] - 1) < 0.001, LOG);
    assert_true(fabs(l[3] - -1) < 0.001, LOG);
}


void VectorTest::test_logistic()
{
    Vector<double> m;
    Vector<double> l;

    m.set(4);
    l.set(4);

    m[0] = -1;
    m[1] = 1;
    m[2] = 2;
    m[3] = -2;

    l = logistic(m);

    assert_true(fabs(l[0] - 0.268941) < 0.000001, LOG);
    assert_true(fabs(l[1] - 0.731059) < 0.000001, LOG);
    assert_true(fabs(l[2] - 0.880797) < 0.000001, LOG);
    assert_true(fabs(l[3] - 0.119203) < 0.000001, LOG);
}


void VectorTest::test_hyperbolic_tangent()
{
    Vector<double> m;
    Vector<double> l;

    m.set(4);
    l.set(4);

    m[0] = -1;
    m[1] = 1;
    m[2] = 2;
    m[3] = -2;

    l = hyperbolic_tangent(m);

    assert_true(fabs(l[0] - -0.761594) < 0.000001, LOG);
    assert_true(fabs(l[1] - 0.761594) < 0.000001, LOG);
    assert_true(fabs(l[2] - 0.964028) < 0.000001, LOG);
    assert_true(fabs(l[3] - -0.964028) < 0.000001, LOG);
}


void VectorTest::test_hyperbolic_tangent_derivatives()
{
    Vector<double> m;
    Vector<double> l;

    m.set(4);
    l.set(4);

    m[0] = -1;
    m[1] = 1;
    m[2] = 2;
    m[3] = -2;

    l = hyperbolic_tangent_derivatives(m);

    assert_true(fabs(l[0] - 0.419974) < 0.000001, LOG);
    assert_true(fabs(l[1] - 0.419974) < 0.000001, LOG);
    assert_true(fabs(l[2] - 0.070651) < 0.000001, LOG);
    assert_true(fabs(l[3] - 0.070651) < 0.000001, LOG);
}


void VectorTest::test_hyperbolic_tangent_second_derivatives()
{
    Vector<double> m;
    Vector<double> l;

    m.set(4);
    l.set(4);

    m[0] = -1;
    m[1] = 1;
    m[2] = 2;
    m[3] = -2;

    l = hyperbolic_tangent_second_derivatives(m);

    assert_true(fabs(l[0] - 0.639700) < 0.000001, LOG);
    assert_true(fabs(l[1] - -0.639700) < 0.000001, LOG);
    assert_true(fabs(l[2] - -0.136219) < 0.000001, LOG);
    assert_true(fabs(l[3] - 0.136219) < 0.000001, LOG);
}


void VectorTest::test_logistic_derivatives()
{
    Vector<double> m;
    Vector<double> l;

    m.set(4);
    l.set(4);

    m[0] = -1;
    m[1] = 1;
    m[2] = 2;
    m[3] = -2;

    l = logistic_derivatives(m);

    assert_true(fabs(l[0] - 0.196612) < 0.000001, LOG);
    assert_true(fabs(l[1] - 0.196612) < 0.000001, LOG);
    assert_true(fabs(l[2] - 0.104994) < 0.000001, LOG);
    assert_true(fabs(l[3] - 0.104994) < 0.000001, LOG);
}


void VectorTest::test_logistic_second_derivatives()
{
    Vector<double> m;
    Vector<double> l;

    m.set(4);
    l.set(4);

    m[0] = -1;
    m[1] = 1;
    m[2] = 2;
    m[3] = -2;

    l = logistic_second_derivatives(m);

    assert_true(fabs(l[0] - 0.090858) < 0.000001, LOG);
    assert_true(fabs(l[1] - -0.090858) < 0.000001, LOG);
    assert_true(fabs(l[2] - -0.079963) < 0.000001, LOG);
    assert_true(fabs(l[3] - 0.079963) < 0.000001, LOG);
}


void VectorTest::test_threshold_derivatives()
{
    Vector<double> m;
    Vector<double> l;

    m.set(2);
    l.set(2);

    m[0] = 2;
    m[1] = -2;

    l = threshold_derivatives(m);

    assert_true(fabs(l[0] - 0) < 0.000001, LOG);
    assert_true(fabs(l[1] - 0) < 0.000001, LOG);
}


void VectorTest::test_threshold_second_derivatives()
{
    Vector<double> m;
    Vector<double> l;

    m.set(2);
    l.set(2);

    m[0] = 2;
    m[1] = -2;

    l = threshold_derivatives(m);

    assert_true(fabs(l[0] - 0) < 0.000001, LOG);
    assert_true(fabs(l[1] - 0) < 0.000001, LOG);
}


void VectorTest::test_symmetric_threshold_derivatives()
{
    Vector<double> m;
    Vector<double> l;

    m.set(2);
    l.set(2);

    m[0] = 2;
    m[1] = -2;

    l = threshold_derivatives(m);

    assert_true(fabs(l[0] - 0) < 0.000001, LOG);
    assert_true(fabs(l[1] - 0) < 0.000001, LOG);
}


void VectorTest::test_symmetric_threshold_second_derivatives()
{
    Vector<double> m;
    Vector<double> l;

    m.set(2);
    l.set(2);

    m[0] = 2;
    m[1] = -2;

    l = threshold_derivatives(m);

    assert_true(fabs(l[0] - 0) < 0.000001, LOG);
    assert_true(fabs(l[1] - 0) < 0.000001, LOG);
}


void VectorTest::test_scale_minimum_maximum()
{
    message += "test_scale_minimum_maximum\n";

    Vector<double> v;
    Statistics<double> statistics;

    // Test

    v.set(2);
    v.randomize_uniform(-2000.0, 2000.0);

    statistics = v.scale_minimum_maximum();

    assert_true(v.calculate_statistics().has_minimum_minus_one_maximum_one(), LOG);
}


void VectorTest::test_scale_mean_standard_deviation()
{
    message += "test_scale_mean_standard_deviation\n";

    Vector<double> v;
    Statistics<double> statistics;

    // Test

    v.set(2);
    v.randomize_uniform(-2000.0, 2000.0);

    statistics = v.scale_mean_standard_deviation();

    assert_true(v.calculate_statistics().has_mean_zero_standard_deviation_one(), LOG);
}


void VectorTest::test_unscale_minimum_maximum()
{
    message += "test_unscale_minimum_maximum\n";

    Vector<double> v;
    Vector<double> copy;
    Statistics<double> statistics;

    // Test

    v.set(2);
    v.randomize_uniform(-2000.0, 2000.0);

    copy = v;

    statistics = v.scale_minimum_maximum();

    v.unscale_minimum_maximum(Vector<double>(2,statistics.minimum), Vector<double>(2,statistics.maximum));

    assert_true((v - copy).calculate_absolute_value() < 1.0e-3 , LOG);
}


void VectorTest::test_unscale_mean_standard_deviation()
{
    message += "test_unscale_mean_standard_deviation\n";

    Vector<double> v;
    Vector<double> copy;
    Statistics<double> statistics;

    // Test

    v.set(2);
    v.randomize_uniform(-2000.0, 2000.0);

    copy = v;

    statistics = v.scale_mean_standard_deviation();

    v.unscale_mean_standard_deviation(Vector<double>(2,statistics.mean), Vector<double>(2,statistics.standard_deviation));

    assert_true((v - copy).calculate_absolute_value() < 1.0e-3 , LOG);
}


void VectorTest::test_parse()
{
   message += "test_parse\n";

   Vector<int> v;

   string str;

   // Test

   str = "1 2 3";

   v.parse(str);

   assert_true(v.size() == 3, LOG);
   assert_true(v[0] == 1, LOG);
   assert_true(v[1] == 2, LOG);
   assert_true(v[2] == 3, LOG);
}


void VectorTest::test_load()
{
   message += "test_load\n";

   string file_name = "../data/vector.dat";

   Vector<int> v;
      
   // Test

   v.set(3, 1);

   v.save(file_name);
   v.load(file_name);

   assert_true(v.size() == 3, LOG);
   assert_true(v == 1, LOG);

   // Test

   v.set(2);
   v[0] = -1;
   v[1] = 1;

   v.save(file_name);
   v.load(file_name);

   assert_true(v.size() == 2, LOG);
   assert_true(v[0] == -1, LOG);
   assert_true(v[1] == 1, LOG);
}


void VectorTest::test_save()
{
   message += "test_save\n";

   string file_name = "../data/vector.dat";

   Vector<int> v(2, 0);
   Vector<int> w(v);

   v.save(file_name);

   v.load(file_name);

   assert_true(v == w, LOG);
}


void VectorTest::run_test_case()
{
   message += "Running vector test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Arithmetic operators

   test_sum_operator();
   test_rest_operator();
   test_multiplication_operator();
   test_division_operator();

   // Operation and assignment operators

   test_sum_assignment_operator();
   test_rest_assignment_operator();
   test_multiplication_assignment_operator();
   test_division_assignment_operator();

   // Equality and relational operators

   test_equal_to_operator();
   test_not_equal_to_operator();

   test_greater_than_operator();
   test_greater_than_or_equal_to_operator();

   test_less_than_operator();
   test_less_than_or_equal_to_operator();

   // Output operator

   test_output_operator();

   // Get methods

   test_get_display();

   test_get_subvector_random();

   // Set methods

   test_set();
   test_set_display();

   // Resize methods

   test_resize();

   test_insert();
   test_take_out();

   test_insert_element();
   test_split_element();

   test_remove_element();

   test_get_assembly();
   test_difference();

   test_intersection();

   test_get_unique();

   // Initialization methods

   test_initialize();
   test_initialize_sequential();
   test_randomize_uniform();
   test_randomize_normal();

   // Checking methods

   test_contains();
   test_is_in();
   test_is_constant();
   test_is_crescent();
   test_is_decrescent();

   test_impute_time_series_missing_values_mean();

   // Mathematical methods

   test_dot_vector();
   test_dot_matrix();

   test_calculate_sum();
   test_calculate_partial_sum();
   test_calculate_product();

   test_calculate_mean();
   test_calculate_standard_deviation();
   test_calculate_covariance();

   test_calculate_mean_standard_deviation();

   test_calculate_minimum();
   test_calculate_maximum();

   test_calculate_minimum_maximum();

   test_calculate_minimum_missing_values();
   test_calculate_maximum_missing_values();

   test_calculate_minimum_maximum_missing_values();

   test_calculate_explained_variance();

   test_calculate_quartiles();

   test_calculate_histogram();

   test_calculate_bin();
   test_calculate_frequency();
   test_calculate_total_frequencies();

   test_calculate_minimal_index();
   test_calculate_maximal_index();

   test_calculate_minimal_indices();
   test_calculate_maximal_indices();

   test_calculate_minimal_maximal_index();

   test_calculate_cumulative_index();
   test_calculate_closest_index();

   test_calculate_norm();
   test_calculate_normalized();

   test_calculate_sum_squared_error();
   test_calculate_mean_squared_error();
   test_calculate_root_mean_squared_error();

   test_apply_absolute_value();

   test_calculate_lower_bounded();
   test_calculate_upper_bounded();

   test_calculate_lower_upper_bounded();

   test_apply_lower_bound();
   test_apply_upper_bound();
   test_apply_lower_upper_bounds();

   test_calculate_less_rank();
   test_calculate_greater_rank();

   test_calculate_linear_regression_parameters();

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

   // Scaling and unscaling

   test_scale_minimum_maximum();
   test_scale_mean_standard_deviation();

   test_unscale_minimum_maximum();
   test_unscale_mean_standard_deviation();

   // Parsing methods

   test_parse();

   // Serialization methods

   test_save();

   test_load();

   message += "End vector test case\n";
}


double VectorTest::dot(const Vector<double>& vector, const Vector<double>& other_vector)
{
    double dot_product = 0.0;

    for(size_t i = 0; i < vector.size(); i++)
    {
       dot_product += vector[i]*other_vector[i];
    }

    return(dot_product);
}


Vector<double> VectorTest::dot(const Vector<double>& vector, const Matrix<double>& matrix)
{
    const size_t rows_number = matrix.get_rows_number();
    const size_t columns_number = matrix.get_columns_number();

    Vector<double> product(columns_number);

    for(size_t j = 0; j < columns_number; j++)
    {
       product[j] = 0;

       for(size_t i = 0; i < rows_number; i++)
       {
          product[j] += vector[i]*matrix(i,j);
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
