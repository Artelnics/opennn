//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V E C T O R   T E S T   C L A S S                                     
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "vector_test.h"


VectorTest::VectorTest() : UnitTesting() 
{   
}


VectorTest::~VectorTest()
{
}


void VectorTest::test_constructor()
{
   cout << "test_constructor\n";

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
   assert_true(abs(v7[4] - 3.8) < numeric_limits<double>::epsilon(), LOG);

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
   cout << "test_sum_operator\n";

   Vector<int> a, b, c, d;

   // Scalar

   a.set(1, 1);
   b = a + 1;

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
   cout << "test_rest_operator\n";

   Vector<double> a, b, c, d;

   // Scalar

   a.set(1, 1.0);
   b = a - 1.0;

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
   cout << "test_multiplication_operator\n";

   Vector<double> a, b, c, d;

   // Scalar

   a.set(1, 1.0);
   b = a*2.0;

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
   cout << "test_division_operator\n";

   Vector<double> a, b, c, d;

   // Scalar

   a.set(1, 1.0);
   b = a/2.0;

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
   cout << "test_sum_assignment_operator\n";

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
   cout << "test_rest_assignment_operator\n";

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
   cout << "test_multiplication_assignment_operator\n";

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
   cout << "test_division_assignment_operator\n";

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
   cout << "test_equal_to_operator\n";

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
   cout << "test_not_equal_to_operator\n";

   Vector<double> a(2, -1.0);
   Vector<double> b(2, 1.0);

   assert_true(a != b, LOG);
   assert_true(a != 0.0, LOG);
   assert_true(b != 0.0, LOG);
}


void VectorTest::test_greater_than_operator()
{
   cout << "test_greater_than_operator\n";

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
   cout << "test_less_than_operator\n";

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
   cout << "test_greater_than_or_equal_to_operator\n";

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
   cout << "test_less_than_or_equal_to_operator\n";

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
   cout << "test_output_operator\n";

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
   cout << "test_get_size\n";

   Vector<int> v;

   assert_true(v.size() == 0, LOG);

   v.set(1);

   assert_true(v.size() == 1, LOG);

   v.set(0);

   assert_true(v.size() == 0, LOG);
}


void VectorTest::test_get_display()
{
   cout << "test_get_display\n";
}


void VectorTest::test_get_subvector()
{
    cout << "test_get_subvector\n";

    //String case

    Vector<string> vector({"hello", "house", "car", "dog", "horse"});
    Vector<string> solution({"car", "dog", "horse"});
    Vector<string> check_vector(vector.size());

    check_vector = vector.get_subvector(2,4);

    assert_true(check_vector == solution, LOG);

    //Size_t case

    Vector<size_t> vector1({1, 2, 3, 4, 5});
    Vector<size_t> solution1({1, 2, 3});
    Vector<size_t> check_vector1(vector.size());

    check_vector1 = vector1.get_subvector(0,2);

    assert_true(check_vector1 == solution1, LOG);

    //Double case

    Vector<double> vector2({1.2, 2.5, 6.8, 5});
    Vector<double> solution2({2.5, 6.8});
    Vector<double> check_vector2(vector.size());

    check_vector2 = vector2.get_subvector(1,2);

    assert_true(check_vector2 == solution2, LOG);
}


void VectorTest::test_get_subvector_random()
{
    cout << "test_get_subvector_random\n";

    Vector<double> a(2, 0.0);

    assert_true(a.get_subvector_random(2) == a, LOG);
    assert_true(a.get_subvector_random(1).size() == 1, LOG);
    assert_true(a.get_subvector_random(1)[0] == 0.0, LOG);

    Vector<int> b(1,1,10);

    assert_true(b.get_subvector_random(5) >= 1, LOG);
    assert_true(b.get_subvector_random(5) <= 10, LOG);
}


void VectorTest::test_get_first()
{
    cout << "test_get_first\n";

    Vector<size_t> vector ({ 2, 4, 6, 9, 3});
    Vector<size_t> chek_vector(vector.size());
    Vector<size_t> solution ({ 2, 4, 6});

    chek_vector = vector.get_first(3);

    assert_true(chek_vector == solution, LOG);
}


void VectorTest::test_get_last()
{
    cout << "test_get_last\n";

    Vector<size_t> vector ({ 2, 4, 6, 9, 3});
    Vector<size_t> chek_vector(vector.size());
    Vector<size_t> solution ({ 6, 9, 3});

    chek_vector = vector.get_last(3);

    assert_true(chek_vector == solution, LOG);
}


void VectorTest::test_get_before_last()
{
    cout << "test_get_before_last \n";

    // String case

    Vector<string> vector({"Hello", "Bye", "Home"});
    string solution("Bye");
    string check_vector;

    check_vector = vector.get_before_last();

    assert_true(check_vector == solution, LOG);

    // Size_t case

    Vector<size_t> vector1({1, 2, 5, 6});
    size_t solution1 = 5;
    size_t check_vector1;

    check_vector1 = vector1.get_before_last();

    assert_true(check_vector1 == solution1, LOG);

    //Double case

    Vector<double> vector2({1.5, 2, 2.9, 4.6});
    double solution2 = 2.9;
    double check_vector2;

    check_vector2 = vector2.get_before_last();

    assert_true(check_vector2 - solution2 < 0.000001, LOG);
}


void VectorTest::test_delete_first()
{
    cout <<"test_delete_first\n";

    Vector<size_t> vector ({2, 4, 6, 9, 3});
    Vector<size_t> chek_vector(vector.size());
    Vector<size_t> solution ({ 6, 9, 3});

    chek_vector = vector.delete_first(2);

    assert_true(chek_vector == solution, LOG);
}


void VectorTest::test_delete_last()
{
    cout <<"test_delete_last\n";

    Vector<size_t> vector({2, 4, 5, 8, 7});
    Vector<size_t> check_vector(vector.size());
    Vector<size_t> solution ({2, 4, 5});

    check_vector = vector.delete_last(2);

    assert_true(check_vector == solution, LOG);
}


void VectorTest::test_get_integer_elements()
{
    cout << "test_get_integer_elements\n";

    //Normal case

    Vector<double> vector({2, 4, 5});
    Vector<double> solution({2, 4, 5});
    Vector<double> check_vector(vector.size());

    check_vector = vector.get_integer_elements(vector.size());

    assert_true(check_vector == solution, LOG);

    //Empty case

    Vector<double> vector1({});
    Vector<double> solution1({});
    Vector<double> check_vector1;

    check_vector1 = vector1.get_integer_elements(5);

    assert_true(check_vector1 == solution1, LOG);
}


void VectorTest::test_get_integers()
{
    cout <<"test_get_integers\n";

    // Normal case

    Vector<size_t> vector({2, 3, 5, 3, 8, 5});
    Vector<size_t> solution({2, 3, 5, 8});
    Vector<size_t> check_vector(vector.size());

    check_vector = vector.get_integer_elements(vector.size());

    assert_true(check_vector == solution, LOG);
}


void VectorTest::test_set()
{
   cout << "test_set\n";

   string file_name = "../data/vector.dat";

   Vector<int> vector(3, 0);

   // Default

   vector.set();

   assert_true(vector.size() == 0, LOG);

   // Size

   vector.set(1);

   assert_true(vector.size() == 1, LOG);

   // Size initialization

   vector.set(1, 0);

   assert_true(vector.size() == 1, LOG);
   assert_true(vector == 0, LOG);

   // File

   vector.set(file_name);
   vector.save(file_name);

   assert_true(vector == 0, LOG);

   // Sequential

   vector.set(10, 5, 50);

   assert_true(vector.size() == 9, LOG);
   assert_true(vector[0] == 10, LOG);
   assert_true(vector[8] == 50, LOG);

   vector.set(9, -1, 1);

   assert_true(vector.size() == 9, LOG);
   assert_true(vector[0] == 9, LOG);
   assert_true(vector[8] == 1, LOG);

   // Copy

   vector.set(1, 0);
   vector.set(vector);

   assert_true(vector.size() == 1, LOG);
   assert_true(vector == 0, LOG);
}


void VectorTest::test_set_display()
{
   cout << "test_set_display\n";
}


void VectorTest::test_resize()
{
   cout << "test_resize\n";

   Vector<int> vector(1, 0);

   // Decrease size

   vector.resize(2);

   assert_true(vector.size() == 2, LOG);

   // Increase size

   vector.resize(0);

   assert_true(vector.size() == 0, LOG);
}


void VectorTest::test_initialize()
{
   cout << "test_initialize\n";

   Vector<int> vector(2);

   vector.initialize(0);

   Vector<int> vector2(2, 0);
   
   assert_true(vector == vector2, LOG);
}


void VectorTest::test_initialize_first()
{
    cout << "test_initialize_first\n";

    //String case

    Vector<string> vector({"Hello", "Bye", "home"});
    Vector<string> solution({"Dog", "Bye", "home"});

    vector.initialize_first(1, "Dog");

    assert_true(vector == solution, LOG);

    //Size_t case

    Vector<size_t> vector1({1, 2, 5, 4});
    Vector<size_t> solution1({3, 3, 5, 4 });

    vector1.initialize_first(2, 3);

    assert_true(vector1 == solution1, LOG);

    //Double case

    Vector<double> vector2({1, 2.5, 6, 2.7});
    Vector<double> solution2({2.4, 2.5, 6, 2.7});

    vector2.initialize_first(1, 2.4);

    assert_true(vector2 == solution2, LOG);
}


void VectorTest::test_initialize_sequential()
{
   cout << "test_initialize_sequential\n";

   Vector<double> vector(2);

   vector.initialize_sequential();

   Vector<double> vector1(2);
   vector1[0] = 0.0;
   vector1[1] = 1.0;
   
   assert_true(vector == vector1, LOG);
}


void VectorTest::test_randomize_uniform()
{
   cout << "test_randomize_uniform\n";

   Vector<double> vector(3);

   vector.randomize_uniform(-1,1);

   assert_true(vector >= -1.0, LOG);
   assert_true(vector <= 1.0, LOG);
  
   vector.randomize_uniform(0.0, 2.0);
   
   assert_true(vector >= 0.0, LOG);
   assert_true(vector <= 2.0, LOG);
}


void VectorTest::test_randomize_normal()
{
   cout << "test_randomize_normal\n";

   Vector<double> vector(2);

   vector.randomize_normal();

   vector.randomize_normal(0.0, 0.0);

   assert_true(vector == 0.0, LOG);
}


void VectorTest::test_randomize_binary()
{
    cout << "test_randomize_binary\n";

    Vector<double> vector(10);

    vector.randomize_binary(-0.4, 0.8);

    assert_true(vector <= 1, LOG);
    assert_true(vector >= 0, LOG);
}


void VectorTest::test_fill_from()
{
    cout << "test_fill_from\n";

    //String case

    Vector<string> vector(3);
    Vector<string> solution({"5", "8", "7"});
    Vector<string> check_vector(vector.size());

    check_vector = vector.fill_from(0, {"5", "8", "7"});

    assert_true(check_vector == solution, LOG);

    //Size_t case

    Vector<size_t> vector1({2, 3, 5, 6, 7});
    Vector<size_t> solution1({2, 14, 5, 6, 7});
    Vector<size_t> check_vector1(vector1.size());

    check_vector1 = vector1.fill_from(1, {14});

    assert_true(check_vector1 == solution1, LOG);

    //Double case

    Vector<double> vector2({2.5, 3.7, 5, 6, 7});
    Vector<double> solution2({2.5, 14.2, 5, 6, 7});
    Vector<double> check_vector2(vector2.size());

    check_vector2 = vector2.fill_from(1, {14.2});

    assert_true(check_vector2 == solution2, LOG);
}


void VectorTest::test_contains()
{
   cout << "test_contains\n";

   Vector<int> vector;
   Vector<int> vector1;

   // Test

   assert_true(vector.contains(0) == false, LOG);

   //Test

   vector.set(5, -1);

   assert_true(vector.contains(0) == false, LOG);

   //Test

   vector.set(5, -1);
   vector[3] = 1;
   vector1.set(3, 1);

   assert_true(vector.contains(vector1) == true, LOG);

   //Test

   vector.set(10, -1);

   assert_true(vector.contains(0) == false, LOG);
}


void VectorTest::test_contains_greater_than()
{
    cout << "test_contains_greater_than \n";

    Vector<size_t> vector({2,3,6,14,2,5});

    assert_true(vector.contains_greater_than(10), LOG);
}


void VectorTest::test_is_in()
{
   cout << "test_is_in\n";

   Vector<size_t> vector(5, 0);

   assert_true(vector.is_in(0, 0), LOG);
}


void VectorTest::test_is_constant()
{
   cout << "test_is_constant\n";

   Vector<int> vector1(3,2);
   Vector<double> vector2(1,0.5,2.5);

   assert_true(vector1.is_constant(1), LOG);
   assert_true(vector2.is_constant(2), LOG);
   assert_false(vector2.is_constant(1), LOG);
}


void VectorTest::test_is_crescent()
{
   cout << "test_is_crescent\n";

   Vector<double> vector1(1,0.5,2.5);
   Vector<double> vector2(3);

   vector2[0]=1;
   vector2[1]=2;
   vector2[2]=2;

   assert_true(vector1.is_crescent(), LOG);
   assert_false(!vector2.is_crescent(), LOG);
}


void VectorTest::test_is_decrescent()
{
   cout << "test_is_decrescent\n";

   Vector<double> vector1(2.5,-0.5,1);
   Vector<double> vector2(3);

   vector2[0]=2;
   vector2[1]=1;
   vector2[2]=1;

   assert_true(vector1.is_decrescent(), LOG);
   assert_false(!vector2.is_decrescent(), LOG);
}

void VectorTest::test_is_constant_string()
{
    cout << "test_is_constant_string\n";

    Vector<string> vector1({"aaa","aaa","aaa"});
    Vector<string> vector2({"aaa","eee","iii"});

    assert_true(vector1.is_constant_string(), LOG);
    assert_true(!vector2.is_constant_string(), LOG);

}

void VectorTest::test_has_same_elements()
{
    cout << "test_has_same_elements\n";

    size_t size = 100;
    Vector<double> vector1(size);
    vector1.initialize_sequential();

    assert_true(vector1.has_same_elements(vector1), LOG);
}


void VectorTest::test_is_binary()
{
    cout << "test_is_binary \n";

    Vector<double> vector(3);

    vector[0]=2;
    vector[1]=1;
    vector[2]=1;

    assert_true(vector.is_binary(), LOG);
}


void VectorTest::test_is_binary_0_1()
{
    cout << "test_is_binary \n";

    Vector<size_t> vector({0, 1, 1, 1, 0, 1});
    Vector<size_t> vector1({0, 1, 0, 1, 2});

    assert_true(vector.is_binary_0_1(), LOG);
    assert_true(!vector1.is_binary_0_1(), LOG);
}

void VectorTest::test_is_positive()
{
    cout << "test_is_positive \n";

    Vector<double> vector({2,6,8,9});

    assert_true(vector.is_positive(), LOG);
}

void VectorTest::test_is_negative()
{
    cout << "test_is_negative\n";

    Vector<size_t> vector1({5,4,9,8});
    Vector<double> vector2({-5,-6,-8,-4});

    assert_true(!vector1.is_negative(), LOG);
    assert_true(vector2.is_negative(), LOG);

}

void VectorTest::test_is_integer()
{
    cout << "test_is_integer \n";

    size_t size = 100;
    Vector<double> vector(size);

    vector.initialize(100);

    assert_true(vector.is_integer(), LOG);
}


void VectorTest::test_check_period()
{
    cout << "test_check_period\n";

    Vector<size_t> vector({1, 3, 5, 7, 9});
    Vector<size_t> vector1({1, 3, 5, 6});

    assert_false(vector1.check_period(2), LOG);
    assert_true(vector.check_period(2), LOG);
}


void VectorTest::test_get_reverse()
{
    cout << "test_get_reverse\n";

    Vector<size_t> vector({1, 3, 5, 7, 9});
    Vector<size_t> vector_reverse({9, 7, 5, 3, 1});
    Vector<size_t> check_vector(vector.size());

    check_vector = vector.get_reverse();

    assert_true(check_vector == vector_reverse, LOG);

}


void VectorTest::test_calculate_sum()
{
   cout << "test_calculate_sum\n";

   Vector<int> vector;

   assert_true(vector.calculate_sum() == 0, LOG);

   vector.set(2);
   vector.initialize(1);

   assert_true(vector.calculate_sum() == 2, LOG);
}


void VectorTest::test_calculate_partial_sum()
{
    cout << "test_calculate_partial_sum\n";

    Vector<size_t> vector(5, 1);

    // Test

    Vector<size_t> indices(1, 0);

    assert_true(vector.calculate_partial_sum(indices) == 1, LOG);

    // Test

    indices.set(2);

    vector[4] = 8;

    indices[0] = 0;
    indices[1] = 4;

    assert_true(vector.calculate_partial_sum(indices) == 9, LOG);
}


void VectorTest::test_calculate_product()
{
   cout << "test_calculate_product\n";

   Vector<double> vector;
   Vector<int> vector1;

   assert_true(vector.calculate_product() == 1.0, LOG);

   vector.set(2);
   vector[0] = 0.5;
   vector[1] = 1.5;

   vector1.set(3,1,6);


   assert_true(vector.calculate_product() == 0.75, LOG);
   assert_true(vector1.calculate_product() == 360, LOG);
}


void VectorTest::test_calculate_cumulative_index()
{
   cout << "test_calculate_cumulative_index\n";

   Vector<double> vector;
   double value;
   size_t index;

   // Test

   vector.set(0.0, 1.0, 1.0);
   value = 0.0;
   index = vector.calculate_cumulative_index(value);

   assert_true(index == 0, LOG);

   // Test

   vector.set(0.0, 1.0, 1.0);
   value = 0.5;
   index = vector.calculate_cumulative_index(value);

   assert_true(index == 1, LOG);

   // Test

   vector.set(0.0, 1.0, 1.0);
   value = 1.0;
   index = vector.calculate_cumulative_index(value);

   assert_true(index == 1, LOG);
}


void VectorTest::test_dot_vector()
{
   cout << "test_dot_vector\n";

   Vector<double> vector1;
   Vector<double> vector2;

   double number;

   // Test

   vector1.set(1, 2.0);
   vector2.set(1, 2.0);

   number = dot(vector1, vector2);

   assert_true(number == 4.0, LOG);

   // Test

   vector1.set(2, 0.0);
   vector2.set(2, 0.0);

   number = dot(vector1, vector2);

   assert_true(number == 0.0, LOG);

   // Test

   vector1.set(3);
   vector2.randomize_normal();

   vector2.set(3);
   vector2.randomize_normal();

   number = dot(vector1, vector2);

   assert_true(abs(number - dot(vector1, vector2)) < numeric_limits<double>::epsilon(), LOG);
}


void VectorTest::test_dot_matrix()
{
   cout << "test_dot_matrix\n";

   Vector<double> vector;
   Matrix<double> matrix;

   Vector<double> number;

   // Test

   vector.set(2, 0.0);
   matrix.set(2, 2, 0.0);

   number = dot(vector, matrix);

   assert_true(number == 0.0, LOG);

   // Test

   vector.set(2, 1.0);
   matrix.set(2, 2, 1.0);

   number = dot(vector, matrix);

   assert_true(number == 2.0, LOG);

   // Test

   vector.set(2);
   vector[0] = -1.0;
   vector[1] = 1.0;

   matrix.set(2, 2);
   matrix(0,0) = 1.0;
   matrix(0,1) = 2.0;
   matrix(1,0) = 3.0;
   matrix(1,1) = 4.0;

   number = dot(vector, matrix);
   assert_true(number == 2, LOG);

   vector.set(3);
   vector.randomize_normal();

   matrix.set(3, 2);
   matrix.randomize_normal();

   number = dot(vector, matrix);

   assert_true(number == dot(vector, matrix), LOG);
}


void VectorTest::test_insert()
{
   cout << "test_insert\n";

   Vector<int> vector1(4, 0);
   Vector<int> vector2(2, 1);

   vector1.embed(1, vector2);

   Vector<int> vector3(4);
   vector3[0] = 0;
   vector3[1] = 1;
   vector3[2] = 1;
   vector3[3] = 0;

   assert_true(vector1 == vector3, LOG);
}


void VectorTest::test_take_out()
{
   cout << "test_take_out\n";

   Vector<int> vector1(4);
   vector1[0] = 0;
   vector1[1] = 1;
   vector1[2] = 1;
   vector1[3] = 0;

   Vector<int> vector2 = vector1.get_subvector(1, 2);

   Vector<int> vector3(2, 1);

   assert_true(vector2 == vector3, LOG);
}


void VectorTest::test_insert_element()
{
    cout << "test_insert_element\n";
}


void VectorTest::test_split_element()
{
//    cout << "test_split_element\n";

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
    cout << "test_remove_element\n";

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
   cout << "test_get_assembly\n";

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
    cout << "test_difference\n";

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

}


void VectorTest::test_intersection()
{
    cout << "test_intersection\n";

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
    cout << "test_unique\n";

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


void VectorTest::test_calculate_less_rank()
{
    cout << "test_calculate_less_rank\n";

    Vector<double> vector;

    Vector<size_t> rank;

    // Test

    vector.set(3);
    vector[0] = 0.0;
    vector[1] = -1.0;
    vector[2] = 1.0;

    rank = vector.calculate_less_rank();

    assert_true(vector.size() == 3, LOG);

    assert_true(rank[0] == 1, LOG);
    assert_true(rank[1] == 0, LOG);
    assert_true(rank[2] == 2, LOG);

    // Test

    vector.set(10);
    vector.randomize_normal();

    rank = vector.calculate_less_rank();

//    assert_true(vector.minimal_index() == rank.minimal_index(), LOG);
//    assert_true(vector.maximal_index() == rank.maximal_index(), LOG);

    //Test

    vector.set(2, 0.0);

    rank = vector.calculate_less_rank();

    assert_true(rank[0] == 0 || rank[0] == 1, LOG);
}


void VectorTest::test_calculate_greater_rank()
{
   cout << "test_calculate_greater_rank\n";

   Vector<size_t> vector ({2,4,5,6,7});
   Vector<size_t> check_vector = vector.calculate_greater_rank();
   Vector<size_t> solution({4,3,2,1,0});

   assert_true(check_vector == solution, LOG);
}


void VectorTest::test_calculate_sort_rank()
{
    cout << "test_calculate_sort_rank\n";

    Vector<size_t> vector({2, 5, 8, 9, 6, 7});
    Vector<size_t> check_vector;
    Vector<size_t> solution({2, 9, 8, 5, 6, 7});

    check_vector = vector.sort_rank({0, 3, 2, 1, 4, 5});

    assert_true(check_vector == solution, LOG);
}


void VectorTest::test_parse()
{
   cout << "test_parse\n";

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
   cout << "test_load\n";

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
   cout << "test_save\n";

   string file_name = "../data/vector.dat";

   Vector<int> v(2, 0);
   Vector<int> w(v);

   v.save(file_name);

   v.load(file_name);

   assert_true(v == w, LOG);
}


void VectorTest::test_count_equal_to()
{
    cout << "test_count_equal_to\n";

    Vector<double> vector ({1,2,6,8,9,7,4,5,5,6,6,5,2,3,1});

    assert_true(vector.count_equal_to(5) == 3, LOG);
}


void VectorTest::test_count_not_equal_to()
{
    cout <<"test_count_not_equal_to";

    size_t solution = 4;

    Vector<size_t> vector({1,2,5,4,7,4});

    assert_true(vector.count_not_equal_to(4) == solution, LOG);
}


void VectorTest::test_count_NAN()
{
//    cout << "test_count_NAN";

//    size_t size = 10;
//    size_t solution = 0;
//    size_t check;

//    Vector<size_t> vector(size);

//    for(size_t i = 0 ; i < size ; i++ )
//    {
//        vector[i] = i;
//    }

////    check = vector.count_NAN();

//    assert_true(check == solution, LOG);
}


void VectorTest::test_count_negative()
{
    cout << "test_count_negative\n";

    // Normal case

    Vector<double> vector({1, 2, 3, -4});

    size_t solution = 1;

    size_t elements;

    elements = vector.count_negative();

    assert_true(elements == solution, LOG);

    // All elements negative

    Vector<double> vector_negatives({-1, -2, -3, -4});

    size_t solution_neg = 4;

    size_t elements_neg;

    elements_neg = vector_negatives.count_negative();

    assert_true(elements_neg == solution_neg, LOG);

    // All elements positive

    Vector<double> vector_positives({1, 2, 3, 4});

    size_t solution_pos = 0;

    size_t elements_pos;

    elements_pos = vector_positives.count_negative();

    assert_true(elements_pos == solution_pos, LOG);

    // Empty  vector

    Vector<double> vector_empty({});

    size_t solution_empty = 0;

    size_t elements_empty;

    elements_empty = vector_empty.count_negative();

    assert_true(elements_empty == solution_empty, LOG);
}


void VectorTest::test_count_positive()
{
    cout << "test_count_positive\n";

    // Normal case

    Vector<double> vector({1, 2, 3, -4});

    size_t solution = 3;

    size_t elements;

    elements = vector.count_positive();

    assert_true(elements == solution, LOG);

    // All elements negative

    Vector<double> vector_negatives({-1, -2, -3, -4});

    size_t solution_neg = 0;

    size_t elements_neg;

    elements_neg = vector_negatives.count_positive();

    assert_true(elements_neg == solution_neg, LOG);

    // All elements positive

    Vector<double> vector_positives({1, 2, 3, 4});

    size_t solution_pos = 4;

    size_t elements_pos;

    elements_pos = vector_positives.count_positive();

    assert_true(elements_pos == solution_pos, LOG);

    // Empty  vector

    Vector<double> vector_empty({});

    size_t solution_empty = 0;

    size_t elements_empty;

    elements_empty = vector_empty.count_positive();

    assert_true(elements_empty == solution_empty, LOG);

    // All elements equal zero

    Vector<double> vector_zeros({0, 0, 0, 0});

    size_t solution_zeros = 4;

    size_t elements_zeros;

    elements_zeros = vector_zeros.count_positive();

    assert_true(elements_zeros == solution_zeros, LOG);
}


void VectorTest::test_count_integers()
{
    cout << "test_count_integers\n";

    // Normal case

    Vector<double> vector({1,5,9,6,5,6});

    size_t solution = 4;

    size_t elements;

    elements = vector.count_integers(6);

    assert_true(elements == solution, LOG);


    // All elements different

    Vector<double> vector_integer({-1, -2, -3, -4});

    size_t solution_different = 4;

    size_t elements_differents;

    elements_differents = vector_integer.count_integers(4);

    assert_true(elements_differents == solution_different, LOG);

    // All elements equal

    Vector<double> vector_equal({1, 1, 1, 1});

    size_t solution_equal = 1;

    size_t elements_equal;

    elements_equal = vector_equal.count_integers(4);

    assert_true(elements_equal == solution_equal, LOG);

    // Empty  vector

    Vector<double> vector_empty({});

    size_t solution_empty = 0;

    size_t elements_empty;

    elements_empty = vector_empty.count_integers(4);

    assert_true(elements_empty == solution_empty, LOG);

    // All elements equal zero

    Vector<double> vector_zeros({0, 0, 0, 0});

    size_t solution_zeros = 1;

    size_t elements_zeros;

    elements_zeros = vector_zeros.count_integers(4);

    assert_true(elements_zeros == solution_zeros, LOG);
}


void VectorTest::test_filter_equal_to()
{
    cout << "test_filter_equal\n";

    //Normal case

    Vector<double> vector({1, 2, 3, 6, 8, 9, 9, 5, 6});

    Vector<double> equal;

    equal = vector.filter_equal_to(1);

    Vector<double> solution({1});

    assert_true(equal == solution, LOG);

    //Element not included

    Vector<double> vector_not_included({1, 2, 3, 6, 8, 9, 9, 5, 6});

    Vector<double> equal_not_included;

    equal_not_included = vector_not_included.filter_equal_to(7);

    Vector<double> solution_not_included({});

    assert_true(equal_not_included == solution_not_included, LOG);
}


void VectorTest::test_filter_not_equal_to()
{
    cout << "test_filter_not_equal\n";

    //Normal case

    Vector<double> vector({1, 2, 3, 6, 8, 9, 9, 5, 6});

    Vector<double> different;

    different = vector.filter_not_equal_to(1);

    Vector<double> solution({2, 3, 6, 8, 9, 9, 5, 6});

    assert_true(different == solution, LOG);

    //Element not included

    Vector<double> vector_not_included({1, 2, 3, 6, 8, 9, 9, 5, 6});

    Vector<double> different_not_included;

    different_not_included = vector_not_included.filter_not_equal_to(7);

    Vector<double> solution_not_included({1, 2, 3, 6, 8, 9, 9, 5, 6});

    assert_true(different_not_included == solution_not_included, LOG);
}


void VectorTest::test_get_positive_elements()
{
    cout << "get_positive_elements\n";

    //Normal case

    size_t size = 9;
    Vector<double> vector({-1, 2, 3, 6, 8, -9, 9, -5, 6});
    Vector<double> vector_2(size);
    Vector<double> vector_solution({2, 3, 6, 8, 9, 6});

    vector_2 = vector.get_positive_elements();

    assert_true(vector_2 == vector_solution, LOG);

    // All elements negative

    Vector<double> vector_negatives({-1, -2, -3, -6, -8, -9, -9, -5, -6});
    Vector<double> vector_2_negatives(0);
    Vector<double> vector_solution_neg(0);

    vector_2_negatives = vector_negatives.get_positive_elements();

    assert_true(vector_2_negatives == vector_solution_neg, LOG);

    //All elements positive

    Vector<double> vector_positives({1, 2, 3, 6, 8, 9, 9, 5, 6});
    Vector<double> vector_2_positives(size);
    Vector<double> vector_solution_pos({1, 2, 3, 6, 8, 9, 9, 5, 6});

    vector_2_positives = vector_positives.get_positive_elements();

    assert_true(vector_2_positives == vector_solution_pos , LOG);

    //Empty vector
    size_t size_2 = 0;
    Vector<double> vector_empty(size_2);
    Vector<double> vector_2_empty(size_2);
    Vector<double> vector_solution_empty(size_2);

    vector_2_empty = vector_empty.get_positive_elements();

    assert_true(vector_2_empty == vector_solution_empty, LOG);
}


void VectorTest::test_get_negative_elements()
{
    cout << "get_negative_elements\n";

    //Normal case

    Vector<double> vector({-1, 2, 3, 6, 8, -9, 9, -5, 6});
    Vector<double> check_vector(vector.size());
    Vector<double> solution({-1, -9, -5});

    check_vector = vector.get_negative_elements();

    assert_true(check_vector == solution, LOG);

    // All elements negative

    Vector<double> vector_negatives({-1, -2, -3, -6, -8, -9, -9, -5, -6});
    Vector<double> check_vector_negatives(vector_negatives.size());
    Vector<double> vector_solution_neg({-1, -2, -3, -6, -8, -9, -9, -5, -6});

    check_vector_negatives = vector_negatives.get_negative_elements();

    assert_true(check_vector_negatives == vector_solution_neg, LOG);

    //All elements positive

    Vector<double> vector_positives({1, 2, 3, 6, 8, 9, 9, 5, 6});
    Vector<double> check_vector_positives(vector_positives.size());
    Vector<double> vector_solution_pos({});

    check_vector_positives = vector_positives.get_negative_elements();

    assert_true(check_vector_positives == vector_solution_pos , LOG);

    //Empty vector
    size_t size_2 = 0;
    Vector<double> vector_empty(size_2);
    Vector<double> check_vector_empty(size_2);
    Vector<double> vector_solution_empty(size_2);

    check_vector_empty = vector_empty.get_negative_elements();

    assert_true(check_vector_empty == vector_solution_empty, LOG);
}


void VectorTest::test_get_between_indices()
{
   cout << "test_get_between_indices\n";

   Vector<size_t> vector({2, 6, 9, 8, 25, 14, 3});
   Vector<size_t> solution({ 0, 1, 2, 3, 6});
   Vector<size_t> indices(vector.size());

   indices = vector.get_between_indices(2, 9);

   assert_true(indices == solution, LOG);
}


void VectorTest::test_get_indices_equal_to()
{
    cout << "test_get_indices_equal_to\n";

    //Normal case

    Vector <double> vector({1, 2, 3, 6, 8, 9, 9, 5, 6});
    Vector<size_t> vector_solution({5, 6});
    Vector<size_t> indices(vector_solution.size());

    indices = vector.get_indices_equal_to(9);

    assert_true(indices == vector_solution, LOG);

    //Element not included

    Vector<double> vector_2({1, 2, 3, 6, 8, 9, 9, 5, 6});
    Vector<size_t> vector_solution_2({});
    Vector<size_t> indices_2(vector_solution_2.size());

    indices = vector_2.get_indices_equal_to(7);

    assert_true(indices_2 == vector_solution_2, LOG);
}


void VectorTest::test_get_indices_not_equal_to()
{
    cout <<"test_get_indices_not_equal_to\n";

    //Normal case

    Vector <double> vector({1, 2, 3, 6, 8, 9, 9, 5, 6});
    Vector<size_t> vector_solution({0, 1, 2, 3, 4, 7, 8});
    Vector<size_t> indices(vector_solution.size());

    indices = vector.get_indices_not_equal_to(9);

    assert_true(indices == vector_solution, LOG);

    //Element not included

    Vector<double> vector_2({1, 2, 3, 6, 8, 9, 9, 5, 6});
    Vector<size_t> vector_solution_2({});
    Vector<size_t> indices_2(vector_solution_2.size());

    indices = vector_2.get_indices_not_equal_to(7);

    assert_true(indices_2 == vector_solution_2, LOG);
}


void VectorTest::test_filter_minimum_maximum()
{
    cout <<"test_filter_minimum_maximun\n";

    Vector<double> vector({1,2,3,4,5,6,7,8,9,10});
    Vector<double> solution({3,4,5,6});
    Vector<double> check_vector = vector.filter_minimum_maximum(3,6);

    assert_true(check_vector == solution, LOG);
}


void VectorTest::test_get_indices_less_than()
{
    cout << "test_get_indices_less_than\n";

    //Normal case

    Vector<double> vector({1, 2, 3, 6, 8, 9, 9, 5, 6});
    Vector<size_t> vector_solution({0,1});
    Vector<size_t> indices(vector_solution.size());

    indices = vector.get_indices_less_than(3);

    assert_true(indices == vector_solution, LOG);

    //Out of limits case

    Vector<double> vector_2({1, 2, 3, 6, 8, 9, 9, 5, 6});
    Vector<size_t> vector_solution_2({});
    Vector<size_t> indices_2(vector_solution_2.size());

    indices = vector_2.get_indices_less_than(12);

    assert_true(indices_2 == vector_solution_2, LOG);
}


void VectorTest::test_get_indices_greater_than()
{
   cout << "test_get_indices_grater_than\n";

    //Normal case

    Vector<double> vector({1, 2, 3, 6, 8, 9, 9, 5, 6});
    Vector<size_t> vector_solution({3, 4, 5, 6, 7, 8});
    Vector<size_t> indices(vector_solution.size());

    indices = vector.get_indices_greater_than(3);

    assert_true(indices == vector_solution, LOG);

    //Out of limits case

    Vector<double> vector_2({1, 2, 3, 6, 8, 9, 9, 5, 6});
    Vector<size_t> vector_solution_2({});
    Vector<size_t> indices_2(vector_solution_2.size());

    indices = vector_2.get_indices_greater_than(12);

    assert_true(indices_2 == vector_solution_2, LOG);
}


void VectorTest::test_count_greater_than()
{
    cout << "test_count_greater_than\n";

    //Normal case

    Vector<double> vector({-11, 2, 3, 6, 8, 9, -9, -5, 6});

    size_t size;

    size = vector.count_greater_than(8);

    assert_true(size == 1, LOG);

    //Element out of range

    Vector<double> vector_2({-11, 2, 3, 6, 8, 9, -9, -5, 6});

    size_t size_2;

    size_2 = vector_2.count_greater_than(10);

    assert_true(size_2 == 0, LOG);
}


void VectorTest::test_count_greater_equal_to()
{
    cout << "test_count_greater_equal_to\n";

    //Normal case

    Vector<size_t> vector({1, 2, 3, 4, 5, 6});
    size_t solution = 3;
    size_t check_vector;

    check_vector = vector.count_greater_equal_to(4);

    assert_true(check_vector == solution, LOG);

    //Equal elements

    Vector<size_t> vector1({1, 1, 1, 1});

    size_t solution1 = 4;
    size_t check_vector1;

    check_vector1 = vector1.count_greater_equal_to(1);

    assert_true(check_vector1 == solution1, LOG);

    // Empty vector

    Vector<size_t> vector2({});

    size_t solution2 = 0;
    size_t check_vector2;

    check_vector2 = vector2.count_greater_equal_to(1);

    assert_true(check_vector2 == solution2, LOG);
}


void VectorTest::test_count_less_than()
{
    cout << "test_count_less_than\n";

    //Normal case

    Vector<double> vector({-11, 2, 3, 6, 8, 9, -9, -5, 6});

    size_t size;

    size = vector.count_less_than(8);

    assert_true(size == 7 , LOG);

    //Element out of range

    Vector<double> vector_2({-11, 2, 3, 6, 8, 9, -9, -5, 6});

    size_t size_2;

    size_2 = vector_2.count_less_than(15);

    assert_true(size_2, LOG);
}


void VectorTest::test_count_less_equal_to()
{
    cout << "test_count_less_equal_to\n";

    //Normal case

    Vector<size_t> vector({1, 2, 3, 4, 5, 6});
    size_t solution = 3;
    size_t check_vector;

    check_vector = vector.count_less_equal_to(3);

    assert_true(check_vector == solution, LOG);

    //Equal elements

    Vector<size_t> vector1({1, 1, 1, 1});

    size_t solution1 = 4;
    size_t check_vector1;

    check_vector1 = vector1.count_less_equal_to(1);

    assert_true(check_vector1 == solution1, LOG);

    // Empty vector

    Vector<size_t> vector2({});

    size_t solution2 = 0;
    size_t check_vector2;

    check_vector2 = vector2.count_greater_equal_to(1);

    assert_true(check_vector2 == solution2, LOG);
}


void VectorTest::test_count_between()
{
    cout << "test_count_between\n";

    //Normal case

    Vector<size_t> vector({1, 2, 3, 4, 5, 6});

    size_t solution = 3;
    size_t check_vector;

    check_vector = vector.count_between(3,5);

    assert_true(check_vector == solution, LOG);

    // Equal elements

    Vector<size_t> vector1({1, 1, 1, 1});

    size_t solution1 = 4;
    size_t check_vector1;

    check_vector1 = vector1.count_between(0,3);

    assert_true(check_vector1 == solution1, LOG);

    // Empty vector

    Vector<size_t> vector2({});

    size_t solution2 = 0;
    size_t check_vector2;

    check_vector2 = vector2.count_between(3,5);

    assert_true(check_vector2 == solution2, LOG);
}


void VectorTest::test_get_indices_that_contains()
{
    cout << "test_get_indices_that_contains\n";

    Vector<string> vector({"aaa", "bbb", "ccc", "ddd"});
    Vector<size_t> solution({0});
    Vector<size_t> indices(solution.size());

    indices = vector.get_indices_that_contains("aaa");

    assert_true(indices == solution, LOG);
}


void VectorTest::test_get_indices_less_equal_to()
{
    cout << "test_get_indices_less_equal_to\n";

    //Normal case

    size_t size = 100;
    Vector<size_t> vector(size);
    vector.initialize_sequential();
    Vector<size_t> vector_solution({0, 1, 2, 3});
    Vector<size_t> indices(vector_solution.size());

    indices = vector.get_indices_less_equal_to(3);

    assert_true(indices == vector_solution, LOG);

    //Empty vector

    Vector<size_t> vector_2(0);
    Vector<size_t> vector_solution_2({});
    Vector<size_t> indices_2(vector_solution_2.size());

    indices_2 = vector_2.get_indices_less_equal_to(6);

    assert_true(indices_2 == vector_solution_2, LOG);
}


void VectorTest::test_get_indices_greater_equal_to()
{
    cout << "test_get_indices_greater_equal_to\n";

    //Normal case

    size_t size = 100;
    Vector<size_t> vector(size);
    vector.initialize_sequential();
    Vector<size_t> vector_solution({96, 97, 98, 99});
    Vector<size_t> indices(vector_solution.size());

    indices = vector.get_indices_greater_equal_to(96);

    assert_true(indices == vector_solution, LOG);

    //Empty vector

    Vector<size_t> vector_2(0);
    Vector<size_t> vector_solution_2({});
    Vector<size_t> indices_2(vector_solution_2.size());

    indices_2 = vector_2.get_indices_greater_equal_to(6);

    assert_true(indices_2 == vector_solution_2, LOG);
}


void VectorTest::test_perform_Box_Cox_transformation()
{
    cout << "test_perform_Box_Cox_transformation\n";

    size_t size = 50;
    Vector<double> x(size, 3.0);
    Vector<double> transformation(x.size());
    Vector<double> solution(size, 4.0);

    transformation = x.perform_Box_Cox_transformation(2);

    assert_true(transformation == solution, LOG);
}


void VectorTest::test_calculate_percentage()
{
    cout << "test_calculate_percentage\n";

    Vector<size_t> vector({1, 2, 3, 4, 5, 6});
    Vector<double> solution({4.7619, 9.5238, 14.286, 19.047, 23.809, 28.571});
    Vector<double> check_vector(vector.size());

    double sum;

    check_vector = vector.calculate_percentage(21);

    sum = check_vector.calculate_sum();

    assert_true(check_vector - solution < 0.1, LOG);
    assert_true(100 - sum < 0.5, LOG);

}


void VectorTest::test_get_first_index()
{
    cout << "test_get_first_index\n";

    size_t size = 100;
    Vector<int> vector(size);
    vector.initialize_sequential();
    size_t solution;
    size_t index = 3;

    solution = vector.get_first_index(3);

    assert_true(index == solution, LOG);

}


void VectorTest::test_tuck_in()
{
    cout << "test_tuck_in\n";

    Vector<size_t> vector({2, 5, 6, 8 ,7, 5});
    Vector<size_t> vector2({11, 12, 13});
    Vector<size_t> solution({2, 5, 11, 12, 13, 5});

    vector.embed(2, vector2);

    assert_true(vector == solution, LOG);
}


void VectorTest::test_delete_index()
{
    cout << "test_delete_index\n";

    Vector<size_t> vector({2, 5, 6, 7, 8});
    Vector<size_t> solution({2, 6, 7 , 8});
    Vector<size_t> check_vector(vector.size()-1);

    check_vector = vector.delete_index(1);

    assert_true(check_vector == solution, LOG);
}


void VectorTest::test_delete_indices()
{
    cout << "test_delete_indices\n";

    Vector<size_t> vector({2, 5, 6, 7, 8});
    Vector<size_t> solution({2, 7, 8});
    Vector<size_t> check_vector(vector.size()-1);

    check_vector = vector.delete_indices({1,2});

    assert_true(check_vector == solution, LOG);
}


void VectorTest::test_delete_value()
{
    cout << "test_delete_value\n";

    // Normal case

        Vector<size_t> vector({2, 5, 6, 7, 8});
        Vector<size_t> solution({2, 5, 7, 8});
        Vector<size_t> check_vector(vector.size()-1);

        check_vector = vector.delete_value(6);

        assert_true(check_vector == solution, LOG);

    // More than one equal element

        Vector<size_t> vector1({2, 6, 6, 7, 8});
        Vector<size_t> solution1({2, 7, 8});
        Vector<size_t> check_vector1(vector1.size()-1);

        check_vector1 = vector1.delete_value(6);

        assert_true(check_vector1 == solution1, LOG);

    // Equal elements

        Vector<size_t> vector2({6, 6, 6, 6, 6});
        Vector<size_t> solution2({});
        Vector<size_t> check_vector2(vector.size()-1);

        check_vector2 = vector2.delete_value(6);

        assert_true(check_vector2 == solution2, LOG);
}


void VectorTest::test_delete_values()
{
    cout << "test_delete_values\n";

    // Normal case

    Vector<size_t> vector({2, 5, 6, 7, 8});
    Vector<size_t> solution({2, 7, 8});
    Vector<size_t> check_vector(vector.size()-1);

    check_vector = vector.delete_values({5, 6});

    assert_true(check_vector == solution, LOG);

    // More than one equal element

    Vector<size_t> vector1({2, 6, 6, 7, 8});
    Vector<size_t> solution1({2, 8});
    Vector<size_t> check_vector1(vector1.size()-1);

    check_vector1 = vector1.delete_values({6, 7});

    assert_true(check_vector1 == solution1, LOG);

    // Equal elements

    Vector<size_t> vector2({6, 6, 6, 6, 6});
    Vector<size_t> solution2({});
    Vector<size_t> check_vector2(vector.size()-1);

    check_vector2 = vector2.delete_values({6});

    assert_true(check_vector2 == solution2, LOG);
}


void VectorTest::test_assemble()
{
    cout << "test_assemble\n";

    // Normal case

    Vector<size_t> vector1({1, 2, 3, 4});
    Vector<size_t> vector2({5, 6, 7, 8});
    Vector<size_t> solution({1, 2, 3, 4, 5, 6, 7, 8});
    Vector<size_t> chek_vector(vector1.size() + vector2.size());

    chek_vector = vector1.assemble(vector2);

    assert_true(chek_vector == solution, LOG);

    //Empty vector

    Vector<size_t> vector3({1, 2, 3, 4});
    Vector<size_t> vector4({});
    Vector<size_t> solution2({1, 2, 3, 4});
    Vector<size_t> chek_vector2(vector3.size() + vector4.size());

    chek_vector2 = vector3.assemble(vector4);

    assert_true(chek_vector2 == solution2, LOG);
}


void VectorTest::test_get_difference()
{
    cout << "test_get_difference\n";

    //Normal case

    Vector<size_t> vector1 ({1, 2, 3, 4});
    Vector<size_t> vector2 ({1, 2, 4});
    Vector<size_t> solution({3});
    Vector<size_t> check_vector(vector1.size());

    check_vector = vector1.get_difference(vector2);

    assert_true(check_vector == solution, LOG);

    // Equal vectors

    Vector<size_t> vector3 ({1, 2, 3, 4});
    Vector<size_t> vector4 ({1, 2, 3, 4});
    Vector<size_t> solution2({});
    Vector<size_t> check_vector2(vector3.size());

    check_vector2 = vector3.get_difference(vector4);

    assert_true(check_vector2 == solution2, LOG);

    //Differents vectors

    Vector<size_t> vector_1 ({1, 2, 3, 4});
    Vector<size_t> vector_2 ({5, 6, 7, 8});
    Vector<size_t> solution_2({1, 2, 3, 4});
    Vector<size_t> check_vector_2(vector_1.size());

    check_vector_2 = vector_1.get_difference(vector_2);

    assert_true(check_vector_2 == solution_2, LOG);

    // Empty vector

    Vector<size_t> vector_3 ({1, 2, 3, 4});
    Vector<size_t> vector_4 ({});
    Vector<size_t> solution_3({1, 2, 3, 4});
    Vector<size_t> check_vector_3(vector_3.size());

    check_vector_3 = vector_3.get_difference(vector_4);

    assert_true(check_vector_3 == solution_3, LOG);
}


void VectorTest::test_get_union()
{
    cout << "test_get_union\n";

    // Normal case

    Vector<size_t> vector_1 ({1, 2, 3, 4});
    Vector<size_t> vector_2 ({5, 6, 7, 8});
    Vector<size_t> solution_2({1, 2, 3, 4, 5, 6, 7, 8});
    Vector<size_t> check_vector_2(vector_1.size() + vector_2.size());

    check_vector_2 = vector_1.get_union(vector_2);

    assert_true(check_vector_2 == solution_2, LOG);

    // Equal vectors

    Vector<size_t> vector1 ({1, 2, 3, 4});
    Vector<size_t> vector2 ({1, 2, 3, 4});
    Vector<size_t> solution2({1, 2, 3, 4});
    Vector<size_t> check_vector2(vector1.size() + vector2.size());

    check_vector2 = vector1.get_union(vector2);

    assert_true(check_vector2 == solution2, LOG);

    // Empty vector

    Vector<size_t> vector ({1, 2, 3, 4});
    Vector<size_t> vector_ ({});
    Vector<size_t> solution_({1, 2, 3, 4});
    Vector<size_t> check_vector_(vector.size() + vector_.size());

    check_vector_ = vector.get_union(vector_);

    assert_true(check_vector_ == solution_, LOG);
}


void VectorTest::test_get_intersection()
{
    cout << "test_get_intersection\n";

    //Normal case

    Vector<size_t> vector1 ({1, 2, 3, 4});
    Vector<size_t> vector2 ({1, 5, 6, 4});
    Vector<size_t> solution2({1, 4});
    Vector<size_t> check_vector2(vector1.size());

    check_vector2 = vector1.get_intersection(vector2);

    assert_true(check_vector2 == solution2, LOG);

    //Equal vectorts

    Vector<size_t> vector_1 ({1, 2, 3, 4});
    Vector<size_t> vector_2 ({1, 2, 3, 4});
    Vector<size_t> solution_2({1, 2, 3, 4});
    Vector<size_t> check_vector_2(vector_1.size());

    check_vector_2 = vector_1.get_intersection(vector_2);

    assert_true(check_vector_2 == solution_2, LOG);

    //Differents vectors

    Vector<size_t> vector ({1, 2, 3, 4});
    Vector<size_t> vector_ ({5, 6, 7});
    Vector<size_t> solution_({});
    Vector<size_t> check_vector_(vector1.size());

    check_vector_ = vector.get_intersection(vector_);

    assert_true(check_vector_ == solution_, LOG);
}


void VectorTest::test_get_unique_elements()
{
    cout << "test_get_unique_elements\n";

    // Normal case

    Vector<size_t> vector({1, 2, 3, 5, 4, 3});
    Vector<size_t> solution({1,2,3,4,5});
    Vector<size_t> check_vector(vector.size());

    check_vector = vector.get_unique_elements();

    assert_true(check_vector == solution, LOG);

    // Equal elements

    Vector<size_t> vector1({1, 1, 1, 1});
    Vector<size_t> solution1({1});
    Vector<size_t> check_vector1(vector1.size());

    check_vector1 = vector1.get_unique_elements();

    assert_true(check_vector1 == solution1, LOG);
}


void VectorTest::test_count_unique()
{
    cout << "test_count_unique\n";

    // Normal case

    Vector<size_t> vector({1, 2, 2, 2, 3, 4, 3});
    Vector<size_t> solution({1, 3, 2, 1});
    Vector<size_t> check_vector;

    check_vector = vector.count_unique();

    assert_true(check_vector == solution, LOG);

    // Equal elements

    Vector<size_t> vector1({1, 1, 1, 1, 1});
    Vector<size_t> solution1({5});
    Vector<size_t> check_vector1;

    check_vector1 = vector1.count_unique();

    assert_true(check_vector1 == solution1, LOG);

    // Empty vector

    Vector<size_t> vector2({});
    Vector<size_t> solution2({});
    Vector<size_t> check_vector2;

    check_vector2 = vector2.count_unique();

    assert_true(check_vector2 == solution2, LOG);
}


void VectorTest::test_sort_ascending_indices()
{
    cout << "test_sort_ascending_indices\n";

    // Normal case

    Vector<double> vector({2, 5, 6, 3});
    Vector<size_t> indices(vector.size());
    Vector<size_t> solution({0, 3, 1, 2});

    indices = vector.sort_ascending_indices();

    assert_true(indices == solution, LOG);

    // All the same elements

    Vector<size_t> vector_2(5, 1);
    Vector<size_t> indices_2(vector_2.size());
    Vector<size_t> solution_2(vector_2.size());
    solution_2.initialize(1);
    solution_2.initialize_sequential();

    indices_2 = vector_2.sort_ascending_indices();

    assert_true(indices_2 == solution_2, LOG);
}


void VectorTest::test_calculate_top_string()
{
    cout << "test_calculate_top_string\n";

    //Normal case

    Vector<string> vector({"Hello", "Hello", "GoodBye", "GoodBye", "GoodBye", "Home"});
    Vector<string> solution({"GoodBye", "Hello", "Home"});
    Vector<string> check_vector(vector.size());

    check_vector = vector.calculate_top_string(10);

    assert_true(check_vector == solution, LOG);

    //Equal number of elements

    Vector<string> vector1({"Alba", "Claudia", "Lucas"});
    Vector<string> solution1({"Alba", "Claudia", "Lucas"});
    Vector<string> check_vector1(vector1.size());

    check_vector1 = vector1.calculate_top_string(10);

    assert_true(check_vector1 == solution1, LOG);
}


void VectorTest::test_calculate_top_number()
{
    cout << "test_calculate_top_number\n";

    //Normal case

    Vector<double> vector({1, 2, 2, 3, 3, 3});
    Vector<double> solution({3, 2, 1});
    Vector<double> check_vector(vector.size());

    check_vector = vector.calculate_top_number(10);

   assert_true(check_vector == solution, LOG);

    //Equal number of elements

    Vector<double> vector1({1, 2, 2, 1, 3, 3});
    Vector<double> solution1({1, 2, 3});
    Vector<double> check_vector1(vector1.size());

    check_vector1 = vector1.calculate_top_number(10);

    assert_true(check_vector1 == solution1, LOG);
}


void VectorTest::test_sort_descending_indices()
{
    cout << "test_sort_descending_indices\n";

    // Normal case

    Vector<double> vector({2, 5, 6, 3});
    Vector<size_t> indices(vector.size());
    Vector<size_t> solution({2, 1, 3, 0});

    indices = vector.sort_descending_indices();

    assert_true(indices == solution, LOG);

    // All the same elements

    Vector<double> vector2({2, 2, 2, 2});
    Vector<size_t> indices2(vector.size());
    Vector<size_t> solution2({0, 1, 2, 3});

    indices2 = vector2.sort_descending_indices();

    assert_true(indices2 == solution2, LOG);
}


void VectorTest::test_calculate_lower_indices()
{
    cout << "test_calculate_lower_indices\n";

    size_t size = 10;
    size_t size_2 = 4;
    Vector<double> vector(size);
    Vector<size_t> solution(size_2);
    Vector<size_t> indices(size_2);
    solution.initialize_sequential();
    indices = vector.calculate_lower_indices(size_2);

    assert_true(indices == solution, LOG);

    size_t size_3 = 0;

    Vector<size_t> solution_2(size_3);
    Vector<size_t> indices_2(size_3);
    solution_2.initialize_sequential();
    indices_2 = vector.calculate_lower_indices(size_3);

    assert_true(indices_2 == solution_2, LOG);
}


void VectorTest::test_calculate_lower_values()
{
    cout << "test_calculate_lower_values\n";

    //Normal case

    size_t size_solution = 3;
    Vector<size_t> vector({1, 2, 3, 4});
    Vector<size_t> values(size_solution);
    Vector<size_t> solution(size_solution);
    solution.set(vector.get_subvector(0, 2));
    vector.get_subvector(0, 2);

    values = vector.calculate_lower_values(size_solution);

    assert_true(values == solution, LOG);
}


void VectorTest::test_sort_ascending_values()
{
    cout << "test_sort_ascending_values\n";

    Vector<size_t> vector(5);
    vector.initialize_sequential();
    Vector<size_t> solution(5);
    solution.initialize_sequential();

    vector.sort_ascending_values();

    assert_true(vector == solution, LOG);
}


void VectorTest::test_count_contains()
{
    cout << "test_count_contains";

    Vector<string> vector({"home","hostel","hearth","hearthbeat"});

    size_t solution = 2;

    size_t count = vector.count_contains("ho");

    assert_true(solution = count, LOG);
}


void VectorTest::test_merge()
{
    cout << "test_merged\n";

    Vector<string> vector ({"home", "hostel", "hearth"});
    Vector<string> vector_merged(vector.size());
    const string word_merge("hello");
    Vector<string> solution({"home-hello", "hostel-hello", "hearth-hello"});

    vector_merged = vector.merge(word_merge, '-');

    assert_true(vector_merged == solution, LOG);

}


void VectorTest::test_sort_descending_values()
{
    cout << "test_sort_descending_values\n";

    Vector<size_t> vector({1,2,3,4,5});
    Vector<size_t> solution({5,4,3,2,1});

    assert_true(vector.sort_descending_values() == solution, LOG);
}


void VectorTest::test_replace_value()
{
    cout << "test_replace_value\n";

    Vector<int> vector({5,6,2,8,4,5});

    vector.replace_value(6,9);

    Vector<int> solution({5,9,2,8,4,5});

    assert_true(vector == solution, LOG);
}


void VectorTest::test_filter_positive()
{
    cout << "test_filter_positive \n";

    //Normal case

    Vector<double> vector({ 1, 5, 3, 4, -4});
    Vector<double> solution({ 1, 5 , 3, 4, 0});
    Vector<double> check_vector = vector.filter_positive();

    assert_true(check_vector == solution, LOG);

    //Elements positives

    Vector<double> vector_positive({ 1, 5, 3, 4, 4});
    Vector<double> solution_positive({ 1, 5 , 3, 4, 4});
    Vector<double> check_vector_positive = vector_positive.filter_positive();

    assert_true(check_vector == solution, LOG);

    // Elements negatives

    Vector<double> vector_negative({ -1, -5, -3, -4, -4});
    Vector<double> solution_negative({ 0, 0 , 0, 0, 0});
    Vector<double> check_vector_negative = vector_negative.filter_positive();

    assert_true(check_vector_negative == solution_negative, LOG);

    //Elements cero

    Vector<double> vector_cero({ 0, 0, 0, 0, 0});
    Vector<double> solution_cero({0, 0, 0, 0, 0});
    Vector<double> check_vector_cero = vector_cero.filter_positive();

    assert_true(check_vector_cero == solution_cero, LOG);
}


void VectorTest::test_filter_negative()
{
    cout << "test_filter_negative \n";

    //Normal case

    Vector<double> vector({ -1, -5, 3, 4, -4});
    Vector<double> solution({ -1, -5 , 0, 0, -4});
    Vector<double> check_vector = vector.filter_negative();

    assert_true(check_vector == solution, LOG);

    //Elements positives

    Vector<double> vector_negative({ 1, 5, 3, 4, 4});
    Vector<double> solution_negative({ 0, 0 , 0, 0, 0});
    Vector<double> check_vector_negative = vector_negative.filter_negative();

    assert_true(check_vector_negative == solution_negative, LOG);

    // Elements negatives

    Vector<double> vector_positive({ -1, -5, -3, -4, -4});
    Vector<double> solution_positive({ -1, -5, -3, -4, -4});
    Vector<double> check_vector_positive = vector_positive.filter_negative();

    assert_true(check_vector_negative == solution_negative, LOG);

    //Elements cero

    Vector<double> vector_cero({ 0, 0, 0, 0, 0});
    Vector<double> solution_cero({0, 0, 0, 0, 0});
    Vector<double> check_vector_cero = vector_cero.filter_negative();

    assert_true(check_vector_cero == solution_cero, LOG);
}


void VectorTest:: test_count_dates()
{
//    cout << "test_count_dates";

//    size_t date = count_dates({6,6,2019},{6,7,2019});
//    size_t solution;

//    assert_true(solution == date, LOG);
}


void VectorTest::run_test_case()
{
   cout << "Running vector test case...\n";

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

   // File operation

    test_filter_positive();
    test_tuck_in();

   // Get methods

   test_get_display();

   test_get_subvector();
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

   test_delete_index();
   test_delete_indices();

   test_delete_value();
   test_delete_values();

   test_assemble();

   test_get_difference();
   test_get_union();
   test_get_intersection();

   test_get_unique_elements();

   test_count_unique();

   test_calculate_top_string();
   test_calculate_top_number();

   // Initialization methods

   test_initialize();
   test_initialize_first();
   test_initialize_sequential();
   test_randomize_uniform();
   test_randomize_normal();
   test_randomize_binary();

   test_fill_from();

   // Checking methods

   test_contains();
   test_contains_greater_than();

   test_is_in();

   test_is_constant();
   test_is_constant_string();

   test_is_crescent();
   test_is_decrescent();

   test_has_same_elements();

   test_is_binary();
   test_is_binary_0_1();

   test_is_positive();
   test_is_negative();

   test_is_integer();

   test_check_period();
   test_get_reverse();

   // Replace methods

   test_replace_value();

   // Mathematical methods

   test_dot_vector();
   test_dot_matrix();

   test_calculate_sum();
   test_calculate_partial_sum();
   test_calculate_product();

   test_calculate_cumulative_index();

   test_calculate_less_rank();
   test_calculate_greater_rank();
   test_calculate_sort_rank();

   // Parsing methods

   test_parse();

   //Count methods

   test_count_equal_to();
   test_count_not_equal_to();

//   test_count_NAN();

   test_count_negative();
   test_count_positive();
   test_count_integers();

   test_filter_equal_to();
   test_filter_not_equal_to();

   test_get_positive_elements();
   test_get_negative_elements();

   test_get_between_indices();

   test_get_indices_equal_to();
   test_get_indices_not_equal_to();

   test_filter_minimum_maximum();

   test_get_indices_equal_to();
   test_get_indices_less_than();
   test_get_indices_greater_than();

   test_count_greater_than();
   test_count_greater_equal_to();
   test_count_less_than();
   test_count_less_equal_to();
   test_count_between();

   test_get_indices_less_than();
   test_get_indices_greater_than();
   test_get_indices_less_equal_to();
   test_get_indices_greater_equal_to();

   test_perform_Box_Cox_transformation();

   test_calculate_percentage();

   test_get_indices_that_contains();

   test_count_contains();

   // Descriptives methods

   test_get_first_index();

   // Filter methods

   test_get_first();
   test_get_last();
   test_get_before_last();
   test_delete_first();
   test_delete_last();

   test_get_integer_elements();
   test_get_integers();

   test_filter_negative();
   test_filter_positive();

   // Ranks methods

   test_sort_ascending_indices();
   test_calculate_lower_indices();
   test_calculate_lower_values();
   test_sort_ascending_values();
   test_sort_descending_values();
   test_sort_descending_indices();
   test_calculate_less_rank();
//   test_calculate_greater_rank();

   // Serialization methods

   test_save();

   test_load();


   cout << "End vector test case\n";
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
