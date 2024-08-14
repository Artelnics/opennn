//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   T E S T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "data_set_test.h"

#include "../opennn/tensors.h"

namespace opennn
{

DataSetTest::DataSetTest() : UnitTesting()
{
    data_set.set_display(false);
}


DataSetTest::~DataSetTest()
{
}


void DataSetTest::test_constructor()
{
    cout << "test_constructor\n";

    // Default constructor

    DataSet data_set_1;

    assert_true(data_set_1.get_variables_number() == 0, LOG);
    assert_true(data_set_1.get_samples_number() == 0, LOG);

    // Samples and variables number constructor

    DataSet data_set_2(1, 2);

    assert_true(data_set_2.get_samples_number() == 1, LOG);
    assert_true(data_set_2.get_variables_number() == 2, LOG);

    // Inputs, targets and samples numbers constructor

    DataSet data_set_3(1, 1, 1);

    assert_true(data_set_3.get_variables_number() == 2, LOG);
    assert_true(data_set_3.get_samples_number() == 1, LOG);
    assert_true(data_set_3.get_target_variables_number() == 1,LOG);
    assert_true(data_set_3.get_input_variables_number() == 1,LOG);
}


void DataSetTest::test_destructor()
{
    cout << "test_destructor\n";

    DataSet* data_set = new DataSet(1, 1, 1);
    delete data_set;
}


void DataSetTest::test_calculate_variables_descriptives()
{
    cout << "test_calculate_variables_descriptives\n";

    Tensor<Descriptives, 1> variables_descriptives;

    ofstream file;
    string data_string;

    const string data_source_path = "../data/data.dat";

    // Test

    data_set.set(1, 1);

    data_set.set_data_constant(type(0));

    variables_descriptives = data_set.calculate_variables_descriptives();

    assert_true(variables_descriptives.size() == 1, LOG);

    assert_true(abs(variables_descriptives[0].minimum) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(variables_descriptives[0].maximum) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(variables_descriptives[0].mean) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(variables_descriptives[0].standard_deviation) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    data_set.set(2, 2, 2);

    data_set.set_data_constant(type(0));

    variables_descriptives = data_set.calculate_variables_descriptives();

    assert_true(variables_descriptives.size() == 4, LOG);

    assert_true(variables_descriptives[0].minimum < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(variables_descriptives[0].maximum < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(variables_descriptives[0].mean < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(variables_descriptives[0].standard_deviation < type(NUMERIC_LIMITS_MIN), LOG);

    assert_true(variables_descriptives[1].minimum < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(variables_descriptives[1].maximum < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(variables_descriptives[1].mean < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(variables_descriptives[1].standard_deviation < type(NUMERIC_LIMITS_MIN), LOG);

    assert_true(variables_descriptives[2].minimum < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(variables_descriptives[2].maximum < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(variables_descriptives[2].mean < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(variables_descriptives[2].standard_deviation < type(NUMERIC_LIMITS_MIN), LOG);

    assert_true(variables_descriptives[3].minimum < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(variables_descriptives[3].maximum < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(variables_descriptives[3].mean < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(variables_descriptives[3].standard_deviation < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    inputs_number = 2;
    targets_number = 1;
    samples_number = 3;

    data.resize(samples_number,inputs_number + targets_number);
    data.setValues({{type(-1000),type(2),type(0)},{type(1),type(4),type(2)},{type(1),type(4),type(0)}});

    data_set.set_data(data);

    variables_descriptives = data_set.calculate_variables_descriptives();

    assert_true(variables_descriptives.size() == 3, LOG);

    assert_true(abs(variables_descriptives[0].minimum + type(1000)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(variables_descriptives[1].minimum - type(2)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(variables_descriptives[2].minimum) < type(NUMERIC_LIMITS_MIN), LOG);

    assert_true(abs(variables_descriptives[0].maximum - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(variables_descriptives[1].maximum - type(4)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(variables_descriptives[2].maximum - type(2)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void DataSetTest::test_calculate_input_variables_descriptives()
{
    cout << "test_calculate_input_variables_descriptives\n";

    Tensor<Descriptives, 1> input_variables_descriptives;

    Tensor<Index, 1> indices;

    // Test

    data.resize(2, 3);
    data.setValues({{type(1), type(2.0), type(3.0)},
                    {type(1), type(2.0), type(3.0)}});

    data_set.set(data);

    indices.resize(2);
    indices.setValues({0, 1});

    input_variables_descriptives = data_set.calculate_input_variables_descriptives();

    assert_true(input_variables_descriptives[0].mean - type(2.0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(input_variables_descriptives[0].standard_deviation - type(1)< type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(input_variables_descriptives[0].minimum - type(1) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(input_variables_descriptives[0].maximum - type(3.0) < type(NUMERIC_LIMITS_MIN), LOG);
}


void DataSetTest::test_calculate_data_distributions()
{
    cout << "test_calculate_data_distributions\n";

    Tensor<Histogram, 1> histograms;

    data_set.set();

    // Test

    data.resize(3, 3);

    data.setValues({{type(2),type(2),type(1)},
                    {type(1),type(1),type(1)},
                    {type(1),type(2),type(2)}});

    data_set.set_data(data);

    histograms = data_set.calculate_raw_variables_distribution(2);

    assert_true(histograms.size() == 3, LOG);

    assert_true(abs( histograms(0).frequencies(0) - 2 )  < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs( histograms(1).frequencies(0) - 1 ) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs( histograms(2).frequencies(0) - 2 ) < type(NUMERIC_LIMITS_MIN), LOG);

    assert_true(abs( histograms(0).centers(0) - 1 )  < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs( histograms(1).centers(0) - 1 ) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs( histograms(2).centers(0) - 1 ) < type(NUMERIC_LIMITS_MIN), LOG);
}


void DataSetTest::test_filter_data()
{
    cout << "test_filter_data\n";

    Index samples_number;
    Index inputs_number;
    Index targets_number;

    Tensor<type, 1> minimums;
    Tensor<type, 1> maximums;

    // Test

    samples_number = 2;
    inputs_number = 1;
    targets_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(type(1));

    minimums.resize(2);
    minimums.setValues({ type(2), type(0)});

    maximums.resize(2);
    maximums.setValues({ type(2), type(0.5)});

    data_set.filter_data(minimums, maximums);

    assert_true(data_set.get_sample_use(0) == DataSet::SampleUse::Unused, LOG);
    assert_true(data_set.get_sample_use(1) == DataSet::SampleUse::Unused, LOG);
}


void DataSetTest::test_scale_data()
{
    cout << "test_scale_data\n";

    Tensor<Descriptives, 1> data_descriptives;
    Tensor<type, 2> scaled_data;

    // Test

    data.resize(2,2);
    data.setValues({{type(1), type(2)},
                    {type(3), type(4)}});

    data_set.set_data(data);

    data_set.set_raw_variables_scalers(Scaler::NoScaling);
    data_descriptives = data_set.scale_data();

    scaled_data = data_set.get_data();

    assert_true(are_equal(scaled_data, data), LOG);

    // Test

    data_set.set_raw_variables_scalers(Scaler::MinimumMaximum);
    data_descriptives = data_set.scale_data();

    scaled_data = data_set.get_data();

    assert_true(abs(scaled_data(0) - type(-1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(scaled_data(1) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(scaled_data(2) - type(-1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(scaled_data(3) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
}


void DataSetTest::test_unuse_constant_raw_variables()
{
    cout << "test_unuse_constant_columns\n";

    // Test

    samples_number = 3;
    inputs_number = 2;
    targets_number = 1;

    data.resize(samples_number, inputs_number + targets_number);
    data.setValues({{type(1),type(2),type(0)},{type(1),type(2),type(1)},{type(1),type(2),type(2)}});

    data_set.set_data(data);
    data_set.set_has_header(false);
    data_set.set_constant_raw_variables();

    data_set.unuse_constant_raw_variables();

    assert_true(data_set.get_input_raw_variables_number() == 0, LOG);
    assert_true(data_set.get_target_raw_variables_number() == 1, LOG);
}


void DataSetTest::test_calculate_target_distribution()
{
    cout << "test_calculate_target_distribution\n";

    Tensor<Index, 1> target_distribution;

    // Test two classes

    data.resize(5, 5);

    data.setValues({{type(2),type(5),type(6),type(9),type(0)},
                    {type(2),type(9),type(1),type(9),type(0)},
                    {type(2),type(9),type(1),type(9),type(NAN)},
                    {type(6),type(5),type(6),type(7),type(1)},
                    {type(0),type(1),type(0),type(1),type(1)}});

    data_set.set(data);

    target_variables_indices.resize(1);
    target_variables_indices.setValues({4});

    input_variables_indices.resize(4);
    input_variables_indices.setValues({0, 1, 2, 3});

    data_set.set_input_target_raw_variables(input_variables_indices, target_variables_indices);
    target_distribution = data_set.calculate_target_distribution();

    Tensor<Index, 1> solution(2);
    solution(0) = 2;
    solution(1) = 2;

    assert_true(target_distribution(0) == solution(0), LOG);
    assert_true(target_distribution(1) == solution(1), LOG);

    // Test more two classes

    data.resize(5, 9);
    data.setZero();

    data.setValues({{type(2),type(5),type(6),type(9),type(8),type(7),type(1),type(0),type(0)},
                    {type(2),type(9),type(1),type(9),type(4),type(5),type(0),type(1),type(0)},
                    {type(6),type(5),type(6),type(7),type(3),type(2),type(0),type(0),type(1)},
                    {type(6),type(5),type(6),type(7),type(3),type(2),type(0),type(0),type(1)},
                    {type(0),type(NAN),type(1),type(0),type(2),type(2),type(0),type(1),type(0)}});

    data_set.set_data(data);

    target_variables_indices.resize(3);
    target_variables_indices.setValues({6,7,8});

    input_variables_indices.resize(2);
    input_variables_indices.setValues({0, 1});

    data_set.set_input_target_raw_variables(input_variables_indices, target_variables_indices);

    target_distribution = data_set.calculate_target_distribution();

    assert_true(target_distribution[0] == 1, LOG);
    assert_true(target_distribution[1] == 2, LOG);
    assert_true(target_distribution[2] == 2, LOG);
}


void DataSetTest::test_calculate_Tukey_outliers()
{
    cout << "test_calculate_Tukey_outliers\n";

    Tensor<type, 1> sample;

    Tensor<Tensor<Index, 1>, 1> outliers_indices;

    // Test

    data_set.set(100, 5, 1);
    data_set.set_data_random();

    outliers_indices = data_set.calculate_Tukey_outliers(type(1.5));

    assert_true(outliers_indices.size() == 2, LOG);
    assert_true(outliers_indices(0)(0) == 0, LOG);
}


void DataSetTest::test_read_csv()
{
/*
    cout << "test_read_csv\n";

    // Test

    data_set.set(2, 2, 2);
    data_set.set_separator(',');
    data_source_path = "../data/data.dat";
    data_set.set_data_source_path(data_source_path);
    data_set.set_data_constant(type(0));
    data_set.save_data();

    data_set.read_csv();
    data = data_set.get_data();

    assert_true(is_equal(data, type(0)), LOG);

    // Test

    data_set.set_separator('\t');
    data_string = "\n\n\n   1\t2   \n\n\n   3\t4   \n\n\n    5\t6    \n\n\n";

    file.open(data_source_path.c_str());
    file << data_string;
    file.close();

    data_set.set();
    data_set.set_data_source_path(data_source_path);
    data_set.read_csv();
    data = data_set.get_data();

    assert_true(data.dimension(0) == 3, LOG);
    assert_true(data.dimension(1) == 2, LOG);

    assert_true(abs(data(0, 0) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(data(0, 1) - type(2)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(data(1, 0) - type(3)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(data(1, 1) - type(4)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(data(2, 0) - type(5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(data(2, 1) - type(6)) < type(NUMERIC_LIMITS_MIN), LOG);

    assert_true(data_set.get_samples_number() == 3, LOG);
    assert_true(data_set.get_variables_number() == 2, LOG);

    // Test

    data_set.set_separator('\t');
    data_string = "\n\n\n   1\t2   \n\n\n   3\t4   \n\n\n   5\t6  \n\n\n";

    file.open(data_source_path.c_str());
    file << data_string;
    file.close();

    data_set.set();
    data_set.set_data_source_path(data_source_path);
    data_set.read_csv();

    data = data_set.get_data();

    assert_true(data.dimension(0) == 3, LOG);
    assert_true(data.dimension(1) == 2, LOG);

    assert_true(abs(data(0,0) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(data(0,1) - type(2)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(data(1,0) - type(3)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(data(1,1) - type(4)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(data(2,0) - type(5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(data(2,1) - type(6)) < type(NUMERIC_LIMITS_MIN), LOG);

    assert_true(data_set.get_samples_number() == 3, LOG);
    assert_true(data_set.get_variables_number() == 2, LOG);

    // Test

    data_set.set();
    data_set.set_has_header(true);
    data_set.set_separator(' ');
    data_string = "x y\n"
                  "1 2\n"
                  "3 4\n"
                  "5 6";

    file.open(data_source_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    data = data_set.get_data();

    assert_true(data_set.get_header_line(), LOG);
    assert_true(data_set.get_variable_name(0) == "x", LOG);
    assert_true(data_set.get_variable_name(1) == "y", LOG);

    assert_true(data.dimension(0) == 3, LOG);
    assert_true(data.dimension(1) == 2, LOG);

    assert_true((data(0,0) - 1.0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(0,1) - 2.0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(1,0) - 3.0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(1,1) - 4.0) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    data_set.set_has_header(true);
    data_set.set_separator(',');

    data_string = "\tx\t,\ty\n"
                  "\t1\t,\t2\n"
                  "\t3\t,\t4";

    file.open(data_source_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    data = data_set.get_data();

    assert_true(data_set.get_variable_name(0) == "x", LOG);
    assert_true(data_set.get_variable_name(1) == "y", LOG);

    assert_true((data(0,0) - 1.0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(0,1) - 2.0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(1,0) - 3.0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(1,1) - 4.0) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    data_set.set_has_header(true);
    data_set.set_separator(',');

    data_string = "x , y\n"
                  "1 , 2\n"
                  "3 , 4\n";

    file.open(data_source_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    data = data_set.get_data();

    assert_true(data_set.get_variable_name(0) == "x", LOG);
    assert_true(data_set.get_variable_name(1) == "y", LOG);

    assert_true((data(0,0) - 1.0 ) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(0,1) - 2.0 ) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(1,0) - 3.0 ) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(1,1) - 4.0 ) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    data_set.set_has_header(false);
    data_set.set_separator(',');
    data_string =
            "5.1,3.5,1.4,0.2,Iris-setosa\n"
            "7.0,3.2,4.7,1.4,Iris-versicolor\n"
            "7.0,3.2,4.7,1.4,Iris-versicolor\n"
            "6.3,3.3,6.0,2.5,Iris-virginica";

    file.open(data_source_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    assert_true(data_set.get_samples_number() == 4, LOG);
    assert_true(data_set.get_variables_number() == 7, LOG);

    // Test

    data_set.set_has_header(false);
    data_set.set_separator(',');
    data_string =
            "5.1,3.5,1.4,0.2,Iris-setosa\n"
            "7.0,3.2,4.7,1.4,Iris-versicolor\n"
            "7.0,3.2,4.7,1.4,Iris-versicolor\n"
            "6.3,3.3,6.0,2.5,Iris-virginica\n";

    file.open(data_source_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    assert_true(data_set.get_variables_number() == 7, LOG);
    assert_true(data_set.get_input_variables_number() == 4, LOG);
    assert_true(data_set.get_target_variables_number() == 3, LOG);
    assert_true(data_set.get_samples_number() == 4, LOG);

    data = data_set.get_data();

    assert_true((data(0,0) - type(5.1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(0,4) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(0,5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(0,6)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(1,4)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(1,5) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(1,6)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(2,4)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(2,5) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(2,6)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(3,4)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(3,5)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true((data(3,6) - type(1)) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    data_set.set_has_header(true);
    data_set.set_separator(',');
    data_set.set_missing_values_label("NaN");
    data_string =
            "sepal length,sepal width,petal length,petal width,class\n"
            "NaN,3.5,1.4,0.2,Iris-setosa\n"
            "7.0,3.2,4.7,1.4,Iris-versicolor\n"
            "7.0,3.2,4.7,1.4,Iris-versicolor\n"
            "6.3,3.3,6.0,2.5,Iris-virginica\n"
            "0.0,0.0,0.0,0.0,NaN\n";

    file.open(data_source_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    assert_true(data_set.get_variables_number() == 7, LOG);
    assert_true(data_set.get_input_variables_number() == 4, LOG);
    assert_true(data_set.get_target_variables_number() == 3, LOG);

    assert_true(data_set.get_variable_name(0) == "sepal length", LOG);
    assert_true(data_set.get_variable_name(1) == "sepal width", LOG);
    assert_true(data_set.get_variable_name(2) == "petal length", LOG);
    assert_true(data_set.get_variable_name(3) == "petal width", LOG);
    assert_true(data_set.get_variable_name(4) == "Iris-setosa", LOG);
    assert_true(data_set.get_variable_name(5) == "Iris-versicolor", LOG);
    assert_true(data_set.get_variable_name(6) == "Iris-virginica", LOG);

    assert_true(data_set.get_samples_number() == 5, LOG);

    data = data_set.get_data();

    assert_true(data(0,4) - type(1) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(data(0,5) - type(0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(data(0,6) - type(0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(data(1,4) - type(0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(data(1,5) - type(1) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(data(1,6) - type(0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(data(2,4) - type(0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(data(2,5) - type(1) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(data(2,6) - type(0) < type(NUMERIC_LIMITS_MIN), LOG);
*/
    // Test
/*
    data_set.set_has_header(false);
    data_set.set_separator(' ');
    data_string = "1 2\n3 4\n5 6";

    file.open(data_source_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();
    data_set.set_variable_name(0, "x");
    data_set.set_variable_name(1, "y");

    data_set.save("../data/data_set.xml");
    data_set.load("../data/data_set.xml");

    assert_true(data_set.get_variable_name(0) == "x", LOG);
    assert_true(data_set.get_variable_name(1) == "y", LOG);
*/
    // Test
/*
    data_set.set_has_header(false);
    data_set.set_separator(' ');
    data_string = "1 true\n"
                  "3 false\n"
                  "5 true\n";

    file.open(data_source_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    assert_true(data_set.get_variables_number() == 2, LOG);
    assert_true(data_set.get_input_variables_number() == 1, LOG);
    assert_true(data_set.get_target_variables_number() == 1, LOG);

    data = data_set.get_data();

    assert_true(data(0,1) - type(1) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(data(1,1) - type(0) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(data(2,1) - type(1) < type(NUMERIC_LIMITS_MIN), LOG);

    // Test

    data_set.set_separator('\t');
    data_set.set_missing_values_label("NaN");
    data_string =
            "f\t52\t1100\t32\t145490\t4\tno\n"
            "f\t57\t8715\t1\t242542\t1\tNaN\n"
            "m\t44\t5145\t28\t79100\t5\tno\n"
            "f\t57\t2857\t16\t1\t1\tNaN\n"
            "f\t47\t3368\t44\t63939\t1\tyes\n"
            "f\t59\t5697\t14\t45278\t1\tno\n"
            "m\t86\t1843\t1\t132799\t2\tyes\n"
            "m\t67\t4394\t25\t6670\t2\tno\n"
            "m\t40\t6619\t23\t168081\t1\tno\n"
            "f\t12\t4204\t17\t1\t2\tno\n";

    file.open(data_source_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    assert_true(data_set.get_variables_number() == 7, LOG);
    assert_true(data_set.get_input_variables_number() == 6, LOG);
    assert_true(data_set.get_target_variables_number() == 1, LOG);

    data = data_set.get_data();

    assert_true(data.dimension(0) == 10, LOG);
    assert_true(data.dimension(1) == 7, LOG);
*/
}


void DataSetTest::test_read_adult_csv()
{
    cout << "test_read_adult_csv\n";

    try {
        data_set.set_missing_values_label("?");
        data_set.set_separator(',');
        data_set.set_data_source_path("../../datasets/adult.data");
        data_set.set_has_header(false);
        data_set.read_csv();

        assert_true(data_set.get_samples_number() == 1000, LOG);
        assert_true(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(1) == DataSet::RawVariableType::Categorical, LOG);
        assert_true(data_set.get_raw_variable_type(2) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(3) == DataSet::RawVariableType::Categorical, LOG);

    }
    catch (const exception&)
    {
        assert_true(true,LOG);
    }

}


void DataSetTest::test_read_car_csv()
{
    cout << "test_read_car_csv\n";

    try
    {
        /*
        data_set.set("../../datasets/car.data",',',false);
        */
        assert_true(data_set.get_samples_number() == 1728, LOG);
        assert_true(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::Categorical, LOG);
        assert_true(data_set.get_raw_variable_type(1) == DataSet::RawVariableType::Categorical, LOG);
        assert_true(data_set.get_raw_variable_type(2) == DataSet::RawVariableType::Categorical, LOG);
        assert_true(data_set.get_raw_variable_type(3) == DataSet::RawVariableType::Categorical, LOG);
        assert_true(data_set.get_raw_variable_type(4) == DataSet::RawVariableType::Categorical, LOG);
        assert_true(data_set.get_raw_variable_type(5) == DataSet::RawVariableType::Categorical, LOG);
        assert_true(data_set.get_raw_variable_type(6) == DataSet::RawVariableType::Categorical, LOG);
    }
    catch(const exception&)
    {
        assert_true(true,LOG);
    }
}


void DataSetTest::test_read_empty_csv()
{
/*    cout << "test_read_empty_csv\n";

    data_set.set();

    try
    {
        data_set.set("../../datasets/empty.csv",' ',false);

        assert_true(data_set.is_empty(), LOG);
        assert_true(data_set.get_samples_number() == 0, LOG);
        assert_true(data_set.get_variables_number() == 2, LOG);

    }
    catch(const exception&)
    {
        assert_true(true,LOG);
    }*/
}


void DataSetTest::test_read_heart_csv()
{
 /*   cout << "test_read_heart_csv\n";
    try
    {
        data_set.set("../../datasets/heart.csv",',',true);

        assert_true(data_set.get_samples_number() == 303, LOG);
        assert_true(data_set.get_variables_number() == 14, LOG);
        assert_true(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(1) == DataSet::RawVariableType::Binary, LOG);
        assert_true(data_set.get_raw_variable_type(2) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(3) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(4) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(5) == DataSet::RawVariableType::Binary, LOG);
        assert_true(data_set.get_raw_variable_type(6) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(7) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(8) == DataSet::RawVariableType::Binary, LOG);
        assert_true(data_set.get_raw_variable_type(9) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(10) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(11) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(12) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(13) == DataSet::RawVariableType::Binary, LOG);
    }
    catch (const exception&)
    {
        assert_true(true,LOG);
    }       */
}


void DataSetTest::test_read_iris_csv()
{
    cout << "test_read_iris_csv\n";

    try
    {
        /*
        data_set.set("../../datasets/iris.data",',',false);
        */
        assert_true(data_set.get_samples_number() == 150, LOG);
        assert_true(data_set.get_variables_number() == 7, LOG);
        assert_true(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(1) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(2) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(3) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(4) == DataSet::RawVariableType::Categorical, LOG);
    }
    catch (const exception&)
    {
        assert_true(true,LOG);
    }
}


void DataSetTest::test_read_mnsit_csv()
{
/*    cout << "test_read_mnist_csv\n";

    try
    {
        data_set.set("../../datasets/mnist.csv",',',false);

        assert_true(data_set.get_samples_number() == 100, LOG);
        assert_true(data_set.get_variables_number() == 785, LOG);
    }
    catch(const exception&)
    {
        assert_true(true,LOG);
    }   */
}


void DataSetTest::test_read_one_variable_csv()
{
    cout << "test_read_one_variable_csv\n";

    try
    {
        /*
        data_set.set("../../datasets/one_variable.csv",',',false);
        */
        assert_true(data_set.get_samples_number() == 7, LOG);
        assert_true(data_set.get_variables_number() == 1, LOG);
        assert_true(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::Numeric, LOG);
    }
    catch (const exception&)
    {
        assert_true(true,LOG);
    }
}


void DataSetTest::test_read_pollution_csv()
{
    cout << "test_read_pollution_csv\n";

    try
    {
        /*
        data_set.set("../../datasets/pollution.csv",',',true);
        */
        assert_true(data_set.get_samples_number() == 1000, LOG);
        assert_true(data_set.get_variables_number() == 13, LOG);
        assert_true(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::DateTime, LOG);
        assert_true(data_set.get_raw_variable_type(1) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(2) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(3) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(4) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(5) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(6) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(7) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(8) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(9) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(10) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(11) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(12) == DataSet::RawVariableType::Numeric, LOG);
    }
    catch (const exception&)
    {
        assert_true(true,LOG);
    }
}


void DataSetTest::test_read_bank_churn_csv()
{
/*    cout << "test_read_bank_churn_csv\n";

    data_set.set_separator(';');
    data_set.set_data_source_path("../../datasets/bankchurn.csv");
    data_set.set_has_header(true);
    data_set.read_csv();*/
}

void DataSetTest::test_read_urinary_inflammations_csv()
{
 /*   cout << "test_read_urinary_inflammations_csv\n";
    try
    {
        data_set.set("../../datasets/urinary_inflammations.csv",';',true);

        assert_true(data_set.get_samples_number() == 120, LOG);
        assert_true(data_set.get_variables_number() == 8, LOG);
        assert_true(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(1) == DataSet::RawVariableType::Binary, LOG);
        assert_true(data_set.get_raw_variable_type(2) == DataSet::RawVariableType::Binary, LOG);
        assert_true(data_set.get_raw_variable_type(3) == DataSet::RawVariableType::Binary, LOG);
        assert_true(data_set.get_raw_variable_type(4) == DataSet::RawVariableType::Binary, LOG);
        assert_true(data_set.get_raw_variable_type(5) == DataSet::RawVariableType::Binary, LOG);
        assert_true(data_set.get_raw_variable_type(6) == DataSet::RawVariableType::Binary, LOG);
        assert_true(data_set.get_raw_variable_type(7) == DataSet::RawVariableType::Binary, LOG);
    }
    catch (const exception&)
    {
        assert_true(true,LOG);
    }   */
}


void DataSetTest::test_read_wine_csv()
{
    cout << "test_read_wine_csv\n";

    try
    {
        /*
        data_set.set("../../datasets/wine.data",',',false);
        */
        assert_true(data_set.get_samples_number() == 178, LOG);
        assert_true(data_set.get_variables_number() == 14, LOG);
        assert_true(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(1) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(2) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(3) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(4) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(5) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(6) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(7) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(8) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(9) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(10) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(11) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(12) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(13) == DataSet::RawVariableType::Numeric, LOG);
    }
    catch (const exception&)
    {
        assert_true(true,LOG);
    }
}


void DataSetTest::test_read_binary_csv()
{
    cout << "test_read_binary_csv\n";
    try
    {
        /*
        data_set.set("../../datasets/binary.csv",',',false);
        */
        assert_true(data_set.get_samples_number() == 8, LOG);
        assert_true(data_set.get_variables_number() == 3, LOG);
        assert_true(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(1) == DataSet::RawVariableType::Numeric, LOG);
        assert_true(data_set.get_raw_variable_type(2) == DataSet::RawVariableType::Binary, LOG);
    }
    catch (const exception&)
    {
        assert_true(true,LOG);
    }
}


void DataSetTest::test_scrub_missing_values()
{

    cout << "test_scrub_missing_values\n";

    const string data_source_path = "../data/data.dat";

    Tensor<DataSet::SampleUse, 1> samples_uses;

    ofstream file;

    data_set.set_data_source_path(data_source_path);

    string data_string;

    // Test
/*
    data_set.set_separator(' ');
    data_set.set_missing_values_label("NaN");

    data_string = "0 0 0\n"
                  "0 0 NaN\n"
                  "0 0 0\n";

    file.open(data_source_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    data_set.scrub_missing_values();

    samples_uses = data_set.get_samples_uses();

    assert_true(samples_uses(1) == DataSet::SampleUse::Unused, LOG);
*/
    // Test
/*
    data_set.set_separator(' ');
    data_set.set_missing_values_label("?");

    data_string ="? 6 3\n"
                 "3 ? 2\n"
                 "2 1 ?\n"
                 "1 2 1";

    file.open(data_source_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    data_set.set_missing_values_method(DataSet::MissingValuesMethod::Mean);
    data_set.scrub_missing_values();

    data = data_set.get_data();

    assert_true(abs(data(0,0) - type(2.0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(abs(data(1,1) - type(3.0)) < type(NUMERIC_LIMITS_MIN), LOG);
    assert_true(isnan(data(2,2)), LOG); */
}


void DataSetTest::test_calculate_variables_means()
{
    cout << "test_calculate_variables_means\n";

    data.resize(3, 4);

    data.setValues({{type(1), type(2), type(3.6), type(9)},
                    {type(2), type(2), type(2), type(2)},
                    {type(3), type(2), type(1), type(1)}});

    data_set.set_data(data);

    Tensor<Index, 1> indices;

    indices.resize(4);
    indices.setValues({0, 1, 2, 3});

    Tensor<type, 1> means = data_set.calculate_variables_means(indices);
    Tensor<type, 1> solution(4);

    solution.setValues({type(2), type(2), type(2.2), type(4)});

    Tensor<bool, 0> eq = (means==solution).all();

    assert_true(eq(0), LOG);
}


void DataSetTest::test_calculate_used_targets_mean()
{
    cout << "test_calculate_used_targets_mean\n";

    data.resize(3, 4);

    data.setValues({
                       {type(1), type(NAN), type(1), type(1)},
                       {type(2), type(2), type(2), type(2)},
                       {type(3), type(3), type(NAN), type(3)}});

    data_set.set_data(data);

    Tensor<Index, 1> indices(3);
    indices.setValues({0, 1, 2});
    Tensor<Index, 1> training_indexes(1);
    training_indexes.setValues({0});

    data_set.set_training(training_indexes);
}


void DataSetTest::test_calculate_selection_targets_mean()
{
    cout << "test_calculate_selection_targets_mean\n";

    Tensor<Index, 1> target_indices;
    Tensor<Index, 1> selection_indices;

    Tensor<type, 1> selection_targets_mean;

    // Test

    data.resize(3, 4);
    data.setValues({{1, type(NAN), 6, 9},
                    {1, 2, 5, 2},
                    {3, 2, type(NAN), 4}});

    data_set.set_data(data);

    target_indices.resize(2);
    target_indices.setValues({2,3});

    selection_indices.resize(2);
    selection_indices.setValues({0, 1});

    data_set.set_input();
    data_set.set_selection(selection_indices);

    data_set.set_input_target_raw_variables(Tensor<Index,1>(), target_indices);

    selection_targets_mean = data_set.calculate_selection_targets_mean();

    assert_true(selection_targets_mean(0) == type(5.5) , LOG);
    assert_true(selection_targets_mean(1) == type(5.5) , LOG);
}


void DataSetTest::test_calculate_input_target_correlations()
{

    cout << "test_calculate_input_target_correlations\n";

    // Test 1 (numeric and numeric trivial case)

    data.resize(3, 4);

    data.setValues({{type(1), type(1), type(-1), type(1)},
                    {type(2), type(2), type(-2), type(2)},
                    {type(3), type(3), type(-3), type(3)} });

    data_set.set_data(data);

    Tensor<Index, 1> input_raw_variables_indices(3);
    input_raw_variables_indices.setValues({0, 1, 2});

    Tensor<Index, 1> target_raw_variables_indices(1);
    target_raw_variables_indices.setValues({3});

    data_set.set_input_target_raw_variables(input_raw_variables_indices, target_raw_variables_indices);

    Tensor<Correlation, 2> input_target_correlations = data_set.calculate_input_target_raw_variables_correlations();

    assert_true(input_target_correlations(0,0).r == 1, LOG);
    assert_true(input_target_correlations(1,0).r == 1, LOG);
    assert_true(input_target_correlations(2,0).r == -1, LOG);

    // Test 2 (numeric and numeric non trivial case)

    data.resize(3, 4);
    data.setValues({{type(1), type(2), type(4), type(1)},
                    {type(2), type(3), type(9), type(2)},
                    {type(3), type(1), type(10), type(2)} });

    data_set.set_data(data);

    input_raw_variables_indices.setValues({0, 1});

    target_raw_variables_indices.resize(2);
    target_raw_variables_indices.setValues({2,3});

    data_set.set_input_target_raw_variables(input_raw_variables_indices, target_raw_variables_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variables_correlations();

    assert_true(input_target_correlations(0,0).r < 1 && input_target_correlations(0,0).r > -1 , LOG);
    assert_true(input_target_correlations(1,0).r < 1 && input_target_correlations(1,0).r > -1, LOG);
    assert_true(input_target_correlations(2,0).r < 1 && input_target_correlations(2,0).r > -1, LOG);

    // Test 3 (binary and binary non trivial case)

    data.setValues({
                       {type(0), type(0), type(1), type(0)},
                       {type(1), type(0), type(0), type(1)},
                       {type(1), type(0), type(0), type(1)} });

    data_set.set_data(data);

    input_raw_variables_indices.resize(3);
    input_raw_variables_indices.setValues({0, 1, 2});

    target_raw_variables_indices.setValues({3});

    data_set.set_input_target_raw_variables(input_raw_variables_indices, target_raw_variables_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variables_correlations();

    assert_true(input_target_correlations(0,0).r == 1, LOG);
    assert_true(input_target_correlations(0,0).form == Correlation::Form::Linear, LOG);

    assert_true(isnan(input_target_correlations(1,0).r), LOG);
    assert_true(input_target_correlations(1,0).form == Correlation::Form::Linear, LOG);

    assert_true(input_target_correlations(2,0).r == -1, LOG);
    assert_true(input_target_correlations(2,0).form == Correlation::Form::Linear, LOG);

    // Test 4 (binary and binary trivial case)

    data.setValues({{type(0), type(0), type(0), type(0)},
                    {type(1), type(1), type(1), type(1)},
                    {type(1), type(1), type(1), type(1)} });

    data_set.set_data(data);

    input_raw_variables_indices.resize(3);
    input_raw_variables_indices.setValues({0, 1, 2});

    target_raw_variables_indices.setValues({3});

    data_set.set_input_target_raw_variables(input_raw_variables_indices, target_raw_variables_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variables_correlations();

    for(Index i = 0; i < input_target_correlations.size(); i++)
    {
        assert_true(input_target_correlations(i).r == 1, LOG);
        assert_true(input_target_correlations(i).form == Correlation::Form::Linear, LOG);
    }

    // Test 5 (categorical and categorical)

    data_set = opennn::DataSet();

    data_set.set("../../datasets/correlation_tests.csv",',', false);

    input_raw_variables_indices.resize(2);
    input_raw_variables_indices.setValues({0, 3});

    target_raw_variables_indices.resize(1);
    target_raw_variables_indices.setValues({4});

    data_set.set_input_target_raw_variables(input_raw_variables_indices, target_raw_variables_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variables_correlations();
/*
    assert_true(input_target_correlations(1,0).r < 1, LOG);
    assert_true(input_target_correlations(1,0).form == Correlation::Form::Logistic, LOG);
*/
    // Test 6 (numeric and binary)
/*
    input_variables_indices.resize(3);
    input_variables_indices.setValues({1, 2, 5});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    data_set.set_input_target_raw_variables(input_variables_indices, target_variables_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variables_correlations();

    assert_true(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1, LOG);
    assert_true(input_target_correlations(0,0).form == Form::Logistic, LOG);

    assert_true(-1 < input_target_correlations(1,0).r && input_target_correlations(1,0).r < 1, LOG);
    assert_true(input_target_correlations(1,0).form == Form::Linear, LOG);

    assert_true(-1 < input_target_correlations(2,0).r && input_target_correlations(1,0).r < 1, LOG);
    assert_true(input_target_correlations(2,0).form == Form::Linear, LOG);


    // Test 7 (numeric and categorical)

    input_variables_indices.resize(1);
    input_variables_indices.setValues({1});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({0});

    data_set.set_input_target_raw_variables(input_variables_indices, target_variables_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variables_correlations();

    assert_true(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1, LOG);
    assert_true(input_target_correlations(0,0).form == Form::Logistic, LOG);

    // Test 8 (binary and categorical)

    input_variables_indices.resize(3);
    input_variables_indices.setValues({2,5,6});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({0});

    data_set.set_input_target_raw_variables(input_variables_indices, target_variables_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variables_correlations();

    assert_true(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1, LOG);
    assert_true(input_target_correlations(0,0).form == Form::Logistic, LOG);

    assert_true(-1 < input_target_correlations(1,0).r && input_target_correlations(1,0).r < 1, LOG);
    assert_true(input_target_correlations(1,0).form == Form::Logistic, LOG);

    assert_true(-1 < input_target_correlations(2,0).r && input_target_correlations(1,0).r < 1, LOG);
    assert_true(input_target_correlations(2,0).form == Form::Logistic, LOG);

    // With missing values or NAN

    // Test 9 (categorical and categorical)

    data_set.set("../../../opennn/datasets/correlation_tests_with_nan.csv",',', false);

    input_raw_variables_indices.resize(2);
    input_raw_variables_indices.setValues({0, 3});

    target_raw_variables_indices.resize(1);
    target_raw_variables_indices.setValues({4});

    data_set.set_input_target_raw_variables(input_raw_variables_indices, target_raw_variables_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variables_correlations();

    assert_true(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1, LOG);
    assert_true(input_target_correlations(0,0).form == Form::Logistic, LOG);

    assert_true(-1 < input_target_correlations(1,0).r && input_target_correlations(1,0).r < 1, LOG);
    assert_true(input_target_correlations(1,0).form == Form::Logistic, LOG);

    // Test 10 (numeric and binary)

    input_variables_indices.resize(3);
    input_variables_indices.setValues({1, 2, 5});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    data_set.set_input_target_raw_variables(input_variables_indices, target_variables_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variables_correlations();

    assert_true(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1, LOG);
    assert_true(input_target_correlations(0,0).form == Form::Logistic, LOG);

    assert_true(-1 < input_target_correlations(1,0).r && input_target_correlations(1,0).r < 1, LOG);
    assert_true(input_target_correlations(1,0).form == Form::Linear, LOG);

    assert_true(-1 < input_target_correlations(2,0).r && input_target_correlations(2,0).r < 1, LOG);
    assert_true(input_target_correlations(2,0).form == Form::Linear, LOG);

    // Test 11 (numeric and categorical)

    input_variables_indices.resize(1);
    input_variables_indices.setValues({1});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({0});

    data_set.set_input_target_raw_variables(input_variables_indices, target_variables_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variables_correlations();

    assert_true(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1, LOG);
    assert_true(input_target_correlations(0,0).form == Form::Logistic, LOG);

    // Test 12 (binary and categorical)

    input_variables_indices.resize(3);
    input_variables_indices.setValues({1, 2, 5});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({0});

    data_set.set_input_target_raw_variables(input_variables_indices, target_variables_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variables_correlations();

    assert_true(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1, LOG);
    assert_true(input_target_correlations(0,0).form == Form::Logistic, LOG);

    assert_true(-1 < input_target_correlations(1,0).r && input_target_correlations(1,0).r < 1, LOG);
    assert_true(input_target_correlations(1,0).form == Form::Logistic, LOG);

    assert_true(-1 < input_target_correlations(2,0).r && input_target_correlations(1,0).r < 1, LOG);
    assert_true(input_target_correlations(2,0).form == Form::Logistic, LOG);
    */

}


void DataSetTest::test_calculate_input_raw_variables_correlations()
{
    cout << "test_calculate_input_raw_variables_correlations\n";

    // Test 1 (numeric and numeric trivial case)

    cout << "Test 1" << endl;

    data.resize(3, 4);

    data.setValues({{type(1), type(1), type(-1), type(1)},
                    {type(2), type(2), type(-2), type(2)},
                    {type(3), type(3), type(-3), type(3)} });

    data_set.set_data(data);

    Tensor<Index, 1> input_raw_variables_indices(3);
    input_raw_variables_indices.setValues({0, 1, 2});

    Tensor<Index, 1> target_raw_variables_indices(1);
    target_raw_variables_indices.setValues({3});

    data_set.set_input_target_raw_variables(input_raw_variables_indices, target_raw_variables_indices);

    Tensor<Correlation, 2> inputs_correlations = data_set.calculate_input_raw_variables_correlations()(0);

    assert_true(inputs_correlations(0,0).r == 1, LOG);
    assert_true(inputs_correlations(0,1).r == 1, LOG);
    assert_true(inputs_correlations(0,2).r == -1, LOG);

    assert_true(inputs_correlations(1,0).r == 1, LOG);
    assert_true(inputs_correlations(1,1).r == 1, LOG);
    assert_true(inputs_correlations(1,2).r == -1, LOG);

    assert_true(inputs_correlations(2,0).r == -1, LOG);
    assert_true(inputs_correlations(2,1).r == -1, LOG);
    assert_true(inputs_correlations(2,2).r == 1, LOG);

    // Test 2 (numeric and numeric non trivial case)

    cout << "Test 2" << endl;

    data.resize(3, 4);
    data.setValues({{type(1), type(2), type(4), type(1)},
                    {type(2), type(3), type(9), type(2)},
                    {type(3), type(1), type(10), type(2)}});

    data_set.set_data(data);

    input_raw_variables_indices.setValues({0, 1});

    target_raw_variables_indices.setValues({2,3});

    data_set.set_input_target_raw_variables(input_raw_variables_indices, target_raw_variables_indices);

    inputs_correlations = data_set.calculate_input_raw_variables_correlations()(0);

    for(Index i = 0; i <  data_set.get_input_raw_variables_number() ; i++)
    {
        assert_true( inputs_correlations(i,i).r == 1, LOG);

        for(Index j = 0; i < j ; j++)
        {
            assert_true(-1 < inputs_correlations(i,j).r && inputs_correlations(i,j).r < 1, LOG);
        }
    }

    // Test 3 (binary and binary non trivial case)

    cout << "Test 3" << endl;

    data.resize(3, 4);
    data.setValues({{type(0), type(0), type(1), type(1)},
                    {type(1), type(0), type(0), type(2)},
                    {type(1), type(0), type(0), type(2)} });

    data_set.set_data(data);

    input_raw_variables_indices.resize(3);
    input_raw_variables_indices.setValues({0, 1, 2});

    target_raw_variables_indices.setValues({3});

    data_set.set_input_target_raw_variables(input_raw_variables_indices, target_raw_variables_indices);

    inputs_correlations = data_set.calculate_input_raw_variables_correlations()(0);

    assert_true(inputs_correlations(0,0).r == 1, LOG);
    assert_true(inputs_correlations(0,0).form == Correlation::Form::Linear, LOG);

    assert_true(isnan(inputs_correlations(1,0).r), LOG);

    assert_true(inputs_correlations(1,0).form == Correlation::Form::Linear, LOG);

    assert_true(isnan(inputs_correlations(1,1).r), LOG);
    assert_true(inputs_correlations(1,1).form == Correlation::Form::Linear, LOG);

    assert_true(inputs_correlations(2,0).r == -1, LOG);
    assert_true(inputs_correlations(2,0).form == Correlation::Form::Linear, LOG);

    assert_true(isnan(inputs_correlations(2,1).r), LOG);
    assert_true(inputs_correlations(2,1).form == Correlation::Form::Linear, LOG);

    assert_true(inputs_correlations(2,2).r == 1, LOG);
    assert_true(inputs_correlations(2,2).form == Correlation::Form::Linear, LOG);

    // Test 4 (binary and binary trivial case)
/*
    cout << "Test 4" << endl;

    data.setValues({{type(1), type(0), type(1), type(1)},
                    {type(2), type(1), type(1), type(2)},
                    {type(3), type(1), type(0), type(2)}});

    data_set.set_data(data);

    input_raw_variables_indices.resize(3);
    input_raw_variables_indices.setValues({0, 1, 2});

    target_raw_variables_indices.setValues({3});

    data_set.set_input_target_raw_variables(input_raw_variables_indices, target_raw_variables_indices);

    inputs_correlations = data_set.calculate_input_raw_variables_correlations()(0);

    assert_true(inputs_correlations(0,0).r == 1, LOG);
    assert_true(inputs_correlations(0,0).form == Correlation::Form::Linear, LOG);

    assert_true(inputs_correlations(0,1).r > 0 && inputs_correlations(0,1).r < 1, LOG);
    assert_true(inputs_correlations(0,1).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(0,2).r < 0 && inputs_correlations(0,2).r > -1, LOG);
    assert_true(inputs_correlations(0,2).form == Correlation::Form::Logistic, LOG);


    assert_true(inputs_correlations(1,0).r > 0 && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(1,0).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(1,1).r == 1, LOG);
    assert_true(inputs_correlations(1,1).form == Correlation::Form::Linear, LOG);

    assert_true(inputs_correlations(1,2).r == -0.5, LOG);
    assert_true(inputs_correlations(1,2).form == Correlation::Form::Linear, LOG);


    assert_true(inputs_correlations(2,0).r < 0 && inputs_correlations(2,0).r > -1, LOG);
    assert_true(inputs_correlations(2,0).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(2,1).r == -0.5, LOG);
    assert_true(inputs_correlations(2,1).form == Correlation::Form::Linear, LOG);

    assert_true(inputs_correlations(2,2).r == 1, LOG);
    assert_true(inputs_correlations(2,2).form == Correlation::Form::Linear, LOG);
*/
    // Test 5 (categorical and categorical)

    cout << "Test 5" << endl;

    data_set.set("../../datasets/correlation_tests.csv",',', false);
/*
    input_raw_variables_indices.resize(3);
    input_raw_variables_indices.setValues({0, 3, 4});

    target_raw_variables_indices.resize(1);
    target_raw_variables_indices.setValues({5});

    data_set.set_input_target_raw_variables(input_raw_variables_indices, target_raw_variables_indices);

    inputs_correlations = data_set.calculate_input_raw_variables_correlations()(0);

    assert_true(inputs_correlations(0,0).r == 1, LOG);
    assert_true(inputs_correlations(0,0).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(1,0).r == 1, LOG);
    assert_true(inputs_correlations(1,0).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(1,1).r == 1, LOG);
    assert_true(inputs_correlations(1,1).form == Correlation::Form::Logistic, LOG);

    assert_true(-1 < inputs_correlations(2,0).r && inputs_correlations(2,0).r < 1, LOG);
    assert_true(inputs_correlations(2,0).form == Correlation::Form::Logistic, LOG);

    assert_true(-1 < inputs_correlations(2,1).r && inputs_correlations(2,1).r < 1, LOG);
    assert_true(inputs_correlations(2,1).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(2,2).r == 1, LOG);
    assert_true(inputs_correlations(2,2).form == Correlation::Form::Logistic, LOG); // CHECK

    // Test 6 (numeric and binary)

    cout << "Test 6" << endl;

    input_variables_indices.resize(3);
    input_variables_indices.setValues({1, 2, 5});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    data_set.set_input_target_raw_variables(input_variables_indices, target_variables_indices);

    inputs_correlations = data_set.calculate_input_raw_variables_correlations()(0);

    assert_true(inputs_correlations(0,0).r == 1, LOG);

    assert_true(-1 < inputs_correlations(1,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(1,0).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(1,1).r == 1, LOG);
    assert_true(inputs_correlations(1,1).form == Correlation::Form::Linear, LOG);

    assert_true(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(2,0).form == Correlation::Form::Logistic, LOG);

    assert_true(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(2,1).form == Correlation::Form::Linear, LOG);

    assert_true(inputs_correlations(2,2).r == 1, LOG);
    assert_true(inputs_correlations(2,2).form == Correlation::Form::Linear, LOG);


    // Test 7 (numeric and categorical)

    cout << "Test 7" << endl;

    input_variables_indices.resize(3);
    input_variables_indices.setValues({0, 1, 3});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    data_set.set_input_target_raw_variables(input_variables_indices, target_variables_indices);


    inputs_correlations = data_set.calculate_input_raw_variables_correlations()(0);

    assert_true(inputs_correlations(0,0).r == 1, LOG);

    assert_true(-1 < inputs_correlations(1,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(1,0).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(1,1).r == 1, LOG);

    assert_true(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(2,0).form == Correlation::Form::Logistic, LOG);

    assert_true(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(2,1).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(2,2).r == 1, LOG);
    assert_true(inputs_correlations(2,2).form == Correlation::Form::Logistic, LOG);

    // Test 8 (binary and categorical)

    cout << "Test 8" << endl;

    input_variables_indices.resize(3);
    input_variables_indices.setValues({0, 2, 3});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    data_set.set_input_target_raw_variables(input_variables_indices, target_variables_indices);

    inputs_correlations = data_set.calculate_input_raw_variables_correlations()(0);

    assert_true(inputs_correlations(0,0).r == 1, LOG);
    assert_true(inputs_correlations(0,0).form == Correlation::Form::Logistic, LOG);

    assert_true(-1 < inputs_correlations(1,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(1,0).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(1,1).r == 1, LOG);
    assert_true(inputs_correlations(1,1).form == Correlation::Form::Linear, LOG);

    assert_true(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(2,0).form == Correlation::Form::Logistic, LOG);

    assert_true(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(2,1).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(2,2).r == 1, LOG);
    assert_true(inputs_correlations(2,2).form == Correlation::Form::Logistic, LOG);

    // With missing values or NAN

    // Test 9 (categorical and categorical)

    cout << "Test 9" << endl;

    data_set.set_missing_values_label("NA");
    data_set.set("../../datasets/correlation_tests_with_nan.csv",',', false);

    input_raw_variables_indices.resize(3);
    input_raw_variables_indices.setValues({0, 3, 4});

    target_raw_variables_indices.resize(1);
    target_raw_variables_indices.setValues({6});

    data_set.set_input_target_raw_variables(input_raw_variables_indices, target_raw_variables_indices);

    inputs_correlations = data_set.calculate_input_raw_variables_correlations()(0);

    assert_true(inputs_correlations(0,0).r == 1, LOG);
    assert_true(inputs_correlations(0,0).form == Correlation::Form::Logistic, LOG);

    assert_true(-1 < inputs_correlations(1,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(1,0).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(1,1).r == 1, LOG);
    assert_true(inputs_correlations(1,1).form == Correlation::Form::Logistic, LOG);

    assert_true(-1 < inputs_correlations(2,0).r && inputs_correlations(2,0).r < 1, LOG);
    assert_true(inputs_correlations(2,0).form == Correlation::Form::Logistic, LOG);

    assert_true(-1 < inputs_correlations(2,1).r && inputs_correlations(2,1).r < 1, LOG);
    assert_true(inputs_correlations(2,1).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(2,2).r == 1, LOG);
    assert_true(inputs_correlations(2,2).form == Correlation::Form::Logistic, LOG);

    // Test 10 (numeric and binary)

    cout << "Test 10" << endl;

    input_variables_indices.resize(3);
    input_variables_indices.setValues({1, 2, 5});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    data_set.set_input_target_raw_variables(input_variables_indices, target_variables_indices);

    inputs_correlations = data_set.calculate_input_raw_variables_correlations()(0);

    assert_true(inputs_correlations(0,0).r == 1, LOG);

    assert_true(-1 < inputs_correlations(1,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(1,0).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(1,1).r == 1, LOG);
    assert_true(inputs_correlations(1,1).form == Correlation::Form::Linear, LOG);

    assert_true(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(2,0).form == Correlation::Form::Logistic, LOG);

    assert_true(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(2,1).form == Correlation::Form::Linear, LOG);

    assert_true(inputs_correlations(2,2).r == 1, LOG);
    assert_true(inputs_correlations(2,2).form == Correlation::Form::Linear, LOG);

    // Test 11 (numeric and categorical)

    cout << "Test 11" << endl;

    input_variables_indices.resize(3);
    input_variables_indices.setValues({0, 1, 3});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    data_set.set_input_target_raw_variables(input_variables_indices, target_variables_indices);

    inputs_correlations = data_set.calculate_input_raw_variables_correlations()(0);

    assert_true(inputs_correlations(0,0).r == 1, LOG);

    assert_true(-1 < inputs_correlations(1,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(1,0).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(1,1).r == 1, LOG);

    assert_true(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(2,0).form == Correlation::Form::Logistic, LOG);

    assert_true(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(2,1).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(2,2).r == 1, LOG);
    assert_true(inputs_correlations(2,2).form == Correlation::Form::Logistic, LOG);

    // Test 12 (binary and categorical)

    cout << "Test 12" << endl;

    input_variables_indices.resize(3);
    input_variables_indices.setValues({0, 2, 3});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    data_set.set_input_target_raw_variables(input_variables_indices, target_variables_indices);

    inputs_correlations = data_set.calculate_input_raw_variables_correlations()(0);

    assert_true(inputs_correlations(0,0).r == 1, LOG);
    assert_true(inputs_correlations(0,0).form == Correlation::Form::Logistic, LOG);

    assert_true(-1 < inputs_correlations(1,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(1,0).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(1,1).r == 1, LOG);
    assert_true(inputs_correlations(1,1).form == Correlation::Form::Linear, LOG);

    assert_true(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(2,0).form == Correlation::Form::Logistic, LOG);

    assert_true(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1, LOG);
    assert_true(inputs_correlations(2,1).form == Correlation::Form::Logistic, LOG);

    assert_true(inputs_correlations(2,2).r == 1, LOG);
    assert_true(inputs_correlations(2,2).form == Correlation::Form::Logistic, LOG);
*/
}


void DataSetTest::test_unuse_repeated_samples()
{
    cout << "test_unuse_repeated_samples\n";

    Tensor<Index, 1> indices;

    data_set.set();

    // Test

    data.resize(3, 3);

    data.setValues({{type(1),type(2),type(2)},
                    {type(1),type(2),type(2)},
                    {type(1),type(6),type(6)}});

    data_set = opennn::DataSet();

    data_set.set_data(data);
    data_set.set_training();

    indices = data_set.unuse_repeated_samples();

    assert_true(indices.size() == 1, LOG);
    assert_true(indices(0) == 1, LOG);

    // Test

    data.resize(4,3);

    data.setValues({{type(1),type(2),type(2)},
                   {type(1),type(2),type(2)},
                   {type(1),type(2),type(4)},
                   {type(1),type(2),type(4)}});

    data_set = opennn::DataSet();

    data_set.set_data(data);
    data_set.set_training();

    indices = data_set.unuse_repeated_samples();

    assert_true(indices.size() == 2, LOG);
    assert_true(contains(indices, 1), LOG);
    assert_true(contains(indices, 3), LOG);

    // Test

    data.resize(5, 3);
    data.setValues({{type(1),type(2),type(2)},
                   {type(1),type(2),type(2)},
                   {type(1),type(2),type(4)},
                   {type(1),type(2),type(4)},
                   {type(1),type(2),type(4)}});

    data_set = opennn::DataSet();

    data_set.set_data(data);
    data_set.set_training();

    indices = data_set.unuse_repeated_samples();

    assert_true(indices.size() == 3, LOG);
    assert_true(contains(indices, 1), LOG);
    assert_true(contains(indices, 3), LOG);
    assert_true(contains(indices, 4), LOG);
}


void DataSetTest::test_unuse_uncorrelated_raw_variables()
{
    cout << "test_unuse_uncorrelated_columns\n";

    data.resize(3, 3);
    data.setValues({{type(1),type(0),type(0)},
                    {type(1),type(0),type(0)},
                    {type(1),type(0),type(1)}});
}


void DataSetTest::test_calculate_training_negatives()
{
    cout << "test_calculate_training_negatives\n";

    Index training_negatives;
    Index target_index;

    // Test

    data.resize(3, 3);

    data.setValues({{ 1, 1, 1},
                    {-1,-1, 0},
                    { 0, 1, 1}});

    data_set.set_data(data);

    training_indices.resize(2);
    training_indices.setValues({0,1});

    input_variables_indices.resize(2);
    input_variables_indices.setValues({0, 1});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({2});

    target_index = 2;

    data_set.set_testing();
    data_set.set_training(training_indices);

    training_negatives = data_set.calculate_training_negatives(target_index);

    assert_true(training_negatives == 1, LOG);
}


void DataSetTest::test_calculate_selection_negatives()
{
    cout << "test_calculate_selection_negatives\n";

    Tensor<Index, 1> selection_indices;
    Tensor<Index, 1> input_variables_indices;
    Tensor<Index, 1> target_variables_indices;

    // Test

    data.resize(3, 3);

    data.setValues({{1, 1, 1},{0, 0, 1},{0, 1, 1}});

    data_set.set_data(data);

    selection_indices.resize(2);
    selection_indices.setValues({0,1});

    input_variables_indices.resize(2);
    input_variables_indices.setValues({0, 1});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({2});

    Index target_index = 2;

    data_set.set_testing();
    data_set.set_selection(selection_indices);

    data_set.set_input_target_raw_variables(input_variables_indices, target_variables_indices);

    Index selection_negatives = data_set.calculate_selection_negatives(target_index);

    data = data_set.get_data();

    assert_true(selection_negatives == 0, LOG);
}


void DataSetTest::test_fill()
{
    cout << "test_fill\n";

    data.resize(3, 3);
    data.setValues({{1,4,7},{2,5,8},{3,6,9}});
    data_set.set_data(data);

    data_set.set_training();

    const Index training_samples_number = data_set.get_training_samples_number();

    const Tensor<Index, 1> training_samples_indices = data_set.get_training_samples_indices();

    const Tensor<Index, 1> input_variables_indices = data_set.get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = data_set.get_target_variables_indices();
/*
    batch.set(training_samples_number, &data_set);
    batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);

    Tensor<type, 2> input_data(3,2);
    input_data.setValues({{1,4},{2,5},{3,6}});

    Tensor<type, 2> target_data(3,1);
    target_data.setValues({{7},{8},{9}});

    const Tensor<pair<type*, dimensions>, 1> inputs_pair = batch.get_inputs_pair();

    const TensorMap<Tensor<type, 2>> inputs(inputs_pair(0).first, inputs_pair(0).second[0], inputs_pair(0).second[1]);

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets(targets_pair.first, targets_pair.second[0], targets_pair.second[1]);

    assert_true(are_equal(inputs, input_data), LOG);
    assert_true(are_equal(targets, target_data), LOG);  */
}


void DataSetTest::run_test_case()
{
    cout << "Running data set test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Data resizing methods

    test_unuse_constant_raw_variables();
    test_unuse_repeated_samples();
    test_unuse_uncorrelated_raw_variables();

    // Statistics methods

    test_calculate_variables_descriptives();
    test_calculate_variables_means();
    test_calculate_input_variables_descriptives();
    test_calculate_used_targets_mean();
    test_calculate_selection_targets_mean();

    // Histrogram methods

    test_calculate_data_distributions();

    // Filtering methods

    test_filter_data();

    // Data scaling

    test_scale_data();

    // Correlations

    test_calculate_input_target_correlations();
/*
    test_calculate_input_raw_variables_correlations();

    // Classification methods

    test_calculate_target_distribution();

    // Outlier detection

    test_calculate_Tukey_outliers();

    // Serialization methods

    test_read_csv();
    test_read_bank_churn_csv();
    test_read_adult_csv();
    test_read_car_csv();
    test_read_empty_csv();
    test_read_heart_csv();
    test_read_iris_csv();
    test_read_mnsit_csv();
    test_read_one_variable_csv();
    test_read_pollution_csv();
    test_read_urinary_inflammations_csv();
    test_read_wine_csv();
    test_read_binary_csv();
    test_calculate_training_negatives();
    test_calculate_selection_negatives();
    test_scrub_missing_values();

    test_fill();
*/
    cout << "End of data set test case.\n\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
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
