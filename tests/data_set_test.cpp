#include "pch.h"

#include "../opennn/data_set.h"


TEST(DataSetTest, DefaultConstructor)
{
    DataSet data_set;

    EXPECT_EQ(data_set.get_variables_number(), 0);
    EXPECT_EQ(data_set.get_samples_number(), 0);
}


TEST(DataSetTest, DimensionsConstructor)
{

    DataSet data_set(1, {1}, {1});
    
    EXPECT_EQ(data_set.get_samples_number(), 1);
    EXPECT_EQ(data_set.get_variables_number(), 2);
    EXPECT_EQ(data_set.get_variables_number(DataSet::VariableUse::Input), 1);
    EXPECT_EQ(data_set.get_variables_number(DataSet::VariableUse::Target), 1);

}


TEST(DataSetTest, VariablesDescriptives)
{

    DataSet data_set(1, { 1 }, { 1 });
    data_set.set_data_constant(type(0));

//    vector<Descriptives> variable_descriptives = data_set.calculate_variable_descriptives();

//    EXPECT_EQ(variable_descriptives.size() == 1);

//    EXPECT_EQ(abs(variable_descriptives[0].minimum) < type(NUMERIC_LIMITS_MIN));
//    EXPECT_EQ(abs(variable_descriptives[0].maximum) < type(NUMERIC_LIMITS_MIN));
//    EXPECT_EQ(abs(variable_descriptives[0].mean) < type(NUMERIC_LIMITS_MIN));
//    EXPECT_EQ(abs(variable_descriptives[0].standard_deviation) < type(NUMERIC_LIMITS_MIN));
}



TEST(DataSetTest, CalculateVariablesDescriptives)
{
    // Test

    DataSet data_set(2, { 2 }, { 2 });

    data_set.set_data_constant(type(0));
/*
    variable_descriptives = data_set.calculate_variable_descriptives();

    EXPECT_EQ(variable_descriptives.size() == 4);

    EXPECT_EQ(variable_descriptives[0].minimum < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(variable_descriptives[0].maximum < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(variable_descriptives[0].mean < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(variable_descriptives[0].standard_deviation < type(NUMERIC_LIMITS_MIN));

    EXPECT_EQ(variable_descriptives[1].minimum < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(variable_descriptives[1].maximum < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(variable_descriptives[1].mean < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(variable_descriptives[1].standard_deviation < type(NUMERIC_LIMITS_MIN));

    EXPECT_EQ(variable_descriptives[2].minimum < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(variable_descriptives[2].maximum < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(variable_descriptives[2].mean < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(variable_descriptives[2].standard_deviation < type(NUMERIC_LIMITS_MIN));

    EXPECT_EQ(variable_descriptives[3].minimum < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(variable_descriptives[3].maximum < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(variable_descriptives[3].mean < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(variable_descriptives[3].standard_deviation < type(NUMERIC_LIMITS_MIN));

    // Test

    inputs_number = 2;
    targets_number = 1;
    samples_number = 3;

    data.resize(samples_number, inputs_number + targets_number);

    data.setValues({{type(-1000),type(2),type(0)},
                    {type(1)    ,type(4),type(2)},
                    {type(1)    ,type(4),type(0)}});

    data_set.set_data(data);

    variable_descriptives = data_set.calculate_variable_descriptives();

    EXPECT_EQ(variable_descriptives.size() == 3);

    EXPECT_EQ(abs(variable_descriptives[0].minimum + type(1000)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(variable_descriptives[1].minimum - type(2)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(variable_descriptives[2].minimum) < type(NUMERIC_LIMITS_MIN));

    EXPECT_EQ(abs(variable_descriptives[0].maximum - type(1)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(variable_descriptives[1].maximum - type(4)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(variable_descriptives[2].maximum - type(2)) < type(NUMERIC_LIMITS_MIN));
*/
}

TEST(DataSetTest, CalculateRawVariablesDistributions)
{
    Tensor<Histogram, 1> histograms;
/*
    data_set.set();

    // Test

    data.resize(3, 3);

    data.setValues({{type(2),type(2),type(1)},
                    {type(1),type(1),type(1)},
                    {type(1),type(2),type(2)}});

    data_set.set_data(data);

    histograms = data_set.calculate_raw_variables_distribution(2);

    EXPECT_EQ(histograms.size() == 3);

    EXPECT_EQ(abs( histograms(0).frequencies(0) - 2 ) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs( histograms(1).frequencies(0) - 1 ) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs( histograms(2).frequencies(0) - 2 ) < type(NUMERIC_LIMITS_MIN));

    EXPECT_EQ(abs( histograms(0).centers(0) - 1 ) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs( histograms(1).centers(0) - 1 ) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs( histograms(2).centers(0) - 1 ) < type(NUMERIC_LIMITS_MIN));
*/
}


TEST(DataSetTest, FilterData)
{
    Index samples_number;
    Index inputs_number;
    Index targets_number;

    Tensor<type, 1> minimums;
    Tensor<type, 1> maximums;

    // Test
/*
    samples_number = 2;
    inputs_number = 1;
    targets_number = 1;

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data_constant(type(1));

    minimums.resize(2);
    minimums.setValues({ type(2), type(0)});

    maximums.resize(2);
    maximums.setValues({type(2), type(0.5)});

    data_set.filter_data(minimums, maximums);

    EXPECT_EQ(data_set.get_sample_use(0) == DataSet::SampleUse::None);
    EXPECT_EQ(data_set.get_sample_use(1) == DataSet::SampleUse::None);
*/
}

/*
TEST(DataSetTest, ScaleData)
{
    vector<Descriptives> data_descriptives;
    Tensor<type, 2> scaled_data;

    // Test

    data.resize(2,2);
    data.setValues({{type(1), type(2)},
                    {type(3), type(4)}});

    data_set.set_data(data);

    data_set.set_raw_variable_scalers(Scaler::None);
    data_descriptives = data_set.scale_data();

    scaled_data = data_set.get_data();

    EXPECT_EQ(are_equal(scaled_data, data));

    // Test

    data_set.set_raw_variable_scalers(Scaler::MinimumMaximum);
    data_descriptives = data_set.scale_data();

    scaled_data = data_set.get_data();

    EXPECT_EQ(abs(scaled_data(0) - type(-1)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(scaled_data(1) - type(1)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(scaled_data(2) - type(-1)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(scaled_data(3) - type(1)) < type(NUMERIC_LIMITS_MIN));
}


TEST(DataSetTest, UnuseConstantRawVariables)
{
    // Test

    samples_number = 3;
    inputs_number = 2;
    targets_number = 1;

    data.resize(samples_number, inputs_number + targets_number);

    data.setValues({{type(1),type(2),type(0)},
                    {type(1),type(2),type(1)},
                    {type(1),type(2),type(2)}});

    data_set.set_data(data);
    data_set.set_has_header(false);
    data_set.unuse_constant_raw_variables();

    data_set.unuse_constant_raw_variables();

    EXPECT_EQ(data_set.get_raw_variables_number(DataSet::VariableUse::Input) == 0);
    EXPECT_EQ(data_set.get_raw_variables_number(DataSet::VariableUse::Target) == 1);
}


TEST(DataSetTest, CalculateTargetDistribution)
{
    Tensor<Index, 1> target_distribution;

    // Test two classes

    data.resize(5, 5);

    data.setValues({{type(2),type(5),type(6),type(9),type(0)},
                    {type(2),type(9),type(1),type(9),type(0)},
                    {type(2),type(9),type(1),type(9),type(NAN)},
                    {type(6),type(5),type(6),type(7),type(1)},
                    {type(0),type(1),type(0),type(1),type(1)}});

    data_set.set(data);

    input_variables_indices.resize(4);
    input_variables_indices.setValues({0, 1, 2, 3});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({4});

    data_set.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    target_distribution = data_set.calculate_target_distribution();

    Tensor<Index, 1> solution(2);
    solution(0) = 2;
    solution(1) = 2;

    EXPECT_EQ(target_distribution(0) == solution(0));
    EXPECT_EQ(target_distribution(1) == solution(1));

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

    data_set.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    target_distribution = data_set.calculate_target_distribution();

    EXPECT_EQ(target_distribution[0] == 1);
    EXPECT_EQ(target_distribution[1] == 2);
    EXPECT_EQ(target_distribution[2] == 2);
}


TEST(DataSetTest, CalculateTukeyOutliers)
{
    Tensor<type, 1> sample;

    Tensor<Tensor<Index, 1>, 1> outliers_indices;

    // Test

    data_set.set(100, 5, 1);
    data_set.set_data_random();

    outliers_indices = data_set.calculate_Tukey_outliers(type(1.5));

    EXPECT_EQ(outliers_indices.size() == 2);
    EXPECT_EQ(outliers_indices(0)(0) == 0);
}

TEST(DataSetTest, ReadCSV)
{
    // Test

    data_set.set(2, 2, 2);
    data_set.set_separator(DataSet::Separator::Comma);
    data_path = "../data/data.dat";
    data_set.set_data_source_path(data_path);
    data_set.set_data_constant(type(0));
    data_set.save_data();

    data_set.read_csv();
    data = data_set.get_data();

    EXPECT_EQ(is_equal(data, type(0)));

    // Test

    data_set.set_separator(DataSet::Separator::Tab);
    data_string = "\n\n\n   1\t2   \n\n\n   3\t4   \n\n\n    5\t6    \n\n\n";

    file.open(data_path.c_str());
    file << data_string;
    file.close();

    data_set.set();
    data_set.set_data_source_path(data_path);
    data_set.read_csv();
    data = data_set.get_data();

    EXPECT_EQ(data.dimension(0) == 3);
    EXPECT_EQ(data.dimension(1) == 2);

    EXPECT_EQ(abs(data(0, 0) - type(1)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(data(0, 1) - type(2)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(data(1, 0) - type(3)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(data(1, 1) - type(4)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(data(2, 0) - type(5)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(data(2, 1) - type(6)) < type(NUMERIC_LIMITS_MIN));

    EXPECT_EQ(data_set.get_samples_number() == 3);
    EXPECT_EQ(data_set.get_variables_number() == 2);

    // Test

    data_set.set_separator(DataSet::Separator::Tab);
    data_string = "\n\n\n   1\t2   \n\n\n   3\t4   \n\n\n   5\t6  \n\n\n";

    file.open(data_path.c_str());
    file << data_string;
    file.close();

    data_set.set();
    data_set.set_data_source_path(data_path);
    data_set.read_csv();

    data = data_set.get_data();

    EXPECT_EQ(data.dimension(0) == 3);
    EXPECT_EQ(data.dimension(1) == 2);

    EXPECT_EQ(abs(data(0,0) - type(1)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(data(0,1) - type(2)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(data(1,0) - type(3)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(data(1,1) - type(4)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(data(2,0) - type(5)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(data(2,1) - type(6)) < type(NUMERIC_LIMITS_MIN));

    EXPECT_EQ(data_set.get_samples_number() == 3);
    EXPECT_EQ(data_set.get_variables_number() == 2);

    // Test

    data_set.set();
    data_set.set_has_header(true);
    data_set.set_separator(DataSet::Separator::Space);
    data_string = "x y\n"
                  "1 2\n"
                  "3 4\n"
                  "5 6";

    file.open(data_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    data = data_set.get_data();

    EXPECT_EQ(data_set.get_header_line());
    EXPECT_EQ(data_set.get_variable_name(0) == "x");
    EXPECT_EQ(data_set.get_variable_name(1) == "y");

    EXPECT_EQ(data.dimension(0) == 3);
    EXPECT_EQ(data.dimension(1) == 2);

    EXPECT_EQ((data(0,0) - 1.0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(0,1) - 2.0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(1,0) - 3.0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(1,1) - 4.0) < type(NUMERIC_LIMITS_MIN));

    // Test

    data_set.set_has_header(true);
    data_set.set_separator(DataSet::Separator::Comma);

    data_string = "\tx\t,\ty\n"
                  "\t1\t,\t2\n"
                  "\t3\t,\t4";

    file.open(data_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    data = data_set.get_data();

    EXPECT_EQ(data_set.get_variable_name(0) == "x");
    EXPECT_EQ(data_set.get_variable_name(1) == "y");

    EXPECT_EQ((data(0,0) - 1.0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(0,1) - 2.0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(1,0) - 3.0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(1,1) - 4.0) < type(NUMERIC_LIMITS_MIN));

    // Test

    data_set.set_has_header(true);
    data_set.set_separator(DataSet::Separator::Comma);

    data_string = "x , y\n"
                  "1 , 2\n"
                  "3 , 4\n";

    file.open(data_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    data = data_set.get_data();

    EXPECT_EQ(data_set.get_variable_name(0) == "x");
    EXPECT_EQ(data_set.get_variable_name(1) == "y");

    EXPECT_EQ((data(0,0) - 1.0 ) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(0,1) - 2.0 ) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(1,0) - 3.0 ) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(1,1) - 4.0 ) < type(NUMERIC_LIMITS_MIN));

    // Test

    data_set.set_has_header(false);
    data_set.set_separator(DataSet::Separator::Comma);
    data_string =
            "5.1,3.5,1.4,0.2,Iris-setosa\n"
            "7.0,3.2,4.7,1.4,Iris-versicolor\n"
            "7.0,3.2,4.7,1.4,Iris-versicolor\n"
            "6.3,3.3,6.0,2.5,Iris-virginica";

    file.open(data_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    EXPECT_EQ(data_set.get_samples_number() == 4);
    EXPECT_EQ(data_set.get_variables_number() == 7);

    // Test

    data_set.set_has_header(false);
    data_set.set_separator(DataSet::Separator::Comma);
    data_string =
            "5.1,3.5,1.4,0.2,Iris-setosa\n"
            "7.0,3.2,4.7,1.4,Iris-versicolor\n"
            "7.0,3.2,4.7,1.4,Iris-versicolor\n"
            "6.3,3.3,6.0,2.5,Iris-virginica\n";

    file.open(data_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    EXPECT_EQ(data_set.get_variables_number() == 7);
    EXPECT_EQ(data_set.get_input_variables_number() == 4);
    EXPECT_EQ(data_set.get_target_variables_number() == 3);
    EXPECT_EQ(data_set.get_samples_number() == 4);

    data = data_set.get_data();

    EXPECT_EQ((data(0,0) - type(5.1)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(0,4) - type(1)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(0,5)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(0,6)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(1,4)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(1,5) - type(1)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(1,6)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(2,4)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(2,5) - type(1)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(2,6)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(3,4)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(3,5)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ((data(3,6) - type(1)) < type(NUMERIC_LIMITS_MIN));

    // Test

    data_set.set_has_header(true);
    data_set.set_separator(DataSet::Separator::Comma);
    data_set.set_missing_values_label("NaN");
    data_string =
            "sepal length,sepal width,petal length,petal width,class\n"
            "NaN,3.5,1.4,0.2,Iris-setosa\n"
            "7.0,3.2,4.7,1.4,Iris-versicolor\n"
            "7.0,3.2,4.7,1.4,Iris-versicolor\n"
            "6.3,3.3,6.0,2.5,Iris-virginica\n"
            "0.0,0.0,0.0,0.0,NaN\n";

    file.open(data_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    EXPECT_EQ(data_set.get_variables_number() == 7);
    EXPECT_EQ(data_set.get_input_variables_number() == 4);
    EXPECT_EQ(data_set.get_target_variables_number() == 3);

    EXPECT_EQ(data_set.get_variable_name(0) == "sepal length");
    EXPECT_EQ(data_set.get_variable_name(1) == "sepal width");
    EXPECT_EQ(data_set.get_variable_name(2) == "petal length");
    EXPECT_EQ(data_set.get_variable_name(3) == "petal width");
    EXPECT_EQ(data_set.get_variable_name(4) == "Iris-setosa");
    EXPECT_EQ(data_set.get_variable_name(5) == "Iris-versicolor");
    EXPECT_EQ(data_set.get_variable_name(6) == "Iris-virginica");

    EXPECT_EQ(data_set.get_samples_number() == 5);

    data = data_set.get_data();

    EXPECT_EQ(data(0,4) - type(1) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(data(0,5) - type(0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(data(0,6) - type(0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(data(1,4) - type(0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(data(1,5) - type(1) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(data(1,6) - type(0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(data(2,4) - type(0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(data(2,5) - type(1) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(data(2,6) - type(0) < type(NUMERIC_LIMITS_MIN));

    // Test

    data_set.set_has_header(false);
    data_set.set_separator(DataSet::Separator::Space);
    data_string = "1 2\n3 4\n5 6";

    file.open(data_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();
    data_set.set_variable_name(0, "x");
    data_set.set_variable_name(1, "y");

    data_set.save("../data/data_set.xml");
    data_set.load("../data/data_set.xml");

    EXPECT_EQ(data_set.get_variable_name(0) == "x");
    EXPECT_EQ(data_set.get_variable_name(1) == "y");

    // Test

    data_set.set_has_header(false);
    data_set.set_separator(DataSet::Separator::Space);
    data_string = "1 true\n"
                  "3 false\n"
                  "5 true\n";

    file.open(data_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    EXPECT_EQ(data_set.get_variables_number() == 2);
    EXPECT_EQ(data_set.get_input_variables_number() == 1);
    EXPECT_EQ(data_set.get_target_variables_number() == 1);

    data = data_set.get_data();

    EXPECT_EQ(data(0,1) - type(1) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(data(1,1) - type(0) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(data(2,1) - type(1) < type(NUMERIC_LIMITS_MIN));

    // Test

    data_set.set_separator(DataSet::Separator::Tab);
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

    file.open(data_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    EXPECT_EQ(data_set.get_variables_number() == 7);
    EXPECT_EQ(data_set.get_input_variables_number() == 6);
    EXPECT_EQ(data_set.get_target_variables_number() == 1);

    data = data_set.get_data();

    EXPECT_EQ(data.dimension(0) == 10);
    EXPECT_EQ(data.dimension(1) == 7);

}


TEST(DataSetTest, ReadAdultCSV)
{
    try
    {
        data_set.set_missing_values_label("?");
        data_set.set_separator_string(",");
        data_set.set_data_source_path("../../datasets/adult.data");
        data_set.set_has_header(false);
        data_set.read_csv();

        EXPECT_EQ(data_set.get_samples_number() == 1000);
        EXPECT_EQ(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(1) == DataSet::RawVariableType::Categorical);
        EXPECT_EQ(data_set.get_raw_variable_type(2) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(3) == DataSet::RawVariableType::Categorical);

    }
    catch (const exception&)
    {
        EXPECT_EQ(true,LOG);
    }
}


TEST(DataSetTest, ReadCarCSV)
{
    try
    {
       
        data_set.set("../../datasets/car.data", ",");
        
        EXPECT_EQ(data_set.get_samples_number() == 1728);
        EXPECT_EQ(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::Categorical);
        EXPECT_EQ(data_set.get_raw_variable_type(1) == DataSet::RawVariableType::Categorical);
        EXPECT_EQ(data_set.get_raw_variable_type(2) == DataSet::RawVariableType::Categorical);
        EXPECT_EQ(data_set.get_raw_variable_type(3) == DataSet::RawVariableType::Categorical);
        EXPECT_EQ(data_set.get_raw_variable_type(4) == DataSet::RawVariableType::Categorical);
        EXPECT_EQ(data_set.get_raw_variable_type(5) == DataSet::RawVariableType::Categorical);
        EXPECT_EQ(data_set.get_raw_variable_type(6) == DataSet::RawVariableType::Categorical);
    }
    catch(const exception&)
    {
        EXPECT_EQ(true,LOG);
    }
}


TEST(DataSetTest, ReadEmptyCSV)
{
    data_set.set();

    try
    {
        data_set.set("../../datasets/empty.csv", " ", false);

        //EXPECT_EQ(data_set.is_empty());
        EXPECT_EQ(data_set.get_samples_number() == 0);
        EXPECT_EQ(data_set.get_variables_number() == 2);

    }
    catch(const exception&)
    {
        EXPECT_EQ(true,LOG);
    }
}


TEST(DataSetTest, ReadHeartCSV)
{
    try
    {
        data_set.set("../../datasets/heart.csv", ",", true);

        EXPECT_EQ(data_set.get_samples_number() == 303);
        EXPECT_EQ(data_set.get_variables_number() == 14);
        EXPECT_EQ(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(1) == DataSet::RawVariableType::Binary);
        EXPECT_EQ(data_set.get_raw_variable_type(2) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(3) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(4) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(5) == DataSet::RawVariableType::Binary);
        EXPECT_EQ(data_set.get_raw_variable_type(6) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(7) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(8) == DataSet::RawVariableType::Binary);
        EXPECT_EQ(data_set.get_raw_variable_type(9) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(10) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(11) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(12) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(13) == DataSet::RawVariableType::Binary);
    }
    catch (const exception&)
    {
        EXPECT_EQ(true,LOG);
    }       
}


TEST(DataSetTest, ReadIrisCSV)
{
    try
    {
        
        data_set.set("../../datasets/iris.data", ",", false);
        
        EXPECT_EQ(data_set.get_samples_number() == 150);
        EXPECT_EQ(data_set.get_variables_number() == 7);
        EXPECT_EQ(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(1) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(2) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(3) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(4) == DataSet::RawVariableType::Categorical);
    }
    catch (const exception&)
    {
        EXPECT_EQ(true,LOG);
    }
}


TEST(DataSetTest, ReadOneVariableCSV)
{
    try
    { 
        data_set.set("../../datasets/one_variable.csv", ",", false);
        
        EXPECT_EQ(data_set.get_samples_number() == 7);
        EXPECT_EQ(data_set.get_variables_number() == 1);
        EXPECT_EQ(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::Numeric);
    }
    catch (const exception&)
    {
        EXPECT_EQ(true,LOG);
    }
}


TEST(DataSetTest, ReadPollutionCSV)
{
    try
    {
        
        data_set.set("../../datasets/pollution.csv", ",", true);
        
        EXPECT_EQ(data_set.get_samples_number() == 1000);
        EXPECT_EQ(data_set.get_variables_number() == 13);
        EXPECT_EQ(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::DateTime);
        EXPECT_EQ(data_set.get_raw_variable_type(1) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(2) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(3) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(4) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(5) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(6) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(7) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(8) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(9) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(10) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(11) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(12) == DataSet::RawVariableType::Numeric);
    }
    catch (const exception&)
    {
        EXPECT_EQ(true,LOG);
    }
}


TEST(DataSetTest, ReadBankChurnCSV)
{
    data_set.set_separator(DataSet::Separator::Semicolon);
    data_set.set_data_source_path("../../datasets/bankchurn.csv");
    data_set.set_has_header(true);
    
    data_set.read_csv();
    
}


void DataSetTest::test_read_urinary_inflammations_csv()
{
    cout << "test_read_urinary_inflammations_csv\n";

    try
    {
        data_set.set("../../datasets/urinary_inflammations.csv", ";", true);

        EXPECT_EQ(data_set.get_samples_number() == 120);
        EXPECT_EQ(data_set.get_variables_number() == 8);
        EXPECT_EQ(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(1) == DataSet::RawVariableType::Binary);
        EXPECT_EQ(data_set.get_raw_variable_type(2) == DataSet::RawVariableType::Binary);
        EXPECT_EQ(data_set.get_raw_variable_type(3) == DataSet::RawVariableType::Binary);
        EXPECT_EQ(data_set.get_raw_variable_type(4) == DataSet::RawVariableType::Binary);
        EXPECT_EQ(data_set.get_raw_variable_type(5) == DataSet::RawVariableType::Binary);
        EXPECT_EQ(data_set.get_raw_variable_type(6) == DataSet::RawVariableType::Binary);
        EXPECT_EQ(data_set.get_raw_variable_type(7) == DataSet::RawVariableType::Binary);
    }
    catch (const exception&)
    {
        EXPECT_EQ(true,LOG);
    }

}


void DataSetTest::test_read_wine_csv()
{
    cout << "test_read_wine_csv\n";

    try
    {        
        data_set.set("../../datasets/wine.data", ",", false);
        
        EXPECT_EQ(data_set.get_samples_number() == 178);
        EXPECT_EQ(data_set.get_variables_number() == 14);
        EXPECT_EQ(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(1) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(2) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(3) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(4) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(5) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(6) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(7) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(8) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(9) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(10) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(11) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(12) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(13) == DataSet::RawVariableType::Numeric);
    }
    catch (const exception&)
    {
        EXPECT_EQ(true,LOG);
    }
}


void DataSetTest::test_read_binary_csv()
{
    cout << "test_read_binary_csv\n";

    try
    {
        
        data_set.set("../../datasets/binary.csv", ",", false);
        
        EXPECT_EQ(data_set.get_samples_number() == 8);
        EXPECT_EQ(data_set.get_variables_number() == 3);
        EXPECT_EQ(data_set.get_raw_variable_type(0) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(1) == DataSet::RawVariableType::Numeric);
        EXPECT_EQ(data_set.get_raw_variable_type(2) == DataSet::RawVariableType::Binary);
    }
    catch (const exception&)
    {
        EXPECT_EQ(true,LOG);
    }
}


void DataSetTest::test_scrub_missing_values()
{
    cout << "test_scrub_missing_values\n";

    const string data_path = "../../datasets/data.dat";

    Tensor<DataSet::SampleUse, 1> sample_uses;

    std::ofstream file;

    data_set.set_data_source_path(data_path);

    string data_string;

    // Test

    data_set.set_separator(DataSet::Separator::Space);
    data_set.set_missing_values_label("NaN");

    data_string = "0 0 0\n"
                  "0 0 NaN\n"
                  "0 0 0\n";

    file.open(data_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    data_set.scrub_missing_values();

    sample_uses = data_set.get_sample_uses();

    EXPECT_EQ(sample_uses(1) == DataSet::SampleUse::None);

    // Test

    data_set.set_separator(DataSet::Separator::Space);
    data_set.set_missing_values_label("?");

    data_string ="? 6 3\n"
                 "3 ? 2\n"
                 "2 1 ?\n"
                 "1 2 1";

    file.open(data_path.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    data_set.set_missing_values_method(DataSet::MissingValuesMethod::Mean);
    data_set.scrub_missing_values();

    data = data_set.get_data();

    EXPECT_EQ(abs(data(0,0) - type(2.0)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(abs(data(1,1) - type(3.0)) < type(NUMERIC_LIMITS_MIN));
    EXPECT_EQ(isnan(data(2,2)));

}


void DataSetTest::test_calculate_used_targets_mean()
{
    cout << "test_calculate_used_targets_mean\n";

    data.resize(3, 4);

    data.setValues({{type(1), type(NAN), type(1), type(1)},
                    {type(2), type(2), type(2), type(2)},
                    {type(3), type(3), type(NAN), type(3)}});

    data_set.set_data(data);

    Tensor<Index, 1> indices(3);
    indices.setValues({0, 1, 2});

    Tensor<Index, 1> training_indices(1);
    training_indices.setValues({0});

    data_set.set_training(training_indices);


}


void DataSetTest::test_calculate_selection_targets_mean()
{
    cout << "test_calculate_selection_targets_mean\n";

    Tensor<Index, 1> target_indices;
    Tensor<Index, 1> selection_indices;

    Tensor<type, 1> selection_targets_mean;

    // Test

    data.resize(3, 4);

    data.setValues({{1, type(NAN),         6, 9},
                    {1,         2,         5, 2},
                    {3,         2, type(NAN), 4}});

    data_set.set_data(data);

    target_indices.resize(2);
    target_indices.setValues({2,3});

    selection_indices.resize(2);
    selection_indices.setValues({0, 1});

    data_set.set(DataSet::VariableUse::Input);

    data_set.set_sample_uses(DataSet::SampleUse::Selection, selection_indices);

    data_set.set_input_target_raw_variables_indices(Tensor<Index,1>(), target_indices);

    selection_targets_mean = data_set.calculate_selection_targets_mean();

//    cout << selection_targets_mean << endl;system("pause");

    EXPECT_EQ(selection_targets_mean(0) == type(5.5));
    EXPECT_EQ(selection_targets_mean(1) == type(5));

}


void DataSetTest::test_calculate_input_target_raw_variable_correlations()
{
    cout << "test_calculate_input_target_raw_variable_correlations\n";

    // Test 1 (numeric and numeric trivial case)

    data.resize(3, 4);

    data.setValues({{type(1), type(1), type(-1), type(1)},
                    {type(2), type(2), type(-2), type(2)},
                    {type(3), type(3), type(-3), type(3)}});

    data_set.set_data(data);

    Tensor<Index, 1> input_raw_variable_indices(3);
    input_raw_variable_indices.setValues({0, 1, 2});

    Tensor<Index, 1> target_raw_variable_indices(1);
    target_raw_variable_indices.setValues({3});

    data_set.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    Tensor<Correlation, 2> input_target_raw_variable_correlations = data_set.calculate_input_target_raw_variable_pearson_correlations();
    
    EXPECT_EQ(input_target_raw_variable_correlations(0,0).r == 1);
    EXPECT_EQ(input_target_raw_variable_correlations(1,0).r == 1);
    EXPECT_EQ(input_target_raw_variable_correlations(2,0).r == -1);
    
    // Test 2 (numeric and numeric non trivial case)

    data.resize(3, 4);
    data.setValues({{type(1), type(2), type(4), type(1)},
                    {type(2), type(3), type(9), type(2)},
                    {type(3), type(1), type(10), type(2)}});

    data_set.set_data(data);

    input_raw_variable_indices.setValues({0, 1});

    target_raw_variable_indices.resize(2);
    target_raw_variable_indices.setValues({2,3});

    data_set.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    input_target_raw_variable_correlations = data_set.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(input_target_raw_variable_correlations(0,0).r < 1 && input_target_raw_variable_correlations(0,0).r > -1 );
    EXPECT_EQ(input_target_raw_variable_correlations(1,0).r < 1 && input_target_raw_variable_correlations(1,0).r > -1);
    EXPECT_EQ(input_target_raw_variable_correlations(2,0).r < 1 && input_target_raw_variable_correlations(2,0).r > -1);

    // Test 3 (binary and binary non trivial case)

    data.setValues({{type(0), type(0), type(1), type(0)},
                    {type(1), type(0), type(0), type(1)},
                    {type(1), type(0), type(0), type(1)}});

    data_set.set_data(data);

    input_raw_variable_indices.resize(3);
    input_raw_variable_indices.setValues({0, 1, 2});

    target_raw_variable_indices.setValues({3});

    data_set.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    input_target_raw_variable_correlations = data_set.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(input_target_raw_variable_correlations(0,0).r == 1);
    EXPECT_EQ(input_target_raw_variable_correlations(0,0).form == Correlation::Form::Linear);

    EXPECT_EQ(isnan(input_target_raw_variable_correlations(1,0).r));
    EXPECT_EQ(input_target_raw_variable_correlations(1,0).form == Correlation::Form::Linear);

    EXPECT_EQ(input_target_raw_variable_correlations(2,0).r == -1);
    EXPECT_EQ(input_target_raw_variable_correlations(2,0).form == Correlation::Form::Linear);

    // Test 4 (binary and binary trivial case)

    data.setValues({{type(0), type(0), type(0), type(0)},
                    {type(1), type(1), type(1), type(1)},
                    {type(1), type(1), type(1), type(1)}});

    data_set.set_data(data);

    input_raw_variable_indices.resize(3);
    input_raw_variable_indices.setValues({0, 1, 2});

    target_raw_variable_indices.setValues({3});

    data_set.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    input_target_raw_variable_correlations = data_set.calculate_input_target_raw_variable_pearson_correlations();

    for(Index i = 0; i < input_target_raw_variable_correlations.size(); i++)
    {
        EXPECT_EQ(input_target_raw_variable_correlations(i).r == 1);
        EXPECT_EQ(input_target_raw_variable_correlations(i).form == Correlation::Form::Linear);
    }

    // Test 5 (categorical and categorical)

    data_set = DataSet();

    data_set.set("../../datasets/correlation_tests.csv", ",", false);

    input_raw_variable_indices.resize(2);
    input_raw_variable_indices.setValues({0, 3});

    target_raw_variable_indices.resize(1);
    target_raw_variable_indices.setValues({4});

    data_set.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    input_target_raw_variable_correlations = data_set.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(input_target_correlations(1,0).r < 1);
    EXPECT_EQ(input_target_correlations(1,0).form == Correlation::Form::Logistic);

    // Test 6 (numeric and binary)

    input_variables_indices.resize(3);
    input_variables_indices.setValues({1, 2, 5});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    data_set.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1);
    EXPECT_EQ(input_target_correlations(0,0).form == Form::Logistic);

    EXPECT_EQ(-1 < input_target_correlations(1,0).r && input_target_correlations(1,0).r < 1);
    EXPECT_EQ(input_target_correlations(1,0).form == Form::Linear);

    EXPECT_EQ(-1 < input_target_correlations(2,0).r && input_target_correlations(1,0).r < 1);
    EXPECT_EQ(input_target_correlations(2,0).form == Form::Linear);

    // Test 7 (numeric and categorical)

    input_variables_indices.resize(1);
    input_variables_indices.setValues({1});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({0});

    data_set.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1);
    EXPECT_EQ(input_target_correlations(0,0).form == Form::Logistic);

    // Test 8 (binary and categorical)

    input_variables_indices.resize(3);
    input_variables_indices.setValues({2,5,6});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({0});

    data_set.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1);
    EXPECT_EQ(input_target_correlations(0,0).form == Form::Logistic);

    EXPECT_EQ(-1 < input_target_correlations(1,0).r && input_target_correlations(1,0).r < 1);
    EXPECT_EQ(input_target_correlations(1,0).form == Form::Logistic);

    EXPECT_EQ(-1 < input_target_correlations(2,0).r && input_target_correlations(1,0).r < 1);
    EXPECT_EQ(input_target_correlations(2,0).form == Form::Logistic);

    // With missing values or NAN

    // Test 9 (categorical and categorical)

    data_set.set("../../../opennn/datasets/correlation_tests_with_nan.csv",',', false);

    input_raw_variable_indices.resize(2);
    input_raw_variable_indices.setValues({0, 3});

    target_raw_variable_indices.resize(1);
    target_raw_variable_indices.setValues({4});

    data_set.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1);
    EXPECT_EQ(input_target_correlations(0,0).form == Form::Logistic);

    EXPECT_EQ(-1 < input_target_correlations(1,0).r && input_target_correlations(1,0).r < 1);
    EXPECT_EQ(input_target_correlations(1,0).form == Form::Logistic);

    // Test 10 (numeric and binary)

    input_variables_indices.resize(3);
    input_variables_indices.setValues({1, 2, 5});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    data_set.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1);
    EXPECT_EQ(input_target_correlations(0,0).form == Form::Logistic);

    EXPECT_EQ(-1 < input_target_correlations(1,0).r && input_target_correlations(1,0).r < 1);
    EXPECT_EQ(input_target_correlations(1,0).form == Form::Linear);

    EXPECT_EQ(-1 < input_target_correlations(2,0).r && input_target_correlations(2,0).r < 1);
    EXPECT_EQ(input_target_correlations(2,0).form == Form::Linear);

    // Test 11 (numeric and categorical)

    input_variables_indices.resize(1);
    input_variables_indices.setValues({1});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({0});

    data_set.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1);
    EXPECT_EQ(input_target_correlations(0,0).form == Form::Logistic);

    // Test 12 (binary and categorical)

    input_variables_indices.resize(3);
    input_variables_indices.setValues({1, 2, 5});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({0});

    data_set.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    input_target_correlations = data_set.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1);
    EXPECT_EQ(input_target_correlations(0,0).form == Form::Logistic);

    EXPECT_EQ(-1 < input_target_correlations(1,0).r && input_target_correlations(1,0).r < 1);
    EXPECT_EQ(input_target_correlations(1,0).form == Form::Logistic);

    EXPECT_EQ(-1 < input_target_correlations(2,0).r && input_target_correlations(1,0).r < 1);
    EXPECT_EQ(input_target_correlations(2,0).form == Form::Logistic);
    
}


void DataSetTest::test_calculate_input_raw_variable_correlations()
{
    cout << "test_calculate_input_raw_variable_correlations\n";

    // Test 1 (numeric and numeric trivial case)

    data.resize(3, 4);

    data.setValues({{type(1), type(1), type(-1), type(1)},
                    {type(2), type(2), type(-2), type(2)},
                    {type(3), type(3), type(-3), type(3)}});

    data_set.set_data(data);

    Tensor<Index, 1> input_raw_variable_indices(3);
    input_raw_variable_indices.setValues({0, 1, 2});

    Tensor<Index, 1> target_raw_variable_indices(1);
    target_raw_variable_indices.setValues({3});

    data_set.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    Tensor<Correlation, 2> inputs_correlations = data_set.calculate_input_raw_variable_pearson_correlations();

    EXPECT_EQ(inputs_correlations(0,0).r == 1);
    EXPECT_EQ(inputs_correlations(0,1).r == 1);
    EXPECT_EQ(inputs_correlations(0,2).r == -1);

    EXPECT_EQ(inputs_correlations(1,0).r == 1);
    EXPECT_EQ(inputs_correlations(1,1).r == 1);
    EXPECT_EQ(inputs_correlations(1,2).r == -1);

    EXPECT_EQ(inputs_correlations(2,0).r == -1);
    EXPECT_EQ(inputs_correlations(2,1).r == -1);
    EXPECT_EQ(inputs_correlations(2,2).r == 1);

    // Test 2 (numeric and numeric non trivial case)

    data.resize(3, 4);
    data.setValues({{type(1), type(2), type(4), type(1)},
                    {type(2), type(3), type(9), type(2)},
                    {type(3), type(1), type(10), type(2)}});

    data_set.set_data(data);

    input_raw_variable_indices.setValues({0, 1});

    target_raw_variable_indices.setValues({2,3});

    data_set.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    inputs_correlations = data_set.calculate_input_raw_variable_pearson_correlations();

    for(Index i = 0; i <  data_set.get_raw_variables_number(DataSet::VariableUse::Input) ; i++)
    {
        EXPECT_EQ(inputs_correlations(i,i).r == 1);

        for(Index j = 0; i < j ; j++)
            EXPECT_EQ(-1 < inputs_correlations(i,j).r && inputs_correlations(i,j).r < 1);
    }

    // Test 3 (binary and binary non trivial case)

    data.resize(3, 4);
    data.setValues({{type(0), type(0), type(1), type(1)},
                    {type(1), type(0), type(0), type(2)},
                    {type(1), type(0), type(0), type(2)}});

    data_set.set_data(data);

    input_raw_variable_indices.resize(3);
    input_raw_variable_indices.setValues({0, 1, 2});

    target_raw_variable_indices.setValues({3});

    data_set.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);
    
    inputs_correlations = data_set.calculate_input_raw_variable_pearson_correlations();

    EXPECT_EQ(inputs_correlations(0,0).r == 1);
    EXPECT_EQ(inputs_correlations(0,0).form == Correlation::Form::Linear);

    EXPECT_EQ(isnan(inputs_correlations(1,0).r));

    EXPECT_EQ(inputs_correlations(1,0).form == Correlation::Form::Linear);

    EXPECT_EQ(isnan(inputs_correlations(1,1).r));
    EXPECT_EQ(inputs_correlations(1,1).form == Correlation::Form::Linear);

    EXPECT_EQ(inputs_correlations(2,0).r == -1);
    EXPECT_EQ(inputs_correlations(2,0).form == Correlation::Form::Linear);

    EXPECT_EQ(isnan(inputs_correlations(2,1).r));
    EXPECT_EQ(inputs_correlations(2,1).form == Correlation::Form::Linear);

    EXPECT_EQ(inputs_correlations(2,2).r == 1);
    EXPECT_EQ(inputs_correlations(2,2).form == Correlation::Form::Linear);

    // Test 4 (binary and binary trivial case)

    data.setValues({{type(1), type(0), type(1), type(1)},
                    {type(2), type(1), type(1), type(2)},
                    {type(3), type(1), type(0), type(2)}});

    data_set.set_data(data);

    input_raw_variable_indices.resize(3);
    input_raw_variable_indices.setValues({0, 1, 2});

    target_raw_variable_indices.setValues({3});

    data_set.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);
    /*
    inputs_correlations = data_set.calculate_input_raw_variable_pearson_correlations();

    EXPECT_EQ(inputs_correlations(0,0).r == 1);
    EXPECT_EQ(inputs_correlations(0,0).form == Correlation::Form::Linear);

    EXPECT_EQ(inputs_correlations(0,1).r > 0 && inputs_correlations(0,1).r < 1);
    EXPECT_EQ(inputs_correlations(0,1).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(0,2).r < 0 && inputs_correlations(0,2).r > -1);
    EXPECT_EQ(inputs_correlations(0,2).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(1,0).r > 0 && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(1,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(1,1).r == 1);
    EXPECT_EQ(inputs_correlations(1,1).form == Correlation::Form::Linear);

    EXPECT_EQ(inputs_correlations(1,2).r == -0.5);
    EXPECT_EQ(inputs_correlations(1,2).form == Correlation::Form::Linear);

    EXPECT_EQ(inputs_correlations(2,0).r < 0 && inputs_correlations(2,0).r > -1);
    EXPECT_EQ(inputs_correlations(2,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(2,1).r == -0.5);
    EXPECT_EQ(inputs_correlations(2,1).form == Correlation::Form::Linear);

    EXPECT_EQ(inputs_correlations(2,2).r == 1);
    EXPECT_EQ(inputs_correlations(2,2).form == Correlation::Form::Linear);

    // Test 5 (categorical and categorical)

    data_set.set("../../datasets/correlation_tests.csv",',', false);

    input_raw_variable_indices.resize(3);
    input_raw_variable_indices.setValues({0, 3, 4});

    target_raw_variable_indices.resize(1);
    target_raw_variable_indices.setValues({5});

    data_set.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    inputs_correlations = data_set.calculate_input_raw_variable_correlations()(0);

    EXPECT_EQ(inputs_correlations(0,0).r == 1);
    EXPECT_EQ(inputs_correlations(0,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(1,0).r == 1);
    EXPECT_EQ(inputs_correlations(1,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(1,1).r == 1);
    EXPECT_EQ(inputs_correlations(1,1).form == Correlation::Form::Logistic);

    EXPECT_EQ(-1 < inputs_correlations(2,0).r && inputs_correlations(2,0).r < 1);
    EXPECT_EQ(inputs_correlations(2,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(-1 < inputs_correlations(2,1).r && inputs_correlations(2,1).r < 1);
    EXPECT_EQ(inputs_correlations(2,1).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(2,2).r == 1);
    EXPECT_EQ(inputs_correlations(2,2).form == Correlation::Form::Logistic); // CHECK

    // Test 6 (numeric and binary)

    input_variables_indices.resize(3);
    input_variables_indices.setValues({1, 2, 5});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    data_set.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    inputs_correlations = data_set.calculate_input_raw_variable_correlations()(0);

    EXPECT_EQ(inputs_correlations(0,0).r == 1);

    EXPECT_EQ(-1 < inputs_correlations(1,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(1,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(1,1).r == 1);
    EXPECT_EQ(inputs_correlations(1,1).form == Correlation::Form::Linear);

    EXPECT_EQ(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(2,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(2,1).form == Correlation::Form::Linear);

    EXPECT_EQ(inputs_correlations(2,2).r == 1);
    EXPECT_EQ(inputs_correlations(2,2).form == Correlation::Form::Linear);

    // Test 7 (numeric and categorical)

    input_variables_indices.resize(3);
    input_variables_indices.setValues({0, 1, 3});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    data_set.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    inputs_correlations = data_set.calculate_input_raw_variable_correlations()(0);

    EXPECT_EQ(inputs_correlations(0,0).r == 1);

    EXPECT_EQ(-1 < inputs_correlations(1,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(1,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(1,1).r == 1);

    EXPECT_EQ(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(2,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(2,1).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(2,2).r == 1);
    EXPECT_EQ(inputs_correlations(2,2).form == Correlation::Form::Logistic);

    // Test 8 (binary and categorical)

    input_variables_indices.resize(3);
    input_variables_indices.setValues({0, 2, 3});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    data_set.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    inputs_correlations = data_set.calculate_input_raw_variable_correlations()(0);

    EXPECT_EQ(inputs_correlations(0,0).r == 1);
    EXPECT_EQ(inputs_correlations(0,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(-1 < inputs_correlations(1,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(1,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(1,1).r == 1);
    EXPECT_EQ(inputs_correlations(1,1).form == Correlation::Form::Linear);

    EXPECT_EQ(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(2,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(2,1).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(2,2).r == 1);
    EXPECT_EQ(inputs_correlations(2,2).form == Correlation::Form::Logistic);

    // With missing values or NAN

    // Test 9 (categorical and categorical)

    data_set.set_missing_values_label("NA");
    data_set.set("../../datasets/correlation_tests_with_nan.csv",',', false);

    input_raw_variable_indices.resize(3);
    input_raw_variable_indices.setValues({0, 3, 4});

    target_raw_variable_indices.resize(1);
    target_raw_variable_indices.setValues({6});

    data_set.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    inputs_correlations = data_set.calculate_input_raw_variable_correlations()(0);

    EXPECT_EQ(inputs_correlations(0,0).r == 1);
    EXPECT_EQ(inputs_correlations(0,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(-1 < inputs_correlations(1,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(1,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(1,1).r == 1);
    EXPECT_EQ(inputs_correlations(1,1).form == Correlation::Form::Logistic);

    EXPECT_EQ(-1 < inputs_correlations(2,0).r && inputs_correlations(2,0).r < 1);
    EXPECT_EQ(inputs_correlations(2,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(-1 < inputs_correlations(2,1).r && inputs_correlations(2,1).r < 1);
    EXPECT_EQ(inputs_correlations(2,1).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(2,2).r == 1);
    EXPECT_EQ(inputs_correlations(2,2).form == Correlation::Form::Logistic);

    // Test 10 (numeric and binary)

    input_variables_indices.resize(3);
    input_variables_indices.setValues({1, 2, 5});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    data_set.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    inputs_correlations = data_set.calculate_input_raw_variable_correlations()(0);

    EXPECT_EQ(inputs_correlations(0,0).r == 1);

    EXPECT_EQ(-1 < inputs_correlations(1,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(1,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(1,1).r == 1);
    EXPECT_EQ(inputs_correlations(1,1).form == Correlation::Form::Linear);

    EXPECT_EQ(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(2,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(2,1).form == Correlation::Form::Linear);

    EXPECT_EQ(inputs_correlations(2,2).r == 1);
    EXPECT_EQ(inputs_correlations(2,2).form == Correlation::Form::Linear);

    // Test 11 (numeric and categorical)

    input_variables_indices.resize(3);
    input_variables_indices.setValues({0, 1, 3});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    data_set.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    inputs_correlations = data_set.calculate_input_raw_variable_correlations()(0);

    EXPECT_EQ(inputs_correlations(0,0).r == 1);

    EXPECT_EQ(-1 < inputs_correlations(1,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(1,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(1,1).r == 1);

    EXPECT_EQ(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(2,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(2,1).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(2,2).r == 1);
    EXPECT_EQ(inputs_correlations(2,2).form == Correlation::Form::Logistic);

    // Test 12 (binary and categorical)

    input_variables_indices.resize(3);
    input_variables_indices.setValues({0, 2, 3});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    data_set.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    inputs_correlations = data_set.calculate_input_raw_variable_correlations()(0);

    EXPECT_EQ(inputs_correlations(0,0).r == 1);
    EXPECT_EQ(inputs_correlations(0,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(-1 < inputs_correlations(1,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(1,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(1,1).r == 1);
    EXPECT_EQ(inputs_correlations(1,1).form == Correlation::Form::Linear);

    EXPECT_EQ(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(2,0).form == Correlation::Form::Logistic);

    EXPECT_EQ(-1 < inputs_correlations(2,0).r && inputs_correlations(1,0).r < 1);
    EXPECT_EQ(inputs_correlations(2,1).form == Correlation::Form::Logistic);

    EXPECT_EQ(inputs_correlations(2,2).r == 1);
    EXPECT_EQ(inputs_correlations(2,2).form == Correlation::Form::Logistic);

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
    data_set.set(DataSet::SampleUse::Training);

    indices = data_set.unuse_repeated_samples();

    EXPECT_EQ(indices.size() == 1);
    EXPECT_EQ(indices(0) == 1);

    // Test

    data.resize(4,3);

    data.setValues({{type(1),type(2),type(2)},
                   {type(1),type(2),type(2)},
                   {type(1),type(2),type(4)},
                   {type(1),type(2),type(4)}});

    data_set = opennn::DataSet();

    data_set.set_data(data);
    data_set.set(DataSet::SampleUse::Training);

    indices = data_set.unuse_repeated_samples();

    EXPECT_EQ(indices.size() == 2);
    EXPECT_EQ(contains(indices, 1));
    EXPECT_EQ(contains(indices, 3));

    // Test

    data.resize(5, 3);
    data.setValues({{type(1),type(2),type(2)},
                   {type(1),type(2),type(2)},
                   {type(1),type(2),type(4)},
                   {type(1),type(2),type(4)},
                   {type(1),type(2),type(4)}});

    data_set.set();

    data_set.set_data(data);
    data_set.set(DataSet::SampleUse::Training);

    indices = data_set.unuse_repeated_samples();

    EXPECT_EQ(indices.size() == 3);
    EXPECT_EQ(contains(indices, 1));
    EXPECT_EQ(contains(indices, 3));
    EXPECT_EQ(contains(indices, 4));
}


void DataSetTest::test_unuse_uncorrelated_raw_variables()
{
    cout << "test_unuse_uncorrelated_raw_variables\n";

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

    data_set.set(DataSet::SampleUse::Testing);
    data_set.set_training(training_indices);

    //training_negatives = data_set.calculate_training_negatives(target_index);
    training_negatives = data_set.calculate_negatives(DataSet::SampleUse::Training,target_index);

    EXPECT_EQ(training_negatives == 1);
}


void DataSetTest::test_calculate_selection_negatives()
{
    cout << "test_calculate_selection_negatives\n";

    vector<Index> selection_indices;
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

    data_set.set(DataSet::SampleUse::Testing);

    data_set.set_selection(selection_indices);

    data_set.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    //Index selection_negatives = data_set.calculate_selection_negatives(target_index);
    Index selection_negatives = data_set.calculate_negatives(DataSet::SampleUse::Selection,target_index);
    data_set.calculate_negatives(DataSet::SampleUse::Training,target_index);
    data = data_set.get_data();

    EXPECT_EQ(selection_negatives == 0);

}


void DataSetTest::test_fill()
{
    cout << "test_fill\n";

    data.resize(3, 3);
    data.setValues({{1,4,7},{2,5,8},{3,6,9}});
    data_set.set_data(data);

    data_set.set(DataSet::SampleUse::Training);

    const Index training_samples_number = data_set.get_samples_number(DataSet::SampleUse::Training);

    const Tensor<Index, 1> training_samples_indices = data_set.get_sample_indices(DataSet::SampleUse::Training);

    const Tensor<Index, 1> input_variables_indices = data_set.get_variable_indices(DataSet::VariableUse::Input);
    const Tensor<Index, 1> target_variables_indices = data_set.get_variable_indices(DataSet::VariableUse::Target);

    batch.set(training_samples_number, &data_set);
    /*
    batch.fill(training_samples_indices, input_variables_indices, target_variables_indices);
    
    Tensor<type, 2> input_data(3,2);
    input_data.setValues({{1,4},{2,5},{3,6}});

    Tensor<type, 2> target_data(3,1);
    target_data.setValues({{7},{8},{9}});

    const vector<pair<type*, dimensions>> input_pairs = batch.get_input_pairs();

    const TensorMap<Tensor<type, 2>> inputs = tensor_map(input_pairs[0]);

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets = tensor_map_2(targets_pair);

    EXPECT_EQ(are_equal(inputs, input_data));
    EXPECT_EQ(are_equal(targets, target_data));  

}

}
*/