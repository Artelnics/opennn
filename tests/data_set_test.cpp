#include "pch.h"

#include "../opennn/dataset.h"

using namespace opennn;

void create_temp_csv_file(const string& file_path, const string& content) 
{
    ofstream outfile(file_path);
    if (!outfile.is_open())
        throw runtime_error("Failed to open temporary CSV file for writing: " + file_path);

    outfile << content;
    outfile.close();
}


TEST(Dataset, DefaultConstructor)
{
    Dataset dataset;

    EXPECT_EQ(dataset.get_variables_number(), 0);
    EXPECT_EQ(dataset.get_samples_number(), 0);
}


TEST(Dataset, DimensionsConstructor)
{

    Dataset dataset(1, { 1 }, { 1 });

    EXPECT_EQ(dataset.get_samples_number(), 1);
    EXPECT_EQ(dataset.get_variables_number(), 2);
    EXPECT_EQ(dataset.get_variables_number("Input"), 1);
    EXPECT_EQ(dataset.get_variables_number("Target"), 1);

}


TEST(Dataset, VariableDescriptivesZero)
{

    Dataset dataset(1, { 1 }, { 1 });
    dataset.set_data_constant(type(0));

    const vector<Descriptives> variable_descriptives = dataset.calculate_variable_descriptives();

    EXPECT_EQ(variable_descriptives.size(), 2);

    EXPECT_NEAR(variable_descriptives[0].minimum, 0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(variable_descriptives[0].maximum, 0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(variable_descriptives[0].mean, 0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(variable_descriptives[0].standard_deviation, 0, NUMERIC_LIMITS_MIN);

}


TEST(Dataset, VariableDescriptives)
{

    const Index samples_number = 3;
    const Index inputs_number = 2;
    const Index targets_number = 1;

    Dataset dataset(samples_number, { inputs_number }, { targets_number });

    Tensor<type, 2> data(samples_number, inputs_number + targets_number);

    data.setValues({ {type(-1000),type(2),type(0)},
                    {type(1)    ,type(4),type(2)},
                    {type(1)    ,type(4),type(0)} });

    dataset.set_data(data);
   
    const vector<Descriptives> variable_descriptives = dataset.calculate_variable_descriptives();

    EXPECT_EQ(variable_descriptives.size(), 3);
    EXPECT_NEAR(variable_descriptives[0].minimum, type(-1000), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(variable_descriptives[1].minimum, type(2), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(variable_descriptives[2].minimum, 0, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(variable_descriptives[0].maximum, type(1), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(variable_descriptives[1].maximum, type(4), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(variable_descriptives[2].maximum, type(2), NUMERIC_LIMITS_MIN);
}


TEST(Dataset, RawVariableDistributions)
{

    Dataset dataset(3, { 2 }, { 1 });

    Tensor<type, 2> data(3, 3);

    data.setValues({ {type(2),type(2),type(1)},
                    {type(1),type(1),type(1)},
                    {type(1),type(2),type(2)} });

    dataset.set_data(data);

    const vector<Histogram> histograms = dataset.calculate_raw_variable_distributions(2);

    EXPECT_EQ(histograms.size(), 3);

    EXPECT_NEAR(histograms[0].frequencies(0), 2, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(histograms[1].frequencies(0), 1, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(histograms[2].frequencies(0), 2, NUMERIC_LIMITS_MIN);

    EXPECT_NEAR(histograms[0].centers(0), 1, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(histograms[1].centers(0), 1, NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(histograms[2].centers(0), 1, NUMERIC_LIMITS_MIN);
}


TEST(Dataset, FilterData_MixedFiltering) {
    // Test
    {
        Dataset dataset(2, { 1 }, { 1 });
        dataset.set_data_constant(type(1));

        Tensor<type, 1> minimums(2);
        minimums.setValues({ type(2), type(0) });

        Tensor<type, 1> maximums(2);
        maximums.setValues({ type(2), type(0.5) });

        Tensor<Index, 1> filtered_data = dataset.filter_data(minimums, maximums);

        EXPECT_EQ(filtered_data.size(), 2);
        EXPECT_EQ(dataset.get_sample_use(0), "None");
        EXPECT_EQ(dataset.get_sample_use(1), "None");
    }

    // Test
    {
        Dataset dataset(2, { 1 }, { 1 });
        Eigen::Tensor<type, 2> data(2, 2);
        data.setValues({ { type(1), type(2) }, { type(3), type(4) } });
        dataset.set_data(data);

        Tensor<type, 1> minimums(2);
        minimums.setValues({ type(0), type(0) });

        Tensor<type, 1> maximums(2);
        maximums.setValues({ type(2), type(3) });

        Tensor<Index, 1> filtered_data = dataset.filter_data(minimums, maximums);

        EXPECT_EQ(filtered_data.size(), 1);
        EXPECT_EQ(dataset.get_sample_use(0), "Training");
        EXPECT_EQ(dataset.get_sample_use(1), "None");
    }
}


TEST(Dataset, ScaleData)
{
    Dataset dataset(2, { 1 }, { 1 });

    Tensor<type, 2> original_data(2, 2);
    original_data.setValues({ {type(10), type(200)},
                              {type(30), type(400)} });

    dataset.set_data(original_data);

    dataset.set_raw_variable_scalers(Scaler::MinimumMaximum);
    vector<Descriptives> data_descriptives_minmax = dataset.scale_data();
    Tensor<type, 2> scaled_data_minmax = dataset.get_data();

    // Expected scaled values for column 0 (original: 10, 30):
    // 10 (min) -> 0.0
    // 30 (max) -> 1.0
    EXPECT_NEAR(scaled_data_minmax(0, 0), type(0.0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(scaled_data_minmax(1, 0), type(1.0), NUMERIC_LIMITS_MIN);

    // Expected scaled values for column 1 (original: 200, 400):
    // 200 (min) -> 0.0
    // 400 (max) -> 1.0
    EXPECT_NEAR(scaled_data_minmax(0, 1), type(0.0), NUMERIC_LIMITS_MIN);
    EXPECT_NEAR(scaled_data_minmax(1, 1), type(1.0), NUMERIC_LIMITS_MIN);
}


TEST(Dataset, UnuseConstantRawVariables)
{

    Dataset dataset(3, { 2 }, { 1 });

    Tensor<type, 2> data(3, 3);

    data.setValues({ {type(1),type(2),type(0)},
                    {type(1),type(2),type(1)},
                    {type(1),type(2),type(2)} });

    dataset.set_data(data);
    dataset.unuse_constant_raw_variables();

    dataset.unuse_constant_raw_variables();

    EXPECT_EQ(dataset.get_raw_variables_number("Input"), 0);
    EXPECT_EQ(dataset.get_raw_variables_number("Target"), 1);

}


TEST(Dataset, CalculateTargetDistribution)
{
/*
    Dataset dataset(5, { 3 }, { 2 });
    Tensor<type, 2> data(5, 4);

    data.setValues({ {type(2),type(5),type(6),type(0)},
                    {type(2),type(9),type(1),type(0)},
                    {type(2),type(9),type(1),type(NAN)},
                    {type(6),type(5),type(6),type(1)},
                    {type(0),type(1),type(0),type(1)} });


    dataset.set_data(data);
    std::vector<Index> input_variables_indices;
    std::vector<Index> target_variables_indices;

    for (Index i = 0; i < 4; i++)
        input_variables_indices.push_back(i);


    target_variables_indices.push_back(0);

    dataset.set_raw_variable_indices(input_variables_indices, target_variables_indices);

    Tensor<Index, 1> target_distribution = dataset.calculate_target_distribution();

    Tensor<Index, 1> solution(2);
    solution(0) = 1;
    solution(1) = 4;

    EXPECT_EQ(target_distribution(0), solution(0));
    EXPECT_EQ(target_distribution(1), solution(1));

    // Test more two classes

    data.resize(5, 9);
    data.setZero();

    data.setValues({ {type(2),type(5),type(6),type(9),type(8),type(7),type(1),type(0),type(0)},
                    {type(2),type(9),type(1),type(9),type(4),type(5),type(0),type(1),type(0)},
                    {type(6),type(5),type(6),type(7),type(3),type(2),type(0),type(0),type(1)},
                    {type(6),type(5),type(6),type(7),type(3),type(2),type(0),type(0),type(1)},
                    {type(0),type(NAN),type(1),type(0),type(2),type(2),type(0),type(1),type(0)} });

    dataset.set(5, { 8 }, { 1 });

    dataset.set_data(data);

    target_variables_indices.resize(3);
    target_variables_indices.push_back(6);
    target_variables_indices.push_back(7);
    target_variables_indices.push_back(8);

    input_variables_indices.resize(2);
    input_variables_indices.push_back(0);
    input_variables_indices.push_back(1);

    dataset.set_raw_variable_indices(input_variables_indices, target_variables_indices);

    target_distribution = dataset.calculate_target_distribution();

    EXPECT_EQ(target_distribution[0], 4);
    EXPECT_EQ(target_distribution[1], 1);
    EXPECT_EQ(target_distribution[2], 2);
*/
}


TEST(Dataset, ReadCSV_Basic)
{
    const string temp_csv_file_path = "temp_data_readcsv_test_basic.csv";

    const string csv_content =
        "variable_1,variable_2,target_1\n"
        "10,10,0\n"
        "20,20,1\n";

    create_temp_csv_file(temp_csv_file_path, csv_content);

    Dataset dataset;

    dataset.set_data_path(temp_csv_file_path);
    dataset.set_separator(Dataset::Separator::Comma);
    dataset.set_has_header(true);
    dataset.set_has_ids(false);
    dataset.set_missing_values_label("NA");
    dataset.set_codification(Dataset::Codification::UTF8);
    dataset.set_display(false);

    ASSERT_NO_THROW(dataset.read_csv());

    dataset.set_default_raw_variables_uses();

    dataset.set_dimensions("Input", { dataset.get_variables_number("Input") });
    dataset.set_dimensions("Target", { dataset.get_variables_number("Target") });

    EXPECT_EQ(dataset.get_samples_number(), 2);
    EXPECT_EQ(dataset.get_raw_variables_number(), 3);
    EXPECT_EQ(dataset.get_variables_number(), 3);

    // Raw Variables
    const auto& raw_vars = dataset.get_raw_variables();
    ASSERT_EQ(raw_vars.size(), 3);
    EXPECT_EQ(raw_vars[0].name, "variable_1");
    EXPECT_EQ(raw_vars[1].name, "variable_2");
    EXPECT_EQ(raw_vars[2].name, "target_1");

    EXPECT_EQ(raw_vars[0].type, Dataset::RawVariableType::Numeric);
    EXPECT_EQ(raw_vars[1].type, Dataset::RawVariableType::Numeric);
    EXPECT_EQ(raw_vars[2].type, Dataset::RawVariableType::Binary);

    EXPECT_EQ(raw_vars[0].use, "Input");
    EXPECT_EQ(raw_vars[1].use, "Input");
    EXPECT_EQ(raw_vars[2].use, "Target");

    // Data Tensor Content
    const Tensor<type, 2>& data = dataset.get_data();
    ASSERT_EQ(data.dimension(0), 2);
    ASSERT_EQ(data.dimension(1), 3);

    vector<Index> var1_indices = dataset.get_variable_indices(0);
    vector<Index> var2_indices = dataset.get_variable_indices(1);
    vector<Index> target1_indices = dataset.get_variable_indices(2);

    ASSERT_EQ(var1_indices.size(), 1);
    ASSERT_EQ(var2_indices.size(), 1);
    ASSERT_EQ(target1_indices.size(), 1);

    Index v1_idx = var1_indices[0];
    Index v2_idx = var2_indices[0];
    Index t1_idx = target1_indices[0];

    vector<Index> all_indices_ordered = { v1_idx, v2_idx, t1_idx };
    vector<Index> sorted_indices = all_indices_ordered;
    sort(sorted_indices.begin(), sorted_indices.end());
    EXPECT_TRUE(sorted_indices[0] == 0 && sorted_indices[1] == 1 && sorted_indices[2] == 2);

    EXPECT_EQ(v1_idx, 0);
    EXPECT_EQ(v2_idx, 1);
    EXPECT_EQ(t1_idx, 2);

    EXPECT_NEAR(data(0, v1_idx), 10.0, 1e-9);
    EXPECT_NEAR(data(0, v2_idx), 10.0, 1e-9);
    EXPECT_NEAR(data(0, t1_idx), 0.0, 1e-9);

    EXPECT_NEAR(data(1, v1_idx), 20.0, 1e-9);
    EXPECT_NEAR(data(1, v2_idx), 20.0, 1e-9);
    EXPECT_NEAR(data(1, t1_idx), 1.0, 1e-9);

    dimensions input_dims = dataset.get_dimensions("Input");
    dimensions target_dims = dataset.get_dimensions("Target");

    ASSERT_EQ(input_dims.size(), 1);
    EXPECT_EQ(input_dims[0], 2);

    ASSERT_EQ(target_dims.size(), 1);
    EXPECT_EQ(target_dims[0], 1);

    // Missing Values Info
    EXPECT_EQ(dataset.get_missing_values_number(), 0);
    EXPECT_FALSE(dataset.has_nan());
    EXPECT_EQ(dataset.count_rows_with_nan(), 0);
    Tensor<Index, 1> nans_per_raw_var = dataset.count_nans_per_raw_variable();
    ASSERT_EQ(nans_per_raw_var.size(), 3);
    for (Index i = 0; i < 3; ++i) EXPECT_EQ(nans_per_raw_var(i), 0);

    // Data File Preview
    const auto& preview = dataset.get_data_file_preview();
    ASSERT_EQ(preview.size(), 3) << "Preview size should be 3 for a 3-line file with header.";
    if (preview.size() >= 1) 
    {
        ASSERT_EQ(preview[0].size(), 3);
        EXPECT_EQ(preview[0][0], "variable_1");
        EXPECT_EQ(preview[0][1], "variable_2");
        EXPECT_EQ(preview[0][2], "target_1");
    }
    if (preview.size() >= 2) 
    {
        ASSERT_EQ(preview[1].size(), 3);
        EXPECT_EQ(preview[1][0], "10");
        EXPECT_EQ(preview[1][1], "10");
        EXPECT_EQ(preview[1][2], "0");
    }
    if (preview.size() >= 3) 
    {
        ASSERT_EQ(preview[2].size(), 3);
        EXPECT_EQ(preview[2][0], "20");
        EXPECT_EQ(preview[2][1], "20");
        EXPECT_EQ(preview[2][2], "1");
    }

    remove(temp_csv_file_path.c_str());
}


TEST(Dataset, ReadCSV_SpaceSeparator)
{
    const string temp_csv_file_path = "temp_data_space_sep.csv";
    const string csv_content =
        "var1 var2 target\n"
        "100 200 1\n"
        "150 250 0\n";

    create_temp_csv_file(temp_csv_file_path, csv_content);

    Dataset dataset;
    dataset.set_data_path(temp_csv_file_path);
    dataset.set_separator(Dataset::Separator::Space);
    dataset.set_has_header(true);
    dataset.set_display(false);

    ASSERT_NO_THROW(dataset.read_csv());

    EXPECT_EQ(dataset.get_samples_number(), 2);
    EXPECT_EQ(dataset.get_raw_variables_number(), 3);

    const auto& raw_vars = dataset.get_raw_variables();
    EXPECT_EQ(raw_vars[0].name, "var1");
    EXPECT_EQ(raw_vars[1].name, "var2");
    EXPECT_EQ(raw_vars[2].name, "target");
    EXPECT_EQ(raw_vars[2].type, Dataset::RawVariableType::Binary);

    const Tensor<type, 2>& data = dataset.get_data();
    Index v1_idx = dataset.get_variable_indices(0)[0];
    Index v2_idx = dataset.get_variable_indices(1)[0];
    Index t_idx = dataset.get_variable_indices(2)[0];

    EXPECT_NEAR(data(0, v1_idx), 100.0, 1e-9);
    EXPECT_NEAR(data(0, v2_idx), 200.0, 1e-9);
    EXPECT_NEAR(data(0, t_idx), 1.0, 1e-9);
    EXPECT_NEAR(data(1, v1_idx), 150.0, 1e-9);
    EXPECT_NEAR(data(1, v2_idx), 250.0, 1e-9);
    EXPECT_NEAR(data(1, t_idx), 0.0, 1e-9);

    remove(temp_csv_file_path.c_str());
}


TEST(Dataset, ReadCSV_WithSampleIDs)
{
    const string temp_csv_file_path = "temp_data_sample_ids.csv";
    const string csv_content =
        "ID,feature1,feature2\n"
        "sampleA,10,20\n"
        "sampleB,15,25\n";

    create_temp_csv_file(temp_csv_file_path, csv_content);

    Dataset dataset;
    dataset.set_data_path(temp_csv_file_path);
    dataset.set_separator(Dataset::Separator::Comma);
    dataset.set_has_header(true);
    dataset.set_has_ids(true);
    dataset.set_display(false);

    ASSERT_NO_THROW(dataset.read_csv());

    EXPECT_EQ(dataset.get_samples_number(), 2);
    EXPECT_EQ(dataset.get_raw_variables_number(), 2);

    const auto& raw_vars = dataset.get_raw_variables();
    ASSERT_EQ(raw_vars.size(), 2);
    EXPECT_EQ(raw_vars[0].name, "feature1");
    EXPECT_EQ(raw_vars[1].name, "feature2");

    const vector<string>& sample_ids = dataset.get_sample_ids();
    ASSERT_EQ(sample_ids.size(), 2);
    EXPECT_EQ(sample_ids[0], "sampleA");
    EXPECT_EQ(sample_ids[1], "sampleB");

    const Tensor<type, 2>& data = dataset.get_data();
    Index f1_idx = dataset.get_variable_indices(0)[0];
    Index f2_idx = dataset.get_variable_indices(1)[0];

    EXPECT_NEAR(data(0, f1_idx), 10.0, 1e-9);
    EXPECT_NEAR(data(0, f2_idx), 20.0, 1e-9);
    EXPECT_NEAR(data(1, f1_idx), 15.0, 1e-9);
    EXPECT_NEAR(data(1, f2_idx), 25.0, 1e-9);

    remove(temp_csv_file_path.c_str());
}


TEST(Dataset, ReadCSV_EmptyLinesAndWhitespaceSkipped)
{
    const string temp_csv_file_path = "temp_data_emptylines.csv";
    const string csv_content =
        "h1,h2\n"
        "\n"
        "  \t  \n"
        "1,10\n"
        "\n"
        "2,20\n";
    create_temp_csv_file(temp_csv_file_path, csv_content);

    Dataset dataset;
    dataset.set_data_path(temp_csv_file_path);
    dataset.set_separator(opennn::Dataset::Separator::Comma);
    dataset.set_has_header(true);
    dataset.set_display(false);

    ASSERT_NO_THROW(dataset.read_csv());

    EXPECT_EQ(dataset.get_samples_number(), 2);
    const Tensor<type, 2>& data = dataset.get_data();
    Index h1_idx = dataset.get_variable_indices(0)[0];
    Index h2_idx = dataset.get_variable_indices(1)[0];
    EXPECT_NEAR(data(0, h1_idx), 1.0, 1e-9);
    EXPECT_NEAR(data(0, h2_idx), 10.0, 1e-9);
    EXPECT_NEAR(data(1, h1_idx), 2.0, 1e-9);
    EXPECT_NEAR(data(1, h2_idx), 20.0, 1e-9);

    remove(temp_csv_file_path.c_str());
}


/*
void Dataset::test_calculate_input_target_raw_variable_correlations()
{
    // Test 1 (numeric and numeric trivial case)

    data.resize(3, 4);

    data.setValues({{type(1), type(1), type(-1), type(1)},
                    {type(2), type(2), type(-2), type(2)},
                    {type(3), type(3), type(-3), type(3)}});

    dataset.set_data(data);

    Tensor<Index, 1> input_raw_variable_indices(3);
    input_raw_variable_indices.setValues({0, 1, 2});

    Tensor<Index, 1> target_raw_variable_indices(1);
    target_raw_variable_indices.setValues({3});

    dataset.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    Tensor<Correlation, 2> input_target_raw_variable_correlations = dataset.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(input_target_raw_variable_correlations(0,0).r == 1);
    EXPECT_EQ(input_target_raw_variable_correlations(1,0).r == 1);
    EXPECT_EQ(input_target_raw_variable_correlations(2,0).r == -1);

    // Test 2 (numeric and numeric non trivial case)

    data.resize(3, 4);
    data.setValues({{type(1), type(2), type(4), type(1)},
                    {type(2), type(3), type(9), type(2)},
                    {type(3), type(1), type(10), type(2)}});

    dataset.set_data(data);

    input_raw_variable_indices.setValues({0, 1});

    target_raw_variable_indices.resize(2);
    target_raw_variable_indices.setValues({2,3});

    dataset.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    input_target_raw_variable_correlations = dataset.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(input_target_raw_variable_correlations(0,0).r < 1 && input_target_raw_variable_correlations(0,0).r > -1 );
    EXPECT_EQ(input_target_raw_variable_correlations(1,0).r < 1 && input_target_raw_variable_correlations(1,0).r > -1);
    EXPECT_EQ(input_target_raw_variable_correlations(2,0).r < 1 && input_target_raw_variable_correlations(2,0).r > -1);

    // Test 3 (binary and binary non trivial case)

    data.setValues({{type(0), type(0), type(1), type(0)},
                    {type(1), type(0), type(0), type(1)},
                    {type(1), type(0), type(0), type(1)}});

    dataset.set_data(data);

    input_raw_variable_indices.resize(3);
    input_raw_variable_indices.setValues({0, 1, 2});

    target_raw_variable_indices.setValues({3});

    dataset.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    input_target_raw_variable_correlations = dataset.calculate_input_target_raw_variable_pearson_correlations();

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

    dataset.set_data(data);

    input_raw_variable_indices.resize(3);
    input_raw_variable_indices.setValues({0, 1, 2});

    target_raw_variable_indices.setValues({3});

    dataset.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    input_target_raw_variable_correlations = dataset.calculate_input_target_raw_variable_pearson_correlations();

    for(Index i = 0; i < input_target_raw_variable_correlations.size(); i++)
    {
        EXPECT_EQ(input_target_raw_variable_correlations(i).r == 1);
        EXPECT_EQ(input_target_raw_variable_correlations(i).form == Correlation::Form::Linear);
    }

    // Test 5 (categorical and categorical)

    dataset = Dataset();

    dataset.set("../../datasets/correlation_tests.csv", ",", false);

    input_raw_variable_indices.resize(2);
    input_raw_variable_indices.setValues({0, 3});

    target_raw_variable_indices.resize(1);
    target_raw_variable_indices.setValues({4});

    dataset.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    input_target_raw_variable_correlations = dataset.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(input_target_correlations(1,0).r < 1);
    EXPECT_EQ(input_target_correlations(1,0).form == Correlation::Form::Logistic);

    // Test 6 (numeric and binary)

    input_variables_indices.resize(3);
    input_variables_indices.setValues({1, 2, 5});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    dataset.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    input_target_correlations = dataset.calculate_input_target_raw_variable_pearson_correlations();

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

    dataset.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    input_target_correlations = dataset.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1);
    EXPECT_EQ(input_target_correlations(0,0).form == Form::Logistic);

    // Test 8 (binary and categorical)

    input_variables_indices.resize(3);
    input_variables_indices.setValues({2,5,6});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({0});

    dataset.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    input_target_correlations = dataset.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1);
    EXPECT_EQ(input_target_correlations(0,0).form == Form::Logistic);

    EXPECT_EQ(-1 < input_target_correlations(1,0).r && input_target_correlations(1,0).r < 1);
    EXPECT_EQ(input_target_correlations(1,0).form == Form::Logistic);

    EXPECT_EQ(-1 < input_target_correlations(2,0).r && input_target_correlations(1,0).r < 1);
    EXPECT_EQ(input_target_correlations(2,0).form == Form::Logistic);

    // With missing values or NAN

    // Test 9 (categorical and categorical)

    dataset.set("../../../opennn/datasets/correlation_tests_with_nan.csv",',', false);

    input_raw_variable_indices.resize(2);
    input_raw_variable_indices.setValues({0, 3});

    target_raw_variable_indices.resize(1);
    target_raw_variable_indices.setValues({4});

    dataset.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    input_target_correlations = dataset.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1);
    EXPECT_EQ(input_target_correlations(0,0).form == Form::Logistic);

    EXPECT_EQ(-1 < input_target_correlations(1,0).r && input_target_correlations(1,0).r < 1);
    EXPECT_EQ(input_target_correlations(1,0).form == Form::Logistic);

    // Test 10 (numeric and binary)

    input_variables_indices.resize(3);
    input_variables_indices.setValues({1, 2, 5});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({6});

    dataset.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    input_target_correlations = dataset.calculate_input_target_raw_variable_pearson_correlations();

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

    dataset.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    input_target_correlations = dataset.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1);
    EXPECT_EQ(input_target_correlations(0,0).form == Form::Logistic);

    // Test 12 (binary and categorical)

    input_variables_indices.resize(3);
    input_variables_indices.setValues({1, 2, 5});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({0});

    dataset.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    input_target_correlations = dataset.calculate_input_target_raw_variable_pearson_correlations();

    EXPECT_EQ(-1 < input_target_correlations(0,0).r && input_target_correlations(0,0).r < 1);
    EXPECT_EQ(input_target_correlations(0,0).form == Form::Logistic);

    EXPECT_EQ(-1 < input_target_correlations(1,0).r && input_target_correlations(1,0).r < 1);
    EXPECT_EQ(input_target_correlations(1,0).form == Form::Logistic);

    EXPECT_EQ(-1 < input_target_correlations(2,0).r && input_target_correlations(1,0).r < 1);
    EXPECT_EQ(input_target_correlations(2,0).form == Form::Logistic);

}


void Dataset::test_calculate_input_raw_variable_correlations()
{
    // Test 1 (numeric and numeric trivial case)

    data.resize(3, 4);

    data.setValues({{type(1), type(1), type(-1), type(1)},
                    {type(2), type(2), type(-2), type(2)},
                    {type(3), type(3), type(-3), type(3)}});

    dataset.set_data(data);

    Tensor<Index, 1> input_raw_variable_indices(3);
    input_raw_variable_indices.setValues({0, 1, 2});

    Tensor<Index, 1> target_raw_variable_indices(1);
    target_raw_variable_indices.setValues({3});

    dataset.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    Tensor<Correlation, 2> inputs_correlations = dataset.calculate_input_raw_variable_pearson_correlations();

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

    dataset.set_data(data);

    input_raw_variable_indices.setValues({0, 1});

    target_raw_variable_indices.setValues({2,3});

    dataset.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    inputs_correlations = dataset.calculate_input_raw_variable_pearson_correlations();

    for(Index i = 0; i <  dataset.get_raw_variables_number("Input") ; i++)
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

    dataset.set_data(data);

    input_raw_variable_indices.resize(3);
    input_raw_variable_indices.setValues({0, 1, 2});

    target_raw_variable_indices.setValues({3});

    dataset.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    inputs_correlations = dataset.calculate_input_raw_variable_pearson_correlations();

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

    dataset.set_data(data);

    input_raw_variable_indices.resize(3);
    input_raw_variable_indices.setValues({0, 1, 2});

    target_raw_variable_indices.setValues({3});

    dataset.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);
    /*
    inputs_correlations = dataset.calculate_input_raw_variable_pearson_correlations();

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

    dataset.set("../../datasets/correlation_tests.csv",',', false);

    input_raw_variable_indices.resize(3);
    input_raw_variable_indices.setValues({0, 3, 4});

    target_raw_variable_indices.resize(1);
    target_raw_variable_indices.setValues({5});

    dataset.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    inputs_correlations = dataset.calculate_input_raw_variable_correlations()(0);

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

    dataset.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    inputs_correlations = dataset.calculate_input_raw_variable_correlations()(0);

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

    dataset.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    inputs_correlations = dataset.calculate_input_raw_variable_correlations()(0);

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

    dataset.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    inputs_correlations = dataset.calculate_input_raw_variable_correlations()(0);

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

    dataset.set_missing_values_label("NA");
    dataset.set("../../datasets/correlation_tests_with_nan.csv",',', false);

    input_raw_variable_indices.resize(3);
    input_raw_variable_indices.setValues({0, 3, 4});

    target_raw_variable_indices.resize(1);
    target_raw_variable_indices.setValues({6});

    dataset.set_input_target_raw_variables_indices(input_raw_variable_indices, target_raw_variable_indices);

    inputs_correlations = dataset.calculate_input_raw_variable_correlations()(0);

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

    dataset.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    inputs_correlations = dataset.calculate_input_raw_variable_correlations()(0);

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

    dataset.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    inputs_correlations = dataset.calculate_input_raw_variable_correlations()(0);

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

    dataset.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    inputs_correlations = dataset.calculate_input_raw_variable_correlations()(0);

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


void Dataset::test_unuse_repeated_samples()
{
    Tensor<Index, 1> indices;

    dataset.set();

    // Test

    data.resize(3, 3);

    data.setValues({{type(1),type(2),type(2)},
                    {type(1),type(2),type(2)},
                    {type(1),type(6),type(6)}});

    dataset = opennn::Dataset();

    dataset.set_data(data);
    dataset.set("Training");

    indices = dataset.unuse_repeated_samples();

    EXPECT_EQ(indices.size() == 1);
    EXPECT_EQ(indices(0) == 1);

    // Test

    data.resize(4,3);

    data.setValues({{type(1),type(2),type(2)},
                   {type(1),type(2),type(2)},
                   {type(1),type(2),type(4)},
                   {type(1),type(2),type(4)}});

    dataset = opennn::Dataset();

    dataset.set_data(data);
    dataset.set("Training");

    indices = dataset.unuse_repeated_samples();

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

    dataset.set();

    dataset.set_data(data);
    dataset.set("Training");

    indices = dataset.unuse_repeated_samples();

    EXPECT_EQ(indices.size() == 3);
    EXPECT_EQ(contains(indices, 1));
    EXPECT_EQ(contains(indices, 3));
    EXPECT_EQ(contains(indices, 4));
}


void Dataset::test_unuse_uncorrelated_raw_variables()
{
    data.resize(3, 3);
    data.setValues({{type(1),type(0),type(0)},
                    {type(1),type(0),type(0)},
                    {type(1),type(0),type(1)}});
}


void Dataset::test_calculate_training_negatives()
{
    Index training_negatives;
    Index target_index;

    // Test

    data.resize(3, 3);

    data.setValues({{ 1, 1, 1},
                    {-1,-1, 0},
                    { 0, 1, 1}});

    dataset.set_data(data);

    training_indices.resize(2);
    training_indices.setValues({0,1});

    input_variables_indices.resize(2);
    input_variables_indices.setValues({0, 1});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({2});

    target_index = 2;

    dataset.set("Testing");
    dataset.set_training(training_indices);

    //training_negatives = dataset.calculate_training_negatives(target_index);
    training_negatives = dataset.calculate_negatives("Training",target_index);

    EXPECT_EQ(training_negatives == 1);
}


void Dataset::test_calculate_selection_negatives()
{
    vector<Index> selection_indices;
    Tensor<Index, 1> input_variables_indices;
    Tensor<Index, 1> target_variables_indices;

    // Test

    data.resize(3, 3);

    data.setValues({{1, 1, 1},{0, 0, 1},{0, 1, 1}});

    dataset.set_data(data);

    selection_indices.resize(2);
    selection_indices.setValues({0,1});

    input_variables_indices.resize(2);
    input_variables_indices.setValues({0, 1});

    target_variables_indices.resize(1);
    target_variables_indices.setValues({2});

    Index target_index = 2;

    dataset.set("Testing");

    dataset.set_selection(selection_indices);

    dataset.set_input_target_raw_variables_indices(input_variables_indices, target_variables_indices);

    //Index selection_negatives = dataset.calculate_selection_negatives(target_index);
    Index selection_negatives = dataset.calculate_negatives("Selection",target_index);
    dataset.calculate_negatives("Training",target_index);
    data = dataset.get_data();

    EXPECT_EQ(selection_negatives == 0);

}


void Dataset::test_fill()
{
    data.resize(3, 3);
    data.setValues({{1,4,7},{2,5,8},{3,6,9}});
    dataset.set_data(data);

    dataset.set("Training");

    const Index training_samples_number = dataset.get_samples_number("Training");

    const Tensor<Index, 1> training_samples_indices = dataset.get_sample_indices("Training");

    const Tensor<Index, 1> input_variables_indices = dataset.get_variable_indices("Input");
    const Tensor<Index, 1> target_variables_indices = dataset.get_variable_indices("Target");

    batch.set(training_samples_number, &dataset);
    /*
    batch.fill(training_samples_indices, input_variables_indices, {}, target_variables_indices);

    Tensor<type, 2> input_data(3,2);
    input_data.setValues({{1,4},{2,5},{3,6}});

    Tensor<type, 2> target_data(3,1);
    target_data.setValues({{7},{8},{9}});

    const vector<pair<type*, dimensions>> input_pairs = batch.get_input_pairs();

    const TensorMap<Tensor<type, 2>> inputs = tensor_map<2>(input_pairs[0]);

    const pair<type*, dimensions> targets_pair = batch.get_targets_pair();

    const TensorMap<Tensor<type, 2>> targets = tensor_map<2>(targets_pair);

    EXPECT_EQ(are_equal(inputs, input_data));
    EXPECT_EQ(are_equal(targets, target_data));

}

}
*/
