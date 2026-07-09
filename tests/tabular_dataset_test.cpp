#include "pch.h"

#include "opennn/dataset.h"
#include "opennn/tabular_dataset.h"

using namespace opennn;

namespace
{

void create_temp_csv_file(const string& file_path, const string& content)
{
    ofstream outfile(file_path);
    if(!outfile.is_open())
        throw runtime_error("Failed to open temporary CSV file for writing: " + file_path);

    outfile << content;
    outfile.close();
}

}


TEST(TabularDataset, DefaultConstructor)
{
    TabularDataset dataset;

    EXPECT_EQ(dataset.get_variables_number(), 0);
    EXPECT_EQ(dataset.get_samples_number(), 0);
}


TEST(TabularDataset, DimensionsConstructor)
{
    TabularDataset dataset(1, { 1 }, { 1 });

    EXPECT_EQ(dataset.get_samples_number(), 1);
    EXPECT_EQ(dataset.get_variables_number(), 2);
    EXPECT_EQ(dataset.get_variables_number("Input"), 1);
    EXPECT_EQ(dataset.get_variables_number("Target"), 1);
}


TEST(TabularDataset, VariableDescriptivesZero)
{
    TabularDataset dataset(1, { 1 }, { 1 });
    dataset.set_data_constant(type(0));

    const vector<Descriptives> variable_descriptives = dataset.calculate_feature_descriptives();

    EXPECT_EQ(variable_descriptives.size(), 2);

    EXPECT_NEAR(variable_descriptives[0].minimum, 0, EPSILON);
    EXPECT_NEAR(variable_descriptives[0].maximum, 0, EPSILON);
    EXPECT_NEAR(variable_descriptives[0].mean, 0, EPSILON);
    EXPECT_NEAR(variable_descriptives[0].standard_deviation, 0, EPSILON);
}


TEST(TabularDataset, VariableDescriptives)
{
    const Index samples_number = 3;
    const Index inputs_number = 2;
    const Index targets_number = 1;

    TabularDataset dataset(samples_number, { inputs_number }, { targets_number });

    MatrixR data(samples_number, inputs_number + targets_number);

    data << type(-1000),type(2),type(0),
            type(1)    ,type(4),type(2),
            type(1)    ,type(4),type(0);

    dataset.set_data(data);

    const vector<Descriptives> variable_descriptives = dataset.calculate_feature_descriptives();

    EXPECT_EQ(variable_descriptives.size(), 3);
    EXPECT_NEAR(variable_descriptives[0].minimum, type(-1000), EPSILON);
    EXPECT_NEAR(variable_descriptives[1].minimum, type(2), EPSILON);
    EXPECT_NEAR(variable_descriptives[2].minimum, 0, EPSILON);
    EXPECT_NEAR(variable_descriptives[0].maximum, type(1), EPSILON);
    EXPECT_NEAR(variable_descriptives[1].maximum, type(4), EPSILON);
    EXPECT_NEAR(variable_descriptives[2].maximum, type(2), EPSILON);
}


TEST(TabularDataset, RawVariableDistributions)
{
    TabularDataset dataset(3, { 2 }, { 1 });

    MatrixR data(3, 3);

    data << type(2),type(2),type(1),
            type(1),type(1),type(1),
            type(1),type(2),type(2);

    dataset.set_data(data);

    const vector<Histogram> histograms = dataset.calculate_variable_distributions(2);

    EXPECT_EQ(histograms.size(), 3);

    EXPECT_NEAR(histograms[0].frequencies(0), 2, EPSILON);
    EXPECT_NEAR(histograms[1].frequencies(0), 1, EPSILON);
    EXPECT_NEAR(histograms[2].frequencies(0), 2, EPSILON);

    EXPECT_NEAR(histograms[0].centers(0), 1, EPSILON);
    EXPECT_NEAR(histograms[1].centers(0), 1, EPSILON);
    EXPECT_NEAR(histograms[2].centers(0), 1, EPSILON);
}


TEST(TabularDataset, ScaleData)
{
    TabularDataset dataset(2, { 1 }, { 1 });

    MatrixR original_data(2, 2);
    original_data << type(10), type(200),
                     type(30), type(400);

    dataset.set_data(original_data);

    dataset.set_variable_scalers("MinimumMaximum");
    vector<Descriptives> data_descriptives_minmax = dataset.scale_data();
    MatrixR scaled_data_minmax = dataset.get_data();

    EXPECT_NEAR(scaled_data_minmax(0, 0), type(-1.0), EPSILON);
    EXPECT_NEAR(scaled_data_minmax(1, 0), type(1.0), EPSILON);

    EXPECT_NEAR(scaled_data_minmax(0, 1), type(-1.0), EPSILON);
    EXPECT_NEAR(scaled_data_minmax(1, 1), type(1.0), EPSILON);
}


TEST(TabularDataset, UnuseConstantRawVariables)
{
    TabularDataset dataset(3, { 2 }, { 1 });

    MatrixR data(3, 3);

    data << type(1),type(2),type(0),
            type(1),type(2),type(1),
            type(1),type(2),type(2);

    dataset.set_data(data);

    EXPECT_GE(dataset.get_variables_number(), 0);
}


TEST(TabularDataset, CalculateTargetDistribution)
{
    TabularDataset dataset(5, { 3 }, { 1 });
    MatrixR data(5, 4);

    data << type(2),type(5),type(6),type(0),
            type(2),type(9),type(1),type(0),
            type(2),type(9),type(1),type(NAN),
            type(6),type(5),type(6),type(1),
            type(0),type(1),type(0),type(1);

    dataset.set_data(data);
    vector<Index> input_variables_indices;
    vector<Index> target_variables_indices;

    for (Index i = 0; i < 3; i++)
        input_variables_indices.push_back(i);

    target_variables_indices.push_back(3);

    dataset.set_variable_indices(input_variables_indices, target_variables_indices);

    VectorI target_distribution = dataset.calculate_target_distribution();

    VectorI solution(2);
    solution(0) = 2;
    solution(1) = 2;

    EXPECT_EQ(target_distribution(0), solution(0));
    EXPECT_EQ(target_distribution(1), solution(1));

    TabularDataset dataset_2(5, { 6 }, { 3 });

    data.resize(5, 9);
    data.setZero();

    data << type(2),type(5),type(6),type(9),type(8),type(7),type(1),type(0),type(0),
            type(2),type(9),type(1),type(9),type(4),type(5),type(0),type(1),type(0),
            type(6),type(5),type(6),type(7),type(3),type(2),type(0),type(0),type(1),
            type(6),type(5),type(6),type(7),type(3),type(2),type(0),type(0),type(1),
            type(0),type(NAN),type(1),type(0),type(2),type(2),type(0),type(1),type(0);

    dataset_2.set_data(data);

    vector<Index> input_variables_indices_2;
    vector<Index> target_variables_indices_2;

    target_variables_indices_2.push_back(6);
    target_variables_indices_2.push_back(7);
    target_variables_indices_2.push_back(8);

    for (Index i = 0; i < 6; i++)
        input_variables_indices_2.push_back(i);

    dataset_2.set_variable_indices(input_variables_indices_2, target_variables_indices_2);

    VectorI target_distribution_2 = dataset_2.calculate_target_distribution();

    EXPECT_EQ(target_distribution_2[0], 1);
    EXPECT_EQ(target_distribution_2[1], 2);
    EXPECT_EQ(target_distribution_2[2], 2);
}


TEST(TabularDataset, ReadCSV_Basic)
{
    const string temp_csv_file_path = "temp_data_readcsv_test_basic.csv";

    const string csv_content =
        "variable_1,variable_2,target_1\n"
        "10,10,0\n"
        "20,20,1\n";

    create_temp_csv_file(temp_csv_file_path, csv_content);

    TabularDataset dataset;

    dataset.set_data_path(temp_csv_file_path);
    dataset.set_separator(Dataset::Separator::Comma);
    dataset.set_has_header(true);
    dataset.set_has_ids(false);
    dataset.set_missing_values_label("NA");
    dataset.set_codification(Dataset::Codification::UTF8);
    dataset.set_display(false);

    ASSERT_NO_THROW(dataset.read_csv());

    EXPECT_EQ(dataset.get_samples_number(), 2);
    EXPECT_EQ(dataset.get_variables_number(), 3);

    const auto& raw_vars = dataset.get_variables();
    ASSERT_EQ(raw_vars.size(), 3);
    EXPECT_EQ(raw_vars[0].name, "variable_1");
    EXPECT_EQ(raw_vars[1].name, "variable_2");
    EXPECT_EQ(raw_vars[2].name, "target_1");

    EXPECT_EQ(raw_vars[0].type, VariableType::Numeric);
    EXPECT_EQ(raw_vars[1].type, VariableType::Numeric);
    EXPECT_EQ(raw_vars[2].type, VariableType::Binary);

    dataset.set_variable_role(0, "Input");
    dataset.set_variable_role(1, "Input");
    dataset.set_variable_role(2, "Target");

    dataset.set_shape("Input", { dataset.get_variables_number("Input") });
    dataset.set_shape("Target", { dataset.get_variables_number("Target") });

    EXPECT_EQ(dataset.get_variables()[0].role, VariableRole::Input);
    EXPECT_EQ(dataset.get_variables()[1].role, VariableRole::Input);
    EXPECT_EQ(dataset.get_variables()[2].role, VariableRole::Target);

    const MatrixR& data = dataset.get_data();
    ASSERT_EQ(data.rows(), 2);
    ASSERT_EQ(data.cols(), 3);

    Index var1_index = dataset.get_variable_index("variable_1");
    Index var2_index = dataset.get_variable_index("variable_2");
    Index target1_index = dataset.get_variable_index("target_1");

    EXPECT_EQ(var1_index, 0);
    EXPECT_EQ(var2_index, 1);
    EXPECT_EQ(target1_index, 2);

    vector<Index> all_indices_ordered = { var1_index, var2_index, target1_index };
    vector<Index> sorted_indices = all_indices_ordered;
    sort(sorted_indices.begin(), sorted_indices.end());
    EXPECT_TRUE(sorted_indices[0] == 0 && sorted_indices[1] == 1 && sorted_indices[2] == 2);

    EXPECT_NEAR(data(0, var1_index), 10.0, 1e-9);
    EXPECT_NEAR(data(0, var2_index), 10.0, 1e-9);
    EXPECT_NEAR(data(0, target1_index), 0.0, 1e-9);

    EXPECT_NEAR(data(1, var1_index), 20.0, 1e-9);
    EXPECT_NEAR(data(1, var2_index), 20.0, 1e-9);
    EXPECT_NEAR(data(1, target1_index), 1.0, 1e-9);

    Shape input_shape = dataset.get_shape("Input");
    Shape target_shape = dataset.get_shape("Target");

    ASSERT_EQ(input_shape.rank, 1);
    EXPECT_EQ(input_shape[0], 2);

    ASSERT_EQ(target_shape.rank, 1);
    EXPECT_EQ(target_shape[0], 1);

    EXPECT_EQ(dataset.get_missing_values_number(), 0);
    EXPECT_FALSE(dataset.has_nan());
    EXPECT_EQ(dataset.count_rows_with_nan(), 0);
    VectorI nans_per_raw_var = dataset.count_nans_per_variable();
    ASSERT_EQ(nans_per_raw_var.size(), 3);
    for (Index i = 0; i < 3; ++i) EXPECT_EQ(nans_per_raw_var(i), 0);

    const auto& preview = dataset.get_data_file_preview();
    ASSERT_EQ(preview.size(), 3);
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


TEST(TabularDataset, ReadCSV_SpaceSeparator)
{
    const string temp_csv_file_path = "temp_data_space_sep.csv";
    const string csv_content =
        "var1 var2 target\n"
        "100 200 1\n"
        "150 250 0\n";

    create_temp_csv_file(temp_csv_file_path, csv_content);

    TabularDataset dataset;
    dataset.set_data_path(temp_csv_file_path);
    dataset.set_separator(Dataset::Separator::Space);
    dataset.set_has_header(true);
    dataset.set_display(false);

    ASSERT_NO_THROW(dataset.read_csv());

    EXPECT_EQ(dataset.get_samples_number(), 2);
    EXPECT_EQ(dataset.get_variables_number(), 3);

    const auto& raw_vars = dataset.get_variables();
    EXPECT_EQ(raw_vars[0].name, "var1");
    EXPECT_EQ(raw_vars[1].name, "var2");
    EXPECT_EQ(raw_vars[2].name, "target");
    EXPECT_EQ(raw_vars[2].type, VariableType::Binary);

    const MatrixR& data = dataset.get_data();
    Index v1_idx = dataset.get_variable_index(0);
    Index v2_idx = dataset.get_variable_index(1);
    Index t_idx = dataset.get_variable_index(2);

    EXPECT_NEAR(data(0, v1_idx), 100.0, 1e-9);
    EXPECT_NEAR(data(0, v2_idx), 200.0, 1e-9);
    EXPECT_NEAR(data(0, t_idx), 1.0, 1e-9);
    EXPECT_NEAR(data(1, v1_idx), 150.0, 1e-9);
    EXPECT_NEAR(data(1, v2_idx), 250.0, 1e-9);
    EXPECT_NEAR(data(1, t_idx), 0.0, 1e-9);

    remove(temp_csv_file_path.c_str());
}


TEST(TabularDataset, ReadCSV_WithSampleIDs)
{
    const string temp_csv_file_path = "temp_data_sample_ids.csv";
    const string csv_content =
        "ID,feature1,feature2\n"
        "sampleA,10,20\n"
        "sampleB,15,25\n";

    create_temp_csv_file(temp_csv_file_path, csv_content);

    TabularDataset dataset;
    dataset.set_data_path(temp_csv_file_path);
    dataset.set_separator(Dataset::Separator::Comma);
    dataset.set_has_header(true);
    dataset.set_has_ids(true);
    dataset.set_display(false);

    ASSERT_NO_THROW(dataset.read_csv());

    EXPECT_EQ(dataset.get_samples_number(), 2);
    EXPECT_EQ(dataset.get_variables_number(), 2);

    const auto& raw_vars = dataset.get_variables();
    ASSERT_EQ(raw_vars.size(), 2);
    EXPECT_EQ(raw_vars[0].name, "feature1");
    EXPECT_EQ(raw_vars[1].name, "feature2");

    const MatrixR& data = dataset.get_data();
    Index f1_idx = dataset.get_variable_index(0);
    Index f2_idx = dataset.get_variable_index(1);

    EXPECT_NEAR(data(0, f1_idx), 10.0, 1e-9);
    EXPECT_NEAR(data(0, f2_idx), 20.0, 1e-9);
    EXPECT_NEAR(data(1, f1_idx), 15.0, 1e-9);
    EXPECT_NEAR(data(1, f2_idx), 25.0, 1e-9);

    remove(temp_csv_file_path.c_str());
}


TEST(TabularDataset, ReadCSV_EmptyLinesAndWhitespaceSkipped)
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

    TabularDataset dataset;
    dataset.set_data_path(temp_csv_file_path);
    dataset.set_separator(opennn::Dataset::Separator::Comma);
    dataset.set_has_header(true);
    dataset.set_display(false);

    ASSERT_NO_THROW(dataset.read_csv());

    EXPECT_EQ(dataset.get_samples_number(), 2);
    const MatrixR& data = dataset.get_data();
    Index h1_idx = dataset.get_variable_index(0);
    Index h2_idx = dataset.get_variable_index(1);
    EXPECT_NEAR(data(0, h1_idx), 1.0, 1e-9);
    EXPECT_NEAR(data(0, h2_idx), 10.0, 1e-9);
    EXPECT_NEAR(data(1, h1_idx), 2.0, 1e-9);
    EXPECT_NEAR(data(1, h2_idx), 20.0, 1e-9);

    remove(temp_csv_file_path.c_str());
}


TEST(TabularDataset, InputTargetVariableCorrelations)
{
    MatrixR data;
    TabularDataset dataset(3, {3}, {1});

    data.resize(3, 4);

    data << type(1), type(1), type(-1), type(1),
        type(2), type(2), type(-2), type(2),
        type(-1), type(-1), type(1), type(-1);

    dataset.set_data(data);
    dataset.set_display(false);
    dataset.set_sample_roles("Training");
    vector<Index> input_features_indices(3);
    input_features_indices[0] = Index(0);
    input_features_indices[1] = Index(1);
    input_features_indices[2] = Index(2);

    vector<Index> target_features_indices(1);
    target_features_indices[0] = Index(3);

    dataset.set_variable_indices(input_features_indices, target_features_indices);

    Tensor<Correlation, 2> input_target_raw_variable_correlations = dataset.calculate_input_target_variable_pearson_correlations();

    EXPECT_EQ(input_target_raw_variable_correlations(0, 0).coefficient, 1.0);

    EXPECT_EQ(input_target_raw_variable_correlations(1,0).coefficient, 1.0);
    EXPECT_EQ(input_target_raw_variable_correlations(2, 0).coefficient, -1.0);

    data.resize(3, 4);
    data << type(1), type(2), type(4), type(1),
        type(2), type(3), type(9), type(2),
        type(3), type(1), type(10), type(2);

    dataset.set_data(data);

    input_features_indices = { 0, 1 };
    target_features_indices.resize(2);
    target_features_indices = { 2, 3 };

    dataset.set_variable_indices(input_features_indices, target_features_indices);

    input_target_raw_variable_correlations = dataset.calculate_input_target_variable_pearson_correlations();

    EXPECT_EQ(input_target_raw_variable_correlations.dimension(0), 2);
    EXPECT_EQ(input_target_raw_variable_correlations.dimension(1), 2);

    for(Index i = 0; i < input_target_raw_variable_correlations.dimension(0); ++i)
        for(Index j = 0; j < input_target_raw_variable_correlations.dimension(1); ++j)
            EXPECT_TRUE(-1 < input_target_raw_variable_correlations(i, j).coefficient
                        && input_target_raw_variable_correlations(i, j).coefficient < 1);

    data << type(0), type(0), type(1), type(0),
        type(1), type(0), type(0), type(1),
        type(1), type(0), type(0), type(1);

    dataset.set_data(data);

    input_features_indices.resize(3);
    input_features_indices = {0, 1, 2};

    target_features_indices = {3};

    dataset.set_variable_indices(input_features_indices, target_features_indices);

    input_target_raw_variable_correlations = dataset.calculate_input_target_variable_pearson_correlations();

    EXPECT_EQ(input_target_raw_variable_correlations(0,0).coefficient, 1);
    EXPECT_EQ(input_target_raw_variable_correlations(0,0).form, Correlation::Form::Identity);

    EXPECT_TRUE(isnan(input_target_raw_variable_correlations(1,0).coefficient));
    EXPECT_EQ(input_target_raw_variable_correlations(1,0).form, Correlation::Form::Identity);

    EXPECT_EQ(input_target_raw_variable_correlations(2,0).coefficient, -1);
    EXPECT_EQ(input_target_raw_variable_correlations(2,0).form, Correlation::Form::Identity);

    data << type(0), type(0), type(0), type(0),
        type(1), type(1), type(1), type(1),
        type(1), type(1), type(1), type(1);

    dataset.set_data(data);

    input_features_indices.resize(3);
    input_features_indices = {0, 1, 2};

    target_features_indices = {3};

    dataset.set_variable_indices(input_features_indices, target_features_indices);

    input_target_raw_variable_correlations = dataset.calculate_input_target_variable_pearson_correlations();

    for(Index i = 0; i < input_target_raw_variable_correlations.size(); i++)
    {
        EXPECT_EQ(input_target_raw_variable_correlations(i).coefficient, 1);
        EXPECT_EQ(input_target_raw_variable_correlations(i).form, Correlation::Form::Identity);
    }
}


TEST(TabularDataset, InputVariableCorrelations)
{
    MatrixR data;
    TabularDataset dataset(3, { 3 }, { 1 });
    data.resize(3, 4);

    data << type(1), type(1), type(-1), type(1),
            type(2), type(2), type(-2), type(2),
            type(3), type(3), type(-3), type(3);

    dataset.set_data(data);
    dataset.set_display(false);

    vector<Index> input_features_indices(3);
    input_features_indices[0] = Index(0);
    input_features_indices[1] = Index(1);
    input_features_indices[2] = Index(2);

    vector<Index> target_features_indices(1);
    target_features_indices[0] = Index(3);

    dataset.set_variable_indices(input_features_indices, target_features_indices);

    Tensor<Correlation, 2> input_correlations = dataset.calculate_input_variable_pearson_correlations();

    EXPECT_EQ(input_correlations(0,0).coefficient, 1);
    EXPECT_EQ(input_correlations(0,1).coefficient, 1);
    EXPECT_EQ(input_correlations(0,2).coefficient, -1);

    EXPECT_EQ(input_correlations(1,0).coefficient, 1);
    EXPECT_EQ(input_correlations(1,1).coefficient, 1);
    EXPECT_EQ(input_correlations(1,2).coefficient, -1);

    EXPECT_EQ(input_correlations(2,0).coefficient, -1);
    EXPECT_EQ(input_correlations(2,1).coefficient, -1);
    EXPECT_EQ(input_correlations(2,2).coefficient, 1);

    data.resize(3, 4);
    data << type(1), type(2), type(4), type(1),
            type(2), type(3), type(9), type(2),
            type(3), type(1), type(10), type(2);

    dataset.set_data(data);

    input_features_indices = {0, 1};

    target_features_indices = {2,3};

    dataset.set_variable_indices(input_features_indices, target_features_indices);

    input_correlations = dataset.calculate_input_variable_pearson_correlations();

    for(Index i = 0; i <  dataset.get_variables_number("Input"); i++)
    {
        EXPECT_EQ(input_correlations(i,i).coefficient, 1);

        for(Index j = 0; i < j ; j++)
            EXPECT_TRUE(-1 < input_correlations(i,j).coefficient && input_correlations(i,j).coefficient < 1);
    }

    data.resize(3, 4);
    data << type(0), type(0), type(1), type(1),
            type(1), type(0), type(0), type(2),
            type(1), type(0), type(0), type(2);

    dataset.set_data(data);

    input_features_indices.resize(3);
    input_features_indices = {0, 1, 2};

    target_features_indices = {3};

    dataset.set_variable_indices(input_features_indices, target_features_indices);

    input_correlations = dataset.calculate_input_variable_pearson_correlations();

    EXPECT_EQ(input_correlations(0,0).coefficient, 1);
    EXPECT_EQ(input_correlations(0,0).form, Correlation::Form::Identity);

    EXPECT_TRUE(isnan(input_correlations(1,0).coefficient));

    EXPECT_EQ(input_correlations(1,0).form, Correlation::Form::Identity);

    EXPECT_TRUE(isnan(input_correlations(1,1).coefficient));
    EXPECT_EQ(input_correlations(1,1).form, Correlation::Form::Identity);

    EXPECT_EQ(input_correlations(2,0).coefficient, -1);
    EXPECT_EQ(input_correlations(2,0).form, Correlation::Form::Identity);

    EXPECT_TRUE(isnan(input_correlations(2,1).coefficient));
    EXPECT_EQ(input_correlations(2,1).form, Correlation::Form::Identity);

    EXPECT_EQ(input_correlations(2,2).coefficient, 1);
    EXPECT_EQ(input_correlations(2,2).form, Correlation::Form::Identity);

    data << type(1), type(0), type(1), type(1),
            type(0), type(1), type(1), type(1),
            type(1), type(1), type(0), type(0);

    dataset.set_data(data);

    input_features_indices.resize(3);
    input_features_indices = {0, 1, 2};

    target_features_indices = {3};

    dataset.set_variable_indices(input_features_indices, target_features_indices);

    input_correlations = dataset.calculate_input_variable_pearson_correlations();

    EXPECT_EQ(input_correlations(0,0).coefficient, 1);
    EXPECT_EQ(input_correlations(0,0).form, Correlation::Form::Identity);

    EXPECT_TRUE(input_correlations(0,1).coefficient < 0 && input_correlations(0,1).coefficient > -1);
    EXPECT_EQ(input_correlations(0,1).form, Correlation::Form::Identity);

    EXPECT_TRUE(input_correlations(0,2).coefficient < 0 && input_correlations(0,2).coefficient > -1);
    EXPECT_EQ(input_correlations(0,2).form, Correlation::Form::Identity);

    EXPECT_TRUE(input_correlations(1,0).coefficient < 0 && input_correlations(1,0).coefficient > -1);
    EXPECT_EQ(input_correlations(1,0).form, Correlation::Form::Identity);

    EXPECT_EQ(input_correlations(1,1).coefficient, 1);
    EXPECT_EQ(input_correlations(1,1).form, Correlation::Form::Identity);

    EXPECT_EQ(input_correlations(1,2).coefficient, -0.5);
    EXPECT_EQ(input_correlations(1,2).form, Correlation::Form::Identity);

    EXPECT_TRUE(input_correlations(2,0).coefficient < 0 && input_correlations(2,0).coefficient > -1);
    EXPECT_EQ(input_correlations(2,0).form, Correlation::Form::Identity);

    EXPECT_EQ(input_correlations(2,1).coefficient, -0.5);
    EXPECT_EQ(input_correlations(2,1).form, Correlation::Form::Identity);

    EXPECT_EQ(input_correlations(2,2).coefficient, 1);
    EXPECT_EQ(input_correlations(2,2).form, Correlation::Form::Identity);
}


TEST(TabularDataset, UnuseUncorrelatedVariables)
{
    MatrixR data;
    data.resize(4, 4);

    data << type(1), type(1), type(0), type(2),
            type(2), type(0), type(0), type(4),
            type(3), type(0), type(0), type(6),
            type(4), type(1), type(0), type(8);

    TabularDataset dataset(4, { 3 }, { 1 });
    dataset.set_variable_names({ "A", "B", "C", "T" });
    dataset.set_data(data);

    vector<Index> input_indices = { 0, 1, 2 };
    vector<Index> target_indices = { 3 };
    dataset.set_variable_indices(input_indices, target_indices);

    type min_correlation = 0.25;
    vector<string> unused = dataset.unuse_uncorrelated_variables(min_correlation);

    ASSERT_EQ(unused.size(), 2);

    sort(unused.begin(), unused.end());
    EXPECT_EQ(unused[0], "B");
    EXPECT_EQ(unused[1], "C");

    const auto& raw_vars = dataset.get_variables();
    EXPECT_EQ(raw_vars[0].role, VariableRole::Input);
    EXPECT_EQ(raw_vars[1].role, VariableRole::None);
    EXPECT_EQ(raw_vars[2].role, VariableRole::None);
    EXPECT_EQ(raw_vars[3].role, VariableRole::Target);
}


TEST(TabularDataset, TargetDistributionBinaryCountsUnusedSamples)
{
    TabularDataset dataset(4, { 2 }, { 1 });

    MatrixR data(4, 3);
    data << type(1), type(2), type(0),
            type(3), type(4), type(1),
            type(5), type(6), type(0),
            type(7), type(8), type(1);
    dataset.set_data(data);

    dataset.set_sample_roles("Training");
    dataset.set_sample_role(0, "None");
    dataset.set_sample_role(1, "None");

    const VectorI distribution = dataset.calculate_target_distribution();

    ASSERT_EQ(distribution.size(), 2);
    EXPECT_EQ(distribution(0), 2);
    EXPECT_EQ(distribution(1), 2);
    EXPECT_EQ(distribution(0) + distribution(1), dataset.get_samples_number());
}


TEST(TabularDataset, MissingValuesUnuseMarksRowsUnused)
{
    TabularDataset dataset(3, { 2 }, { 1 });

    MatrixR data(3, 3);
    data << type(1),   type(2), type(0),
            type(NAN), type(5), type(1),
            type(3),   type(4), type(1);
    dataset.set_data(data);
    dataset.set_sample_roles("Training");

    EXPECT_EQ(dataset.count_nan(), 1);

    dataset.set_missing_values_method(TabularDataset::MissingValuesMethod::Unuse);
    dataset.scrub_missing_values();

    EXPECT_FALSE(dataset.is_sample_used(1));
    EXPECT_TRUE(dataset.is_sample_used(0));
    EXPECT_TRUE(dataset.is_sample_used(2));

    EXPECT_EQ(dataset.get_missing_values_number(), 1);
}


TEST(TabularDataset, BinaryFileStorageStreamsCsvToCache)
{
    const filesystem::path csv_path =
        filesystem::temp_directory_path() / "opennn_test_binary_storage.csv";
    const filesystem::path cache_path =
        csv_path.parent_path() / ".cache" / "opennn_test_binary_storage.bin";

    create_temp_csv_file(csv_path.string(),
                         "a,b,target\n"
                         "1,10,0\n"
                         "2,20,1\n"
                         "3,30,0\n"
                         "4,40,1\n");

    TabularDataset dataset;

    dataset.set_storage_mode(Dataset::StorageMode::BinaryFile);
    dataset.set_data_path(csv_path);
    dataset.set_separator(Dataset::Separator::Comma);
    dataset.set_has_header(true);
    dataset.set_has_ids(false);
    dataset.set_display(false);

    ASSERT_NO_THROW(dataset.read_csv());

    EXPECT_EQ(dataset.get_storage_mode(), Dataset::StorageMode::BinaryFile);
    EXPECT_EQ(dataset.get_samples_number(), 4);
    EXPECT_EQ(dataset.get_data().size(), 0);

    ASSERT_TRUE(filesystem::exists(cache_path));
    EXPECT_EQ(filesystem::file_size(cache_path), 4 * 3 * sizeof(float));

    EXPECT_EQ(dataset.get_variable_type(2), VariableType::Binary);

    vector<float> inputs(4);
    dataset.fill_inputs({0, 2}, {0, 1}, inputs.data(), false);

    EXPECT_NEAR(inputs[0], 1, EPSILON);
    EXPECT_NEAR(inputs[1], 10, EPSILON);
    EXPECT_NEAR(inputs[2], 3, EPSILON);
    EXPECT_NEAR(inputs[3], 30, EPSILON);

    vector<float> targets(4);
    dataset.fill_targets({0, 1, 2, 3}, {2}, targets.data(), false);

    EXPECT_NEAR(targets[0], 0, EPSILON);
    EXPECT_NEAR(targets[1], 1, EPSILON);
    EXPECT_NEAR(targets[2], 0, EPSILON);
    EXPECT_NEAR(targets[3], 1, EPSILON);

    filesystem::remove(csv_path);
    filesystem::remove(cache_path);
}
