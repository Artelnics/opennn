//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   T E S T   C L A S S                                 
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "data_set_test.h"

DataSetTest::DataSetTest() : UnitTesting() 
{
}


DataSetTest::~DataSetTest()
{
}


void DataSetTest::test_constructor()
{
   cout << "test_constructor\n";


   // Default constructor

   DataSet ds1;

   assert_true(ds1.get_variables_number() == 0, LOG);
   assert_true(ds1.get_instances_number() == 0, LOG);

   // Instances and variables number constructor

   DataSet ds2(1, 2);

   assert_true(ds2.get_instances_number() == 1, LOG);
   assert_true(ds2.get_variables_number() == 2, LOG);

   // Inputs, targets and instances numbers constructor

   DataSet ds3(1, 1, 1);

   assert_true(ds3.get_variables_number() == 2, LOG);
   assert_true(ds3.get_instances_number() == 1, LOG);

   // XML constructor

   tinyxml2::XMLDocument* document = ds3.to_XML();

   DataSet ds4(*document);

   assert_true(ds4.get_variables_number() == 2, LOG);
   assert_true(ds4.get_instances_number() == 1, LOG);

 //  delete document;

   DataSet ds5(*document);

   assert_true(ds5.get_variables_number() == 2, LOG);
   assert_true(ds5.get_instances_number() == 1, LOG);

   delete document;

   // Copy constructor

   DataSet ds6(ds1);

   assert_true(ds6.get_variables_number() == 0, LOG);
   assert_true(ds6.get_instances_number() == 0, LOG);
}


void DataSetTest::test_destructor()
{
   cout << "test_destructor\n";

   DataSet* dsp = new DataSet(1, 1, 1);

   delete dsp;
}


void DataSetTest::test_assignment_operator()
{
   cout << "test_assignment_operator\n";

   DataSet ds1(1, 1, 1);
   DataSet ds2 = ds1;

   assert_true(ds2.get_instances_number() == 1, LOG);
   assert_true(ds2.get_variables_number() == 2, LOG);
}


void DataSetTest::test_get_instances_number() 
{
   cout << "test_get_instances_number\n";

   DataSet data_set;

   assert_true(data_set.get_instances_number() == 0, LOG);
}


void DataSetTest::test_get_variables_number() 
{
   cout << "test_get_variables_number\n";

   DataSet data_set;

   assert_true(data_set.get_variables_number() == 0, LOG);
}


void DataSetTest::test_get_variables() 
{
   cout << "test_get_variables\n";

   DataSet data_set(1, 3, 2);

   assert_true(data_set.get_input_variables_number() == 3, LOG);
   assert_true(data_set.get_target_variables_number() == 2, LOG);
}


void DataSetTest::test_get_display() 
{
   cout << "test_get_display\n";

   DataSet data_set;

   data_set.set_display(true);

   assert_true(data_set.get_display() == true, LOG);

   data_set.set_display(false);

   assert_true(data_set.get_display() == false, LOG);
}


void DataSetTest::test_get_data() 
{
   cout << "test_get_data\n";

   DataSet data_set(1,1,1);

   data_set.initialize_data(0.0);

   const Matrix<double>& data = data_set.get_data();

   assert_true(data.get_rows_number() == 1, LOG);
   assert_true(data.get_columns_number() == 2, LOG);
   assert_true(data == 0.0, LOG);
}


void DataSetTest::test_get_training_data()
{
   cout << "test_get_training_data\n";

   Matrix<double> matrix({{1,4,6},{4,3,6},{7,8,9}});

   DataSet data_set;

   data_set.set_data(matrix);

   Vector<size_t> training_indices({0,1});


   data_set.set_testing();
   data_set.set_training(training_indices);

   Matrix<double> training_data = data_set.get_training_data();
   Matrix<double> solution({{1,4},{4,3},{7,8}});
   assert_true(training_data == solution, LOG);
}


void DataSetTest::test_get_selection_data()
{
   cout << "test_get_selection_data\n";

   Matrix<double> matrix({{1,4,6},{4,3,6},{7,8,9}});

   DataSet data_set;

   data_set.set_data(matrix);

   Vector<size_t> selection_indices({0,1});


   data_set.set_testing();
   data_set.set_selection(selection_indices);

   Matrix<double> selection_data = data_set.get_selection_data();
   Matrix<double> solution({{1,4},{4,3},{7,8}});
   assert_true(selection_data == solution, LOG);
}


void DataSetTest::test_get_testing_data()
{
   cout << "test_get_testing_data\n";

   Matrix<double> matrix({{1,4,6},{4,3,6},{7,8,9}});

   DataSet data_set;

   data_set.set_data(matrix);

   Vector<size_t> testing_indices({0,1});


   data_set.set_training();
   data_set.set_testing(testing_indices);

   Matrix<double> testing_data = data_set.get_testing_data();
   Matrix<double> solution({{1,4},{4,3},{7,8}});
   assert_true(testing_data == solution, LOG);
}


void DataSetTest::test_get_inputs() 
{
   cout << "test_get_inputs\n";

   DataSet data_set(1, 3, 2);

   size_t instances_number = data_set.get_instances_number();
   size_t inputs_number = data_set.get_input_variables_number();

   Matrix<double> inputs = data_set.get_input_data();

   size_t rows_number = inputs.get_rows_number();
   size_t columns_number = inputs.get_columns_number();

   assert_true(instances_number == rows_number, LOG);
   assert_true(inputs_number == columns_number, LOG);
}


void DataSetTest::test_get_targets()
{
   cout << "test_get_targets\n";

   DataSet data_set(1,3,2);

   size_t instances_number = data_set.get_instances_number();
   size_t targets_number = data_set.get_target_variables_number();

   Matrix<double> targets = data_set.get_target_data();

   size_t rows_number = targets.get_rows_number();
   size_t columns_number = targets.get_columns_number();

   assert_true(instances_number == rows_number, LOG);
   assert_true(targets_number == columns_number, LOG);
}


void DataSetTest::test_get_instance()
{
   cout << "test_get_instance\n";

   DataSet data_set;
   Vector<double> instance;

   // Test

   data_set.set(1, 1, 1);
   data_set.initialize_data(1.0);

   instance = data_set.get_instance_data(0);

   assert_true(instance.size() == 2, LOG);
   assert_true(instance == 1.0, LOG);

   // Test several variables

   data_set.set(4, 3, 1);
   data_set.initialize_data(1.0);

   Vector<size_t> indices_variables({1,3});
   Vector<double> instance_0 = data_set.get_instance_data(0, indices_variables);
   Vector<double> instance_1 = data_set.get_instance_data(1, indices_variables);

   assert_true(instance_0 == instance_1, LOG);
}


void DataSetTest::test_set() 
{
   cout << "test_set\n";

   DataSet data_set;

   Matrix<double> data;

   // Instances and inputs and target variables

   data_set.set(1, 2, 3);

   assert_true(data_set.get_instances_number() == 1, LOG);
   assert_true(data_set.get_input_variables_number() == 2, LOG);
   assert_true(data_set.get_target_variables_number() == 3, LOG);

   data = data_set.get_data();

   assert_true(data.get_rows_number() == 1, LOG);
   assert_true(data.get_columns_number() == 5, LOG);
}


void DataSetTest::test_set_instances_number() 
{
   cout << "test_set_instances_number\n";

   DataSet data_set(1,1,1);

   data_set.set_instances_number(2);

   assert_true(data_set.get_instances_number() == 2, LOG);
}


void DataSetTest::test_set_variables_number() 
{
   cout << "test_set_variables_number\n";

   DataSet data_set(1, 1);

//   data_set.set_variables_number(2);

   assert_true(data_set.get_variables_number() == 2, LOG);
}


void DataSetTest::test_set_display() 
{
   cout << "test_set_display\n";
}


void DataSetTest::test_set_data() 
{
   cout << "test_set_data\n";

   DataSet data_set(1, 1, 1);

   Matrix<double> new_data(1, 2, 0.0);

   data_set.set_data(new_data);

   Matrix<double> data = data_set.get_data();

   assert_true(data == new_data, LOG);
}


void DataSetTest::test_set_instance()
{
   cout << "test_set_instance\n";

   DataSet data_set(1, 1, 1);

   Vector<double> new_instance(2, 0.0);

   data_set.set_instance(0, new_instance);

   Vector<double> instance = data_set.get_instance_data(0);

   assert_true(instance == new_instance, LOG);
}


void DataSetTest::test_calculate_data_descriptives() 
{
   cout << "test_calculate_data_descriptives\n";

   DataSet data_set;

   Vector<Descriptives> descriptives;

   // Test

   data_set.set(1, 1);

   data_set.initialize_data(0.0);

   descriptives = data_set.calculate_columns_descriptives();

   assert_true(descriptives.size() == 1, LOG);

   assert_true(descriptives[0].minimum == 0.0, LOG);
   assert_true(descriptives[0].maximum == 0.0, LOG);
   assert_true(descriptives[0].mean == 0.0, LOG);
   assert_true(descriptives[0].standard_deviation == 0.0, LOG);

   // Test

   data_set.set(2, 2, 2);

   data_set.initialize_data(0.0);

   descriptives = data_set.calculate_columns_descriptives();

   assert_true(descriptives.size() == 4, LOG);

   assert_true(descriptives[0].minimum == 0.0, LOG);
   assert_true(descriptives[0].maximum == 0.0, LOG);
   assert_true(descriptives[0].mean == 0.0, LOG);
   assert_true(descriptives[0].standard_deviation == 0.0, LOG);

   assert_true(descriptives[1].minimum == 0.0, LOG);
   assert_true(descriptives[1].maximum == 0.0, LOG);
   assert_true(descriptives[1].mean == 0.0, LOG);
   assert_true(descriptives[1].standard_deviation == 0.0, LOG);

   assert_true(descriptives[2].minimum == 0.0, LOG);
   assert_true(descriptives[2].maximum == 0.0, LOG);
   assert_true(descriptives[2].mean == 0.0, LOG);
   assert_true(descriptives[2].standard_deviation == 0.0, LOG);

   assert_true(descriptives[3].minimum == 0.0, LOG);
   assert_true(descriptives[3].maximum == 0.0, LOG);
   assert_true(descriptives[3].mean == 0.0, LOG);
   assert_true(descriptives[3].standard_deviation == 0.0, LOG);

}


void DataSetTest::test_calculate_data_descriptives_missing_values()
{
    cout << "test_calculate_data_descriptives_missing_values\n";

    const string data_file_name = "../data/data.dat";

    ofstream file;

    DataSet data_set;

    data_set.set_data_file_name(data_file_name);

    Matrix<double> data;

    string data_string;

    data_set.set_separator(' ');
    data_set.set_missing_values_label("?");

    data_string = "-1000 ? 0 \n 3 4 ? \n ? 4 1";

    file.open(data_file_name.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    data = data_set.get_data();

    assert_true(abs(data_set.calculate_columns_descriptives_matrix()(0, 0) - (-1000)) < 1.0e-4, LOG);
    assert_true(abs(data_set.calculate_columns_descriptives_matrix()(1, 0) - 4.0) < 1.0e-4, LOG);
    assert_true(abs(data_set.calculate_columns_descriptives_matrix()(2, 0) - 0.0) < 1.0e-4, LOG);
}


void DataSetTest::test_calculate_training_instances_descriptives()
{
   cout << "test_calculate_training_instances_descriptives\n";

   DataSet data_set;
   Vector<Descriptives> training_instances_descriptives;

   // Test

   data_set.set(2, 2, 2);

   data_set.set_training();

   data_set.initialize_data(0.0);

   data_set.calculate_columns_descriptives_training_instances();
}


void DataSetTest::test_calculate_selection_instances_descriptives()
{
   cout << "test_calculate_selection_instances_descriptives\n";

   DataSet data_set;
   Vector<Descriptives> selection_instances_descriptives;

   // Test

   data_set.set(2,2,2);

   data_set.set_selection();

   data_set.initialize_data(0.0);

   selection_instances_descriptives = data_set.calculate_columns_descriptives_selection_instances();
}


void DataSetTest::test_calculate_testing_instances_descriptives()
{
   cout << "test_calculate_testing_instances_descriptives\n";

   DataSet data_set;
   Vector<Descriptives> testing_instances_descriptives;

   // Test

   data_set.set(2, 2, 2);


   data_set.set_testing();
   
   data_set.initialize_data(0.0);

   testing_instances_descriptives = data_set.calculate_columns_descriptives_testing_instances();
}


void DataSetTest::test_calculate_inputs_descriptives()
{
   cout << "test_calculate_inputs_descriptives\n";

   Matrix<double> matrix({{1.0,2.0,3.0},{1.0,2.0,3.0}});

   DataSet data_set;
   data_set.set_data(matrix);
   Vector<size_t> indices({0, 1});

   Descriptives descriptives;
   descriptives = data_set.calculate_inputs_descriptives(indices[0]);
   Descriptives descriptives_1;
   descriptives_1 = data_set.calculate_inputs_descriptives(indices[1]);

   assert_true(descriptives.mean == 2.0, LOG);
   assert_true(descriptives.standard_deviation == 1.0, LOG);
   assert_true(descriptives.minimum == 1.0, LOG);
   assert_true(descriptives.maximum == 3.0, LOG);

   assert_true(descriptives_1.mean == 2.0, LOG);
   assert_true(descriptives_1.standard_deviation == 1.0, LOG);
   assert_true(descriptives_1.minimum == 1.0, LOG);
   assert_true(descriptives_1.maximum == 3.0, LOG);
}


void DataSetTest::test_calculate_autocorrelations()
{
    cout << "test_calculate_autocorrelations\n";

    DataSet data_set;

    Matrix<double> autocorrelations;

    data_set.set(20, 1, 1);

    data_set.randomize_data_normal();

    autocorrelations = data_set.calculate_autocorrelations();

    assert_true(autocorrelations.get_columns_number() == 10, LOG);
    assert_true(autocorrelations.get_rows_number() == 2, LOG);
}


void DataSetTest::test_calculate_cross_correlations()
{
    cout << "test_calculate_cross_correlations";

    DataSet data_set;

    Matrix<Vector<double>> cross_correlations;

    data_set.set(20, 5, 1);

    data_set.randomize_data_normal();

    cross_correlations = data_set.calculate_cross_correlations();

    assert_true(cross_correlations.get_columns_number() == 6, LOG);
    assert_true(cross_correlations.get_rows_number() == 6, LOG);
}


void DataSetTest::test_calculate_data_histograms()
{
   cout << "test_calculate_data_histograms\n";

   Matrix<double> matrix(3,3);
   matrix(0,0) = 2.0;
   matrix(0,1) = 2.0;
   matrix(0,2) = 1.0;
   matrix(1,0) = 1.0;
   matrix(1,1) = 1.0;
   matrix(1,2) = 1.0;
   matrix(2,0) = 1.0;
   matrix(2,1) = 2.0;
   matrix(2,2) = 2.0;

   DataSet data_set(matrix);
   Vector<Histogram> histograms;
   histograms = data_set.calculate_columns_histograms();
   Vector<size_t> sol({1, 1});
   Vector<size_t> sol_2({2, 2});
   Vector<double> centers({0,1});

   //Test frequencies
   assert_true(histograms[0].frequencies == sol, LOG);
   assert_true(histograms[1].frequencies == sol_2, LOG);
   assert_true(histograms[2].frequencies == sol, LOG);

   //Test centers
   assert_true(histograms[0].centers == centers, LOG);
   assert_true(histograms[1].centers == centers, LOG);
   assert_true(histograms[2].centers == centers, LOG);
}


void DataSetTest::test_filter_data()
{
   cout << "test_filter_data\n";

   DataSet data_set;

   Vector<double> minimums;
   Vector<double> maximums;

   Matrix<double> data;

   // Test

   data_set.set(2, 1, 1);
   data_set.initialize_data(1.0);

   minimums.set(2, 0.0);
   maximums.set(2, 0.5);

   data_set.filter_data(minimums, maximums);

   data = data_set.get_data();

   assert_true(data_set.get_instance_use(0) == DataSet::UnusedInstance, LOG);
   assert_true(data_set.get_instance_use(1) == DataSet::UnusedInstance, LOG);
}


void DataSetTest::test_scale_inputs_mean_standard_deviation() 
{
   cout << "test_scale_inputs_mean_standard_deviation\n";

   DataSet data_set;

   Vector<Descriptives> inputs_descriptives;

   // Test

   data_set.set(2, 2, 2);
   data_set.randomize_data_normal();

   data_set.scale_inputs_mean_standard_deviation();

   inputs_descriptives = data_set.calculate_input_variables_descriptives();

   assert_true(inputs_descriptives[0].has_mean_zero_standard_deviation_one(), LOG);
}


void DataSetTest::test_scale_targets_mean_standard_deviation() 
{
   cout << "test_scale_targets_mean_standard_deviation\n";

   DataSet data_set;

   Vector<Descriptives> targets_descriptives;

   // Test

   data_set.set(2, 2, 2);
   data_set.randomize_data_normal();

   data_set.scale_targets_mean_standard_deviation();

   targets_descriptives = data_set.calculate_target_variables_descriptives();

   assert_true(abs(targets_descriptives[0].has_mean_zero_standard_deviation_one() - 1.0) < 1.0e-3, LOG);
}


void DataSetTest::test_scale_inputs_minimum_maximum() 
{
   cout << "test_scale_inputs_minimum_maximum\n";

   DataSet data_set;

   Vector<Descriptives> inputs_descriptives;

   // Test

   data_set.set(2, 2, 2);
   data_set.randomize_data_normal();

   data_set.scale_inputs_minimum_maximum();

   inputs_descriptives = data_set.calculate_input_variables_descriptives();

   assert_true(inputs_descriptives[0].has_minimum_minus_one_maximum_one(), LOG);
}


void DataSetTest::test_scale_targets_minimum_maximum() 
{
   cout << "test_scale_targets_minimum_maximum\n";

   DataSet data_set;

   Vector<Descriptives> targets_descriptives;

   // Test

   data_set.set(2, 2, 2);
   data_set.randomize_data_normal();

   data_set.scale_targets_minimum_maximum();

   targets_descriptives = data_set.calculate_target_variables_descriptives();

     //assert_true(targets_descriptives[0].has_minimum_minus_one_maximum_one(), LOG);
}


void DataSetTest::test_scale_data_minimum_maximum()
{
   cout << "test_scale_data_minimum_maximum\n";

   DataSet data_set;

   Vector<Descriptives> data_descriptives;

   Matrix<double> data;
   Matrix<double> scaled_data;

    // Test

   data_set.set(2,2,2);
   data_set.initialize_data(0.0);

   data_set.set_display(false);

   data = data_set.get_data();

   data_descriptives = data_set.scale_data_minimum_maximum();

   scaled_data = data_set.get_data();

   assert_true(scaled_data == data, LOG);
}


void DataSetTest::test_scale_data_mean_standard_deviation()
{
   cout << "test_scale_data_mean_standard_deviation\n";

   DataSet data_set;

   Vector<Descriptives> data_descriptives;

   Matrix<double> data;
   Matrix<double> scaled_data;

    // Test

   data_set.set(2,2,2);
   data_set.initialize_data(0.0);

   data_set.set_display(false);

   data = data_set.get_data();

   data_descriptives = data_set.scale_data_mean_standard_deviation();

   scaled_data = data_set.get_data();

   assert_true(scaled_data == data, LOG);
}


void DataSetTest::test_unscale_data_mean_standard_deviation()
{
   cout << "test_unscale_data_mean_standard_deviation\n";

   Matrix<double> matrix (3,1);
   matrix(0,0) = 2.0;
   matrix(1,0) = 5.0;
   matrix(2,0) = 77.0;

   DataSet data;
   data.set_data(matrix);

   Vector<Descriptives> descrriptives(1);
   Descriptives descriptives;
   descriptives.set_minimum(5.0);
   descriptives.set_maximum(9.0);
   descriptives.set_mean(8.0);
   descriptives.set_standard_deviation(2.0);
   descrriptives[0] = descriptives ;

   Matrix<double> unescale_matrix(3,1);
   DataSet data_unscaled;
   data.set_data(unescale_matrix);

   data.unscale_data_mean_standard_deviation(descrriptives);

   Matrix<double> matrix_solution (3,1);
   matrix_solution(0,0) = descriptives.mean;
   matrix_solution(1,0) = descriptives.mean;
   matrix_solution(2,0) = descriptives.mean;

   assert_true(data.get_data() == matrix_solution, LOG);
}


void DataSetTest::test_unscale_data_minimum_maximum()
{
   cout << "test_unscale_data_minimum_maximum\n";

   Matrix<double> matrix (3,1);
   matrix(0,0) = 2.0;
   matrix(1,0) = 5.0;
   matrix(2,0) = 77.0;

   DataSet data;
   data.set_data(matrix);

   Vector<Descriptives> descrriptives(1);
   Descriptives descriptives;
   descriptives.set_minimum(5.0);
   descriptives.set_maximum(9.0);
   descriptives.set_mean(8.0);
   descriptives.set_standard_deviation(2.0);
   descrriptives[0] = descriptives ;

   Matrix<double> unescale_matrix(3,1);
   DataSet data_unscaled;
   data.set_data(unescale_matrix);

   data.unscale_data_minimum_maximum(descrriptives);

   Matrix<double> matrix_solution (3,1);
   matrix_solution(0,0) = 7.0;
   matrix_solution(1,0) = 7.0;
   matrix_solution(2,0) = 7.0;

   assert_true(data.get_data() == matrix_solution, LOG);
}


void DataSetTest::test_unscale_inputs_mean_standard_deviation() 
{
   cout << "test_unscale_inputs_mean_standard_deviation\n";

   DataSet data_set(2, 2, 2);
   data_set.initialize_data(0.0);

   data_set.set_display(false);

   Vector<Descriptives> data_descriptives;

   // Test

   Matrix<double> inputs = data_set.get_input_data();

   data_descriptives.set(4);

   data_set.unscale_inputs_mean_standard_deviation(data_descriptives);

   Matrix<double> new_inputs = data_set.get_input_data();

   assert_true(new_inputs == inputs, LOG);
}


void DataSetTest::test_unscale_targets_mean_standard_deviation() 
{
   cout << "test_unscale_targets_mean_standard_deviation\n";
   
   DataSet data_set(2, 2, 2);
   data_set.initialize_data(0.0);

   data_set.set_display(false);

   Matrix<double> targets = data_set.get_target_data();

   Vector<Descriptives> data_descriptives(4);

   data_set.unscale_targets_mean_standard_deviation(data_descriptives);

   Matrix<double> new_targets = data_set.get_target_data();

   assert_true(new_targets == targets, LOG);
}


void DataSetTest::test_unscale_inputs_minimum_maximum() 
{
   cout << "test_unscale_inputs_minimum_maximum\n"; 

   DataSet data_set(2, 2, 2);
   data_set.initialize_data(0.0);

   data_set.set_display(false);

   Vector<Descriptives> data_descriptives;

   // Test

   Matrix<double> inputs = data_set.get_input_data();

   data_descriptives.set(4);

   data_set.unscale_inputs_minimum_maximum(data_descriptives);

   Matrix<double> new_inputs = data_set.get_input_data();

   assert_true(new_inputs == inputs, LOG);
}


void DataSetTest::test_unscale_targets_minimum_maximum() 
{
   cout << "test_unscale_targets_minimum_maximum\n";

   DataSet data_set(2, 2, 2);
   data_set.initialize_data(0.0);

   data_set.set_display(false);

   Matrix<double> targets = data_set.get_target_data();

   Vector<Descriptives> data_descriptives(4);

   data_set.unscale_targets_minimum_maximum(data_descriptives);

   Matrix<double> new_targets = data_set.get_target_data();

   assert_true(new_targets == targets, LOG);
}


void DataSetTest::test_unuse_constant_columns()
{
   cout << "test_unuse_constant_columns\n";

   DataSet data_set;

   // Test 

   data_set.set(1, 2, 1);

   data_set.initialize_data(0.0);

   data_set.unuse_constant_columns();

   assert_true(data_set.get_input_columns_number() == 0, LOG);
   assert_true(data_set.get_target_columns_number() == 1, LOG);
}


void DataSetTest::test_initialize_data()
{
   cout << "test_initialize_data\n";

   Matrix<double> matrix(3,3);

   DataSet data_set;

   data_set.set_data(matrix);

   data_set.initialize_data(2);

   Matrix<double> solution({{2,2,2},{2,2,2},{2,2,2}});

   assert_true(data_set.get_data() == solution, LOG);
}


/// @todo Multiple classes.

void DataSetTest::test_calculate_target_columns_distribution()
{
    cout << "test_calculate_target_columns_distribution\n";
/*
    //Test two classes

    Matrix<double> matrix({{2,5,6,9,8},{2,9,1,9,4},{6,5,6,7,3},{0,static_cast<double>(NAN),1,0,1}});

    Vector<size_t> target_indices({3});

    Vector<size_t> inputs_indices({0, 1, 2});

    DataSet data_set;

    data_set.set_data(matrix);

    Vector<size_t> target_distribution = data_set.calculate_target_distribution();

    Vector<size_t> solution;
    solution.set(2);
    solution[0] = 2;
    solution[1] = 2;

    assert_true(target_distribution == solution, LOG);

    // Test more two classes

    Matrix<double> matrix_1({{2,5,6,9,8,7},{2,9,1,9,4,5},{6,5,6,7,3,2},{0,static_cast<double>(NAN),1,0,2,2},{static_cast<double>(NAN),static_cast<double>(NAN),1,0,0,2}});

    Vector<size_t> target_indices_1({2,3});

    Vector<size_t> inputs_indices_1({0, 1});

    DataSet ds_1;

    ds_1.set_data(matrix_1);

    Vector<size_t> calculate_target_distribution_1 = ds_1.calculate_target_distribution();

    assert_true(calculate_target_distribution_1[0] == 6, LOG);
    assert_true(calculate_target_distribution_1[1] == 3, LOG);
    assert_true(calculate_target_distribution_1[2] == 2, LOG);
*/
}


void DataSetTest::test_unuse_most_populated_target()
{
    cout << "test_unused_most_populated_target\n";

    DataSet data_set;

    Vector<size_t> unused_instances_indices;

    // Test

    data_set.set(5,2,5);
    data_set.initialize_data(0.0);

    unused_instances_indices = data_set.unuse_most_populated_target(7);

    assert_true(unused_instances_indices.size() == 5, LOG);
    assert_true(data_set.get_used_instances_number() == 0, LOG);
    assert_true(data_set.get_unused_instances_number() == 5, LOG);

    // Test

    DataSet ds2;

    ds2.set(100, 7,5);
    ds2.initialize_data(1.0);

    unused_instances_indices = ds2.unuse_most_populated_target(99);

    assert_true(unused_instances_indices.size() == 99, LOG);
    assert_true(ds2.get_used_instances_number() == 1, LOG);
    assert_true(ds2.get_unused_instances_number() == 99, LOG);

    // Test

    DataSet ds3;

    ds3.set(1, 10,10);
    ds3.randomize_data_normal();

    unused_instances_indices = ds3.unuse_most_populated_target(50);

    assert_true(unused_instances_indices.size() == 1, LOG);
    assert_true(ds3.get_used_instances_number() == 0, LOG);
    assert_true(ds3.get_unused_instances_number() == 1, LOG);
}


void DataSetTest::test_balance_binary_targets_distribution()
{
    cout << "test_balance_binary_target_distribution\n";

    // Test

    DataSet data_set(22, 2, 1);
    data_set.initialize_data(1.0);
    Vector<double> instance0(3, 0.0);
    Vector<double> instance1(3, 0.0);
    Vector<double> instance2(3);
    Vector<double> instance3(3);
    Vector<double> instance4(3);

    instance2[0] = 4.0;
    instance2[1] = 5.0;
    instance2[2] = 0.0;

    instance3[0] = 3.9;
    instance3[1] = 5.0;
    instance3[2] = 0.0;

    instance4[0] = 0.2;
    instance4[1] = 9.0;
    instance4[2] = 0.0;

    data_set.set_instance(0, instance0);
    data_set.set_instance(1, instance1);
    data_set.set_instance(2, instance2);
    data_set.set_instance(3, instance3);
    data_set.set_instance(4, instance4);

    data_set.balance_binary_targets_distribution();

    assert_true(data_set.get_unused_instances_number() == 12, LOG);
    //assert_true(data_set.calculate_target_distribution()[1] == 5, LOG);
    //assert_true(data_set.calculate_target_distribution()[0] == 5, LOG);

    // Test

    {
    DataSet data_set(10, 3, 1);

    Vector<double> instance0(4);
    Vector<double> instance1(4);
    Vector<double> instance2(4);
    Vector<double> instance3(4);
    Vector<double> instance4(4);
    Vector<double> instance5(4);
    Vector<double> instance6(4);
    Vector<double> instance7(4);
    Vector<double> instance8(4);
    Vector<double> instance9(4);

    instance0[0] = 0.9;
    instance0[1] = 5.0;
    instance0[2] = 0.0;
    instance0[3] = 0.0;

    instance1[0] = 1.1;
    instance1[1] = 2.3;
    instance1[2] = 0.0;
    instance1[3] = 0.0;

    instance2[0] = 2.3;
    instance2[1] = 3.0;
    instance2[2] = 1.0;
    instance2[3] = 0.0;

    instance3[0] = 5.6;
    instance3[1] = 3.4;
    instance3[2] = 1.0;
    instance3[3] = 0.0;

    instance4[0] = 0.8;
    instance4[1] = 3.1;
    instance4[2] = 0.0;
    instance4[3] = 0.0;

    instance5[0] = 3.4;
    instance5[1] = 3.9;
    instance5[2] = 0.0;
    instance5[3] = 0.0;

    instance6[0] = 5.6;
    instance6[1] = 8.0;
    instance6[2] = 1.0;
    instance6[3] = 0.0;

    instance7[0] = 3.9;
    instance7[1] = 9.0;
    instance7[2] = 0.0;
    instance7[3] = 0.0;

    instance8[0] = 1.9;
    instance8[1] = 2.3;
    instance8[2] = 0.0;
    instance8[3] = 0.0;

    instance9[0] = 7.8;
    instance9[1] = 2.8;
    instance9[2] = 0.0;
    instance9[3] = 0.0;

    data_set.set_instance(0, instance0);
    data_set.set_instance(1, instance1);
    data_set.set_instance(2, instance2);
    data_set.set_instance(3, instance3);
    data_set.set_instance(4, instance4);
    data_set.set_instance(5, instance5);
    data_set.set_instance(6, instance6);
    data_set.set_instance(7, instance7);
    data_set.set_instance(8, instance8);
    data_set.set_instance(9, instance9);

    data_set.balance_binary_targets_distribution();

    assert_true(data_set.get_unused_instances_number() == 10, LOG);
    assert_true(data_set.calculate_target_distribution()[1] == 0, LOG);
    assert_true(data_set.calculate_target_distribution()[0] == 0, LOG);
    }

    // Test

    {
    DataSet data_set(10, 3, 1);

    Vector<double> instance0(4);
    Vector<double> instance1(4);
    Vector<double> instance2(4);
    Vector<double> instance3(4);
    Vector<double> instance4(4);
    Vector<double> instance5(4);
    Vector<double> instance6(4);
    Vector<double> instance7(4);
    Vector<double> instance8(4);
    Vector<double> instance9(4);

    instance0[0] = 0.9;
    instance0[1] = 5.0;
    instance0[2] = 0.0;
    instance0[3] = 0.0;

    instance1[0] = 1.1;
    instance1[1] = 2.3;
    instance1[2] = 0.0;
    instance1[3] = 0.0;

    instance2[0] = 2.3;
    instance2[1] = 3.0;
    instance2[2] = 1.0;
    instance2[3] = 0.0;

    instance3[0] = 5.6;
    instance3[1] = 3.4;
    instance3[2] = 1.0;
    instance3[3] = 1.0;

    instance4[0] = 0.8;
    instance4[1] = 3.1;
    instance4[2] = 0.0;
    instance4[3] = 0.0;

    instance5[0] = 3.4;
    instance5[1] = 3.9;
    instance5[2] = 0.0;
    instance5[3] = 0.0;

    instance6[0] = 5.6;
    instance6[1] = 8.0;
    instance6[2] = 1.0;
    instance6[3] = 0.0;

    instance7[0] = 3.9;
    instance7[1] = 9.0;
    instance7[2] = 0.0;
    instance7[3] = 1.0;

    instance8[0] = 1.9;
    instance8[1] = 2.3;
    instance8[2] = 0.0;
    instance8[3] = 0.0;

    instance9[0] = 7.8;
    instance9[1] = 2.8;
    instance9[2] = 0.0;
    instance9[3] = 0.0;

    data_set.set_instance(0, instance0);
    data_set.set_instance(1, instance1);
    data_set.set_instance(2, instance2);
    data_set.set_instance(3, instance3);
    data_set.set_instance(4, instance4);
    data_set.set_instance(5, instance5);
    data_set.set_instance(6, instance6);
    data_set.set_instance(7, instance7);
    data_set.set_instance(8, instance8);
    data_set.set_instance(9, instance9);

    data_set.balance_binary_targets_distribution(50.0);

    assert_true(data_set.get_unused_instances_number() == 3, LOG);
    assert_true(data_set.calculate_target_distribution()[1] == 2, LOG);
    assert_true(data_set.calculate_target_distribution()[0] == 5, LOG);
    }

    // Test
    {
    DataSet ds2(9, 1, 99);

    Matrix<double> data1(9,10);
    Matrix<double> data2(90,10);

    data1.randomize_normal();
    data2.randomize_normal();

    data1.set_column(9, 0.0);
    data2.set_column(9, 1.0);

    Matrix<double> data = data1.assemble_rows(data2);

    ds2.set(data);

    ds2.balance_binary_targets_distribution();

    assert_true(ds2.calculate_target_distribution()[0] == ds2.calculate_target_distribution()[1], LOG);
    assert_true(ds2.calculate_target_distribution()[0] == 9, LOG);
    }

    //Test
    {
    DataSet data_set(4,1,1);

    const string data_file_name = "../data/data.dat";

    data_set.set_data_file_name(data_file_name);
    data_set.set_has_columns_names(false);
    data_set.set_separator(',');
    data_set.set_missing_values_label("NaN");

    const string data_string =
    "5.1,3.5,1.0\n"
    "7.0,3.2,NaN\n"
    "7.0,3.2,0.0\n"
    "6.3,3.3,0.0";

    ofstream file;

    file.open(data_file_name.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    data_set.impute_missing_values_unuse();

    data_set.balance_binary_targets_distribution();

    Vector<size_t> target_distribution = data_set.calculate_target_distribution();

    assert_true(target_distribution[0] == target_distribution[1], LOG);
    assert_true(data_set.get_used_columns_indices().size() == 2, LOG);
    assert_true(data_set.get_unused_instances_indices().size() == 2, LOG);
    }

    //Test
    {
    DataSet data_set(16,1,1);

    const string data_file_name = "../data/data.dat";

    data_set.set_data_file_name(data_file_name);
    data_set.set_has_columns_names(false);
    data_set.set_separator(',');
    data_set.set_missing_values_label("NaN");

    const string data_string =
    "5.1,3.5,1.0\n"
    "7.0,3.2,1.0\n"
    "7.0,3.2,0.0\n"
    "6.3,3.3,0.0\n"
    "5.1,3.5,1.0\n"
    "7.0,3.2,0.0\n"
    "7.0,3.2,NaN\n"
    "6.3,3.3,NaN\n"
    "5.1,3.5,NaN\n"
    "7.0,3.2,NaN\n"
    "7.0,3.2,NaN\n"
    "6.3,3.3,NaN\n"
    "5.1,3.5,NaN\n"
    "7.0,3.2,NaN\n"
    "7.0,3.2,NaN\n"
    "6.3,3.3,NaN\n"
    "5.1,3.5,1.0\n"
    "7.0,3.2,NaN\n"
    "7.0,3.2,NaN\n"
    "6.3,3.3,NaN\n";

    ofstream file;

    file.open(data_file_name.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    data_set.impute_missing_values_unuse();

    data_set.balance_binary_targets_distribution();

    Vector<size_t> target_distribution = data_set.calculate_target_distribution();

    assert_true(target_distribution[0] == target_distribution[1], LOG);
    }

}


void DataSetTest::test_balance_multiple_targets_distribution()
{
    cout << "test_balance_multiple_target_distribution\n";

    DataSet data_set(9, 2, 2);

    Vector<double> instance0(4);
    Vector<double> instance1(4);
    Vector<double> instance2(4);
    Vector<double> instance3(4);
    Vector<double> instance4(4);
    Vector<double> instance5(4);
    Vector<double> instance6(4);
    Vector<double> instance7(4);
    Vector<double> instance8(4);
    Vector<double> instance9(4);

    instance0[0] = 0.9;
    instance0[1] = 5.0;
    instance0[2] = 0.0;
    instance0[3] = 1.0;

    instance1[0] = 1.1;
    instance1[1] = 2.3;
    instance1[2] = 0.0;
    instance1[3] = 1.0;

    instance2[0] = 2.3;
    instance2[1] = 3.0;
    instance2[2] = 0.0;
    instance2[3] = 1.0;

    instance3[0] = 5.6;
    instance3[1] = 3.4;
    instance3[2] = 0.0;
    instance3[3] = 1.0;

    instance4[0] = 0.8;
    instance4[1] = 3.1;
    instance4[2] = 0.0;
    instance4[3] = 1.0;

    instance5[0] = 3.4;
    instance5[1] = 3.9;
    instance5[2] = 0.0;
    instance5[3] = 1.0;

    instance6[0] = 5.6;
    instance6[1] = 8.0;
    instance6[2] = 0.0;
    instance6[3] = 1.0;

    instance7[0] = 3.9;
    instance7[1] = 9.0;
    instance7[2] = 0.0;
    instance7[3] = 1.0;

    instance8[0] = 1.9;
    instance8[1] = 2.3;
    instance8[2] = 0.0;
    instance8[3] = 1.0;

    instance9[0] = 7.8;
    instance9[1] = 2.8;
    instance9[2] = 0.0;
    instance9[3] = 1.0;

    data_set.set_instance(0, instance0);
    data_set.set_instance(1, instance1);
    data_set.set_instance(2, instance2);
    data_set.set_instance(3, instance3);
    data_set.set_instance(4, instance4);
    data_set.set_instance(5, instance5);
    data_set.set_instance(6, instance6);
    data_set.set_instance(7, instance7);
    data_set.set_instance(8, instance8);

    data_set.balance_multiple_targets_distribution();

    assert_true(data_set.get_unused_instances_number() == 9, LOG);
    assert_true(data_set.calculate_target_distribution()[0] == 0, LOG);
    assert_true(data_set.calculate_target_distribution()[1] == 0, LOG);

}


void DataSetTest::test_balance_function_regression_targets_distribution()
{
    cout << "test_balance_function_regression_targets_distribution.\n";

    DataSet data_set;

    Vector<size_t> unused_instances;

    // Test

    data_set.set(10, 3, 1);

    Vector<double> instance0(4);
    Vector<double> instance1(4);
    Vector<double> instance2(4);
    Vector<double> instance3(4);
    Vector<double> instance4(4);
    Vector<double> instance5(4);
    Vector<double> instance6(4);
    Vector<double> instance7(4);
    Vector<double> instance8(4);
    Vector<double> instance9(4);

    instance0[0] = 0.9;
    instance0[1] = 5.0;
    instance0[2] = 6.0;
    instance0[3] = 8.0;

    instance1[0] = 1.1;
    instance1[1] = 2.3;
    instance1[2] = 7.2;
    instance1[3] = 0.52;

    instance2[0] = 2.3;
    instance2[1] = 3.0;
    instance2[2] = 1.0;
    instance2[3] = 1.4;

    instance3[0] = 5.6;
    instance3[1] = 3.4;
    instance3[2] = 4.8;
    instance3[3] = 1.9;

    instance4[0] = 0.8;
    instance4[1] = 3.1;
    instance4[2] = 3.2;
    instance4[3] = 2.7;

    instance5[0] = 3.4;
    instance5[1] = 3.9;
    instance5[2] = 7.8;
    instance5[3] = 3.5;

    instance6[0] = 5.6;
    instance6[1] = 8.0;
    instance6[2] = 4.2;
    instance6[3] = 5.6;

    instance7[0] = 3.9;
    instance7[1] = 9.0;
    instance7[2] = 0.8;
    instance7[3] = 3.1;

    instance8[0] = 1.9;
    instance8[1] = 2.3;
    instance8[2] = 7.0;
    instance8[3] = 7.9;

    instance9[0] = 7.8;
    instance9[1] = 2.8;
    instance9[2] = 0.1;
    instance9[3] = 5.9;

    data_set.set_instance(0, instance0);
    data_set.set_instance(1, instance1);
    data_set.set_instance(2, instance2);
    data_set.set_instance(3, instance3);
    data_set.set_instance(4, instance4);
    data_set.set_instance(5, instance5);
    data_set.set_instance(6, instance6);
    data_set.set_instance(7, instance7);
    data_set.set_instance(8, instance8);
    data_set.set_instance(9, instance9);

    unused_instances = data_set.balance_approximation_targets_distribution(10.0);

    assert_true(data_set.get_unused_instances_number() == 1, LOG);
    assert_true(data_set.get_used_instances_number() == 9, LOG);
    assert_true(unused_instances.size() == 1, LOG);

    // Test

    DataSet ds2;
    ds2.set(1000, 5, 10);
    ds2.randomize_data_normal();

    unused_instances = ds2.balance_approximation_targets_distribution(100.0);

    assert_true(ds2.get_used_instances_number() == 0, LOG);
    assert_true(ds2.get_unused_instances_number() == 1000, LOG);
    assert_true(unused_instances.size() == 1000, LOG);
}


void DataSetTest::test_clean_Tukey_outliers()
{
    cout << "test_clean_Tukey_outliers\n";

    DataSet data_set(100, 5, 1);
    data_set.randomize_data_uniform(1.0, 2.0);

    Vector<double> instance(6);

    instance[0] = 1.0;
    instance[1] = 1.9;
    instance[2] = 10.0;
    instance[3] = 1.1;
    instance[4] = 1.8;

    data_set.set_instance(9, instance);

    const Vector<Vector<size_t>> outliers_indices = data_set.calculate_Tukey_outliers(1.5);

    const Vector<size_t> outliers_instances = outliers_indices[0].get_indices_greater_than(0);
    size_t outliers_number = outliers_instances.size();

    data_set.set_instances_unused(outliers_instances);

    assert_true(data_set.get_unused_instances_number() == 1, LOG);
    assert_true(outliers_number == 1, LOG);
    assert_true(outliers_instances[0] == 9, LOG);
}




void DataSetTest::test_generate_data_binary_classification()
{
    cout << "test_generate_data_binary_classification\n";

    DataSet data_set;

    Vector<size_t> target_distribution;

    // Test

    data_set.generate_data_binary_classification(2, 1);

    target_distribution = data_set.calculate_target_distribution();

    assert_true(target_distribution.size() == 2, LOG);
    assert_true(target_distribution[0] == 1, LOG);
    assert_true(target_distribution[1] == 1, LOG);
}


void DataSetTest::test_generate_data_multiple_classification()
{
    cout << "test_generate_data_multiple_classification\n";

}


void DataSetTest::test_to_XML() 
{
   cout << "test_to_XML\n";

   DataSet data_set;

   tinyxml2::XMLDocument* document;

   // Test

   document = data_set.to_XML();

   assert_true(document != nullptr, LOG);
}


/// @todo

void DataSetTest::test_from_XML() 
{
   cout << "test_from_XML\n";
/*
   DataSet data_set;

   tinyxml2::XMLDocument* document;
   
   // Test

   document = data_set.to_XML();

   data_set.from_XML(*document);

   // Test

   data_set.set(2, 2);

   data_set.set_variable_use(0, DataSet::Target);
   data_set.set_variable_use(1, DataSet::UnusedVariable);

   data_set.set_instance_use(0, DataSet::UnusedInstance);
   data_set.set_instance_use(1, DataSet::Testing);

   document = data_set.to_XML();

   data_set.set();

   data_set.from_XML(*document);

   assert_true(data_set.get_variables_number() == 2, LOG);
   assert_true(data_set.get_variable_use(0) == DataSet::Target, LOG);
   assert_true(data_set.get_variable_use(1) == DataSet::UnusedVariable, LOG);
   assert_true(data_set.get_instances_number() == 2, LOG);
   assert_true(data_set.get_instance_use(0) == DataSet::UnusedInstance, LOG);
   assert_true(data_set.get_instance_use(1) == DataSet::Testing, LOG);
*/
}


void DataSetTest::test_read_csv() 
{
   cout << "test_read_csv\n";

   DataSet data_set;
//   data_set.set_data_file_name("../../datasets/empty.csv");
//   data_set.set_data_file_name("../../datasets/iris.data");
//   data_set.set_data_file_name("../../datasets/airline_passengers.csv");
//   data_set.set_data_file_name("../../datasets/pollution.csv");
   data_set.set_data_file_name("../../datasets/heart.csv");

   data_set.set_separator(DataSet::Comma);
   data_set.set_has_columns_names(true);
   data_set.read_csv();

//   data_set.print_data();

//   data_set.print_summary();

   const string data_file_name = "../data/data.dat";

   ofstream file;

   data_set.set_data_file_name(data_file_name);

   Matrix<double> data;

   string data_string;

   // Test

   data_set.set();
   data_set.set_data_file_name(data_file_name);

   data_set.set_display(false);

   data_set.save_data();

   data_set.read_csv();

   data = data_set.get_data();

   assert_true(data.empty(), LOG);

   // Test

   data_set.set(2, 2, 2);
   data_set.set_data_file_name(data_file_name);

   data_set.initialize_data(0.0);

   data_set.set_display(false);

   data_set.save_data();

   data_set.read_csv();

   data = data_set.get_data();

   assert_true(data == 0.0, LOG);

   // Test

   data_set.set_separator(' ');

   data_string = "\n\t\n   1 \t 2   \n\n\n   3 \t 4   \n\t\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   data = data_set.get_data();

   assert_true(data.get_rows_number() == 2, LOG);
   assert_true(data.get_columns_number() == 2, LOG);

   assert_true(abs(data(0,0) - 1) < 1.0e-4, LOG);
   assert_true(abs(data(0,1) - 2) < 1.0e-4, LOG);
   assert_true(abs(data(1,0) - 3) < 1.0e-4, LOG);
   assert_true(abs(data(1,1) - 4) < 1.0e-4, LOG);

   assert_true(data_set.get_instances_number() == 2, LOG);
   assert_true(data_set.get_variables_number() == 2, LOG);

   // Test

   data_set.set_separator('\t');

   data_string = "\n\n\n1 \t 2\n3 \t 4\n\n\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   data = data_set.get_data();

   assert_true((data(0,0) - 1.0) < 1.0e-4, LOG);
   assert_true((data(0,1) - 2.0) < 1.0e-4, LOG);
   assert_true((data(1,0) - 3.0) < 1.0e-4, LOG);
   assert_true((data(1,1) - 4.0) < 1.0e-4, LOG);

   // Test

   data_set.set_has_columns_names(true);
   data_set.set_separator(' ');

   data_string = "\n"
                 "x y\n"
                 "\n"
                 "1   2\n"
                 "3   4\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   data = data_set.get_data();

   assert_true(data_set.get_header_line() == true, LOG);
   assert_true(data_set.get_variable_name(0) == "x", LOG);
   assert_true(data_set.get_variable_name(1) == "y", LOG);

   assert_true(data.get_rows_number() == 2, LOG);
   assert_true(data.get_columns_number() == 2, LOG);

   assert_true((data(0,0) - 1.0) < 1.0e-4, LOG);
   assert_true((data(0,1) - 2.0) < 1.0e-4, LOG);
   assert_true((data(1,0) - 3.0) < 1.0e-4, LOG);
   assert_true((data(1,1) - 4.0) < 1.0e-4, LOG);

   // Test

   data_set.set_has_columns_names(true);
   data_set.set_separator(',');

   data_string = "\tx \t ,\t y \n"
                 "\t1 \t, \t 2 \n"
                 "\t3 \t, \t 4 \n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   data = data_set.get_data();

   assert_true(data_set.get_variable_name(0) == "x", LOG);
   assert_true(data_set.get_variable_name(1) == "y", LOG);

   assert_true((data(0,0) - 1.0) < 1.0e-4, LOG);
   assert_true((data(0,1) - 2.0) < 1.0e-4, LOG);
   assert_true((data(1,0) - 3.0) < 1.0e-4, LOG);
   assert_true((data(1,1) - 4.0) < 1.0e-4, LOG);

   // Test

   data_set.set_has_columns_names(true);
   data_set.set_separator(',');

   data_string = "x , y\n"
                 "1 , 2\n"
                 "3 , 4\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   data = data_set.get_data();

   assert_true(data_set.get_variable_name(0) == "x", LOG);
   assert_true(data_set.get_variable_name(1) == "y", LOG);

   assert_true((data(0,0) - 1.0 ) < 1.0e-4, LOG);
   assert_true((data(0,1) - 2.0 ) < 1.0e-4, LOG);
   assert_true((data(1,0) - 3.0 ) < 1.0e-4, LOG);
   assert_true((data(1,1) - 4.0 ) < 1.0e-4, LOG);

   // Test

   data_set.set_has_columns_names(false);
   data_set.set_separator(',');

   data_string =
   "5.1,3.5,1.4,0.2,Iris-setosa\n"
   "7.0,3.2,4.7,1.4,Iris-versicolor\n"
   "7.0,3.2,4.7,1.4,Iris-versicolor\n"
   "6.3,3.3,6.0,2.5,Iris-virginica";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   assert_true(data_set.get_instances_number() == 4, LOG);
   assert_true(data_set.get_variables_number() == 7, LOG);

   // Test

   data_set.set_has_columns_names(false);
   data_set.set_separator(',');

   data_string =
   "5.1,3.5,1.4,0.2,Iris-setosa\n"
   "7.0,3.2,4.7,1.4,Iris-versicolor\n"
   "7.0,3.2,4.7,1.4,Iris-versicolor\n"
   "6.3,3.3,6.0,2.5,Iris-virginica\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   assert_true(data_set.get_variables_number() == 7, LOG);
   assert_true(data_set.get_input_variables_number() == 4, LOG);
   assert_true(data_set.get_target_variables_number() == 3, LOG);
   assert_true(data_set.get_instances_number() == 4, LOG);

   data = data_set.get_data();

   assert_true((data(0,0) - 5.1) < 1.0e-4, LOG);
   assert_true((data(0,4) - 1) < 1.0e-4, LOG);
   assert_true((data(0,5) - 0) < 1.0e-4, LOG);
   assert_true((data(0,6) - 0) < 1.0e-4, LOG);
   assert_true((data(1,4) - 0) < 1.0e-4, LOG);
   assert_true((data(1,5) - 1) < 1.0e-4, LOG);
   assert_true((data(1,6) - 0) < 1.0e-4, LOG);
   assert_true((data(2,4) - 0) < 1.0e-4, LOG);
   assert_true((data(2,5) - 1) < 1.0e-4, LOG);
   assert_true((data(2,6) - 0) < 1.0e-4, LOG);
   assert_true((data(3,4) - 0) < 1.0e-4, LOG);
   assert_true((data(3,5) - 0) < 1.0e-4, LOG);
   assert_true((data(3,6) - 1) < 1.0e-4, LOG);

   // Test

   data_set.set_has_columns_names(true);
   data_set.set_separator(',');
   data_set.set_missing_values_label("NaN");

   data_string =
   "sepal length,sepal width,petal length,petal width,class\n"
   "NaN,3.5,1.4,0.2,Iris-setosa\n"
   "7.0,3.2,4.7,1.4,Iris-versicolor\n"
   "7.0,3.2,4.7,1.4,Iris-versicolor\n"
   "6.3,3.3,6.0,2.5,Iris-virginica\n"
   "0.0,0.0,0.0,0.0,NaN\n";

   file.open(data_file_name.c_str());
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

   assert_true(data_set.get_instances_number() == 5, LOG);

   data = data_set.get_data();

   assert_true(data(0,4) == 1.0, LOG);
   assert_true(data(0,5) == 0.0, LOG);
   assert_true(data(0,6) == 0.0, LOG);
   assert_true(data(1,4) == 0.0, LOG);
   assert_true(data(1,5) == 1.0, LOG);
   assert_true(data(1,6) == 0.0, LOG);
   assert_true(data(2,4) == 0.0, LOG);
   assert_true(data(2,5) == 1.0, LOG);
   assert_true(data(2,6) == 0.0, LOG);

   // Test

   data_set.set_has_columns_names(false);
   data_set.set_separator(',');
   data_set.set_missing_values_label("NaN");

   data_string =
   "0,0,0\n"
   "0,0,NaN\n"
   "0,0,0\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   // Test

   data_set.set_separator(' ');

   data_string = "1 2\n3 4\n5 6\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

//   data_set_pointer->set_name(0, "x");
//   data_set_pointer->set_name(1, "y");

   data_set.save("../data/data_set.xml");

   data_set.load("../data/data_set.xml");

   assert_true(data_set.get_variable_name(0) == "x", LOG);
   assert_true(data_set.get_variable_name(1) == "y", LOG);

   // Test

   data_set.set_has_columns_names(false);
   data_set.set_separator(' ');

   data_string = "1 true\n"
                 "3 false\n"
                 "5 true\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   assert_true(data_set.get_variables_number() == 2, LOG);
   assert_true(data_set.get_input_variables_number() == 1, LOG);
   assert_true(data_set.get_target_variables_number() == 1, LOG);

   data = data_set.get_data();

   assert_true(data(0,1) == 1.0, LOG);
   assert_true(data(1,1) == 0.0, LOG);
   assert_true(data(2,1) == 1.0, LOG);

   // Test

   data_set.set_separator('\t');
   data_set.set_missing_values_label("NaN");

   data_string =
   "f	52	1100	32	145490	4	no\n"
   "f	57	8715	1	242542	1	NaN\n"
   "m	44	5145	28	79100	5	no\n"
   "f	57	2857	16	1	1	NaN\n"
   "f	47	3368	44	63939	1	yes\n"
   "f	59	5697	14	45278	1	no\n"
   "m	86	1843	1	132799	2	yes\n"
   "m	67	4394	25	6670	2	no\n"
   "m	40	6619	23	168081	1	no\n"
   "f	12	4204	17	1	2	no\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   data_set.read_csv();

   assert_true(data_set.get_variables_number() == 7, LOG);
   assert_true(data_set.get_input_variables_number() == 6, LOG);
   assert_true(data_set.get_target_variables_number() == 1, LOG);

   data = data_set.get_data();

   assert_true(data.get_rows_number() == 10, LOG);
   assert_true(data.get_columns_number() == 7, LOG);
}


void DataSetTest::test_read_adult_csv()
{
    cout << "test_read_adult_csv\n";

    DataSet data_set;

    data_set.set_missing_values_label("?");
    data_set.set_separator(',');
    data_set.set_data_file_name("../../datasets/adult.data");
    data_set.set_has_columns_names(false);
    data_set.read_csv();

    assert_true(data_set.get_instances_number() == 1000, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(1) == DataSet::Categorical, LOG);
    assert_true(data_set.get_column_type(2) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(3) == DataSet::Categorical, LOG);

}

void DataSetTest::test_read_airline_passengers_csv()
{
    cout << "test_read_airline_passengers_csv\n";

    try
    {
        DataSet data_set("../../datasets/adult.data",',',true);

        assert_true(data_set.get_column_type(0) == DataSet::DateTime, LOG);
        assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
    }
    catch (exception& )
    {
        assert_true(true, LOG);
        cout << "Exception, date below 1970" << endl;
    }



}

void DataSetTest::test_read_car_csv()
{
    cout << "test_read_car_csv\n";

    try
    {
        DataSet data_set("../../datasets/car.data",',',false);

        assert_true(data_set.get_instances_number() == 1728, LOG);
        assert_true(data_set.get_column_type(0) == DataSet::Categorical, LOG);
        assert_true(data_set.get_column_type(1) == DataSet::Categorical, LOG);
        assert_true(data_set.get_column_type(2) == DataSet::Categorical, LOG);
        assert_true(data_set.get_column_type(3) == DataSet::Categorical, LOG);
        assert_true(data_set.get_column_type(4) == DataSet::Categorical, LOG);
        assert_true(data_set.get_column_type(5) == DataSet::Categorical, LOG);
        assert_true(data_set.get_column_type(6) == DataSet::Categorical, LOG);

    }
    catch (exception)
    {
        //Exception
        assert_true(true,LOG);

    }
}


void DataSetTest::test_read_empty_csv()
{
    cout << "test_read_empty_csv\n";

    try
    {
        DataSet data_set("../../datasets/empty.csv",',',false);

        assert_true(data_set.get_instances_number() == 1, LOG);
        assert_true(data_set.get_variables_number() == 0, LOG);

    }
    catch (exception)
    {
        //Exception, File is empty
        assert_true(true,LOG);
    }

}

void DataSetTest::test_read_heart_csv()
{
    cout << "test_read_heart_csv\n";

    DataSet data_set("../../datasets/heart.csv",',',true);

    assert_true(data_set.get_instances_number() == 303, LOG);
    assert_true(data_set.get_variables_number() == 14, LOG);
    assert_true(data_set.get_columns_number() == 14, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(1) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(2) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(3) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(4) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(5) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(6) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(7) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(8) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(9) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(10) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(11) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(12) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(13) == DataSet::Binary, LOG);
}

void DataSetTest::test_read_iris_csv()
{
    cout << "test_read_iris_csv\n";

    DataSet data_set("../../datasets/iris.data",',',false);

    assert_true(data_set.get_instances_number() == 150, LOG);
    assert_true(data_set.get_variables_number() == 7, LOG);
    assert_true(data_set.get_columns_number() == 5, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(1) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(2) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(3) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(4) == DataSet::Categorical, LOG);

}


void DataSetTest::test_read_mnsit_csv()
{
    cout << "test_read_mnist_csv\n";

    DataSet data_set("../../datasets/mnist.csv",',',false);

    assert_true(data_set.get_instances_number() == 100, LOG);
    assert_true(data_set.get_variables_number() == 785, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(1) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(2) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(3) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(4) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(20) == DataSet::Binary, LOG);

}


void DataSetTest::test_read_one_variable_csv()
{
    cout << "test_read_one_variable_csv\n";

    DataSet data_set("../../datasets/one_variable.csv",',',false);

    assert_true(data_set.get_instances_number() == 7, LOG);
    assert_true(data_set.get_variables_number() == 1, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
}


void DataSetTest::test_read_pollution_csv()
{
    cout << "test_read_pollution_csv\n";

    DataSet data_set("../../datasets/pollution.csv",',',true);

    assert_true(data_set.get_instances_number() == 1000, LOG);
    assert_true(data_set.get_variables_number() == 13, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::DateTime, LOG);
    assert_true(data_set.get_column_type(1) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(2) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(3) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(4) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(5) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(6) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(7) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(8) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(9) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(10) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(11) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(12) == DataSet::Numeric, LOG);
}

void DataSetTest::test_read_urinary_inflammations_csv()
{
    cout << "test_read_urinary_inflammations_csv\n";

    DataSet data_set("../../datasets/urinary_inflammations.csv",';',true);

    assert_true(data_set.get_instances_number() == 120, LOG);
    assert_true(data_set.get_variables_number() == 8, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(1) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(2) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(3) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(4) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(5) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(6) == DataSet::Binary, LOG);
    assert_true(data_set.get_column_type(7) == DataSet::Binary, LOG);
}

void DataSetTest::test_read_wine_csv()
{
    cout << "test_read_wine_csv\n";

    DataSet data_set("../../datasets/wine.data",',',false);

    assert_true(data_set.get_instances_number() == 178, LOG);
    assert_true(data_set.get_variables_number() == 14, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(1) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(2) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(3) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(4) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(5) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(6) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(7) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(8) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(9) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(10) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(11) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(12) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(13) == DataSet::Numeric, LOG);
}


void DataSetTest::test_read_binary_csv()
{
    cout << "test_read_binary_csv\n";

    DataSet data_set("../../datasets/binary.csv",',',false);

    assert_true(data_set.get_instances_number() == 8, LOG);
    assert_true(data_set.get_variables_number() == 3, LOG);
    assert_true(data_set.get_column_type(0) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(1) == DataSet::Numeric, LOG);
    assert_true(data_set.get_column_type(2) == DataSet::Binary, LOG);
}


void DataSetTest::test_convert_time_series()
{
    //@todo
   cout << "test_convert_time_series\n";

   DataSet data_set;

   Matrix<double> data;

   // Test

   data.set(2, 2, 3.1416);

   data_set.set_data(data);

   data_set.set_variable_name(0, "x");
   data_set.set_variable_name(1, "y");

   data_set.set_lags_number(1);

   data_set.transform_time_series();

   data = data_set.get_data();

   assert_true(data.get_rows_number() == 1, LOG);
   assert_true(data.get_columns_number() == 4, LOG);

   assert_true(data_set.get_instances_number() == 1, LOG);
   assert_true(data_set.get_variables_number() == 4, LOG);

   assert_true(data_set.get_input_variables_number() == 2, LOG);
   assert_true(data_set.get_target_variables_number() == 2, LOG);

   assert_true(data_set.get_variable_name(0) == "x", LOG);
   assert_true(data_set.get_variable_name(1) == "y", LOG);
   assert_true(data_set.get_variable_name(2) == "lag_1_x", LOG);
   assert_true(data_set.get_variable_name(3) == "lag_1_y", LOG);

}


void DataSetTest::test_convert_autoassociation()
{
   //@todo
   cout << "test_convert_autoassociation\n";

   DataSet data_set;

   Matrix<double> data;

   // Test

   data.set(2, 2, 3.1416);

   data_set.set_data(data);

   data_set.set_variable_name(0, "x");
   data_set.set_variable_name(1, "y");

   data_set.transform_association();

   data = data_set.get_data();

   assert_true(data.get_rows_number() == 2, LOG);
   assert_true(data.get_columns_number() == 4, LOG);

   assert_true(data_set.get_instances_number() == 2, LOG);
   assert_true(data_set.get_variables_number() == 4, LOG);

   assert_true(data_set.get_input_variables_number() == 2, LOG);
   assert_true(data_set.get_target_variables_number() == 2, LOG);

   assert_true(data_set.get_variable_name(0) == "x", LOG);
   assert_true(data_set.get_variable_name(1) == "y", LOG);
   assert_true(data_set.get_variable_name(2) == "autoassociation_x", LOG);
   assert_true(data_set.get_variable_name(3) == "autoassociation_y", LOG);
}


void DataSetTest::test_scrub_missing_values()
{
    cout << "test_scrub_missing_values\n";
/*
    const string data_file_name = "../data/data.dat";

    ofstream file;

    DataSet data_set;

    data_set.set_data_file_name(data_file_name);

    Instances instances;

    Matrix<double> data;

    string data_string;

    // Test

    data_set.set_separator(' ');
    data_set.set_missing_values_label("NaN");
    data_set.set_file_type("dat");

    data_string = "0 0 0\n"
                  "0 0 NaN\n"
                  "0 0 0\n";

    file.open(data_file_name.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    data_set.scrub_missing_values();

//    instances = data_set.get_instances();

    assert_true(instances.get_use(1) == Instances::Unused, LOG);

    // Test

    data_set.set_separator(' ');
    data_set.set_missing_values_label("NaN");
    data_set.set_file_type("dat");

    data_string = "NaN 3   3\n"
                  "2   NaN 3\n"
                  "0   1   NaN\n";

    file.open(data_file_name.c_str());
    file << data_string;
    file.close();

    data_set.read_csv();

    data_set.scrub_missing_values();

//    instances = data_set.get_instances();

    data = data_set.get_data();

    assert_true(abs(data(0,0) - 1.0) < 1.0e-3, LOG);
    assert_true(abs(data(1,1) - 2.0) < 1.0e-3, LOG);
    assert_true(abs(data(2,2) - 3.0) < 1.0e-3, LOG);
*/
}


void DataSetTest::test_empty()
{
    cout << "test_emprty\n";
/*
    Matrix<double> matrix(9,9);

    DataSet data_set;

    data_set.set(matrix);

    assert_true(abs(data_set.empty() - 0) < 1.0e-6 , LOG);
    */
}


void DataSetTest::test_filter_variable()
{
    cout << "test_filter_variable";
/*
    Matrix<double> matrix({{1,2,3},{4,2,8},{7,8,6}});
    Vector<size_t> solution({0,1});
    Vector<size_t> solution_1({0, 1, 2});
    Vector<size_t> solution_2({});
    DataSet data_set;
    data_set.set_data(matrix);

    //Test
    assert_true(data_set.filter_variable(1, 1, 3) == solution, LOG);
    assert_true(data_set.filter_variable(0, 4, 5) == solution_1, LOG);

    //Test
    Vector<string> header({"a","b","c"});
    Matrix<double> matrix_1({{1,2,3},{4,2,8},{7,8,6}}, {"a","b","c"});
    DataSet ds_1;
    ds_1.set_columns_names(true);
    ds_1.set_data(matrix_1);
*/
}


void DataSetTest::test_calculate_variables_means()
{
    cout << "test_calculate_variables_means\n";

    Matrix<double> matrix({{1, 2, 3, 4},{2, 2, 2, 2},{1, 1, 1, 1}});

    DataSet data_set;
    data_set.set_data(matrix);
    Vector<size_t> index({0, 1});
    Vector<double> means = data_set.calculate_variables_means(index);
    Vector<double> solution(2, 2.0);

    assert_true(means == solution, LOG);
}


/// @todo

void DataSetTest::test_calculate_training_targets_mean()
{
    cout << "test_calculate_training_targets_mean\n";
/*
    Matrix<double> matrix({{1, static_cast<double>(NAN), 1, 1},{2, 2, 2, 2},{3, 3, static_cast<double>(NAN), 3}});
    Vector<size_t> indices({0, 1, 2});
    Vector<size_t> training_indexes({0, 1, 2});
    DataSet data_set;
    data_set.set_data(matrix);

    data_set.set_training(training_indexes);

    // Test 3 targets
    data_set.set_target_variables_indices(indices);

    Vector<double> means = data_set.calculate_training_targets_mean();
    Vector<double> solutions({1.0 , 2.0, 3.0});

    assert_true(means == solutions, LOG);

    // Test 1 target
    Vector<size_t> index_target({0});
    Vector<size_t> indexes_inputs({1, 2});

    data_set.set_target_variables_indices(index_target);
    data_set.set_input_variables_indices(indexes_inputs);

    Vector<double> mean = data_set.calculate_training_targets_mean();
    Vector<double> solution({1.0});

    assert_true(mean == solution, LOG);
*/
}


/// @todo

void DataSetTest::test_calculate_selection_targets_mean()
{
    cout << "test_calculate_selection_targets_mean\n";
/*
    Matrix<double> matrix({{1, static_cast<double>(NAN), 6, 9},{1, 2, 5, 2},{3, 2, static_cast<double>(NAN), 4}});
    Vector<size_t> indexes_targets({2});
    Vector<size_t> selection_indexes({0, 1});
    DataSet data_set;
    data_set.set_data(matrix);

    data_set.set_input();

    data_set.set_selection(selection_indexes);

    // Test 3 targets
    data_set.set_target_variables_indices(indexes_targets);

    Vector<double> means = data_set.calculate_selection_targets_mean();
    Vector<double> solutions({2.0, 3.0});

    cout << "means: " << means << endl;

    assert_true(means == solutions, LOG);
*/
}


/// @todo

void DataSetTest::test_calculate_testing_targets_mean()
{
    cout << "test_calculate_testing_targets_mean\n";
/*
    Matrix<double> matrix({{1, 1, 1, 1},{2, 2, 2, 2},{3, 3, 3, 3}});

    DataSet data_set;
    data_set.set_data(matrix);
    vector<size_t> targets_indices({2});
    vector<size_t> testing_indices({2, 3});

    data_set.set_target_variables_indices(targets_indices);

    data_set.set_testing(testing_indices);

    Vector<double> mean = data_set.calculate_testing_targets_mean();

    assert_true(mean == 3.0, LOG);
*/
}


/// @todo

void DataSetTest::test_calculate_input_target_correlations()
{
    cout << "test_calculate_input_target_correlations\n";
/*
    Matrix<double> matrix({{1, 1, 1, 1},{2, 2, 2, 2},{3, 3, 3, 3}});

    DataSet data_set;
    data_set.set_data(matrix);
    Vector<size_t> inputs_indices({0, 1});

    data_set.set_input_variables_indices(inputs_indices);

    Matrix<double> correlations_targets = data_set.calculate_inputs_targets_correlations();

    //Test linear correlation
    assert_true(correlations_targets - 1.0 < 1.0e-3, LOG);

    //Test logistic correlation

*/
}


/// @todo

void DataSetTest::test_calculate_total_input_correlations()
{
    cout << "test_calculate_total_input_correlations\n";
/*
    Matrix<double> matrix({{1, 1, 1, 1},{2, 2, 2, 2},{3, 3, 3, 3}});

    DataSet data_set;
    data_set.set_data(matrix);
    Vector<size_t> inputs_indices({0, 1});
    Vector<size_t> targets_indices({2});
    Vector<double> solution({1, 1});

    data_set.set_input_variables_indices(inputs_indices);

    Vector<double> correlations_inputs = data_set.calculate_total_input_correlations();

    assert_true(correlations_inputs == solution, LOG);
*/
}


void DataSetTest::test_unuse_repeated_instances()
{
    cout << "test_unuse_repeated_instances\n";
/*
    Matrix<double> matrix({{1,2,2},{1,2,2},{1,6,6}});
    DataSet data_set;
    data_set.set_data(matrix);
    Vector<size_t> indices({2});

    assert_true(data_set.unuse_repeated_instances() == indices, LOG);

    Matrix<double> matrix_1({{1,2,2,2},{1,2,2,2},{1,6,6,6}});
    DataSet ds_1;
    ds_1.set_data(matrix_1);
    Vector<size_t> indices_1({2, 3});

    assert_true(ds_1.unuse_repeated_instances() == indices_1, LOG);

    Matrix<double> matrix_2({{1,2,2,4,4},{1,2,2,4,4},{1,6,6,4,4}});
    DataSet ds_2;
    ds_2.set_data(matrix_2);
    Vector<size_t> indices_2({2,4});

    assert_true(ds_2.unuse_repeated_instances() == indices_2, LOG);
*/
}


void DataSetTest::test_unuse_non_significant_inputs()
{

    /*
    cout << "test_unuse_non_significant_inputs\n";

    Matrix<double> matrix({{1,0,0},{1,0,0},{1,0,1}});
    DataSet data_set;

    cout << "unuse: " << data_set.unuse_non_significant_inputs() << endl;
*/
}


/// @todo

void DataSetTest::test_unuse_columns_missing_values()
{
    cout << "test_unuse_variables_missing_values\n";

    Matrix<double> matrix({{1,2,2,4,4},{1,2,2,4,4},{1,6,6,4,4}});
//    matrix.set_header(Vector<string>({"var1","var2","var3","var4"}));

    DataSet data_set;

    data_set.set_data(matrix);
    data_set.set_variables_names(Vector<string>({"var1","var2","var3"}));

//    data_set.unuse_variables_missing_values(1);
}


/// @todo

void DataSetTest::test_perform_principal_components_analysis()
{
    cout << "test_perform_principal_components_analysis\n";

    Matrix<double> matrix({{1,1},{-1,-1},{1,1}});
    DataSet data_set;
    data_set.set_data(matrix);

    Vector<double> solution({1,1});

    //data_set.perform_principal_components_analysis();
    //Matrix<double> PCA = data_set.get_data();
    //assert_true(PCA.get_column(2) == solution, LOG);
}


void DataSetTest::test_calculate_training_negatives()
{
    cout << "test_calculate_training_negatives\n";
/*
    Matrix<double> matrix({{1,1,1},{-1,-1,-1},{0,1,1}});

    DataSet data_set;
    data_set.set_data(matrix);
    Vector<size_t> training_indices({0,1});
    Vector<size_t> inputs_indices({0, 1});
    Vector<size_t> target_indices({2});
    size_t target_index = 2;

    data_set.set_testing();
    data_set.set_training(training_indices);

    data_set.set_input_variables_indices(inputs_indices);
    data_set.set_target_variables_indices(target_indices);

    size_t training_negatives = data_set.calculate_training_negatives(target_index);

    Matrix<double> data = data_set.get_data();

    assert_true(training_negatives == 1, LOG);
*/
}


/// @todo

void DataSetTest::test_calculate_selection_negatives()
{
    cout << "test_calculate_selection_negatives\n";
/*
    Matrix<double> matrix({{1,1,1},{-1,-1,-1},{0,1,1}});

    DataSet data_set;
    data_set.set_data(matrix);
    Vector<size_t> selection_indices({0,1});
    Vector<size_t> inputs_indices({0, 1});
    Vector<size_t> target_indices({2});
    size_t target_index = 2;

    data_set.set_testing();
    data_set.set_selection(selection_indices);

    data_set.set_input_variables_indices(inputs_indices);
    data_set.set_target_variables_indices(target_indices);

    size_t selection_negatives = data_set.calculate_training_negatives(target_index);

    Matrix<double> data = data_set.get_data();

    assert_true(selection_negatives == 0, LOG);
*/
}


void DataSetTest::test_is_binary_classification()
{
    cout << "test_is_binary_classification\n";
/*
    Matrix<double> matrix({{1,2,1},{1,1,0},{0,1,2}});
    DataSet data_set;
    data_set.set_data(matrix);
    Vector<size_t> input_indices({0,1});
    Vector<size_t> target_indices({2});

    data_set.set_input_variables_indices(input_indices);
    data_set.set_target_variables_indices(target_indices);

    bool classification = data_set.is_binary_classification();

    assert_true(classification == false, LOG);
*/
}


/// @todo

void DataSetTest::test_is_multiple_classification()
{
    cout << "test_is_multiple_classification\n";
/*
    Matrix<double> matrix({{1,0,1},{1,0,1},{2,1,1}});
    DataSet data_set;
    data_set.set_data(matrix);
    Vector<size_t> input_indices({0,1});
    Vector<size_t> target_indices({2});

    data_set.set_input_variables_indices(input_indices);
    data_set.set_target_variables_indices(target_indices);
    bool classification = data_set.is_multiple_classification();


    assert_true(classification == true, LOG);
*/
}


void DataSetTest::run_test_case()
{
   cout << "Running data set test case...\n";
/*
   // Constructor and destructor methods
   test_constructor();
   test_destructor();

   // Assignment operators methods
   test_assignment_operator();

   // Get methods
   test_get_instances_number();
   test_get_variables_number();
   test_get_variables();
   test_get_display();
   test_is_binary_variable();
   test_is_binary_classification();
   test_is_multiple_classification();

   // Data methods
   test_empty();
   test_get_data();
   test_get_training_data();
   test_get_selection_data();
   test_get_testing_data();
   test_get_selection_data_set();
   test_get_testing_data_set();
   test_get_inputs();
   test_get_targets();

   // Instance methods
   test_get_instance();

   // Set methods
   test_set();
   test_set_display();

   // Data methods
   test_set_data();
   test_set_instances_number();
   test_set_variables_number();

   // Instance methods
   test_set_instance();

   // Data resizing methods
  // test_subtract_instance();    @todo fail test
   test_unuse_constant_columns();
   test_unuse_repeated_instances();
  // test_unuse_non_significant_inputs();
   test_unuse_variables_missing_values();

   // Initialization methods
   test_initialize_data();

   // Statistics methods
   test_calculate_data_descriptives();
   test_calculate_data_descriptives_missing_values();
   test_calculate_training_instances_descriptives();
   test_calculate_selection_instances_descriptives();
   test_calculate_testing_instances_descriptives();
   test_calculate_inputs_descriptives();
   test_calculate_training_targets_mean();
   test_calculate_selection_targets_mean();
   test_calculate_testing_targets_mean();

   // Histrogram methods
   test_calculate_data_histograms();

   // Filtering methods
   test_filter_data();
   test_filter_variable();

   // Data scaling
   test_scale_data_mean_standard_deviation();
   test_scale_data_minimum_maximum();

   // Input variables scaling
    test_scale_inputs_mean_standard_deviation();
    test_scale_inputs_minimum_maximum();

   // Target variables scaling
    test_scale_targets_mean_standard_deviation();
    test_scale_targets_minimum_maximum();

   // Data unscaling
   test_unscale_data_mean_standard_deviation();
   test_unscale_data_minimum_maximum();

   // Input variables unscaling
    test_unscale_inputs_mean_standard_deviation();
    test_unscale_inputs_minimum_maximum();

   // Target variables unscaling
   test_unscale_targets_mean_standard_deviation();
   test_unscale_targets_minimum_maximum();

   // Classificatios methods
   test_get_binary_inputs_indices();
   //test_balance_binary_targets_distribution();

   // Correlations
   test_calculate_input_target_correlations();
   test_calculate_total_input_correlations();
   test_calculate_multiple_linear_correlations();

   // Input-target variables unscaling
   //test_unscale_variables_mean_standard_deviation();
   //test_unscale_variables_minimum_maximum();

   // Pattern recognition methods
   test_calculate_target_columns_distribution();
  // test_unuse_most_populated_target();
  //test_balance_multiple_targets_distribution();

  // test_balance_function_regression_targets_distribution();

   // Outlier detection

   test_clean_Tukey_outliers();
   //test_calculate_instances_distances();

   //test_calculate_k_distances();
   //test_calculate_reachability_distances();
   //test_calculate_reachability_density();
   //test_calculate_local_outlier_factor();

   //test_clean_local_outlier_factor();

   // Data generation
   //test_generate_data_binary_classification();
   //test_generate_data_multiple_classification();

   //test_generate_constant_data();

   // Serialization methods

  test_to_XML();
  test_from_XML();

   test_read_csv();
*/
   test_read_adult_csv();
   test_read_airline_passengers_csv();
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


//   test_convert_time_series();
//   test_convert_autoassociation();

//   test_calculate_training_negatives();
//   test_calculate_selection_negatives();

//   test_scrub_missing_values();

   // Principal components mehtod

//   test_perform_principal_components_analysis();

   // String utilities

//   test_trim();
//   test_get_trimmed();

//   test_count_tokens();
//   test_get_tokens();

//   test_is_numeric();


   cout << "End of data set test case.\n";
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
