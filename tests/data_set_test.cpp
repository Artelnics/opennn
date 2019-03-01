/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   D A T A   S E T   T E S T   C L A S S                                                                      */
/*                                                                                                              */ 
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// Unit testing includes

#include "data_set_test.h"

using namespace OpenNN;

// GENERAL CONSTRUCTOR

DataSetTest::DataSetTest() : UnitTesting() 
{
}


// DESTRUCTOR

DataSetTest::~DataSetTest()
{
}


// METHODS

void DataSetTest::test_constructor()
{
   message += "test_constructor\n";

   // Default constructor

   DataSet ds1;

   assert_true(ds1.get_variables().get_variables_number() == 0, LOG);
   assert_true(ds1.get_instances().get_instances_number() == 0, LOG);

   // Instances and variables number constructor

   DataSet ds2(1, 2);

   assert_true(ds2.get_instances().get_instances_number() == 1, LOG);
   assert_true(ds2.get_variables().get_variables_number() == 2, LOG);

   // Inputs, targets and instances numbers constructor

   DataSet ds3(1, 1, 1);

   assert_true(ds3.get_variables().get_variables_number() == 2, LOG);
   assert_true(ds3.get_instances().get_instances_number() == 1, LOG);

   // XML constructor

   tinyxml2::XMLDocument* document = ds3.to_XML();

   DataSet ds4(*document);

   assert_true(ds4.get_variables().get_variables_number() == 2, LOG);
   assert_true(ds4.get_instances().get_instances_number() == 1, LOG);

   delete document;

   // File constructor

//   const string file_name = "../data/data_set.xml";

//   ds1.save(file_name);

//   DataSet ds5(file_name);

//   assert_true(ds5.get_variables().get_variables_number() == 0, LOG);
//   assert_true(ds5.get_instances().get_instances_number() == 0, LOG);

   // Copy constructor

   DataSet ds6(ds1);

   assert_true(ds6.get_variables().get_variables_number() == 0, LOG);
   assert_true(ds6.get_instances().get_instances_number() == 0, LOG);

}


void DataSetTest::test_destructor()
{
   message += "test_destructor\n";

   DataSet* dsp = new DataSet(1, 1, 1);

   delete dsp;
}


void DataSetTest::test_assignment_operator()
{
   message += "test_assignment_operator\n";

   DataSet ds1(1, 1, 1);
   DataSet ds2 = ds1;

   assert_true(ds2.get_instances().get_instances_number() == 1, LOG);
   assert_true(ds2.get_variables().get_variables_number() == 2, LOG);
}


void DataSetTest::test_get_instances_number() 
{
   message += "test_get_instances_number\n";

   DataSet ds;

   assert_true(ds.get_instances().get_instances_number() == 0, LOG);
}


void DataSetTest::test_get_variables_number() 
{
   message += "test_get_variables_number\n";

   DataSet ds;

   assert_true(ds.get_variables().get_variables_number() == 0, LOG);
}


void DataSetTest::test_get_variables() 
{
   message += "test_get_variables\n";

   DataSet ds(1, 3, 2);

   const Variables& variables = ds.get_variables();

   assert_true(variables.get_inputs_number() == 3, LOG);
   assert_true(variables.get_targets_number() == 2, LOG);
}


void DataSetTest::test_get_display() 
{
   message += "test_get_display\n";

   DataSet ds;

   ds.set_display(true);

   assert_true(ds.get_display() == true, LOG);

   ds.set_display(false);

   assert_true(ds.get_display() == false, LOG);
}


void DataSetTest::test_get_data() 
{
   message += "test_get_data\n";

   DataSet ds(1,1,1);

   ds.initialize_data(0.0);

   const Matrix<double>& data = ds.get_data();

   assert_true(data.get_rows_number() == 1, LOG);
   assert_true(data.get_columns_number() == 2, LOG);
   assert_true(data == 0.0, LOG);
}


void DataSetTest::test_get_training_data()
{
   message += "test_get_training_data\n";
}


void DataSetTest::test_get_selection_data()
{
   message += "test_get_selection_data\n";
}


void DataSetTest::test_get_testing_data()
{
   message += "test_get_testing_data\n";
}


void DataSetTest::test_get_inputs() 
{
   message += "test_get_inputs\n";

//   DataSet ds(1, 3, 2);

//   size_t instances_number = ds.get_instances().get_instances_number();
//   size_t inputs_number = ds.get_variables().get_inputs_number();

//   Matrix<double> inputs = ds.get_inputs();

//   size_t rows_number = inputs.get_rows_number();
//   size_t columns_number = inputs.get_columns_number();

//   assert_true(instances_number == rows_number, LOG);
//   assert_true(inputs_number == columns_number, LOG);
}


void DataSetTest::test_get_targets() 
{
   message += "test_get_targets\n";

   DataSet ds(1,3,2);

   size_t instances_number = ds.get_instances().get_instances_number();
   size_t targets_number = ds.get_variables().get_targets_number();

   Matrix<double> targets = ds.get_targets();

   size_t rows_number = targets.get_rows_number();
   size_t columns_number = targets.get_columns_number();

   assert_true(instances_number == rows_number, LOG);
   assert_true(targets_number == columns_number, LOG);
}


void DataSetTest::test_get_instance()
{
   message += "test_get_instance\n";

   DataSet ds;
   Vector<double> instance;

   // Test

   ds.set(1, 1, 1);
   ds.initialize_data(1.0);

   instance = ds.get_instance(0);

   assert_true(instance.size() == 2, LOG);
   assert_true(instance == 1.0, LOG);
}


void DataSetTest::test_set() 
{
   message += "test_set\n";

   DataSet ds;

   Matrix<double> data;

   // Instances and inputs and target variables

   ds.set(1, 2, 3);

   assert_true(ds.get_instances().get_instances_number() == 1, LOG);
   assert_true(ds.get_variables().get_inputs_number() == 2, LOG);
   assert_true(ds.get_variables().get_targets_number() == 3, LOG);

   data = ds.get_data();

   assert_true(data.get_rows_number() == 1, LOG);
   assert_true(data.get_columns_number() == 5, LOG);
}


void DataSetTest::test_set_instances_number() 
{
   message += "test_set_instances_number\n";

   DataSet ds(1,1,1);

   ds.set_instances_number(2);

   assert_true(ds.get_instances().get_instances_number() == 2, LOG);
}


void DataSetTest::test_set_variables_number() 
{
   message += "test_set_variables_number\n";

   DataSet ds(1, 1);

   ds.set_variables_number(2);

   assert_true(ds.get_variables().get_variables_number() == 2, LOG);
}


void DataSetTest::test_set_display() 
{
   message += "test_set_display\n";
}


void DataSetTest::test_set_data() 
{
   message += "test_set_data\n";

   DataSet ds(1, 1, 1);

   Matrix<double> new_data(1, 2, 0.0);

   ds.set_data(new_data);

   Matrix<double> data = ds.get_data();

   assert_true(data == new_data, LOG);
}


void DataSetTest::test_set_instance()
{
   message += "test_set_instance\n";

   DataSet ds(1, 1, 1);

   Vector<double> new_instance(2, 0.0);

   ds.set_instance(0, new_instance);

   Vector<double> instance = ds.get_instance(0);

   assert_true(instance == new_instance, LOG);
}


void DataSetTest::test_add_instance() 
{
   message += "test_add_instance\n";

   DataSet ds(1,1,1);

   Vector<double> new_instance(2, 0.0);

   ds.add_instance(new_instance);

   assert_true(ds.get_instances().get_instances_number() == 2, LOG);

   Vector<double> instance = ds.get_instance(1);

   assert_true(instance == new_instance, LOG);

}


// @todo

void DataSetTest::test_subtract_instance() 
{
//   message += "test_subtract_instance\n";

//   DataSet ds(3, 1, 1);

//   ds.subtract_instance(1);

//   assert_true(ds.get_instances().get_instances_number() == 2, LOG);
}


void DataSetTest::test_calculate_data_statistics() 
{
   message += "test_calculate_data_statistics\n";

   DataSet ds;

   Vector< Statistics<double> > statistics;

   // Test

   ds.set(1, 1);

   ds.initialize_data(0.0);

   statistics = ds.calculate_data_statistics();

   assert_true(ds.get_missing_values().get_missing_values_number() == 0, LOG);

   assert_true(statistics.size() == 1, LOG);

   assert_true(statistics[0].minimum == 0.0, LOG);
   assert_true(statistics[0].maximum == 0.0, LOG);
   assert_true(statistics[0].mean == 0.0, LOG);
   assert_true(statistics[0].standard_deviation == 0.0, LOG);

   // Test

   ds.set(2, 2, 2);

   ds.initialize_data(0.0);

   statistics = ds.calculate_data_statistics();

   assert_true(statistics.size() == 4, LOG);

   assert_true(statistics[0].minimum == 0.0, LOG);
   assert_true(statistics[0].maximum == 0.0, LOG);
   assert_true(statistics[0].mean == 0.0, LOG);
   assert_true(statistics[0].standard_deviation == 0.0, LOG);

   assert_true(statistics[1].minimum == 0.0, LOG);
   assert_true(statistics[1].maximum == 0.0, LOG);
   assert_true(statistics[1].mean == 0.0, LOG);
   assert_true(statistics[1].standard_deviation == 0.0, LOG);

   assert_true(statistics[2].minimum == 0.0, LOG);
   assert_true(statistics[2].maximum == 0.0, LOG);
   assert_true(statistics[2].mean == 0.0, LOG);
   assert_true(statistics[2].standard_deviation == 0.0, LOG);

   assert_true(statistics[3].minimum == 0.0, LOG);
   assert_true(statistics[3].maximum == 0.0, LOG);
   assert_true(statistics[3].mean == 0.0, LOG);
   assert_true(statistics[3].standard_deviation == 0.0, LOG);

}


void DataSetTest::test_calculate_data_statistics_missing_values()
{
    message += "test_calculate_data_statistics_missing_values\n";

    const string data_file_name = "../data/data.dat";

    ofstream file;

    DataSet ds;

    ds.set_data_file_name(data_file_name);

      ds.set_file_type("dat");

    Matrix<double> data;

    string data_string;

    ds.set_separator("Space");
    ds.set_missing_values_label("?");

    data_string = "-1000 ? 0 \n 3 4 ? \n ? 4 1";

    file.open(data_file_name.c_str());
    file << data_string;
    file.close();

    ds.load_data();

    data = ds.get_data();

    assert_true(ds.calculate_data_statistics_matrix()(0, 0) == -1000, LOG);
    assert_true(ds.calculate_data_statistics_matrix()(1, 0) == 4, LOG);
    assert_true(ds.calculate_data_statistics_matrix()(2, 0) == 0, LOG);
}


void DataSetTest::test_calculate_training_instances_statistics()
{
   message += "test_calculate_training_instances_statistics\n";

   DataSet ds;
   Vector< Statistics<double> > training_instances_statistics;

   Instances* instances_pointer;

   // Test

   ds.set(2, 2, 2);

   instances_pointer = ds.get_instances_pointer();
   instances_pointer->set_training();

   ds.initialize_data(0.0);

   ds.calculate_training_instances_statistics();

}


void DataSetTest::test_calculate_selection_instances_statistics()
{
   message += "test_calculate_selection_instances_statistics\n";

   DataSet ds;
   Vector< Statistics<double> > selection_instances_statistics;

   Instances* instances_pointer;

   // Test

   ds.set(2,2,2);

   instances_pointer = ds.get_instances_pointer();
   instances_pointer->set_selection();

   ds.initialize_data(0.0);

   selection_instances_statistics = ds.calculate_selection_instances_statistics();
}


void DataSetTest::test_calculate_testing_instances_statistics()
{
   message += "test_calculate_testing_instances_statistics\n";

   DataSet ds;
   Vector< Statistics<double> > testing_instances_statistics;

   Instances* instances_pointer;

   // Test

   ds.set(2, 2, 2);

   instances_pointer = ds.get_instances_pointer();
   instances_pointer->set_testing();
   
   ds.initialize_data(0.0);

   testing_instances_statistics = ds.calculate_testing_instances_statistics();
}


void DataSetTest::test_calculate_input_statistics()
{
   message += "test_calculate_input_statistics\n";
}


void DataSetTest::test_calculate_targets_statistics()
{
   message += "test_calculate_targets_statistics\n";
}


void DataSetTest::test_calculate_linear_correlations()
{
   message += "test_calculate_linear_correlations\n";
}

void DataSetTest::test_calculate_autocorrelations()
{
    message += "test_calculate_autocorrelations\n";

    DataSet ds;

    Matrix<double> autocorrelations;

    ds.set(20, 1, 1);

    ds.randomize_data_normal();

    autocorrelations = ds.calculate_autocorrelations();

    assert_true(autocorrelations.get_columns_number() == 10, LOG);
    assert_true(autocorrelations.get_rows_number() == 2, LOG);
}


void DataSetTest::test_calculate_cross_correlations()
{
    message += "test_calculate_cross_correlations";

    DataSet ds;

    Matrix< Vector<double> > cross_correlations;

    ds.set(20, 5, 1);

    ds.randomize_data_normal();

    cross_correlations = ds.calculate_cross_correlations();

    assert_true(cross_correlations.get_columns_number() == 6, LOG);
    assert_true(cross_correlations.get_rows_number() == 6, LOG);
}


void DataSetTest::test_calculate_data_histograms()
{
   message += "test_calculate_data_histograms\n";
}


void DataSetTest::test_filter_data()
{
   message += "test_filter_data\n";

   DataSet ds;

   Vector<double> minimums;
   Vector<double> maximums;

   Matrix<double> data;

   // Test

   ds.set(2, 1, 1);
   ds.initialize_data(1.0);

   minimums.set(2, 0.0);
   maximums.set(2, 0.5);

   ds.filter_data(minimums, maximums);

   data = ds.get_data();

   assert_true(ds.get_instances().get_use(0) == Instances::Unused, LOG);
   assert_true(ds.get_instances().get_use(1) == Instances::Unused, LOG);
}


void DataSetTest::test_scale_inputs_mean_standard_deviation() 
{
   message += "test_scale_inputs_mean_standard_deviation\n";

   DataSet ds;

   Vector< Statistics<double> > inputs_statistics;

   // Test

   ds.set(2, 2, 2);
   ds.randomize_data_normal();

   ds.scale_inputs_mean_standard_deviation();

   inputs_statistics = ds.calculate_inputs_statistics();

   assert_true(inputs_statistics[0].has_mean_zero_standard_deviation_one(), LOG);
}


void DataSetTest::test_scale_targets_mean_standard_deviation() 
{
   message += "test_scale_targets_mean_standard_deviation\n";

   DataSet ds;

   Vector< Statistics<double> > targets_statistics;

   // Test

   ds.set(2, 2, 2);
   ds.randomize_data_normal();

   ds.scale_targets_mean_standard_deviation();

   targets_statistics = ds.calculate_targets_statistics();

   assert_true(targets_statistics[0].has_mean_zero_standard_deviation_one(), LOG);
}


void DataSetTest::test_scale_inputs_minimum_maximum() 
{
   message += "test_scale_inputs_minimum_maximum\n";

   DataSet ds;

   Vector< Statistics<double> > inputs_statistics;

   // Test

   ds.set(2, 2, 2);
   ds.randomize_data_normal();

   ds.scale_inputs_minimum_maximum();

   inputs_statistics = ds.calculate_inputs_statistics();

   assert_true(inputs_statistics[0].has_minimum_minus_one_maximum_one(), LOG);
}


void DataSetTest::test_scale_targets_minimum_maximum() 
{
   message += "test_scale_targets_minimum_maximum\n";

   DataSet ds;

   Vector< Statistics<double> > targets_statistics;

   // Test

   ds.set(2, 2, 2);
   ds.randomize_data_normal();

   ds.scale_targets_minimum_maximum();

   targets_statistics = ds.calculate_targets_statistics();

   assert_true(targets_statistics[0].has_minimum_minus_one_maximum_one(), LOG);
}


void DataSetTest::test_scale_data_minimum_maximum()
{
   message += "test_scale_data_minimum_maximum\n";

   DataSet ds;

   Vector< Statistics<double> > data_statistics;

   Matrix<double> data;
   Matrix<double> scaled_data;

    // Test

   ds.set(2,2,2);
   ds.initialize_data(0.0);

   ds.set_display(false);

   data = ds.get_data();

   data_statistics = ds.scale_data_minimum_maximum();

   scaled_data = ds.get_data();

   assert_true(scaled_data == data, LOG);
}


void DataSetTest::test_scale_data_mean_standard_deviation()
{
   message += "test_scale_data_mean_standard_deviation\n";

   DataSet ds;

   Vector< Statistics<double> > data_statistics;

   Matrix<double> data;
   Matrix<double> scaled_data;

    // Test

   ds.set(2,2,2);
   ds.initialize_data(0.0);

   ds.set_display(false);

   data = ds.get_data();

   data_statistics = ds.scale_data_mean_standard_deviation();

   scaled_data = ds.get_data();

   assert_true(scaled_data == data, LOG);
}


void DataSetTest::test_unscale_data_mean_standard_deviation()
{
   message += "test_unscale_data_mean_standard_deviation\n";
}


void DataSetTest::test_unscale_data_minimum_maximum()
{
   message += "test_unscale_data_minimum_maximum\n";
}


void DataSetTest::test_unscale_inputs_mean_standard_deviation() 
{
   message += "test_unscale_inputs_mean_standard_deviation\n";

//   DataSet ds(2, 2, 2);
//   ds.initialize_data(0.0);

//   ds.set_display(false);

//   Vector< Statistics<double> > data_statistics;

//   // Test

//   Matrix<double> inputs = ds.get_inputs();

//   data_statistics.set(4);

//   ds.unscale_inputs_mean_standard_deviation(data_statistics);

//   Matrix<double> new_inputs = ds.get_inputs();

//   assert_true(new_inputs == inputs, LOG);

}


void DataSetTest::test_unscale_targets_mean_standard_deviation() 
{
   message += "test_unscale_targets_mean_standard_deviation\n";
   
   DataSet ds(2, 2, 2);
   ds.initialize_data(0.0);

   ds.set_display(false);

   Matrix<double> targets = ds.get_targets();

   Vector< Statistics<double> > data_statistics(4);

   ds.unscale_targets_mean_standard_deviation(data_statistics);

   Matrix<double> new_targets = ds.get_targets();

   assert_true(new_targets == targets, LOG);
}


void DataSetTest::test_unscale_variables_mean_standard_deviation() 
{
   message += "test_unscale_variables_mean_standard_deviation\n";
}


void DataSetTest::test_unscale_inputs_minimum_maximum() 
{
   message += "test_unscale_inputs_minimum_maximum\n"; 

   DataSet ds(2, 2, 2);
   ds.initialize_data(0.0);

   ds.set_display(false);

   Vector< Statistics<double> > data_statistics;

   // Test

   Matrix<double> inputs = ds.get_inputs();

   data_statistics.set(4);

   ds.unscale_inputs_minimum_maximum(data_statistics);

   Matrix<double> new_inputs = ds.get_inputs();

   assert_true(new_inputs == inputs, LOG);
}


void DataSetTest::test_unscale_targets_minimum_maximum() 
{
   message += "test_unscale_targets_minimum_maximum\n";

   DataSet ds(2, 2, 2);
   ds.initialize_data(0.0);

   ds.set_display(false);

   Matrix<double> targets = ds.get_targets();

   Vector< Statistics<double> > data_statistics(4);

   ds.unscale_targets_minimum_maximum(data_statistics);

   Matrix<double> new_targets = ds.get_targets();

   assert_true(new_targets == targets, LOG);

}


void DataSetTest::test_unscale_variables_minimum_maximum() 
{
   message += "test_unscale_variables_minimum_maximum\n"; 
}


void DataSetTest::test_subtract_constant_variables()
{
   message += "test_subtract_constant_variables\n"; 

   DataSet ds;

   // Test 

   ds.set(1, 2, 1);

   ds.initialize_data(0.0);

   ds.unuse_constant_variables();

   assert_true(ds.get_variables().get_inputs_number() == 0, LOG);
   assert_true(ds.get_variables().get_targets_number() == 1, LOG);
}


void DataSetTest::test_subtract_repeated_instances()
{
   message += "test_subtract_repeated_instances\n"; 
}


void DataSetTest::test_initialize_data()
{
   message += "test_initialize_data\n";
}


void DataSetTest::test_calculate_target_distribution()
{
    message += "test_calculate_target_distribution\n";

}


void DataSetTest::test_unuse_most_populated_target()
{
    message += "test_unused_most_populated_target\n";

    DataSet ds;

    Vector<size_t> unused_instances_indices;

    // Test

    ds.set(5,2,5);
    ds.initialize_data(0.0);

    unused_instances_indices = ds.unuse_most_populated_target(7);

    assert_true(unused_instances_indices.size() == 5, LOG);
    assert_true(ds.get_instances().get_used_instances_number() == 0, LOG);
    assert_true(ds.get_instances().get_unused_instances_number() == 5, LOG);

    // Test

    DataSet ds2;

    ds2.set(100, 7,5);
    ds2.initialize_data(1.0);

    unused_instances_indices = ds2.unuse_most_populated_target(99);

    assert_true(unused_instances_indices.size() == 99, LOG);
    assert_true(ds2.get_instances().get_used_instances_number() == 1, LOG);
    assert_true(ds2.get_instances().get_unused_instances_number() == 99, LOG);

    // Test

    DataSet ds3;

    ds3.set(1, 10,10);
    ds3.randomize_data_normal();

    unused_instances_indices = ds3.unuse_most_populated_target(50);

    assert_true(unused_instances_indices.size() == 1, LOG);
    assert_true(ds3.get_instances().get_used_instances_number() == 0, LOG);
    assert_true(ds3.get_instances().get_unused_instances_number() == 1, LOG);
}

void DataSetTest::test_balance_binary_targets_distribution()
{
    message += "test_balance_binary_target_distribution\n";

    // Test

    DataSet ds(22, 2, 1);
    ds.initialize_data(1.0);

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

    ds.set_instance(0, instance0);
    ds.set_instance(1, instance1);
    ds.set_instance(2, instance2);
    ds.set_instance(3, instance3);
    ds.set_instance(4, instance4);

    ds.balance_binary_targets_distribution();

    assert_true(ds.get_instances().get_unused_instances_number() == 12, LOG);
    assert_true(ds.calculate_target_distribution()[1] == 5, LOG);
    assert_true(ds.calculate_target_distribution()[0] == 5, LOG);

    // Test

    {
    DataSet ds(10, 3, 1);

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

    ds.set_instance(0, instance0);
    ds.set_instance(1, instance1);
    ds.set_instance(2, instance2);
    ds.set_instance(3, instance3);
    ds.set_instance(4, instance4);
    ds.set_instance(5, instance5);
    ds.set_instance(6, instance6);
    ds.set_instance(7, instance7);
    ds.set_instance(8, instance8);
    ds.set_instance(9, instance9);

    ds.balance_binary_targets_distribution();

    assert_true(ds.get_instances().get_unused_instances_number() == 10, LOG);
    assert_true(ds.calculate_target_distribution()[1] == 0, LOG);
    assert_true(ds.calculate_target_distribution()[0] == 0, LOG);
    }

    // Test

    {
    DataSet ds(10, 3, 1);

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

    ds.set_instance(0, instance0);
    ds.set_instance(1, instance1);
    ds.set_instance(2, instance2);
    ds.set_instance(3, instance3);
    ds.set_instance(4, instance4);
    ds.set_instance(5, instance5);
    ds.set_instance(6, instance6);
    ds.set_instance(7, instance7);
    ds.set_instance(8, instance8);
    ds.set_instance(9, instance9);

    ds.balance_binary_targets_distribution(50.0);

    assert_true(ds.get_instances().get_unused_instances_number() == 3, LOG);
    assert_true(ds.calculate_target_distribution()[1] == 2, LOG);
    assert_true(ds.calculate_target_distribution()[0] == 5, LOG);
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
    DataSet ds(4,1,1);

    const string data_file_name = "../data/data.dat";

    ds.set_data_file_name(data_file_name);
    ds.set_header_line(false);
    ds.set_separator("Comma");
    ds.set_file_type("dat");
    ds.set_missing_values_label("NaN");

    const string data_string =
    "5.1,3.5,1.0\n"
    "7.0,3.2,NaN\n"
    "7.0,3.2,0.0\n"
    "6.3,3.3,0.0";

    ofstream file;

    file.open(data_file_name.c_str());
    file << data_string;
    file.close();

    ds.load_data();

    ds.impute_missing_values_unuse();

    ds.balance_binary_targets_distribution();

    Vector<size_t> target_distribution = ds.calculate_target_distribution();

    assert_true(target_distribution[0] == target_distribution[1], LOG);
    assert_true(ds.get_instances().get_used_indices().size() == 2, LOG);
    assert_true(ds.get_instances().get_unused_indices().size() == 2, LOG);
    }

    //Test
    {
    DataSet ds(16,1,1);

    const string data_file_name = "../data/data.dat";

    ds.set_data_file_name(data_file_name);
    ds.set_header_line(false);
    ds.set_separator("Comma");
    ds.set_file_type("dat");
    ds.set_missing_values_label("NaN");

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

    ds.load_data();

    ds.impute_missing_values_unuse();

    ds.balance_binary_targets_distribution();

    Vector<size_t> target_distribution = ds.calculate_target_distribution();

    assert_true(target_distribution[0] == target_distribution[1], LOG);
    }
}


void DataSetTest::test_balance_multiple_targets_distribution()
{
    message += "test_balance_multiple_target_distribution\n";

    DataSet ds(9, 2, 2);

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

    ds.set_instance(0, instance0);
    ds.set_instance(1, instance1);
    ds.set_instance(2, instance2);
    ds.set_instance(3, instance3);
    ds.set_instance(4, instance4);
    ds.set_instance(5, instance5);
    ds.set_instance(6, instance6);
    ds.set_instance(7, instance7);
    ds.set_instance(8, instance8);

    ds.balance_multiple_targets_distribution();

    assert_true(ds.get_instances().get_unused_instances_number() == 9, LOG);
    assert_true(ds.calculate_target_distribution()[0] == 0, LOG);
    assert_true(ds.calculate_target_distribution()[1] == 0, LOG);

}


void DataSetTest::test_balance_function_regression_targets_distribution()
{
    message += "test_balance_function_regression_targets_distribution.\n";

    DataSet ds;

    Vector<size_t> unused_instances;

    // Test

    ds.set(10, 3, 1);

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

    ds.set_instance(0, instance0);
    ds.set_instance(1, instance1);
    ds.set_instance(2, instance2);
    ds.set_instance(3, instance3);
    ds.set_instance(4, instance4);
    ds.set_instance(5, instance5);
    ds.set_instance(6, instance6);
    ds.set_instance(7, instance7);
    ds.set_instance(8, instance8);
    ds.set_instance(9, instance9);

    unused_instances = ds.balance_approximation_targets_distribution(10.0);

    assert_true(ds.get_instances().get_unused_instances_number() == 1, LOG);
    assert_true(ds.get_instances().get_used_instances_number() == 9, LOG);
    assert_true(unused_instances.size() == 1, LOG);

    // Test

    DataSet ds2;
    ds2.set(1000, 5, 10);
    ds2.randomize_data_normal();

    unused_instances = ds2.balance_approximation_targets_distribution(100.0);

    assert_true(ds2.get_instances().get_used_instances_number() == 0, LOG);
    assert_true(ds2.get_instances().get_unused_instances_number() == 1000, LOG);
    assert_true(unused_instances.size() == 1000, LOG);
}

/*
void DataSetTest::test_calculate_instances_distances()
{
    message += "test_calculate_instances_distances\n";

    DataSet ds(5, 5, 2);
    ds.randomize_data_normal();

    Matrix<double> distances;

    //distances = ds.calculate_instances_distances();

    assert_true(distances(0, 0) == 0, LOG);
    assert_true(distances(1, 1) == 0, LOG);
    assert_true(distances(0, 1) == distances(1, 0), LOG);
}


void DataSetTest::test_calculate_k_distances()
{
    message += "test_calculate_k_distances\n";

    DataSet ds(5, 5, 10);
    ds.randomize_data_normal();

    size_t nearest_neighbors_number = 2;

    Vector<double> k_distances;

    Matrix<double> distances(10, 10);
    distances.randomize_uniform(0, 100);

    for(size_t i = 0; i < 10; i++)
    {
        distances(i, i) = 0;
    }

    k_distances = ds.calculate_k_distances(distances, nearest_neighbors_number);

    assert_true(k_distances.size() == 10, LOG);
    assert_true(k_distances[0] == distances(0, distances.get_row(0).calculate_minimal_indices(nearest_neighbors_number + 1)[nearest_neighbors_number]), LOG);
    assert_true(k_distances[9] == distances(9, distances.get_row(9).calculate_minimal_indices(nearest_neighbors_number + 1)[nearest_neighbors_number]), LOG);
}


void DataSetTest::test_calculate_reachability_distances()
{
    message += "test_calculate_reachability_distances\n";

    DataSet ds(5, 5, 10);
    ds.randomize_data_normal();

    Matrix<double> distances(10, 10);
    Vector<double> k_distances(10);

    Matrix<double> reachability_distances;

    // Test

    distances.initialize(1.0);
    k_distances.initialize(3.0);

    reachability_distances = ds.calculate_reachability_distances(distances, k_distances);

    assert_true(reachability_distances.get_columns_number() == 10, LOG);
    assert_true(reachability_distances.get_rows_number() == 10, LOG);
    assert_true(reachability_distances.get_row(0).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(1).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(2).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(3).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(4).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(5).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(6).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(7).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(8).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(9).is_in(3.0, 3.0), LOG);

    // Test

    distances.initialize(3.0);
    k_distances.initialize(1.0);

    reachability_distances = ds.calculate_reachability_distances(distances, k_distances);

    assert_true(reachability_distances.get_columns_number() == 10, LOG);
    assert_true(reachability_distances.get_rows_number() == 10, LOG);
    assert_true(reachability_distances.get_row(0).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(1).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(2).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(3).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(4).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(5).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(6).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(7).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(8).is_in(3.0, 3.0), LOG);
    assert_true(reachability_distances.get_row(9).is_in(3.0, 3.0), LOG);
}


void DataSetTest::test_calculate_reachability_density()
{
    message += "test_calculate_reachability_density\n";

    DataSet ds(5, 5, 10);
    ds.randomize_data_normal();

    size_t nearest_neighbors_number = 2;

    Matrix<double> distances(10, 10, 1.0);

    Vector<double> reachability_density;

    reachability_density = ds.calculate_reachability_density(distances, nearest_neighbors_number);

    assert_true(reachability_density.size() == 10, LOG);
    assert_true(reachability_density.is_in(1.0, 1.0), LOG);
}


void DataSetTest::test_calculate_local_outlier_factor()
{
    message += "test_calculate_local_outlier_factor\n";

    DataSet ds(2, 1, 100);
    ds.generate_artificial_data(100, 3);

    Vector<double> instance1(3, 1.0);
    instance1[2] = 50.0;

    Vector<double> instance2(3, 1.0);
    instance2[2] = 5.0;

    Vector<double> instance3(3);

    instance3[0] = 1.0;
    instance3[1] = 1.0;
    instance3[2] = 0.0;

    Vector<double> instance4(3, 1.0);
    instance4[2] = -10.0;

    ds.set_instance(96, instance1);
    ds.set_instance(97, instance2);
    ds.set_instance(98, instance3);
    ds.set_instance(99, instance4);

    ds.scale_data_minimum_maximum();

    Vector<double> local_outlier_factor = ds.calculate_local_outlier_factor(8);

    assert_true(local_outlier_factor.size() == 100, LOG);
}


void DataSetTest::test_clean_local_outlier_factor()
{
    message += "test_clean_local_outlier_factor\n";

    DataSet ds(5, 1, 100);
    ds.randomize_data_uniform(1.0, 2.0);

    Vector<double> instance(6);

    instance[0] = 1.0;
    instance[1] = 1.9;
    instance[2] = 2.5;
    instance[3] = 1.1;
    instance[4] = 1.8;

    ds.set_instance(9, instance);

    Vector<size_t> unused_instances;

    unused_instances = ds.clean_local_outlier_factor(5);

    assert_true(ds.get_instances().get_unused_instances_number() == 1, LOG);
    assert_true(unused_instances.size() == 1, LOG);
    assert_true(unused_instances[0] == 9, LOG);
}
*/

void DataSetTest::test_clean_Tukey_outliers()
{
    message += "test_clean_Tukey_outliers\n";

    DataSet ds(100, 5, 1);
    ds.randomize_data_uniform(1.0, 2.0);

    Vector<double> instance(6);

    instance[0] = 1.0;
    instance[1] = 1.9;
    instance[2] = 10.0;
    instance[3] = 1.1;
    instance[4] = 1.8;

    ds.set_instance(9, instance);

    const Vector< Vector<size_t> > outliers_indices = ds.calculate_Tukey_outliers(1.5);

    const Vector<size_t> outliers_instances = outliers_indices[0].calculate_greater_than_indices(0);
    size_t outliers_number = outliers_instances.size();

    ds.get_instances_pointer()->set_unused(outliers_instances);

    assert_true(ds.get_instances().get_unused_instances_number() == 1, LOG);
    assert_true(outliers_number == 1, LOG);
    assert_true(outliers_instances[0] == 9, LOG);
}


void DataSetTest::test_generate_data_function_regression()
{
    message += "test_generate_data_function_regression\n";

}


void DataSetTest::test_generate_data_binary_classification()
{
    message += "test_generate_data_binary_classification\n";

    DataSet ds;

    Vector<size_t> target_distribution;

    // Test

    ds.generate_data_binary_classification(2, 1);

    target_distribution = ds.calculate_target_distribution();

    assert_true(target_distribution.size() == 2, LOG);
    assert_true(target_distribution[0] == 1, LOG);
    assert_true(target_distribution[1] == 1, LOG);
}


void DataSetTest::test_generate_data_multiple_classification()
{
    message += "test_generate_data_multiple_classification\n";

}


void DataSetTest::test_to_XML() 
{
   message += "test_to_XML\n";

   DataSet ds;

   tinyxml2::XMLDocument* document;

   // Test

   document = ds.to_XML();

   assert_true(document != nullptr, LOG);
}


void DataSetTest::test_from_XML() 
{
   message += "test_from_XML\n";

   DataSet ds;

   Variables* v = ds.get_variables_pointer();
   Instances* i = ds.get_instances_pointer();

   tinyxml2::XMLDocument* document;
   
   // Test

   document = ds.to_XML();

   ds.from_XML(*document);

   // Test

   ds.set(2, 2);

   v->set_use(0, Variables::Target);
   v->set_use(1, Variables::Unused);

   i->set_use(0, Instances::Unused);
   i->set_use(1, Instances::Testing);

   document = ds.to_XML();

   ds.set();

   ds.from_XML(*document);

   assert_true(v->get_variables_number() == 2, LOG);
   assert_true(v->get_use(0) == Variables::Target, LOG);
   assert_true(v->get_use(1) == Variables::Unused, LOG);
   assert_true(i->get_instances_number() == 2, LOG);
   assert_true(i->get_use(0) == Instances::Unused, LOG);
   assert_true(i->get_use(1) == Instances::Testing, LOG);
}


void DataSetTest::test_print() 
{
   message += "test_print\n";

   DataSet ds;

   ds.set_display(false);

//   ds.print();
}


void DataSetTest::test_save() 
{
   message += "test_save\n";

   string file_name = "../data/data_set.xml";

   DataSet ds;

   ds.set_display(false);

   ds.save(file_name);
}


void DataSetTest::test_load() 
{
   message += "test_load\n";

   string file_name = "../data/data_set.xml";
   string data_file_name = "../data/data.dat";

   DataSet ds;
   DataSet ds_copy;

   Matrix<double> data;

   // Test

   ds.set();

   ds.save(file_name);
   ds.load(file_name);

   // Test;

   ds.set();

   data.set(1, 2, 0.0);

   data.save(data_file_name);

   ds.set(2, 1);

   ds.set_data_file_name(data_file_name);

   ds.get_variables_pointer()->set_name(0, "x");
   ds.get_variables_pointer()->set_units(0, "[m]");
   ds.get_variables_pointer()->set_description(0, "distance");

   ds.get_variables_pointer()->set_name(1, "y");
   ds.get_variables_pointer()->set_units(1, "[s]");
   ds.get_variables_pointer()->set_description(1, "time");

   ds.save(file_name);
   ds_copy.load(file_name);

   assert_true(ds_copy.get_variables().get_variables_number() == 2, LOG);
   assert_true(ds_copy.get_instances().get_instances_number() == 1, LOG);

   assert_true(ds_copy.get_variables_pointer()->get_name(0) == "x", LOG);
   assert_true(ds_copy.get_variables_pointer()->get_unit(0) == "[m]", LOG);
   assert_true(ds_copy.get_variables_pointer()->get_description(0) == "distance", LOG);

   assert_true(ds_copy.get_variables_pointer()->get_name(1) == "y", LOG);
   assert_true(ds_copy.get_variables_pointer()->get_unit(1) == "[s]", LOG);
   assert_true(ds_copy.get_variables_pointer()->get_description(1) == "time", LOG);
}


void DataSetTest::test_print_data()
{
   message += "test_print_data\n";
}


void DataSetTest::test_save_data()
{
   message += "test_save_data\n";

   string data_file_name = "../data/data.dat";

   DataSet ds(2,2,2);

   ds.set_data_file_name(data_file_name);

   ds.set_display(false);

   ds.save_data();
}


void DataSetTest::test_load_data() 
{
   message += "test_load_data\n";

   const string data_file_name = "../data/data.dat";

   ofstream file;

   DataSet ds;

   Variables* variables_pointer;

   ds.set_data_file_name(data_file_name);

   Matrix<double> data;

   string data_string;

   // Test

   ds.set();
   ds.set_data_file_name(data_file_name);
   ds.set_file_type("dat");

   ds.set_display(false);

   ds.save_data();

   ds.load_data();

   data = ds.get_data();

   assert_true(data.empty(), LOG);

   // Test

   ds.set(2, 2, 2);
   ds.set_data_file_name(data_file_name);
   ds.set_file_type("dat");

   ds.initialize_data(0.0);

   ds.set_display(false);

   ds.save_data();

   ds.load_data();

   data = ds.get_data();

   assert_true(data == 0.0, LOG);

   // Test

   ds.set_separator("Space");
   ds.set_file_type("dat");

   data_string = "\n\t\n   1 \t 2   \n\n\n   3 \t 4   \n\t\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   ds.load_data();

   data = ds.get_data();

   assert_true(data.get_rows_number() == 2, LOG);
   assert_true(data.get_columns_number() == 2, LOG);

   assert_true(data(0,0) == 1, LOG);
   assert_true(data(0,1) == 2, LOG);
   assert_true(data(1,0) == 3, LOG);
   assert_true(data(1,1) == 4, LOG);

   assert_true(ds.get_instances().get_instances_number() == 2, LOG);
   assert_true(ds.get_variables().get_variables_number() == 2, LOG);

   // Test

   ds.set_separator("Tab");
   ds.set_file_type("dat");

   data_string = "\n\n\n1 \t 2\n3 \t 4\n\n\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   ds.load_data();

   data = ds.get_data();

   assert_true(data(0,0) == 1, LOG);
   assert_true(data(0,1) == 2, LOG);
   assert_true(data(1,0) == 3, LOG);
   assert_true(data(1,1) == 4, LOG);

   // Test

   ds.set_header_line(true);
   ds.set_separator("Space");
   ds.set_file_type("dat");

   data_string = "\n"
                 "x y\n"
                 "\n"
                 "1   2\n"
                 "3   4\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   ds.load_data();

   data = ds.get_data();

   assert_true(ds.get_header_line() == true, LOG);
   assert_true(ds.get_variables_pointer()->get_name(0) == "x", LOG);
   assert_true(ds.get_variables_pointer()->get_name(1) == "y", LOG);

   assert_true(data.get_rows_number() == 2, LOG);
   assert_true(data.get_columns_number() == 2, LOG);

   assert_true(data(0,0) == 1, LOG);
   assert_true(data(0,1) == 2, LOG);
   assert_true(data(1,0) == 3, LOG);
   assert_true(data(1,1) == 4, LOG);

   // Test

   ds.set_header_line(true);
   ds.set_separator("Comma");
   ds.set_file_type("dat");

   data_string = "\tx \t ,\t y \n"
                 "\t1 \t, \t 2 \n"
                 "\t3 \t, \t 4 \n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   ds.load_data();

   data = ds.get_data();

   assert_true(ds.get_variables_pointer()->get_name(0) == "x", LOG);
   assert_true(ds.get_variables_pointer()->get_name(1) == "y", LOG);

   assert_true(data(0,0) == 1, LOG);
   assert_true(data(0,1) == 2, LOG);
   assert_true(data(1,0) == 3, LOG);
   assert_true(data(1,1) == 4, LOG);

   // Test

   ds.set_header_line(true);
   ds.set_separator("Comma");
   ds.set_file_type("dat");

   data_string = "x , y\n"
                 "1 , 2\n"
                 "3 , 4\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   ds.load_data();

   data = ds.get_data();

   assert_true(ds.get_variables_pointer()->get_name(0) == "x", LOG);
   assert_true(ds.get_variables_pointer()->get_name(1) == "y", LOG);

   assert_true(data(0,0) == 1, LOG);
   assert_true(data(0,1) == 2, LOG);
   assert_true(data(1,0) == 3, LOG);
   assert_true(data(1,1) == 4, LOG);

   // Test

   ds.set_header_line(false);
   ds.set_separator("Comma");
   ds.set_file_type("dat");

   data_string =
   "5.1,3.5,1.4,0.2,Iris-setosa\n"
   "7.0,3.2,4.7,1.4,Iris-versicolor\n"
   "7.0,3.2,4.7,1.4,Iris-versicolor\n"
   "6.3,3.3,6.0,2.5,Iris-virginica";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   ds.load_data();

   assert_true(ds.get_instances().get_instances_number() == 4, LOG);
   assert_true(ds.get_variables().get_variables_number() == 7, LOG);

   // Test

   ds.set_header_line(false);
   ds.set_separator("Comma");
   ds.set_file_type("dat");

   data_string =
   "5.1,3.5,1.4,0.2,Iris-setosa\n"
   "7.0,3.2,4.7,1.4,Iris-versicolor\n"
   "7.0,3.2,4.7,1.4,Iris-versicolor\n"
   "6.3,3.3,6.0,2.5,Iris-virginica\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   ds.load_data();

   assert_true(ds.get_variables_pointer()->get_variables_number() == 7, LOG);
   assert_true(ds.get_variables_pointer()->get_inputs_number() == 4, LOG);
   assert_true(ds.get_variables_pointer()->get_targets_number() == 3, LOG);
   assert_true(ds.get_instances_pointer()->get_instances_number() == 4, LOG);

   data = ds.get_data();

   assert_true(data(0,0) == 5.1, LOG);
   assert_true(data(0,4) == 1, LOG);
   assert_true(data(0,5) == 0, LOG);
   assert_true(data(0,6) == 0, LOG);
   assert_true(data(1,4) == 0, LOG);
   assert_true(data(1,5) == 1, LOG);
   assert_true(data(1,6) == 0, LOG);
   assert_true(data(2,4) == 0, LOG);
   assert_true(data(2,5) == 1, LOG);
   assert_true(data(2,6) == 0, LOG);
   assert_true(data(3,4) == 0, LOG);
   assert_true(data(3,5) == 0, LOG);
   assert_true(data(3,6) == 1, LOG);

   // Test

   ds.set_header_line(true);
   ds.set_separator("Comma");
   ds.set_missing_values_label("NaN");
   ds.set_file_type("dat");

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

   ds.load_data();

   assert_true(ds.get_variables_pointer()->get_variables_number() == 7, LOG);
   assert_true(ds.get_variables_pointer()->get_inputs_number() == 4, LOG);
   assert_true(ds.get_variables_pointer()->get_targets_number() == 3, LOG);

   assert_true(ds.get_variables_pointer()->get_name(0) == "sepal length", LOG);
   assert_true(ds.get_variables_pointer()->get_name(1) == "sepal width", LOG);
   assert_true(ds.get_variables_pointer()->get_name(2) == "petal length", LOG);
   assert_true(ds.get_variables_pointer()->get_name(3) == "petal width", LOG);
   assert_true(ds.get_variables_pointer()->get_name(4) == "Iris-setosa", LOG);
   assert_true(ds.get_variables_pointer()->get_name(5) == "Iris-versicolor", LOG);
   assert_true(ds.get_variables_pointer()->get_name(6) == "Iris-virginica", LOG);

   assert_true(ds.get_instances_pointer()->get_instances_number() == 5, LOG);

   assert_true(ds.get_missing_values().get_missing_values_number() == 4, LOG);
   assert_true(ds.get_missing_values().get_item(0).instance_index == 0, LOG);
   assert_true(ds.get_missing_values().get_item(0).variable_index == 0, LOG);

   assert_true(ds.get_missing_values().get_item(1).instance_index == 4, LOG);
   assert_true(ds.get_missing_values().get_item(1).variable_index == 4, LOG);
   assert_true(ds.get_missing_values().get_item(2).instance_index == 4, LOG);
   assert_true(ds.get_missing_values().get_item(2).variable_index == 5, LOG);
   assert_true(ds.get_missing_values().get_item(3).instance_index == 4, LOG);
   assert_true(ds.get_missing_values().get_item(3).variable_index == 6, LOG);

   data = ds.get_data();

   assert_true(data(0,4) == 1, LOG);
   assert_true(data(0,5) == 0, LOG);
   assert_true(data(0,6) == 0, LOG);
   assert_true(data(1,4) == 0, LOG);
   assert_true(data(1,5) == 1, LOG);
   assert_true(data(1,6) == 0, LOG);
   assert_true(data(2,4) == 0, LOG);
   assert_true(data(2,5) == 1, LOG);
   assert_true(data(2,6) == 0, LOG);

   // Test

   ds.set_header_line(false);
   ds.set_separator("Comma");
   ds.set_missing_values_label("NaN");
   ds.set_file_type("dat");

   data_string =
   "0,0,0\n"
   "0,0,NaN\n"
   "0,0,0\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   ds.load_data();

   assert_true(ds.get_missing_values().count_missing_instances() == 1, LOG);
   assert_true(ds.get_missing_values().get_item(0).instance_index == 1, LOG);
   assert_true(ds.get_missing_values().get_item(0).variable_index == 2, LOG);

   // Test

   ds.set_separator("Space");
   ds.set_file_type("dat");

   data_string = "1 2\n3 4\n5 6\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   ds.load_data();

   variables_pointer = ds.get_variables_pointer();

   variables_pointer->set_name(0, "x");
   variables_pointer->set_name(1, "y");

   ds.save("../data/data_set.xml");

   ds.load("../data/data_set.xml");

   assert_true(ds.get_variables().get_name(0) == "x", LOG);
   assert_true(ds.get_variables().get_name(1) == "y", LOG);

   // Test

   ds.set_header_line(false);
   ds.set_separator("Space");
   ds.set_file_type("dat");

   data_string = "1 true\n"
                 "3 false\n"
                 "5 true\n";

   file.open(data_file_name.c_str());
   file << data_string;
   file.close();

   ds.load_data();

   assert_true(ds.get_variables_pointer()->get_variables_number() == 2, LOG);
   assert_true(ds.get_variables_pointer()->get_inputs_number() == 1, LOG);
   assert_true(ds.get_variables_pointer()->get_targets_number() == 1, LOG);

   data = ds.get_data();

   assert_true(data(0,1) == 1, LOG);
   assert_true(data(1,1) == 0, LOG);
   assert_true(data(2,1) == 1, LOG);

   // Test

   ds.set_separator("Tab");
   ds.set_missing_values_label("NaN");
   ds.set_file_type("dat");

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

   ds.load_data();

   assert_true(ds.get_variables_pointer()->get_variables_number() == 7, LOG);
   assert_true(ds.get_variables_pointer()->get_inputs_number() == 6, LOG);
   assert_true(ds.get_variables_pointer()->get_targets_number() == 1, LOG);

   assert_true(ds.get_missing_values_pointer()->get_missing_values_number() == 2, LOG);

   data = ds.get_data();

   assert_true(data.get_rows_number() == 10, LOG);
   assert_true(data.get_columns_number() == 7, LOG);

}


void DataSetTest::test_get_data_statistics()
{
   message += "test_get_data_statistics\n";

   DataSet ds(1,1,1);
}


void DataSetTest::test_print_data_statistics()
{
   message += "test_print_data_statistics\n";
}


void DataSetTest::test_get_training_instances_statistics()
{
   message += "test_get_training_instances_statistics\n";

}


void DataSetTest::test_save_training_instances_statistics()
{
   message += "test_save_training_instances_statistics\n";
}


void DataSetTest::test_print_training_instances_statistics()
{
   message += "test_print_training_instances_statistics\n";
}


void DataSetTest::test_get_selection_instances_statistics()
{
   message += "test_get_selection_instances_statistics\n";
}


void DataSetTest::test_save_selection_instances_statistics()
{
   message += "test_save_selection_instances_statistics\n";
}


void DataSetTest::test_print_selection_instances_statistics()
{
   message += "test_print_selection_instances_statistics\n";
}


void DataSetTest::test_get_testing_instances_statistics()
{
   message += "test_get_testing_instances_statistics\n";
}


void DataSetTest::test_save_testing_instances_statistics()
{
   message += "test_save_testing_instances_statistics\n";
}


void DataSetTest::test_print_testing_instances_statistics()
{
   message += "test_print_testing_instances_statistics\n";
}


void DataSetTest::test_get_instances_statistics()
{
   message += "test_get_instances_statistics\n";
}


void DataSetTest::test_save_instances_statistics()
{
   message += "test_save_instances_statistics\n";
}


void DataSetTest::test_print_instances_statistics()
{
   message += "test_print_instances_statistics\n";
}


// @todo Complete method and tests.

void DataSetTest::test_convert_time_series()
{
   message += "test_convert_time_series\n";

   DataSet ds;

   Matrix<double> data;

   // Test

   data.set(2, 2, 3.1416);

   ds.set_data(data);

   ds.get_variables_pointer()->set_name(0, "x");
   ds.get_variables_pointer()->set_name(1, "y");

   ds.set_lags_number(1);

   ds.convert_time_series();

   data = ds.get_data();

   assert_true(data.get_rows_number() == 1, LOG);
   assert_true(data.get_columns_number() == 4, LOG);

   assert_true(ds.get_instances().get_instances_number() == 1, LOG);
   assert_true(ds.get_variables().get_variables_number() == 4, LOG);

   assert_true(ds.get_variables().get_inputs_number() == 2, LOG);
   assert_true(ds.get_variables().get_targets_number() == 2, LOG);

//   assert_true(ds.get_variables().get_name(0) == "x", LOG);
//   assert_true(ds.get_variables().get_name(1) == "y", LOG);
//   assert_true(ds.get_variables().get_name(2) == "lag_1_x", LOG);
//   assert_true(ds.get_variables().get_name(3) == "lag_1_y", LOG);
}


void DataSetTest::test_convert_autoassociation()
{
   message += "test_convert_autoassociation\n";

   DataSet ds;

   Matrix<double> data;

   // Test

   data.set(2, 2, 3.1416);

   ds.set_data(data);

   ds.get_variables_pointer()->set_name(0, "x");
   ds.get_variables_pointer()->set_name(1, "y");

   ds.set_autoassociation(true);

   ds.convert_association();

   data = ds.get_data();

   assert_true(data.get_rows_number() == 2, LOG);
   assert_true(data.get_columns_number() == 4, LOG);

   assert_true(ds.get_instances().get_instances_number() == 2, LOG);
   assert_true(ds.get_variables().get_variables_number() == 4, LOG);

   assert_true(ds.get_variables().get_inputs_number() == 2, LOG);
   assert_true(ds.get_variables().get_targets_number() == 2, LOG);

   assert_true(ds.get_variables().get_name(0) == "x", LOG);
   assert_true(ds.get_variables().get_name(1) == "y", LOG);
   assert_true(ds.get_variables().get_name(2) == "autoassociation_x", LOG);
   assert_true(ds.get_variables().get_name(3) == "autoassociation_y", LOG);
}


void DataSetTest::test_convert_angular_variable_degrees()
{
   message += "test_convert_angular_variable_degrees\n";
}


void DataSetTest::test_convert_angular_variable_radians()
{
   message += "test_convert_angular_variable_radians\n";
}


void DataSetTest::test_convert_angular_variables_degrees()
{
   message += "test_convert_angular_variables_degrees\n";

   DataSet ds;

   Matrix<double> data;

   Vector<size_t> angular_variables;

   // Test

   data.set(2,2, 1.234);

   ds.set(data);

   angular_variables.set(1, 0);

   ds.convert_angular_variables_degrees(angular_variables);

   assert_true(ds.get_variables().get_variables_number() == 3, LOG);
   assert_true(ds.get_data().get_column(0).calculate_absolute_value() <= 1.0, LOG);
   assert_true(ds.get_data().get_column(1).calculate_absolute_value() <= 1.0, LOG);
   assert_true(ds.get_data().get_column(2).calculate_absolute_value() == 1.234, LOG);

   // Test

   data.set(2,2, 1.234);

   ds.set(data);

   angular_variables.set(0,1,1);

   ds.convert_angular_variables_degrees(angular_variables);

   assert_true(ds.get_variables().get_variables_number() == 4, LOG);
   assert_true(ds.get_data().get_column(0).calculate_absolute_value() <= 1.0, LOG);
   assert_true(ds.get_data().get_column(1).calculate_absolute_value() <= 1.0, LOG);
   assert_true(ds.get_data().get_column(2).calculate_absolute_value() <= 1.0, LOG);
   assert_true(ds.get_data().get_column(3).calculate_absolute_value() <= 1.0, LOG);
}


void DataSetTest::test_convert_angular_variables_radians()
{
   message += "test_convert_angular_variables_radians\n";

   DataSet ds;

   Matrix<double> data;

   Vector<size_t> angular_variables;

   // Test

   data.set(2,2, 1.234);

   ds.set(data);

   angular_variables.set(1, 0);

   ds.convert_angular_variables_radians(angular_variables);

   assert_true(ds.get_variables().get_variables_number() == 3, LOG);
   assert_true(ds.get_data().get_column(0).calculate_absolute_value() <= 1.0, LOG);
   assert_true(ds.get_data().get_column(1).calculate_absolute_value() <= 1.0, LOG);
   assert_true(ds.get_data().get_column(2).calculate_absolute_value() == 1.234, LOG);

   // Test

   data.set(2,2, 1.234);

   ds.set(data);

   angular_variables.set(0,1,1);

   ds.convert_angular_variables_radians(angular_variables);

   assert_true(ds.get_variables().get_variables_number() == 4, LOG);
   assert_true(ds.get_data().get_column(0).calculate_absolute_value() <= 1.0, LOG);
   assert_true(ds.get_data().get_column(1).calculate_absolute_value() <= 1.0, LOG);
   assert_true(ds.get_data().get_column(2).calculate_absolute_value() <= 1.0, LOG);
   assert_true(ds.get_data().get_column(3).calculate_absolute_value() <= 1.0, LOG);
}


void DataSetTest::test_convert_angular_variables()
{
   message += "test_convert_angular_variables\n";
}


// @todo

void DataSetTest::test_scrub_missing_values()
{
    message += "test_scrub_missing_values\n";

    const string data_file_name = "../data/data.dat";

    ofstream file;

    DataSet ds;

    ds.set_data_file_name(data_file_name);

    Instances instances;

    MissingValues* mv = ds.get_missing_values_pointer();

    Matrix<double> data;

    string data_string;

    // Test

    ds.set_separator("Space");
    ds.set_missing_values_label("NaN");
    ds.set_file_type("dat");

    mv->set_scrubbing_method("Unuse");

    data_string = "0 0 0\n"
                  "0 0 NaN\n"
                  "0 0 0\n";

    file.open(data_file_name.c_str());
    file << data_string;
    file.close();

    ds.load_data();

    assert_true(ds.get_missing_values().count_missing_instances() == 1, LOG);
    assert_true(ds.get_missing_values().get_item(0).instance_index == 1, LOG);
    assert_true(ds.get_missing_values().get_item(0).variable_index == 2, LOG);

    ds.scrub_missing_values();

    instances = ds.get_instances();

    assert_true(instances.get_use(1) == Instances::Unused, LOG);

    // Test

    ds.set_separator("Space");
    ds.set_missing_values_label("NaN");
    ds.set_file_type("dat");

    data_string = "NaN 3   3\n"
                  "2   NaN 3\n"
                  "0   1   NaN\n";

    file.open(data_file_name.c_str());
    file << data_string;
    file.close();

    ds.load_data();

    mv->set_scrubbing_method("Mean");

    ds.scrub_missing_values();

    instances = ds.get_instances();

    data = ds.get_data();

    assert_true(fabs(data(0,0) - 1.0) < 1.0e-3, LOG);
    assert_true(fabs(data(1,1) - 2.0) < 1.0e-3, LOG);
    assert_true(fabs(data(2,2) - 3.0) < 1.0e-3, LOG);
}


void DataSetTest::test_impute_missing_values_time_series_mean()
{
    message += "test_impute_missing_values_time_series_mean\n";

    Matrix<double> data;
    string data_string;

    ofstream file;


    // Test 1

    DataSet dataset_test1;
    const string data_file_name_test1 = "../data/data_test1.dat";
    dataset_test1.set_separator("Space");
    dataset_test1.set_missing_values_label("-999.9");
    dataset_test1.set_data_file_name(data_file_name_test1);
    dataset_test1.set_file_type("dat");

    data_string = "1.0\n"
                  "-999.9\n"
                  "3.0\n";

    file.open(data_file_name_test1.c_str());
    file << data_string;
    file.close();

    dataset_test1.load_data();

    dataset_test1.impute_missing_values_time_series_mean();

    data = dataset_test1.get_data();

    assert_true(fabs(data(0,0) - 1.0) < 1.0e-3, LOG);
    assert_true(fabs(data(1,0) - 2.0) < 1.0e-3, LOG);
    assert_true(fabs(data(2,0) - 3.0) < 1.0e-3, LOG);


//    dataset_test1.set_data_file_name("I:/Patri/test_data/data_test1.dat");
//    dataset_test1.save_data();


    // Test 2

    DataSet dataset_test2;
    const string data_file_name_test2 = "../data/data_test2.dat";
    dataset_test2.set_separator("Space");
    dataset_test2.set_missing_values_label("-999.9");
    dataset_test2.set_data_file_name(data_file_name_test2);
    dataset_test2.set_file_type("dat");

    data_string = "1.0 4.0\n"
                  "-999.9 -999.9\n"
                  "3.0 6.0\n";

    file.open(data_file_name_test2.c_str());
    file << data_string;
    file.close();

    dataset_test2.load_data();

    dataset_test2.impute_missing_values_time_series_mean();

    data = dataset_test2.get_data();

    assert_true(fabs(data(0,0) - 1.0) < 1.0e-3, LOG);
    assert_true(fabs(data(1,0) - 2.0) < 1.0e-3, LOG);
    assert_true(fabs(data(2,0) - 3.0) < 1.0e-3, LOG);
    assert_true(fabs(data(0,1) - 4.0) < 1.0e-3, LOG);
    assert_true(fabs(data(1,1) - 5.0) < 1.0e-3, LOG);
    assert_true(fabs(data(2,1) - 6.0) < 1.0e-3, LOG);


//    dataset_test2.set_data_file_name("I:/Patri/test_data/data_test2.dat");
//    dataset_test2.save_data();


    // Test 3

    DataSet dataset_test3;
    const string data_file_name_test3 = "../data/data_test3.dat";
    dataset_test3.set_separator("Space");
    dataset_test3.set_missing_values_label("-999.9");
    dataset_test3.set_data_file_name(data_file_name_test3);
    dataset_test3.set_file_type("dat");

    data_string = "-999.9 4.0\n"
                  "2.0 5.0\n"
                  "3.0 -999.9\n";

    file.open(data_file_name_test3.c_str());
    file << data_string;
    file.close();

    dataset_test3.load_data();

    dataset_test3.impute_missing_values_time_series_mean();

    data = dataset_test3.get_data();

    assert_true(fabs(data(0,0) - 1.0) < 1.0e-3, LOG);
    assert_true(fabs(data(1,0) - 2.0) < 1.0e-3, LOG);
    assert_true(fabs(data(2,0) - 3.0) < 1.0e-3, LOG);
    assert_true(fabs(data(0,1) - 4.0) < 1.0e-3, LOG);
    assert_true(fabs(data(1,1) - 5.0) < 1.0e-3, LOG);
    assert_true(fabs(data(2,1) - 6.0) < 1.0e-3, LOG);


//    dataset_test3.set_data_file_name("I:/Patri/test_data/data_test3.dat");
//    dataset_test3.save_data();


    // Test 4

    DataSet dataset_test4;
    const string data_file_name_test4 = "../data/data_test4.dat";
    dataset_test4.set_separator("Space");
    dataset_test4.set_missing_values_label("-999.9");
    dataset_test4.set_data_file_name(data_file_name_test4);
    dataset_test4.set_file_type("dat");

    data_string = "1.0 5.0\n"
                  "-999.9 -999.9\n"
                  "-999.9 -999.9\n"
                  "4.0 8.0\n";

    file.open(data_file_name_test4.c_str());
    file << data_string;
    file.close();

    dataset_test4.load_data();

    dataset_test4.impute_missing_values_time_series_mean();

    data = dataset_test4.get_data();

    assert_true(fabs(data(0,0) - 1.0) < 1.0e-3, LOG);
    assert_true(fabs(data(1,0) - 2.0) < 1.0e-3, LOG);
    assert_true(fabs(data(2,0) - 3.0) < 1.0e-3, LOG);
    assert_true(fabs(data(3,0) - 4.0) < 1.0e-3, LOG);
    assert_true(fabs(data(0,1) - 5.0) < 1.0e-3, LOG);
    assert_true(fabs(data(1,1) - 6.0) < 1.0e-3, LOG);
    assert_true(fabs(data(2,1) - 7.0) < 1.0e-3, LOG);
    assert_true(fabs(data(3,1) - 8.0) < 1.0e-3, LOG);


//    dataset_test4.set_data_file_name("I:/Patri/test_data/data_test4.dat");
//    dataset_test4.save_data();


    // Test 5

    DataSet dataset_test5;
    const string data_file_name_test5 = "../data/data_test5.dat";
    dataset_test5.set_separator("Space");
    dataset_test5.set_missing_values_label("-999.9");
    dataset_test5.set_data_file_name(data_file_name_test5);
    dataset_test5.set_file_type("dat");

    data_string = "-999.9\n"
                  "-999.9\n"
                  "3.0\n"
                  "4.0\n";

    file.open(data_file_name_test5.c_str());
    file << data_string;
    file.close();

    dataset_test5.load_data();

    dataset_test5.impute_missing_values_time_series_mean();

    data = dataset_test5.get_data();

    assert_true(fabs(data(0,0) - 1.0) < 1.0e-3, LOG);
    assert_true(fabs(data(1,0) - 2.0) < 1.0e-3, LOG);
    assert_true(fabs(data(2,0) - 3.0) < 1.0e-3, LOG);
    assert_true(fabs(data(3,0) - 4.0) < 1.0e-3, LOG);


//    dataset_test5.set_data_file_name("I:/Patri/test_data/data_test5.dat");
//    dataset_test5.save_data();


    // Test 6

    DataSet dataset_test6;
    const string data_file_name_test6 = "../data/data_test6.dat";
    dataset_test6.set_separator("Space");
    dataset_test6.set_missing_values_label("-999.9");
    dataset_test6.set_data_file_name(data_file_name_test6);
    dataset_test6.set_file_type("dat");

    data_string = "-999.9 5.0\n"
                  "-999.9 6.0\n"
                  "3.0 -999.9\n"
                  "4.0 -999.9\n";

    file.open(data_file_name_test6.c_str());
    file << data_string;
    file.close();

    dataset_test6.load_data();

    dataset_test6.impute_missing_values_time_series_mean();

    data = dataset_test6.get_data();

    assert_true(fabs(data(0,0) - 1.0) < 1.0e-3, LOG);
    assert_true(fabs(data(1,0) - 2.0) < 1.0e-3, LOG);
    assert_true(fabs(data(2,0) - 3.0) < 1.0e-3, LOG);
    assert_true(fabs(data(3,0) - 4.0) < 1.0e-3, LOG);
    assert_true(fabs(data(0,1) - 5.0) < 1.0e-3, LOG);
    assert_true(fabs(data(1,1) - 6.0) < 1.0e-3, LOG);
    assert_true(fabs(data(2,1) - 7.0) < 1.0e-3, LOG);
    assert_true(fabs(data(3,1) - 8.0) < 1.0e-3, LOG);


//    dataset_test6.set_data_file_name("I:/Patri/test_data/data_test6.dat");
//    dataset_test6.save_data();
}



void DataSetTest::test_trim()
{
   message += "test_trim\n";

   DataSet ds;

   string str;

   // Test

   str.assign(" hello");

   ds.trim(str);

   assert_true(str.compare("hello") == 0, LOG);

   // Test

   str.assign("hello ");

   ds.trim(str);

   assert_true(str.compare("hello") == 0, LOG);

   // Test

   str.assign(" hello ");

   ds.trim(str);

   assert_true(str.compare("hello") == 0, LOG);

   // Test

   str.assign("   hello   ");

   ds.trim(str);

   assert_true(str.compare("hello") == 0, LOG);
}


void DataSetTest::test_get_trimmed()
{
   message += "test_get_trimmed\n";

   DataSet ds;

   string str1;
   string str2;

   // Test

   str1.assign(" hello");

   str2 = ds.get_trimmed(str1);

   assert_true(str2.compare("hello") == 0, LOG);

   // Test

   str1.assign("hello ");

   str2 = ds.get_trimmed(str1);

   assert_true(str2.compare("hello") == 0, LOG);

   // Test

   str1.assign(" hello ");

   str2 = ds.get_trimmed(str1);

   assert_true(str2.compare("hello") == 0, LOG);

   // Test

   str1.assign("   hello   ");

   str2 = ds.get_trimmed(str1);

   assert_true(str2.compare("hello") == 0, LOG);
}


void DataSetTest::test_count_tokens()
{
   message += "test_count_tokens\n";

   DataSet ds;

   string str;

   size_t tokens_number;

   // Test

   str.assign(" hello ");

   tokens_number = ds.count_tokens(str);

   assert_true(tokens_number == 1, LOG);

   // Test

   str.assign(" hello");

   tokens_number = ds.count_tokens(str);

   assert_true(tokens_number == 1, LOG);

   // Test

   str.assign(" hello bye ");

   tokens_number = ds.count_tokens(str);

   assert_true(tokens_number == 2, LOG);

   // Test

   str.assign(" hello   bye ");

   tokens_number = ds.count_tokens(str);

   assert_true(tokens_number == 2, LOG);

   // Test

   ds.set_separator("Comma");

   str.assign("1, 2, 3, 4");

   tokens_number = ds.count_tokens(str);

   assert_true(tokens_number == 4, LOG);

   // Test

   ds.set_separator("Comma");

   str.assign(",1, 2, 3, 4,");

   tokens_number = ds.count_tokens(str);

   assert_true(tokens_number == 4, LOG);

   // Test

   str.assign(",1,2,3,4,");

   tokens_number = ds.count_tokens(str);

   assert_true(tokens_number == 4, LOG);

   // Test

   str.assign(", 1 , 2 , 3 , 4, ");

   tokens_number = ds.count_tokens(str);

   assert_true(tokens_number == 4, LOG);

   // Test

   str.assign("5.1,3.5,1.4,0.2,Iris-setosa");

   tokens_number = ds.count_tokens(str);

   assert_true(tokens_number == 5, LOG);
}


void DataSetTest::test_get_tokens()
{
   message += "test_get_tokens\n";

   DataSet ds;

   string str;

   Vector<string> tokens;

   // Test

   str.assign(" hello ");

   tokens = ds.get_tokens(str);

   assert_true(tokens.size() == 1, LOG);
   assert_true(tokens[0].compare("hello") == 0, LOG);

   // Test

   str.assign(" hello");

   tokens = ds.get_tokens(str);

   assert_true(tokens.size() == 1, LOG);
   assert_true(tokens[0].compare("hello") == 0, LOG);

   // Test

   str.assign(" hello bye ");

   tokens = ds.get_tokens(str);

   assert_true(tokens.size() == 2, LOG);
   assert_true(tokens[0].compare("hello") == 0, LOG);
   assert_true(tokens[1].compare("bye") == 0, LOG);

   // Test

   ds.set_separator("Comma");

   str.assign("1,2,3,4");

   tokens = ds.get_tokens(str);

   assert_true(tokens.size() == 4, LOG);
   assert_true(tokens[0].compare("1") == 0, LOG);
   assert_true(tokens[1].compare("2") == 0, LOG);
   assert_true(tokens[2].compare("3") == 0, LOG);
   assert_true(tokens[3].compare("4") == 0, LOG);

   // Test

   ds.set_separator("Comma");

   str.assign(",1,2,3,4,");

   tokens = ds.get_tokens(str);

   assert_true(tokens.size() == 4, LOG);
   assert_true(tokens[0].compare("1") == 0, LOG);
   assert_true(tokens[1].compare("2") == 0, LOG);
   assert_true(tokens[2].compare("3") == 0, LOG);
   assert_true(tokens[3].compare("4") == 0, LOG);

   // Test

   ds.set_separator("Comma");

   str.assign(", 1 , 2 , 3 , 4 , ");

   tokens = ds.get_tokens(str);

   assert_true(tokens.size() == 4, LOG);

   assert_true(ds.get_trimmed(tokens[0]).compare("1") == 0, LOG);
   assert_true(ds.get_trimmed(tokens[1]).compare("2") == 0, LOG);
   assert_true(ds.get_trimmed(tokens[2]).compare("3") == 0, LOG);
   assert_true(ds.get_trimmed(tokens[3]).compare("4") == 0, LOG);

   // Test

   ds.set_separator("Comma");

   str.assign(", 1 , 2 , 3 , 4 , ");

   tokens = ds.get_tokens(str);

   assert_true(tokens.size() == 4, LOG);

   assert_true(ds.get_trimmed(tokens[0]).compare("1") == 0, LOG);
   assert_true(ds.get_trimmed(tokens[1]).compare("2") == 0, LOG);
   assert_true(ds.get_trimmed(tokens[2]).compare("3") == 0, LOG);
   assert_true(ds.get_trimmed(tokens[3]).compare("4") == 0, LOG);

}


void DataSetTest::test_is_numeric()
{
   message += "test_is_numeric\n";

   DataSet ds;

   string str;

   // Test

   str.assign("hello");

   assert_true(!ds.is_numeric(str), LOG);

   // Test

   str.assign("0");

   assert_true(ds.is_numeric(str), LOG);

   // Test

   str.assign("-1.0e-99");

   assert_true(ds.is_numeric(str), LOG);
}


void DataSetTest::run_test_case()
{
   message += "Running data set test case...\n";

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

   // Data methods

   test_get_data();

   test_get_training_data();
   test_get_selection_data();
   test_get_testing_data();

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

   test_add_instance();
   test_subtract_instance();

   test_subtract_constant_variables();
   test_subtract_repeated_instances();

   // Initialization methods

   test_initialize_data();

   // Statistics methods

   test_calculate_data_statistics();
   test_calculate_data_statistics_missing_values();

   test_calculate_training_instances_statistics();
   test_calculate_selection_instances_statistics();
   test_calculate_testing_instances_statistics();

   test_calculate_input_statistics();
   test_calculate_targets_statistics();

   // Correlation methods

   test_calculate_linear_correlations();

   // Histrogram methods

   test_calculate_data_histograms();

   // Filtering methods

   test_filter_data();

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

   // Input-target variables unscaling

   test_unscale_variables_mean_standard_deviation();
   test_unscale_variables_minimum_maximum();

   // Pattern recognition methods

   test_calculate_target_distribution();

   test_unuse_most_populated_target();

   test_balance_binary_targets_distribution();
   test_balance_multiple_targets_distribution();
   test_balance_function_regression_targets_distribution();

   // Outlier detection

   //test_calculate_instances_distances();
   //test_calculate_k_distances();
   //test_calculate_reachability_distances();
   //test_calculate_reachability_density();
   //test_calculate_local_outlier_factor();

   //test_clean_local_outlier_factor();
   test_clean_Tukey_outliers();

   // Data generation

   test_generate_data_function_regression();

   test_generate_data_binary_classification();
   test_generate_data_multiple_classification();

   // Serialization methods

//   test_to_XML();
//   test_from_XML();

//   test_print();
//   test_save();
//   test_load();

//   test_print_data();
//   test_save_data();

//   test_load_data();

//   test_get_data_statistics();
//   test_print_data_statistics();

//   test_get_training_instances_statistics();
//   test_print_training_instances_statistics();
//   test_save_training_instances_statistics();

//   test_get_selection_instances_statistics();
//   test_print_selection_instances_statistics();
//   test_save_selection_instances_statistics();

//   test_get_testing_instances_statistics();
//   test_print_testing_instances_statistics();
//   test_save_testing_instances_statistics();

//   test_get_instances_statistics();
//   test_print_instances_statistics();
//   test_save_instances_statistics();

//   test_convert_time_series();
//   test_convert_autoassociation();

//   test_convert_angular_variable_degrees();
//   test_convert_angular_variable_radians();

//   test_convert_angular_variables();

//   test_scrub_missing_values();

   test_impute_missing_values_time_series_mean();

   // String utilities

//   test_trim();
//   test_get_trimmed();

//   test_count_tokens();
//   test_get_tokens();

//   test_is_numeric();

   message += "End of data set test case.\n";
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
