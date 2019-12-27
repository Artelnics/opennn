//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   S T R A T E G Y   T E S T   C L A S S               
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "training_strategy_test.h"


TrainingStrategyTest::TrainingStrategyTest() : UnitTesting() 
{
}


TrainingStrategyTest::~TrainingStrategyTest()
{
}


void TrainingStrategyTest::test_constructor()
{
   cout << "test_constructor\n";

   NeuralNetwork neural_network;
   DataSet data_set;

   // Test

   TrainingStrategy ts1(&neural_network, &data_set);

   assert_true(ts1.has_loss_index() == true, LOG);

   // Test

   TrainingStrategy ts2;

   assert_true(ts2.has_loss_index() == true, LOG);
}


void TrainingStrategyTest::test_destructor()
{
   cout << "test_destructor\n";

   TrainingStrategy* ts = new TrainingStrategy();

   delete ts;
}


void TrainingStrategyTest::test_get_loss_index_pointer()
{
   cout << "test_get_loss_index_pointer\n";

   NeuralNetwork neural_network;
   DataSet data_set;

   SumSquaredError sum_squared_error;

   TrainingStrategy ts(&neural_network, &data_set);

   LossIndex* loss_index_pointer = ts.get_loss_index_pointer();

   assert_true(loss_index_pointer != nullptr, LOG);
}


void TrainingStrategyTest::test_get_display()
{
   cout << "test_get_display\n";

   NeuralNetwork nn;
   DataSet ds;

   TrainingStrategy training_strategy(&nn, &ds);

   training_strategy.set_display(false);

   assert_true(training_strategy.get_display() == false, LOG);
}


void TrainingStrategyTest::test_set()
{
   cout << "test_set\n"; 
}


void TrainingStrategyTest::test_set_default()
{
   cout << "test_set_default\n"; 
}


void TrainingStrategyTest::test_set_loss_index_pointer()
{
   cout << "test_set_loss_index_pointer\n"; 
}


void TrainingStrategyTest::test_perform_training()
{
   cout << "test_perform_training\n";

    NeuralNetwork neural_network;
    DataSet data_set;
    SumSquaredError sum_squared_error(&neural_network, &data_set);
    TrainingStrategy ts(&neural_network, &data_set);

    // Test

    neural_network.set(NeuralNetwork::Approximation, {1, 1});
    data_set.set(1,1,2);

//    ts.perform_training();

}

///@todo

void TrainingStrategyTest::test_to_XML()
{
   cout << "test_to_XML\n";
/*
   TrainingStrategy training_strategy;

   // Test

   ts.set_optimization_method(TrainingStrategy::GRADIENT_DESCENT);

   tinyxml2::XMLDocument* document = ts.to_XML();

   assert_true(document != nullptr, LOG);

   delete document;
*/
}


///@todo

void TrainingStrategyTest::test_from_XML()
{
   cout << "test_from_XML\n";
/*
   TrainingStrategy ts1;
   TrainingStrategy ts2;

   ts1.set_optimization_method(TrainingStrategy::GRADIENT_DESCENT);

   tinyxml2::XMLDocument* document = ts1.to_XML();

    ts2.from_XML(*document);

   delete document;

    assert_true(ts2.get_optimization_method() == TrainingStrategy::GRADIENT_DESCENT, LOG);
*/
}


void TrainingStrategyTest::test_print()
{
   cout << "test_print\n";
}


///@todo

void TrainingStrategyTest::test_save()
{
   cout << "test_save\n";
/*
   string file_name = "../data/training_strategy.xml";

   TrainingStrategy training_strategy;

   ts.set_optimization_method(TrainingStrategy::GRADIENT_DESCENT);

   ts.save(file_name);
*/
}


void TrainingStrategyTest::test_load()
{
   cout << "test_load\n";

   string file_name = "../data/training_strategy.xml";

   NeuralNetwork nn;
   DataSet ds;

   TrainingStrategy training_strategy(&nn, &ds);

   // Test

   training_strategy.save(file_name);
   training_strategy.load(file_name);
}


void TrainingStrategyTest::test_results_constructor()
{
    cout << "test_results_constructor\n";

//    TrainingStrategy::TrainingStrategyResults results;

}


void TrainingStrategyTest::test_results_destructor()
{
    cout << "test_results_destructor\n";

//    TrainingStrategy::TrainingStrategyResults* results = new TrainingStrategy::TrainingStrategyResults();

//    delete results;
}


void TrainingStrategyTest::run_test_case()
{
   cout << "Running training strategy test case...\n";

   // Constructor and destructor methods

   test_constructor();
   test_destructor();

   // Get methods

   test_get_loss_index_pointer();

   // Utilities
   
   test_get_display();

   // Set methods

   test_set();
   test_set_default();   

   test_set_loss_index_pointer();

   // Training methods

   test_perform_training();

   // Serialization methods

   test_to_XML();
   test_from_XML();

   test_print();
   test_save();
   test_load();

   // Results methods

   test_results_constructor();
   test_results_destructor();

   cout << "End of training strategy test case.\n";
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
