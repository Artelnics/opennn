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

   TrainingStrategy training_strategy1(&neural_network, &data_set);


}


void TrainingStrategyTest::test_destructor()
{
   cout << "test_destructor\n";

   NeuralNetwork neural_network;
   DataSet data_set;

   TrainingStrategy* ts = new TrainingStrategy(&neural_network, &data_set);

   delete ts;

}


void TrainingStrategyTest::test_get_loss_index_pointer()
{
   cout << "test_get_loss_index_pointer\n";

   NeuralNetwork neural_network;
   DataSet data_set;

   SumSquaredError sum_squared_error;

   TrainingStrategy training_strategy(&neural_network, &data_set);

   LossIndex* loss_index_pointer = training_strategy.get_loss_index_pointer();

   assert_true(loss_index_pointer != nullptr, LOG);
}


void TrainingStrategyTest::test_get_display()
{
   cout << "test_get_display\n";

   NeuralNetwork neural_network;
   DataSet data_set;

   TrainingStrategy training_strategy(&neural_network, &data_set);

   training_strategy.set_display(false);

   assert_true(!training_strategy.get_display(), LOG);
}


void TrainingStrategyTest::test_set()
{
   cout << "test_set\n"; 
}


void TrainingStrategyTest::test_set_default()
{
   cout << "test_set_default\n"; 

   TrainingStrategy training_strategy;

   training_strategy.set_default();

}


void TrainingStrategyTest::test_set_loss_index_pointer()
{
   cout << "test_set_loss_index_pointer\n"; 
}


void TrainingStrategyTest::test_perform_training()
{
   cout << "test_perform_training\n";

    NeuralNetwork neural_network;
    Tensor<Index, 1> architecture(3);

    DataSet data_set(2, 2);

    Tensor<type, 2> new_data(2, 2);
    new_data(0,0) = 0.0;
    new_data(0,1) = 0.0;
    new_data(1,0) = 1.0;
    new_data(1,1) = 1.0;

    data_set.set_data(new_data);

    NormalizedSquaredError normalized_squared_error(&neural_network, &data_set);

    TrainingStrategy training_strategy(&neural_network, &data_set);

    // Test

    architecture.setValues({1, 1, 1});

    neural_network.set(NeuralNetwork::Approximation, architecture);
    training_strategy.set_optimization_method(TrainingStrategy::ADAPTIVE_MOMENT_ESTIMATION);
    neural_network.set_parameters_random();

    training_strategy.set_maximum_epochs_number(10);
    training_strategy.set_display(false);

    training_strategy.perform_training();
}


void TrainingStrategyTest::test_to_XML()
{
   cout << "test_to_XML\n";

   TrainingStrategy training_strategy;

   FILE *pFile;

   string file_name = "../data/training_strategy.xml";

   pFile = fopen(file_name.c_str(), "w");

   tinyxml2::XMLPrinter document(pFile);

   training_strategy.write_XML(document);

   fclose(pFile);

}


void TrainingStrategyTest::test_from_XML()
{
   cout << "test_from_XML\n";

   TrainingStrategy training_strategy;

   training_strategy.set_optimization_method(TrainingStrategy::GRADIENT_DESCENT);

   training_strategy.set_default();

   tinyxml2::XMLDocument document;

   string file_name = "../data/training_strategy.xml";

   if(document.LoadFile(file_name.c_str()))
   {
       ostringstream buffer;

       buffer << "OpenNN Exception: TrainingStrategy class.\n"
              << "void load(const string&) method.\n"
              << "Cannot load XML file " << file_name << ".\n";

       throw logic_error(buffer.str());
   }

   training_strategy.from_XML(document);

}


void TrainingStrategyTest::test_print()
{
   cout << "test_print\n";
}


void TrainingStrategyTest::test_save()
{
   cout << "test_save\n";

   string file_name = "../data/training_strategy.xml";

   TrainingStrategy training_strategy;

   training_strategy.set_optimization_method(TrainingStrategy::GRADIENT_DESCENT);

   training_strategy.save(file_name);
}


void TrainingStrategyTest::test_load()
{
   cout << "test_load\n";

   string file_name = "../data/training_strategy.xml";

   NeuralNetwork neural_network;
   DataSet data_set;

   TrainingStrategy training_strategy(&neural_network, &data_set);

   // Test

   training_strategy.save(file_name);
   training_strategy.load(file_name);
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

   cout << "End of training strategy test case.\n\n";
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2021 Artificial Intelligence Techniques, SL.
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
