//   OpenNN: Open Neural Networks Library
//   www.opennn.org
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <stdint.h>
#include <limits.h>
#include <statistics.h>
#include <regex>
#include <iomanip>

// Systems Complementaries

#include <cmath>
#include <cstdlib>
#include <ostream>

// Qt includes

#include <QFileInfo>
#include <QCoreApplication>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>
#include <QString>
#include <QFile>
#include <QTextStream>
#include <QTextCodec>
#include <QVector>
#include <QDebug>
#include <QDir>

#include <QObject>

// OpenNN includes

#include "../opennn/opennn.h"
#include "device.h"

#include <../eigen/unsupported/Eigen/KroneckerProduct>

using namespace OpenNN;
using namespace std;
using namespace chrono;

using Eigen::MatrixXd;
using Eigen::Vector3d;

int main(void)
{
    try
    {
        cout << "Blank application" << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        DataSet data;





/*
        DataSet data_set("C:/Users/Usuario/Documents/rosenbrock_1000000_1000.csv", ',', true);

//



        // Data set

        Tensor<type, 2> data(samples, variables+1);

        data.setRandom();

        DataSet data_set(data);
*/

        Index samples = 1000000;
        Index variables = 1000;

         // Device

        Device device(Device::EigenSimpleThreadPool);

        DataSet data_set;

        data_set.generate_Rosenbrock_data(samples, variables+1);
/*
        Tensor<string, 1> uses(variables+1);
        uses.setValues({"Input", "Input", "Target", "Target"});

        data_set.set_columns_uses(uses);aa
*/
        data_set.set_device_pointer(&device);

        data_set.set_training();

        // Neural network

        const Index inputs_number = data_set.get_input_variables_number();

        const Index hidden_neurons_number = variables;

        const Index outputs_number = data_set.get_target_variables_number();

        Tensor<Index, 1> arquitecture(3);

        arquitecture.setValues({inputs_number, hidden_neurons_number, outputs_number});

        NeuralNetwork neural_network(NeuralNetwork::Approximation, arquitecture);
        neural_network.set_device_pointer(&device);

        // Training strategyy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::MEAN_SQUARED_ERROR);

        training_strategy.set_optimization_method(TrainingStrategy::STOCHASTIC_GRADIENT_DESCENT);

        training_strategy.get_mean_squared_error_pointer()->set_regularization_method(LossIndex::NoRegularization);

        training_strategy.get_stochastic_gradient_descent_pointer()->set_maximum_epochs_number(10);
        training_strategy.get_stochastic_gradient_descent_pointer()->set_display_period(1);

//        training_strategy.get_stochastic_gradient_descent_pointer()->set_batch_instances_number(1);

//        training_strategy.get_quasi_Newton_method_pointer()->set_display_period(1);

//        training_strategy.get_quasi_Newton_method_pointer()->set_maximum_epochs_number(20);

//        training_strategy.get_stochastic_gradient_descent_pointer()->set_batch_instances_number(variables);

//        training_strategy.get_stochastic_gradient_descent_pointer()->set_maximum_epochs_number(20);

//        training_strategy.get_gradient_descent_pointer()->get_learning_rate_algorithm_pointer()->set_learning_rate_method(LearningRateAlgorithm::Fixed);

        training_strategy.set_device_pointer(&device);

        training_strategy.perform_training();

        cout << "End" << endl;

        return 0;

    }
       catch(exception& e)
    {
       cerr << e.what() << endl;
    }
  }


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
