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

using namespace OpenNN;
using namespace std;
using namespace chrono;


int main(void)
{
    try
    {
        cout << "Hello Blank Application" << endl;

        const size_t instances_number = 10;
        const size_t inputs_number = 5;
        const size_t targets_number = 1;
        const size_t batch_instances_number = 5;
        const size_t neurons_number = 1000;
        const size_t epochs_number = 100;

        // Data Set

        cout << "Loading data..." << endl;

        DataSet data_set;

        data_set.generate_Rosenbrock_data(instances_number, inputs_number+targets_number);

        data_set.set_training();

        data_set.set_batch_instances_number(batch_instances_number);

        const Vector<size_t> inputs_indices = data_set.get_input_columns_indices();
        const Vector<size_t> targets_indices = data_set.get_target_columns_indices();

        const Vector< Vector<size_t> > training_batches = data_set.get_training_batches();

        DataSet::CudaBatch cuda_batch(&data_set);

        for(size_t epoch = 1; epoch < epochs_number; epoch++)
        {

            for(size_t interation = 0; interation < training_batches.size(); interation++)
            {
                // DataSet

                cuda_batch.copy_host(data_set.get_data(), training_batches[interation], inputs_indices, targets_indices);

                cuda_batch.copy_device();

                cuda_batch.print();

                // NeuralNetwork

            }

        }

        cout << "Bye Blank Application" << endl;

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
