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

        DataSet data_set("C:/OpenNN/blank/data/irisflowers.csv", ',', true);

        data_set.read_csv();

//        data_set.calculate_columns_box_plots();

        Tensor<type, 2> data = data_set.get_data();

        const Tensor<Index, 1> used_instances_indices = data_set.get_used_instances_indices();

        BoxPlot box_plots = box_plot(data.chip(1, 1), used_instances_indices);

        cout << "minimum: " << box_plots.minimum << endl;
        cout << "first_quartile: " << box_plots.first_quartile << endl;
        cout << "median: " << box_plots.median << endl;
        cout << "third_quartile: " << box_plots.third_quartile << endl;
        cout << "maximum: " << box_plots.maximum << endl;

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
