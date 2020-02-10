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

using Eigen::MatrixXd;
using Eigen::Vector3d;

int main(void)
{
    try
    {
        cout << "Blank application" << endl;

        Tensor<type, 1> a(6);
        Tensor<type, 1> b(3);
//        a.setValues({1,2,3,4,5,6});
//        b.setConstant(9);

        Tensor<type, 1> copy(9);
        copy.setValues({1,2,3,4,5,6,7,8,9});

        // split vector in two vectors.

        memcpy(a.data(), copy.data(), static_cast<size_t>(a.size())*sizeof (type));
        memcpy(b.data(), copy.data() + a.size(), static_cast<size_t>(b.size())*sizeof (type));

        cout << a << endl;
        cout << "." << endl << b << endl;

        cout << "---" << endl;

        // fill vector with two vectors.

        Tensor<type, 1> a1(6);
        Tensor<type, 1> b1(3);
        a1.setValues({1,2,3,4,5,6});
        b1.setConstant(9);

        Tensor<type, 1> copy1(9);
//        copy.setValues({1,2,3,4,5,6,7,8,9});

        memcpy(copy1.data(), a1.data(),static_cast<size_t>(a1.size())*sizeof (type));
        memcpy(copy1.data() + a1.size(), b1.data(),static_cast<size_t>(b1.size())*sizeof (type));

        cout << copy1 << endl;

        Tensor<type, 2> matrix(4,6);
        matrix.setConstant(1);

        Tensor<type, 1> vector(6);
        vector = matrix.sum(Eigen::array<Index, 1>({0}));

        cout << matrix << endl << "." << endl << vector << endl;

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
