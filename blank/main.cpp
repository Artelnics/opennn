//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <stdio.h>

// OpenNN includes

#include "../../opennn/opennn/opennn.h"

using namespace OpenNN;
using namespace std;
using namespace Eigen;

int main()
{
    try{
        cout << "Hello, OpenNN!" << endl;

        ofstream file;

        string data_string;

        string data_file_name = "../data/data.dat";

        DataSet data_set;

        Index rows;

        data_set.set_has_columns_names(true);
        data_set.set_separator(',');
        data_set.set_data_file_name(data_file_name);


        data_string = "x,y\n"
                      "1,2\n"
                      "1,2\n"
                      "1,2\n"
                      "3,4\n";



/*
        data_string = "\n"
                      "x,y\n"
                      "\n"
                      "1,2\n"
                      "1,2\n"
                      "3,4\n";
*/

        file.open(data_file_name.c_str());
        file << data_string;
        file.close();

        cout << data_string << endl;

        data_set.read_csv();

        rows = data_set.get_samples_number();


        cout<<"The number of rows of the data set is:"<< data_set.get_samples_number() <<endl;
        cout << "Good bye!" << endl;

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
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
