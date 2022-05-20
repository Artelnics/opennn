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
#include <time.h>

// OpenNN includes

#include "../opennn/opennn.h"
#include "../opennn/text_analytics.h"

using namespace opennn;
using namespace std;
using namespace Eigen;

#include "data_set.h"




int main()
{
    try
    {
        cout << "Blank script! " << endl;

//        const string list = "17";

        TextAnalytics text_analytics;

        std::ifstream file("/home/artelnics2020/Escritorio/transformedchars.txt");
        std::ofstream file_2("/home/artelnics2020/Escritorio/result.txt");

        string line;

        int line_number = 1;

        while(std::getline(file, line))
        {
            cout << "line number: " << line_number << endl;
            line_number++;
            remove_non_printable_chars(line);
//            replace(line.begin(), line.end() + line.size(), '\f' ,'-');
//            replace(line.begin(), line.end() + line.size(), '\r' ,'-');
            Tensor<string,1> tensor(1);
            tensor.setValues({line});
//            text_analytics.delete_unicode_non_printable_chars(tensor);
            text_analytics.delete_non_printable_chars(tensor);
//            replace(line.begin(), line.end() + line.size(), '\u007f' ,'-');
//            replace(line, "\007f","-");
            file_2 << tensor(0) << endl;
//            file_2 << line << endl;
        }

        getchar();

        DataSet data_set;

        data_set.set_data_file_name("/home/artelnics2020/Escritorio/Untitled 1.csv");

        data_set.read_txt();

        cout << "Goodbye!" << endl;

        return 0;
    }
    catch(const exception& e)
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
