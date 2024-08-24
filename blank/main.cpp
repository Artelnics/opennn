//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

//#include <stdio.h>
//#include <cstring>
#include <iostream>
//#include <fstream>
//#include <sstream>
#include <string>
//#include <time.h>
//#include <chrono>
//#include <algorithm>
//#include <execution>

// OpenNN includes

#include "../opennn/data_set.h"

using namespace std;
using namespace opennn;
using namespace std::chrono;
using namespace Eigen;

int main()
{
   try
   {
        cout << "Blank\n";

       DataSet data_set("C:/Users/Roberto Lopez/Documents/5_years_mortality.csv", ",", true, false);
        data_set.save("C:/Users/Roberto Lopez/Documents/5_years_mortality.xml");
       data_set.load("C:/Users/Roberto Lopez/Documents/5_years_mortality.xml");
/*
        data_set.print();

        data_set.print_data();
*/
        cout << "Bye!" << endl;

        return 0;
   }
   catch (const exception& e)
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
