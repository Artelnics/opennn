#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

// OpenNN includes

#include "../opennn/opennn.h"
using namespace opennn;

int main(int argc, char* argv[])
{
    try
    {
        DataSet data_set("D:/sum.csv",',',false);
        
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}