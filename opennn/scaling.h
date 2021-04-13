//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T A T I S T I C S   H E A D E R
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#ifndef SCALING_H
#define SCALING_H

// System includes

#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <vector>

// OpenNN includes

#include "config.h"

using namespace std;
using namespace Eigen;

namespace OpenNN
{
/// Enumeration of available methods for scaling and unscaling the data.

enum Scaler{NoScaling, NoUnscaling, MinimumMaximum, MeanStandardDeviation, StandardDeviation, Logarithm};

}

#endif // STATISTICS_H
