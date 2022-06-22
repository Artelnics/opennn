//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E N N N   I M A G E S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef OPENNN_IMAGES_H
#define OPENNN_IMAGES_H

// System includes

#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <limits>
#include <cmath>
#include <ctime>

// OpenNN includes

#include "opennn.h"

// Eigen includes

#include "config.h"

using namespace std;
using namespace Eigen;

namespace opennn
{
    Tensor<type, 1> resize_image(Tensor<type, 1>&);
    Tensor<type, 1> get_ground_truth_values(Tensor<unsigned char, 1>&, Index&, Index&, Index&, Index&);
}

#endif // OPENNN_IMAGES_H
