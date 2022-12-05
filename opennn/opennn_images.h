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

    Tensor<Tensor<type, 1>, 1> read_bmp_image_data(const string&);

    Tensor<type, 1> resize_image(Tensor<type, 1>&);

    Tensor<Tensor<type, 1>, 1> propose_single_random_region(const Tensor<Tensor<type, 1>, 1>&, const Index&, const Index&);

    type intersection_over_union(const Index&, const Index&, const Index&, const Index&,
                                 const Index&, const Index&, const Index&, const Index&);

    Tensor<type, 1> get_ground_truth_values(Tensor<unsigned char, 1>&, Index&, Index&, Index&, Index&);

    Tensor<type, 1> get_bounding_box(const Tensor<Tensor<type, 1>, 1>&, const Index&,
                                     const Index&, const Index&, const Index&);

    void sort_channel(Tensor<unsigned char,1>&, Tensor<unsigned char,1>&, const int&);

    Tensor<unsigned char, 1> remove_padding(Tensor<unsigned char, 1>&, const int&,const int&, const int&);

    Tensor<type, 1> resize_proposed_region(const Tensor<type, 1>, const Index&, const Index&,
                                           const Index&, const Index&, const Index&);
}

#endif // OPENNN_IMAGES_H
