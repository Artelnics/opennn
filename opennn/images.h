//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E N N N   I M A G E S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef OPENNN_IMAGES_H
#define OPENNN_IMAGES_H





#include "config.h"

using namespace std;
using namespace Eigen;

namespace opennn
{
    // Read image

    Tensor<unsigned char, 3> read_bmp_image(const string&);
    
    // Type

    void bilinear_interpolation_resize_image(const Tensor<unsigned char, 3>&, Tensor<unsigned char, 3>&, Index, Index);

    void reflect_image_x(const ThreadPoolDevice*, TensorMap<Tensor<type, 3>>&);
    void reflect_image_y(const ThreadPoolDevice*, TensorMap<Tensor<type, 3>>&);
    void rotate_image(const ThreadPoolDevice*, const Tensor<type, 3>&, Tensor<type, 3>&, const type&);
    void rescale_image(const ThreadPoolDevice*, const Tensor<type, 3>&, TensorMap<Tensor<type, 3>>&, const type&);
    void translate_image(const ThreadPoolDevice*, const Tensor<type, 3>&, Tensor<type, 3>&, const Index&);

    Tensor<unsigned char, 1> remove_padding(Tensor<unsigned char, 1>& image, const int& rows_number, const int& columns_number, const int& padding);

    //const Eigen::array<bool, 3> reflect_horizontal_dimesions = {false, true, false};
    //const Eigen::array<bool, 3> reflect_vertical_dimesions = {true, false, false};

}

#endif // OPENNN_IMAGES_H
