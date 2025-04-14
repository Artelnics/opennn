//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P E N N N   I M A G E S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef OPENNN_IMAGES_H
#define OPENNN_IMAGES_H

#include "pch.h"

namespace opennn
{

    Tensor<type, 3> read_bmp_image(const filesystem::path&);
    
    Tensor<type, 3> resize_image(const Tensor<type, 3>&,
                                 const Index&,
                                 const Index&);

    void reflect_image_x(const ThreadPoolDevice*, Tensor<type, 3>&);
    void reflect_image_y(const ThreadPoolDevice*, Tensor<type, 3>&);
    void rotate_image(const ThreadPoolDevice*, const Tensor<type, 3>&, Tensor<type, 3>&, const type&);
    void rescale_image(const ThreadPoolDevice*, const Tensor<type, 3>&, TensorMap<Tensor<type, 3>>&, const type&);
    void translate_image_x(const ThreadPoolDevice*, const Tensor<type, 3>&, Tensor<type, 3>&, const Index&);
    void translate_image_y(const ThreadPoolDevice*, const Tensor<type, 3>&, Tensor<type, 3>&, const Index&);

    //Tensor<unsigned char, 1> remove_padding(Tensor<unsigned char, 1>& image, const int& rows_number, const int& columns_number, const int& padding);
}

#endif // OPENNN_IMAGES_H
