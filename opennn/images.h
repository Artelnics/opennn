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
    uint8_t read_u8(ifstream&, const string&);
    uint16_t read_u16_le(ifstream&, const string&);
    uint32_t read_u32_le(ifstream&, const string&);
    int32_t read_s32_le(ifstream&, const string&);

    Tensor<type, 3> read_bmp_image(const filesystem::path&);
    
    Tensor<type, 3> resize_image(const Tensor<type, 3>&, const Index&, const Index&);

    void reflect_image_x(const ThreadPoolDevice*, Tensor<type, 3>&);
    void reflect_image_y(const ThreadPoolDevice*, Tensor<type, 3>&);
    void rotate_image(const ThreadPoolDevice*, const Tensor<type, 3>&, Tensor<type, 3>&, const type&);
    void translate_image_x(const ThreadPoolDevice*, const Tensor<type, 3>&, Tensor<type, 3>&, const Index&);
    void translate_image_y(const ThreadPoolDevice*, const Tensor<type, 3>&, Tensor<type, 3>&, const Index&);
}

#endif // OPENNN_IMAGES_H
