#include "opennn_images.h"

namespace opennn
{
    Tensor<type, 1> get_ground_truth_values(Tensor<unsigned char, 1>& input_image,
                                            Index& x_top_left, Index& y_top_left,
                                            Index& x_bottom_right, Index& y_bottom_right)
    {
        Tensor<type, 1> ground_truth_image;
        return ground_truth_image;
    }

    Tensor<type, 1> resize_image(Tensor<type, 1>& input_image)
    {
        Tensor<type, 1> output_image;
        return output_image;
    }
}
