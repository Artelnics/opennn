//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   4 D  D I M E N S I O N S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com


#include "config.h"

#ifndef FOURD_DIMENSIONS_H
#define FOURD_DIMENSIONS_H


namespace opennn
{
struct Convolutional4dDimensions
{
    static constexpr Index sample_index = 0U; 
    static constexpr Index channel_index = 1U; 
    static constexpr Index row_index = 3U; 
    static constexpr Index column_index = 2U; 
};

struct Kernel4dDimensions
{
    static constexpr Index channel_index = 0U; 
    static constexpr Index row_index = 2U; 
    static constexpr Index column_index = 1U; 
    static constexpr Index kernel_index = 3U; 
};
    
}
#endif