//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   V I S I O N   T R A N S F O R M E R   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef VISION_TRANSFORMER_H
#define VISION_TRANSFORMER_H

#include "neural_network.h"

namespace opennn
{

class VisionTransformer : public NeuralNetwork
{
public:

    VisionTransformer(const Index& image_height = 32,
                      const Index& image_width = 32,
                      const Index& image_channels = 3,
                      const Index& patch_size = 4,
                      const Index& target_number = 3,
                      const Index& embedding_dimension = 128,
                      const Index& hidden_dimension = 512,
                      const Index& heads_number = 4,
                      const Index& layers_number = 1);

     void set(const Index& image_height = 0,
              const Index& image_width = 0,
              const Index& image_channels = 0,
              const Index& patch_size = 0,
              const Index& target_number = 0,
              const Index& embedding_dimension = 0,
              const Index& hidden_dimension = 0,
              const Index& heads_number = 0,
              const Index& layers_number = 1);

     Index calculate_image_output(const filesystem::path&);

     void save(const filesystem::path&) const;

     void load(const filesystem::path&);

     void set_dropout_rate(const type&);

private:

      type dropout_rate = 0;
};

};

#endif // VISION_TRANSFORMER_H
