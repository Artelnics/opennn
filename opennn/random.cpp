//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R A N D O M   N U M B E R   G E N E R A T O R   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com
#include "random.h"


namespace
{
    opennn::DefaultRandomGenerator default_generator;
}

opennn::DefaultRandomGenerator& opennn::getGlobalRandomGenerator()
{
    return default_generator;
}
