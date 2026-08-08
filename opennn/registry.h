//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R E G I S T R Y   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <memory>
#include <string>

namespace opennn
{

using namespace std;

class Layer;
class Optimizer;
class InputsSelection;

unique_ptr<Layer> create_layer(const string& name);
unique_ptr<Optimizer> create_optimizer(const string& name);
unique_ptr<InputsSelection> create_inputs_selection(const string& name);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
