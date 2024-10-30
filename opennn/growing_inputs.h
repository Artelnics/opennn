//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   I N P U T S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef GROWINGINPUTS_H
#define GROWINGINPUTS_H

#include "training_strategy.h"
#include "inputs_selection.h"

namespace opennn
{

class GrowingInputs : public InputsSelection
{

public:

    explicit GrowingInputs(TrainingStrategy* = nullptr);

    const Index& get_maximum_inputs_number() const;

    const Index& get_minimum_inputs_number() const;

    const Index& get_maximum_selection_failures() const;

    virtual void set_default() final;

    void set_maximum_inputs_number(const Index&);

    void set_minimum_inputs_number(const Index&);

    void set_maximum_selection_failures(const Index&);

    InputsSelectionResults perform_inputs_selection() final;

    Tensor<string, 2> to_string_matrix() const;

    void from_XML(const tinyxml2::XMLDocument&);

    void to_XML(tinyxml2::XMLPrinter&) const;
    
    void save(const string&) const;
    void load(const string&);

private:

    Index maximum_inputs_number = 1;

    Index minimum_inputs_number = 1;

    Index maximum_selection_failures = 100;
};

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
