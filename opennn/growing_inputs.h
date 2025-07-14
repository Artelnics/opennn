//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   I N P U T S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef GROWINGINPUTS_H
#define GROWINGINPUTS_H

#include "inputs_selection.h"

namespace opennn
{

class GrowingInputs : public InputsSelection
{

public:

    GrowingInputs(const TrainingStrategy* = nullptr);

    const Index& get_maximum_inputs_number() const;

    const Index& get_minimum_inputs_number() const;

    const Index& get_maximum_selection_failures() const;

    void set_default();

    void set_maximum_inputs_number(const Index&);

    void set_minimum_inputs_number(const Index&);

    void set_maximum_selection_failures(const Index&);

    InputsSelectionResults perform_input_selection() override;

    Tensor<string, 2> to_string_matrix() const;

    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;
    
    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

    string get_name() const override
    {
        return "GrowingInputs";
    }

private:

    Index maximum_inputs_number = 1;

    Index minimum_inputs_number = 1;

    Index maximum_selection_failures = 100;
};

}

#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2025 Artificial Intelligence Techniques, SL.
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
