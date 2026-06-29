//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   I N P U T S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "inputs_selection.h"

namespace opennn
{

class GrowingInputs final : public InputsSelection
{

public:

    GrowingInputs(TrainingStrategy* = nullptr);

    Index get_minimum_inputs_number() const override { return minimum_inputs_number; }
    Index get_maximum_inputs_number() const override { return maximum_inputs_number; }

    void set_default();

    void set_maximum_inputs_number(const Index);
    void set_minimum_inputs_number(const Index new_minimum_inputs_number) { minimum_inputs_number = new_minimum_inputs_number; }

    InputsSelectionResult perform_input_selection() override;

    void from_JSON(const JsonDocument&) override;

    void to_JSON(JsonWriter&) const override;

private:

    Index minimum_inputs_number = 1;
    Index maximum_inputs_number = 1;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
