// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#include "opennn/model_selection/input_selection.h"

namespace opennn
{

class GrowingInputs final : public InputSelection
{

public:

    explicit GrowingInputs(Training* = nullptr);

    Index get_minimum_inputs_number() const override { return minimum_inputs_number; }
    Index get_maximum_inputs_number() const override { return maximum_inputs_number; }

    void set_default();

    void set_maximum_inputs_number(const Index);
    void set_minimum_inputs_number(const Index new_minimum_inputs_number) { minimum_inputs_number = new_minimum_inputs_number; }

    void set_warm_start(bool new_warm_start) { warm_start = new_warm_start; }

    InputSelectionResult perform_input_selection() override;

    void from_JSON(const JsonDocument&) override;

    void to_JSON(JsonWriter&) const override;

private:

    Index minimum_inputs_number = 1;
    Index maximum_inputs_number = 1;

    bool warm_start = true;

};

}
