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

/// @brief Selects the optimal subset of input features by greedily growing the input set.
class GrowingInputs final : public InputsSelection
{

public:

    /// @brief Constructs the algorithm bound to an optional training strategy.
    GrowingInputs(TrainingStrategy* = nullptr);

    /// @brief Returns the minimum number of inputs the algorithm is allowed to select.
    Index get_minimum_inputs_number() const override;

    /// @brief Returns the maximum number of inputs the algorithm is allowed to select.
    Index get_maximum_inputs_number() const override;

    /// @brief Restores default bounds, correlation thresholds and stopping criteria.
    void set_default();

    /// @brief Sets the upper bound on the number of inputs that may be selected.
    void set_maximum_inputs_number(const Index);

    /// @brief Sets the lower bound on the number of inputs that may be selected.
    void set_minimum_inputs_number(const Index);

    /// @brief Sets the maximum allowed correlation between selected inputs.
    void set_maximum_correlation(const float);

    /// @brief Sets the minimum correlation an input must have with the targets to be considered.
    void set_minimum_correlation(const float);

    /// @brief Runs the greedy input growing procedure until the stopping criterion is met.
    /// @return Selection results including the chosen input indices and error history.
    InputsSelectionResults perform_input_selection() override;

    /// @brief Loads algorithm configuration from a JSON document.
    void from_JSON(const JsonDocument&) override;

    /// @brief Writes algorithm configuration to a JSON writer.
    void to_JSON(JsonWriter&) const override;

private:

    Index minimum_inputs_number = 1;
    Index maximum_inputs_number = 1;

    float minimum_correlation = 0;
    float maximum_correlation = 0;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
