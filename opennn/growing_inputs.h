//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   I N P U T S   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file growing_inputs.h
 * @brief Declares the GrowingInputs feature-selection method.
 */

#pragma once

#include "inputs_selection.h"

namespace opennn
{

/**
 * @class GrowingInputs
 * @brief Forward-selection of input features driven by feature-target correlation.
 *
 * Starts from the input with the highest correlation to the target and
 * iteratively adds the next-best input as long as the validation error
 * keeps improving and the configured min/max bounds are respected.
 */
class GrowingInputs final : public InputsSelection
{

public:

    /**
     * @brief Constructs the selector.
     * @param training_strategy Training strategy used to evaluate candidate subsets.
     */
    GrowingInputs(TrainingStrategy* training_strategy = nullptr);

    /** @brief Lower bound on the number of selected inputs. */
    Index get_minimum_inputs_number() const override;
    /** @brief Upper bound on the number of selected inputs. */
    Index get_maximum_inputs_number() const override;

    /** @brief Resets all hyperparameters to their default values. */
    void set_default();

    /**
     * @brief Sets the maximum number of selected inputs.
     *
     * Receives the upper bound on the number of selected inputs.
     */
    void set_maximum_inputs_number(const Index);
    /**
     * @brief Sets the minimum number of selected inputs.
     *
     * Receives the lower bound on the number of selected inputs.
     */
    void set_minimum_inputs_number(const Index);

    /**
     * @brief Sets the upper correlation threshold.
     *
     * Receives the threshold above which two inputs are considered redundant.
     */
    void set_maximum_correlation(const float);
    /**
     * @brief Sets the lower correlation threshold.
     *
     * Receives the threshold below which an input is considered uninformative.
     */
    void set_minimum_correlation(const float);

    /**
     * @brief Runs the forward-selection algorithm.
     * @return Best-of-run input subset and supporting statistics.
     */
    InputsSelectionResults perform_input_selection() override;

    /**
     * @brief Loads selector hyperparameters from a parsed JSON document.
     */
    void from_JSON(const JsonDocument&) override;

    /**
     * @brief Writes selector hyperparameters to a streaming JSON writer.
     */
    void to_JSON(JsonWriter&) const override;

private:

    /** @brief Lower bound on the number of selected inputs. */
    Index minimum_inputs_number = 1;
    /** @brief Upper bound on the number of selected inputs. */
    Index maximum_inputs_number = 1;

    /** @brief Threshold below which an input is considered uninformative. */
    float minimum_correlation = 0;
    /** @brief Threshold above which two inputs are considered redundant. */
    float maximum_correlation = 0;

};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
