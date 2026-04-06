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

    GrowingInputs(const TrainingStrategy* = nullptr);

    Index get_minimum_inputs_number() const override;
    Index get_maximum_inputs_number() const override;

    void set_default();

    void set_maximum_inputs_number(const Index);
    void set_minimum_inputs_number(const Index);

    void set_maximum_correlation(const type);
    void set_minimum_correlation(const type);

    void set_maximum_validation_failures(const Index);

    InputsSelectionResults perform_input_selection() override;

    void from_XML(const XMLDocument&) override;

    void to_XML(XMLPrinter&) const override;
    
    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

private:

    Index minimum_inputs_number = 1;
    Index maximum_inputs_number = 1;

    type minimum_correlation = 0;
    type maximum_correlation = 0;

    Index maximum_validation_failures = 100;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
