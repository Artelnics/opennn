//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A N S F O R M E R   T E S T   C L A S S   H E A D E R       
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TRANSFORMERTEST_H
#define TRANSFORMERTEST_H

// Unit testing includes

#include "../opennn/unit_testing.h"
#include "../opennn/transformer.h"
#include "../opennn/language_data_set.h"
#include "../opennn/batch.h"

namespace opennn
{

class TransformerTest : public UnitTesting
{

public:

    explicit TransformerTest();

    void test_constructor();

    void test_forward_propagate();

    void test_calculate_outputs();

    bool check_activations_sums(const Tensor<type, 3>&);

    void run_test_case();

private:

    Index batch_samples_number = 0;

    Index input_length = 0;
    Index context_length = 0;
    Index input_dimensions = 0;
    Index context_dimension = 0;
    Index embedding_depth = 0;
    Index perceptron_depth = 0;
    Index heads_number = 0;
    Index number_of_layers = 0;

    Tensor<type, 2> data;

    LanguageDataSet data_set;

    Batch batch;

    Tensor<Index, 1> training_samples_indices;
    Tensor<Index, 1> context_variables_indices;
    Tensor<Index, 1> input_variables_indices;
    Tensor<Index, 1> target_variables_indices;

    Transformer transformer;
};

}

#endif


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
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
