//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R A N D O M   U T I L I T I E S
//
//   Artificial Intelligence Techniques, SL
//   artelnics artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{
    void set_seed(Index);

    type random_uniform(type = -1, type = 1);
    type random_normal(type = 0, type = 1);
    Index random_integer(Index, Index);
    bool random_bool(type = 0.5);

    void set_random_uniform(VectorR&, type = -0.1, type = 0.1);
    void set_random_uniform(MatrixR&, type = -0.1, type = 0.1);

    void set_random_uniform(VectorMap, type = -0.1, type = 0.1);
    void set_random_uniform(MatrixMap, type = -0.1, type = 0.1);

    void set_random_integer(MatrixR&, Index, Index);

    void shuffle(VectorB& v);

    template<typename T>
    void shuffle_vector(vector<T>&);

    void shuffle_vector_blocks(vector<Index>&, size_t = 20);


    Index get_random_element(const vector<Index>&);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
