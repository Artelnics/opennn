//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A U T O   A S S O C I A T I O N   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef AUTOASSOCIATIONDATASET_H
#define AUTOASSOCIATIONDATASET_H

#include "data_set.h"

namespace opennn
{

class AutoAssociationDataSet : public DataSet
{

public:

    explicit AutoAssociationDataSet();

    vector<RawVariable> get_associative_raw_variables() const;
    const Tensor<type, 2>& get_associative_data() const;
    void set_auto_associative_samples_uses();

    Index get_associative_raw_variables_number() const;
    void set_associative_data(const Tensor<type, 2>&);
    void set_associative_raw_variables_number(const Index&);

    void transform_associative_dataset();
    void transform_associative_raw_variables();
    void transform_associative_data();

    void save_auto_associative_data_binary(const string&) const;
    void load_auto_associative_data_binary(const string&);

private:

    Tensor<type, 2> associative_data;

    vector<RawVariable> associative_raw_variables;

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
