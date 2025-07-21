//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T I M E   S E R I E S   D A T A S E T   C L A S S   H E A D E R        
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TIMESERIESDATASET_H
#define TIMESERIESDATASET_H

#include "dataset.h"

namespace opennn
{

class TimeSeriesDataset : public Dataset
{

public:

    TimeSeriesDataset(const Index& = 0,
                      const dimensions& = {},
                      const dimensions& = {});

    TimeSeriesDataset(const filesystem::path&,
                      const string&,
                      const bool& = true,
                      const bool& = false,
                      const Codification& = Codification::UTF8);

    void fill_gaps();

    const Index& get_past_time_steps() const;
    const Index& get_future_time_steps() const;

    const Index& get_time_raw_variable_index() const;

    void set_past_time_steps(const Index&);
    void set_future_time_steps(const Index&);
    void set_time_raw_variable_index(const Index&);

    Tensor<type, 2> calculate_autocorrelations(const Index& = 10) const;
    Tensor<type, 3> calculate_cross_correlations(const Index& = 10) const;

    void print() const override;

    void to_XML(XMLPrinter&) const override;
    void from_XML(const XMLDocument&) override;

    void impute_missing_values_mean();

    void fill_input_tensor(const vector<Index>&,
                           const vector<Index>&,
                           type*) const override;

    void fill_target_tensor(const vector<Index>&,
                            const vector<Index>&,
                            type*) const override;

    vector<vector<Index>> get_batches(const vector<Index>&, const Index&, const bool&) const override;


private:

    Index past_time_steps = 2;

    Index future_time_steps = 1;

    Index time_raw_variable_index = 0;
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
