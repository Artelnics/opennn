//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T I M E   S E R I E S   D A T A S E T   C L A S S   H E A D E R        
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef TIMESERIESDATASET_H
#define TIMESERIESDATASET_H

#include "data_set.h"

namespace opennn
{

class TimeSeriesDataSet : public DataSet
{

public:

    TimeSeriesDataSet(const Index& = 0,
                      const dimensions& = {},
                      const dimensions& = {});

    TimeSeriesDataSet(const filesystem::path&,
                      const string&,
                      const bool& = true,
                      const bool& = false,
                      const Codification& = Codification::UTF8);

    void fill_gaps();

    const Index& get_lags_number() const;
    const Index& get_steps_ahead() const;

    const Index& get_time_raw_variable_index() const;

    const Index& get_group_raw_variable_index() const;

    void set_lags_number(const Index&);
    void set_steps_ahead_number(const Index&);

    void set_time_raw_variable(const string&);
    void set_group_by_raw_variable(const string&);

    Tensor<type, 2> calculate_autocorrelations(const Index& = 10) const;
    Tensor<type, 3> calculate_cross_correlations(const Index& = 10) const;

    void print() const override;

    void to_XML(XMLPrinter&) const override;
    void from_XML(const XMLDocument&) override;

    Index get_time_series_time_raw_variable_index() const;

    void impute_missing_values_mean();

private:

    Index lags_number = 0;

    Index steps_ahead = 0;

    Index time_raw_variable_index = 0;

    Index group_raw_variable_index = 0;

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
