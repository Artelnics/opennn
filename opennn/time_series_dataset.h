//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T I M E   S E R I E S   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tabular_dataset.h"

namespace opennn
{

class TimeSeriesDataset final : public TabularDataset
{

public:

    TimeSeriesDataset(const Index = 0,
                      const Shape& = {},
                      const Shape& = {});

    TimeSeriesDataset(const filesystem::path&,
                      const string&,
                      bool = true,
                      bool = false,
                      const Codification& = Codification::UTF8);

    Index get_past_time_steps() const;
    Index get_future_time_steps() const;
    bool get_multi_target() const;

    Tensor3 get_data(const string&, const string&) const;

    void set_past_time_steps(const Index);
    void set_future_time_steps(const Index);
    void set_multi_target(const bool);

    MatrixR calculate_autocorrelations(const Index = 10) const;
    Tensor3 calculate_cross_correlations(const Index = 10) const;

    void to_JSON(JsonWriter&) const override;
    void from_JSON(const JsonDocument&) override;

    void read_csv();

    void impute_missing_values_unuse() override;
    void impute_missing_values_interpolate() override;

    void fill_inputs(const vector<Index>&,
                           const vector<Index>&,
                           float*,
                           FillMode,
                           int contiguous = -1) const override;

    void fill_targets(const vector<Index>&,
                            const vector<Index>&,
                            float*,
                            FillMode,
                            int contiguous = -1) const override;

    void fill_batch(Batch&,
                    const vector<Index>&,
                    const vector<Index>&,
                    const vector<Index>&,
                    const vector<Index>&,
                    FillMode) const override;

    void resize_input_shape(Index) override;

    void refresh_forecasting_roles();

private:

    void configure_forecasting();

    Index past_time_steps = 2;

    Index future_time_steps = 1;

    bool multi_target = false;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
