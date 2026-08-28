//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T I M E   S E R I E S   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/dataset/tabular_dataset.h"

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
                      bool has_header = true,
                      bool has_sample_ids = false,
                      const Codification& = Codification::UTF8);

    Index get_past_time_steps() const { return past_time_steps; }
    Index get_future_time_steps() const { return future_time_steps; }
    bool get_multi_target() const { return multi_target; }

    Tensor3 get_data(const string&, const string&) const;

    void set_past_time_steps(const Index);
    void set_future_time_steps(const Index);
    void set_multi_target(bool new_multi_target);

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
                           ColumnContiguity column_contiguity = ColumnContiguity::Unknown) const override;

    void fill_targets(const vector<Index>&,
                            const vector<Index>&,
                            float*,
                            FillMode,
                            ColumnContiguity column_contiguity = ColumnContiguity::Unknown) const override;

    void fill_batch(Batch&,
                    const vector<Index>& sample_indices,
                    const FeatureSelection&,
                    FillMode) const override;

    FeatureScaling prepare_training_scaling(
        VariableRole,
        const FeatureScaling&,
        Index) override;

    vector<Variable> get_model_input_variables() const override;
    bool sample_order_matters() const noexcept override { return true; }

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
