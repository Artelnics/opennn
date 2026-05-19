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

/// @brief Time series dataset with configurable past/future windows and autocorrelation analysis.
class TimeSeriesDataset final : public TabularDataset
{

public:

    /// @brief Creates a time series dataset with the given sample count and input/target shapes.
    /// @param sample_count Number of samples to allocate.
    /// @param input_shape Shape of input features.
    /// @param target_shape Shape of target features.
    TimeSeriesDataset(const Index = 0,
                      const Shape& = {},
                      const Shape& = {});

    /// @brief Creates a time series dataset by reading the given file with the given separator.
    /// @param data_path Path to the source CSV/text file.
    /// @param separator Field separator name.
    /// @param has_header Whether the first row contains column names.
    /// @param has_ids Whether the first column contains sample identifiers.
    /// @param codification Text encoding of the file.
    TimeSeriesDataset(const filesystem::path&,
                      const string&,
                      bool = true,
                      bool = false,
                      const Codification& = Codification::UTF8);

    /// @brief Fills missing rows between time stamps so the series has a uniform cadence.
    void fill_gaps();

    /// @brief Returns the number of past time steps used as input context.
    Index get_past_time_steps() const;
    /// @brief Returns the number of future time steps used as prediction horizon.
    Index get_future_time_steps() const;
    /// @brief Returns the index of the variable acting as the time axis.
    Index get_time_variable_index() const;
    /// @brief Returns whether the dataset is configured for multi-target forecasting.
    bool get_multi_target() const;

    /// @brief Returns the windowed 3D tensor for the given sample and feature roles.
    Tensor3 get_data(const string& sample_role, const string& feature_role) const;

    /// @brief Sets the number of past time steps used as input context.
    void set_past_time_steps(const Index);
    /// @brief Sets the number of future time steps used as prediction horizon.
    void set_future_time_steps(const Index);
    /// @brief Sets the index of the variable acting as the time axis.
    void set_time_variable_index(const Index);
    /// @brief Sets whether the dataset is configured for multi-target forecasting.
    void set_multi_target(const bool);

    /// @brief Returns the autocorrelation matrix up to the given maximum lag.
    MatrixR calculate_autocorrelations(const Index = 10) const;
    /// @brief Returns the Pearson cross-correlations between variables up to the given lag.
    Tensor3 calculate_cross_correlations(const Index = 10) const;
    /// @brief Returns the Spearman cross-correlations between variables up to the given lag.
    Tensor3 calculate_cross_correlations_spearman(const Index = 10) const;

    void to_JSON(JsonWriter&) const override;
    void from_JSON(const JsonDocument&) override;

    /// @brief Reads the configured CSV file into the time series dataset.
    void read_csv();

    /// @brief Marks samples around missing values as unused (time-series aware).
    void impute_missing_values_unuse() override;
    /// @brief Interpolates missing values along the time axis.
    void impute_missing_values_interpolate() override;

    /// @brief Copies the past-window input features of the selected samples into the destination buffer.
    void fill_inputs(const vector<Index>&,
                           const vector<Index>&,
                           float*,
                           bool is_training,
                           bool parallelize = true,
                           int contiguous = -1) const override;

    /// @brief Copies the future-window target features of the selected samples into the destination buffer.
    void fill_targets(const vector<Index>&,
                            const vector<Index>&,
                            float*,
                            bool is_training,
                            bool parallelize = true,
                            int contiguous = -1) const override;

    /// @brief Resizes the input shape, accounting for the configured past/future windows.
    void resize_input_shape(Index) override;

private:

    Index past_time_steps = 2;

    Index future_time_steps = 1;

    bool multi_target = false;

    Index time_variable_index = 0;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
