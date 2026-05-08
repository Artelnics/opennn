//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T I M E   S E R I E S   D A T A S E T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

/**
 * @file time_series_dataset.h
 * @brief Declares the TimeSeriesDataset specialization of Dataset for
 *        univariate and multivariate time series forecasting.
 */

#pragma once

#include "dataset.h"

namespace opennn
{

/**
 * @class TimeSeriesDataset
 * @brief Dataset specialization for time series with explicit past / future
 *        windows.
 *
 * Stores a temporally ordered table of samples and exposes per-sample
 * (past_time_steps, future_time_steps) windows that the network consumes
 * as inputs and targets respectively. Supports auto- and cross-correlation
 * analysis, gap filling and several missing-value imputation strategies.
 */
class TimeSeriesDataset final : public Dataset
{

public:

    /**
     * @brief Constructs an empty TimeSeriesDataset of given dimensions.
     * @param samples_number Number of samples (time steps).
     * @param input_shape Per-sample input shape.
     * @param target_shape Per-sample target shape.
     */
    TimeSeriesDataset(const Index samples_number = 0,
                      const Shape& input_shape = {},
                      const Shape& target_shape = {});

    /**
     * @brief Constructs a TimeSeriesDataset by loading from a CSV file.
     * @param path Path to the CSV file.
     * @param separator Column separator character.
     * @param has_header Whether the first row contains column names.
     * @param has_sample_index Whether the first column is a sample index.
     * @param codification Source-file character encoding.
     */
    TimeSeriesDataset(const filesystem::path& path,
                      const string& separator,
                      bool has_header = true,
                      bool has_sample_index = false,
                      const Codification& codification = Codification::UTF8);

    /**
     * @brief Fills missing time steps with imputed rows so the time variable
     *        becomes evenly spaced.
     */
    void fill_gaps();

    /** @brief Number of past time steps used as inputs. */
    Index get_past_time_steps() const;
    /** @brief Number of future time steps used as targets. */
    Index get_future_time_steps() const;
    /** @brief Column index of the time variable. */
    Index get_time_variable_index() const;
    /** @brief Whether the dataset has more than one target variable. */
    bool get_multi_target() const;

    /**
     * @brief Returns the data tensor for the given sample / feature roles.
     * @param sample_role Sample role ("Training", "Validation", "Testing").
     * @param feature_role Feature role ("Input" or "Target").
     * @return Rank-3 tensor (sample, time_step, variable).
     */
    Tensor3 get_data(const string& sample_role, const string& feature_role) const;

    /**
     * @brief Sets the input window length.
     *
     * Receives the number of past time steps used as inputs.
     */
    void set_past_time_steps(const Index);
    /**
     * @brief Sets the forecast horizon.
     *
     * Receives the number of future time steps used as targets.
     */
    void set_future_time_steps(const Index);
    /**
     * @brief Sets the column index of the time variable.
     *
     * Receives the new time-variable column index.
     */
    void set_time_variable_index(const Index);
    /**
     * @brief Sets whether the dataset has more than one target variable.
     *
     * Receives true for multi-target, false for single-target.
     */
    void set_multi_target(const bool);

    /**
     * @brief Computes the autocorrelation function of every variable.
     * @param maximum_lag Maximum lag (in time steps) considered.
     * @return Matrix with one row per variable and one column per lag.
     */
    MatrixR calculate_autocorrelations(const Index maximum_lag = 10) const;
    /**
     * @brief Computes the Pearson cross-correlation between every variable pair.
     * @param maximum_lag Maximum lag (in time steps) considered.
     * @return Rank-3 tensor (variable_a, variable_b, lag).
     */
    Tensor3 calculate_cross_correlations(const Index maximum_lag = 10) const;
    /**
     * @brief Computes the Spearman cross-correlation between every variable pair.
     * @param maximum_lag Maximum lag (in time steps) considered.
     * @return Rank-3 tensor (variable_a, variable_b, lag).
     */
    Tensor3 calculate_cross_correlations_spearman(const Index maximum_lag = 10) const;

    /**
     * @brief Writes dataset metadata (windows, time variable) to JSON.
     */
    void to_JSON(JsonWriter&) const override;
    /**
     * @brief Loads dataset metadata (windows, time variable) from JSON.
     */
    void from_JSON(const JsonDocument&) override;

    /** @brief Reads time-series rows from the configured CSV file. */
    void read_csv() override;

    /** @brief Marks rows with missing values as None (excluded from training). */
    void impute_missing_values_unuse() override;
    /** @brief Imputes missing values via temporal interpolation. */
    void impute_missing_values_interpolate() override;

    /**
     * @brief Fills the input batch buffer with past-window data.
     *
     * Receives the sample indices, the input feature indices, the device
     * buffer pointer, an unused legacy flag, and an optional contiguous
     * stride hint (-1 to ignore).
     */
    void fill_inputs(const vector<Index>&,
                           const vector<Index>&,
                           float*,
                           bool = true,
                           int contiguous = -1) const override;

    /**
     * @brief Fills the target batch buffer with future-window data.
     *
     * Receives the sample indices, the target feature indices, the device
     * buffer pointer, an unused legacy flag, and an optional contiguous
     * stride hint (-1 to ignore).
     */
    void fill_targets(const vector<Index>&,
                            const vector<Index>&,
                            float*,
                            bool = true,
                            int contiguous = -1) const override;

private:

    /** @brief Number of past time steps used as inputs. */
    Index past_time_steps = 2;

    /** @brief Number of future time steps used as targets. */
    Index future_time_steps = 1;

    /** @brief Whether the dataset has more than one target variable. */
    bool multi_target = false;

    /** @brief Column index of the time variable. */
    Index time_variable_index = 0;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
