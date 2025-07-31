//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z A T I O N   A L G O R I T H M   C L A S S   H E A D E R 
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef OPTIMIZATIONALGORITHM_H
#define OPTIMIZATIONALGORITHM_H

#include "tinyxml2.h"
#include "statistics.h"

using namespace tinyxml2;

namespace opennn
{

class LossIndex;

struct TrainingResults;

class OptimizationAlgorithm
{

public:

    OptimizationAlgorithm(const LossIndex* = nullptr);

    enum class StoppingCondition{None,
                                 MinimumLossDecrease,
                                 LossGoal,
                                 MaximumSelectionErrorIncreases,
                                 MaximumEpochsNumber,
                                 MaximumTime};

    LossIndex* get_loss_index() const;

    string get_hardware_use() const;

    void set_hardware_use(const string&);

    bool has_loss_index() const;

    const bool& get_display() const;

    const Index& get_display_period() const;

    const Index& get_save_period() const;

    const string& get_neural_network_file_name() const;

    string write_time(const type&) const;

    void set(const LossIndex* = nullptr);

    virtual void set_loss_index(LossIndex*);

    virtual void set_display(const bool&);

    void set_display_period(const Index&);

    void set_save_period(const Index&);
    void set_neural_network_file_name(const string&);

    // Training

    virtual void check() const;

    virtual TrainingResults train() = 0;

    virtual string get_name() const;

    virtual void print() const;

    virtual Tensor<string, 2> to_string_matrix() const;

    virtual void from_XML(const XMLDocument&);

    virtual void to_XML(XMLPrinter&) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

    static type get_elapsed_time(const time_t& beginning_time);

    void set_names();
    void set_scaling();
    void set_unscaling();
    void set_vocabularies();

protected:

   LossIndex* loss_index = nullptr;

   Index maximum_epochs_number = 10000;

   type maximum_time = type(360000);

   BoxPlot auto_association_box_plot;

   string hardware_use = "Multi-core";

   Index display_period = 10;

   Index save_period = numeric_limits<Index>::max();

   string neural_network_file_name = "neural_network.xml";

   bool display = true;

#ifdef OPENNN_CUDA

protected:

    cublasHandle_t cublas_handle;
    cudnnHandle_t cudnn_handle;

public:

    void create_cuda()
    {
        cublasCreate(&cublas_handle);
        cudnnCreate(&cudnn_handle);
    }

    void destroy_cuda()
    {
        cublasDestroy(cublas_handle);
        cudnnDestroy(cudnn_handle);
    }

    virtual TrainingResults train_cuda() = 0;

#endif

};


struct OptimizationAlgorithmData
{
    OptimizationAlgorithmData();

    void print() const;

    Tensor<type, 1> potential_parameters;
    Tensor<type, 1> training_direction;
    type initial_learning_rate = type(0);

};


struct TrainingResults
{
    TrainingResults(const Index& = 0);

    string write_stopping_condition() const;

    type get_training_error() const;

    type get_selection_error() const;

    Index get_epochs_number() const;

    void save(const filesystem::path&) const;

    void print(const string& message = string());

    OptimizationAlgorithm::StoppingCondition stopping_condition = OptimizationAlgorithm::StoppingCondition::None;

    Tensor<string, 2> write_override_results(const Index& = 3) const;

    void resize_training_error_history(const Index&);

    void resize_selection_error_history(const Index&);

    Tensor<type, 1> training_error_history;

    Tensor<type, 1> selection_error_history;

    string elapsed_time;

    type loss = NAN;

    Index selection_failures = 0;

    type loss_decrease = type(0);
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
