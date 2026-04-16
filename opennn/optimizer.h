//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   O P T I M I Z A T I O N   A L G O R I T H M   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

//#include "statistics.h"
#include "tinyxml2.h"

using namespace tinyxml2;

namespace opennn
{

class Loss;

struct TrainingResults;

class Optimizer
{

public:

    Optimizer(Loss* = nullptr);
    virtual ~Optimizer() = default;

    enum class StoppingCondition{None,
                                 MinimumLossDecrease,
                                 LossGoal,
                                 MaximumSelectionErrorIncreases,
                                 MaximumEpochsNumber,
                                 MaximumTime};

    const Loss* get_loss() const { return loss; }

    bool get_display() const { return display; }

    void set(Loss* new_loss) { loss = new_loss; }

    virtual void set_loss(Loss* new_loss) { loss = new_loss; }

    virtual void set_display(bool new_display) { display = new_display; }

    void set_display_period(const Index new_display_period) { display_period = new_display_period; }

    void set_maximum_epochs(const Index new_maximum_epochs) { maximum_epochs = new_maximum_epochs; }
    void set_maximum_time(const type new_maximum_time) { maximum_time = new_maximum_time; }

    void set_loss_goal(const type new_loss_goal) { training_loss_goal = new_loss_goal; }
    void set_maximum_validation_failures(const Index n) { maximum_validation_failures = n; }

    // Training

    virtual void check() const;

    virtual TrainingResults train() = 0;

    virtual TrainingResults train_cuda();

    const string& get_name() const { return name; }

    virtual void print() const {}

    virtual void from_XML(const XmlDocument&);

    virtual void to_XML(XmlPrinter&) const;

    void save(const filesystem::path&) const;
    void load(const filesystem::path&);

    static type get_elapsed_time(const time_t& beginning_time);

protected:

    void set_names();
    void set_scaling();
    void set_unscaling();

    bool check_stopping_condition(TrainingResults&, Index epoch, type elapsed_time,
                                   type training_error, Index validation_failures) const;

    void write_common_xml(XmlPrinter&) const;
    void read_common_xml(const XmlElement*);

    Loss* loss = nullptr;

    type training_loss_goal = type(0);

    Index maximum_validation_failures = numeric_limits<Index>::max();

    Index maximum_epochs = 10000;

    type maximum_time = type(360000);

    Index display_period = 10;

    bool display = true;

    string name;
};

struct OptimizerData
{
    OptimizerData() = default;
    virtual ~OptimizerData() = default;

    virtual void print() const;

    VectorR potential_parameters;
    VectorR training_direction;
    type initial_learning_rate = type(0);
};

struct TrainingResults
{
    TrainingResults(const Index = 0);
    virtual ~TrainingResults() = default;

    string write_stopping_condition() const;

    type get_training_error() const;

    type get_validation_error() const;

    Index get_epochs_number() const;

    void save(const filesystem::path&) const;

    void print(const string& message = string()) const;

    Optimizer::StoppingCondition stopping_condition = Optimizer::StoppingCondition::None;

    Tensor<string, 2> write_override_results(const Index = 3) const;

    void resize_training_error_history(const Index);

    void resize_validation_error_history(const Index);

    VectorR training_error_history;

    VectorR validation_error_history;

    string elapsed_time;

    type loss = NAN;

    Index validation_failures = 0;

    type loss_decrease = type(0);
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
