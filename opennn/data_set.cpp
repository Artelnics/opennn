//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <codecvt>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>

#include "data_set.h"
#include "statistics.h"
#include "correlations.h"
#include "tensors.h"
#include "codification.h"
#include "strings_utilities.h"

namespace opennn
{

DataSet::DataSet()
{
    set();

    set_default();
}


DataSet::DataSet(const Tensor<type, 2>& data)
{
    set(data);

    set_default();
}


DataSet::DataSet(const Index& new_samples_number, const Index& new_variables_number)
{
    set(new_samples_number, new_variables_number);

    set_default();
}


DataSet::DataSet(const Index& new_samples_number, const Index& new_inputs_number, const Index& new_targets_number)
{
    set(new_samples_number, new_inputs_number, new_targets_number);

    set_default();
}


DataSet::DataSet(const Tensor<type, 1>& inputs_variables_dimensions, const Index& channels)
{
    set(inputs_variables_dimensions, channels);

    set_default();
}


DataSet::DataSet(const string& data_path,
                 const string& separator,
                 const bool& has_header,
                 const bool& has_samples_id,
                 const Codification& data_codification)
{
    set(data_path, separator, has_header, has_samples_id, data_codification);
}


DataSet::~DataSet()
{
    delete thread_pool;
    delete thread_pool_device;
}


const bool& DataSet::get_display() const
{
    return display;
}


DataSet::RawVariable::RawVariable()
{
    name = "";
    use = VariableUse::None;
    type = RawVariableType::None;
    categories.resize(0);
    scaler = Scaler::None;
}


DataSet::RawVariable::RawVariable(const string& new_name,
                        const VariableUse& new_raw_variable_use,
                        const RawVariableType& new_type,
                        const Scaler& new_scaler,
                        const Tensor<string, 1>& new_categories)
{
    name = new_name;
    scaler = new_scaler;
    use = new_raw_variable_use;
    type = new_type;
    categories = new_categories;
}


void DataSet::RawVariable::set_scaler(const Scaler& new_scaler)
{
    scaler = new_scaler;
}


void DataSet::RawVariable::set_scaler(const string& new_scaler)
{
    if(new_scaler == "None")
        set_scaler(Scaler::None);
    else if(new_scaler == "MinimumMaximum")
        set_scaler(Scaler::MinimumMaximum);
    else if(new_scaler == "MeanStandardDeviation")
        set_scaler(Scaler::MeanStandardDeviation);
    else if(new_scaler == "StandardDeviation")
        set_scaler(Scaler::StandardDeviation);
    else if(new_scaler == "Logarithm")
        set_scaler(Scaler::Logarithm);
    else
        throw runtime_error("Unknown scaler: " + new_scaler + "\n");
}


void DataSet::RawVariable::set_use(const VariableUse& new_raw_variable_use)
{
    use = new_raw_variable_use;
}


void DataSet::RawVariable::set_use(const string& new_raw_variable_use)
{
    if(new_raw_variable_use == "Input")
        set_use(VariableUse::Input);
    else if(new_raw_variable_use == "Target")
        set_use(VariableUse::Target);
    else if(new_raw_variable_use == "Time")
        set_use(VariableUse::Time);
    else if(new_raw_variable_use == "None")
        set_use(VariableUse::None);
    else
        throw runtime_error("Unknown raw_variable use: " + new_raw_variable_use + "\n");
}


void DataSet::RawVariable::set_type(const string& new_raw_variable_type)
{
    if(new_raw_variable_type == "Numeric")
        type = RawVariableType::Numeric;
    else if(new_raw_variable_type == "Binary")
        type = RawVariableType::Binary;
    else if(new_raw_variable_type == "Categorical")
        type = RawVariableType::Categorical;
    else if(new_raw_variable_type == "DateTime")
        type = RawVariableType::DateTime;
    else if(new_raw_variable_type == "Constant")
        type = RawVariableType::Constant;
    else
        throw runtime_error("Raw variable type is not valid (" + new_raw_variable_type + ").\n");
}


void DataSet::RawVariable::add_category(const string & new_category)
{
    const Index old_categories_number = categories.size();

    Tensor<string, 1> old_categories(old_categories_number+1);

    for(Index category_index = 0; category_index < old_categories_number; category_index++)
        categories(category_index) = old_categories(category_index);

    categories(old_categories_number) = new_category;
}


void DataSet::RawVariable::set_categories(const Tensor<string, 1>& new_categories)
{
    categories.resize(new_categories.size());

    categories = new_categories;
}


void DataSet::RawVariable::from_XML(const tinyxml2::XMLDocument& document)
{
    // Name

    const tinyxml2::XMLElement* name_element = document.FirstChildElement("Name");

    if(!name_element)
        throw runtime_error("Name element is nullptr.\n");

    if(name_element->GetText())
        name = name_element->GetText();

    // Scaler

    const tinyxml2::XMLElement* scaler_element = document.FirstChildElement("Scaler");

    if(!scaler_element)
        throw runtime_error("Scaler element is nullptr.\n");

    if(scaler_element->GetText())
        set_scaler(scaler_element->GetText());

    // Use

    const tinyxml2::XMLElement* use_element = document.FirstChildElement("Use");

    if(!use_element)
        throw runtime_error("RawVariableUse element is nullptr.\n");

    if(use_element->GetText())
        set_use(use_element->GetText());

    // Type

    const tinyxml2::XMLElement* type_element = document.FirstChildElement("Type");

    if(!type_element)
        throw runtime_error("Type element is nullptr.\n");

    if(type_element->GetText())
        set_type(type_element->GetText());

    if(type == RawVariableType::Categorical)
    {
        // Categories

        const tinyxml2::XMLElement* categories_element = document.FirstChildElement("Categories");

        if(!categories_element)
            throw runtime_error("Categories element is nullptr.\n");

        if(categories_element->GetText())
            categories = get_tokens(categories_element->GetText(), ";");
    }
}


void DataSet::RawVariable::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    // Name

    file_stream.OpenElement("Name");
    file_stream.PushText(name.c_str());
    file_stream.CloseElement();

    // Scaler

    file_stream.OpenElement("Scaler");
    file_stream.PushText(get_scaler_string().c_str());
    file_stream.CloseElement();

    // raw_variable use

    file_stream.OpenElement("Use");
    file_stream.PushText(get_use_string().c_str());
    file_stream.CloseElement();

    // Type

    file_stream.OpenElement("Type");
    file_stream.PushText(get_type_string().c_str());
    file_stream.CloseElement();

    if(type == RawVariableType::Categorical || type == RawVariableType::Binary)
    {
        if(categories.size() == 0) return;

        // Categories

        file_stream.OpenElement("Categories");

        for(Index i = 0; i < categories.size(); i++)
        {
            file_stream.PushText(categories(i).c_str());

            if(i != categories.size()-1)
                file_stream.PushText(";");
        }

        file_stream.CloseElement();
    }
}


void DataSet::RawVariable::print() const
{
    cout << "Raw variable" << endl
         << "Name: " << name << endl
         << "Use: " << get_use_string() << endl
         << "Type: " << get_type_string() << endl
         << "Scaler: " << get_scaler_string() << endl
         << "Categories: " << categories << endl;
}


DataSet::ModelType DataSet::get_model_type() const
{
    return model_type;
}


string DataSet::get_model_type_string() const
{
    switch(model_type)
    {
    case ModelType::Approximation:
        return "Approximation";
    case ModelType::Classification:
        return "Classification";
    case ModelType::Forecasting:
        return "Forecasting";
    case ModelType::AutoAssociation:
        return "AutoAssociation";
    case ModelType::TextClassification:
        return "TextClassification";
    case ModelType::ImageClassification:
        return "ImageClassification";
    default:
        throw runtime_error("Unknown model type");
    }
}


string DataSet::RawVariable::get_use_string() const 
{
    switch (use)
    {
    case VariableUse::Input:
        return "Input";

    case VariableUse::Target:
        return "Target";

    case VariableUse::Time:
        return "Time";

    case VariableUse::None:
        return "None";

    default:
        throw runtime_error("Unknow raw variable use");
    }
}


Index DataSet::RawVariable::get_categories_number() const
{
    return categories.size();
}


bool DataSet::is_sample_used(const Index& index) const
{
    return samples_uses(index) != SampleUse::None;
}


bool DataSet::is_sample_unused(const Index& index) const
{
    return samples_uses(index) == SampleUse::None;
}


Tensor<Index, 1> DataSet::get_samples_uses_numbers() const
{
    Tensor<Index, 1> count(4);
    count.setZero();

    const Index samples_number = get_samples_number();

    for(Index i = 0; i < samples_number; i++)
        switch (samples_uses(i))
        {
        case SampleUse::Training: count[0]++; break;
        case SampleUse::Selection: count[1]++; break;
        case SampleUse::Testing: count[2]++; break;
        default: count[3]++; break;
        }

    return count;
}


Tensor<type, 1> DataSet::get_samples_uses_percentages() const
{
    const Index samples_number = get_samples_number();

    return (get_samples_uses_numbers().cast<type>()) * (100 / type(samples_number));
}


string DataSet::get_sample_string(const Index& sample_index, const string& separator) const
{
    const Tensor<type, 1> sample = data.chip(sample_index, 0);

    string sample_string;

    const Index raw_variables_number = get_raw_variables_number();

    Index variable_index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        switch(raw_variables(i).type)
        {
        case RawVariableType::Numeric:
            sample_string += isnan(data(sample_index, variable_index))
                ? missing_values_label
                : to_string(double(data(sample_index, variable_index)));

            variable_index++;
            break;

        case RawVariableType::Binary:
            sample_string += isnan(data(sample_index, variable_index))
                ? missing_values_label
                : raw_variables(i).categories(Index(data(sample_index, variable_index)));

            variable_index++;
            break;

        case RawVariableType::DateTime:
            sample_string += isnan(data(sample_index, variable_index))
                ? missing_values_label
                : to_string(double(data(sample_index, variable_index)));

            variable_index++;
            break;

        case RawVariableType::Categorical:
            if(isnan(data(sample_index, variable_index)))
            {
                sample_string += missing_values_label;
            }
            else
            {
                const Index categories_number = raw_variables(i).get_categories_number();

                for(Index j = 0; j < categories_number; j++)
                {
                    if(abs(data(sample_index, variable_index+j) - type(1)) < type(NUMERIC_LIMITS_MIN))
                    {
                        sample_string += raw_variables(i).categories(j);
                        break;
                    }
                }

                variable_index += categories_number;
            }
            break;

        case RawVariableType::Constant:
            sample_string += isnan(data(sample_index, variable_index))
                ? missing_values_label
                : to_string(double(data(sample_index, variable_index)));

            variable_index++;
            break;

        default:
            break;
        }

        if(i != raw_variables_number-1) 
            sample_string += separator + " ";
    }

    return sample_string;
}


Tensor<Index, 1> DataSet::get_training_samples_indices() const
{
    const Index samples_number = get_samples_number();

    const Index training_samples_number = get_training_samples_number();

    Tensor<Index, 1> training_indices(training_samples_number);

    Index index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(samples_uses(i) == SampleUse::Training)
        {
            training_indices(index) = i;
            index++;
        }
    }

    return training_indices;
}


Tensor<Index, 1> DataSet::get_selection_samples_indices() const
{
    const Index samples_number = get_samples_number();

    const Index selection_samples_number = get_selection_samples_number();

    Tensor<Index, 1> selection_indices(selection_samples_number);

    Index count = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(samples_uses(i) == SampleUse::Selection)
        {
            selection_indices(count) = i;
            count++;
        }
    }

    return selection_indices;
}


Tensor<Index, 1> DataSet::get_testing_samples_indices() const
{
    const Index samples_number = get_samples_number();

    const Index testing_samples_number = get_testing_samples_number();

    Tensor<Index, 1> testing_indices(testing_samples_number);

    Index count = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(samples_uses(i) == SampleUse::Testing)
        {
            testing_indices(count) = i;
            count++;
        }
    }

    return testing_indices;
}


Tensor<Index, 1> DataSet::get_used_samples_indices() const
{
    const Index samples_number = get_samples_number();

    const Index used_samples_number = samples_number - get_unused_samples_number();

    Tensor<Index, 1> used_indices(used_samples_number);

    Index index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(samples_uses(i) != SampleUse::None)
        {
            used_indices(index) = i;
            index++;
        }
    }

    return used_indices;
}


Tensor<Index, 1> DataSet::get_unused_samples_indices() const
{
    const Index samples_number = get_samples_number();

    const Index unused_samples_number = get_unused_samples_number();

    Tensor<Index, 1> unused_indices(unused_samples_number);

    Index count = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(samples_uses(i) == SampleUse::None)
        {
            unused_indices(count) = i;
            count++;
        }
    }

    return unused_indices;
}


DataSet::SampleUse DataSet::get_sample_use(const Index& index) const
{
    return samples_uses(index);
}


const Tensor<DataSet::SampleUse,1 >& DataSet::get_samples_uses() const
{
    return samples_uses;
}


Tensor<Index, 1> DataSet::get_samples_uses_tensor() const
{
    const Index samples_number = get_samples_number();

    Tensor<Index, 1> samples_uses_tensor(samples_number);

    #pragma omp parallel for

    for(Index i = 0; i < samples_number; i++)
        samples_uses_tensor(i) = Index(samples_uses(i));

    return samples_uses_tensor;
}


Tensor<Index, 2> DataSet::get_batches(const Tensor<Index,1>& samples_indices,
                                      const Index& batch_samples_number,
                                      const bool& shuffle,
                                      const Index& new_buffer_size) const
{
    if(!shuffle) return split_samples(samples_indices, batch_samples_number);

    random_device rng;
    mt19937 urng(rng());

    const Index samples_number = samples_indices.size();

    Index buffer_size = new_buffer_size;
    Index batches_number;
    Index batch_size = batch_samples_number;

    // When samples_number is less than 100 (small sample)

    if(buffer_size > samples_number)
        buffer_size = samples_number;

    // Check batch size and samples number

    if(samples_number < batch_size)
    {
        batches_number = 1;
        batch_size = samples_number;
        buffer_size = batch_size;
    }
    else
    {
        batches_number = samples_number / batch_size;
    }

    Tensor<Index, 2> batches(batches_number, batch_size);

    Tensor<Index, 1> samples_copy(samples_indices);

    // Shuffle

    std::shuffle(samples_copy.data(), samples_copy.data() + samples_copy.size(), urng);

    #pragma omp parallel for

    for(Index i = 0; i < batches_number; i++)
    {
        const Index offset = i * batches_number;

        for(Index j = 0; j < batch_size; j++)
            batches(i, j) = samples_copy(offset + j);
    }

    return batches;
    
/*
    Tensor<Index, 1> buffer(buffer_size);

    for(Index i = 0; i < buffer_size; i++) 
        buffer(i) = i;

    Index next_index = buffer_size;
    Index random_index = 0;
    Index leftover_batch_samples = batch_size;

    // Heuristic cases for batch shuffling
    
    if(batch_size < buffer_size)
    {
        Index diff = buffer_size/ batch_size;

        // Main Loop

        for(Index i = 0; i < batches_number; i++)
        {
            // Last batch

            if(i == batches_number-diff)
            {
                Index buffer_index = 0;

                for(Index j = leftover_batch_samples; j < batch_size; j++)
                {
                    batches(i - 1, j) = buffer(buffer_index);

                    buffer_index++;
                }

                for(Index k = batches_number-diff; k < batches_number; k++)
                {
                    for(Index j = 0; j < batch_size; j++)
                    {
                        batches(k,j) = buffer(buffer_index);

                        buffer_index++;
                    }
                }

                break;
            }

            // Shuffle batches

            for(Index j = 0; j < batch_size; j++)
            {
                random_index = Index(rand()%buffer_size);

                batches(i, j) = buffer(random_index);

                buffer(random_index) = samples_indices(next_index);

                next_index++;

                if(next_index == samples_number)
                {
                    leftover_batch_samples = j + 1;
                    break;
                }
            }
        }

        return batches;
    }
    else // buffer_size <= batch_size
    {
        // Main Loop

        for(Index i = 0; i < batches_number; i++)
        {
            // Last batch

            if(i == batches_number-1)
            {
                std::shuffle(buffer.data(), buffer.data() +  buffer.size(), urng);

                if(batch_size <= buffer_size)
                {
                    for(Index j = 0; j < batch_size;j++)
                    {
                        batches(i, j) = buffer(j);
                    }
                }
                else //buffer_size < batch_size
                {
                    for(Index j = 0; j < buffer_size; j++)
                    {
                        batches(i, j) = buffer(j);
                    }

                    for(Index j = buffer_size; j < batch_size; j++)
                    {
                        batches(i, j) = samples_indices(next_index);

                        next_index++;
                    }
                }

                break;
            }

            // Shuffle batches

            for(Index j = 0; j < batch_size; j++)
            {
                random_index = Index(rand()%buffer_size);

                batches(i, j) = buffer(random_index);

                buffer(random_index) = samples_indices(next_index);

                next_index++;
            }
        }
    }

    std::shuffle(batches.data(), batches.data() + batches.size(), urng);
    return batches;
*/
}


Index DataSet::get_training_samples_number() const
{
    const Index samples_number = get_samples_number();

    Index training_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
        if(samples_uses(i) == SampleUse::Training)
            training_samples_number++;

    return training_samples_number;
}


Index DataSet::get_selection_samples_number() const
{
    const Index samples_number = get_samples_number();

    Index selection_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
        if(samples_uses(i) == SampleUse::Selection)
            selection_samples_number++;

    return selection_samples_number;
}


Index DataSet::get_testing_samples_number() const
{
    const Index samples_number = get_samples_number();

    Index testing_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
        if(samples_uses(i) == SampleUse::Testing)
            testing_samples_number++;

    return testing_samples_number;
}


Index DataSet::get_used_samples_number() const
{
    const Index samples_number = get_samples_number();
    const Index unused_samples_number = get_unused_samples_number();

    return samples_number - unused_samples_number;
}


Index DataSet::get_unused_samples_number() const
{
    const Index samples_number = get_samples_number();

    Index unused_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
        if(samples_uses(i) == SampleUse::None)
            unused_samples_number++;

    return unused_samples_number;
}


void DataSet::set_training()
{
    samples_uses.setConstant(SampleUse::Training);
}


void DataSet::set_selection()
{
    samples_uses.setConstant(SampleUse::Selection);
}


void DataSet::set_testing()
{
    samples_uses.setConstant(SampleUse::Testing);
}


void DataSet::set_training(const Tensor<Index, 1>& indices)
{
    Index index = 0;

    for(Index i = 0; i < indices.size(); i++)
    {
        index = indices(i);

        samples_uses(index) = SampleUse::Training;
    }
}


void DataSet::set_selection(const Tensor<Index, 1>& indices)
{
    Index index = 0;

    for(Index i = 0; i < indices.size(); i++)
    {
        index = indices(i);

        samples_uses(index) = SampleUse::Selection;
    }
}


void DataSet::set_testing(const Tensor<Index, 1>& indices)
{
    Index index = 0;

    for(Index i = 0; i < indices.size(); i++)
    {
        index = indices(i);

        samples_uses(index) = SampleUse::Testing;
    }
}


void DataSet::set_samples_unused()
{
    const Index samples_number = get_samples_number();

    for(Index i = 0; i < samples_number; i++)
        samples_uses(i) = SampleUse::None;
}


void DataSet::set_samples_unused(const Tensor<Index, 1>& indices)
{
    for(Index i = 0; i < Index(indices.size()); i++)
    {
        const Index index = indices(i);

        samples_uses(index) = SampleUse::None;
    }
}


void DataSet::set_sample_use(const Index& index, const SampleUse& new_use)
{
    const Index samples_number = get_samples_number();

    if(index >= samples_number)
        throw runtime_error("Index must be less than samples number.\n");

    samples_uses(index) = new_use;
}


void DataSet::set_sample_use(const Index& index, const string& new_use)
{
    if(new_use == "Training")
        samples_uses(index) = SampleUse::Training;
    else if(new_use == "Selection")
        samples_uses(index) = SampleUse::Selection;
    else if(new_use == "Testing")
        samples_uses(index) = SampleUse::Testing;
    else if(new_use == "None")
        samples_uses(index) = SampleUse::None;
    else
        throw runtime_error("Unknown sample use: " + new_use + "\n");
}


void DataSet::set_sample_uses(const Tensor<SampleUse, 1>& new_uses)
{
    const Index samples_number = get_samples_number();

#ifdef OPENNN_DEBUG

    const Index new_uses_size = new_uses.size();

    if(new_uses_size != samples_number)
        throw runtime_error("Size of uses(" + to_string(new_uses_size) + ") "
                            "must be equal to number of samples(" + to_string(samples_number) + ").\n");

#endif

    for(Index i = 0; i < samples_number; i++)
        samples_uses(i) = new_uses(i);
}


void DataSet::set_sample_uses(const Tensor<string, 1>& new_uses)
{
    const Index samples_number = get_samples_number();

#ifdef OPENNN_DEBUG

    const Index new_uses_size = new_uses.size();

    if(new_uses_size != samples_number)
        throw runtime_error("Size of uses(" + to_string(new_uses_size) + ") "
                            "must be equal to number of samples(" + to_string(samples_number) + ").\n");

#endif

    for(Index i = 0; i < samples_number; i++)
    {
        if(new_uses(i) == "Training" || new_uses(i) == "0")
            samples_uses(i) = SampleUse::Training;
        else if(new_uses(i) == "Selection" || new_uses(i) == "1")
            samples_uses(i) = SampleUse::Selection;
        else if(new_uses(i) == "Testing" || new_uses(i) == "2")
            samples_uses(i) = SampleUse::Testing;
        else if(new_uses(i) == "None" || new_uses(i) == "3")
            samples_uses(i) = SampleUse::None;
        else
            throw runtime_error("Unknown sample use: " + new_uses(i) + ".\n");
    }
}


void DataSet::set_sample_uses(const Tensor<Index, 1>& indices, const SampleUse sample_use)
{
    for(Index i = 0; i < indices.size(); i++)
        set_sample_use(indices(i), sample_use);
}


void DataSet::split_samples_random(const type& training_samples_ratio,
                                   const type& selection_samples_ratio,
                                   const type& testing_samples_ratio)
{
    random_device rng;
    mt19937 urng(rng());

    const Index used_samples_number = get_used_samples_number();

    if(used_samples_number == 0) return;

    const type total_ratio = training_samples_ratio + selection_samples_ratio + testing_samples_ratio;

    // Get number of samples for training, selection and testing

    const Index selection_samples_number = Index((selection_samples_ratio * used_samples_number)/total_ratio);
    const Index testing_samples_number = Index((testing_samples_ratio * used_samples_number)/ total_ratio);

    const Index training_samples_number = used_samples_number - selection_samples_number - testing_samples_number;

    const Index sum_samples_number = training_samples_number + selection_samples_number + testing_samples_number;

    if(sum_samples_number != used_samples_number)
        throw runtime_error("Sum of numbers of training, selection and testing samples is not equal to number of used samples.\n");

    const Index samples_number = get_samples_number();

    Tensor<Index, 1> indices;

    initialize_sequential(indices, 0, 1, samples_number-1);

    std::shuffle(indices.data(), indices.data() + indices.size(), urng);

    Index i = 0;
    Index index;

    // Training

    Index count_training = 0;

    while(count_training != training_samples_number)
    {
        index = indices(i);

        i++;

        if (samples_uses(index) == SampleUse::None) continue;

        samples_uses(index)= SampleUse::Training;
        count_training++;
    }

    // Selection

    Index count_selection = 0;

    while(count_selection != selection_samples_number)
    {
        index = indices(i);

        i++;

        if (samples_uses(index) == SampleUse::None) continue;
        
        samples_uses(index) = SampleUse::Selection;
        count_selection++;
    }

    // Testing

    Index count_testing = 0;

    while(count_testing != testing_samples_number)
    {
        index = indices(i);

        if (samples_uses(index) == SampleUse::None) continue;

        samples_uses(index) = SampleUse::Testing;
        count_testing++;

        i++;
    }
}


void DataSet::split_samples_sequential(const type& training_samples_ratio,
                                       const type& selection_samples_ratio,
                                       const type& testing_samples_ratio)
{
    const Index used_samples_number = get_used_samples_number();

    if(used_samples_number == 0) return;

    const type total_ratio = training_samples_ratio + selection_samples_ratio + testing_samples_ratio;

    // Get number of samples for training, selection and testing

    const Index selection_samples_number = Index(selection_samples_ratio* type(used_samples_number)/ type(total_ratio));
    const Index testing_samples_number = Index(testing_samples_ratio* type(used_samples_number)/ type(total_ratio));
    const Index training_samples_number = used_samples_number - selection_samples_number - testing_samples_number;

    const Index sum_samples_number = training_samples_number + selection_samples_number + testing_samples_number;

    if(sum_samples_number != used_samples_number)
        throw runtime_error("Sum of numbers of training, selection and testing samples is not equal to number of used samples.\n");

    Index i = 0;

    // Training

    Index count_training = 0;

    while(count_training != training_samples_number)
    {
        if(samples_uses(i) != SampleUse::None)
        {
            samples_uses(i) = SampleUse::Training;
            count_training++;
        }

        i++;
    }

    // Selection

    Index count_selection = 0;

    while(count_selection != selection_samples_number)
    {
        if(samples_uses(i) != SampleUse::None)
        {
            samples_uses(i) = SampleUse::Selection;
            count_selection++;
        }

        i++;
    }

    // Testing

    Index count_testing = 0;

    while(count_testing != testing_samples_number)
    {
        if(samples_uses(i) != SampleUse::None)
        {
            samples_uses(i) = SampleUse::Testing;
            count_testing++;
        }

        i++;
    }
}


void DataSet::set_raw_variables(const Tensor<RawVariable, 1>& new_raw_variables)
{
    raw_variables = new_raw_variables;
}


void DataSet::set_default_raw_variables_uses()
{
    const Index raw_variables_number = raw_variables.size();

    bool target = false;

    if(raw_variables_number == 0)
    {
        return;
    }
    else if(raw_variables_number == 1)
    {
        raw_variables(0).set_use(VariableUse::None);
    }
    else
    {
        set_input();

        for(Index i = raw_variables.size()-1; i >= 0; i--)
        {
            if(raw_variables(i).type == RawVariableType::Constant || raw_variables(i).type == RawVariableType::DateTime)
            {
                raw_variables(i).set_use(VariableUse::None);
                continue;
            }

            if(!target)
            {
                raw_variables(i).set_use(VariableUse::Target);

                target = true;

                continue;
            }
        }

        input_dimensions.resize(1);

        target_dimensions.resize(1);
    }
}


void DataSet::set_default_raw_variables_names()
{
    const Index raw_variables_number = raw_variables.size();

    for(Index i = 0; i < raw_variables_number; i++)
        raw_variables(i).name = "variable_" + to_string(1+i);
}


void DataSet::set_raw_variable_name(const Index& raw_variable_index, const string& new_name)
{
    raw_variables(raw_variable_index).name = new_name;
}


DataSet::VariableUse DataSet::get_raw_variable_use(const Index&  index) const
{
    return raw_variables(index).use;
}


Tensor<DataSet::VariableUse, 1> DataSet::get_raw_variables_uses() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Tensor<DataSet::VariableUse, 1> raw_variables_uses(raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
        raw_variables_uses(i) = raw_variables(i).use;

    return raw_variables_uses;
}


Tensor<DataSet::VariableUse, 1> DataSet::get_variables_uses() const
{
    const Index raw_variables_number = get_raw_variables_number();
    const Index variables_number = get_variables_number();

    Tensor<VariableUse, 1> variables_uses(variables_number);

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            index += raw_variables(i).categories.size();
        }
        else
        {
            variables_uses(index) = raw_variables(i).use;
            index++;
        }
    }

    return variables_uses;
}


string DataSet::get_variable_name(const Index& variable_index) const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            for(Index j = 0; j < raw_variables(i).get_categories_number(); j++)
                if(index == variable_index)
                    return raw_variables(i).categories(j);
                else
                    index++;
        }
        else
        {
            if(index == variable_index)
                return raw_variables(i).name;
            else
                index++;
        }
    }

    return string();
}


Tensor<string, 1> DataSet::get_variables_names() const
{    
    const Index raw_variables_number = get_raw_variables_number();

    const Index variables_number = get_variables_number();

    Tensor<string, 1> variables_names(variables_number);

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            for(Index j = 0; j < raw_variables(i).categories.size(); j++)
            {
                variables_names(index) = raw_variables(i).categories(j);

                index++;
            }
        }
        else
        {
            variables_names(index) = raw_variables(i).name;

            index++;
        }
    }

    return variables_names;
}


Tensor<string, 1> DataSet::get_input_variables_names() const
{
    const Index input_variables_number = get_input_variables_number();

    Tensor<string, 1> input_variables_names(input_variables_number);

    const Index raw_variables_number = get_raw_variables_number();

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).use != VariableUse::Input) continue;

        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            const Index categories_number = raw_variables(i).get_categories_number();

            for(Index j = 0; j < categories_number; j++)
            {
                input_variables_names(index) = raw_variables(i).categories(j);
                index++;
            }
        }
        else
        {
            input_variables_names(index) = raw_variables(i).name;
            index++;
        }
    }

    return input_variables_names;
}


Tensor<string, 1> DataSet::get_target_variables_names() const
{
    const Index target_variables_number = get_target_variables_number();

    Tensor<string, 1> target_variables_names(target_variables_number);

    const Index raw_variables_number = get_raw_variables_number();

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).use != VariableUse::Target) continue;

        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            const Index categories_number = raw_variables(i).get_categories_number();

            for(Index j = 0; j < categories_number; j++)
            {
                target_variables_names(index) = raw_variables(i).categories(j);
                index++;
            }
        }
        else
        {
            target_variables_names(index) = raw_variables(i).name;
            index++;
        }
    }

    return target_variables_names;
}


const dimensions& DataSet::get_input_dimensions() const
{
    return input_dimensions;
}


const dimensions& DataSet::get_target_dimensions() const
{
    return target_dimensions;
}


Index DataSet::get_used_variables_number() const
{
    const Index variables_number = get_variables_number();

    const Index unused_variables_number = get_unused_variables_number();

    return variables_number - unused_variables_number;
}


Tensor<Index, 1> DataSet::get_input_raw_variables_indices() const
{
    const Index input_raw_variables_number = get_input_raw_variables_number();

    Tensor<Index, 1> input_raw_variables_indices(input_raw_variables_number);

    const Index raw_variables_number = get_raw_variables_number();

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).use == VariableUse::Input)
        {
            input_raw_variables_indices(index) = i;
            index++;
        }
    }

    return input_raw_variables_indices;
}


Tensor<Index, 1> DataSet::get_target_raw_variables_indices() const
{
    const Index raw_variables_number = get_raw_variables_number();

    const Index target_raw_variables_number = get_target_raw_variables_number();

    Tensor<Index, 1> target_raw_variables_indices(target_raw_variables_number);

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).use == VariableUse::Target)
        {
            target_raw_variables_indices(index) = i;
            index++;
        }
    }

    return target_raw_variables_indices;
}


Tensor<Index, 1> DataSet::get_unused_raw_variables_indices() const
{
    const Index raw_variables_number = get_raw_variables_number();

    const Index unused_raw_variables_number = get_unused_raw_variables_number();

    Tensor<Index, 1> unused_raw_variables_indices(unused_raw_variables_number);

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).use == VariableUse::None)
        {
            unused_raw_variables_indices(index) = i;
            index++;
        }
    }

    return unused_raw_variables_indices;
}


Tensor<Index, 1> DataSet::get_used_raw_variables_indices() const
{
    const Index raw_variables_number = get_raw_variables_number();

    const Index used_raw_variables_number = get_used_raw_variables_number();

    Tensor<Index, 1> used_indices(used_raw_variables_number);

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).use  == VariableUse::Input
        || raw_variables(i).use  == VariableUse::Target
        || raw_variables(i).use  == VariableUse::Time)
        {
            used_indices(index) = i;
            index++;
        }
    }

    return used_indices;
}


Tensor<Scaler, 1> DataSet::get_raw_variables_scalers() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Tensor<Scaler, 1> raw_variables_scalers(raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
        raw_variables_scalers(i) = raw_variables(i).scaler;

    return raw_variables_scalers;
}


Tensor<Scaler, 1> DataSet::get_input_variables_scalers() const
{
    const Index input_raw_variables_number = get_input_raw_variables_number();
    const Index input_variables_number = get_input_variables_number();

    const Tensor<RawVariable, 1> input_raw_variables = get_input_raw_variables();

    Tensor<Scaler, 1> input_variables_scalers(input_variables_number);

    Index index = 0;

    for(Index i = 0; i < input_raw_variables_number; i++)
    {
        if(input_raw_variables(i).type == RawVariableType::Categorical)
        {
            for(Index j = 0; j < input_raw_variables(i).get_categories_number(); j++)
            {
                input_variables_scalers(index) = input_raw_variables(i).scaler;
                index++;
            }
        }
        else
        {
            input_variables_scalers(index) = input_raw_variables(i).scaler;
            index++;
        }
    }

    return input_variables_scalers;
}


Tensor<Scaler, 1> DataSet::get_target_variables_scalers() const
{
    const Index target_raw_variables_number = get_target_raw_variables_number();
    const Index target_variables_number = get_target_variables_number();

    const Tensor<RawVariable, 1> target_raw_variables = get_target_raw_variables();

    Tensor<Scaler, 1> target_variables_scalers(target_variables_number);

    Index index = 0;

    for(Index i = 0; i < target_raw_variables_number; i++)
    {
        if(target_raw_variables(i).type == RawVariableType::Categorical)
        {
            for(Index j = 0; j < target_raw_variables(i).get_categories_number(); j++)
            {
                target_variables_scalers(index) = target_raw_variables(i).scaler;
                index++;
            }
        }
        else
        {
            target_variables_scalers(index) = target_raw_variables(i).scaler;
            index++;
        }
    }

    return target_variables_scalers;
}


Tensor<string, 1> DataSet::get_raw_variables_names() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Tensor<string, 1> raw_variables_names(raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
        raw_variables_names(i) = raw_variables(i).name;

    return raw_variables_names;
}


Tensor<string, 1> DataSet::get_input_raw_variable_names() const
{
    const Index raw_variables_number = get_raw_variables_number();

    const Index input_raw_variables_number = get_input_raw_variables_number();

    Tensor<string, 1> input_raw_variables_names(input_raw_variables_number);

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if (raw_variables(i).use != VariableUse::Input) continue;
        
        input_raw_variables_names(index) = raw_variables(i).name;
        index++;
        
    }

    return input_raw_variables_names;
}


Tensor<string, 1> DataSet::get_target_raw_variables_names() const
{
    const Index raw_variables_number = get_raw_variables_number();

    const Index target_raw_variables_number = get_target_raw_variables_number();

    Tensor<string, 1> target_raw_variables_names(target_raw_variables_number);

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if (raw_variables(i).use != VariableUse::Target) continue;
        
        target_raw_variables_names(index) = raw_variables(i).name;
        index++;        
    }

    return target_raw_variables_names;
}


Tensor<string, 1> DataSet::get_used_raw_variables_names() const
{
    const Index raw_variables_number = get_raw_variables_number();
    const Index used_raw_variables_number = get_used_raw_variables_number();

    Tensor<string, 1> names(used_raw_variables_number);

    Index index = 0 ;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if (raw_variables(i).use == VariableUse::None) continue;

        names(index) = raw_variables(i).name;
        index++;
    }

    return names;
}


Index DataSet::get_input_raw_variables_number() const
{    
    const Index raw_variables_number = get_raw_variables_number();

    Index input_raw_variables_number = 0;

    for(Index i = 0; i < raw_variables_number; i++)
        if(raw_variables(i).use == VariableUse::Input)
            input_raw_variables_number++;

    return input_raw_variables_number;
}


Index DataSet::get_target_raw_variables_number() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index target_raw_variables_number = 0;

    for(Index i = 0; i < raw_variables_number; i++)
        if(raw_variables(i).use == VariableUse::Target)
            target_raw_variables_number++;

    return target_raw_variables_number;
}


Index DataSet::get_time_raw_variables_number() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index time_raw_variables_number = 0;

    for(Index i = 0; i < raw_variables_number; i++)
        if(raw_variables(i).use == VariableUse::Time)
            time_raw_variables_number++;

    return time_raw_variables_number;
}


Index DataSet::get_unused_raw_variables_number() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index unused_raw_variables_number = 0;

    for(Index i = 0; i < raw_variables_number; i++)
        if(raw_variables(i).use == VariableUse::None)
            unused_raw_variables_number++;

    return unused_raw_variables_number;
}


Index DataSet::get_used_raw_variables_number() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index used_raw_variables_number = 0;

    for(Index i = 0; i < raw_variables_number; i++)
        if(raw_variables(i).use != VariableUse::None)
            used_raw_variables_number++;

    return used_raw_variables_number;
}


Index DataSet::get_input_and_unused_variables_number() const
{
    Index raw_variables_number = 0;

    for(Index i = 0; i < raw_variables_number; i++)
        if(raw_variables(i).type == RawVariableType::Categorical)
            if(raw_variables(i).use == VariableUse::Input || raw_variables(i).use == VariableUse::None)
                raw_variables_number += raw_variables(i).categories.size();
        else
            if(raw_variables(i).use == VariableUse::Input || raw_variables(i).use == VariableUse::None)
                raw_variables_number++;

    return raw_variables_number;
}


Tensor<type, 1> DataSet::box_plot_from_histogram(const Histogram& histogram, 
                                                 const Index& bins_number) const
{
    const Index samples_number = get_training_samples_number();

    const Tensor<type, 1>relative_frequencies = histogram.frequencies.cast<type>() *
           histogram.frequencies.constant(100.0).cast<type>() /
           histogram.frequencies.constant(samples_number).cast<type>();

    const Tensor<type, 1> bin_centers = histogram.centers;
    const Tensor<type, 1> bin_frequencies = relative_frequencies;

    vector<type> cumulative_frequencies(1000);

    cumulative_frequencies[0] = bin_frequencies[0];

    for(size_t i = 1; i < 1000; i++)
        cumulative_frequencies[i] = cumulative_frequencies[i-1] + bin_frequencies[i];

    const type total_frequency = cumulative_frequencies[999];

    const type q1_position = type(0.25) * total_frequency;
    const type q2_position = type(0.5) * total_frequency;
    const type q3_position = type(0.75) * total_frequency;

    size_t q1_bin = 0;
    size_t q2_bin = 0;
    size_t q3_bin = 0;

    for(size_t i = 0; i < 1000; i++) 
    {
        if(cumulative_frequencies[i] >= q1_position)
        {
            q1_bin = i;
            break;
        }
    }

    for(size_t i = 0; i < 1000; i++) 
    {
        if(cumulative_frequencies[i] >= q2_position)
        {
            q2_bin = i;
            break;
        }
    }

    for(size_t i = 0; i < 1000; i++) 
    {
        if(cumulative_frequencies[i] >= q3_position)
        {
            q3_bin = i;
            break;
        }
    }

    const type bin_width = bin_centers[1] - bin_centers[0];

    const type q1 = bin_centers[q1_bin] + (q1_position - cumulative_frequencies[q1_bin-1]) * bin_width / bin_frequencies[q1_bin];

    const type q2 = bin_centers[q2_bin] + (q2_position - cumulative_frequencies[q2_bin-1]) * bin_width / bin_frequencies[q2_bin];

    const type q3 = bin_centers[q3_bin] + (q3_position - cumulative_frequencies[q3_bin-1]) * bin_width / bin_frequencies[q3_bin];

    const type minimum = bin_centers[0] - bin_width / type(2);
    const type maximum = bin_centers[999] + bin_width / type(2);

    Tensor<type, 1> iqr_values(5);
    iqr_values.setValues({ minimum, q1, q2, q3, maximum });

    return iqr_values;
}


Tensor<DataSet::RawVariable, 1> DataSet::get_raw_variables() const
{
    return raw_variables;
}


Tensor<DataSet::RawVariable, 1> DataSet::get_input_raw_variables() const
{
    const Index raw_variables_number = get_raw_variables_number();

    const Index inputs_number = get_input_raw_variables_number();

    Tensor<RawVariable, 1> input_raw_variables(inputs_number);
    Index input_index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if (raw_variables(i).use != VariableUse::Input) continue;
        
        input_raw_variables(input_index) = raw_variables(i);
        input_index++;        
    }

    return input_raw_variables;
}


Tensor<DataSet::RawVariable, 1> DataSet::get_target_raw_variables() const
{
    const Index raw_variables_number = get_raw_variables_number();

    const Index targets_number = get_target_raw_variables_number();

    Tensor<RawVariable, 1> target_raw_variables(targets_number);
    Index target_index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if (raw_variables(i).use != VariableUse::Target) continue;
        
        target_raw_variables(target_index) = raw_variables(i);
        target_index++;        
    }

    return target_raw_variables;
}


Tensor<DataSet::RawVariable, 1> DataSet::get_used_raw_variables() const
{
    const Index used_raw_variables_number = get_used_raw_variables_number();

    const Tensor<Index, 1> used_raw_variables_indices = get_used_raw_variables_indices();

    Tensor<DataSet::RawVariable, 1> used_raw_variables(used_raw_variables_number);

    for(Index i = 0; i < used_raw_variables_number; i++)
        used_raw_variables(i) = raw_variables(used_raw_variables_indices(i));

    return used_raw_variables;
}


Index DataSet::get_constant_raw_variables_number() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index constant_number = 0;

    for(Index i = 0; i < raw_variables_number; i++)
        if(raw_variables(i).type == RawVariableType::Constant)
            constant_number++;

    return constant_number;
}


Index DataSet::get_variables_number() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index variables_number = 0;

    for(Index i = 0; i < raw_variables_number; i++)       
        variables_number += (raw_variables(i).type == RawVariableType::Categorical)
            ? raw_variables(i).get_categories_number()
            : 1;

    return variables_number;
}


Index DataSet::get_input_variables_number() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index input_variables_number = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).use != VariableUse::Input) 
            continue;

        input_variables_number += (raw_variables(i).type == RawVariableType::Categorical)
            ? raw_variables(i).get_categories_number()
            : 1;
    }

    return input_variables_number;
}


Index DataSet::get_target_variables_number() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index target_variables_number = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).use != VariableUse::Target) 
            continue;

        target_variables_number += (raw_variables(i).type == RawVariableType::Categorical)
            ? raw_variables(i).get_categories_number()
            : 1;
    }

    return target_variables_number;
}


Index DataSet::get_unused_variables_number() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index unused_variables_number = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).use != VariableUse::None) 
            continue;

        unused_variables_number += (raw_variables(i).type == RawVariableType::Categorical)
            ? raw_variables(i).get_categories_number()
            : 1;

        unused_variables_number += raw_variables(i).get_categories_number();
    }

    return unused_variables_number;
}


Tensor<Index, 1> DataSet::get_used_variables_indices() const
{
    const Index used_variables_number = get_used_variables_number();
    Tensor<Index, 1> used_variables_indices(used_variables_number);

    const Index raw_variables_number = get_raw_variables_number();

    Index variable_index = 0;
    Index used_variable_index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        const Index categories_number = raw_variables(i).get_categories_number();

        if(raw_variables(i).use == VariableUse::None)
        {
            variable_index += categories_number;
            continue;
        }

        for(Index j = 0; j < categories_number; j++)
        {
            used_variables_indices(used_variable_index) = variable_index;
            variable_index++;
            used_variable_index++;
        }
    }

    return used_variables_indices;
}


Tensor<Index, 1> DataSet::get_input_variables_indices() const
{
    const Index input_variables_number = get_input_variables_number();
    Tensor<Index, 1> input_variables_indices(input_variables_number);

    const Index raw_variables_number = get_raw_variables_number();

    Index variable_index = 0;
    Index input_variable_index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).use != VariableUse::Input)
        {
            if(raw_variables(i).type == RawVariableType::Categorical)
                variable_index += raw_variables(i).get_categories_number();
            else
                variable_index++;

//            continue;
        }

        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            const Index categories_number = raw_variables(i).get_categories_number();

            for(Index j = 0; j < categories_number; j++)
            {
                input_variables_indices(input_variable_index) = variable_index;
                variable_index++;
                input_variable_index++;
            }
        }
        else
        {
            input_variables_indices(input_variable_index) = variable_index;
            variable_index++;
            input_variable_index++;
        }
    }

    return input_variables_indices;
}


Tensor<Index, 1> DataSet::get_target_variables_indices() const
{
    const Index target_variables_number = get_target_variables_number();
    Tensor<Index, 1> target_variables_indices(target_variables_number);

    const Index raw_variables_number = get_raw_variables_number();

    Index variable_index = 0;
    Index target_variable_index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).use != VariableUse::Target)
        {
            variable_index += (raw_variables(i).type == RawVariableType::Categorical)
                ? raw_variables(i).get_categories_number()
                : 1;

            continue;
        }

        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            const Index categories_number = raw_variables(i).get_categories_number();

            for(Index j = 0; j < categories_number; j++)
            {
                target_variables_indices(target_variable_index) = variable_index;
                variable_index++;
                target_variable_index++;
            }
        }
        else
        {
            target_variables_indices(target_variable_index) = variable_index;
            variable_index++;
            target_variable_index++;
        }
    }

    return target_variables_indices;
}


void DataSet::set_raw_variables_uses(const Tensor<string, 1>& new_raw_variables_uses)
{
    const Index new_raw_variables_uses_size = new_raw_variables_uses.size();

    if(new_raw_variables_uses_size != raw_variables.size())
        throw runtime_error("Size of raw_variables uses (" + to_string(new_raw_variables_uses_size) + ") "
                            "must be equal to raw_variables size (" + to_string(raw_variables.size()) + "). \n");

    for(Index i = 0; i < new_raw_variables_uses.size(); i++)
        raw_variables(i).set_use(new_raw_variables_uses(i));

    input_dimensions = {get_input_variables_number()};

    target_dimensions = {get_target_variables_number()};
}


void DataSet::set_raw_variables_uses(const Tensor<VariableUse, 1>& new_raw_variables_uses)
{
    const Index new_raw_variables_uses_size = new_raw_variables_uses.size();

    if(new_raw_variables_uses_size != raw_variables.size())
        throw runtime_error("Size of raw_variables uses (" + to_string(new_raw_variables_uses_size) + ") "
                            "must be equal to raw_variables size (" + to_string(raw_variables.size()) + ").\n");

    for(Index i = 0; i < new_raw_variables_uses.size(); i++)
        raw_variables(i).set_use(new_raw_variables_uses(i));

    input_dimensions = {get_input_variables_number()};

    target_dimensions = {get_target_variables_number()};
}


void DataSet::set_raw_variables_unused()
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
        set_raw_variable_use(i, VariableUse::None);
}


void DataSet::set_input_target_raw_variables_indices(const Tensor<Index, 1>& input_raw_variables,
                                                     const Tensor<Index, 1>& target_raw_variables)
{
    set_raw_variables_unused();

    for(Index i = 0; i < input_raw_variables.size(); i++)
        set_raw_variable_use(input_raw_variables(i), VariableUse::Input);

    for(Index i = 0; i < target_raw_variables.size(); i++)
        set_raw_variable_use(target_raw_variables(i), VariableUse::Target);
}


void DataSet::set_input_target_raw_variables_indices(const Tensor<string, 1>& input_raw_variables,
                                                     const Tensor<string, 1>& target_raw_variables)
{
    set_raw_variables_unused();

    for(Index i = 0; i < input_raw_variables.size(); i++)
        set_raw_variable_use(input_raw_variables(i), VariableUse::Input);

    for(Index i = 0; i < target_raw_variables.size(); i++)
        set_raw_variable_use(target_raw_variables(i), VariableUse::Target);
}


void DataSet::set_input_raw_variables_unused()
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
        if(raw_variables(i).use == DataSet::VariableUse::Input) 
            set_raw_variable_use(i, VariableUse::None);
}


void DataSet::set_input_raw_variables(const Tensor<Index, 1>& input_raw_variables_indices, const Tensor<bool, 1>& input_raw_variables_use)
{
    for(Index i = 0; i < input_raw_variables_indices.size(); i++)
        set_raw_variable_use(input_raw_variables_indices(i),
            input_raw_variables_use(i) ? VariableUse::Input : VariableUse::None);
}


void DataSet::set_raw_variable_use(const Index& index, const VariableUse& new_use)
{
    raw_variables(index).use = new_use;
}


void DataSet::set_raw_variables_unused(const Tensor<Index, 1>& unused_raw_variables_index)
{
    for(Index i = 0; i < unused_raw_variables_index.size(); i++)
        set_raw_variable_use(unused_raw_variables_index(i), VariableUse::None);
}


void DataSet::set_raw_variable_use(const string& name, const VariableUse& new_use)
{
    const Index index = get_raw_variable_index(name);

    set_raw_variable_use(index, new_use);
}


void DataSet::set_raw_variable_type(const Index& index, const RawVariableType& new_type)
{
    raw_variables[index].type = new_type;
}


void DataSet::set_raw_variable_type(const string& name, const RawVariableType& new_type)
{
    const Index index = get_raw_variable_index(name);

    set_raw_variable_type(index, new_type);
}


void DataSet::set_all_raw_variables_type(const RawVariableType& new_type)
{
    for(Index i = 0; i < raw_variables.size(); i ++)
        raw_variables[i].type = new_type;
}


void DataSet::set_variable_name(const Index& variable_index, const string& new_variable_name)
{
    const Index raw_variables_number = get_raw_variables_number();

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            for(Index j = 0; j < raw_variables(i).get_categories_number(); j++)
            {
                if(index == variable_index)
                {
                    raw_variables(i).categories(j) = new_variable_name;
                    return;
                }
                else
                {
                    index++;
                }
            }
        }
        else
        {
            if(index == variable_index)
            {
                raw_variables(i).name = new_variable_name;
                return;
            }
            else
            {
                index++;
            }
        }
    }
}


void DataSet::set_variables_names(const Tensor<string, 1>& new_variables_names)
{
    const Index raw_variables_number = get_raw_variables_number();

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            for(Index j = 0; j < raw_variables(i).get_categories_number(); j++)
            {
                raw_variables(i).categories(j) = new_variables_names(index);
                index++;
            }
        }
        else
        {
            raw_variables(i).name = new_variables_names(index);
            index++;
        }
    }
}


void DataSet::set_variables_names_from_raw_variables(const Tensor<string, 1>& new_variables_names,
                                               const Tensor<DataSet::RawVariable, 1>& new_raw_variables)
{
    const Index raw_variables_number = get_raw_variables_number();

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            raw_variables(i).categories.resize(new_raw_variables(i).get_categories_number());

            for(Index j = 0; j < new_raw_variables(i).get_categories_number(); j++)
            {
                raw_variables(i).categories(j) = new_variables_names(index);
                index++;
            }
        }
        else
        {
            raw_variables(i).name = new_variables_names(index);
            index++;
        }
    }
}


void DataSet::set_raw_variables_names(const Tensor<string, 1>& new_names)
{
    const Index new_names_size = new_names.size();
    const Index raw_variables_number = get_raw_variables_number();

    if(new_names_size != raw_variables_number)
        throw runtime_error("Size of names (" + to_string(new_names.size()) + ") "
                            "is not equal to raw_variables number (" + to_string(raw_variables_number) + ").\n");

    for(Index i = 0; i < raw_variables_number; i++)
        raw_variables(i).name = get_trimmed(new_names(i));
}


void DataSet::set_input()
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Constant) 
            continue;

        raw_variables(i).set_use(VariableUse::Input);
    }
}


void DataSet::set_target()
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
        raw_variables(i).set_use(VariableUse::Target);
}


void DataSet::set_variables_unused()
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
        raw_variables(i).set_use(VariableUse::None);
}


void DataSet::set_raw_variables_number(const Index& new_raw_variables_number)
{
    raw_variables.resize(new_raw_variables_number);

    set_default_raw_variables_uses();
}


void DataSet::set_raw_variables_scalers(const Scaler& scalers)
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
        raw_variables(i).scaler = scalers;
}


void DataSet::set_raw_variables_scalers(const Tensor<Scaler, 1>& new_scalers)
{
    const Index raw_variables_number = get_raw_variables_number();

    if(new_scalers.size() != raw_variables_number)
        throw runtime_error("Size of raw_variable scalers(" + to_string(new_scalers.size()) + ") "
                            "has to be the same as raw_variables numbers(" + to_string(raw_variables_number) + ").\n");

    for(Index i = 0; i < raw_variables_number; i++)
        raw_variables(i).scaler = new_scalers[i];
}


void DataSet::set_binary_raw_variables()
{
    if(display) cout << "Setting binary raw variables..." << endl;

    Index variable_index = 0;

    const Index raw_variables_number = get_raw_variables_number();

    for(Index raw_variable_index = 0; raw_variable_index < raw_variables_number; raw_variable_index++)
    {
        RawVariable raw_variable = raw_variables(raw_variable_index);

        if(raw_variable.type == RawVariableType::Numeric)
        {
            const TensorMap<Tensor<type, 1>> data_column = tensor_map(data, variable_index);

            if(is_binary_vector(data_column))
                raw_variable.type = RawVariableType::Binary;

            variable_index++;
        }
        else if(raw_variable.type == RawVariableType::Categorical)
        {
            variable_index += raw_variable.get_categories_number();
        }
        else if(raw_variable.type == RawVariableType::DateTime
             || raw_variable.type == RawVariableType::Constant
             || raw_variable.type == RawVariableType::Binary)
        {
            variable_index++;
        }

        raw_variables(raw_variable_index) = raw_variable;
    }
}


void DataSet::unuse_constant_raw_variables()
{
    if(display) cout << "Setting constant raw variables..." << endl;

    Index variable_index = 0;

    const Index raw_variables_number = get_raw_variables_number();

    for(Index raw_variable_index = 0; raw_variable_index < raw_variables_number; raw_variable_index++)
    {
        RawVariable raw_variable = raw_variables(raw_variable_index);

        if(raw_variable.type == RawVariableType::Numeric)
        {
            const TensorMap<Tensor<type, 1>> data_column = tensor_map(data, variable_index);

            if(is_constant_vector(data_column))
            {
                raw_variable.type = RawVariableType::Constant;
                raw_variable.use = VariableUse::None;
            }

            variable_index++;
        }
        else if(raw_variable.type == RawVariableType::DateTime || raw_variable.type == RawVariableType::Constant)
        {
            variable_index++;
        }
        else if(raw_variable.type == RawVariableType::Binary)
        {
            if(raw_variable.get_categories_number() == 1)
            {
                raw_variable.type = RawVariableType::Constant;
                raw_variable.use = VariableUse::None;
            }

            variable_index++;
        }
        else if(raw_variable.type == RawVariableType::Categorical)
        {           
            if(raw_variable.get_categories_number() == 1)
            {
                raw_variable.type = RawVariableType::Constant;
                raw_variable.use = VariableUse::None;
            }

            variable_index += raw_variable.get_categories_number();        
        }

        raw_variables(raw_variable_index) = raw_variable;
    }
}


void DataSet::set_input_variables_dimensions(const dimensions& new_input_dimensions)
{
    input_dimensions = new_input_dimensions;
}


void DataSet::set_target_dimensions(const dimensions& new_targets_dimensions)
{
    target_dimensions = new_targets_dimensions;
}


bool DataSet::is_empty() const
{
    return data.dimension(0) == 0 || data.dimension(1) == 0;
}


const Tensor<type, 2>& DataSet::get_data() const
{
    return data;
}


Tensor<type, 2>* DataSet::get_data_p()
{
    return &data;
}


DataSet::MissingValuesMethod DataSet::get_missing_values_method() const
{
    return missing_values_method;
}


string DataSet::get_missing_values_method_string() const
{
    switch (missing_values_method)
    {
        case MissingValuesMethod::Mean:
            return "Mean";
        case MissingValuesMethod::Median:
            return "Median";
        case MissingValuesMethod::Unuse:
            return "Unuse";
        case MissingValuesMethod::Interpolation:
            return "Interpolation";
        default:
            throw runtime_error("Unknown missing values method");
    }
}


const string& DataSet::get_data_source_path() const
{
    return data_path;
}


const bool& DataSet::get_header_line() const
{
    return has_header;
}


const bool& DataSet::get_has_ids() const
{
    return has_samples_id;
}


Tensor<string, 1> DataSet::get_sample_ids() const
{
    return samples_id;
}


const DataSet::Separator& DataSet::get_separator() const
{
    return separator;
}


string DataSet::get_separator_string() const
{
    switch(separator)
    {
    case Separator::Space:
        return " ";

    case Separator::Tab:
        return "\t";

    case Separator::Comma:
        return ",";

    case Separator::Semicolon:
        return ";";

    default:
        return string();
    }
}


string DataSet::get_separator_name() const
{
    switch(separator)
    {
    case Separator::Space:
        return "Space";

    case Separator::Tab:
        return "Tab";

    case Separator::Comma:
        return "Comma";

    case Separator::Semicolon:
        return "Semicolon";

    default:
        return string();
    }
}


const DataSet::Codification DataSet::get_codification() const
{
    return codification;
}


const string DataSet::get_codification_string() const
{
    switch(codification)
    {
    case Codification::UTF8:
        return "UTF-8";

    case Codification::SHIFT_JIS:
        return "SHIFT_JIS";

    default:
        return "UTF-8";
    }
}


const string& DataSet::get_missing_values_label() const
{
    return missing_values_label;
}


Tensor<type, 2> DataSet::get_training_data() const
{
    const Tensor<Index, 1> variables_indices = get_used_variables_indices();

    const Tensor<Index, 1> training_indices = get_training_samples_indices();

    Tensor<type, 2> training_data(training_indices.size(), variables_indices.size());

    fill_tensor_data(data, training_indices, variables_indices, training_data.data());

    return training_data;
}


Tensor<type, 2> DataSet::get_selection_data() const
{
    const Tensor<Index, 1> selection_indices = get_selection_samples_indices();

    const Index variables_number = get_variables_number();

    Tensor<Index, 1> variables_indices;
    initialize_sequential(variables_indices, 0, 1, variables_number-1);

    Tensor<type, 2> selection_data(selection_indices.size(), variables_indices.size());

    fill_tensor_data(data, selection_indices, variables_indices, selection_data.data());

    return selection_data;
}


Tensor<type, 2> DataSet::get_testing_data() const
{
    const Index variables_number = get_variables_number();

    Tensor<Index, 1> variables_indices;
    initialize_sequential(variables_indices, 0, 1, variables_number-1);

    const Tensor<Index, 1> testing_indices = get_testing_samples_indices();

    Tensor<type, 2> testing_data(testing_indices.size(), variables_indices.size());

    fill_tensor_data(data, testing_indices, variables_indices, testing_data.data());

    return testing_data;
}


Tensor<type, 2> DataSet::get_input_data() const
{
    const Index samples_number = get_samples_number();

    Tensor<Index, 1> indices;
    initialize_sequential(indices, 0, 1, samples_number-1);

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    Tensor<type, 2> input_data(indices.size(), input_variables_indices.size());

    fill_tensor_data(data, indices, input_variables_indices, input_data.data());

    return input_data;
}


Tensor<type, 2> DataSet::get_target_data() const
{
    const Tensor<Index, 1> indices = get_used_samples_indices();

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    Tensor<type, 2> target_data(indices.size(), target_variables_indices.size());

    fill_tensor_data(data, indices, target_variables_indices, target_data.data());

    return target_data;
}


Tensor<type, 2> DataSet::get_input_data(const Tensor<Index, 1>& samples_indices) const
{
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    Tensor<type, 2> input_data(samples_indices.size(), input_variables_indices.size());

    fill_tensor_data(data, samples_indices, input_variables_indices, input_data.data());

    return input_data;
}


Tensor<type, 2> DataSet::get_target_data(const Tensor<Index, 1>& samples_indices) const
{
    const Index samples_number = get_samples_number();

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    Tensor<type, 2> target_data(samples_number, target_variables_indices.size());

    fill_tensor_data(data, samples_number, target_variables_indices, target_data.data());

    return target_data;
}


Tensor<type, 2> DataSet::get_training_input_data() const
{
    const Tensor<Index, 1> training_indices = get_training_samples_indices();

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    Tensor<type, 2> training_input_data(training_indices.size(), input_variables_indices.size());

    fill_tensor_data(data, training_indices, input_variables_indices, training_input_data.data());

    return training_input_data;
}


Tensor<type, 2> DataSet::get_training_target_data() const
{
    const Tensor<Index, 1> training_indices = get_training_samples_indices();

    const Tensor<Index, 1>& target_variables_indices = get_target_variables_indices();

    Tensor<type, 2> training_target_data(training_indices.size(), target_variables_indices.size());

    fill_tensor_data(data, training_indices, target_variables_indices, training_target_data.data());

    return training_target_data;
}


Tensor<type, 2> DataSet::get_selection_input_data() const
{
    const Tensor<Index, 1> selection_indices = get_selection_samples_indices();

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    Tensor<type, 2> selection_input_data(selection_indices.size(), input_variables_indices.size());

    fill_tensor_data(data, selection_indices, input_variables_indices, selection_input_data.data());

    return selection_input_data;
}


Tensor<type, 2> DataSet::get_selection_target_data() const
{
    const Tensor<Index, 1> selection_indices = get_selection_samples_indices();

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    Tensor<type, 2> selection_target_data(selection_indices.size(), target_variables_indices.size());

    fill_tensor_data(data, selection_indices, target_variables_indices, selection_target_data.data());

    return selection_target_data;
}


Tensor<type, 2> DataSet::get_testing_input_data() const
{
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    const Tensor<Index, 1> testing_indices = get_testing_samples_indices();

    Tensor<type, 2> testing_input_data(testing_indices.size(), input_variables_indices.size());

    fill_tensor_data(data, testing_indices, input_variables_indices, testing_input_data.data());

    return testing_input_data;
}


Tensor<type, 2> DataSet::get_testing_target_data() const
{
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    const Tensor<Index, 1> testing_indices = get_testing_samples_indices();

    Tensor<type, 2> testing_target_data(testing_indices.size(), target_variables_indices.size());

    fill_tensor_data(data, testing_indices, target_variables_indices, testing_target_data.data());

    return testing_target_data;
}


Tensor<type, 1> DataSet::get_sample_data(const Index& index) const
{

#ifdef OPENNN_DEBUG

    const Index samples_number = get_samples_number();

    if(index >= samples_number)
        throw runtime_error("Index of sample (" + to_string(index) + ") "
                            "must be less than number of samples (" + to_string(samples_number) + ").\n");

#endif

    // Get sample

    return data.chip(index,0);
}


Tensor<type, 1> DataSet::get_sample_data(const Index& sample_index, const Tensor<Index, 1>& variables_indices) const
{
#ifdef OPENNN_DEBUG

    const Index samples_number = get_samples_number();

    if(sample_index >= samples_number)
        throw runtime_error("Index of sample must be less than number of \n");

#endif

    const Index variables_number = variables_indices.size();

    Tensor<type, 1 > row(variables_number);

    for(Index i = 0; i < variables_number; i++)
    {
        const Index variable_index = variables_indices(i);

        row(i) = data(sample_index, variable_index);
    }

    return row;
}


Tensor<type, 2> DataSet::get_sample_input_data(const Index&  sample_index) const
{
    const Index input_variables_number = get_input_variables_number();

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    Tensor<type, 2> inputs(1, input_variables_number);

    for(Index i = 0; i < input_variables_number; i++)
        inputs(0, i) = data(sample_index, input_variables_indices(i));

    return inputs;
}


Tensor<type, 2> DataSet::get_sample_target_data(const Index&  sample_index) const
{
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    Tensor<type, 2> sample_target_data(1, target_variables_indices.size());

    fill_tensor_data(data, Tensor<Index, 1>(sample_index), target_variables_indices, sample_target_data.data());

    return sample_target_data;
}


Tensor<Index, 1> DataSet::get_raw_variables_index(const Tensor<string, 1>& raw_variables_names) const
{
    Tensor<Index, 1> raw_variables_index(raw_variables_names.size());

    for(Index i = 0; i < raw_variables_names.size(); i++)
        raw_variables_index(i) = get_raw_variable_index(raw_variables_names(i));

    return raw_variables_index;
}


Index DataSet::get_raw_variable_index(const string& column_name) const
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
        if(raw_variables(i).name == column_name) return i;

    throw runtime_error("Cannot find " + column_name + "\n");
}


Index DataSet::get_raw_variable_index(const Index& variable_index) const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index total_variables_number = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        total_variables_number += (raw_variables(i).type == RawVariableType::Categorical)
            ? raw_variables(i).get_categories_number()
            : 1;

        if(variable_index+1 <= total_variables_number) return i;
    }

    throw runtime_error("Cannot find variable index: " + to_string(variable_index) + ".\n");
}


Tensor<Index, 1> DataSet::get_variable_indices(const Index& raw_variable_index) const
{
    Index index = 0;

    for(Index i = 0; i < raw_variable_index; i++)
        index += (raw_variables(i).type == RawVariableType::Categorical)
            ? raw_variables(i).categories.size()
            : 1;

    if(raw_variables(raw_variable_index).type == RawVariableType::Categorical)
    {
        Tensor<Index, 1> variable_indices(raw_variables(raw_variable_index).categories.size());

        for(Index j = 0; j<raw_variables(raw_variable_index).categories.size(); j++)
            variable_indices(j) = index+j;

        return variable_indices;
    }
    else
    {
        Tensor<Index, 1> indices(1);
        indices.setConstant(index);

        return indices;
    }
}


Tensor<type, 2> DataSet::get_raw_variable_data(const Index& raw_variable_index) const
{
    Index raw_variables_number = 1;
    const Index rows_number = data.dimension(0);

    if(raw_variables(raw_variable_index).type == RawVariableType::Categorical)
        raw_variables_number = raw_variables(raw_variable_index).get_categories_number();

    const Eigen::array<Index, 2> extents = {rows_number, raw_variables_number};
    const Eigen::array<Index, 2> offsets = {0, get_variable_indices(raw_variable_index)(0)};

    return data.slice(offsets, extents);
}


Tensor<type, 1> DataSet::get_sample(const Index& sample_index) const
{
    if(sample_index >= data.dimension(0))
        throw runtime_error("Sample index out of bounds.");

    return data.chip(sample_index, 0);
}


void DataSet::add_sample(const Tensor<type, 1>& sample)
{
    const Index current_samples = data.dimension(0);

    if(current_samples == 0)
    {
        Tensor<type, 2> new_data(1, sample.dimension(0));
        new_data.chip(0, 0) = sample;
        data = new_data;
        return;
    }

    if(sample.dimension(0) != data.dimension(1))
        throw runtime_error("Sample size doesn't match data raw_variable size.");

    Tensor<type, 2> new_data(current_samples + 1, data.dimension(1));

    for(Index i = 0; i < current_samples; i++)
        new_data.chip(i, 0) = data.chip(i, 0);

    new_data.chip(current_samples, 0) = sample;

    data = new_data;
}


string DataSet::get_sample_category(const Index& sample_index, const Index& column_index_start) const
{
    if(raw_variables[column_index_start].type != RawVariableType::Categorical)
        throw runtime_error("The specified raw_variable is not of categorical type.");

    for(Index raw_variable_index = column_index_start; raw_variable_index < raw_variables.size(); raw_variable_index++)
        if(data(sample_index, raw_variable_index) == 1)
            return raw_variables[column_index_start].categories(raw_variable_index - column_index_start);

    throw runtime_error("Sample does not have a valid one-hot encoded category.");
}


Tensor<type, 2> DataSet::get_raw_variables_data(const Tensor<Index, 1>& selected_raw_variable_indices) const
{
    const Index raw_variables_number = selected_raw_variable_indices.size();
    const Index rows_number = data.dimension(0);

    Tensor<type, 2> data_slice(rows_number, raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
    {
        const Eigen::array<Index, 1> rows_number_to_reshape{{rows_number}};

        const Tensor<type, 2> single_raw_variable_data = get_raw_variable_data(selected_raw_variable_indices(i));

        const Tensor<type, 1> column_data = single_raw_variable_data.reshape(rows_number_to_reshape);

        data_slice.chip(i,1) = column_data;
    }

    return data_slice;
}


Tensor<type, 2> DataSet::get_raw_variable_data(const Index& raw_variable_index, const Tensor<Index, 1>& rows_indices) const
{
    Tensor<type, 2> raw_variable_data(rows_indices.size(), get_variable_indices(raw_variable_index).size());

    fill_tensor_data(data, rows_indices, get_variable_indices(raw_variable_index), raw_variable_data.data());

    return raw_variable_data;
}


Tensor<type, 2> DataSet::get_raw_variable_data(const string& column_name) const
{
    const Index raw_variable_index = get_raw_variable_index(column_name);

    return get_raw_variable_data(raw_variable_index);
}


Tensor<type, 1> DataSet::get_variable_data(const Index& index) const
{
    return data.chip(index, 1);
}


Tensor<type, 1> DataSet::get_variable_data(const string& variable_name) const
{
    const Tensor<string, 1> variable_names = get_variables_names();

    Index size = 0;

    for(Index i = 0; i < variable_names.size(); i++)
        if(variable_names(i) ==  variable_name) size++;

    Tensor<Index, 1> variable_index(size);

    Index index = 0;

    for(Index i = 0; i < variable_names.size(); i++)
    {
        if(variable_names(i) ==  variable_name)
        {
            variable_index(index) = i;

            index++;
        }
    }

#ifdef OPENNN_DEBUG

    const Index variables_size = variable_index.size();

    if(variables_size == 0)
        throw runtime_error("Variable: " + variable_name + " does not exist.\n");

    if(variables_size > 1)
        throw runtime_error("Variable: " + variable_name + " appears more than once in the data set.\n");

#endif

    return data.chip(variable_index(0), 1);
}


Tensor<type, 1> DataSet::get_variable_data(const Index& variable_index, const Tensor<Index, 1>& samples_indices) const
{
    const Index samples_indices_size = samples_indices.size();

    Tensor<type, 1 > raw_variable(samples_indices_size);

    for(Index i = 0; i < samples_indices_size; i++)
    {
        const Index sample_index = samples_indices(i);

        raw_variable(i) = data(sample_index, variable_index);
    }

    return raw_variable;
}


Tensor<type, 1> DataSet::get_variable_data(const string& variable_name, const Tensor<Index, 1>& samples_indices) const
{
    const Tensor<string, 1> variable_names = get_variables_names();

    Index size = 0;

    for(Index i = 0; i < variable_names.size(); i++)
        if(variable_names(i) ==  variable_name)
            size++;

    Tensor<Index, 1> variable_index(size);

    Index index = 0;

    for(Index i = 0; i < variable_names.size(); i++)
    {
        if(variable_names(i) ==  variable_name)
        {
            variable_index(index) = i;

            index++;
        }
    }

#ifdef OPENNN_DEBUG

    const Index variables_size = variable_index.size();

    if(variables_size == 0)
        throw runtime_error("Variable: " + variable_name + " does not exist.\n");

    if(variables_size > 1)
        throw runtime_error("Variable: " + variable_name + " appears more than once in the data set.\n");

#endif

    const Index samples_indices_size = samples_indices.size();

    Tensor<type, 1 > raw_variable(samples_indices_size);

    for(Index i = 0; i < samples_indices_size; i++)
    {
        const Index sample_index = samples_indices(i);

        raw_variable(i) = data(sample_index, variable_index(0));
    }

    return raw_variable;
}


Tensor<Tensor<string, 1>, 1> DataSet::get_data_file_preview() const
{
    return data_file_preview;
}


void DataSet::set()
{
    thread_pool = nullptr;
    thread_pool_device = nullptr;

    data.resize(0,0);

    samples_uses.resize(0);

    raw_variables.resize(0);

    //time_series_raw_variables.resize(0);

    raw_variables_missing_values_number.resize(0);
}


void DataSet::set(const Tensor<type, 1>& inputs_variables_dimensions, const Index& channels)
{
    // Set data

    const Index variables_number = inputs_variables_dimensions.dimension(0) + channels;
    const Index samples_number = 1;
    data.resize(samples_number, variables_number);

    // Set raw variables

    for(Index i = 0; i < inputs_variables_dimensions.dimension(0);i++)
    {
        for(Index j = 0; j < inputs_variables_dimensions(i); j++)
        {
            raw_variables(i+j).name = "variable_" + to_string(i+j+1);
            raw_variables(i+j).use = VariableUse::Input;
            raw_variables(i+j).type = RawVariableType::Numeric;
        }
    }

    for(Index i = 0; i < channels; i++)
    {
        raw_variables(inputs_variables_dimensions.dimension(0) + i).name = "variable_" + to_string(inputs_variables_dimensions.dimension(0) + i + 1);
        raw_variables(inputs_variables_dimensions.dimension(0) + i).use = VariableUse::Target;
        raw_variables(inputs_variables_dimensions.dimension(0) + i).type = RawVariableType::Numeric;
    }
}


void DataSet::set(const string& data_path,
                  const string& separator,
                  const bool& new_has_header,
                  const bool& new_has_ids,
                  const DataSet::Codification& new_codification)
{
    set();

    set_default();

    set_data_source_path(data_path);

    set_separator_string(separator);

    set_has_header(new_has_header);

    set_has_ids(new_has_ids);

    set_codification(new_codification);

    read_csv();

    set_default_raw_variables_scalers();

    set_default_raw_variables_uses();

    const Index input_variables_number = get_input_variables_number();
    const Index target_variables_number = get_target_variables_number();

    input_dimensions = {input_variables_number};

    target_dimensions = {target_variables_number};
}


void DataSet::set(const Tensor<type, 2>& new_data)
{
    data_path = "";
    
    const Index variables_number = new_data.dimension(1);
    const Index samples_number = new_data.dimension(0);

    set(samples_number, variables_number);

    data = new_data;

    set_default_raw_variables_uses();
}


void DataSet::set(const Index& new_samples_number, const Index& new_variables_number)
{
    
    data.resize(new_samples_number, new_variables_number);

    raw_variables.resize(new_variables_number);
    
    for(Index index = 0; index < new_variables_number-1; index++)
    {
        raw_variables(index).name = "variable_" + to_string(index+1);
        raw_variables(index).use = VariableUse::Input;
        raw_variables(index).type = RawVariableType::Numeric;
    }

    raw_variables(new_variables_number - 1).name = "variable_" + to_string(new_variables_number);
    raw_variables(new_variables_number - 1).use = VariableUse::Target;
    raw_variables(new_variables_number - 1).type = RawVariableType::Numeric;

    samples_uses.resize(new_samples_number);

    split_samples_random();
}


void DataSet::set(const Index& new_samples_number,
                  const Index& new_inputs_number,
                  const Index& new_targets_number)
{

    data_path = "";

    const Index new_variables_number = new_inputs_number + new_targets_number;

    data.resize(new_samples_number, new_variables_number);

    raw_variables.resize(new_variables_number);

    for(Index i = 0; i < new_variables_number; i++)
    {
        raw_variables(i).type = RawVariableType::Numeric;
        raw_variables(i).name = "variable_" + to_string(i+1);

        raw_variables(i).use = (i < new_inputs_number)
            ? VariableUse::Input
            : VariableUse::Target;
    }

    input_dimensions = { new_inputs_number };

    target_dimensions = { new_targets_number };

    samples_uses.resize(new_samples_number);
    split_samples_random();
}


void DataSet::set(const DataSet& other_data_set)
{
    data_path = other_data_set.data_path;

    has_header = other_data_set.has_header;

    separator = other_data_set.separator;

    missing_values_label = other_data_set.missing_values_label;

    data = other_data_set.data;

    raw_variables = other_data_set.raw_variables;

    display = other_data_set.display;
}


void DataSet::set(const tinyxml2::XMLDocument& data_set_document)
{
    if(thread_pool != nullptr) delete thread_pool;
    if(thread_pool_device != nullptr) delete thread_pool_device;

    set_default();

    from_XML(data_set_document);
}


void DataSet::set(const string& file_name)
{
    load(file_name);
}


void DataSet::set_display(const bool& new_display)
{
    display = new_display;
}


void DataSet::set_default()
{
    const int n = omp_get_max_threads();
    thread_pool = new ThreadPool(n);
    thread_pool_device = new ThreadPoolDevice(thread_pool, n);

    has_header = false;

    separator = Separator::Comma;

    missing_values_label = "NA";

    //set_default_raw_variables_uses();

    set_default_raw_variables_names();

    input_dimensions = {get_input_variables_number()};

    target_dimensions = {get_target_variables_number()};
}


void DataSet::set_model_type_string(const string& new_model_type)
{
    if(new_model_type == "Approximation")
        set_model_type(ModelType::Approximation);
    else if(new_model_type == "Classification")
        set_model_type(ModelType::Classification);
    else if(new_model_type == "Forecasting")
        set_model_type(ModelType::Forecasting);
    else if(new_model_type == "ImageClassification")
        set_model_type(ModelType::ImageClassification);
    else if(new_model_type == "TextClassification")
        set_model_type(ModelType::TextClassification);
    else if(new_model_type == "AutoAssociation")
        set_model_type(ModelType::AutoAssociation);
    else
        throw runtime_error("Unknown model type: " + new_model_type + "\n");
}


void DataSet::set_model_type(const DataSet::ModelType& new_model_type)
{
    model_type = new_model_type;
}


void DataSet::set_data(const Tensor<type, 2>& new_data)
{
    const Index samples_number = new_data.dimension(0);
    const Index variables_number = new_data.dimension(1);

    set(samples_number, variables_number);

    data = new_data;
}


void DataSet::set_data(const Tensor<type, 2>& new_data, const bool& new_samples)
{
    data = new_data;
}


void DataSet::set_data_source_path(const string& new_data_file_name)
{
    data_path = new_data_file_name;
}


void DataSet::set_has_header(const bool& new_has_header)
{
    has_header = new_has_header;
}


void DataSet::set_has_ids(const bool& new_has_ids)
{
    has_samples_id = new_has_ids;
}


void DataSet::set_separator(const Separator& new_separator)
{
    separator = new_separator;
}


void DataSet::set_separator_string(const string& new_separator_string)
{
    if(new_separator_string == " ")
        separator = Separator::Space;
    else if(new_separator_string == "\t")
        separator = Separator::Tab;
    else if(new_separator_string == ",")
        separator = Separator::Comma;
    else if(new_separator_string == ";")
        separator = Separator::Semicolon;
    else
        throw runtime_error("Unknown separator: " + new_separator_string);
}


void DataSet::set_separator_name(const string& new_separator_name)
{
    if(new_separator_name == "Space")
        separator = Separator::Space;
    else if(new_separator_name == "Tab")
        separator = Separator::Tab;
    else if(new_separator_name == "Comma")
        separator = Separator::Comma;
    else if(new_separator_name == "Semicolon")
        separator = Separator::Semicolon;
    else
        throw runtime_error("Unknown separator: " + new_separator_name + ".\n");
}


void DataSet::set_codification(const DataSet::Codification& new_codification)
{
    codification = new_codification;
}


void DataSet::set_codification(const string& new_codification_string)
{
    if(new_codification_string == "UTF-8")
        codification = Codification::UTF8;
    else if(new_codification_string == "SHIFT_JIS")
        codification = Codification::SHIFT_JIS;
    else
        throw runtime_error("Unknown codification: " + new_codification_string + ".\n");
}


void DataSet::set_missing_values_label(const string& new_missing_values_label)
{
#ifdef OPENNN_DEBUG

    if(get_trimmed(new_missing_values_label).empty())
        throw runtime_error("Missing values label cannot be empty.\n");

#endif

    missing_values_label = new_missing_values_label;
}


void DataSet::set_missing_values_method(const DataSet::MissingValuesMethod& new_missing_values_method)
{
    missing_values_method = new_missing_values_method;
}


void DataSet::set_missing_values_method(const string & new_missing_values_method)
{
    if(new_missing_values_method == "Unuse")
        missing_values_method = MissingValuesMethod::Unuse;
    else if(new_missing_values_method == "Mean")
        missing_values_method = MissingValuesMethod::Mean;
    else if(new_missing_values_method == "Median")
        missing_values_method = MissingValuesMethod::Median;
    else if(new_missing_values_method == "Interpolation")
        missing_values_method = MissingValuesMethod::Interpolation;
    else
        throw runtime_error("Unknown method type.\n");
}


void DataSet::set_threads_number(const int& new_threads_number)
{
    if(thread_pool != nullptr) delete thread_pool;
    if(thread_pool_device != nullptr) delete thread_pool_device;

    thread_pool = new ThreadPool(new_threads_number);
    thread_pool_device = new ThreadPoolDevice(thread_pool, new_threads_number);
}


void DataSet::set_samples_number(const Index& new_samples_number)
{
    const Index variables_number = get_variables_number();

    set(new_samples_number,variables_number);
}


Tensor<Index, 1> DataSet::unuse_repeated_samples()
{
    const Index samples_number = get_samples_number();

#ifdef OPENNN_DEBUG

    if(samples_number == 0)
        throw runtime_error("Number of samples is zero.\n");

#endif

    Tensor<Index, 1> repeated_samples(0);

    Tensor<type, 1> sample_i;
    Tensor<type, 1> sample_j;

    for(Index i = 0; i < samples_number; i++)
    {
        sample_i = get_sample_data(i);

        for(Index j = Index(i+1); j < samples_number; j++)
        {
            sample_j = get_sample_data(j);

            if(get_sample_use(j) != SampleUse::None
            && equal(sample_i.data(), sample_i.data()+sample_i.size(), sample_j.data()))
            {
                set_sample_use(j, SampleUse::None);

                push_back_index(repeated_samples, j);
            }
        }
    }

    return repeated_samples;
}


Tensor<string, 1> DataSet::unuse_uncorrelated_raw_variables(const type& minimum_correlation)
{
    Tensor<string, 1> unused_raw_variables;

    const Tensor<Correlation, 2> correlations = calculate_input_target_raw_variables_correlations();

    const Index input_raw_variables_number = get_input_raw_variables_number();
    const Index target_raw_variables_number = get_target_raw_variables_number();

    const Tensor<Index, 1> input_raw_variables_indices = get_input_raw_variables_indices();

    for(Index i = 0; i < input_raw_variables_number; i++)
    {
        const Index input_raw_variable_index = input_raw_variables_indices(i);

        for(Index j = 0; j < target_raw_variables_number; j++)
        {
            if(!isnan(correlations(i, j).r)
                && abs(correlations(i, j).r) < minimum_correlation
                && raw_variables(input_raw_variable_index).use != VariableUse::None)
            {
                raw_variables(input_raw_variable_index).set_use(VariableUse::None);

                push_back_string(unused_raw_variables, raw_variables(input_raw_variable_index).name);
            }
        }
    }

    return unused_raw_variables;
}


Tensor<string, 1> DataSet::unuse_multicollinear_raw_variables(Tensor<Index, 1>& original_variable_indices, Tensor<Index, 1>& final_variable_indices)
{
    // Original_raw_variables_indices and final_raw_variables_indices refers to the indices of the variables

    Tensor<string, 1> unused_raw_variables;

    for(Index i = 0; i < original_variable_indices.size(); i++)
    {
        const Index original_raw_variable_index = original_variable_indices(i);

        bool found = false;

        for(Index j = 0; j < final_variable_indices.size(); j++)
        {
            if(original_raw_variable_index == final_variable_indices(j))
            {
                found = true;
                break;
            }
        }

        const Index raw_variable_index = get_raw_variable_index(original_raw_variable_index);

        if(!found && raw_variables(raw_variable_index).use != VariableUse::None)
        {
            raw_variables(raw_variable_index).set_use(VariableUse::None);

            push_back_string(unused_raw_variables, raw_variables(raw_variable_index).name);
        }
    }

    return unused_raw_variables;
}


Tensor<Histogram, 1> DataSet::calculate_raw_variables_distribution(const Index& bins_number) const
{
    const Index raw_variables_number = raw_variables.size();
    const Index used_raw_variables_number = get_used_raw_variables_number();
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Index used_samples_number = used_samples_indices.size();

    Tensor<Histogram, 1> histograms(used_raw_variables_number);

    Index variable_index = 0;
    Index used_raw_variable_index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Numeric)
        {
            if(raw_variables(i).use == VariableUse::None)
            {
                variable_index++;
            }
            else
            {
                Tensor<type, 1> raw_variable(used_samples_number);

                for(Index j = 0; j < used_samples_number; j++)
                    raw_variable(j) = data(used_samples_indices(j), variable_index);

                histograms(used_raw_variable_index) = histogram(raw_variable, bins_number);

                variable_index++;
                used_raw_variable_index++;
            }
        }
        else if(raw_variables(i).type == RawVariableType::Categorical)
        {
            const Index categories_number = raw_variables(i).get_categories_number();

            if(raw_variables(i).use == VariableUse::None)
            {
                variable_index += categories_number;
            }
            else
            {
                Tensor<Index, 1> categories_frequencies(categories_number);
                categories_frequencies.setZero();
                Tensor<type, 1> centers(categories_number);

                for(Index j = 0; j < categories_number; j++)
                {
                    for(Index k = 0; k < used_samples_number; k++)
                        if(abs(data(used_samples_indices(k), variable_index) - type(1)) < type(NUMERIC_LIMITS_MIN))
                            categories_frequencies(j)++;

                    centers(j) = type(j);

                    variable_index++;
                }

                histograms(used_raw_variable_index).frequencies = categories_frequencies;
                histograms(used_raw_variable_index).centers = centers;

                used_raw_variable_index++;
            }
        }
        else if(raw_variables(i).type == RawVariableType::Binary)
        {
            if(raw_variables(i).use == VariableUse::None)
            {
                variable_index++;
            }
            else
            {
                Tensor<Index, 1> binary_frequencies(2);
                binary_frequencies.setZero();

                for(Index j = 0; j < used_samples_number; j++)
                {
                    binary_frequencies(abs(data(used_samples_indices(j), variable_index) - type(1)) < type(NUMERIC_LIMITS_MIN)
                        ? 0
                        : 1)++;
                }

                histograms(used_raw_variable_index).frequencies = binary_frequencies;
                variable_index++;
                used_raw_variable_index++;
            }
        }
        else if(raw_variables(i).type == RawVariableType::DateTime)
        {
            // @todo

            if(raw_variables(i).use == VariableUse::None)
            {
            }
            else
            {
            }

            variable_index++;
        }
        else
        {
            variable_index++;
        }
    }

    return histograms;
}


Tensor<BoxPlot, 1> DataSet::calculate_data_raw_variables_box_plot(Tensor<type,2>& data) const
{
    const Index raw_variables_number = data.dimension(1);

    Tensor<BoxPlot, 1> box_plots(raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
        box_plots(i) = box_plot(data.chip(i, 1));

    return box_plots;
}


Tensor<BoxPlot, 1> DataSet::calculate_raw_variables_box_plots() const
{
    const Index raw_variables_number = get_raw_variables_number();

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();

    Tensor<BoxPlot, 1> box_plots(raw_variables_number);

//    Index used_raw_variable_index = 0;
    Index variable_index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).type == RawVariableType::Numeric || raw_variables(i).type == RawVariableType::Binary)
        {
            if(raw_variables(i).use != VariableUse::None)
            {
                box_plots(i) = box_plot(data.chip(variable_index, 1), used_samples_indices);

//                used_raw_variable_index++;
            }
            else
            {
                box_plots(i) = BoxPlot();
            }

            variable_index++;
        }
        else if(raw_variables(i).type == RawVariableType::Categorical)
        {
            variable_index += raw_variables(i).get_categories_number();

            box_plots(i) = BoxPlot();
        }
        else
        {
            variable_index++;
            box_plots(i) = BoxPlot();
        }
    }

    return box_plots;
}


Index DataSet::calculate_used_negatives(const Index& target_index)
{
    Index negatives = 0;

    const Tensor<Index, 1> used_indices = get_used_samples_indices();

    const Index used_samples_number = used_indices.size();

    for(Index i = 0; i < used_samples_number; i++)
    {
        const Index training_index = used_indices(i);

        if (isnan(data(training_index, target_index))) continue;
        
        if(abs(data(training_index, target_index)) < type(NUMERIC_LIMITS_MIN))
            negatives++;
        else if(abs(data(training_index, target_index) - type(1)) > type(NUMERIC_LIMITS_MIN)
             || data(training_index, target_index) < type(0))
            throw runtime_error("Training sample is neither a positive nor a negative: "
                                + to_string(training_index) + "-" + to_string(target_index) + "-" + to_string(data(training_index, target_index)));        
    }

    return negatives;
}


Index DataSet::calculate_training_negatives(const Index& target_index) const
{
    Index negatives = 0;

    const Tensor<Index, 1> training_indices = get_training_samples_indices();

    const Index training_samples_number = training_indices.size();

    for(Index i = 0; i < training_samples_number; i++)
    {
        const Index training_index = training_indices(i);

        if(abs(data(training_index, target_index)) < type(NUMERIC_LIMITS_MIN))
            negatives++;
        else if(abs(data(training_index, target_index) - type(1)) > type(1.0e-3))
            throw runtime_error("Training sample is neither positive nor negative: "
                                + to_string(data(training_index, target_index)));
    }

    return negatives;
}


Index DataSet::calculate_selection_negatives(const Index& target_index) const
{
    Index negatives = 0;

    const Index selection_samples_number = get_selection_samples_number();

    const Tensor<Index, 1> selection_indices = get_selection_samples_indices();

    for(Index i = 0; i < Index(selection_samples_number); i++)
    {
        const Index selection_index = selection_indices(i);

        if(abs(data(selection_index, target_index)) < type(NUMERIC_LIMITS_MIN))
            negatives++;
        else if(abs(data(selection_index, target_index) - type(1)) > type(NUMERIC_LIMITS_MIN))
            throw runtime_error("Selection sample is neither a positive nor a negative: "
                                + to_string(data(selection_index, target_index)));
    }

    return negatives;
}


Index DataSet::calculate_testing_negatives(const Index& target_index) const
{
    Index negatives = 0;

    const Index testing_samples_number = get_testing_samples_number();

    const Tensor<Index, 1> testing_indices = get_testing_samples_indices();

    for(Index i = 0; i < Index(testing_samples_number); i++)
    {
        const Index testing_index = testing_indices(i);

        if(data(testing_index, target_index) < type(NUMERIC_LIMITS_MIN))
            negatives++;
    }

    return negatives;
}


Tensor<Descriptives, 1> DataSet::calculate_variables_descriptives() const
{
    return descriptives(data);
}


Tensor<Descriptives, 1> DataSet::calculate_used_variables_descriptives() const
{
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> used_variables_indices = get_used_variables_indices();

    return descriptives(data, used_samples_indices, used_variables_indices);
}


Tensor<Descriptives, 1> DataSet::calculate_raw_variables_descriptives_positive_samples() const
{
    const Index target_index = get_target_variables_indices()(0);

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    const Index samples_number = used_samples_indices.size();

    // Count used positive samples

    Index positive_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = used_samples_indices(i);

        if(abs(data(sample_index, target_index) - type(1)) < type(NUMERIC_LIMITS_MIN)) 
            positive_samples_number++;
    }

    // Get used positive samples indices

    Tensor<Index, 1> positive_used_samples_indices(positive_samples_number);
    Index positive_sample_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = used_samples_indices(i);

        if(abs(data(sample_index, target_index) - type(1)) < type(NUMERIC_LIMITS_MIN))
        {
            positive_used_samples_indices(positive_sample_index) = sample_index;
            positive_sample_index++;
        }
    }

    return descriptives(data, positive_used_samples_indices, input_variables_indices);
}


Tensor<Descriptives, 1> DataSet::calculate_raw_variables_descriptives_negative_samples() const
{
    const Index target_index = get_target_variables_indices()(0);

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    const Index samples_number = used_samples_indices.size();

    // Count used negative samples

    Index negative_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = used_samples_indices(i);

        if(data(sample_index, target_index) < type(NUMERIC_LIMITS_MIN)) 
            negative_samples_number++;
    }

    // Get used negative samples indices

    Tensor<Index, 1> negative_used_samples_indices(negative_samples_number);
    Index negative_sample_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = used_samples_indices(i);

        if(data(sample_index, target_index) < type(NUMERIC_LIMITS_MIN))
        {
            negative_used_samples_indices(negative_sample_index) = sample_index;
            negative_sample_index++;
        }
    }

    return descriptives(data, negative_used_samples_indices, input_variables_indices);
}


Tensor<Descriptives, 1> DataSet::calculate_raw_variables_descriptives_categories(const Index& class_index) const
{
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    const Index samples_number = used_samples_indices.size();

    // Count used class samples

    Index class_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = used_samples_indices(i);

        if(abs(data(sample_index, class_index) - type(1)) < type(NUMERIC_LIMITS_MIN)) 
            class_samples_number++;
    }

    // Get used class samples indices

    Tensor<Index, 1> class_used_samples_indices(class_samples_number);
    class_used_samples_indices.setZero();

    Index class_sample_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = used_samples_indices(i);

        if(abs(data(sample_index, class_index) - type(1)) < type(NUMERIC_LIMITS_MIN))
        {
            class_used_samples_indices(class_sample_index) = sample_index;
            class_sample_index++;
        }
    }

    return descriptives(data, class_used_samples_indices, input_variables_indices);
}


Tensor<Descriptives, 1> DataSet::calculate_raw_variables_descriptives_training_samples() const
{
    const Tensor<Index, 1> training_indices = get_training_samples_indices();

    const Tensor<Index, 1> used_indices = get_used_raw_variables_indices();

    return descriptives(data, training_indices, used_indices);
}


Tensor<Descriptives, 1> DataSet::calculate_raw_variables_descriptives_selection_samples() const
{
    const Tensor<Index, 1> selection_indices = get_selection_samples_indices();

    const Tensor<Index, 1> used_indices = get_used_raw_variables_indices();

    return descriptives(data, selection_indices, used_indices);
}


Tensor<Descriptives, 1> DataSet::calculate_input_variables_descriptives() const
{
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    return descriptives(data, used_samples_indices, input_variables_indices);
}


Tensor<Descriptives, 1> DataSet::calculate_target_variables_descriptives() const
{
    const Tensor<Index, 1> used_indices = get_used_samples_indices();

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    return descriptives(data, used_indices, target_variables_indices);
}


Tensor<Descriptives, 1> DataSet::calculate_testing_target_variables_descriptives() const
{
    const Tensor<Index, 1> testing_indices = get_testing_samples_indices();

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    return descriptives(data, testing_indices, target_variables_indices);
}


Tensor<type, 1> DataSet::calculate_input_variables_minimums() const
{
    return columns_minimums(data, get_used_samples_indices(), get_input_variables_indices());
}


Tensor<type, 1> DataSet::calculate_target_variables_minimums() const
{
    return columns_minimums(data, get_used_samples_indices(), get_target_variables_indices());
}


Tensor<type, 1> DataSet::calculate_input_variables_maximums() const
{
    return columns_maximums(data, get_used_samples_indices(), get_input_variables_indices());
}


Tensor<type, 1> DataSet::calculate_target_variables_maximums() const
{
    return columns_maximums(data, get_used_samples_indices(), get_target_variables_indices());
}


Tensor<type, 1> DataSet::calculate_used_variables_minimums() const
{
    return columns_minimums(data, get_used_samples_indices(), get_used_variables_indices());
}


Tensor<type, 1> DataSet::calculate_used_targets_mean() const
{
    const Tensor<Index, 1> used_indices = get_used_samples_indices();

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    return mean(data, used_indices, target_variables_indices);
}


Tensor<type, 1> DataSet::calculate_selection_targets_mean() const
{
    const Tensor<Index, 1> selection_indices = get_selection_samples_indices();

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    return mean(data, selection_indices, target_variables_indices);
}


Index DataSet::get_gmt() const
{
    return gmt;
}


void DataSet::set_gmt(Index& new_gmt)
{
    gmt = new_gmt;
}


Tensor<Correlation, 2> DataSet::calculate_input_target_raw_variables_correlations() const
{
    const Index input_raw_variables_number = get_input_raw_variables_number();
    const Index target_raw_variables_number = get_target_raw_variables_number();

    const Tensor<Index, 1> input_raw_variables_indices = get_input_raw_variables_indices();
    const Tensor<Index, 1> target_raw_variables_indices = get_target_raw_variables_indices();

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();

    Tensor<Correlation, 2> correlations(input_raw_variables_number, target_raw_variables_number);

//#pragma omp parallel for

    for(Index i = 0; i < input_raw_variables_number; i++)
    {
        const Index input_raw_variable_index = input_raw_variables_indices(i);

        const Tensor<type, 2> input_raw_variable_data 
            = get_raw_variable_data(input_raw_variable_index, used_samples_indices);

        for(Index j = 0; j < target_raw_variables_number; j++)
        {
            const Index target_raw_variable_index = target_raw_variables_indices(j);

            const Tensor<type, 2> target_raw_variable_data 
                = get_raw_variable_data(target_raw_variable_index, used_samples_indices);
            
            correlations(i, j) = correlation(thread_pool_device, input_raw_variable_data, target_raw_variable_data);
        
        }
    }

    return correlations;
}


Tensor<Correlation, 2> DataSet::calculate_input_target_raw_variables_correlations_spearman() const
{
    const Index input_raw_variables_number = get_input_raw_variables_number();
    const Index target_raw_variables_number = get_target_raw_variables_number();

    const Tensor<Index, 1> input_raw_variables_indices = get_input_raw_variables_indices();
    const Tensor<Index, 1> target_raw_variables_indices = get_target_raw_variables_indices();

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();

    Tensor<Correlation, 2> correlations(input_raw_variables_number, target_raw_variables_number);

    for(Index i = 0; i < input_raw_variables_number; i++)
    {
        const Index input_index = input_raw_variables_indices(i);

        const Tensor<type, 2> input_raw_variable_data = get_raw_variable_data(input_index, used_samples_indices);

        for(Index j = 0; j < target_raw_variables_number; j++)
        {
            const Index target_index = target_raw_variables_indices(j);

            const Tensor<type, 2> target_raw_variable_data = get_raw_variable_data(target_index, used_samples_indices);

            correlations(i, j) = correlation_spearman(thread_pool_device, input_raw_variable_data, target_raw_variable_data);
        }
    }

    return correlations;
}


bool DataSet::has_nan() const
{
    const Index rows_number = data.dimension(0);

    for(Index i = 0; i < rows_number; i++)
        if(samples_uses(i) != SampleUse::None)
            if(has_nan_row(i)) 
                return true;

    return false;
}


bool DataSet::has_nan_row(const Index& row_index) const
{
    const Index variables_number = get_variables_number();

    for(Index j = 0; j < variables_number; j++)
        if(isnan(data(row_index,j)))
            return true;

    return false;
}


void DataSet::print_missing_values_information() const
{
    const Index missing_values_number = count_nan();

    const Tensor<Index, 0> raw_variables_with_missing_values = count_raw_variables_with_nan().sum();

    const Index samples_with_missing_values = count_rows_with_nan();

    cout << "Missing values number: " << missing_values_number << " (" << missing_values_number*100/data.size() << "%)" << endl
         << "Raw variables with missing values: " << raw_variables_with_missing_values(0)
         << " (" << raw_variables_with_missing_values(0)*100/data.dimension(1) << "%)" << endl
         << "Samples with missing values: "
         << samples_with_missing_values << " (" << samples_with_missing_values*100/data.dimension(0) << "%)" << endl;
}


void DataSet::print_input_target_raw_variables_correlations() const
{
    const Index inputs_number = get_input_variables_number();
    const Index targets_number = get_target_raw_variables_number();

    const Tensor<string, 1> inputs_name = get_input_raw_variable_names();
    const Tensor<string, 1> targets_name = get_target_raw_variables_names();

    const Tensor<Correlation, 2> correlations = calculate_input_target_raw_variables_correlations();

    for(Index j = 0; j < targets_number; j++)
        for(Index i = 0; i < inputs_number; i++)
            cout << targets_name(j) << " - " << inputs_name(i) << ": " << correlations(i, j).r << endl;
}


void DataSet::print_top_input_target_raw_variables_correlations() const
{
    const Index inputs_number = get_input_raw_variables_number();
    const Index targets_number = get_target_raw_variables_number();

    const Tensor<string, 1> inputs_name = get_input_variables_names();
    const Tensor<string, 1> targets_name = get_target_variables_names();

    const Tensor<type, 2> correlations = get_correlation_values(calculate_input_target_raw_variables_correlations());

    Tensor<type, 1> target_correlations(inputs_number);

    Tensor<string, 2> top_correlations(inputs_number, 2);

    map<type,string> top_correlation;

    for(Index i = 0 ; i < inputs_number; i++)
        for(Index j = 0 ; j < targets_number ; j++)
            top_correlation.insert(pair<type,string>(correlations(i, j), inputs_name(i) + " - " + targets_name(j)));

    map<type,string>::iterator it;

    for(it = top_correlation.begin(); it != top_correlation.end(); it++)
        cout << "Correlation: " << (*it).first << "  between  " << (*it).second << "" << endl;
}


Tensor<Tensor<Correlation, 2>, 1> DataSet::calculate_input_raw_variable_correlations(const bool& calculate_pearson_correlations,
                                                                                     const bool& calculate_spearman_correlations) const
{
    // list to return

    Tensor<Tensor<Correlation, 2>, 1> correlations_list(2);

    const Tensor<Index, 1> input_raw_variables_indices = get_input_raw_variables_indices();

    const Index input_raw_variables_number = get_input_raw_variables_number();

    Tensor<Correlation, 2> correlations_pearson(input_raw_variables_number, input_raw_variables_number);
    Tensor<Correlation, 2> correlations_spearman(input_raw_variables_number, input_raw_variables_number);

    for(Index i = 0; i < input_raw_variables_number; i++)
    {
        const Index current_input_index_i = input_raw_variables_indices(i);

        const Tensor<type, 2> input_i = get_raw_variable_data(current_input_index_i);

        //if(display) cout << "Calculating " << raw_variables(current_input_index_i).name << " correlations. " << endl;

        for(Index j = i; j < input_raw_variables_number; j++)
        {
            if(j == i)
            {
                if(calculate_pearson_correlations)
                {
                    correlations_pearson(i, j).r = type(1);
                    correlations_pearson(i, j).b = type(1);
                    correlations_pearson(i, j).a = type(0);

                    correlations_pearson(i, j).upper_confidence = type(1);
                    correlations_pearson(i, j).lower_confidence = type(1);
                    correlations_pearson(i, j).form = Correlation::Form::Linear;
                    correlations_pearson(i, j).method = Correlation::Method::Pearson;

                    if(is_constant_matrix(input_i))
                    {
                        correlations_pearson(i, j).r = NAN;
                        correlations_pearson(i, j).b = NAN;
                        correlations_pearson(i, j).a = NAN;
                    }
                }

                if(calculate_spearman_correlations)
                {
                    correlations_spearman(i, j).r = type(1);
                    correlations_spearman(i, j).b = type(1);
                    correlations_spearman(i, j).a = type(0);

                    correlations_spearman(i, j).upper_confidence = type(1);
                    correlations_spearman(i, j).lower_confidence = type(1);
                    correlations_spearman(i, j).form = Correlation::Form::Linear;
                    correlations_spearman(i, j).method = Correlation::Method::Spearman;

                    if(is_constant_matrix(input_i))
                    {
                        correlations_spearman(i, j).r = NAN;
                        correlations_spearman(i, j).b = NAN;
                        correlations_spearman(i, j).a = NAN;
                    }
                }
            }
            else
            {
                const Index current_input_index_j = input_raw_variables_indices(j);

                const Tensor<type, 2> input_j = get_raw_variable_data(current_input_index_j);

                if(calculate_pearson_correlations)
                {
                    correlations_pearson(i, j) = correlation(thread_pool_device, input_i, input_j);

                    if(correlations_pearson(i, j).r > type(1) - NUMERIC_LIMITS_MIN)
                       correlations_pearson(i, j).r =  type(1);
                }

                if(calculate_spearman_correlations)
                {
                    correlations_spearman(i, j) = correlation_spearman(thread_pool_device, input_i, input_j);

                    if(correlations_spearman(i, j).r > type(1) - NUMERIC_LIMITS_MIN)
                        correlations_spearman(i, j).r = type(1);
                }
            }
        }
    }

    if(calculate_pearson_correlations)
        for(Index i = 0; i < input_raw_variables_number; i++)
            for(Index j = 0; j < i; j++)
                correlations_pearson(i, j) = correlations_pearson(j,i);

    if(calculate_spearman_correlations)
        for(Index i = 0; i < input_raw_variables_number; i++)
            for(Index j = 0; j < i; j++)
                correlations_spearman(i, j) = correlations_spearman(j,i);

    correlations_list(0) = correlations_pearson;
    correlations_list(1) = correlations_spearman;

    return correlations_list;
}

/*
Tensor<Tensor<Correlation, 2>, 1> DataSet::calculate_input_raw_variable_correlations(const bool& calculate_pearson_correlations,
    const bool& calculate_spearman_correlations) const
{
    // list to return

    Tensor<Tensor<Correlation, 2>, 1> correlations_list(2);

    const Tensor<Index, 1> input_raw_variables_indices = get_input_raw_variables_indices();

    const Index input_raw_variables_number = get_input_raw_variables_number();

    Tensor<Correlation, 2> correlations_pearson(input_raw_variables_number, input_raw_variables_number);
    Tensor<Correlation, 2> correlations_spearman(input_raw_variables_number, input_raw_variables_number);

    for (Index i = 0; i < input_raw_variables_number; i++)
    {
        const Index current_input_index_i = input_raw_variables_indices(i);

        const Tensor<type, 2> input_i = get_raw_variable_data(current_input_index_i);

        //if(display) cout << "Calculating " << raw_variables(current_input_index_i).name << " correlations. " << endl;

        for (Index j = i; j < input_raw_variables_number; j++)
        {
            if (j == i)
            {
                if (calculate_pearson_correlations)
                {
                    correlations_pearson(i, j).r = type(1);
                    correlations_pearson(i, j).b = type(1);
                    correlations_pearson(i, j).a = type(0);

                    correlations_pearson(i, j).upper_confidence = type(1);
                    correlations_pearson(i, j).lower_confidence = type(1);
                    correlations_pearson(i, j).form = Correlation::Form::Linear;
                    correlations_pearson(i, j).method = Correlation::Method::Pearson;

                    if (is_constant_matrix(input_i))
                    {
                        correlations_pearson(i, j).r = NAN;
                        correlations_pearson(i, j).b = NAN;
                        correlations_pearson(i, j).a = NAN;
                    }
                }

                if (calculate_spearman_correlations)
                {
                    correlations_spearman(i, j).r = type(1);
                    correlations_spearman(i, j).b = type(1);
                    correlations_spearman(i, j).a = type(0);

                    correlations_spearman(i, j).upper_confidence = type(1);
                    correlations_spearman(i, j).lower_confidence = type(1);
                    correlations_spearman(i, j).form = Correlation::Form::Linear;
                    correlations_spearman(i, j).method = Correlation::Method::Spearman;

                    if (is_constant_matrix(input_i))
                    {
                        correlations_spearman(i, j).r = NAN;
                        correlations_spearman(i, j).b = NAN;
                        correlations_spearman(i, j).a = NAN;
                    }
                }
            }
            else
            {
                const Index current_input_index_j = input_raw_variables_indices(j);

                const Tensor<type, 2> input_j = get_raw_variable_data(current_input_index_j);

                if (calculate_pearson_correlations)
                {
                    correlations_pearson(i, j) = correlation(thread_pool_device, input_i, input_j);

                    if (correlations_pearson(i, j).r > type(1) - NUMERIC_LIMITS_MIN)
                        correlations_pearson(i, j).r = type(1);
                }

                if (calculate_spearman_correlations)
                {
                    correlations_spearman(i, j) = correlation_spearman(thread_pool_device, input_i, input_j);

                    if (correlations_spearman(i, j).r > type(1) - NUMERIC_LIMITS_MIN)
                        correlations_spearman(i, j).r = type(1);
                }
            }
        }
    }

    if (calculate_pearson_correlations)
        for (Index i = 0; i < input_raw_variables_number; i++)
            for (Index j = 0; j < i; j++)
                correlations_pearson(i, j) = correlations_pearson(j, i);

    if (calculate_spearman_correlations)
        for (Index i = 0; i < input_raw_variables_number; i++)
            for (Index j = 0; j < i; j++)
                correlations_spearman(i, j) = correlations_spearman(j, i);

    correlations_list(0) = correlations_pearson;
    correlations_list(1) = correlations_spearman;

    return correlations_list;
}
*/

void DataSet::print_inputs_correlations() const
{
    const Tensor<type, 2> inputs_correlations = get_correlation_values(calculate_input_raw_variable_correlations()(0));

    cout << inputs_correlations << endl;
}


void DataSet::print_data_file_preview() const
{
    const Index size = data_file_preview.size();

    for(Index i = 0;  i < size; i++)
    {
        for(Index j = 0; j < data_file_preview(i).size(); j++)
            cout << data_file_preview(i)(j) << " ";

        cout << endl;
    }
}


void DataSet::print_top_inputs_correlations() const
{
    const Index variables_number = get_input_variables_number();

    const Tensor<string, 1> variables_name = get_input_variables_names();

    const Tensor<type, 2> variables_correlations = get_correlation_values(calculate_input_raw_variable_correlations()(0));

    const Index correlations_number = variables_number*(variables_number-1)/2;

    Tensor<string, 2> top_correlations(correlations_number, 3);

    map<type, string> top_correlation;

    for(Index i = 0; i < variables_number; i++)
    {
        for(Index j = i; j < variables_number; j++)
        {
            if(i == j) continue;

            top_correlation.insert(pair<type,string>(variables_correlations(i, j), variables_name(i) + " - " + variables_name(j)));
        }
    }

    map<type,string> ::iterator it;

    for(it = top_correlation.begin(); it != top_correlation.end(); it++)
        cout << "Correlation: " << (*it).first << "  between  " << (*it).second << "" << endl;
}


void DataSet::set_default_raw_variables_scalers()
{
    if(model_type == ModelType::ImageClassification)
    {
        set_raw_variables_scalers(Scaler::ImageMinMax);
    }
    else
    {
        const Index raw_variables_number = raw_variables.size();

        for(Index i = 0; i < raw_variables_number; i++)
            raw_variables(i).scaler = (raw_variables(i).type == RawVariableType::Numeric)
                ? Scaler::MeanStandardDeviation
                : Scaler::MinimumMaximum;
    }
}


Tensor<Descriptives, 1> DataSet::scale_data()
{
    const Index variables_number = get_variables_number();

    const Tensor<Descriptives, 1> variables_descriptives = calculate_variables_descriptives();

    Index raw_variable_index;

    for(Index i = 0; i < variables_number; i++)
    {
        raw_variable_index = get_raw_variable_index(i);

        switch(raw_variables(raw_variable_index).scaler)
        {
        case Scaler::None:
            break;

        case Scaler::MinimumMaximum:
            scale_minimum_maximum(data, i, variables_descriptives(i));
            break;

        case Scaler::MeanStandardDeviation:
            scale_mean_standard_deviation(data, i, variables_descriptives(i));
            break;

        case Scaler::StandardDeviation:
            scale_standard_deviation(data, i, variables_descriptives(i));
            break;

        case Scaler::Logarithm:
            scale_logarithmic(data, i);
            break;

        default:
            throw runtime_error("Unknown scaler: " + to_string(int(raw_variables(i).scaler)) + "\n");
        }
    }

    return variables_descriptives;
}


void DataSet::unscale_data(const Tensor<Descriptives, 1>& variables_descriptives)
{
    const Index variables_number = get_variables_number();

    for(Index i = 0; i < variables_number; i++)
    {
        switch(raw_variables(i).scaler)
        {
        case Scaler::None:
            break;

        case Scaler::MinimumMaximum:
            unscale_minimum_maximum(data, i, variables_descriptives(i));
            break;

        case Scaler::MeanStandardDeviation:
            unscale_mean_standard_deviation(data, i, variables_descriptives(i));
            break;

        case Scaler::StandardDeviation:
            unscale_standard_deviation(data, i, variables_descriptives(i));
            break;

        case Scaler::Logarithm:
            unscale_logarithmic(data, i);
            break;

        default:
            throw runtime_error("Unknown scaler: " + to_string(int(raw_variables(i).scaler)) + "\n");
        }
    }
}


Tensor<Descriptives, 1> DataSet::scale_input_variables()
{
    const Index input_variables_number = get_input_variables_number();

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();
    const Tensor<Scaler, 1> input_variables_scalers = get_input_variables_scalers();

    const Tensor<Descriptives, 1> input_variables_descriptives = calculate_input_variables_descriptives();

    for(Index i = 0; i < input_variables_number; i++)
    {
        switch(input_variables_scalers(i))
        {
        case Scaler::None:
            break;

        case Scaler::MinimumMaximum:
            scale_minimum_maximum(data, input_variables_indices(i), input_variables_descriptives(i));
            break;

        case Scaler::MeanStandardDeviation:
            scale_mean_standard_deviation(data, input_variables_indices(i), input_variables_descriptives(i));
            break;

        case Scaler::StandardDeviation:
            scale_standard_deviation(data, input_variables_indices(i), input_variables_descriptives(i));
            break;

        case Scaler::Logarithm:
            scale_logarithmic(data, input_variables_indices(i));
            break;

        case Scaler::ImageMinMax:
            scale_image_minimum_maximum(data, input_variables_indices(i));
            break;

        default:
            throw runtime_error("Unknown scaling inputs method: " + to_string(int(input_variables_scalers(i))) + "\n");
        }
    }

    return input_variables_descriptives;
}


Tensor<Descriptives, 1> DataSet::scale_target_variables()
{
    const Index target_variables_number = get_target_variables_number();

    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();
    const Tensor<Scaler, 1> target_variables_scalers = get_target_variables_scalers();

    const Tensor<Descriptives, 1> target_variables_descriptives = calculate_target_variables_descriptives();

    for(Index i = 0; i < target_variables_number; i++)
    {
        switch(target_variables_scalers(i))
        {
        case Scaler::None:
            break;

        case Scaler::MinimumMaximum:
            scale_minimum_maximum(data, target_variables_indices(i), target_variables_descriptives(i));
            break;

        case Scaler::MeanStandardDeviation:
            scale_mean_standard_deviation(data, target_variables_indices(i), target_variables_descriptives(i));
            break;

        case Scaler::StandardDeviation:
            scale_standard_deviation(data, target_variables_indices(i), target_variables_descriptives(i));
            break;

        case Scaler::Logarithm:
            scale_logarithmic(data, target_variables_indices(i));
            break;

        default:        
            throw runtime_error("Unknown scaling targets method: " + to_string(int(target_variables_scalers(i))) + "\n");
        }
    }

    return target_variables_descriptives;
}


void DataSet::unscale_input_variables(const Tensor<Descriptives, 1>& input_variables_descriptives)
{
    const Index input_variables_number = get_input_variables_number();

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    const Tensor<Scaler, 1> input_variables_scalers = get_input_variables_scalers();

    for(Index i = 0; i < input_variables_number; i++)
    {
        switch(input_variables_scalers(i))
        {
        case Scaler::None:
            break;

        case Scaler::MinimumMaximum:
            unscale_minimum_maximum(data, input_variables_indices(i), input_variables_descriptives(i));
            break;

        case Scaler::MeanStandardDeviation:
            unscale_mean_standard_deviation(data, input_variables_indices(i), input_variables_descriptives(i));
            break;

        case Scaler::StandardDeviation:
            unscale_standard_deviation(data, input_variables_indices(i), input_variables_descriptives(i));
            break;

        case Scaler::Logarithm:
            unscale_logarithmic(data, input_variables_indices(i));
            break;

        case Scaler::ImageMinMax:
            unscale_image_minimum_maximum(data, input_variables_indices(i));
            break;

        default:
            throw runtime_error("Unknown unscaling and unscaling method: " + to_string(int(input_variables_scalers(i))) + "\n");
        }
    }
}


void DataSet::unscale_target_variables(const Tensor<Descriptives, 1>& targets_descriptives)
{
    const Index target_variables_number = get_target_variables_number();
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();
    const Tensor<Scaler, 1> target_variables_scalers = get_target_variables_scalers();

    for(Index i = 0; i < target_variables_number; i++)
    {
        switch(target_variables_scalers(i))
        {
        case Scaler::None:
            break;

        case Scaler::MinimumMaximum:
            unscale_minimum_maximum(data, target_variables_indices(i), targets_descriptives(i));
            break;

        case Scaler::MeanStandardDeviation:
            unscale_mean_standard_deviation(data, target_variables_indices(i), targets_descriptives(i));
            break;

        case Scaler::StandardDeviation:
            unscale_standard_deviation(data, target_variables_indices(i), targets_descriptives(i));
            break;

        case Scaler::Logarithm:
            unscale_logarithmic(data, target_variables_indices(i));
            break;

        default:
            throw runtime_error("Unknown unscaling and unscaling method.\n");
        }
    }
}


void DataSet::set_data_constant(const type& new_value)
{
    data.setConstant(new_value);
    data.dimensions();
}


void DataSet::set_data_random()
{
    set_random(data);
}


void DataSet::set_data_binary_random()
{
    set_random(data);

    const Index samples_number = data.dimension(0);
    const Index variables_number = data.dimension(1);

    const Index input_variables_number = get_input_variables_number();
    const Index target_variables_number = variables_number - input_variables_number;

    Index target_variable_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        target_variable_index = (target_variables_number == 1)
            ? rand() % 2
            : rand() % (variables_number - input_variables_number) + input_variables_number;

        for(Index j = input_variables_number; j < variables_number; j++)
        {
            if(target_variables_number == 1) 
                data(i, j) = type(target_variable_index);
            else 
                data(i, j) = (j == target_variable_index)
                        ? type(1)
                        : type(0);
        }
    }
}


void DataSet::to_XML(tinyxml2::XMLPrinter& file_stream) const
{
    if(model_type == ModelType::Forecasting)
        throw runtime_error("Forecasting");

    if(model_type == ModelType::ImageClassification)
        throw runtime_error("Image classification");

    file_stream.OpenElement("DataSet");

    // Data file

    file_stream.OpenElement("DataSource");

    file_stream.OpenElement("FileType");
    file_stream.PushText("csv");
    file_stream.CloseElement();

    // Data file name

    file_stream.OpenElement("Path");
    file_stream.PushText(data_path.c_str());
    file_stream.CloseElement();

    // Separator

    file_stream.OpenElement("Separator");
    file_stream.PushText(get_separator_name().c_str());
    file_stream.CloseElement();

    // Has header

    file_stream.OpenElement("HasHeader");
    file_stream.PushText(to_string(has_header).c_str());
    file_stream.CloseElement();

    // Has Ids

    file_stream.OpenElement("HasSamplesId");
    file_stream.PushText(to_string(has_samples_id).c_str());
    file_stream.CloseElement();

    // Missing values label

    file_stream.OpenElement("MissingValuesLabel");
    file_stream.PushText(missing_values_label.c_str());
    file_stream.CloseElement();

    // Codification

    file_stream.OpenElement("Codification");
    file_stream.PushText(get_codification_string().c_str());
    file_stream.CloseElement();

    // Close Data source

    file_stream.CloseElement();

    // Raw variables

    file_stream.OpenElement("RawVariables");

    // Raw variables number

    file_stream.OpenElement("RawVariablesNumber");
    file_stream.PushText(to_string(get_raw_variables_number()).c_str());
    file_stream.CloseElement();

    // Raw variables items

    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
    {
        file_stream.OpenElement("RawVariable");
        file_stream.PushAttribute("Item", to_string(i+1).c_str());
        raw_variables(i).to_XML(file_stream);
        file_stream.CloseElement();
    }

    // Close raw variables

    file_stream.CloseElement();

    // Samples

    file_stream.OpenElement("Samples");

    // Samples number

    file_stream.OpenElement("SamplesNumber");
    file_stream.PushText(to_string(get_samples_number()).c_str());
    file_stream.CloseElement();

    // Samples id

    if(has_samples_id)
    {
        file_stream.OpenElement("SamplesId");
        file_stream.PushText(string_tensor_to_string(samples_id).c_str());
        file_stream.CloseElement();
    }

    // Samples uses

    file_stream.OpenElement("SamplesUses");
    file_stream.PushText(tensor_to_string(get_samples_uses_tensor()).c_str());
    file_stream.CloseElement();

    // Close samples

    file_stream.CloseElement();

    // Missing values

    file_stream.OpenElement("MissingValues");

    // Missing values method

    file_stream.OpenElement("MissingValuesMethod");
    file_stream.PushText(get_missing_values_method_string().c_str());
    file_stream.CloseElement();

    // Missing values number

    file_stream.OpenElement("MissingValuesNumber");
    file_stream.PushText(to_string(missing_values_number).c_str());
    file_stream.CloseElement();

    if(missing_values_number > 0)
    {
        // Raw variables missing values number

        file_stream.OpenElement("RawVariablesMissingValuesNumber");
        file_stream.PushText(tensor_to_string(raw_variables_missing_values_number).c_str());
        file_stream.CloseElement();

        // Rows missing values number

        file_stream.OpenElement("RowsMissingValuesNumber");
        file_stream.PushText(to_string(rows_missing_values_number).c_str());
        file_stream.CloseElement();
    }

    // Missing values

    file_stream.CloseElement();
/*
    // Preview data

    file_stream.OpenElement("PreviewData");

    file_stream.OpenElement("PreviewSize");

    buffer << data_file_preview.size();

    file_stream.PushText(buffer.str().c_str());

    file_stream.CloseElement();

    for(Index i = 0; i < data_file_preview.size(); i++)
    {
        file_stream.OpenElement("Row");

        file_stream.PushAttribute("Item", to_string(i+1).c_str());

        for(Index j = 0; j < data_file_preview(i).size(); j++)
        {
            file_stream.PushText(data_file_preview(i)(j).c_str());

            if(j != data_file_preview(i).size()-1)
                file_stream.PushText(",");
        }

        file_stream.CloseElement();
    }

    // Close preview data

    file_stream.CloseElement();
*/
    // Close data set

    file_stream.CloseElement();
}


void DataSet::from_XML(const tinyxml2::XMLDocument& data_set_document)
{
    // Data set element

    const tinyxml2::XMLElement* data_set_element = data_set_document.FirstChildElement("DataSet");

    if(!data_set_element)
        throw runtime_error("Data set element is nullptr.\n");

    // Data file

    const tinyxml2::XMLElement* data_source_element = data_set_element->FirstChildElement("DataSource");

    if(!data_source_element)
        throw runtime_error("Data source element is nullptr.\n");

    // Data file name

    const tinyxml2::XMLElement* data_source_path_element = data_source_element->FirstChildElement("Path");

    if(!data_source_path_element)
        throw runtime_error("Path element is nullptr.\n");

    if(data_source_path_element->GetText())
        set_data_source_path(data_source_path_element->GetText());

    // Separator

    const tinyxml2::XMLElement* separator_element = data_source_element->FirstChildElement("Separator");

    if(separator_element)
        if(separator_element->GetText())
            set_separator_name(separator_element->GetText());

    // Has raw_variables names

    const tinyxml2::XMLElement* raw_variables_names_element = data_source_element->FirstChildElement("HasHeader");

    if(raw_variables_names_element)
    {
        const string new_raw_variables_names_string = raw_variables_names_element->GetText();

        try
        {
            set_has_header(new_raw_variables_names_string == "1");
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Samples id

    const tinyxml2::XMLElement* rows_label_element = data_source_element->FirstChildElement("HasSamplesId");

    if(rows_label_element)
    {
        const string new_rows_label_string = rows_label_element->GetText();

        try
        {
            set_has_ids(new_rows_label_string == "1");
        }
        catch(const exception& e)
        {
            cerr << e.what() << endl;
        }
    }

    // Missing values label

    const tinyxml2::XMLElement* missing_values_label_element = data_source_element->FirstChildElement("MissingValuesLabel");

    if(missing_values_label_element)
        if(missing_values_label_element->GetText())
            set_missing_values_label(missing_values_label_element->GetText());

    // Codification

    const tinyxml2::XMLElement* codification_element = data_source_element->FirstChildElement("Codification");

    if(codification_element)
        if(codification_element->GetText())
            set_codification(codification_element->GetText());

    // Raw variables

    const tinyxml2::XMLElement* raw_variables_element = data_set_element->FirstChildElement("RawVariables");

    if(!raw_variables_element)
        throw runtime_error("RawVariables element is nullptr.\n");

    // Raw variables number

    const tinyxml2::XMLElement* raw_variables_number_element = raw_variables_element->FirstChildElement("RawVariablesNumber");

    if(!raw_variables_number_element)
        throw runtime_error("RawVariablesNumber element is nullptr.\n");

    if(raw_variables_number_element->GetText())
        set_raw_variables_number(Index(atoi(raw_variables_number_element->GetText())));

    // Raw variables

    const tinyxml2::XMLElement* start_element = raw_variables_number_element;

    for(Index i = 0; i < raw_variables.size(); i++)
    {
        const tinyxml2::XMLElement* raw_variable_element = start_element->NextSiblingElement("RawVariable");
        start_element = raw_variable_element;

        if(raw_variable_element->Attribute("Item") != to_string(i+1))
            throw runtime_error("Raw variable item number (" + to_string(i+1) + ") does not match (" + raw_variable_element->Attribute("Item") + ").\n");

        // Name

        const tinyxml2::XMLElement* name_element = raw_variable_element->FirstChildElement("Name");

        if(!name_element)
            throw runtime_error("Name element is nullptr.\n");

        if(name_element->GetText())
            raw_variables(i).name = name_element->GetText();

        // Scaler

        const tinyxml2::XMLElement* scaler_element = raw_variable_element->FirstChildElement("Scaler");

        if(!scaler_element)
            throw runtime_error("Scaler element is nullptr.\n");

        if(scaler_element->GetText())
            raw_variables(i).set_scaler(scaler_element->GetText());

        // raw_variable use

        const tinyxml2::XMLElement* use_element = raw_variable_element->FirstChildElement("Use");

        if(!use_element)
            throw runtime_error("RawVariableUse element is nullptr.\n");

        if(use_element->GetText())
            raw_variables(i).set_use(use_element->GetText());

        // Type

        const tinyxml2::XMLElement* type_element = raw_variable_element->FirstChildElement("Type");

        if(!type_element)
            throw runtime_error("Type element is nullptr.\n");

        if(type_element->GetText())
            raw_variables(i).set_type(type_element->GetText());

        if(raw_variables(i).type == RawVariableType::Categorical
        || raw_variables(i).type == RawVariableType::Binary)
        {
            // Categories

            const tinyxml2::XMLElement* categories_element = raw_variable_element->FirstChildElement("Categories");

            if(!categories_element)
                throw runtime_error("Categories element is nullptr.\n");

            if(categories_element->GetText())
                raw_variables(i).categories = get_tokens(categories_element->GetText(), ";");
        }
    }

    // Has Ids

    if(has_samples_id)
    {
        // Samples id begin tag

        const tinyxml2::XMLElement* has_ids_element = data_set_element->FirstChildElement("HasSamplesId");

        if(!has_ids_element)
            throw runtime_error("HasSamplesId element is nullptr.\n");

        // Ids

        if(has_ids_element->GetText())
            samples_id = get_tokens(has_ids_element->GetText(), " ");
    }

    // Samples

    const tinyxml2::XMLElement* samples_element = data_set_element->FirstChildElement("Samples");

    if(!samples_element)
        throw runtime_error("Samples element is nullptr.\n");

    // Samples number

    const tinyxml2::XMLElement* samples_number_element = samples_element->FirstChildElement("SamplesNumber");

    if(!samples_number_element)
        throw runtime_error("Samples number element is nullptr.\n");

    if(samples_number_element->GetText())
        samples_uses.resize(Index(atoi(samples_number_element->GetText())));

    // Samples uses

    const tinyxml2::XMLElement* samples_uses_element = samples_element->FirstChildElement("SamplesUses");

    if(!samples_uses_element)
        throw runtime_error("Samples uses element is nullptr.\n");

    if(samples_uses_element->GetText())
        set_sample_uses(get_tokens(samples_uses_element->GetText(), " "));

    // Missing values

    const tinyxml2::XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");

    if(!missing_values_element)
        throw runtime_error("Missing values element is nullptr.\n");

    // Missing values method

    const tinyxml2::XMLElement* missing_values_method_element = missing_values_element->FirstChildElement("MissingValuesMethod");

    if(!missing_values_method_element)
        throw runtime_error("Missing values method element is nullptr.\n");

    if(missing_values_method_element->GetText())
        set_missing_values_method(missing_values_method_element->GetText());

    // Missing values number

    const tinyxml2::XMLElement* missing_values_number_element = missing_values_element->FirstChildElement("MissingValuesNumber");

    if(!missing_values_number_element)
        throw runtime_error("Missing values number element is nullptr.\n");

    if(missing_values_number_element->GetText())
        missing_values_number = Index(atoi(missing_values_number_element->GetText()));

    if(missing_values_number > 0)
    {
        // Raw variables Missing values number

        const tinyxml2::XMLElement* raw_variables_missing_values_number_element
                = missing_values_element->FirstChildElement("RawVariablesMissingValuesNumber");

        if(!raw_variables_missing_values_number_element)
            throw runtime_error("RawVariablesMissingValuesNumber element is nullptr.\n");

        if(raw_variables_missing_values_number_element->GetText())
        {
            Tensor<string, 1> new_raw_variables_missing_values_number
                    = get_tokens(raw_variables_missing_values_number_element->GetText(), " ");

            raw_variables_missing_values_number.resize(new_raw_variables_missing_values_number.size());

            for(Index i = 0; i < new_raw_variables_missing_values_number.size(); i++)
                raw_variables_missing_values_number(i) = atoi(new_raw_variables_missing_values_number(i).c_str());
        }

        // Rows missing values number

        const tinyxml2::XMLElement* rows_missing_values_number_element = missing_values_element->FirstChildElement("RowsMissingValuesNumber");

        if(!rows_missing_values_number_element)
            throw runtime_error("Rows missing values number element is nullptr.\n");

        if(rows_missing_values_number_element->GetText())
            rows_missing_values_number = Index(atoi(rows_missing_values_number_element->GetText()));
    }
/*
    // Preview data

    const tinyxml2::XMLElement* preview_data_element = data_set_element->FirstChildElement("PreviewData");

    if(!preview_data_element)
        throw runtime_error("Preview data element is nullptr.\n");

    // Preview size

    const tinyxml2::XMLElement* preview_size_element = preview_data_element->FirstChildElement("PreviewSize");

    if(!preview_size_element)
        throw runtime_error("Preview size element is nullptr.\n");

    Index new_preview_size = 0;

    if(preview_size_element->GetText())
    {
        new_preview_size = Index(atoi(preview_size_element->GetText()));

        if(new_preview_size > 0) data_file_preview.resize(new_preview_size);
    }

    // Preview data

    start_element = preview_size_element;

    for(Index i = 0; i < new_preview_size; i++)
    {
        const tinyxml2::XMLElement* row_element = start_element->NextSiblingElement("Row");
        start_element = row_element;

        if(row_element->Attribute("Item") != to_string(i+1))
            throw runtime_error("Row item number (" + to_string(i+1) + ") "
                                "does not match (" + row_element->Attribute("Item") + ").\n");

        if(row_element->GetText())
        {
            data_file_preview(i) = get_tokens(row_element->GetText(), ",");
        }
    }
*/
    // Display

    const tinyxml2::XMLElement* display_element = data_set_element->FirstChildElement("Display");

    if(display_element)
        set_display(display_element->GetText() != string("0"));
}


void DataSet::print() const
{
    if(!display) return;
    
    const Index variables_number = get_variables_number();
    const Index input_variables_number = get_input_variables_number();
    const Index samples_number = get_samples_number();
    const Index target_variables_bumber = get_target_variables_number();
    const Index training_samples_number = get_training_samples_number();
    const Index selection_samples_number = get_selection_samples_number();
    const Index testing_samples_number = get_testing_samples_number();
    const Index unused_samples_number = get_unused_samples_number();

    cout << "Data set object summary:\n"
         << "Number of samples: " << samples_number << "\n"
         << "Number of variables: " << variables_number << "\n"
         << "Number of input variables: " << input_variables_number << "\n"
         << "Number of target variables: " << target_variables_bumber << "\n"
         << "Input variables dimensions: ";
   
    print_dimensions(input_dimensions);
         
    cout << "Target variables dimensions: ";
    
    print_dimensions(target_dimensions);
    
    cout << "Number of training samples: " << training_samples_number << endl
         << "Number of selection samples: " << selection_samples_number << endl
         << "Number of testing samples: " << testing_samples_number << endl
         << "Number of unused samples: " << unused_samples_number << endl;
   
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
    {
        cout << endl;
        raw_variables(i).print();
    }
}


void DataSet::save(const string& file_name) const
{
    FILE* pFile = fopen(file_name.c_str(), "w");

    tinyxml2::XMLPrinter document(pFile);

    to_XML(document);

    fclose(pFile);
}


void DataSet::load(const string& file_name)
{
    tinyxml2::XMLDocument document;

    if(document.LoadFile(file_name.c_str()))
        throw runtime_error("Cannot load XML file " + file_name + ".\n");

    from_XML(document);
}


void DataSet::print_raw_variables() const
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
    {
        raw_variables(i).print();
        cout << endl;
    }

    cout << endl;
}


void DataSet::print_data() const
{
    if(display) cout << data << endl;
}


void DataSet::print_data_preview() const
{
    if(!display) return;

    const Index samples_number = get_samples_number();

    if(samples_number > 0)
    {
        const Tensor<type, 1> first_sample = data.chip(0, 0);

        cout << "First sample: \n";

        for(int i = 0; i< first_sample.dimension(0); i++)
            cout  << first_sample(i) << "  ";

        cout << endl;
    }

    if(samples_number > 1)
    {
        const Tensor<type, 1> second_sample = data.chip(1, 0);

        cout << "Second sample: \n";

        for(int i = 0; i< second_sample.dimension(0); i++)
            cout  << second_sample(i) << "  ";

        cout << endl;
    }

    if(samples_number > 2)
    {
        const Tensor<type, 1> last_sample = data.chip(samples_number-1, 0);

        cout << "Last sample: \n";

        for(int i = 0; i< last_sample.dimension(0); i++)
            cout  << last_sample(i) << "  ";

        cout << endl;
    }
}


void DataSet::save_data() const
{
    ofstream file(data_path.c_str());

    if(!file.is_open())
        throw runtime_error("Cannot open matrix data file: " + data_path + "\n");

    file.precision(20);

    const Index samples_number = get_samples_number();
    const Index variables_number = get_variables_number();

    const Tensor<string, 1> variables_names = get_variables_names();

    const string separator_string = get_separator_string();

    if(has_samples_id)
        file << "id" << separator_string;

    for(Index j = 0; j < variables_number; j++)
    {
        file << variables_names[j];

        if(j != variables_number-1)
            file << separator_string;
    }

    file << endl;

    for(Index i = 0; i < samples_number; i++)
    {
        if(has_samples_id)
            file << samples_id(i) << separator_string;

        for(Index j = 0; j < variables_number; j++)
        {
            file << data(i, j);

            if(j != variables_number-1)
                file << separator_string;
        }

        file << endl;
    }

    file.close();
}


void DataSet::save_data_binary(const string& binary_data_file_name) const
{
    ofstream file;

    open_file(binary_data_file_name, file);

    if(!file.is_open())
        throw runtime_error("Cannot open data binary file.");

    // Write data

    streamsize size = sizeof(Index);

    Index columns_number = data.dimension(1);
    Index rows_number = data.dimension(0);

    cout << "Saving binary data file..." << endl;

    file.write(reinterpret_cast<char*>(&columns_number), size);
    file.write(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(type);

    const Index total_elements = columns_number * rows_number;

    file.write(reinterpret_cast<const char*>(data.data()), total_elements * size);

    file.close();

    cout << "Binary data file saved." << endl;
}


void DataSet::load_data_binary()
{
    ifstream file;

    open_file(data_path, file);

    streamsize size = sizeof(Index);

    Index columns_number = 0;
    Index rows_number = 0;

    file.read(reinterpret_cast<char*>(&columns_number), size);
    file.read(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(type);

    data.resize(rows_number, columns_number);

    const Index total_elements = rows_number * columns_number;

    file.read(reinterpret_cast<char*>(data.data()), total_elements * size);

    file.close();
}


Tensor<Index, 1> DataSet::calculate_target_distribution() const
{
    const Index samples_number = get_samples_number();
    const Index targets_number = get_target_variables_number();
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    Tensor<Index, 1> class_distribution;

    if(targets_number == 1) 
    {
        class_distribution.resize(2);

        const Index target_index = target_variables_indices(0);

        Index positives = 0;
        Index negatives = 0;

        for(Index sample_index = 0; sample_index < samples_number; sample_index++)
            if(!isnan(data(sample_index,target_index)))
                (data(sample_index, target_index) < type(0.5)) 
                    ? negatives++ 
                    : positives++;

        class_distribution(0) = negatives;
        class_distribution(1) = positives;
    }
    else // More than two classes
    {
        class_distribution.resize(targets_number);

        class_distribution.setZero();

        for(Index i = 0; i < samples_number; i++)
        {
            if (get_sample_use(i) == SampleUse::None) 
                continue;

            for(Index j = 0; j < targets_number; j++)
            {
                if(isnan(data(i,target_variables_indices(j)))) 
                    continue;

                if(data(i,target_variables_indices(j)) > type(0.5)) 
                    class_distribution(j)++;
            }
        }
    }

    return class_distribution;
}


Tensor<Tensor<Index, 1>, 1> DataSet::calculate_Tukey_outliers(const type& cleaning_parameter) const
{
    const Index samples_number = get_used_samples_number();
    const Tensor<Index, 1> samples_indices = get_used_samples_indices();

    const Index raw_variables_number = get_raw_variables_number();
    const Index used_raw_variables_number = get_used_raw_variables_number();
    const Tensor<Index, 1> used_raw_variables_indices = get_used_raw_variables_indices();

    Tensor<Tensor<Index, 1>, 1> return_values(2);

    return_values(0) = Tensor<Index, 1>(samples_number);
    return_values(1) = Tensor<Index, 1>(used_raw_variables_number);

    return_values(0).setZero();
    return_values(1).setZero();

    const Tensor<BoxPlot, 1> box_plots = calculate_raw_variables_box_plots();

    Index variable_index = 0;
    Index used_variable_index = 0;

    #pragma omp parallel for

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).use == VariableUse::None 
        && raw_variables(i).type == RawVariableType::Categorical)
        {
            variable_index += raw_variables(i).get_categories_number();
            continue;
        }
        else if(raw_variables(i).use == VariableUse::None) // Numeric, Binary or DateTime
        {
            variable_index++;
            continue;
        }

        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            variable_index += raw_variables(i).get_categories_number();
            used_variable_index++;
            continue;
        }
        else if(raw_variables(i).type == RawVariableType::Binary 
             || raw_variables(i).type == RawVariableType::DateTime)
        {
            variable_index++;
            used_variable_index++;
            continue;
        }
        else // Numeric
        {
            const type interquartile_range = box_plots(i).third_quartile - box_plots(i).first_quartile;

            if(interquartile_range < numeric_limits<type>::epsilon())
            {
                variable_index++;
                used_variable_index++;
                continue;
            }

            Index raw_variables_outliers = 0;

            for(Index j = 0; j < samples_number; j++)
            {
                const Tensor<type, 1> sample = get_sample_data(samples_indices(Index(j)));

                if(sample(variable_index) < box_plots(i).first_quartile - cleaning_parameter*interquartile_range
                || sample(variable_index) > box_plots(i).third_quartile + cleaning_parameter*interquartile_range)
                {
                    return_values(0)(j) = 1;

                    raw_variables_outliers++;
                }
            }

            return_values(1)(used_variable_index) = raw_variables_outliers;

            variable_index++;
            used_variable_index++;
        }
    }

    return return_values;
}


Tensor<Tensor<Index, 1>, 1> DataSet::replace_Tukey_outliers_with_NaN(const type& cleaning_parameter)
{
    const Index samples_number = get_used_samples_number();
    const Tensor<Index, 1> samples_indices = get_used_samples_indices();

    const Index raw_variables_number = get_raw_variables_number();
    const Index used_raw_variables_number = get_used_raw_variables_number();
    const Tensor<Index, 1> used_raw_variables_indices = get_used_raw_variables_indices();

    Tensor<Tensor<Index, 1>, 1> return_values(2);

    return_values(0) = Tensor<Index, 1>(samples_number);
    return_values(1) = Tensor<Index, 1>(used_raw_variables_number);

    return_values(0).setZero();
    return_values(1).setZero();

    const Tensor<BoxPlot, 1> box_plots = calculate_raw_variables_box_plots();

    Index variable_index = 0;
    Index used_variable_index = 0;

    #pragma omp parallel for

    for(Index i = 0; i < raw_variables_number; i++)
    {
        if(raw_variables(i).use == VariableUse::None 
        && raw_variables(i).type == RawVariableType::Categorical)
        {
            variable_index += raw_variables(i).get_categories_number();
            continue;
        }
        else if(raw_variables(i).use == VariableUse::None) // Numeric, Binary or DateTime
        {
            variable_index++;
            continue;
        }

        if(raw_variables(i).type == RawVariableType::Categorical)
        {
            variable_index += raw_variables(i).get_categories_number();
            used_variable_index++;
            continue;
        }
        else if(raw_variables(i).type == RawVariableType::Binary 
             || raw_variables(i).type == RawVariableType::DateTime)
        {
            variable_index++;
            used_variable_index++;
            continue;
        }
        else // Numeric
        {
            const type interquartile_range = box_plots(i).third_quartile - box_plots(i).first_quartile;

            if(interquartile_range < numeric_limits<type>::epsilon())
            {
                variable_index++;
                used_variable_index++;
                continue;
            }

            Index raw_variables_outliers = 0;

            for(Index j = 0; j < samples_number; j++)
            {
                const Tensor<type, 1> sample = get_sample_data(samples_indices(Index(j)));

                if(sample(variable_index) < (box_plots(i).first_quartile - cleaning_parameter * interquartile_range)
                || sample(variable_index) > (box_plots(i).third_quartile + cleaning_parameter * interquartile_range))
                {
                    return_values(0)(Index(j)) = 1;

                    raw_variables_outliers++;

                    data(samples_indices(Index(j)), variable_index) = numeric_limits<type>::quiet_NaN();
                }
            }

            return_values(1)(used_variable_index) = raw_variables_outliers;

            variable_index++;
            used_variable_index++;
        }
    }

    return return_values;
}


void DataSet::unuse_Tukey_outliers(const type& cleaning_parameter)
{
    const Tensor<Tensor<Index, 1>, 1> outliers_indices = calculate_Tukey_outliers(cleaning_parameter);

    const Tensor<Index, 1> outliers_samples = get_elements_greater_than(outliers_indices, 0);

    set_sample_uses(outliers_samples, DataSet::SampleUse::None);
}


void DataSet::generate_constant_data(const Index& samples_number, const Index& variables_number, const type& value)
{
    set(samples_number, variables_number);

    data.setConstant(value);

    set_default_raw_variables_uses();
}


void DataSet::generate_random_data(const Index& samples_number, const Index& variables_number)
{
    set(samples_number, variables_number);

    set_random(data);
}


void DataSet::generate_sequential_data(const Index& samples_number, const Index& variables_number)
{
    set(samples_number, variables_number);

    for(Index i = 0; i < samples_number; i++)
        for(Index j = 0; j < variables_number; j++)
            data(i, j) = type(j);
}


void DataSet::generate_Rosenbrock_data(const Index& samples_number, const Index& variables_number)
{
    const Index inputs_number = variables_number-1;

    set(samples_number, variables_number);

    set_random(data);
    
#pragma omp parallel for

    for(Index i = 0; i < samples_number; i++)
    {
        type rosenbrock(0);

        for(Index j = 0; j < inputs_number-1; j++)
        {
            const type value = data(i, j);
            const type next_value = data(i, j+1);

            rosenbrock += (type(1) - value)*(type(1) - value) + type(100)*(next_value-value*value)*(next_value-value*value);
        }

        data(i, inputs_number) = rosenbrock;
    }

    set_default_raw_variables_uses();    
}


void DataSet::generate_classification_data(const Index& samples_number, const Index& variables_number, const Index& classes_number)
{
    cout << "Generating Classification Data..." << endl;

    set(samples_number, variables_number + classes_number);

    set_random(data);

    data.setConstant(0.0);

#pragma omp parallel for

    for(Index i = 0; i < samples_number; i++)
    {
        for(Index j = 0; j < variables_number; j++)
            data(i, j) = rand();

        const Index random_class = rand() % classes_number;

        data(i, variables_number + random_class) = 1;
    }

    cout << "Done." << endl;
}


void DataSet::generate_sum_data(const Index& samples_number, const Index& variables_number)
{
    set(samples_number,variables_number);

    set_random(data);

    for(Index i = 0; i < samples_number; i++)
    {
        data(i,variables_number-1) = type(0);

        for(Index j = 0; j < variables_number-1; j++)
            data(i,variables_number-1) += data(i, j);
    }
}


Tensor<Index, 1> DataSet::filter_data(const Tensor<type, 1>& minimums, 
                                      const Tensor<type, 1>& maximums)
{
    const Tensor<Index, 1> used_variables_indices = get_used_variables_indices();

    const Index used_variables_number = used_variables_indices.size();

    const Index samples_number = get_samples_number();

    Tensor<type, 1> filtered_indices(samples_number);
    filtered_indices.setZero();

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Index used_samples_number = used_samples_indices.size();

    Index sample_index = 0;

    for(Index i = 0; i < used_variables_number; i++)
    {
        const Index variable_index = used_variables_indices(i);

        for(Index j = 0; j < used_samples_number; j++)
        {
            sample_index = used_samples_indices(j);

            if(get_sample_use(sample_index) == SampleUse::None)
                continue;

            if(isnan(data(sample_index, variable_index)))
                continue;

            if(abs(data(sample_index, variable_index) - minimums(i)) <= type(NUMERIC_LIMITS_MIN)
            || abs(data(sample_index, variable_index) - maximums(i)) <= type(NUMERIC_LIMITS_MIN))
                continue;

            if(minimums(i) == maximums(i))
            {
                if(data(sample_index, variable_index) != minimums(i))
                {
                    filtered_indices(sample_index) = type(1);
                    set_sample_use(sample_index, SampleUse::None);
                }
            }
            else if(data(sample_index, variable_index) < minimums(i) 
                 || data(sample_index, variable_index) > maximums(i))
            {
                filtered_indices(sample_index) = type(1);
                set_sample_use(sample_index, SampleUse::None);
            }
        }
    }

    const Index filtered_samples_number =
            Index(count_if(filtered_indices.data(),
                           filtered_indices.data()+filtered_indices.size(),
                           [](type value)
                           {
                               return value > type(0.5);
                           }));

    Tensor<Index, 1> filtered_samples_indices(filtered_samples_number);

    Index index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        if(filtered_indices(i) > type(0.5))
        {
            filtered_samples_indices(index) = i;
            index++;
        }
    }

    return filtered_samples_indices;
}


void DataSet::impute_missing_values_unuse()
{
    const Index samples_number = get_samples_number();

    #pragma omp parallel for

    for(Index i = 0; i <samples_number; i++)
        if(has_nan_row(i)) set_sample_use(i, "None");
}


void DataSet::impute_missing_values_mean()
{
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> used_variables_indices = get_used_variables_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    const Tensor<type, 1> means = mean(data, used_samples_indices, used_variables_indices);

    const Index samples_number = used_samples_indices.size();
    const Index variables_number = used_variables_indices.size();
    const Index target_variables_number = target_variables_indices.size();

    Index current_variable;
    Index current_sample;

    #pragma omp parallel for schedule(dynamic)

    for(Index j = 0; j < variables_number - target_variables_number; j++)
    {
        current_variable = input_variables_indices(j);

        for(Index i = 0; i < samples_number; i++)
        {
            current_sample = used_samples_indices(i);

            if(isnan(data(current_sample, current_variable)))
                data(current_sample,current_variable) = means(j);
        }
    }

    #pragma omp parallel for schedule(dynamic)

    for(Index j = 0; j < target_variables_number; j++)
    {
        current_variable = target_variables_indices(j);

        for(Index i = 0; i < samples_number; i++)
        {
            current_sample = used_samples_indices(i);

            if(isnan(data(current_sample, current_variable)))
                set_sample_use(i, "None");
        }
    }
}


void DataSet::impute_missing_values_median()
{
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> used_variables_indices = get_used_variables_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    const Tensor<type, 1> medians = median(data, used_samples_indices, used_variables_indices);

    const Index samples_number = used_samples_indices.size();
    const Index variables_number = used_variables_indices.size();
    const Index target_variables_number = target_variables_indices.size();

    #pragma omp parallel for schedule(dynamic)

    for(Index j = 0; j < variables_number - target_variables_number; j++)
    {
        const Index current_variable = input_variables_indices(j);

        for(Index i = 0; i < samples_number; i++)
        {
            const Index current_sample = used_samples_indices(i);

            if(isnan(data(current_sample, current_variable)))
                data(current_sample,current_variable) = medians(j);
        }
    }

    #pragma omp parallel for schedule(dynamic)

    for(Index j = 0; j < target_variables_number; j++)
    {
        const Index current_variable = target_variables_indices(j);

        for(Index i = 0; i < samples_number; i++)
        {
            const Index current_sample = used_samples_indices(i);

            if(isnan(data(current_sample, current_variable)))
                set_sample_use(i, "None");
        }
    }
}


void DataSet::impute_missing_values_interpolate()
{
    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> used_variables_indices = get_used_variables_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();
    const Tensor<Index, 1> target_variables_indices = get_target_variables_indices();

    const Index samples_number = used_samples_indices.size();
    const Index variables_number = used_variables_indices.size();
    const Index target_variables_number = target_variables_indices.size();

    Index current_variable;
    Index current_sample;

#pragma omp parallel for schedule(dynamic)
    for(Index j = 0; j < variables_number - target_variables_number; j++)
    {
        current_variable = input_variables_indices(j);

        for(Index i = 0; i < samples_number; i++)
        {
            current_sample = used_samples_indices(i);

            if(isnan(data(current_sample, current_variable)))
            {
                type x1 = type(0);
                type x2 = type(0);
                type y1 = type(0);
                type y2 = type(0);
                type x = type(0);
                type y = type(0);

                for(Index k = i - 1; k >= 0; k--)
                {
                    if(!isnan(data(used_samples_indices(k), current_variable)))
                    {
                        x1 = type(used_samples_indices(k));
                        y1 = data(x1, current_variable);
                        break;
                    }
                }

                for(Index k = i + 1; k < samples_number; k++)
                {
                    if(!isnan(data(used_samples_indices(k), current_variable)))
                    {
                        x2 = type(used_samples_indices(k));
                        y2 = data(x2, current_variable);
                        break;
                    }
                }

                if(x2 != x1)
                {
                    x = type(current_sample);
                    y = y1 + (x - x1) * (y2 - y1) / (x2 - x1);
                }
                else
                {
                    y = y1;
                }

                data(current_sample,current_variable) = y;
            }
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for(Index j = 0; j < target_variables_number; j++)
    {
        current_variable = target_variables_indices(j);

        for(Index i = 0; i < samples_number; i++)
        {
            current_sample = used_samples_indices(i);

            if(isnan(data(current_sample, current_variable)))
                set_sample_use(i, "None");
        }
    }
}


void DataSet::scrub_missing_values()
{
    switch(missing_values_method)
    {
    case MissingValuesMethod::Unuse:
        impute_missing_values_unuse();
        break;

    case MissingValuesMethod::Mean:
        impute_missing_values_mean();
        break;

    case MissingValuesMethod::Median:
        impute_missing_values_median();
        break;

    case MissingValuesMethod::Interpolation:
        impute_missing_values_interpolate();
        break;
    }
}


void DataSet::read_csv()
{
    if(data_path.empty())
        throw runtime_error("Data source path is empty.\n");

    ifstream file;
    file.open(data_path.c_str());

    if(!file.is_open())
        throw runtime_error("Error: Cannot open file " + data_path + "\n");

    const string separator_string = get_separator_string();
    string line;

    Tensor<string, 1> tokens;

    Tensor<string, 1> positive_words(4);
    positive_words.setValues({"yes", "positive", "+", "true"});

    Tensor<string, 1> negative_words(4);
    negative_words.setValues({"no", "negative", "-", "false"});

    Index columns_number = 0;

    // Read first line

    while(file.good())
    {
        getline(file, line);
        decode(line);
        trim(line);
        erase(line, '"');

        if(line.empty()) continue;

        check_separators(line);

        tokens = get_tokens(line, separator_string);

        columns_number = tokens.size();

        if(columns_number != 0) break;
    }

    const Index raw_variables_number = has_samples_id
            ? columns_number - 1
            : columns_number;

    raw_variables.resize(raw_variables_number);

    Index samples_number = 0;

    if(has_header)
    {
        if(has_numbers(tokens))
            throw runtime_error("Error: Some header names are numeric: " + line + "\n");

        if(!has_samples_id)
            set_raw_variables_names(tokens);
        else
            for(Index i = 0; i < raw_variables_number; i++)
                raw_variables(i).name = tokens(i+1);
    }
    else
    {
        samples_number++;
        set_default_raw_variables_names();
    }

    // Rest of lines

    while(file.good())
    {
        getline(file, line);
        decode(line);
        trim(line);
        erase(line, '"');
        if(line.empty()) continue;
        check_separators(line);

        tokens = get_tokens(line, separator_string);

        if(tokens.size() != columns_number)
            throw runtime_error("Tokens number is not equal to columns number.");

        //#pragma omp parallel for reduction(+:missing_values_number)

        for(Index i = 0; i < raw_variables_number; i++)
        {
            const RawVariableType type = raw_variables(i).type;

            const string token = has_samples_id ? tokens(i+1) : tokens(i);

            if(token.empty() || token == missing_values_label)
            {
                missing_values_number++;
                continue;
            }
            else if(is_numeric_string(token))
            {
                if(type == RawVariableType::None)
                    raw_variables(i).type = RawVariableType::Numeric;

                if(type == RawVariableType::Categorical)
                    throw runtime_error("Error: Found number in categorical variable: " + raw_variables(i).name);
            }
            else if(is_date_time_string(token))
            {
                if(type == RawVariableType::None)
                    raw_variables(i).type = RawVariableType::DateTime;
            }
            else // is string
            {
                if(type == RawVariableType::None)
                    raw_variables(i).type = RawVariableType::Categorical;

                if(!contains(raw_variables(i).categories, token))
                    push_back_string(raw_variables(i).categories, token);
            }
        }

        samples_number++;
    }

    for(Index i = 0; i < raw_variables_number; i++)
        if(raw_variables(i).type == RawVariableType::Categorical
        && raw_variables(i).get_categories_number() == 2)
            raw_variables(i).type = RawVariableType::Binary;

    samples_uses.resize(samples_number);

    samples_id.resize(samples_number);

    const Index variables_number = get_variables_number();

    data.resize(samples_number, variables_number);
    data.setZero();

    rows_missing_values_number = 0;

    missing_values_number = 0;

    raw_variables_missing_values_number.resize(raw_variables_number);
    raw_variables_missing_values_number.setZero();

    // Fill data

    file.clear();
    file.seekg(0);

    if(has_header)
    {
        while(file.good())
        {
            getline(file, line);
            decode(line);
            trim(line);
            erase(line, '"');
            if(line.empty()) continue;
            break;
        }
    }

    Index sample_index = 0;

    while(file.good())
    {
        getline(file, line);
        decode(line);
        trim(line);
        erase(line, '"');
        if(line.empty()) continue;
        check_separators(line);

        tokens = get_tokens(line, separator_string);

        if(has_missing_values(tokens))
        {
            rows_missing_values_number ++;

            for(Index i = 0; i < tokens.size(); i++)
            {
                if(tokens(i).empty() || tokens(i) == missing_values_label)
                {
                    missing_values_number++;

                    raw_variables_missing_values_number(i)++;
                }
            }
        }

        if(has_samples_id) samples_id(sample_index) = tokens(0);

        #pragma omp parallel for

        for(Index raw_variable_index = 0; raw_variable_index < raw_variables_number; raw_variable_index++)
        {
            const RawVariableType raw_variable_type = raw_variables(raw_variable_index).type;

            const string token = has_samples_id ? tokens(raw_variable_index+1) : tokens(raw_variable_index);

            const Tensor<Index, 1> variable_indices = get_variable_indices(raw_variable_index);

            if(raw_variable_type == RawVariableType::Numeric)
            {
                if(token.empty() || token == missing_values_label)
                    data(sample_index, variable_indices(0)) = NAN;
                else
                    data(sample_index, variable_indices(0)) = stof(token);
            }
            else if(raw_variable_type == RawVariableType::DateTime)
            {              
                data(sample_index, raw_variable_index) = time_t(date_to_timestamp(tokens(raw_variable_index)));
            }
            else if(raw_variable_type == RawVariableType::Categorical)
            {
                const Index categories_number = raw_variables(raw_variable_index).get_categories_number();

                if(token.empty() || token == missing_values_label)
                {
                    for(Index category_index = 0; category_index < categories_number; category_index++)
                        data(sample_index, variable_indices(category_index)) = NAN;
                }
                else
                {
                    const Tensor<string, 1> categories = raw_variables(raw_variable_index).categories;

                    for(Index category_index = 0; category_index < categories_number; category_index++)
                        if(token == categories(category_index))
                            data(sample_index, variable_indices(category_index)) = 1;
                }
            }
            else if(raw_variable_type == RawVariableType::Binary)
            {
                if(contains(positive_words, token) || contains(negative_words, token))
                {
                    data(sample_index, variable_indices(0)) = contains(positive_words, token) ? 1 : 0;
                }
                else
                {
                    const Tensor<string, 1> categories = raw_variables(raw_variable_index).categories;

                    if(token.empty() || token == missing_values_label)
                        data(sample_index, variable_indices(0)) = type(NAN);
                    else if(token == categories(0))
                        data(sample_index, variable_indices(0)) = 1;
                    else if(token == categories(1))
                        data(sample_index, variable_indices(0)) = 0;
                    else
                        throw runtime_error("Unknown token " + token);
                }
            }
        }

        sample_index++;
    }

    file.close();

    unuse_constant_raw_variables();
    set_binary_raw_variables();
    split_samples_random();
}


Tensor<string, 1> DataSet::get_default_raw_variables_names(const Index& raw_variables_number)
{
    Tensor<string, 1> raw_variables_names(raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
        raw_variables_names(i) = "variable_" + to_string(i+1);

    return raw_variables_names;
}


string DataSet::RawVariable::get_type_string() const
{
    switch(type)
    {
    case RawVariableType::None:
        return "None";

    case RawVariableType::Numeric:
        return "Numeric";

    case RawVariableType::Constant:
        return "Constant";

    case RawVariableType::Binary:
        return "Binary";

    case RawVariableType::Categorical:
        return "Categorical";

    case RawVariableType::DateTime:
        return "DateTime";

    default:
        throw runtime_error("Unknown raw variable type");
    }
}


string DataSet::RawVariable::get_scaler_string() const
{
    switch(scaler)
    {
    case Scaler::None:
        return "None";

    case Scaler::MinimumMaximum:
        return "MinimumMaximum";

    case Scaler::MeanStandardDeviation:
        return "MeanStandardDeviation";

    case Scaler::StandardDeviation:
        return "StandardDeviation";
        break;

    case Scaler::Logarithm:
        return "Logarithm";

    default:
        return "";
    }
}


void DataSet::open_file(const string& data_file_name, ifstream& file) const
{
    #ifdef _WIN32

    const regex accent_regex("[\\xC0-\\xFF]");

    if(regex_search(data_file_name, accent_regex))
    {
        wstring_convert<codecvt_utf8<wchar_t>> conv;
        const wstring file_name_wide = conv.from_bytes(data_file_name);
        file.open(file_name_wide, ios::binary);
    }
    else
    {
        file.open(data_file_name.c_str(), ios::binary);
    }
    #else
        file.open(data_file_name.c_str(), ios::binary);
    #endif

    if(!file.is_open())
        throw runtime_error("Error: Cannot open file " + data_file_name + "\n");
}


void DataSet::open_file(const string& file_name, ofstream& file) const
{
    #ifdef _WIN32

    const regex accent_regex("[\\xC0-\\xFF]");

    if(regex_search(file_name, accent_regex))
    {
        wstring_convert<codecvt_utf8<wchar_t>> conv;
        wstring file_name_wide = conv.from_bytes(file_name);
        file.open(file_name_wide, ios::binary);
    }
    else
    {
        file.open(file_name.c_str(), ios::binary);
    }

    #else
        file.open(file_name.c_str(), ios::binary);
    #endif
}


void DataSet::read_data_file_preview(ifstream& file)
{
    if(display) cout << "Reading data file preview..." << endl;

    // @todo Not implemented

    const string separator_string = get_separator_string();

    Index lines_number = has_header ? 4 : 3;

    data_file_preview.resize(lines_number);

    string line;

    Index lines_count = 0;

    while(file.good())
    {
        getline(file, line);

        decode(line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        check_separators(line);

        data_file_preview(lines_count) = get_tokens(line, separator_string);

        lines_count++;

        if(lines_count == lines_number) break;
    }

    file.close();

    // Check empty file

    if(data_file_preview(0).size() == 0)
        throw runtime_error("File " + data_path + " is empty.\n");

    // Resize data file preview to original

    if(data_file_preview.size() > 4)
    {
        lines_number = has_header ? 4 : 3;

        Tensor<Tensor<string, 1>, 1> data_file_preview_copy(data_file_preview);

        data_file_preview.resize(lines_number);

        data_file_preview(0) = data_file_preview_copy(1);
        data_file_preview(1) = data_file_preview_copy(1);
        data_file_preview(2) = data_file_preview_copy(2);
        data_file_preview(lines_number - 2) = data_file_preview_copy(data_file_preview_copy.size()-2);
        data_file_preview(lines_number - 1) = data_file_preview_copy(data_file_preview_copy.size()-1);
    }
}


void DataSet::check_separators(const string& line) const
{
    if(line.find(',') == string::npos
    && line.find(';') == string::npos
    && line.find(' ') == string::npos
    && line.find('\t') == string::npos) return;

    const string separator_string = get_separator_string();

    if(line.find(separator_string) == string::npos)
        throw runtime_error("Error: Separarot '" + separator_string + "' not found in line " + line + ".\n");

    if(separator == Separator::Space)
    {
        if(line.find(',') != string::npos)
            throw runtime_error("Error: Found comma (',') in data file " + data_path + ", but separator is space (' ').");

        if(line.find(';') != string::npos)
            throw runtime_error("Error: Found semicolon (';') in data file " + data_path + ", but separator is space (' ').");
    }
    else if(separator == Separator::Tab)
    {
        if(line.find(',') != string::npos)
            throw runtime_error("Error: Found comma (',') in data file " + data_path + ", but separator is tab ('   ').");

        if(line.find(';') != string::npos)
            throw runtime_error("Error: Found semicolon (';') in data file " + data_path + ", but separator is tab ('   ').");
    }
    else if(separator == Separator::Comma)
    {
        if(line.find(";") != string::npos)
            throw runtime_error("Error: Found semicolon (';') in data file " + data_path + ", but separator is comma (',').");
    }
    else if(separator == Separator::Semicolon)
    {
        if(line.find(",") != string::npos)
            throw runtime_error("Error: Found comma (',') in data file " + data_path + ", but separator is semicolon (';').");
    }
}


void DataSet::check_special_characters(const string & line) const
{
    if(line.find_first_of("|@#~€¬^*") != string::npos)
        throw runtime_error("Error: found special characters in line: " + line + ". Please, review the file.");

    //#ifdef __unix__
    //    if(line.find("\r") != string::npos)
    //    {
    //        const string message =
    //                "Error: mixed break line characters in line: " + line + ". Please, review the file.";
    //        throw runtime_error(message);
    //    }
    //#endif
}


bool DataSet::has_binary_raw_variables() const
{
    const Index variables_number = raw_variables.size();

    for(Index i = 0; i < variables_number; i++)
        if(raw_variables(i).type == RawVariableType::Binary) 
            return true;

    return false;
}


bool DataSet::has_categorical_raw_variables() const
{
    const Index variables_number = raw_variables.size();

    for(Index i = 0; i < variables_number; i++)
        if(raw_variables(i).type == RawVariableType::Categorical) 
            return true;

    return false;
}


bool DataSet::has_binary_or_categorical_raw_variables() const
{
    return has_binary_raw_variables() || has_categorical_raw_variables();
}


bool DataSet::has_selection() const
{
    return get_selection_samples_number() != 0;
}


bool DataSet::has_missing_values(const Tensor<string, 1>& row) const
{
    for(Index i = 0; i < row.size(); i++)
        if(row(i).empty() || row(i) == missing_values_label)
            return true;

    return false;
}


Tensor<Index, 1> DataSet::count_raw_variables_with_nan() const
{
    const Index raw_variables_number = get_raw_variables_number();
    const Index rows_number = get_samples_number();

    Tensor<Index, 1> raw_variables_with_nan(raw_variables_number);
    raw_variables_with_nan.setZero();

    for(Index raw_variable_index = 0; raw_variable_index < raw_variables_number; raw_variable_index++)
    {
        const Index current_variable_index = get_variable_indices(raw_variable_index)(0);

        for(Index row_index = 0; row_index < rows_number; row_index++)
            if(isnan(data(row_index, current_variable_index)))
                raw_variables_with_nan(raw_variable_index)++;
    }

    return raw_variables_with_nan;
}


Index DataSet::count_rows_with_nan() const
{
    Index rows_with_nan = 0;

    const Index rows_number = data.dimension(0);
    const Index raw_variables_number = data.dimension(1);

    bool has_nan = true;

    for(Index row_index = 0; row_index < rows_number; row_index++)
    {
        has_nan = false;

        for(Index raw_variable_index = 0; raw_variable_index < raw_variables_number; raw_variable_index++)
        {
            if(isnan(data(row_index, raw_variable_index)))
            {
                has_nan = true;
                break;
            }
        }

        if(has_nan) rows_with_nan++;
    }

    return rows_with_nan;
}


Index DataSet::count_nan() const
{
    return count_NAN(data);
}


void DataSet::set_missing_values_number(const Index& new_missing_values_number)
{
    missing_values_number = new_missing_values_number;
}


void DataSet::set_missing_values_number()
{
    missing_values_number = count_nan();
}


void DataSet::set_raw_variables_missing_values_number(const Tensor<Index, 1>& new_raw_variables_missing_values_number)
{
    raw_variables_missing_values_number = new_raw_variables_missing_values_number;
}


void DataSet::set_raw_variables_missing_values_number()
{
    raw_variables_missing_values_number = count_raw_variables_with_nan();
}


void DataSet::set_samples_missing_values_number(const Index& new_rows_missing_values_number)
{
    rows_missing_values_number = new_rows_missing_values_number;
}


void DataSet::set_samples_missing_values_number()
{
    rows_missing_values_number = count_rows_with_nan();
}


void DataSet::fix_repeated_names()
{
    // Fix raw_variables names

    const Index raw_variables_number = raw_variables.size();

    map<string, Index> raw_variables_count_map;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        auto result = raw_variables_count_map.insert(pair<string, Index>(raw_variables(i).name, 1));

        if(!result.second) result.first->second++;
    }

    for(const auto & element : raw_variables_count_map)
    {
        if(element.second > 1)
        {
            const string repeated_name = element.first;
            Index repeated_index = 1;

            for(Index i = 0; i < raw_variables_number; i++)
            {
                if(raw_variables(i).name == repeated_name)
                {
                    raw_variables(i).name = raw_variables(i).name + "_" + to_string(repeated_index);
                    repeated_index++;
                }
            }
        }
    }

    // Fix variables names

    if(has_categorical_raw_variables() || has_binary_raw_variables())
    {
        Tensor<string, 1> variables_names = get_variables_names();

        const Index variables_number = variables_names.size();

        map<string, Index> variables_count_map;

        for(Index i = 0; i < variables_number; i++)
        {
            auto result = variables_count_map.insert(pair<string, Index>(variables_names(i), 1));

            if(!result.second) result.first->second++;
        }

        for(const auto & element : variables_count_map)
        {
            if(element.second > 1)
            {
                const string repeated_name = element.first;

                for(Index i = 0; i < variables_number; i++)
                {
                    if(variables_names(i) == repeated_name)
                    {
                        const Index raw_variable_index = get_raw_variable_index(i);

                        if(raw_variables(raw_variable_index).type != RawVariableType::Categorical) continue;

                        variables_names(i) = variables_names(i) + "_" + raw_variables(raw_variable_index).name;
                    }
                }
            }
        }

        set_variables_names(variables_names);
    }
}


Tensor<Index, 2> DataSet::split_samples(const Tensor<Index, 1>& samples_indices, const Index& new_batch_size) const
{
    const Index samples_number = samples_indices.dimension(0);

    Index batches_number;
    Index batch_size = new_batch_size;

    if(samples_number < batch_size)
    {
        batches_number = 1;
        batch_size = samples_number;
    }
    else
    {
        batches_number = samples_number / batch_size;
    }

    Tensor<Index, 2> batches(batches_number, batch_size);

    Index count = 0;

    for(Index i = 0; i < batches_number;i++)
    {
        for(Index j = 0; j < batch_size;++j)
        {
            batches(i, j) = samples_indices(count);

            count++;
        }
    }

    return batches;
}


void DataSet::shuffle()
{
    random_device rng;
    mt19937 urng(rng());

    const Index data_rows = data.dimension(0);
    const Index data_raw_variables = data.dimension(1);

    Tensor<Index, 1> indices(data_rows);

    for(Index i = 0; i < data_rows; i++)
        indices(i) = i;

    std::shuffle(&indices(0), &indices(data_rows-1), urng);

    Tensor<type, 2> new_data(data_rows, data_raw_variables);
    Tensor<string, 1> new_rows_labels(data_rows);

    Index index = 0;

    for(Index i = 0; i < data_rows; i++)
    {
        index = indices(i);

        new_rows_labels(i) = samples_id(index);

        for(Index j = 0; j < data_raw_variables; j++)
            new_data(i, j) = data(index,j);
    }

    data = new_data;
    samples_id = new_rows_labels;
}


bool DataSet::get_has_rows_labels() const
{
    return has_samples_id;
}


void DataSet::decode(string& input_string) const
{
    switch(codification)
    {
    case DataSet::Codification::SHIFT_JIS:
    {
        input_string = sj2utf8(input_string);
    }

    default:
    {
        // do nothing;
    }
    }
}


Tensor<type, 2> DataSet::read_input_csv(const string& input_data_file_name,
                                        const string& separator_string,
                                        const string& missing_values_label,
                                        const bool& has_raw_variables_name,
                                        const bool& has_samples_id) const
{
    const Index raw_variables_number = get_raw_variables_number();

    ifstream file;

    open_file(input_data_file_name, file);

    // Count samples number

    Index input_samples_count = 0;

    string line;
    Index line_number = 0;

    Index tokens_count;

    Index input_raw_variables_number = get_input_raw_variables_number();

    if(model_type == ModelType::AutoAssociation)
        input_raw_variables_number = get_raw_variables_number() - get_target_raw_variables_number() - get_unused_raw_variables_number()/2;

    while(file.good())
    {
        line_number++;

        getline(file, line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        tokens_count = count_tokens(line, separator_string);

        if(tokens_count != input_raw_variables_number)
            throw runtime_error("Line " + to_string(line_number) + ": Size of tokens(" + to_string(tokens_count) + ") "
                                "is not equal to number of raw_variables(" + to_string(input_raw_variables_number) + ").\n");

        input_samples_count++;
    }

    file.close();

    const Index input_variables_number = get_input_variables_number();

    if(has_raw_variables_name) input_samples_count--;

    Tensor<type, 2> input_data(input_samples_count, input_variables_number);
    input_data.setZero();

    open_file(input_data_file_name, file);

    //skip_header(file);

    // Read rest of the lines

    Tensor<string, 1> tokens;

    line_number = 0;
    Index variable_index = 0;
    Index token_index = 0;
    bool is_ID = has_samples_id;

    const bool is_float = is_same<type, float>::value;
    bool has_missing_values = false;

    while(file.good())
    {
        getline(file, line);

        trim(line);

        erase(line, '"');

        if(line.empty()) continue;

        tokens = get_tokens(line, separator_string);

        variable_index = 0;
        token_index = 0;
        is_ID = has_samples_id;

        for(Index i = 0; i < raw_variables_number; i++)
        {
            if(is_ID)
            {
                is_ID = false;
                continue;
            }

            if(raw_variables(i).use == VariableUse::None)
            {
                token_index++;
                continue;
            }
            else if(raw_variables(i).use != VariableUse::Input)
            {
                continue;
            }

            if(raw_variables(i).type == RawVariableType::Numeric)
            {
                if(tokens(token_index) == missing_values_label || tokens(token_index).empty())
                {
                    has_missing_values = true;
                    input_data(line_number, variable_index) = type(NAN);
                }
                else if(is_float)
                {
                    input_data(line_number, variable_index) = type(strtof(tokens(token_index).data(), nullptr));
                }
                else
                {
                    input_data(line_number, variable_index) = type(stof(tokens(token_index)));
                }

                variable_index++;
            }
            else if(raw_variables(i).type == RawVariableType::Binary)
            {
                if(tokens(token_index) == missing_values_label)
                {
                    has_missing_values = true;
                    input_data(line_number, variable_index) = type(NAN);
                }
                else if(raw_variables(i).categories.size() > 0 && tokens(token_index) == raw_variables(i).categories(0))
                {
                    input_data(line_number, variable_index) = type(1);
                }
                else if(tokens(token_index) == raw_variables(i).name)
                {
                    input_data(line_number, variable_index) = type(1);
                }

                variable_index++;
            }
            else if(raw_variables(i).type == RawVariableType::Categorical)
            {
                for(Index k = 0; k < raw_variables(i).get_categories_number(); k++)
                {
                    if(tokens(token_index) == missing_values_label)
                    {
                        has_missing_values = true;
                        input_data(line_number, variable_index) = type(NAN);
                    }
                    else if(tokens(token_index) == raw_variables(i).categories(k))
                    {
                        input_data(line_number, variable_index) = type(1);
                    }

                    variable_index++;
                }
            }
            else if(raw_variables(i).type == RawVariableType::DateTime)
            {
                if(tokens(token_index) == missing_values_label || tokens(token_index).empty())
                {
                    has_missing_values = true;
                    input_data(line_number, variable_index) = type(NAN);
                }
                else
                {
                    input_data(line_number, variable_index) = type(date_to_timestamp(tokens(token_index), gmt));
                }

                variable_index++;
            }
            else if(raw_variables(i).type == RawVariableType::Constant)
            {
                if(tokens(token_index) == missing_values_label || tokens(token_index).empty())
                {
                    has_missing_values = true;
                    input_data(line_number, variable_index) = type(NAN);
                }
                else if(is_float)
                {
                    input_data(line_number, variable_index) = type(strtof(tokens(token_index).data(), nullptr));
                }
                else
                {
                    input_data(line_number, variable_index) = type(stof(tokens(token_index)));
                }

                variable_index++;
            }

            token_index++;
        }

        line_number++;
    }

    file.close();

    if(!has_missing_values)
    {
        return input_data;
    }
    else
    {
        // Scrub missing values

        const MissingValuesMethod missing_values_method = get_missing_values_method();

        if(missing_values_method == MissingValuesMethod::Unuse || missing_values_method == MissingValuesMethod::Mean)
        {
            const Tensor<type, 1> means = mean(input_data);

            const Index samples_number = input_data.dimension(0);
            const Index variables_number = input_data.dimension(1);

            #pragma omp parallel for schedule(dynamic)

            for(Index j = 0; j < variables_number; j++)
                for(Index i = 0; i < samples_number; i++)
                    if(isnan(input_data(i, j)))
                        input_data(i, j) = means(j);
        }
        else
        {
            const Tensor<type, 1> medians = median(input_data);

            const Index samples_number = input_data.dimension(0);
            const Index variables_number = input_data.dimension(1);

            #pragma omp parallel for schedule(dynamic)

            for(Index j = 0; j < variables_number; j++)
                for(Index i = 0; i < samples_number; i++)
                    if(isnan(input_data(i, j)))
                        input_data(i, j) = medians(j);
        }

        return input_data;
    }
}


bool DataSet::get_augmentation() const
{
    return augmentation;
}


// Virtual functions

// Image Models
void DataSet::fill_image_data(const int& width, const int& height, const int& channels, const Tensor<type, 2>& data) {}

// Languaje Models
void DataSet::read_txt_language_model(){}

// AutoAssociation Models
void DataSet::transform_associative_dataset(){}
void DataSet::save_auto_associative_data_binary(const string&) const {};

} // namespace opennn


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
