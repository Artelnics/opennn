//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "data_set.h"
#include "statistics.h"
#include "correlations.h"
#include "tensors.h"
#include "strings_utilities.h"

namespace opennn
{

DataSet::DataSet(const Index& new_samples_number, 
                 const dimensions& new_input_dimensions, 
                 const dimensions& new_target_dimensions)
{
    set(new_samples_number, new_input_dimensions, new_target_dimensions);
}


DataSet::DataSet(const filesystem::path& data_path,
                 const string& separator,
                 const bool& has_header,
                 const bool& has_sample_ids,
                 const Codification& data_codification)
{
    set(data_path, separator, has_header, has_sample_ids, data_codification);
}


const bool& DataSet::get_display() const
{
    return display;
}


DataSet::RawVariable::RawVariable(const string& new_name,
                                  const VariableUse& new_raw_variable_use,
                                  const RawVariableType& new_type,
                                  const Scaler& new_scaler,
                                  const vector<string>& new_categories)
{
    name = new_name;
    use = new_raw_variable_use;
    type = new_type;
    scaler = new_scaler;
    categories = new_categories;
}


void DataSet::RawVariable::set(const string& new_name,
                               const VariableUse& new_raw_variable_use,
                               const RawVariableType& new_type,
                               const Scaler& new_scaler,
                               const vector<string>& new_categories)
{
    name = new_name;
    use = new_raw_variable_use;
    type = new_type;
    scaler = new_scaler;
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
    else if (new_scaler == "ImageMinMax")
        set_scaler(Scaler::ImageMinMax);
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


void DataSet::RawVariable::set_categories(const vector<string>& new_categories)
{
    categories.resize(new_categories.size());

    categories = new_categories;
}


void DataSet::RawVariable::from_XML(const XMLDocument& document)
{
    name = read_xml_string(document.FirstChildElement(), "Name");
    set_scaler(read_xml_string(document.FirstChildElement(), "Scaler"));
    set_use(read_xml_string(document.FirstChildElement(), "Use"));
    set_type(read_xml_string(document.FirstChildElement(), "Type"));

    if (type == RawVariableType::Categorical) 
    {
        const string categories_text = read_xml_string(document.FirstChildElement(), "Categories");
        categories = get_tokens(categories_text, ";");
    }
}


void DataSet::RawVariable::to_XML(XMLPrinter& printer) const
{
    add_xml_element(printer,"Name", name);
    add_xml_element(printer,"Scaler", get_scaler_string());
    add_xml_element(printer,"Use", get_use_string());
    add_xml_element(printer,"Type", get_type_string());

    if(type == RawVariableType::Categorical || type == RawVariableType::Binary)
    {
        if(categories.size() == 0) 
            return;

        add_xml_element(printer,"Categories", string_tensor_to_string(categories));
    }
}


void DataSet::RawVariable::print() const
{
    cout << "Raw variable" << endl
         << "Name: " << name << endl
         << "Use: " << get_use_string() << endl
         << "Type: " << get_type_string() << endl
         << "Scaler: " << get_scaler_string() << endl;

    if (categories.size() != 0)
    {
        cout << "Categories: " << endl;
        print_vector(categories);
    }
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
    return sample_uses[index] != SampleUse::None;
}


Tensor<Index, 1> DataSet::get_sample_use_numbers() const
{
    Tensor<Index, 1> count(4);
    count.setZero();

    const Index samples_number = get_samples_number();

    #pragma omp parallel for

    for(Index i = 0; i < samples_number; i++)
        switch (sample_uses[i])
        {
        case SampleUse::Training: count[0]++; break;
        case SampleUse::Selection: count[1]++; break;
        case SampleUse::Testing: count[2]++; break;
        default: count[3]++; break;
        }

    return count;
}


Tensor<type, 1> DataSet::get_sample_use_percentages() const
{
    const Index samples_number = get_samples_number();

    return (get_sample_use_numbers().cast<type>()) * (100 / type(samples_number));
}


string DataSet::get_sample_string(const Index& sample_index, const string& separator) const
{
    const Tensor<type, 1> sample = data.chip(sample_index, 0);

    string sample_string;

    const Index raw_variables_number = get_raw_variables_number();

    Index variable_index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        const RawVariable& raw_variable = raw_variables[i];

        switch(raw_variable.type)
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
                : raw_variable.categories[Index(data(sample_index, variable_index))];

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
                const Index categories_number = raw_variable.get_categories_number();

                for(Index j = 0; j < categories_number; j++)
                {
                    if(abs(data(sample_index, variable_index+j) - type(1)) < NUMERIC_LIMITS_MIN)
                    {
                        sample_string += raw_variable.categories[j];
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


vector<Index> DataSet::get_sample_indices(const SampleUse& sample_use) const
{
    const Index samples_number = get_samples_number();

    const Index count = get_samples_number(sample_use);

    vector<Index> indices(count);

    Index index = 0;

    for (Index i = 0; i < samples_number; i++)
        if (sample_uses[i] == sample_use)
            indices[index++] = i;

    return indices;
}


vector<Index> DataSet::get_used_sample_indices() const
{
    const Index samples_number = get_samples_number();

    const Index used_samples_number = samples_number - get_samples_number(SampleUse::None);

    vector<Index> used_indices(used_samples_number);

    Index index = 0;

    for(Index i = 0; i < samples_number; i++)
        if(sample_uses[i] != SampleUse::None)
            used_indices[index++] = i;

    return used_indices;
}


DataSet::SampleUse DataSet::get_sample_use(const Index& index) const
{
    return sample_uses[index];
}


const vector<DataSet::SampleUse>& DataSet::get_sample_uses() const
{
    return sample_uses;
}


Tensor<Index, 1> DataSet::get_sample_uses_vector() const
{
    const Index samples_number = get_samples_number();

    Tensor<Index, 1> samples_uses_tensor(samples_number);

    #pragma omp parallel for

    for(Index i = 0; i < samples_number; i++)
        samples_uses_tensor(i) = Index(sample_uses[i]);

    return samples_uses_tensor;
}


vector<vector<Index>> DataSet::get_batches(const vector<Index>& sample_indices,
                                           const Index& batch_samples_number,
                                           const bool& shuffle,
                                           const Index& new_buffer_size) const
{
    if(!shuffle) return split_samples(sample_indices, batch_samples_number);

    random_device rng;
    mt19937 urng(rng());

    const Index samples_number = sample_indices.size();

    Index buffer_size = new_buffer_size;
    Index batches_number;

    const Index batch_size = min(batch_samples_number, samples_number);

    if(buffer_size > samples_number)
        buffer_size = samples_number;

    if(samples_number < batch_size)
    {
        batches_number = 1;
        buffer_size = batch_size;
    }
    else
    {
        batches_number = samples_number / batch_size;
    }

    vector<vector<Index>> batches(batches_number);

    vector<Index> samples_copy(sample_indices);

    // Shuffle

    std::shuffle(samples_copy.data(), samples_copy.data() + samples_copy.size(), urng);

    #pragma omp parallel for

    for(Index i = 0; i < batches_number; i++)
    {
        batches[i].resize(batch_size);

        const Index offset = i * batches_number;

        for(Index j = 0; j < batch_size; j++)
            batches[i][j] = samples_copy[offset + j];
    }

    return batches;
}


Index DataSet::get_samples_number(const SampleUse& sample_use) const
{
    return count_if(sample_uses.begin(), sample_uses.end(),
                    [&sample_use](const SampleUse& new_sample_use) { return new_sample_use == sample_use; });
}


Index DataSet::get_used_samples_number() const
{
    const Index samples_number = get_samples_number();
    const Index unused_samples_number = get_samples_number(SampleUse::None);

    return samples_number - unused_samples_number;
}


void DataSet::set(const SampleUse& sample_use)
{
    fill(sample_uses.begin(), sample_uses.end(), sample_use);
}


void DataSet::set_sample_use(const Index& index, const SampleUse& new_use)
{
    const Index samples_number = get_samples_number();

    if(index >= samples_number)
        throw runtime_error("Index must be less than samples number.\n");

    sample_uses[index] = new_use;
}


void DataSet::set_sample_use(const Index& index, const string& new_use)
{
    if(new_use == "Training")
        sample_uses[index] = SampleUse::Training;
    else if(new_use == "Selection")
        sample_uses[index] = SampleUse::Selection;
    else if(new_use == "Testing")
        sample_uses[index] = SampleUse::Testing;
    else if(new_use == "None")
        sample_uses[index] = SampleUse::None;
    else
        throw runtime_error("Unknown sample use: " + new_use + "\n");
}


void DataSet::set_sample_uses(const vector<SampleUse>& new_uses)
{
    const Index samples_number = get_samples_number();

    for(Index i = 0; i < samples_number; i++)
        sample_uses[i] = new_uses[i];
}


void DataSet::set_sample_uses(const vector<string>& new_uses)
{
    const Index samples_number = get_samples_number();

    for(Index i = 0; i < samples_number; i++)
        if(new_uses[i] == "Training" || new_uses[i] == "0")
            sample_uses[i] = SampleUse::Training;
        else if(new_uses[i] == "Selection" || new_uses[i] == "1")
            sample_uses[i] = SampleUse::Selection;
        else if(new_uses[i] == "Testing" || new_uses[i] == "2")
            sample_uses[i] = SampleUse::Testing;
        else if(new_uses[i] == "None" || new_uses[i] == "3")
            sample_uses[i] = SampleUse::None;
        else
            throw runtime_error("Unknown sample use: " + new_uses[i] + ".\n");
}


void DataSet::set_sample_uses(const vector<Index>& indices, const SampleUse& sample_use)
{
    for(size_t i = 0; i < indices.size(); i++)
        set_sample_use(indices[i], sample_use);
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

    const Index selection_samples_number = Index((selection_samples_ratio * used_samples_number)/total_ratio);
    const Index testing_samples_number = Index((testing_samples_ratio * used_samples_number)/ total_ratio);

    const Index training_samples_number = used_samples_number - selection_samples_number - testing_samples_number;

    const Index sum_samples_number = training_samples_number + selection_samples_number + testing_samples_number;

    if(sum_samples_number != used_samples_number)
        throw runtime_error("Sum of numbers of training, selection and testing samples is not equal to number of used samples.\n");

    const Index samples_number = get_samples_number();
    
    vector<Index> indices(samples_number);
    iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.data(), indices.data() + indices.size(), urng);

    auto assign_sample_use = [this, &indices](SampleUse use, Index count, Index& i) 
    {
        Index assigned_count = 0;

        while (assigned_count < count) 
        {
            const Index index = indices[i++];

            if (sample_uses[index] != SampleUse::None) 
            {
                sample_uses[index] = use;
                assigned_count++;
            }
        }
    };

    Index index = 0;

    assign_sample_use(SampleUse::Training, training_samples_number, index);
    assign_sample_use(SampleUse::Selection, selection_samples_number, index);
    assign_sample_use(SampleUse::Testing, testing_samples_number, index);
}


void DataSet::split_samples_sequential(const type& training_samples_ratio,
                                       const type& selection_samples_ratio,
                                       const type& testing_samples_ratio)
{
    const Index used_samples_number = get_used_samples_number();

    if(used_samples_number == 0) return;

    const type total_ratio = training_samples_ratio + selection_samples_ratio + testing_samples_ratio;

    const Index selection_samples_number = Index(selection_samples_ratio* type(used_samples_number)/ type(total_ratio));
    const Index testing_samples_number = Index(testing_samples_ratio* type(used_samples_number)/ type(total_ratio));
    const Index training_samples_number = used_samples_number - selection_samples_number - testing_samples_number;

    const Index sum_samples_number = training_samples_number + selection_samples_number + testing_samples_number;

    if(sum_samples_number != used_samples_number)
        throw runtime_error("Sum of numbers of training, selection and testing samples is not equal to number of used samples.\n");

    auto set_sample_uses = [this](SampleUse use, Index count, Index& i) 
    {
        Index current_count = 0;

        while (current_count < count) 
        {
            if (sample_uses[i] != SampleUse::None) 
            {
                sample_uses[i] = use;
                current_count++;
            }

            i++;
        }
    };

    Index index = 0;

    set_sample_uses(SampleUse::Training, training_samples_number, index);
    set_sample_uses(SampleUse::Selection, selection_samples_number, index);
    set_sample_uses(SampleUse::Testing, testing_samples_number, index);
}


void DataSet::set_raw_variables(const vector<RawVariable>& new_raw_variables)
{
    raw_variables = new_raw_variables;
}


void DataSet::set_default_raw_variables_uses()
{
    const Index raw_variables_number = raw_variables.size();

    bool target = false;

    if(raw_variables_number == 0)   
        return;
    
    if(raw_variables_number == 1)
    {
        raw_variables[0].set_use(VariableUse::None);
    }
    else
    {
        set(VariableUse::Input);

        for(Index i = raw_variables.size()-1; i >= 0; i--)
        {
            if(raw_variables[i].type == RawVariableType::Constant 
            || raw_variables[i].type == RawVariableType::DateTime)
            {
                raw_variables[i].set_use(VariableUse::None);
                continue;
            }

            if(!target)
            {
                raw_variables[i].set_use(VariableUse::Target);

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
        raw_variables[i].name = "variable_" + to_string(1+i);
}


vector<string> DataSet::get_variable_names() const
{    
    const Index raw_variables_number = get_raw_variables_number();

    const Index variables_number = get_variables_number();

    vector<string> variable_names(variables_number);

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
        if(raw_variables[i].type == RawVariableType::Categorical)
            for(size_t j = 0; j < raw_variables[i].categories.size(); j++)
                variable_names[index++] = raw_variables[i].categories[j];
        else
            variable_names[index++] = raw_variables[i].name;

    return variable_names;
}


vector<string> DataSet::get_variable_names(const VariableUse& variable_use) const
{
    const Index variables_number = get_variables_number(VariableUse::Input);

    vector<string> variable_names(variables_number);

    const Index raw_variables_number = get_raw_variables_number();

    Index index = 0;

    for (Index i = 0; i < raw_variables_number; i++)
    {
        if (raw_variables[i].use != variable_use)
            continue;

        if (raw_variables[i].type == RawVariableType::Categorical)
            for (Index j = 0; j < raw_variables[i].get_categories_number(); j++)
                variable_names[index++] = raw_variables[i].categories[j];
        else
            variable_names[index++] = raw_variables[i].name;
    }

    return variable_names;
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

    const Index unused_variables_number = get_variables_number(VariableUse::None);

    return variables_number - unused_variables_number;
}


vector<Index> DataSet::get_variable_indices(const VariableUse& variable_use) const
{
    const Index this_variables_number = get_variables_number(variable_use);
    vector<Index> this_variable_indices(this_variables_number);

    const Index raw_variables_number = get_raw_variables_number();

    Index variable_index = 0;
    Index this_variable_index = 0;

    for (Index i = 0; i < raw_variables_number; i++)
    {
        if (raw_variables[i].use != variable_use)
        {
            if (raw_variables[i].type == RawVariableType::Categorical)
                variable_index += raw_variables[i].get_categories_number();
            else
                variable_index++;

            continue;
        }

        if (raw_variables[i].type == RawVariableType::Categorical)
        {
            const Index categories_number = raw_variables[i].get_categories_number();

            for (Index j = 0; j < categories_number; j++)
                this_variable_indices[this_variable_index++] = variable_index++;
        }
        else
        {
            this_variable_indices[this_variable_index++] = variable_index++;
        }
    }

    return this_variable_indices;
}


vector<Index> DataSet::get_raw_variable_indices(const VariableUse& variable_use) const
{
    const Index count = get_raw_variables_number(variable_use);

    vector<Index> indices(count);

    const Index raw_variables_number = get_raw_variables_number();

    Index index = 0;

    for (Index i = 0; i < raw_variables_number; i++)
        if (raw_variables[i].use == variable_use)
            indices[index++] = i;

    return indices;
}


vector<Index> DataSet::get_used_raw_variables_indices() const
{
    const Index raw_variables_number = get_raw_variables_number();

    const Index used_raw_variables_number = get_used_raw_variables_number();

    vector<Index> used_indices(used_raw_variables_number);

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
        if(raw_variables[i].use  == VariableUse::Input
        || raw_variables[i].use  == VariableUse::Target
        || raw_variables[i].use  == VariableUse::Time)
            used_indices[index++] = i;

    return used_indices;
}


vector<Scaler> DataSet::get_variable_scalers(const VariableUse& variable_use) const
{
    const Index input_raw_variables_number = get_raw_variables_number(variable_use);
    const Index input_variables_number = get_variables_number(variable_use);

    const vector<RawVariable> input_raw_variables = get_raw_variables(variable_use);

    vector<Scaler> input_variable_scalers(input_variables_number);

    Index index = 0;

    for(Index i = 0; i < input_raw_variables_number; i++)
        if(input_raw_variables[i].type == RawVariableType::Categorical)
            for(Index j = 0; j < input_raw_variables[i].get_categories_number(); j++)
                input_variable_scalers[index++] = input_raw_variables[i].scaler;
        else
            input_variable_scalers[index++] = input_raw_variables[i].scaler;

    return input_variable_scalers;
}


vector<string> DataSet::get_raw_variable_names() const
{
    const Index raw_variables_number = get_raw_variables_number();

    vector<string> raw_variable_names(raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
        raw_variable_names[i] = raw_variables[i].name;

    return raw_variable_names;
}


vector<string> DataSet::get_raw_variable_names(const VariableUse& variable_use) const
{
    const Index raw_variables_number = get_raw_variables_number();

    const Index count = get_raw_variables_number(variable_use);

    vector<string> names(count);

    Index index = 0;

    for (Index i = 0; i < raw_variables_number; i++)
    {
        if (raw_variables[i].use != variable_use)
            continue;

        names[index++] = raw_variables[i].name;
    }

    return names;
}


Index DataSet::get_raw_variables_number(const VariableUse& variable_use) const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index count = 0;

    for (Index i = 0; i < raw_variables_number; i++)
        if (raw_variables[i].use == variable_use)
            count++;

    return count;
}


Index DataSet::get_used_raw_variables_number() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index used_raw_variables_number = 0;

    for(Index i = 0; i < raw_variables_number; i++)
        if(raw_variables[i].use != VariableUse::None)
            used_raw_variables_number++;

    return used_raw_variables_number;
}


Index DataSet::get_input_and_unused_variables_number() const
{
    Index raw_variables_number = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        const RawVariable& raw_variable = raw_variables[i];

        if(raw_variable.use != VariableUse::Input && raw_variable.use != VariableUse::None)
            continue;

        if(raw_variable.type == RawVariableType::Categorical)
                raw_variables_number += raw_variable.categories.size();
        else
                raw_variables_number++;
    }

    return raw_variables_number;
}


const vector<DataSet::RawVariable>& DataSet::get_raw_variables() const
{
    return raw_variables;
}


vector<DataSet::RawVariable> DataSet::get_raw_variables(const VariableUse& variable_use) const
{
    const Index raw_variables_number = get_raw_variables_number();

    const Index count = get_raw_variables_number(variable_use);

    vector<RawVariable> this_raw_variables(count);
    Index index = 0;

    for (Index i = 0; i < raw_variables_number; i++)
        if (raw_variables[i].use == variable_use)
            this_raw_variables[index++] = raw_variables[i];

    return this_raw_variables;
}


Index DataSet::get_variables_number() const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index count = 0;

    for (Index i = 0; i < raw_variables_number; i++)
        count += raw_variables[i].type == RawVariableType::Categorical
                     ? raw_variables[i].get_categories_number()
                     : 1;

    return count;
}


Index DataSet::get_variables_number(const VariableUse& variable_use) const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index count = 0;

    for (Index i = 0; i < raw_variables_number; i++)
    {
        if (raw_variables[i].use != variable_use)
            continue;

        count += (raw_variables[i].type == RawVariableType::Categorical)
            ? raw_variables[i].get_categories_number()
            : 1;
    }

    return count;
}


vector<Index> DataSet::get_used_variable_indices() const
{
    const Index used_variables_number = get_used_variables_number();
    vector<Index> used_variable_indices(used_variables_number);

    const Index raw_variables_number = get_raw_variables_number();

    Index variable_index = 0;
    Index used_variable_index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        const Index categories_number = raw_variables[i].get_categories_number();

        if(raw_variables[i].use == VariableUse::None)
        {
            variable_index += categories_number;
            continue;
        }

        for(Index j = 0; j < categories_number; j++)
            used_variable_indices[used_variable_index++] = variable_index++;
    }

    return used_variable_indices;
}


void DataSet::set_raw_variable_uses(const vector<string>& new_raw_variables_uses)
{
    const Index new_raw_variables_uses_size = new_raw_variables_uses.size();

    if(new_raw_variables_uses_size != raw_variables.size())
        throw runtime_error("Size of raw_variables uses (" + to_string(new_raw_variables_uses_size) + ") "
                            "must be equal to raw_variables size (" + to_string(raw_variables.size()) + "). \n");

    for(size_t i = 0; i < new_raw_variables_uses.size(); i++)
        raw_variables[i].set_use(new_raw_variables_uses[i]);

    input_dimensions = {get_variables_number(VariableUse::Input)};

    target_dimensions = {get_variables_number(VariableUse::Target)};
}


void DataSet::set_raw_variable_uses(const vector<VariableUse>& new_raw_variables_uses)
{
    const Index new_raw_variables_uses_size = new_raw_variables_uses.size();

    if(new_raw_variables_uses_size != raw_variables.size())
        throw runtime_error("Size of raw_variables uses (" + to_string(new_raw_variables_uses_size) + ") "
                            "must be equal to raw_variables size (" + to_string(raw_variables.size()) + ").\n");

    for(size_t i = 0; i < new_raw_variables_uses.size(); i++)
        raw_variables[i].set_use(new_raw_variables_uses[i]);

    input_dimensions = {get_variables_number(VariableUse::Input)};

    target_dimensions = {get_variables_number(VariableUse::Target)};
}


void DataSet::set_raw_variables(const VariableUse& variable_use)
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
        set_raw_variable_use(i, variable_use);
}


void DataSet::set_input_target_raw_variable_indices(const vector<Index>& input_raw_variables,
                                                    const vector<Index>& target_raw_variables)
{
    set_raw_variables(VariableUse::None);

    for(size_t i = 0; i < input_raw_variables.size(); i++)
        set_raw_variable_use(input_raw_variables[i], VariableUse::Input);

    for(size_t i = 0; i < target_raw_variables.size(); i++)
        set_raw_variable_use(target_raw_variables[i], VariableUse::Target);
}


// void DataSet::set_input_target_raw_variable_indices(const vector<string>& input_raw_variables,
//                                                     const vector<string>& target_raw_variables)
// {
//     set_raw_variables(VariableUse::None);

//     for(size_t i = 0; i < input_raw_variables.size(); i++)
//         set_raw_variable_use(input_raw_variables[i], VariableUse::Input);

//     for(size_t i = 0; i < target_raw_variables.size(); i++)
//         set_raw_variable_use(target_raw_variables[i], VariableUse::Target);
// }


void DataSet::set_input_raw_variables_unused()
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
        if(raw_variables[i].use == DataSet::VariableUse::Input) 
            set_raw_variable_use(i, VariableUse::None);
}


void DataSet::set_raw_variable_use(const Index& index, const VariableUse& new_use)
{
    raw_variables[index].use = new_use;
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


void DataSet::set_raw_variable_types(const RawVariableType& new_type)
{
    for(size_t i = 0; i < raw_variables.size(); i ++)
        raw_variables[i].type = new_type;
}


void DataSet::set_variable_names(const vector<string>& new_variables_names)
{
    const Index raw_variables_number = get_raw_variables_number();

    Index index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
        if(raw_variables[i].type == RawVariableType::Categorical)
            for(Index j = 0; j < raw_variables[i].get_categories_number(); j++)
                raw_variables[i].categories[j] = new_variables_names[index++];
        else
            raw_variables[i].name = new_variables_names[index++];
}


void DataSet::set_raw_variable_names(const vector<string>& new_names)
{
    const Index new_names_size = new_names.size();
    const Index raw_variables_number = get_raw_variables_number();

    if(new_names_size != raw_variables_number)
        throw runtime_error("Size of names (" + to_string(new_names.size()) + ") "
                            "is not equal to raw_variables number (" + to_string(raw_variables_number) + ").\n");

    for(Index i = 0; i < raw_variables_number; i++)
        raw_variables[i].name = get_trimmed(new_names[i]);
}


void DataSet::set(const VariableUse& variable_use)
{
    const Index raw_variables_number = get_raw_variables_number();

    for (Index i = 0; i < raw_variables_number; i++)
    {
        if (raw_variables[i].type == RawVariableType::Constant)
            continue;

        raw_variables[i].set_use(variable_use);
    }
}


void DataSet::set_raw_variables_number(const Index& new_raw_variables_number)
{
    raw_variables.resize(new_raw_variables_number);
}


void DataSet::set_raw_variable_scalers(const Scaler& scalers)
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
        raw_variables[i].scaler = scalers;
}


void DataSet::set_raw_variable_scalers(const vector<Scaler>& new_scalers)
{
    const Index raw_variables_number = get_raw_variables_number();

    if(new_scalers.size() != raw_variables_number)
        throw runtime_error("Size of raw_variable scalers(" + to_string(new_scalers.size()) + ") "
                            "has to be the same as raw_variables numbers(" + to_string(raw_variables_number) + ").\n");

    for(Index i = 0; i < raw_variables_number; i++)
        raw_variables[i].scaler = new_scalers[i];
}


void DataSet::set_binary_raw_variables()
{
    Index variable_index = 0;

    const Index raw_variables_number = get_raw_variables_number();

    for(Index raw_variable_index = 0; raw_variable_index < raw_variables_number; raw_variable_index++)
    {
        RawVariable& raw_variable = raw_variables[raw_variable_index];

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
    }
}


void DataSet::unuse_constant_raw_variables()
{
    Index variable_index = 0;

    const Index raw_variables_number = get_raw_variables_number();

    for(Index raw_variable_index = 0; raw_variable_index < raw_variables_number; raw_variable_index++)
    {
        RawVariable& raw_variable = raw_variables[raw_variable_index];

        if(raw_variable.type == RawVariableType::Numeric)
        {
            const TensorMap<Tensor<type, 1>> data_column = tensor_map(data, variable_index);

            if(is_constant_vector(data_column))
                raw_variable.set(raw_variable.name, VariableUse::None, RawVariableType::Constant);

            variable_index++;
        }
        else if(raw_variable.type == RawVariableType::DateTime || raw_variable.type == RawVariableType::Constant)
        {
            variable_index++;
        }
        else if(raw_variable.type == RawVariableType::Binary)
        {
            if(raw_variable.get_categories_number() == 1)
                raw_variable.set(raw_variable.name, VariableUse::None, RawVariableType::Constant);

            variable_index++;
        }
        else if(raw_variable.type == RawVariableType::Categorical)
        {           
            if(raw_variable.get_categories_number() == 1)
                raw_variable.set(raw_variable.name, VariableUse::None, RawVariableType::Constant);

            variable_index += raw_variable.get_categories_number();        
        }
    }
}


void DataSet::set_input_dimensions(const dimensions& new_input_dimensions)
{
    input_dimensions = new_input_dimensions;
}


void DataSet::set_target_dimensions(const dimensions& new_targets_dimensions)
{
    target_dimensions = new_targets_dimensions;
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


const filesystem::path& DataSet::get_data_path() const
{
    return data_path;
}


const bool& DataSet::get_header_line() const
{
    return has_header;
}


const bool& DataSet::get_has_sample_ids() const
{
    return has_sample_ids;
}


vector<string> DataSet::get_sample_ids() const
{
    return sample_ids;
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


Tensor<type, 2> DataSet::get_data(const SampleUse& sample_use) const
{
    const vector<Index> variable_indices = get_used_variable_indices();

    const vector<Index> sample_indices = get_sample_indices(SampleUse::Training);

    Tensor<type, 2> this_data(sample_indices.size(), variable_indices.size());

    fill_tensor_data(data, sample_indices, variable_indices, this_data.data());

    return this_data;
}


Tensor<type, 2> DataSet::get_data(const VariableUse& variable_use) const
{
    const Index samples_number = get_samples_number();

    vector<Index> indices(samples_number);
    iota(indices.begin(), indices.end(), 0);

    const vector<Index> variable_indices = get_variable_indices(variable_use);

    Tensor<type, 2> this_data(indices.size(), variable_indices.size());

    fill_tensor_data(data, indices, variable_indices, this_data.data());

    return this_data;
}


Tensor<type, 2> DataSet::get_data(const SampleUse& sample_use, const VariableUse& variable_use) const
{
    const vector<Index> sample_indices = get_sample_indices(sample_use);

    const vector<Index> variable_indices = get_variable_indices(variable_use);

    Tensor<type, 2> this_data(sample_indices.size(), variable_indices.size());

    fill_tensor_data(data, sample_indices, variable_indices, this_data.data());

    return this_data;
}


Tensor<type, 1> DataSet::get_sample_data(const Index& index) const
{
    return data.chip(index,0);
}


Tensor<type, 1> DataSet::get_sample_data(const Index& sample_index, const vector<Index>& variable_indices) const
{
    const Index variables_number = variable_indices.size();

    Tensor<type, 1 > row(variables_number);

    #pragma omp parallel for
    for(Index i = 0; i < variables_number; i++)
        row(i) = data(sample_index, variable_indices[i]);

    return row;
}


Tensor<type, 2> DataSet::get_sample_input_data(const Index&  sample_index) const
{
    const Index input_variables_number = get_variables_number(VariableUse::Input);

    const vector<Index> input_variable_indices = get_variable_indices(DataSet::VariableUse::Input);

    Tensor<type, 2> inputs(1, input_variables_number);

    for(Index i = 0; i < input_variables_number; i++)
        inputs(0, i) = data(sample_index, input_variable_indices[i]);

    return inputs;
}


Tensor<type, 2> DataSet::get_sample_target_data(const Index&  sample_index) const
{
    const vector<Index> target_variable_indices = get_variable_indices(DataSet::VariableUse::Target);

    Tensor<type, 2> sample_target_data(1, target_variable_indices.size());

    fill_tensor_data(data, vector<Index>(sample_index), target_variable_indices, sample_target_data.data());

    return sample_target_data;
}


Index DataSet::get_raw_variable_index(const string& column_name) const
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
        if(raw_variables[i].name == column_name) 
            return i;

    throw runtime_error("Cannot find " + column_name + "\n");
}


Index DataSet::get_raw_variable_index(const Index& variable_index) const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index total_variables_number = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        total_variables_number += (raw_variables[i].type == RawVariableType::Categorical)
            ? raw_variables[i].get_categories_number()
            : 1;

        if(variable_index+1 <= total_variables_number) 
            return i;
    }

    throw runtime_error("Cannot find variable index: " + to_string(variable_index) + ".\n");
}


vector<vector<Index>> DataSet::get_variable_indices() const
{
    const Index raw_variables_number = get_raw_variables_number();

    vector<vector<Index>> indices(raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
        indices[i] = get_variable_indices(i);

    return indices;
}


vector<Index> DataSet::get_variable_indices(const Index& raw_variable_index) const
{
    Index index = 0;

    for(Index i = 0; i < raw_variable_index; i++)
        index += (raw_variables[i].type == RawVariableType::Categorical)
            ? raw_variables[i].categories.size()
            : 1;

    const RawVariable& raw_variable = raw_variables[raw_variable_index];

    if(raw_variable.type == RawVariableType::Categorical)
    {
        vector<Index> indices(raw_variable.categories.size());

        for(size_t j = 0; j < raw_variable.categories.size(); j++)
            indices[j] = index + j;

        return indices;
    }

    return vector<Index>(1, index);
}


Tensor<type, 2> DataSet::get_raw_variable_data(const Index& raw_variable_index) const
{
    Index raw_variables_number = 1;
    const Index rows_number = data.dimension(0);

    if(raw_variables[raw_variable_index].type == RawVariableType::Categorical)
        raw_variables_number = raw_variables[raw_variable_index].get_categories_number();

    const Eigen::array<Index, 2> extents = {rows_number, raw_variables_number};
    const Eigen::array<Index, 2> offsets = {0, get_variable_indices(raw_variable_index)[0]};

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

    for(size_t raw_variable_index = column_index_start; raw_variable_index < raw_variables.size(); raw_variable_index++)
        if(data(sample_index, raw_variable_index) == 1)
            return raw_variables[column_index_start].categories[raw_variable_index - column_index_start];

    throw runtime_error("Sample does not have a valid one-hot encoded category.");
}


Tensor<type, 2> DataSet::get_raw_variable_data(const Index& raw_variable_index, const vector<Index>& rows_indices) const
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


vector<vector<string>> DataSet::get_data_file_preview() const
{
    return data_file_preview;
}


void DataSet::set(const filesystem::path& data_path,
                  const string& separator,
                  const bool& new_has_header,
                  const bool& new_has_ids,
                  const DataSet::Codification& new_codification)
{
    set_default();

    set_data_path(data_path);

    set_separator_string(separator);

    set_has_header(new_has_header);

    set_has_ids(new_has_ids);

    set_codification(new_codification);

    read_csv();

    set_default_raw_variables_scalers();

    set_default_raw_variables_uses();

    const Index input_variables_number = get_variables_number(VariableUse::Input);
    const Index target_variables_number = get_variables_number(VariableUse::Target);

    input_dimensions = {input_variables_number};

    target_dimensions = {target_variables_number};

}


void DataSet::set(const Index& new_samples_number,
                  const dimensions& new_input_dimensions,
                  const dimensions& new_target_dimensions)
{
    input_dimensions = new_input_dimensions;

    target_dimensions = new_target_dimensions;

    if (new_samples_number == 0 
    || new_input_dimensions.empty() 
    || new_target_dimensions.empty())
        return;

    const Index new_inputs_number = accumulate(new_input_dimensions.begin(), 
                                               new_input_dimensions.end(), 
                                               1, 
                                               multiplies<Index>());

    const Index new_targets_number = accumulate(new_target_dimensions.begin(),
                                                new_target_dimensions.end(),
                                                1,
                                                multiplies<Index>());
    
    const Index new_variables_number = new_inputs_number + new_targets_number;

    data.resize(new_samples_number, new_variables_number);

    raw_variables.resize(new_variables_number);

    set_default();
    
    if (model_type == ModelType::ImageClassification)
    {        
        const Index raw_variables_number = new_inputs_number + 1;

        raw_variables.resize(raw_variables_number);

        for (Index i = 0; i < new_inputs_number; i++)
            raw_variables[i].set("p_" + to_string(i + 1),
                VariableUse::Input,
                RawVariableType::Numeric,
                Scaler::ImageMinMax);
        
        if (new_targets_number == 1)
            raw_variables[raw_variables_number - 1].set("target",
                VariableUse::Target,
                RawVariableType::Binary,
                Scaler::None); 
        else
            raw_variables[raw_variables_number - 1].set("target",
                VariableUse::Target,
                RawVariableType::Categorical,
                Scaler::None);     
    }
    else
    {
        for (Index i = 0; i < new_variables_number; i++)
        {
            RawVariable& raw_variable = raw_variables[i];

            raw_variable.type = RawVariableType::Numeric;
            raw_variable.name = "variable_" + to_string(i + 1);

            raw_variable.use = (i < new_inputs_number)
                ? VariableUse::Input
                : VariableUse::Target;
        }
    }

    sample_uses.resize(new_samples_number);
    
    split_samples_random();
}


void DataSet::set(const filesystem::path& file_name)
{
    load(file_name);
}


void DataSet::set_display(const bool& new_display)
{
    display = new_display;
}


void DataSet::set_default()
{
    const unsigned int threads_number = thread::hardware_concurrency();
    thread_pool = make_unique<ThreadPool>(threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);

    has_header = false;

    separator = Separator::Comma;

    missing_values_label = "NA";

    //set_default_raw_variables_uses();

    set_default_raw_variables_names();
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
    if (new_data.dimension(0) != get_samples_number())
        throw runtime_error("Rows number is not equal to samples number");

    if (new_data.dimension(1) != get_variables_number())
        throw runtime_error("Columns number is not equal to variables number");

    data = new_data;
}


void DataSet::set_data_path(const filesystem::path& new_data_path)
{
    data_path = new_data_path;
}


void DataSet::set_has_header(const bool& new_has_header)
{
    has_header = new_has_header;
}


void DataSet::set_has_ids(const bool& new_has_ids)
{
    has_sample_ids = new_has_ids;
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
    thread_pool = make_unique<ThreadPool>(new_threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), new_threads_number);
}


Tensor<Index, 1> DataSet::unuse_repeated_samples()
{
    const Index samples_number = get_samples_number();

    Tensor<Index, 1> repeated_samples;

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

                push_back(repeated_samples, j);
            }
        }
    }

    return repeated_samples;
}


vector<string> DataSet::unuse_uncorrelated_raw_variables(const type& minimum_correlation)
{
    vector<string> unused_raw_variables;

    const Tensor<Correlation, 2> correlations = calculate_input_target_raw_variable_pearson_correlations();

    const Index input_raw_variables_number = get_raw_variables_number(VariableUse::Input);
    const Index target_raw_variables_number = get_raw_variables_number(VariableUse::Target);

    const vector<Index> input_raw_variable_indices = get_raw_variable_indices(VariableUse::Input);

    for(Index i = 0; i < input_raw_variables_number; i++)
    {
        const Index input_raw_variable_index = input_raw_variable_indices[i];

        for(Index j = 0; j < target_raw_variables_number; j++)
        {
            if(!isnan(correlations(i, j).r)
            && abs(correlations(i, j).r) < minimum_correlation
            && raw_variables[input_raw_variable_index].use != VariableUse::None)
            {
                raw_variables[input_raw_variable_index].set_use(VariableUse::None);

                unused_raw_variables.push_back(raw_variables[input_raw_variable_index].name);
            }
        }
    }

    return unused_raw_variables;
}


vector<string> DataSet::unuse_multicollinear_raw_variables(Tensor<Index, 1>& original_variable_indices, Tensor<Index, 1>& override_variable_indices)
{
    vector<string> unused_raw_variables;

    for(Index i = 0; i < original_variable_indices.size(); i++)
    {
        const Index original_raw_variable_index = original_variable_indices(i);

        bool found = false;

        for(Index j = 0; j < override_variable_indices.size(); j++)
        {
            if(original_raw_variable_index == override_variable_indices(j))
            {
                found = true;
                break;
            }
        }

        const Index raw_variable_index = get_raw_variable_index(original_raw_variable_index);

        if(!found && raw_variables[raw_variable_index].use != VariableUse::None)
        {
            raw_variables[raw_variable_index].set_use(VariableUse::None);

            unused_raw_variables.push_back(raw_variables[raw_variable_index].name);
        }
    }

    return unused_raw_variables;
}


vector<Histogram> DataSet::calculate_raw_variable_distributions(const Index& bins_number) const
{
    const Index raw_variables_number = raw_variables.size();
    const Index used_raw_variables_number = get_used_raw_variables_number();
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const Index used_samples_number = used_sample_indices.size();

    vector<Histogram> histograms(used_raw_variables_number);

    Index variable_index = 0;
    Index used_raw_variable_index = 0;

    for (Index i = 0; i < raw_variables_number; i++)
    {
        const RawVariable& raw_variable = raw_variables[i];

        if (raw_variable.use == VariableUse::None)
        {
            variable_index += (raw_variable.type == RawVariableType::Categorical)
                ? raw_variable.get_categories_number()
                : 1;
            continue;
        }

        switch (raw_variable.type)
        {

        case RawVariableType::Numeric:
        {
            Tensor<type, 1> raw_variable_data(used_samples_number);

            for (Index j = 0; j < used_samples_number; j++)
                raw_variable_data(j) = data(used_sample_indices[j], variable_index);

            histograms[used_raw_variable_index++] = histogram(raw_variable_data, bins_number);

            variable_index++;
        }
            break;

        case RawVariableType::Categorical:
        {
            const Index categories_number = raw_variable.get_categories_number();

            Tensor<Index, 1> categories_frequencies(categories_number);
            categories_frequencies.setZero();
            Tensor<type, 1> centers(categories_number);

            for (Index j = 0; j < categories_number; j++)
            {
                for (Index k = 0; k < used_samples_number; k++)
                    if (abs(data(used_sample_indices[k], variable_index) - type(1)) < NUMERIC_LIMITS_MIN)
                        categories_frequencies(j)++;

                centers(j) = type(j);

                variable_index++;
            }

            histograms[used_raw_variable_index].frequencies = categories_frequencies;
            histograms[used_raw_variable_index].centers = centers;

            used_raw_variable_index++;
        }
            break;

        case RawVariableType::Binary:
        {
            Tensor<Index, 1> binary_frequencies(2);
            binary_frequencies.setZero();

            for (Index j = 0; j < used_samples_number; j++)
                binary_frequencies(abs(data(used_sample_indices[j], variable_index) - type(1)) < NUMERIC_LIMITS_MIN
                    ? 0
                    : 1)++;

            histograms[used_raw_variable_index].frequencies = binary_frequencies;
            variable_index++;
            used_raw_variable_index++;
        }
            break;
        
        case RawVariableType::DateTime:

            variable_index++;

            break;

        default:

            throw runtime_error("Unknown raw variable type.");
        }
    }

    return histograms;
}


vector<BoxPlot> DataSet::calculate_raw_variables_box_plots() const
{
    const Index raw_variables_number = get_raw_variables_number();

    const vector<Index> used_sample_indices = get_used_sample_indices();

    vector<BoxPlot> box_plots(raw_variables_number);

//    Index used_raw_variable_index = 0;
    Index variable_index = 0;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        const RawVariable& raw_variable = raw_variables[i];

        if(raw_variable.type == RawVariableType::Numeric
        || raw_variable.type == RawVariableType::Binary)
        {
            if(raw_variable.use != VariableUse::None)
            {
                box_plots[i] = box_plot(data.chip(variable_index, 1), used_sample_indices);

//                used_raw_variable_index++;
            }

            variable_index++;
        }
        else if(raw_variable.type == RawVariableType::Categorical)
        {
            variable_index += raw_variable.get_categories_number();
        }
        else
        {
            variable_index++;
        }
    }

    return box_plots;
}


Index DataSet::calculate_used_negatives(const Index& target_index)
{
    Index negatives = 0;

    const vector<Index> used_indices = get_used_sample_indices();

    const Index used_samples_number = used_indices.size();

    for(Index i = 0; i < used_samples_number; i++)
    {
        const Index training_index = used_indices[i];

        if (isnan(data(training_index, target_index))) 
            continue;
        
        if(abs(data(training_index, target_index)) < NUMERIC_LIMITS_MIN)
            negatives++;
        else if(abs(data(training_index, target_index) - type(1)) > NUMERIC_LIMITS_MIN
             || data(training_index, target_index) < type(0))
            throw runtime_error("Training sample is neither a positive nor a negative: "
                                + to_string(training_index) + "-" + to_string(target_index) + "-" + to_string(data(training_index, target_index)));        
    }

    return negatives;
}


vector<Descriptives> DataSet::calculate_variable_descriptives() const
{
    return descriptives(data);
}


vector<Descriptives> DataSet::calculate_used_variable_descriptives() const
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_variable_indices = get_used_variable_indices();

    return descriptives(data, used_sample_indices, used_variable_indices);
}


vector<Descriptives> DataSet::calculate_raw_variable_descriptives_positive_samples() const
{
    const Index target_index = get_variable_indices(DataSet::VariableUse::Target)[0];

    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> input_variable_indices = get_variable_indices(DataSet::VariableUse::Input);

    const Index samples_number = used_sample_indices.size();

    Index positive_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
        if(abs(data(used_sample_indices[i], target_index) - type(1)) < NUMERIC_LIMITS_MIN)
            positive_samples_number++;

    vector<Index> positive_used_sample_indices(positive_samples_number);
    Index positive_sample_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = used_sample_indices[i];

        if(abs(data(sample_index, target_index) - type(1)) < NUMERIC_LIMITS_MIN)
            positive_used_sample_indices[positive_sample_index++] = sample_index;
    }

    return descriptives(data, positive_used_sample_indices, input_variable_indices);
}


vector<Descriptives> DataSet::calculate_raw_variable_descriptives_negative_samples() const
{
    const Index target_index = get_variable_indices(DataSet::VariableUse::Target)[0];

    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> input_variable_indices = get_variable_indices(DataSet::VariableUse::Input);

    const Index samples_number = used_sample_indices.size();

    Index negative_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
        if(data(used_sample_indices[i], target_index) < NUMERIC_LIMITS_MIN)
            negative_samples_number++;

    vector<Index> negative_used_sample_indices(negative_samples_number);
    Index negative_sample_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = used_sample_indices[i];

        if(data(sample_index, target_index) < NUMERIC_LIMITS_MIN)
            negative_used_sample_indices[negative_sample_index++] = sample_index;
    }

    return descriptives(data, negative_used_sample_indices, input_variable_indices);
}


vector<Descriptives> DataSet::calculate_raw_variable_descriptives_categories(const Index& class_index) const
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> input_variable_indices = get_variable_indices(DataSet::VariableUse::Input);

    const Index samples_number = used_sample_indices.size();

    // Count used class samples

    Index class_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
        if(abs(data(used_sample_indices[i], class_index) - type(1)) < NUMERIC_LIMITS_MIN)
            class_samples_number++;

    vector<Index> class_used_sample_indices(class_samples_number, 0);

    Index class_sample_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = used_sample_indices[i];

        if(abs(data(sample_index, class_index) - type(1)) < NUMERIC_LIMITS_MIN)
            class_used_sample_indices[class_sample_index++] = sample_index;
    }

    return descriptives(data, class_used_sample_indices, input_variable_indices);
}


vector<Descriptives> DataSet::calculate_variable_descriptives(const VariableUse& variable_use) const
{
    const vector<Index> used_sample_indices = get_used_sample_indices();

    const vector<Index> input_variable_indices = get_variable_indices(variable_use);

    return descriptives(data, used_sample_indices, input_variable_indices);
}


vector<Descriptives> DataSet::calculate_testing_target_variable_descriptives() const
{
    const vector<Index> testing_indices = get_sample_indices(SampleUse::Testing);

    const vector<Index> target_variable_indices = get_variable_indices(DataSet::VariableUse::Target);

    return descriptives(data, testing_indices, target_variable_indices);
}


Tensor<type, 1> DataSet::calculate_used_variables_minimums() const
{
    return column_minimums(data, get_used_sample_indices(), get_used_variable_indices());
}


Tensor<type, 1> DataSet::calculate_means(const DataSet::SampleUse& sample_use, 
                                         const DataSet::VariableUse& variable_use) const
{
    const vector<Index> sample_indices = get_sample_indices(sample_use);

    const vector<Index> variable_indices = get_variable_indices(variable_use);

    return mean(data, sample_indices, variable_indices);
}


Index DataSet::get_gmt() const
{
    return gmt;
}


void DataSet::set_gmt(const Index& new_gmt)
{
    gmt = new_gmt;
}


Tensor<Correlation, 2> DataSet::calculate_input_target_raw_variable_pearson_correlations() const
{
    const Index input_raw_variables_number = get_raw_variables_number(VariableUse::Input);
    const Index target_raw_variables_number = get_raw_variables_number(VariableUse::Target);

    const vector<Index> input_raw_variable_indices = get_raw_variable_indices(VariableUse::Input);
    const vector<Index> target_raw_variable_indices = get_raw_variable_indices(VariableUse::Target);

    const vector<Index> used_sample_indices = get_used_sample_indices();

    Tensor<Correlation, 2> correlations(input_raw_variables_number, target_raw_variables_number);

//#pragma omp parallel for

    for(Index i = 0; i < input_raw_variables_number; i++)
    {
        const Index input_raw_variable_index = input_raw_variable_indices[i];

        const Tensor<type, 2> input_raw_variable_data
            = get_raw_variable_data(input_raw_variable_index, used_sample_indices);

        for(Index j = 0; j < target_raw_variables_number; j++)
        {
            const Index target_raw_variable_index = target_raw_variable_indices[j];

            const Tensor<type, 2> target_raw_variable_data 
                = get_raw_variable_data(target_raw_variable_index, used_sample_indices);

            correlations(i, j) = correlation(thread_pool_device.get(), input_raw_variable_data, target_raw_variable_data);
        }
    }

    return correlations;
}


Tensor<Correlation, 2> DataSet::calculate_input_target_raw_variable_spearman_correlations() const
{
    const Index input_raw_variables_number = get_raw_variables_number(VariableUse::Input);
    const Index target_raw_variables_number = get_raw_variables_number(VariableUse::Target);

    const vector<Index> input_raw_variable_indices = get_raw_variable_indices(VariableUse::Input);
    const vector<Index> target_raw_variable_indices = get_raw_variable_indices(VariableUse::Target);

    const vector<Index> used_sample_indices = get_used_sample_indices();

    Tensor<Correlation, 2> correlations(input_raw_variables_number, target_raw_variables_number);

    for(Index i = 0; i < input_raw_variables_number; i++)
    {
        const Index input_index = input_raw_variable_indices[i];

        const Tensor<type, 2> input_raw_variable_data = get_raw_variable_data(input_index, used_sample_indices);

        for(Index j = 0; j < target_raw_variables_number; j++)
        {
            const Index target_index = target_raw_variable_indices[j];

            const Tensor<type, 2> target_raw_variable_data = get_raw_variable_data(target_index, used_sample_indices);

            correlations(i, j) = correlation_spearman(thread_pool_device.get(), input_raw_variable_data, target_raw_variable_data);
        }
    }

    return correlations;
}


bool DataSet::has_nan() const
{
    const Index rows_number = data.dimension(0);

    for(Index i = 0; i < rows_number; i++)
        if(sample_uses[i] != SampleUse::None)
            if(has_nan_row(i)) 
                return true;

    return false;
}


bool DataSet::has_nan_row(const Index& row_index) const
{
    const Index variables_number = get_variables_number();

    for(Index j = 0; j < variables_number; j++)
        if(isnan(data(row_index, j)))
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
    const Index inputs_number = get_variables_number(VariableUse::Input);
    const Index targets_number = get_raw_variables_number(VariableUse::Target);

    const vector<string> input_names = get_raw_variable_names(VariableUse::Input);
    const vector<string> targets_name = get_raw_variable_names(VariableUse::Target);

    const Tensor<Correlation, 2> correlations = calculate_input_target_raw_variable_pearson_correlations();

    for(Index j = 0; j < targets_number; j++)
        for(Index i = 0; i < inputs_number; i++)
            cout << targets_name[j] << " - " << input_names[i] << ": " << correlations(i, j).r << endl;
}


void DataSet::print_top_input_target_raw_variables_correlations() const
{
    const Index inputs_number = get_raw_variables_number(VariableUse::Input);
    const Index targets_number = get_raw_variables_number(VariableUse::Target);

    const vector<string> input_names = get_variable_names(DataSet::VariableUse::Input);
    const vector<string> targets_name = get_variable_names(DataSet::VariableUse::Target);

    const Tensor<type, 2> correlations = get_correlation_values(calculate_input_target_raw_variable_pearson_correlations());

    Tensor<type, 1> target_correlations(inputs_number);

    Tensor<string, 2> top_correlations(inputs_number, 2);

    map<type,string> top_correlation;

    for(Index i = 0 ; i < inputs_number; i++)
        for(Index j = 0 ; j < targets_number ; j++)
            top_correlation.insert(pair<type,string>(correlations(i, j), input_names[i] + " - " + targets_name[j]));

    map<type,string>::iterator it;

    for(it = top_correlation.begin(); it != top_correlation.end(); it++)
        cout << "Correlation: " << (*it).first << "  between  " << (*it).second << endl;
}


Tensor<Correlation, 2> DataSet::calculate_input_raw_variable_pearson_correlations() const
{
    // list to return

    const vector<Index> input_raw_variable_indices = get_raw_variable_indices(VariableUse::Input);

    const Index input_raw_variables_number = get_raw_variables_number(VariableUse::Input);

    Tensor<Correlation, 2> correlations_pearson(input_raw_variables_number, input_raw_variables_number);

    for (Index i = 0; i < input_raw_variables_number; i++)
    {
        const Index current_input_index_i = input_raw_variable_indices[i];

        const Tensor<type, 2> input_i = get_raw_variable_data(current_input_index_i);

        //if(display) cout << "Calculating " << raw_variables(current_input_index_i).name << " correlations. " << endl;

        if (is_constant_matrix(input_i)) continue;

        correlations_pearson(i, i).set_perfect();
        correlations_pearson(i, i).method = Correlation::Method::Pearson;

        for (Index j = i+1; j < input_raw_variables_number; j++)
        {
            const Index current_input_index_j = input_raw_variable_indices[j];

            const Tensor<type, 2> input_j = get_raw_variable_data(current_input_index_j);
                correlations_pearson(i, j) = correlation(thread_pool_device.get(), input_i, input_j);

            if (correlations_pearson(i, j).r > type(1) - NUMERIC_LIMITS_MIN)
                correlations_pearson(i, j).r = type(1);

            correlations_pearson(j, i) = correlations_pearson(i, j);
        }
    }

    return correlations_pearson;
}


Tensor<Correlation, 2> DataSet::calculate_input_raw_variable_spearman_correlations() const
{
    const vector<Index> input_raw_variable_indices = get_raw_variable_indices(VariableUse::Input);

    const Index input_raw_variables_number = get_raw_variables_number(VariableUse::Input);

    Tensor<Correlation, 2> correlations_spearman(input_raw_variables_number, input_raw_variables_number);

    for(Index i = 0; i < input_raw_variables_number; i++)
    {
        const Index input_raw_variable_index_i = input_raw_variable_indices[i];

        const Tensor<type, 2> input_i = get_raw_variable_data(input_raw_variable_index_i);

        //if(display) cout << "Calculating " << raw_variables(current_input_index_i).name << " correlations. " << endl;

        if (is_constant_matrix(input_i)) continue;

        correlations_spearman(i, i).set_perfect();
        correlations_spearman(i, i).method = Correlation::Method::Spearman;

        for(Index j = i + 1; j < input_raw_variables_number; j++)
        {
            const Index input_raw_variable_index_j = input_raw_variable_indices[j];

            const Tensor<type, 2> input_j = get_raw_variable_data(input_raw_variable_index_j);

            correlations_spearman(i, j) = correlation_spearman(thread_pool_device.get(), input_i, input_j);

            if(correlations_spearman(i, j).r > type(1) - NUMERIC_LIMITS_MIN)
                correlations_spearman(i, j).r = type(1);

            correlations_spearman(j, i) = correlations_spearman(i, j);
        }
    }

    return correlations_spearman;
}


void DataSet::print_inputs_correlations() const
{
    const Tensor<type, 2> inputs_correlations 
        = get_correlation_values(calculate_input_raw_variable_pearson_correlations());

    cout << inputs_correlations << endl;
}


void DataSet::print_data_file_preview() const
{
    const Index size = data_file_preview.size();

    for(Index i = 0;  i < size; i++)
    {
        for(size_t j = 0; j < data_file_preview[i].size(); j++)
            cout << data_file_preview[i][j] << " ";

        cout << endl;
    }
}


void DataSet::print_top_inputs_correlations() const
{
    const Index variables_number = get_variables_number(VariableUse::Input);

    const vector<string> variables_name = get_variable_names(DataSet::VariableUse::Input);

    const Tensor<type, 2> variables_correlations = get_correlation_values(calculate_input_raw_variable_pearson_correlations());

    const Index correlations_number = variables_number*(variables_number-1)/2;

    Tensor<string, 2> top_correlations(correlations_number, 3);

    map<type, string> top_correlation;

    for(Index i = 0; i < variables_number; i++)
    {
        for(Index j = i; j < variables_number; j++)
        {
            if(i == j) continue;

            top_correlation.insert(pair<type,string>(variables_correlations(i, j), variables_name[i] + " - " + variables_name[j]));
        }
    }

    map<type,string> ::iterator it;

    for(it = top_correlation.begin(); it != top_correlation.end(); it++)
        cout << "Correlation: " << (*it).first << "  between  " << (*it).second << endl;
}


void DataSet::set_default_raw_variables_scalers()
{
    const Index raw_variables_number = raw_variables.size();

    if(model_type == ModelType::ImageClassification)
        set_raw_variable_scalers(Scaler::ImageMinMax);
    else
        for(Index i = 0; i < raw_variables_number; i++)
            raw_variables[i].scaler = (raw_variables[i].type == RawVariableType::Numeric)
                ? Scaler::MeanStandardDeviation
                : Scaler::MinimumMaximum;
}


vector<Descriptives> DataSet::scale_data()
{
    const Index variables_number = get_variables_number();

    const vector<Descriptives> variable_descriptives = calculate_variable_descriptives();

    Index raw_variable_index;

    for(Index i = 0; i < variables_number; i++)
    {
        raw_variable_index = get_raw_variable_index(i);

        switch(raw_variables[raw_variable_index].scaler)
        {
        case Scaler::None:
            break;

        case Scaler::MinimumMaximum:
            scale_minimum_maximum(data, i, variable_descriptives[i]);
            break;

        case Scaler::MeanStandardDeviation:
            scale_mean_standard_deviation(data, i, variable_descriptives[i]);
            break;

        case Scaler::StandardDeviation:
            scale_standard_deviation(data, i, variable_descriptives[i]);
            break;

        case Scaler::Logarithm:
            scale_logarithmic(data, i);
            break;

        default:
            throw runtime_error("Unknown scaler: " + to_string(int(raw_variables[i].scaler)) + "\n");
        }
    }

    return variable_descriptives;
}


vector<Descriptives> DataSet::scale_variables(const VariableUse& variable_use)
{
    const Index input_variables_number = get_variables_number(variable_use);

    const vector<Index> input_variable_indices = get_variable_indices(variable_use);
    const vector<Scaler> input_variable_scalers = get_variable_scalers(DataSet::VariableUse::Input);

    const vector<Descriptives> input_variable_descriptives = calculate_variable_descriptives(variable_use);

    for(Index i = 0; i < input_variables_number; i++)
    {
        switch(input_variable_scalers[i])
        {
        case Scaler::None:
            break;

        case Scaler::MinimumMaximum:
            scale_minimum_maximum(data, input_variable_indices[i], input_variable_descriptives[i]);
            break;

        case Scaler::MeanStandardDeviation:
            scale_mean_standard_deviation(data, input_variable_indices[i], input_variable_descriptives[i]);
            break;

        case Scaler::StandardDeviation:
            scale_standard_deviation(data, input_variable_indices[i], input_variable_descriptives[i]);
            break;

        case Scaler::Logarithm:
            scale_logarithmic(data, input_variable_indices[i]);
            break;

        default:
            throw runtime_error("Unknown scaling inputs method: " + to_string(int(input_variable_scalers[i])) + "\n");
        }
    }

    return input_variable_descriptives;
}


void DataSet::unscale_variables(const VariableUse& variable_use, 
                                const vector<Descriptives>& input_variable_descriptives)
{
    const Index input_variables_number = get_variables_number(variable_use);

    const vector<Index> input_variable_indices = get_variable_indices(variable_use);

    const vector<Scaler> input_variable_scalers = get_variable_scalers(DataSet::VariableUse::Input);

    for(Index i = 0; i < input_variables_number; i++)
    {
        switch(input_variable_scalers[i])
        {
        case Scaler::None:
            break;

        case Scaler::MinimumMaximum:
            unscale_minimum_maximum(data, input_variable_indices[i], input_variable_descriptives[i]);
            break;

        case Scaler::MeanStandardDeviation:
            unscale_mean_standard_deviation(data, input_variable_indices[i], input_variable_descriptives[i]);
            break;

        case Scaler::StandardDeviation:
            unscale_standard_deviation(data, input_variable_indices[i], input_variable_descriptives[i]);
            break;

        case Scaler::Logarithm:
            unscale_logarithmic(data, input_variable_indices[i]);
            break;

        case Scaler::ImageMinMax:
            unscale_image_minimum_maximum(data, input_variable_indices[i]);
            break;

        default:
            throw runtime_error("Unknown unscaling and unscaling method: " + to_string(int(input_variable_scalers[i])) + "\n");
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


void DataSet::to_XML(XMLPrinter& printer) const
{
    if (model_type == ModelType::Forecasting)
        throw runtime_error("Forecasting");

    if (model_type == ModelType::ImageClassification)
        throw runtime_error("Image classification");

    printer.OpenElement("DataSet");

    printer.OpenElement("DataSource");
    add_xml_element(printer, "FileType", "csv");

    add_xml_element(printer, "Path", data_path.string());

    add_xml_element(printer, "Separator", get_separator_name());
    add_xml_element(printer, "HasHeader", to_string(has_header));
    add_xml_element(printer, "HasSamplesId", to_string(has_sample_ids));
    add_xml_element(printer, "MissingValuesLabel", missing_values_label);
    add_xml_element(printer, "Codification", get_codification_string());
    printer.CloseElement();  

    printer.OpenElement("RawVariables");
    add_xml_element(printer, "RawVariablesNumber", to_string(get_raw_variables_number()));

    for (Index i = 0; i < get_raw_variables_number(); i++) 
    {
        printer.OpenElement("RawVariable");
        printer.PushAttribute("Item", to_string(i + 1).c_str());
        raw_variables[i].to_XML(printer);
        printer.CloseElement();  
    }

    printer.CloseElement();  

    printer.OpenElement("Samples");
    
    add_xml_element(printer, "SamplesNumber", to_string(get_samples_number()));
    
    if (has_sample_ids) 
        add_xml_element(printer, "SamplesId", string_tensor_to_string(sample_ids));
    
    add_xml_element(printer, "SamplesUses", tensor_to_string(get_sample_uses_vector()));
    printer.CloseElement();  

    printer.OpenElement("MissingValues");
    add_xml_element(printer, "MissingValuesMethod", get_missing_values_method_string());
    add_xml_element(printer, "MissingValuesNumber", to_string(missing_values_number));

    if (missing_values_number > 0) 
    {
        add_xml_element(printer, "RawVariablesMissingValuesNumber", tensor_to_string(raw_variables_missing_values_number));
        add_xml_element(printer, "RowsMissingValuesNumber", to_string(rows_missing_values_number));
    }

    printer.CloseElement();  

    printer.CloseElement();
}


void DataSet::from_XML(const XMLDocument& data_set_document) 
{
    const XMLElement* data_set_element = data_set_document.FirstChildElement("DataSet");
    if (!data_set_element) 
        throw runtime_error("Data set element is nullptr.\n");
    
    const XMLElement* data_source_element = data_set_element->FirstChildElement("DataSource");
    if (!data_source_element) 
        throw runtime_error("Data source element is nullptr.\n");
    
    set_data_path(read_xml_string(data_source_element, "Path"));
    set_separator_name(read_xml_string(data_source_element, "Separator"));
    set_has_header(read_xml_bool(data_source_element, "HasHeader"));
    set_has_ids(read_xml_bool(data_source_element, "HasSamplesId"));
    set_missing_values_label(read_xml_string(data_source_element, "MissingValuesLabel"));
    set_codification(read_xml_string(data_source_element, "Codification"));

    const XMLElement* raw_variables_element = data_set_element->FirstChildElement("RawVariables");

    if (!raw_variables_element)
        throw runtime_error("RawVariables element is nullptr.\n");

    set_raw_variables_number(read_xml_index(raw_variables_element, "RawVariablesNumber"));

    const XMLElement* start_element = raw_variables_element->FirstChildElement("RawVariablesNumber");

    for (size_t i = 0; i < raw_variables.size(); i++)
    {
        RawVariable& raw_variable = raw_variables[i];
        const XMLElement* raw_variable_element = start_element->NextSiblingElement("RawVariable");
        start_element = raw_variable_element;

        if (raw_variable_element->Attribute("Item") != std::to_string(i + 1))
            throw runtime_error("Raw variable item number (" + std::to_string(i + 1) + ") does not match (" + raw_variable_element->Attribute("Item") + ").\n");

        raw_variable.name = read_xml_string(raw_variable_element, "Name");
        raw_variable.set_scaler(read_xml_string(raw_variable_element, "Scaler"));
        raw_variable.set_use(read_xml_string(raw_variable_element, "Use"));
        raw_variable.set_type(read_xml_string(raw_variable_element, "Type"));

        if (raw_variable.type == RawVariableType::Categorical || raw_variable.type == RawVariableType::Binary) 
            raw_variable.categories = get_tokens(read_xml_string(raw_variable_element, "Categories"), ";");
    }

    if (has_sample_ids)
        sample_ids = get_tokens(read_xml_string(data_set_element, "SamplesId"), " ");

    const XMLElement* samples_element = data_set_element->FirstChildElement("Samples");

    if (!samples_element)
        throw runtime_error("Samples element is nullptr.\n");

    sample_uses.resize(read_xml_index(samples_element, "SamplesNumber"));
    set_sample_uses(get_tokens(read_xml_string(samples_element, "SamplesUses"), " "));

    // Missing values
    const XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");

    if (!missing_values_element)
        throw runtime_error("Missing values element is nullptr.\n");

    set_missing_values_method(read_xml_string(missing_values_element, "MissingValuesMethod"));
    missing_values_number = read_xml_index(missing_values_element, "MissingValuesNumber");

    if (missing_values_number > 0)
    {
        raw_variables_missing_values_number.resize(get_tokens(read_xml_string(missing_values_element, "RawVariablesMissingValuesNumber"), " ").size());

        for (Index i = 0; i < raw_variables_missing_values_number.size(); i++)
            raw_variables_missing_values_number(i) = stoi(get_tokens(read_xml_string(missing_values_element, "RawVariablesMissingValuesNumber"), " ")[i]);

        rows_missing_values_number = read_xml_index(missing_values_element, "RowsMissingValuesNumber");
    }

    set_display(read_xml_bool(data_set_element, "Display"));
}


void DataSet::print() const
{
    if(!display) return;
    
    const Index variables_number = get_variables_number();
    const Index input_variables_number = get_variables_number(VariableUse::Input);
    const Index samples_number = get_samples_number();
    const Index target_variables_bumber = get_variables_number(VariableUse::Target);
    const Index training_samples_number = get_samples_number(SampleUse::Training);
    const Index selection_samples_number = get_samples_number(SampleUse::Selection);
    const Index testing_samples_number = get_samples_number(SampleUse::Testing);
    const Index unused_samples_number = get_samples_number(SampleUse::None);

    cout << "Data set object summary:\n"
         << "Number of samples: " << samples_number << "\n"
         << "Number of variables: " << variables_number << "\n"
         << "Number of input variables: " << input_variables_number << "\n"
         << "Number of target variables: " << target_variables_bumber << "\n"
         << "Input variables dimensions: ";
   
    print_vector(get_input_dimensions());
         
    cout << "Target variables dimensions: ";
    
    print_vector(get_target_dimensions());
    
    cout << "Number of training samples: " << training_samples_number << endl
         << "Number of selection samples: " << selection_samples_number << endl
         << "Number of testing samples: " << testing_samples_number << endl
         << "Number of unused samples: " << unused_samples_number << endl;
   
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
        raw_variables[i].print();
}


void DataSet::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        return;

    XMLPrinter document;

    to_XML(document);

    file << document.CStr();
}


void DataSet::load(const filesystem::path& file_name)
{
    XMLDocument document;

    if (document.LoadFile(file_name.string().c_str()))
        throw runtime_error("Cannot load XML file " + file_name.string() + ".\n");

    from_XML(document);
}


void DataSet::print_raw_variables() const
{
    const Index raw_variables_number = get_raw_variables_number();

    for(Index i = 0; i < raw_variables_number; i++)
        raw_variables[i].print();

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
    }

    if(samples_number > 1)
    {
        const Tensor<type, 1> second_sample = data.chip(1, 0);

        cout << "Second sample: \n";

        for(int i = 0; i< second_sample.dimension(0); i++)
            cout  << second_sample(i) << "  ";
    }

    if(samples_number > 2)
    {
        const Tensor<type, 1> last_sample = data.chip(samples_number-1, 0);

        cout << "Last sample: \n";

        for(int i = 0; i< last_sample.dimension(0); i++)
            cout  << last_sample(i) << "  ";
    }

    cout << endl;
}


void DataSet::save_data() const
{
    ofstream file(data_path.c_str());

    if(!file.is_open())
        throw runtime_error("Cannot open matrix data file: " + data_path.string() + "\n");

    file.precision(20);

    const Index samples_number = get_samples_number();
    const Index variables_number = get_variables_number();

    const vector<string> variable_names = get_variable_names();

    const string separator_string = get_separator_string();

    if(has_sample_ids)
        file << "id" << separator_string;

    for(Index j = 0; j < variables_number; j++)
    {
        file << variable_names[j];

        if(j != variables_number-1)
            file << separator_string;
    }

    file << endl;

    for(Index i = 0; i < samples_number; i++)
    {
        if(has_sample_ids)
            file << sample_ids[i] << separator_string;

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


void DataSet::save_data_binary(const filesystem::path& binary_data_file_name) const
{
    ofstream file(binary_data_file_name);

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
    ifstream file(data_path);

    if (!file.is_open())
        throw runtime_error("Failed to open file: " + data_path.string());

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
    const Index targets_number = get_variables_number(VariableUse::Target);
    const vector<Index> target_variable_indices = get_variable_indices(DataSet::VariableUse::Target);

    Tensor<Index, 1> class_distribution;

    if(targets_number == 1) 
    {
        class_distribution.resize(2);

        const Index target_index = target_variable_indices[0];

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
                if(isnan(data(i,target_variable_indices[j])))
                    continue;

                if(data(i,target_variable_indices[j]) > type(0.5))
                    class_distribution(j)++;
            }
        }
    }

    return class_distribution;
}


vector<vector<Index>> DataSet::calculate_Tukey_outliers(const type& cleaning_parameter) const
{
    const Index samples_number = get_used_samples_number();
    const vector<Index> sample_indices = get_used_sample_indices();

    const Index raw_variables_number = get_raw_variables_number();
    const Index used_raw_variables_number = get_used_raw_variables_number();
    const vector<Index> used_raw_variables_indices = get_used_raw_variables_indices();

    vector<vector<Index>> return_values(2);

    return_values[0].resize(samples_number, 0);
    return_values[1].resize(used_raw_variables_number, 0);

    const vector<BoxPlot> box_plots = calculate_raw_variables_box_plots();

    Index variable_index = 0;
    Index used_variable_index = 0;

    #pragma omp parallel for

    for(Index i = 0; i < raw_variables_number; i++)
    {
        const RawVariable& raw_variable = raw_variables[i];

        if(raw_variable.use == VariableUse::None
        && raw_variable.type == RawVariableType::Categorical)
        {
            variable_index += raw_variable.get_categories_number();
            continue;
        }
        else if(raw_variable.use == VariableUse::None) // Numeric, Binary or DateTime
        {
            variable_index++;
            continue;
        }

        if(raw_variable.type == RawVariableType::Categorical)
        {
            variable_index += raw_variable.get_categories_number();
            used_variable_index++;
            continue;
        }
        else if(raw_variable.type == RawVariableType::Binary
             || raw_variable.type == RawVariableType::DateTime)
        {
            variable_index++;
            used_variable_index++;
            continue;
        }
        else // Numeric
        {
            const type interquartile_range = box_plots[i].third_quartile - box_plots[i].first_quartile;

            if(interquartile_range < numeric_limits<type>::epsilon())
            {
                variable_index++;
                used_variable_index++;
                continue;
            }

            Index raw_variables_outliers = 0;

            for(Index j = 0; j < samples_number; j++)
            {
                const Tensor<type, 1> sample = get_sample_data(sample_indices[Index(j)]);

                if(sample(variable_index) < box_plots[i].first_quartile - cleaning_parameter*interquartile_range
                || sample(variable_index) > box_plots[i].third_quartile + cleaning_parameter*interquartile_range)
                {
                    return_values[0][j] = 1;

                    raw_variables_outliers++;
                }
            }

            return_values[1][used_variable_index] = raw_variables_outliers;

            variable_index++;
            used_variable_index++;
        }
    }

    return return_values;
}


vector<vector<Index>> DataSet::replace_Tukey_outliers_with_NaN(const type& cleaning_parameter)
{
    const Index samples_number = get_used_samples_number();
    const vector<Index> sample_indices = get_used_sample_indices();

    const Index raw_variables_number = get_raw_variables_number();
    const Index used_raw_variables_number = get_used_raw_variables_number();
    const vector<Index> used_raw_variables_indices = get_used_raw_variables_indices();

    vector<vector<Index>> return_values(2);

    return_values[0].resize(samples_number, 0);
    return_values[1].resize(used_raw_variables_number, 0);

    const vector<BoxPlot> box_plots = calculate_raw_variables_box_plots();

    Index variable_index = 0;
    Index used_variable_index = 0;

    #pragma omp parallel for

    for(Index i = 0; i < raw_variables_number; i++)
    {
        const RawVariable& raw_variable = raw_variables[i];

        if(raw_variable.use == VariableUse::None
        && raw_variable.type == RawVariableType::Categorical)
        {
            variable_index += raw_variable.get_categories_number();
            continue;
        }
        else if(raw_variable.use == VariableUse::None) // Numeric, Binary or DateTime
        {
            variable_index++;
            continue;
        }

        if(raw_variable.type == RawVariableType::Categorical)
        {
            variable_index += raw_variable.get_categories_number();
            used_variable_index++;
            continue;
        }
        else if(raw_variable.type == RawVariableType::Binary
             || raw_variable.type == RawVariableType::DateTime)
        {
            variable_index++;
            used_variable_index++;
            continue;
        }
        else // Numeric
        {
            const type interquartile_range = box_plots[i].third_quartile - box_plots[i].first_quartile;

            if(interquartile_range < numeric_limits<type>::epsilon())
            {
                variable_index++;
                used_variable_index++;
                continue;
            }

            Index raw_variables_outliers = 0;

            for(Index j = 0; j < samples_number; j++)
            {
                const Tensor<type, 1> sample = get_sample_data(sample_indices[Index(j)]);

                if(sample[variable_index] < (box_plots[i].first_quartile - cleaning_parameter * interquartile_range)
                || sample[variable_index] > (box_plots[i].third_quartile + cleaning_parameter * interquartile_range))
                {
                    return_values[0][Index(j)] = 1;

                    raw_variables_outliers++;

                    data(sample_indices[Index(j)], variable_index) = numeric_limits<type>::quiet_NaN();
                }
            }

            return_values[1][used_variable_index] = raw_variables_outliers;

            variable_index++;
            used_variable_index++;
        }
    }

    return return_values;
}


void DataSet::unuse_Tukey_outliers(const type& cleaning_parameter)
{
    const vector<vector<Index>> outliers_indices = calculate_Tukey_outliers(cleaning_parameter);

    const vector<Index> outliers_samples = get_elements_greater_than(outliers_indices, 0);

    set_sample_uses(outliers_samples, DataSet::SampleUse::None);
}


void DataSet::set_data_rosenbrock()
{
    const Index samples_number = get_samples_number();
    const Index variables_number = get_variables_number();

    set_data_random();
    
    #pragma omp parallel for

    for(Index i = 0; i < samples_number; i++)
    {
        type rosenbrock(0);

        for(Index j = 0; j < variables_number-1; j++)
        {
            const type value = data(i, j);
            const type next_value = data(i, j+1);

            rosenbrock += (type(1) - value)*(type(1) - value) + type(100)*(next_value-value*value)*(next_value-value*value);
        }

        data(i, variables_number-1) = rosenbrock;
    }
}


void DataSet::set_data_classification()
{
    const Index samples_number = get_samples_number();
    const Index input_variables_number = get_variables_number(VariableUse::Input);
    const Index target_variables_number = get_variables_number(VariableUse::Target);

    data.setConstant(0.0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 1);

    #pragma omp parallel for
    for(Index i = 0; i < samples_number; i++)
    {
        for(Index j = 0; j < input_variables_number; j++)
            data(i, j) = get_random_type(-1, 1);

        target_variables_number == 1
            ? data(i, input_variables_number) = dist(gen)
            : data(i, input_variables_number + get_random_index(0, target_variables_number-1)) = 1;
    }

}


void DataSet::set_data_sum()
{
    set_random(data);
/*
    for(Index i = 0; i < samples_number; i++)
    {
        data(i,variables_number-1) = type(0);

        for(Index j = 0; j < variables_number-1; j++)
            data(i,variables_number-1) += data(i, j);
    }
*/
}


Tensor<Index, 1> DataSet::filter_data(const Tensor<type, 1>& minimums, 
                                      const Tensor<type, 1>& maximums)
{
    const vector<Index> used_variable_indices = get_used_variable_indices();

    const Index used_variables_number = used_variable_indices.size();

    const Index samples_number = get_samples_number();

    Tensor<type, 1> filtered_indices(samples_number);
    filtered_indices.setZero();

    const vector<Index> used_sample_indices = get_used_sample_indices();
    const Index used_samples_number = used_sample_indices.size();

    Index sample_index = 0;

    for(Index i = 0; i < used_variables_number; i++)
    {
        const Index variable_index = used_variable_indices[i];

        for(Index j = 0; j < used_samples_number; j++)
        {
            sample_index = used_sample_indices[j];

            if(get_sample_use(sample_index) == SampleUse::None 
            || isnan(data(sample_index, variable_index)))
                continue;

            const type value = data(sample_index, variable_index);

            if(abs(value - minimums(i)) <= NUMERIC_LIMITS_MIN
            || abs(value - maximums(i)) <= NUMERIC_LIMITS_MIN)
                continue;

            if(minimums(i) == maximums(i))
            {
                if(value != minimums(i))
                {
                    filtered_indices(sample_index) = type(1);
                    set_sample_use(sample_index, SampleUse::None);
                }
            }
            else if(value < minimums(i) 
                 || value > maximums(i))
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
        if(filtered_indices(i) > type(0.5))
            filtered_samples_indices(index++) = i;

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
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_variable_indices = get_used_variable_indices();
    const vector<Index> input_variable_indices = get_variable_indices(DataSet::VariableUse::Input);
    const vector<Index> target_variable_indices = get_variable_indices(DataSet::VariableUse::Target);

    const Tensor<type, 1> means = mean(data, used_sample_indices, used_variable_indices);

    const Index samples_number = used_sample_indices.size();
    const Index variables_number = used_variable_indices.size();
    const Index target_variables_number = target_variable_indices.size();

    Index current_variable;
    Index current_sample;

    #pragma omp parallel for schedule(dynamic)

    for(Index j = 0; j < variables_number - target_variables_number; j++)
    {
        current_variable = input_variable_indices[j];

        for(Index i = 0; i < samples_number; i++)
        {
            current_sample = used_sample_indices[i];

            if(isnan(data(current_sample, current_variable)))
                data(current_sample,current_variable) = means(j);
        }
    }

    #pragma omp parallel for schedule(dynamic)

    for(Index j = 0; j < target_variables_number; j++)
    {
        current_variable = target_variable_indices[j];

        for(Index i = 0; i < samples_number; i++)
        {
            current_sample = used_sample_indices[i];

            if(isnan(data(current_sample, current_variable)))
                set_sample_use(i, "None");
        }
    }
}


void DataSet::impute_missing_values_median()
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_variable_indices = get_used_variable_indices();
    const vector<Index> input_variable_indices = get_variable_indices(DataSet::VariableUse::Input);
    const vector<Index> target_variable_indices = get_variable_indices(DataSet::VariableUse::Target);

    const Tensor<type, 1> medians = median(data, used_sample_indices, used_variable_indices);

    const Index samples_number = used_sample_indices.size();
    const Index variables_number = used_variable_indices.size();
    const Index target_variables_number = target_variable_indices.size();

    #pragma omp parallel for schedule(dynamic)

    for(Index j = 0; j < variables_number - target_variables_number; j++)
    {
        const Index current_variable = input_variable_indices[j];

        for(Index i = 0; i < samples_number; i++)
        {
            const Index current_sample = used_sample_indices[i];

            if(isnan(data(current_sample, current_variable)))
                data(current_sample,current_variable) = medians(j);
        }
    }

    #pragma omp parallel for schedule(dynamic)

    for(Index j = 0; j < target_variables_number; j++)
    {
        const Index current_variable = target_variable_indices[j];

        for(Index i = 0; i < samples_number; i++)
        {
            const Index current_sample = used_sample_indices[i];

            if(isnan(data(current_sample, current_variable)))
                set_sample_use(i, "None");
        }
    }
}


void DataSet::impute_missing_values_interpolate()
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_variable_indices = get_used_variable_indices();
    const vector<Index> input_variable_indices = get_variable_indices(DataSet::VariableUse::Input);
    const vector<Index> target_variable_indices = get_variable_indices(DataSet::VariableUse::Target);

    const Index samples_number = used_sample_indices.size();
    const Index variables_number = used_variable_indices.size();
    const Index target_variables_number = target_variable_indices.size();

    Index current_variable;
    Index current_sample;

    #pragma omp parallel for schedule(dynamic)
    for(Index j = 0; j < variables_number - target_variables_number; j++)
    {
        current_variable = input_variable_indices[j];

        for(Index i = 0; i < samples_number; i++)
        {
            current_sample = used_sample_indices[i];

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
                    if (isnan(data(used_sample_indices[k], current_variable))) continue;

                    x1 = type(used_sample_indices[k]);
                    y1 = data(x1, current_variable);
                    break;
                }

                for(Index k = i + 1; k < samples_number; k++)
                {
                    if (isnan(data(used_sample_indices[k], current_variable))) continue;
                    
                    x2 = type(used_sample_indices[k]);
                    y2 = data(x2, current_variable);
                    break;                    
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
        current_variable = target_variable_indices[j];

        for(Index i = 0; i < samples_number; i++)
        {
            current_sample = used_sample_indices[i];

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


void DataSet::prepare_line(string& line) const
{
    decode(line);
    trim(line);
    erase(line, '"');
}


void DataSet::process_tokens(vector<string>& tokens)
{
    const Index raw_variables_number = tokens.size();

    //#pragma omp parallel for reduction(+:missing_values_number)

    for(Index i = 0; i < raw_variables_number; i++)
    {
        RawVariable& raw_variable = raw_variables[i];

        const string token = has_sample_ids ? tokens[i+1] : tokens[i];

        if(token.empty() || token == missing_values_label)
        {
            missing_values_number++;
            continue;
        }
        else if(is_numeric_string(token))
        {
            if(raw_variable.type != RawVariableType::Numeric)
                raw_variable.type = RawVariableType::Numeric;

            if(raw_variable.type == RawVariableType::Categorical)
                throw runtime_error("Error: Found number in categorical variable: " + raw_variable.name);
        }
        else if(is_date_time_string(token))
        {
            if(raw_variable.type != RawVariableType::DateTime)
                raw_variable.type = RawVariableType::DateTime;
        }
        else // is string
        {
            if(raw_variable.type != RawVariableType::Categorical)
                raw_variable.type = RawVariableType::Categorical;

            if(!contains(raw_variable.categories, token))
                raw_variable.categories.push_back(token);
        }
    }
}


void DataSet::read_csv()
{    
    if(data_path.empty())
        throw runtime_error("Data path is empty.\n");

    ifstream file(data_path);

    if(!file.is_open())
        throw runtime_error("Error: Cannot open file " + data_path.string() + "\n");

    const string separator_string = get_separator_string();
    
    const vector<string> positive_words = {"yes", "positive", "+", "true"};

    const vector<string> negative_words = {"no", "negative", "-", "false"};

    string line;

    vector<string> tokens;

    Index columns_number = 0;

    // Read first line

    while(getline(file, line))
    {
        prepare_line(line);

        if(line.empty()) continue;

//        check_separators(line);

        tokens = get_tokens(line, separator_string);

        columns_number = tokens.size();

        if(columns_number != 0) break;
    }

    const Index raw_variables_number = has_sample_ids
            ? columns_number - 1
            : columns_number;

    raw_variables.resize(raw_variables_number);

    Index samples_number = 0;

    if(has_header)
    {
        if(has_numbers(tokens))
            throw runtime_error("Error: Some header names are numeric: " + line + "\n");

        if(has_sample_ids)
            for(Index i = 0; i < raw_variables_number; i++)
                raw_variables[i].name = tokens[i+1];
        else
            set_raw_variable_names(tokens);
    }
    else
    {
        samples_number++;
        set_default_raw_variables_names();
    }

    // Rest of lines

    while(getline(file, line))
    {
        prepare_line(line);

        if(line.empty()) continue;

        //check_separators(line);

        tokens = get_tokens(line, separator_string);

        if(tokens.size() != columns_number)
            throw runtime_error("Sample " + to_string(samples_number+1) + ": "
                                "Tokens number is not equal to columns number.");

        process_tokens(tokens);

        samples_number++;
    }

    for(Index i = 0; i < raw_variables_number; i++)
        if(raw_variables[i].type == RawVariableType::Categorical
        && raw_variables[i].get_categories_number() == 2)
            raw_variables[i].type = RawVariableType::Binary;

    sample_uses.resize(samples_number);

    sample_ids.resize(samples_number);

    const Index variables_number = get_variables_number();

    const vector<vector<Index>> all_variable_indices = get_variable_indices();

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
        while(getline(file, line))
        {
            prepare_line(line);

            if(line.empty()) continue;
            break;
        }
    }

    Index sample_index = 0;

    while(getline(file, line))
    {
        prepare_line(line);

        if(line.empty()) continue;

        check_separators(line);

        tokens = get_tokens(line, separator_string);

        if(has_missing_values(tokens))
        {
            rows_missing_values_number ++;

            for(size_t i = 0; i < tokens.size(); i++)
            {
                if(tokens[i].empty() || tokens[i] == missing_values_label)
                {
                    missing_values_number++;

                    raw_variables_missing_values_number(i)++;
                }
            }
        }

        if(has_sample_ids)
            sample_ids[sample_index] = tokens[0];

        #pragma omp parallel for

        for(Index raw_variable_index = 0; raw_variable_index < raw_variables_number; raw_variable_index++)
        {
            const RawVariableType raw_variable_type = raw_variables[raw_variable_index].type;

            const string token = has_sample_ids
                ? tokens[raw_variable_index+1]
                : tokens[raw_variable_index];

            const vector<Index>& variable_indices = all_variable_indices[raw_variable_index];

            if(raw_variable_type == RawVariableType::Numeric)
            {
                (token.empty() || token == missing_values_label)
                    ? data(sample_index, variable_indices[0]) = NAN
                    : data(sample_index, variable_indices[0]) = stof(token);

            }
            else if(raw_variable_type == RawVariableType::DateTime)
            {              
                data(sample_index, raw_variable_index) = time_t(date_to_timestamp(tokens[raw_variable_index]));
            }
            else if(raw_variable_type == RawVariableType::Categorical)
            {
                const Index categories_number = raw_variables[raw_variable_index].get_categories_number();

                if(token.empty() || token == missing_values_label)
                {
                    for(Index category_index = 0; category_index < categories_number; category_index++)
                        data(sample_index, variable_indices[category_index]) = NAN;
                }
                else
                {
                    const vector<string> categories = raw_variables[raw_variable_index].categories;

                    for(Index category_index = 0; category_index < categories_number; category_index++)
                        if(token == categories[category_index])
                            data(sample_index, variable_indices[category_index]) = 1;
                }
            }
            else if(raw_variable_type == RawVariableType::Binary)
            {
                if(contains(positive_words, token) || contains(negative_words, token))
                {
                    data(sample_index, variable_indices[0]) = contains(positive_words, token)
                        ? 1
                        : 0;
                }
                else
                {
                    const vector<string> categories = raw_variables[raw_variable_index].categories;

                    if(token.empty() || token == missing_values_label)
                        data(sample_index, variable_indices[0]) = type(NAN);
                    else if(token == categories[0])
                        data(sample_index, variable_indices[0]) = 1;
                    else if(token == categories[1])
                        data(sample_index, variable_indices[0]) = 0;
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


vector<string> DataSet::get_default_raw_variables_names(const Index& raw_variables_number)
{
    vector<string> raw_variable_names(raw_variables_number);

    for(Index i = 0; i < raw_variables_number; i++)
        raw_variable_names[i] = "variable_" + to_string(i+1);

    return raw_variable_names;
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
    case Scaler::Logarithm:
        return "Logarithm";
    case Scaler::ImageMinMax:
        return "ImageMinMax";
    default:
        return "";
    }
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

    while(getline(file, line))
    {
        prepare_line(line);

        if(line.empty()) continue;

        check_separators(line);

        data_file_preview[lines_count] = get_tokens(line, separator_string);

        lines_count++;

        if(lines_count == lines_number) break;
    }

    file.close();

    // Check empty file

    if(data_file_preview[0].size() == 0)
        throw runtime_error("File " + data_path.string() + " is empty.\n");

    // Resize data file preview to original

    if(data_file_preview.size() > 4)
    {
        lines_number = has_header ? 4 : 3;

        vector<vector<string>> data_file_preview_copy(data_file_preview);

        data_file_preview.resize(lines_number);

        data_file_preview[0] = data_file_preview_copy[1];
        data_file_preview[1] = data_file_preview_copy[1];
        data_file_preview[2] = data_file_preview_copy[2];
        data_file_preview[lines_number - 2] = data_file_preview_copy[data_file_preview_copy.size()-2];
        data_file_preview[lines_number - 1] = data_file_preview_copy[data_file_preview_copy.size()-1];
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
        throw runtime_error("Error: Separator '" + separator_string + "' not found in line " + line + ".\n");

    if(separator == Separator::Space)
    {
        if(line.find(',') != string::npos)
            throw runtime_error("Error: Found comma (',') in data file " + data_path.string() + ", but separator is space (' ').");
        else if(line.find(';') != string::npos)
            throw runtime_error("Error: Found semicolon (';') in data file " + data_path.string() + ", but separator is space (' ').");
    }
    else if(separator == Separator::Tab)
    {
        if(line.find(',') != string::npos)
            throw runtime_error("Error: Found comma (',') in data file " + data_path.string() + ", but separator is tab ('   ').");
        else if(line.find(';') != string::npos)
            throw runtime_error("Error: Found semicolon (';') in data file " + data_path.string() + ", but separator is tab ('   ').");
    }
    else if(separator == Separator::Comma)
    {
        if(line.find(";") != string::npos)
            throw runtime_error("Error: Found semicolon (';') in data file " + data_path.string() + ", but separator is comma (',').");
    }
    else if(separator == Separator::Semicolon)
    {
        if(line.find(",") != string::npos)
            throw runtime_error("Error: Found comma (',') in data file " + data_path.string() + ", but separator is semicolon (';').");
    }
}


bool DataSet::has_binary_raw_variables() const
{
    return any_of(raw_variables.begin(), raw_variables.end(),
                  [](const RawVariable& raw_variable) { return raw_variable.type == RawVariableType::Binary; });
}


bool DataSet::has_categorical_raw_variables() const
{
    return any_of(raw_variables.begin(), raw_variables.end(),
                  [](const RawVariable& raw_variable) { return raw_variable.type == RawVariableType::Categorical; });
}


bool DataSet::has_binary_or_categorical_raw_variables() const
{
    for (const auto& raw_variable : raw_variables)
        if (raw_variable.type == RawVariableType::Binary || raw_variable.type == RawVariableType::Categorical)
            return true;

    return false;
}


bool DataSet::has_selection() const
{
    return get_samples_number(SampleUse::Selection) != 0;
}


bool DataSet::has_missing_values(const vector<string>& row) const
{
    for(size_t i = 0; i < row.size(); i++)
        if(row[i].empty() || row[i] == missing_values_label)
            return true;

    return false;
}


Tensor<Index, 1> DataSet::count_raw_variables_with_nan() const
{
    const Index raw_variables_number = get_raw_variables_number();
    const Index rows_number = get_samples_number();

    Tensor<Index, 1> raw_variables_with_nan(raw_variables_number);
    raw_variables_with_nan.setZero();

    #pragma omp parallel for

    for(Index raw_variable_index = 0; raw_variable_index < raw_variables_number; raw_variable_index++)
    {
        const Index current_variable_index = get_variable_indices(raw_variable_index)[0];

        Index counter = 0;

        for(Index row_index = 0; row_index < rows_number; row_index++)
            if(isnan(data(row_index, current_variable_index)))
                counter++;

        raw_variables_with_nan(raw_variable_index) = counter;
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

        if(has_nan) 
            rows_with_nan++;
    }

    return rows_with_nan;
}


Index DataSet::count_nan() const
{
    return count_NAN(data);
}


// void DataSet::set_missing_values_number()
// {
//     missing_values_number = count_nan();
// }


// void DataSet::set_raw_variables_missing_values_number()
// {
//     raw_variables_missing_values_number = count_raw_variables_with_nan();
// }


// void DataSet::set_samples_missing_values_number()
// {
//     rows_missing_values_number = count_rows_with_nan();
// }


void DataSet::fix_repeated_names()
{
    // Fix raw_variables names

    const Index raw_variables_number = raw_variables.size();

    map<string, Index> raw_variables_count_map;

    for(Index i = 0; i < raw_variables_number; i++)
    {
        auto result = raw_variables_count_map.insert(pair<string, Index>(raw_variables[i].name, 1));

        if(!result.second) 
            result.first->second++;
    }

    for(const auto & element : raw_variables_count_map)
    {
        if(element.second > 1)
        {
            const string repeated_name = element.first;

            Index repeated_index = 1;

            for(Index i = 0; i < raw_variables_number; i++)
                if(raw_variables[i].name == repeated_name)
                    raw_variables[i].name = raw_variables[i].name + "_" + to_string(repeated_index++);
        }
    }

    // Fix variables names

    if(has_categorical_raw_variables() || has_binary_raw_variables())
    {
        vector<string> variable_names = get_variable_names();

        const Index variables_number = variable_names.size();

        map<string, Index> variables_count_map;

        for(Index i = 0; i < variables_number; i++)
        {
            auto result = variables_count_map.insert(pair<string, Index>(variable_names[i], 1));

            if(!result.second) result.first->second++;
        }

        for(const auto & element : variables_count_map)
        {
            if(element.second > 1)
            {
                const string repeated_name = element.first;

                for(Index i = 0; i < variables_number; i++)
                {
                    if(variable_names[i] == repeated_name)
                    {
                        const Index raw_variable_index = get_raw_variable_index(i);

                        if(raw_variables[raw_variable_index].type != RawVariableType::Categorical)
                            continue;

                        variable_names[i] += "_" + raw_variables[raw_variable_index].name;
                    }
                }
            }
        }

        set_variable_names(variable_names);
    }
}


vector<vector<Index>> DataSet::split_samples(const vector<Index>& sample_indices, const Index& new_batch_size) const
{
    const Index samples_number = sample_indices.size();

    Index batch_size = new_batch_size;

    Index batches_number;

    if(samples_number < batch_size)
    {
        batches_number = 1;
        batch_size = samples_number;
    }
    else
    {
        batches_number = samples_number / batch_size;
    }

//    const Index batches_number = (samples_number + new_batch_size - 1) / new_batch_size; // Round up division

    vector<vector<Index>> batches(batches_number);

    Index count = 0;

    for (Index i = 0; i < batches_number; i++)
    {   
        batches[i].resize(batch_size);

        for (Index j = 0; j < batch_size; ++j)
            batches[i][j] = sample_indices[count++];
    }

    return batches;
}


bool DataSet::get_has_rows_labels() const
{
    return has_sample_ids;
}


void DataSet::decode(string& input_string) const
{
    switch(codification)
    {
    case DataSet::Codification::SHIFT_JIS:
//        input_string = sj2utf8(input_string);
        break;
    default:
        break;
    }
}


Tensor<type, 2> DataSet::read_input_csv(const filesystem::path& input_data_file_name,
                                        const string& separator_string,
                                        const string& missing_values_label,
                                        const bool& has_raw_variables_name,
                                        const bool& has_sample_ids) const
{
    const Index raw_variables_number = get_raw_variables_number();

    ifstream file(input_data_file_name);

    // Count samples number

    Index input_samples_count = 0;

    string line;
    Index line_number = 0;

    Index tokens_count;

    Index input_raw_variables_number = get_raw_variables_number(VariableUse::Input);

    if(model_type == ModelType::AutoAssociation)
        input_raw_variables_number = get_raw_variables_number() 
                                   - get_raw_variables_number(VariableUse::Target) 
                                   - get_raw_variables_number(VariableUse::None)/2;

    while(getline(file, line))
    {
        prepare_line(line);

        line_number++;

        if(line.empty()) continue;

        tokens_count = count_tokens(line, separator_string);

        if(tokens_count != input_raw_variables_number)
            throw runtime_error("Line " + to_string(line_number) + ": Size of tokens(" + to_string(tokens_count) + ") "
                                "is not equal to number of raw_variables(" + to_string(input_raw_variables_number) + ").\n");

        input_samples_count++;
    }

    file.close();

    const Index input_variables_number = get_variables_number(VariableUse::Input);

    if(has_raw_variables_name) input_samples_count--;

    Tensor<type, 2> input_data(input_samples_count, input_variables_number);
    input_data.setZero();

    file.open(input_data_file_name);

    //skip_header(file);

    // Read rest of the lines

    vector<string> tokens;

    line_number = 0;
    Index variable_index = 0;
    Index token_index = 0;
    bool is_ID = has_sample_ids;

    const bool is_float = is_same<type, float>::value;
    bool has_missing_values = false;

    while(getline(file, line))
    {
        prepare_line(line);

        if(line.empty()) continue;

        tokens = get_tokens(line, separator_string);

        variable_index = 0;
        token_index = 0;
        is_ID = has_sample_ids;

        for(Index i = 0; i < raw_variables_number; i++)
        {
            const RawVariable& raw_variable = raw_variables[i];

            if(is_ID)
            {
                is_ID = false;
                continue;
            }

            if(raw_variable.use == VariableUse::None)
            {
                token_index++;
                continue;
            }
            else if(raw_variable.use != VariableUse::Input)
            {
                continue;
            }

            const string& token = tokens[token_index];

            if(raw_variable.type == RawVariableType::Numeric)
            {
                if(token == missing_values_label || token.empty())
                    input_data(line_number, variable_index) = type(NAN);
                else if(is_float)
                    input_data(line_number, variable_index) = type(strtof(token.data(), nullptr));
                else
                    input_data(line_number, variable_index) = type(stof(token));

                variable_index++;
            }
            else if(raw_variable.type == RawVariableType::Binary)
            {
                if(token == missing_values_label)
                    input_data(line_number, variable_index) = type(NAN);
                else if(token == raw_variable.name
                     || (raw_variable.categories.size() > 0 && token == raw_variable.categories[0]))
                    input_data(line_number, variable_index) = type(1);

                variable_index++;
            }
            else if(raw_variable.type == RawVariableType::Categorical)
            {
                for(Index k = 0; k < raw_variable.get_categories_number(); k++)
                {
                    if(token == missing_values_label)
                        input_data(line_number, variable_index) = type(NAN);
                    else if(token == raw_variable.categories[k])
                        input_data(line_number, variable_index) = type(1);

                    variable_index++;
                }
            }
            else if(raw_variable.type == RawVariableType::DateTime)
            {
                if(token == missing_values_label || token.empty())
                    input_data(line_number, variable_index) = type(NAN);
                else
                    input_data(line_number, variable_index) = type(date_to_timestamp(token, gmt));

                variable_index++;
            }
            else if(raw_variable.type == RawVariableType::Constant)
            {
                if(token == missing_values_label || token.empty())
                    input_data(line_number, variable_index) = type(NAN);
                else if(is_float)
                    input_data(line_number, variable_index) = type(strtof(token.data(), nullptr));
                else
                    input_data(line_number, variable_index) = type(stof(token));

                variable_index++;
            }

            token_index++;
        }

        line_number++;
    }

    file.close();

    if(!has_missing_values)
        return input_data;

    // Scrub missing values

    const MissingValuesMethod missing_values_method = get_missing_values_method();

    const Index samples_number = input_data.dimension(0);
    const Index variables_number = input_data.dimension(1);

    if(missing_values_method == MissingValuesMethod::Unuse 
    || missing_values_method == MissingValuesMethod::Mean)
    {
        const Tensor<type, 1> means = mean(input_data);

        #pragma omp parallel for schedule(dynamic)

        for(Index j = 0; j < variables_number; j++)
            for(Index i = 0; i < samples_number; i++)
                if(isnan(input_data(i, j)))
                    input_data(i, j) = means(j);
    }
    else
    {
        const Tensor<type, 1> medians = median(input_data);

        #pragma omp parallel for schedule(dynamic)

        for(Index j = 0; j < variables_number; j++)
            for(Index i = 0; i < samples_number; i++)
                if(isnan(input_data(i, j)))
                    input_data(i, j) = medians(j);
    }

    return input_data;
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
