//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "dataset.h"
#include "statistics.h"
#include "correlations.h"
#include "tensors.h"
#include "strings_utilities.h"
#include "time_series_dataset.h"

namespace opennn
{

Dataset::Dataset(const Index& new_samples_number,
                 const dimensions& new_input_dimensions,
                 const dimensions& new_target_dimensions)
{
    set(new_samples_number, new_input_dimensions, new_target_dimensions);
}


Dataset::Dataset(const filesystem::path& data_path,
                 const string& separator,
                 const bool& has_header,
                 const bool& has_sample_ids,
                 const Codification& data_codification)
{
    set(data_path, separator, has_header, has_sample_ids, data_codification);
}


const bool& Dataset::get_display() const
{
    return display;
}


dimensions Dataset::get_input_dimensions() const
{
    return dimensions({get_variables_number("Input")});
}


dimensions Dataset::get_target_dimensions() const
{
    return dimensions({get_variables_number("Target")});
}


Dataset::RawVariable::RawVariable(const string& new_name,
                                  const string& new_raw_variable_use,
                                  const RawVariableType& new_type,
                                  const Scaler& new_scaler,
                                  const vector<string>& new_categories)
{
    set(new_name, new_raw_variable_use, new_type, new_scaler, new_categories);
}


void Dataset::RawVariable::set(const string& new_name,
                               const string& new_raw_variable_use,
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


void Dataset::RawVariable::set_scaler(const Scaler& new_scaler)
{
    scaler = new_scaler;
}


void Dataset::RawVariable::set_scaler(const string& new_scaler_string)
{
    const Scaler new_scaler = string_to_scaler(new_scaler_string);

    set_scaler(new_scaler);
}


void Dataset::RawVariable::set_use(const string& new_raw_variable_use)
{
    use = new_raw_variable_use;
}


void Dataset::RawVariable::set_type(const string& new_raw_variable_type)
{
    if (new_raw_variable_type == "Numeric")
        type = RawVariableType::Numeric;
    else if (new_raw_variable_type == "Binary")
        type = RawVariableType::Binary;
    else if (new_raw_variable_type == "Categorical")
        type = RawVariableType::Categorical;
    else if (new_raw_variable_type == "DateTime")
        type = RawVariableType::DateTime;
    else if (new_raw_variable_type == "Constant")
        type = RawVariableType::Constant;
    else
        throw runtime_error("Raw variable type is not valid (" + new_raw_variable_type + ").\n");
}


void Dataset::RawVariable::set_categories(const vector<string>& new_categories)
{
    categories = new_categories;
}


void Dataset::RawVariable::from_XML(const XMLDocument& document)
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


void Dataset::RawVariable::to_XML(XMLPrinter& printer) const
{
    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "Scaler", scaler_to_string(scaler));
    add_xml_element(printer, "Use", get_use());
    add_xml_element(printer, "Type", get_type_string());

    if (type == RawVariableType::Categorical || type == RawVariableType::Binary)
        add_xml_element(printer, "Categories", vector_to_string(categories,";"));
}


void Dataset::RawVariable::print() const
{
    cout << "Raw variable" << endl
         << "Name: " << name << endl
         << "Use: " << get_use() << endl
         << "Type: " << get_type_string() << endl
         << "Scaler: " << scaler_to_string(scaler) << endl;

    if (categories.size() != 0)
    {
        cout << "Categories: " << endl;
        print_vector(categories);
    }

    cout << endl;
}


string Dataset::RawVariable::get_use() const
{
    return use;
}


Index Dataset::RawVariable::get_categories_number() const
{
    return categories.size();
}


bool Dataset::is_sample_used(const Index& index) const
{
    return sample_uses[index] != "None";
}


Tensor<Index, 1> Dataset::get_sample_use_numbers() const
{
    Tensor<Index, 1> count(4);
    count.setZero();

    const Index samples_number = get_samples_number();

    for (Index i = 0; i < samples_number; i++)
    {
        const string& use = sample_uses[i];

        if (use == "Training")         count[0]++;
        else if (use == "Selection")   count[1]++;
        else if (use == "Testing")     count[2]++;
        else if (use == "None")        count[3]++;
        else throw runtime_error("Unknown sample use: " + use);
    }

    return count;
}


// Tensor<type, 1> Dataset::get_sample_use_percentages() const
// {
//     const Index samples_number = get_samples_number();

//     return (get_sample_use_numbers().cast<type>()) * (100 / type(samples_number));
// }


// string Dataset::get_sample_string(const Index& sample_index) const
// {
//     const Tensor<type, 1> sample = data.chip(sample_index, 0);

//     string sample_string;

//     const Index raw_variables_number = get_raw_variables_number();

//     Index variable_index = 0;

//     for (Index i = 0; i < raw_variables_number; i++)
//     {
//         const RawVariable& raw_variable = raw_variables[i];

//         switch (raw_variable.type)
//         {
//         case RawVariableType::Numeric:
//         case RawVariableType::DateTime:
//         case RawVariableType::Constant:
//             sample_string += isnan(data(sample_index, variable_index))
//                                  ? missing_values_label
//                                  : to_string(double(data(sample_index, variable_index)));

//             variable_index++;
//             break;

//         case RawVariableType::Binary:
//             sample_string += isnan(data(sample_index, variable_index))
//                 ? missing_values_label
//                 : raw_variable.categories[Index(data(sample_index, variable_index))];

//             variable_index++;
//             break;

//         case RawVariableType::Categorical:
//             if (isnan(data(sample_index, variable_index)))
//             {
//                 sample_string += missing_values_label;
//             }
//             else
//             {
//                 const Index categories_number = raw_variable.get_categories_number();

//                 for (Index j = 0; j < categories_number; j++)
//                 {
//                     if (abs(data(sample_index, variable_index + j) - type(1)) < NUMERIC_LIMITS_MIN)
//                     {
//                         sample_string += raw_variable.categories[j];
//                         break;
//                     }
//                 }

//                 variable_index += categories_number;
//             }
//             break;


//         default:
//             break;
//         }

//         if (i != raw_variables_number - 1)
//             sample_string += get_separator_string() + string(" ");
//     }

//     return sample_string;
// }


vector<Index> Dataset::get_sample_indices(const string& sample_use) const
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


vector<Index> Dataset::get_used_sample_indices() const
{
    const Index samples_number = get_samples_number();

    const Index used_samples_number = samples_number - get_samples_number("None");

    assert(used_samples_number >= 0);
    vector<Index> used_indices(used_samples_number);

    Index index = 0;

    for (Index i = 0; i < samples_number; i++)
        if (sample_uses[i] != "None")
            used_indices[index++] = i;

    return used_indices;
}


string Dataset::get_sample_use(const Index& index) const
{
    return sample_uses[index];
}


const vector<string>& Dataset::get_sample_uses() const
{
    return sample_uses;
}


vector<Index> Dataset::get_sample_uses_vector() const
{
    const Index samples_number = get_samples_number();

    vector<Index> sample_uses_vector(samples_number);

#pragma omp parallel for

    for (Index i = 0; i < samples_number; i++)    
    {
        const string& use = sample_uses[i];

        if(use == "Training")
            sample_uses_vector[i] = 0;
        else if(use == "Selection")
            sample_uses_vector[i] = 1;
        else if(use == "Testing")
            sample_uses_vector[i] = 2;
        else if(use == "None")
            sample_uses_vector[i] = 3;
        else
            throw runtime_error("Unknown sample use: " + use);
    }

    return sample_uses_vector;
}


vector<vector<Index>> Dataset::get_batches(const vector<Index>& sample_indices,
                                           const Index& batch_size,
                                           const bool& shuffle) const
{
    if (!shuffle) return split_samples(sample_indices, batch_size);

    random_device rng;
    mt19937 urng(rng());

    const Index samples_number = sample_indices.size();

    const Index batches_number = (samples_number + batch_size - 1) / batch_size;

    vector<vector<Index>> batches(batches_number);

    vector<Index> samples_copy(sample_indices);

    std::shuffle(samples_copy.begin(), samples_copy.end(), urng);

#pragma omp parallel for
    for (Index i = 0; i < batches_number; i++)
    {
        const Index start_index = i * batch_size;

        const Index end_index = min(start_index + batch_size, samples_number);

        batches[i].assign(samples_copy.begin() + start_index,
            samples_copy.begin() + end_index);
    }

    return batches;
}


Index Dataset::get_samples_number(const string& sample_use) const
{
    return count_if(sample_uses.begin(), sample_uses.end(),
        [&sample_use](const string& new_sample_use) { return new_sample_use == sample_use; });
}


Index Dataset::get_used_samples_number() const
{
    const Index samples_number = get_samples_number();
    const Index unused_samples_number = get_samples_number("None");

    return samples_number - unused_samples_number;
}


void Dataset::set_sample_uses(const string& sample_use)
{
    fill(sample_uses.begin(), sample_uses.end(), sample_use);
}


void Dataset::set_sample_use(const Index& index, const string& new_use)
{
    if (new_use == "Training")
        sample_uses[index] = "Training";
    else if (new_use == "Selection")
        sample_uses[index] = "Selection";
    else if (new_use == "Testing")
        sample_uses[index] = "Testing";
    else if (new_use == "None")
        sample_uses[index] = "None";
    else
        throw runtime_error("Unknown sample use: " + new_use + "\n");
}


void Dataset::set_sample_uses(const vector<string>& new_uses)
{
    const Index samples_number = new_uses.size();

    for (Index i = 0; i < samples_number; i++)
        if (new_uses[i] == "Training" || new_uses[i] == "0")
            sample_uses[i] = "Training";
        else if (new_uses[i] == "Selection" || new_uses[i] == "1")
            sample_uses[i] = "Selection";
        else if (new_uses[i] == "Testing" || new_uses[i] == "2")
            sample_uses[i] = "Testing";
        else if (new_uses[i] == "None" || new_uses[i] == "3")
            sample_uses[i] = "None";
        else
            throw runtime_error("Unknown sample use: " + new_uses[i] + ".\n");
}


void Dataset::set_sample_uses(const vector<Index>& indices, const string& sample_use)
{
    for (const auto& i : indices)
        set_sample_use(i, sample_use);
}


void Dataset::split_samples_random(const type& training_samples_ratio,
                                   const type& selection_samples_ratio,
                                   const type& testing_samples_ratio)
{
    random_device rng;
    mt19937 urng(rng());

    const Index used_samples_number = get_used_samples_number();

    if (used_samples_number == 0) return;

    const type total_ratio = training_samples_ratio + selection_samples_ratio + testing_samples_ratio;

    const Index selection_samples_number = Index((selection_samples_ratio * used_samples_number) / total_ratio);
    const Index testing_samples_number = Index((testing_samples_ratio * used_samples_number) / total_ratio);

    const Index training_samples_number = used_samples_number - selection_samples_number - testing_samples_number;

    const Index sum_samples_number = training_samples_number + selection_samples_number + testing_samples_number;

    if (sum_samples_number != used_samples_number)
        throw runtime_error("Sum of numbers of training, selection and testing samples is not equal to number of used samples.\n");

    const Index samples_number = get_samples_number();

    vector<Index> indices(samples_number);
    iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.data(), indices.data() + indices.size(), urng);

    auto assign_sample_use = [this, &indices](string use, Index count, Index& i)
        {
            Index assigned_count = 0;

            while (assigned_count < count)
            {
                const Index index = indices[i++];

                if (sample_uses[index] != "None")
                {
                    sample_uses[index] = use;
                    assigned_count++;
                }
            }
        };

    Index index = 0;

    assign_sample_use("Training", training_samples_number, index);
    assign_sample_use("Selection", selection_samples_number, index);
    assign_sample_use("Testing", testing_samples_number, index);
}


void Dataset::split_samples_sequential(const type& training_samples_ratio,
                                       const type& selection_samples_ratio,
                                       const type& testing_samples_ratio)
{
    const Index used_samples_number = get_used_samples_number();

    if (used_samples_number == 0) return;

    const type total_ratio = training_samples_ratio + selection_samples_ratio + testing_samples_ratio;

    const Index selection_samples_number = Index(selection_samples_ratio * type(used_samples_number) / type(total_ratio));
    const Index testing_samples_number = Index(testing_samples_ratio * type(used_samples_number) / type(total_ratio));
    const Index training_samples_number = used_samples_number - selection_samples_number - testing_samples_number;

    const Index sum_samples_number = training_samples_number + selection_samples_number + testing_samples_number;

    if (sum_samples_number != used_samples_number)
        throw runtime_error("Sum of numbers of training, selection and testing samples is not equal to number of used samples.\n");

    auto set_sample_uses = [this](string use, Index count, Index& i)
        {
            Index current_count = 0;

            while (current_count < count)
            {
                if (sample_uses[i] != "None")
                {
                    sample_uses[i] = use;
                    current_count++;
                }

                i++;
            }
        };

    Index index = 0;

    set_sample_uses("Training", training_samples_number, index);
    set_sample_uses("Selection", selection_samples_number, index);
    set_sample_uses("Testing", testing_samples_number, index);
}


void Dataset::set_raw_variables(const vector<RawVariable>& new_raw_variables)
{
    raw_variables = new_raw_variables;
}


void Dataset::set_default_raw_variables_uses()
{
    const Index raw_variables_number = raw_variables.size();

    if (raw_variables_number == 0)
        return;

    if (raw_variables_number == 1)
    {
        raw_variables[0].set_use("None");
        return;
    }

    set_variable_uses("Input");

    for (Index i = raw_variables_number - 1; i >= 0; i--)
    {
        RawVariable& raw_variable = raw_variables[i];

        if (raw_variable.type == RawVariableType::Constant ||
            raw_variable.type == RawVariableType::DateTime)
        {
            raw_variable.set_use("None");
        }
        //else if (model_type != ModelType::Classification ||
        //    raw_variable.type == RawVariableType::Binary ||
        //    raw_variable.type == RawVariableType::Categorical))
        {
            raw_variable.set_use("Target");
            break;
        }
    }
}


void Dataset::set_default_raw_variables_uses_forecasting()
{
    const Index raw_variables_number = raw_variables.size();

    bool target = false;
    bool timeRawVariable = false;

    if (raw_variables_number == 0)
        return;

    if (raw_variables_number == 1)
    {
        raw_variables[0].set_use("None");
        return;
    }

    set_variable_uses("Input");

    for (Index i = raw_variables.size() - 1; i >= 0; i--)
    {
        if (raw_variables[i].type == RawVariableType::DateTime && !timeRawVariable)
        {
            raw_variables[i].set_use("Time");

            timeRawVariable = true;
            continue;
        }

        if (raw_variables[i].type == RawVariableType::Constant)
        {
            raw_variables[i].set_use("None");
            continue;
        }

        if (!target)
        {
            raw_variables[i].set_use("Target");

            target = true;

            continue;
        }
    }
}

void Dataset::set_default_raw_variable_names()
{
    const Index raw_variables_number = raw_variables.size();

    for (Index i = 0; i < raw_variables_number; i++)
        raw_variables[i].name = "variable_" + to_string(1 + i);
}


vector<string> Dataset::get_variable_names() const
{
    const Index variables_number = get_variables_number();

    vector<string> variable_names(variables_number);

    Index index = 0;

    for (const Dataset::RawVariable& raw_variable : raw_variables)
        if (raw_variable.type == RawVariableType::Categorical)
            for (size_t j = 0; j < raw_variable.categories.size(); j++)
                variable_names[index++] = raw_variable.categories[j];
        else
            variable_names[index++] = raw_variable.name;

    return variable_names;
}


vector<string> Dataset::get_variable_names(const string& variable_use) const
{
    const Index variables_number = get_variables_number(variable_use);

    vector<string> variable_names(variables_number);

    Index index = 0;

    for (const Dataset::RawVariable& raw_variable : raw_variables)
    {
        if (raw_variable.use != variable_use)
            continue;

        if (raw_variable.type == RawVariableType::Categorical)
            for (Index j = 0; j < raw_variable.get_categories_number(); j++)
                variable_names[index++] = raw_variable.categories[j];
        else
            variable_names[index++] = raw_variable.name;
    }

    return variable_names;
}


dimensions Dataset::get_dimensions(const string& variable_use) const
{
    if (variable_use == "Input")
        return input_dimensions;
    else if (variable_use == "Target")
        return target_dimensions;
    // else if (variable_use == "Decoder")
    //     return decoder_dimensions;
    else
        throw invalid_argument("get_dimensions: Invalid variable use string: " + variable_use);
}


void Dataset::set_dimensions(const string& variable_use, const dimensions& new_dimensions)
{
    if (variable_use == "Input")
        input_dimensions = new_dimensions;
    else if (variable_use == "Target")
        target_dimensions = new_dimensions;
    // else if (variable_use == "Decoder")
    //     decoder_dimensions = new_dimensions;
    else
        throw invalid_argument("set_dimensions: Invalid variable use string: " + variable_use);
}


Index Dataset::get_used_variables_number() const
{
    const Index variables_number = get_variables_number();

    const Index unused_variables_number = get_variables_number("None");

    return variables_number - unused_variables_number;
}


vector<Index> Dataset::get_variable_indices(const string& variable_use) const
{
    const Index this_variables_number = get_variables_number(variable_use);
    vector<Index> this_variable_indices(this_variables_number);

    Index variable_index = 0;
    Index this_variable_index = 0;

    for (const Dataset::RawVariable& raw_variable : raw_variables)
    {
        if (raw_variable.use != variable_use)
        {
            raw_variable.type == RawVariableType::Categorical
                ? variable_index += raw_variable.get_categories_number()
                : variable_index++;

            continue;
        }

        if (raw_variable.type == RawVariableType::Categorical)
        {
            const Index categories_number = raw_variable.get_categories_number();

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


vector<Index> Dataset::get_raw_variable_indices(const string& variable_use) const
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


vector<Index> Dataset::get_used_raw_variables_indices() const
{
    const Index raw_variables_number = get_raw_variables_number();

    const Index used_raw_variables_number = get_used_raw_variables_number();

    vector<Index> used_indices(used_raw_variables_number);

    Index index = 0;

    for (Index i = 0; i < raw_variables_number; i++)
        if (raw_variables[i].use == "Input"
        || raw_variables[i].use == "Target"
        || raw_variables[i].use == "Time")
            used_indices[index++] = i;

    return used_indices;
}


vector<Scaler> Dataset::get_variable_scalers(const string& variable_use) const
{
    const Index input_raw_variables_number = get_raw_variables_number(variable_use);
    const Index input_variables_number = get_variables_number(variable_use);

    const vector<RawVariable> input_raw_variables = get_raw_variables(variable_use);

    vector<Scaler> input_variable_scalers(input_variables_number);

    Index index = 0;

    for (Index i = 0; i < input_raw_variables_number; i++)
        if (input_raw_variables[i].type == RawVariableType::Categorical)
            for (Index j = 0; j < input_raw_variables[i].get_categories_number(); j++)
                input_variable_scalers[index++] = input_raw_variables[i].scaler;
        else
            input_variable_scalers[index++] = input_raw_variables[i].scaler;

    return input_variable_scalers;
}


vector<string> Dataset::get_raw_variable_names() const
{
    const Index raw_variables_number = get_raw_variables_number();

    vector<string> raw_variable_names(raw_variables_number);

    for (Index i = 0; i < raw_variables_number; i++)
        raw_variable_names[i] = raw_variables[i].name;

    return raw_variable_names;
}


vector<string> Dataset::get_raw_variable_names(const string& variable_use) const
{
    const Index count = get_raw_variables_number(variable_use);

    vector<string> names(count);

    Index index = 0;

    for (const Dataset::RawVariable& raw_variable : raw_variables)
    {
        if (raw_variable.use != variable_use)
            continue;

        names[index++] = raw_variable.name;
    }

    return names;
}


Index Dataset::get_raw_variables_number(const string& variable_use) const
{
    Index count = 0;

    for (const Dataset::RawVariable& raw_variable : raw_variables)
        if (raw_variable.use == variable_use)
            count++;

    return count;
}


Index Dataset::get_used_raw_variables_number() const
{
    Index used_raw_variables_number = 0;

    for (const Dataset::RawVariable& raw_variable : raw_variables)
        if (raw_variable.use != "None")
            used_raw_variables_number++;

    return used_raw_variables_number;
}


const vector<Dataset::RawVariable>& Dataset::get_raw_variables() const
{
    return raw_variables;
}


vector<Dataset::RawVariable> Dataset::get_raw_variables(const string& variable_use) const
{
    const Index count = get_raw_variables_number(variable_use);

    vector<RawVariable> this_raw_variables(count);
    Index index = 0;

    for (const Dataset::RawVariable& raw_variable : raw_variables)
        if (raw_variable.use == variable_use)
            this_raw_variables[index++] = raw_variable;

    return this_raw_variables;
}


Index Dataset::get_variables_number() const
{
    Index count = 0;

    for (const Dataset::RawVariable& raw_variable : raw_variables)
        count += raw_variable.type == RawVariableType::Categorical
        ? raw_variable.get_categories_number()
        : 1;

    return count;
}


Index Dataset::get_variables_number(const string& variable_use) const
{
    Index count = 0;

    for (const Dataset::RawVariable& raw_variable : raw_variables)
    {
        if (raw_variable.use != variable_use)
            continue;

        count += (raw_variable.type == RawVariableType::Categorical)
            ? raw_variable.get_categories_number()
            : 1;
    }

    return count;
}


vector<Index> Dataset::get_used_variable_indices() const
{
    const Index used_variables_number = get_used_variables_number();
    vector<Index> used_variable_indices(used_variables_number);

    Index variable_index = 0;
    Index used_variable_index = 0;

    for (const Dataset::RawVariable& raw_variable : raw_variables)
    {
        const Index categories_number = raw_variable.get_categories_number();

        if (raw_variable.use == "None")
        {
            variable_index += (raw_variable.type == RawVariableType::Categorical)
                                  ? raw_variable.get_categories_number()
                                  : 1;
            continue;
        }

        if(raw_variable.type == RawVariableType::Categorical)
            for (Index j = 0; j < categories_number; j++)
                used_variable_indices[used_variable_index++] = variable_index++;
        else
            used_variable_indices[used_variable_index++] = variable_index++;
    }

    return used_variable_indices;
}


void Dataset::set_raw_variable_uses(const vector<string>& new_raw_variables_uses)
{
    const size_t new_raw_variables_uses_size = new_raw_variables_uses.size();

    if (new_raw_variables_uses_size != raw_variables.size())
        throw runtime_error("Size of raw_variables uses (" + to_string(new_raw_variables_uses_size) + ") "
            "must be equal to raw_variables size (" + to_string(raw_variables.size()) + "). \n");

    for (size_t i = 0; i < new_raw_variables_uses.size(); i++)
        raw_variables[i].set_use(new_raw_variables_uses[i]);
}


void Dataset::set_raw_variables(const string& variable_use)
{
    const Index raw_variables_number = get_raw_variables_number();

    for (Index i = 0; i < raw_variables_number; i++)
        set_raw_variable_use(i, variable_use);
}


void Dataset::set_raw_variable_indices(const vector<Index>& input_raw_variables,
    const vector<Index>& target_raw_variables)
{
    set_raw_variables("None");

    for (size_t i = 0; i < input_raw_variables.size(); i++)
        set_raw_variable_use(input_raw_variables[i], "Input");

    for (size_t i = 0; i < target_raw_variables.size(); i++)
        set_raw_variable_use(target_raw_variables[i], "Target");

    const Index input_dimensions = get_variables_number("Input");
    const Index target_dimensions = get_variables_number("Target");

    set_dimensions("Input", {input_dimensions});
    set_dimensions("Target", {target_dimensions});
}


void Dataset::set_input_raw_variables_unused()
{
    const Index raw_variables_number = get_raw_variables_number();

    for (Index i = 0; i < raw_variables_number; i++)
        if (raw_variables[i].use == "Input")
            set_raw_variable_use(i, "None");
}


void Dataset::set_raw_variable_use(const Index& index, const string& new_use)
{
    raw_variables[index].use = new_use;
}


void Dataset::set_raw_variable_use(const string& name, const string& new_use)
{
    const Index index = get_raw_variable_index(name);

    set_raw_variable_use(index, new_use);
}


void Dataset::set_raw_variable_type(const Index& index, const RawVariableType& new_type)
{
    raw_variables[index].type = new_type;
}


void Dataset::set_raw_variable_type(const string& name, const RawVariableType& new_type)
{
    const Index index = get_raw_variable_index(name);

    set_raw_variable_type(index, new_type);
}


void Dataset::set_raw_variable_types(const RawVariableType& new_type)
{
    for (auto& raw_variable : raw_variables)
        raw_variable.type = new_type;
}


void Dataset::set_variable_names(const vector<string>& new_variables_names)
{
    Index index = 0;

    for (Dataset::RawVariable& raw_variable : raw_variables)
        if (raw_variable.type == RawVariableType::Categorical)
            for (Index j = 0; j < raw_variable.get_categories_number(); j++)
                raw_variable.categories[j] = new_variables_names[index++];
        else
            raw_variable.name = new_variables_names[index++];
}


void Dataset::set_raw_variable_names(const vector<string>& new_names)
{
    const Index new_names_size = new_names.size();
    const Index raw_variables_number = get_raw_variables_number();

    if (new_names_size != raw_variables_number)
        throw runtime_error("Size of names (" + to_string(new_names.size()) + ") "
            "is not equal to raw_variables number (" + to_string(raw_variables_number) + ").\n");

    for (Index i = 0; i < raw_variables_number; i++)
        raw_variables[i].name = get_trimmed(new_names[i]);
}


void Dataset::set_variable_uses(const string& variable_use)
{
    const Index raw_variables_number = get_raw_variables_number();

    for (Index i = 0; i < raw_variables_number; i++)
    {
        if (raw_variables[i].type == RawVariableType::Constant)
            continue;

        raw_variables[i].set_use(variable_use);
    }
}


void Dataset::set_raw_variables_number(const Index& new_raw_variables_number)
{
    raw_variables.resize(new_raw_variables_number);
}


void Dataset::set_raw_variable_scalers(const Scaler& scalers)
{
    for (Dataset::RawVariable& raw_variable : raw_variables)
        raw_variable.scaler = scalers;
}


void Dataset::set_raw_variable_scalers(const vector<Scaler>& new_scalers)
{
    const size_t raw_variables_number = get_raw_variables_number();

    if (new_scalers.size() != raw_variables_number)
        throw runtime_error("Size of raw_variable scalers(" + to_string(new_scalers.size()) + ") "
            "has to be the same as raw_variables numbers(" + to_string(raw_variables_number) + ").\n");

    for (size_t i = 0; i < raw_variables_number; i++)
        raw_variables[i].scaler = new_scalers[i];
}


void Dataset::set_binary_raw_variables()
{
    Index variable_index = 0;

    const Index raw_variables_number = get_raw_variables_number();

    for (Index raw_variable_index = 0; raw_variable_index < raw_variables_number; raw_variable_index++)
    {
        RawVariable& raw_variable = raw_variables[raw_variable_index];

        if (raw_variable.type == RawVariableType::Numeric)
        {
            const Tensor<type, 1> data_column = data.chip(variable_index, 1);

            if (is_binary(data_column))
            {
                raw_variable.type = RawVariableType::Binary;
                raw_variable.categories = { "0","1" };
            }

            variable_index++;
        }
        else if (raw_variable.type == RawVariableType::Categorical)
            variable_index += raw_variable.get_categories_number();
        else if (raw_variable.type == RawVariableType::DateTime
            || raw_variable.type == RawVariableType::Constant
            || raw_variable.type == RawVariableType::Binary)
            variable_index++;
    }
}


void Dataset::unuse_constant_raw_variables()
{
    Index variable_index = 0;

    const Index raw_variables_number = get_raw_variables_number();

    for (Index raw_variable_index = 0; raw_variable_index < raw_variables_number; raw_variable_index++)
    {
        RawVariable& raw_variable = raw_variables[raw_variable_index];

        if (raw_variable.type == RawVariableType::Numeric)
        {
            const Tensor<type, 1> data_column = data.chip(variable_index, 1);

            if (is_constant(data_column))
                raw_variable.set(raw_variable.name, "None", RawVariableType::Constant);

            variable_index++;
        }
        else if (raw_variable.type == RawVariableType::DateTime || raw_variable.type == RawVariableType::Constant)
        {
            variable_index++;
        }
        else if (raw_variable.type == RawVariableType::Binary)
        {
            if (raw_variable.get_categories_number() == 1)
                raw_variable.set(raw_variable.name, "None", RawVariableType::Constant);

            variable_index++;
        }
        else if (raw_variable.type == RawVariableType::Categorical)
        {
            if (raw_variable.get_categories_number() == 1)
                raw_variable.set(raw_variable.name, "None", RawVariableType::Constant);

            variable_index += raw_variable.get_categories_number();
        }
    }
}


const Tensor<type, 2>& Dataset::get_data() const
{
    return data;
}


Tensor<type, 2>* Dataset::get_data_p()
{
    return &data;
}


Dataset::MissingValuesMethod Dataset::get_missing_values_method() const
{
    return missing_values_method;
}


string Dataset::get_missing_values_method_string() const
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


const filesystem::path& Dataset::get_data_path() const
{
    return data_path;
}


const bool& Dataset::get_header_line() const
{
    return has_header;
}


const bool& Dataset::get_has_sample_ids() const
{
    return has_sample_ids;
}


vector<string> Dataset::get_sample_ids() const
{
    return sample_ids;
}


const Dataset::Separator& Dataset::get_separator() const
{
    return separator;
}


string Dataset::get_separator_string() const
{
    switch (separator)
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


string Dataset::get_separator_name() const
{
    switch (separator)
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


const Dataset::Codification& Dataset::get_codification() const
{
    return codification;
}


const string Dataset::get_codification_string() const
{
    switch (codification)
    {
    case Codification::UTF8:
        return "UTF-8";
    case Codification::SHIFT_JIS:
        return "SHIFT_JIS";
    default:
        return "UTF-8";
    }
}


const string& Dataset::get_missing_values_label() const
{
    return missing_values_label;
}


Tensor<type, 2> Dataset::get_data_samples(const string& sample_use) const
{
    const vector<Index> variable_indices = get_used_variable_indices();

    const vector<Index> sample_indices = get_sample_indices(sample_use);

    Tensor<type, 2> this_data(sample_indices.size(), variable_indices.size());

    fill_tensor_data(data, sample_indices, variable_indices, this_data.data());

    return this_data;
}


Tensor<type, 2> Dataset::get_data_variables(const string& variable_use) const
{
    const Index samples_number = get_samples_number();

    vector<Index> indices(samples_number);
    iota(indices.begin(), indices.end(), 0);

    const vector<Index> variable_indices = get_variable_indices(variable_use);

    Tensor<type, 2> this_data(indices.size(), variable_indices.size());

    fill_tensor_data(data, indices, variable_indices, this_data.data());

    return this_data;
}


Tensor<type, 2> Dataset::get_data(const string& sample_use, const string& variable_use) const
{
    const vector<Index> sample_indices = get_sample_indices(sample_use);

    const vector<Index> variable_indices = get_variable_indices(variable_use);

    Tensor<type, 2> this_data(sample_indices.size(), variable_indices.size());

    fill_tensor_data(data, sample_indices, variable_indices, this_data.data());

    return this_data;
}


Tensor<type, 2> Dataset::get_data_from_indices(const vector<Index>& sample_indices, const vector<Index>& variable_indices) const
{
    Tensor<type, 2> this_data(sample_indices.size(), variable_indices.size());

    fill_tensor_data(data, sample_indices, variable_indices, this_data.data());

    return this_data;
}


Tensor<type, 1> Dataset::get_sample_data(const Index& index) const
{
    return data.chip(index, 0);
}


Tensor<type, 1> Dataset::get_sample_data(const Index& sample_index, const vector<Index>& variable_indices) const
{
    const Index variables_number = variable_indices.size();

    Tensor<type, 1 > row(variables_number);

#pragma omp parallel for
    for (Index i = 0; i < variables_number; i++)
        row(i) = data(sample_index, variable_indices[i]);

    return row;
}


Tensor<type, 2> Dataset::get_sample_input_data(const Index& sample_index) const
{
    const Index input_variables_number = get_variables_number("Input");

    const vector<Index> input_variable_indices = get_variable_indices("Input");

    Tensor<type, 2> inputs(1, input_variables_number);

    for (Index i = 0; i < input_variables_number; i++)
        inputs(0, i) = data(sample_index, input_variable_indices[i]);

    return inputs;
}


Tensor<type, 2> Dataset::get_sample_target_data(const Index& sample_index) const
{
    const vector<Index> target_variable_indices = get_variable_indices("Target");

    Tensor<type, 2> sample_target_data(1, target_variable_indices.size());

    fill_tensor_data(data, vector<Index>(sample_index), target_variable_indices, sample_target_data.data());

    return sample_target_data;
}


Index Dataset::get_raw_variable_index(const string& column_name) const
{
    const Index raw_variables_number = get_raw_variables_number();

    for (Index i = 0; i < raw_variables_number; i++)
        if (raw_variables[i].name == column_name)
            return i;

    throw runtime_error("Cannot find " + column_name + "\n");
}


Index Dataset::get_raw_variable_index(const Index& variable_index) const
{
    const Index raw_variables_number = get_raw_variables_number();

    Index total_variables_number = 0;

    for (Index i = 0; i < raw_variables_number; i++)
    {
        total_variables_number += (raw_variables[i].type == RawVariableType::Categorical)
            ? raw_variables[i].get_categories_number()
            : 1;

        if (variable_index + 1 <= total_variables_number)
            return i;
    }

    throw runtime_error("Cannot find variable index: " + to_string(variable_index) + ".\n");
}


vector<vector<Index>> Dataset::get_variable_indices() const
{
    const Index raw_variables_number = get_raw_variables_number();

    vector<vector<Index>> indices(raw_variables_number);

    for (Index i = 0; i < raw_variables_number; i++)
        indices[i] = get_variable_indices(i);

    return indices;
}


vector<Index> Dataset::get_variable_indices(const Index& raw_variable_index) const
{
    Index index = 0;

    for (Index i = 0; i < raw_variable_index; i++)
        index += (raw_variables[i].type == RawVariableType::Categorical)
        ? raw_variables[i].categories.size()
        : 1;

    const RawVariable& raw_variable = raw_variables[raw_variable_index];

    if (raw_variable.type == RawVariableType::Categorical)
    {
        vector<Index> indices(raw_variable.categories.size());

        for (size_t j = 0; j < raw_variable.categories.size(); j++)
            indices[j] = index + j;

        return indices;
    }

    return vector<Index>(1, index);
}


Tensor<type, 2> Dataset::get_raw_variable_data(const Index& raw_variable_index) const
{
    Index raw_variables_number = 1;
    const Index rows_number = data.dimension(0);

    if (raw_variables[raw_variable_index].type == RawVariableType::Categorical)
        raw_variables_number = raw_variables[raw_variable_index].get_categories_number();

    const array<Index, 2> offsets = { 0, get_variable_indices(raw_variable_index)[0] };
    const array<Index, 2> extents = { rows_number, raw_variables_number };

    return data.slice(offsets, extents);
}


Tensor<type, 1> Dataset::get_sample(const Index& sample_index) const
{
    if (sample_index >= data.dimension(0))
        throw runtime_error("Sample index out of bounds.");

    return data.chip(sample_index, 0);
}


string Dataset::get_sample_category(const Index& sample_index, const Index& column_index_start) const
{
    if (raw_variables[column_index_start].type != RawVariableType::Categorical)
        throw runtime_error("The specified raw_variable is not of categorical type.");

    for (size_t raw_variable_index = column_index_start; raw_variable_index < raw_variables.size(); raw_variable_index++)
        if (data(sample_index, raw_variable_index) == 1)
            return raw_variables[column_index_start].categories[raw_variable_index - column_index_start];

    throw runtime_error("Sample does not have a valid one-hot encoded category.");
}


Tensor<type, 2> Dataset::get_raw_variable_data(const Index& raw_variable_index, const vector<Index>& row_indices) const
{
    Tensor<type, 2> raw_variable_data(row_indices.size(), get_variable_indices(raw_variable_index).size());

    fill_tensor_data(data, row_indices, get_variable_indices(raw_variable_index), raw_variable_data.data());

    return raw_variable_data;
}


Tensor<type, 2> Dataset::get_raw_variable_data(const string& column_name) const
{
    const Index raw_variable_index = get_raw_variable_index(column_name);

    return get_raw_variable_data(raw_variable_index);
}


const vector<vector<string>>& Dataset::get_data_file_preview() const
{
    return data_file_preview;
}


void Dataset::set(const filesystem::path& new_data_path,
    const string& new_separator,
    const bool& new_has_header,
    const bool& new_has_ids,
    const Dataset::Codification& new_codification)
{
    set_default();

    set_data_path(new_data_path);

    set_separator_string(new_separator);

    set_has_header(new_has_header);

    set_has_ids(new_has_ids);

    set_codification(new_codification);

    read_csv();

    set_default_raw_variables_scalers();

    set_default_raw_variables_uses();

    missing_values_method = MissingValuesMethod::Mean;
    if(has_nan())
        scrub_missing_values();

    input_dimensions = { get_variables_number("Input") };
    target_dimensions = { get_variables_number("Target") };
}


void Dataset::set(const Index& new_samples_number,
                  const dimensions& new_input_dimensions,
                  const dimensions& new_target_dimensions)
{
    if (new_samples_number == 0
        || new_input_dimensions.empty()
        || new_target_dimensions.empty())
        return;

    input_dimensions = new_input_dimensions;

    const Index new_inputs_number = accumulate(new_input_dimensions.begin(),
        new_input_dimensions.end(),
        1,
        multiplies<Index>());

    Index new_targets_number = accumulate(new_target_dimensions.begin(),
        new_target_dimensions.end(),
        1,
        multiplies<Index>());

    new_targets_number = (new_targets_number == 2) ? 1 : new_targets_number;

    target_dimensions = { new_targets_number };

    const Index new_variables_number = new_inputs_number + new_targets_number;

    data.resize(new_samples_number, new_variables_number);

    raw_variables.resize(new_variables_number);

    set_default();

    for (Index i = 0; i < new_variables_number; i++)
    {
        RawVariable& raw_variable = raw_variables[i];

        raw_variable.type = RawVariableType::Numeric;
        raw_variable.name = "variable_" + to_string(i + 1);

        raw_variable.use = (i < new_inputs_number)
            ? "Input"
            : "Target";
    }

    sample_uses.resize(new_samples_number);

    split_samples_random();
}


void Dataset::set(const filesystem::path& file_name)
{
    load(file_name);
}


void Dataset::set_display(const bool& new_display)
{
    display = new_display;
}


void Dataset::set_default()
{
    const unsigned int threads_number = thread::hardware_concurrency();
    thread_pool = make_unique<ThreadPool>(threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);

    has_header = false;

    has_sample_ids = false;

    separator = Separator::Semicolon;

    missing_values_label = "NA";

    set_default_raw_variable_names();
}


void Dataset::set_data(const Tensor<type, 2>& new_data)
{
    if (new_data.dimension(0) != get_samples_number())
        throw runtime_error("Rows number is not equal to samples number");

    if (new_data.dimension(1) != get_variables_number())
        throw runtime_error("Columns number is not equal to variables number");

    data = new_data;
}


void Dataset::set_data_path(const filesystem::path& new_data_path)
{
    data_path = new_data_path;
}


void Dataset::set_has_header(const bool& new_has_header)
{
    has_header = new_has_header;
}


void Dataset::set_has_ids(const bool& new_has_ids)
{
    has_sample_ids = new_has_ids;
}


void Dataset::set_separator(const Separator& new_separator)
{
    separator = new_separator;
}


void Dataset::set_separator_string(const string& new_separator_string)
{
    if (new_separator_string == " ")
        separator = Separator::Space;
    else if (new_separator_string == "\t")
        separator = Separator::Tab;
    else if (new_separator_string == ",")
        separator = Separator::Comma;
    else if (new_separator_string == ";")
        separator = Separator::Semicolon;
    else
        throw runtime_error("Unknown separator: " + new_separator_string);
}


void Dataset::set_separator_name(const string& new_separator_name)
{
    if (new_separator_name == "Space")
        separator = Separator::Space;
    else if (new_separator_name == "Tab")
        separator = Separator::Tab;
    else if (new_separator_name == "Comma")
        separator = Separator::Comma;
    else if (new_separator_name == "Semicolon")
        separator = Separator::Semicolon;
    else
        throw runtime_error("Unknown separator: " + new_separator_name + ".\n");
}


void Dataset::set_codification(const Dataset::Codification& new_codification)
{
    codification = new_codification;
}


void Dataset::set_codification(const string& new_codification_string)
{
    if (new_codification_string == "UTF-8")
        codification = Codification::UTF8;
    else if (new_codification_string == "SHIFT_JIS")
        codification = Codification::SHIFT_JIS;
    else
        throw runtime_error("Unknown codification: " + new_codification_string + ".\n");
}


void Dataset::set_missing_values_label(const string& new_missing_values_label)
{
    missing_values_label = new_missing_values_label;
}


void Dataset::set_missing_values_method(const Dataset::MissingValuesMethod& new_missing_values_method)
{
    missing_values_method = new_missing_values_method;
}


void Dataset::set_missing_values_method(const string& new_missing_values_method)
{
    if (new_missing_values_method == "Unuse")
        missing_values_method = MissingValuesMethod::Unuse;
    else if (new_missing_values_method == "Mean")
        missing_values_method = MissingValuesMethod::Mean;
    else if (new_missing_values_method == "Median")
        missing_values_method = MissingValuesMethod::Median;
    else if (new_missing_values_method == "Interpolation")
        missing_values_method = MissingValuesMethod::Interpolation;
    else
        throw runtime_error("Unknown method type.\n");
}


void Dataset::set_threads_number(const int& new_threads_number)
{
    thread_pool.reset();
    thread_pool_device.reset();

    thread_pool = make_unique<ThreadPool>(new_threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), new_threads_number);
}


// Tensor<Index, 1> Dataset::unuse_repeated_samples()
// {
//     const Index samples_number = get_samples_number();

//     Tensor<Index, 1> repeated_samples;

//     Tensor<type, 1> sample_i;
//     Tensor<type, 1> sample_j;

//     for (Index i = 0; i < samples_number; i++)
//     {
//         sample_i = get_sample_data(i);

//         for (Index j = Index(i + 1); j < samples_number; j++)
//         {
//             sample_j = get_sample_data(j);

//             if (get_sample_use(j) != "None"
//                 && equal(sample_i.data(), sample_i.data() + sample_i.size(), sample_j.data()))
//             {
//                 set_sample_use(j, "None");

//                 push_back(repeated_samples, j);
//             }
//         }
//     }

//     return repeated_samples;
// }


vector<string> Dataset::unuse_uncorrelated_raw_variables(const type& minimum_correlation)
{
    vector<string> unused_raw_variables;

    const Tensor<Correlation, 2> correlations = calculate_input_target_raw_variable_pearson_correlations();

    const Index input_raw_variables_number = get_raw_variables_number("Input");
    const Index target_raw_variables_number = get_raw_variables_number("Target");

    const vector<Index> input_raw_variable_indices = get_raw_variable_indices("Input");

    for (Index i = 0; i < input_raw_variables_number; i++)
    {
        const Index input_raw_variable_index = input_raw_variable_indices[i];

        for (Index j = 0; j < target_raw_variables_number; j++)
        {
            if (!isnan(correlations(i, j).r)
                && abs(correlations(i, j).r) < minimum_correlation
                && raw_variables[input_raw_variable_index].use != "None")
            {
                raw_variables[input_raw_variable_index].set_use("None");

                unused_raw_variables.push_back(raw_variables[input_raw_variable_index].name);
            }
        }
    }

    return unused_raw_variables;
}


vector<string> Dataset::unuse_collinear_raw_variables(const type& maximum_correlation)
{
    const Tensor<Correlation, 2> correlations = calculate_input_raw_variable_pearson_correlations();
    const vector<Index> input_raw_variable_indices = get_raw_variable_indices("Input");
    const Index input_raw_variables_number = input_raw_variable_indices.size();

    vector<Index> high_corr_counts(input_raw_variables_number, 0);
    vector<type> mean_abs_corr(input_raw_variables_number, 0.0);
    vector<bool> to_be_removed(input_raw_variables_number, false);

    for (Index i = 0; i < input_raw_variables_number; ++i)
    {
        type sum_of_abs_corr = 0.0;
        for (Index j = 0; j < input_raw_variables_number; ++j)
        {
            if (i == j) continue;

            const type abs_r = abs(correlations(i, j).r);
            if (!isnan(abs_r))
            {
                if (abs_r >= maximum_correlation)
                    high_corr_counts[i]++;

                sum_of_abs_corr += abs_r;
            }
        }
        if (input_raw_variables_number > 1)
            mean_abs_corr[i] = sum_of_abs_corr / (input_raw_variables_number - 1);
    }

    for (Index i = 0; i < input_raw_variables_number; ++i)
    {
        for (Index j = i + 1; j < input_raw_variables_number; ++j)
        {

            if (to_be_removed[i] || to_be_removed[j])
                continue;

            if (!isnan(correlations(i, j).r) && abs(correlations(i, j).r) >= maximum_correlation)
            {
                Index index_to_flag_for_removal;

                if (high_corr_counts[i] > high_corr_counts[j])
                    index_to_flag_for_removal = i;
                else if (high_corr_counts[j] > high_corr_counts[i])
                    index_to_flag_for_removal = j;
                else
                {
                    if (mean_abs_corr[i] >= mean_abs_corr[j])
                        index_to_flag_for_removal = i;
                    else
                        index_to_flag_for_removal = j;
                }
                to_be_removed[index_to_flag_for_removal] = true;
            }
        }
    }

    vector<string> unused_raw_variables;
    for (Index i = 0; i < input_raw_variables_number; ++i)
    {
        if (to_be_removed[i])
        {
            const Index global_raw_index = input_raw_variable_indices[i];

            if (raw_variables[global_raw_index].use != "None")
            {
                raw_variables[global_raw_index].set_use("None");
                unused_raw_variables.push_back(raw_variables[global_raw_index].name);
            }
        }
    }

    return unused_raw_variables;
}


vector<Histogram> Dataset::calculate_raw_variable_distributions(const Index& bins_number) const
{
    const Index used_raw_variables_number = get_used_raw_variables_number();
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const Index used_samples_number = used_sample_indices.size();

    vector<Histogram> histograms(used_raw_variables_number);

    Index variable_index = 0;
    Index used_raw_variable_index = 0;

    for (const Dataset::RawVariable& raw_variable : raw_variables)
    {
        if (raw_variable.use == "None")
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


vector<BoxPlot> Dataset::calculate_raw_variables_box_plots() const
{
    const Index raw_variables_number = get_raw_variables_number();

    const vector<Index> used_sample_indices = get_used_sample_indices();

    vector<BoxPlot> box_plots(raw_variables_number);

    Index variable_index = 0;

    for (Index i = 0; i < raw_variables_number; i++)
    {
        const RawVariable& raw_variable = raw_variables[i];

        if (raw_variable.type == RawVariableType::Numeric
            || raw_variable.type == RawVariableType::Binary)
        {
            if (raw_variable.use != "None")
            {
                box_plots[i] = box_plot(data.chip(variable_index, 1), used_sample_indices);

                //used_raw_variable_index++;
            }

            variable_index++;
        }
        else if (raw_variable.type == RawVariableType::Categorical)
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


Index Dataset::calculate_used_negatives(const Index& target_index)
{
    Index negatives = 0;

    const vector<Index> used_indices = get_used_sample_indices();

    const Index used_samples_number = used_indices.size();

    for (Index i = 0; i < used_samples_number; i++)
    {
        const Index training_index = used_indices[i];

        if (isnan(data(training_index, target_index)))
            continue;

        if (abs(data(training_index, target_index)) < NUMERIC_LIMITS_MIN)
            negatives++;
        else if (abs(data(training_index, target_index) - type(1)) > NUMERIC_LIMITS_MIN
            || data(training_index, target_index) < type(0))
            throw runtime_error("Training sample is neither a positive nor a negative: "
                + to_string(training_index) + "-" + to_string(target_index) + "-" + to_string(data(training_index, target_index)));
    }

    return negatives;
}


Index Dataset::calculate_negatives(const Index& target_index, const string& sample_use) const
{
    Index negatives = 0;
    const vector<Index> indices = get_sample_indices(sample_use);
    const Index samples_number = get_samples_number(sample_use);

    for (Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = indices[i];
        type sample_value = data(sample_index, target_index);

        if (sample_use == "Testing")
        {
            if (sample_value < type(NUMERIC_LIMITS_MIN))
                negatives++;
        }
        else
        {
            if (abs(sample_value) < type(NUMERIC_LIMITS_MIN))
                negatives++;
            else
            {
                type threshold = (sample_use == "Training") ? type(1.0e-3) : type(NUMERIC_LIMITS_MIN);
                if (abs(sample_value - type(1)) > threshold)
                    throw runtime_error("Sample is neither a positive nor a negative: "
                        + to_string(sample_value) + "-" + to_string(target_index) + "-" + to_string(data(sample_value, target_index)));
            }
        }
    }

    return negatives;
}


vector<Descriptives> Dataset::calculate_variable_descriptives() const
{
    return descriptives(data);
}


vector<Descriptives> Dataset::calculate_used_variable_descriptives() const
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_variable_indices = get_used_variable_indices();

    return descriptives(data, used_sample_indices, used_variable_indices);
}


vector<Descriptives> Dataset::calculate_raw_variable_descriptives_positive_samples() const
{
    const Index target_index = get_variable_indices("Target")[0];

    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> input_variable_indices = get_variable_indices("Input");

    const Index samples_number = used_sample_indices.size();

    Index positive_samples_number = 0;

    for (Index i = 0; i < samples_number; i++)
        if (abs(data(used_sample_indices[i], target_index) - type(1)) < NUMERIC_LIMITS_MIN)
            positive_samples_number++;

    vector<Index> positive_used_sample_indices(positive_samples_number);
    Index positive_sample_index = 0;

    for (Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = used_sample_indices[i];

        if (abs(data(sample_index, target_index) - type(1)) < NUMERIC_LIMITS_MIN)
            positive_used_sample_indices[positive_sample_index++] = sample_index;
    }

    return descriptives(data, positive_used_sample_indices, input_variable_indices);
}


vector<Descriptives> Dataset::calculate_raw_variable_descriptives_negative_samples() const
{
    const Index target_index = get_variable_indices("Target")[0];

    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> input_variable_indices = get_variable_indices("Input");

    const Index samples_number = used_sample_indices.size();

    Index negative_samples_number = 0;

    for (Index i = 0; i < samples_number; i++)
        if (data(used_sample_indices[i], target_index) < NUMERIC_LIMITS_MIN)
            negative_samples_number++;

    vector<Index> negative_used_sample_indices(negative_samples_number);
    Index negative_sample_index = 0;

    for (Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = used_sample_indices[i];

        if (data(sample_index, target_index) < NUMERIC_LIMITS_MIN)
            negative_used_sample_indices[negative_sample_index++] = sample_index;
    }

    return descriptives(data, negative_used_sample_indices, input_variable_indices);
}


vector<Descriptives> Dataset::calculate_raw_variable_descriptives_categories(const Index& class_index) const
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> input_variable_indices = get_variable_indices("Input");

    const Index samples_number = used_sample_indices.size();

    // Count used class samples

    Index class_samples_number = 0;

    for (Index i = 0; i < samples_number; i++)
        if (abs(data(used_sample_indices[i], class_index) - type(1)) < NUMERIC_LIMITS_MIN)
            class_samples_number++;

    vector<Index> class_used_sample_indices(class_samples_number, 0);

    Index class_sample_index = 0;

    for (Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = used_sample_indices[i];

        if (abs(data(sample_index, class_index) - type(1)) < NUMERIC_LIMITS_MIN)
            class_used_sample_indices[class_sample_index++] = sample_index;
    }

    return descriptives(data, class_used_sample_indices, input_variable_indices);
}


vector<Descriptives> Dataset::calculate_variable_descriptives(const string& variable_use) const
{
    const vector<Index> used_sample_indices = get_used_sample_indices();

    const vector<Index> input_variable_indices = get_variable_indices(variable_use);

    return descriptives(data, used_sample_indices, input_variable_indices);
}


vector<Descriptives> Dataset::calculate_testing_target_variable_descriptives() const
{
    const vector<Index> testing_indices = get_sample_indices("Testing");

    const vector<Index> target_variable_indices = get_variable_indices("Target");

    return descriptives(data, testing_indices, target_variable_indices);
}


// Tensor<type, 1> Dataset::calculate_used_variables_minimums() const
// {
//     return column_minimums(data, get_used_sample_indices(), get_used_variable_indices());
// }


Tensor<type, 1> Dataset::calculate_means(const string& sample_use,
    const string& variable_use) const
{
    const vector<Index> sample_indices = get_sample_indices(sample_use);

    const vector<Index> variable_indices = get_variable_indices(variable_use);

    return mean(data, sample_indices, variable_indices);
}


Index Dataset::get_gmt() const
{
    return gmt;
}


void Dataset::set_gmt(const Index& new_gmt)
{
    gmt = new_gmt;
}


Tensor<Correlation, 2> Dataset::calculate_input_target_raw_variable_pearson_correlations() const
{
    cout << "Calculating pearson correlations..." << endl;

    const Index input_raw_variables_number = get_raw_variables_number("Input");
    const Index target_raw_variables_number = get_raw_variables_number("Target");

    const vector<Index> input_raw_variable_indices = get_raw_variable_indices("Input");
    const vector<Index> target_raw_variable_indices = get_raw_variable_indices("Target");

    const vector<Index> used_sample_indices = get_used_sample_indices();

    Tensor<Correlation, 2> correlations(input_raw_variables_number, target_raw_variables_number);

    //#pragma omp parallel for

    for (Index i = 0; i < input_raw_variables_number; i++)
    {
        cout << "Correlation " << i + 1 << " of " << input_raw_variables_number << endl;
        const Index input_raw_variable_index = input_raw_variable_indices[i];

        const Tensor<type, 2> input_raw_variable_data
            = get_raw_variable_data(input_raw_variable_index, used_sample_indices);

        for (Index j = 0; j < target_raw_variables_number; j++)
        {
            const Index target_raw_variable_index = target_raw_variable_indices[j];

            const Tensor<type, 2> target_raw_variable_data
                = get_raw_variable_data(target_raw_variable_index, used_sample_indices);

            correlations(i, j) = correlation(thread_pool_device.get(), input_raw_variable_data, target_raw_variable_data);
        }
    }

    return correlations;
}


Tensor<Correlation, 2> Dataset::calculate_input_target_raw_variable_spearman_correlations() const
{
    cout << "Calculating spearman correlations..." << endl;

    const Index input_raw_variables_number = get_raw_variables_number("Input");
    const Index target_raw_variables_number = get_raw_variables_number("Target");

    const vector<Index> input_raw_variable_indices = get_raw_variable_indices("Input");
    const vector<Index> target_raw_variable_indices = get_raw_variable_indices("Target");

    const vector<Index> used_sample_indices = get_used_sample_indices();

    Tensor<Correlation, 2> correlations(input_raw_variables_number, target_raw_variables_number);

    for (Index i = 0; i < input_raw_variables_number; i++)
    {
        cout << "Correlation " << i + 1 << " of " << input_raw_variables_number << endl;
        const Index input_index = input_raw_variable_indices[i];

        const Tensor<type, 2> input_raw_variable_data = get_raw_variable_data(input_index, used_sample_indices);

        for (Index j = 0; j < target_raw_variables_number; j++)
        {
            const Index target_index = target_raw_variable_indices[j];

            const Tensor<type, 2> target_raw_variable_data = get_raw_variable_data(target_index, used_sample_indices);

            correlations(i, j) = correlation_spearman(thread_pool_device.get(), input_raw_variable_data, target_raw_variable_data);
        }
    }

    return correlations;
}


bool Dataset::has_nan() const
{
    const Index rows_number = data.dimension(0);

    for (Index i = 0; i < rows_number; i++)
        if (sample_uses[i] != "None")
            if (has_nan_row(i))
                return true;

    return false;
}


bool Dataset::has_nan_row(const Index& row_index) const
{
    const Index variables_number = get_variables_number();

    for (Index j = 0; j < variables_number; j++)
        if (isnan(data(row_index, j)))
            return true;

    return false;
}


void Dataset::print_missing_values_information() const
{
    const Index missing_raw_variables_number = count_raw_variables_with_nan();
    const Index samples_with_missing_values = count_rows_with_nan();

    cout << "Missing values number: " << missing_values_number << " (" << missing_values_number * 100 / data.size() << "%)" << endl
        << "Raw variables with missing values: " << missing_raw_variables_number
        << " (" << missing_raw_variables_number * 100 / data.dimension(1) << "%)" << endl
        << "Samples with missing values: "
        << samples_with_missing_values << " (" << samples_with_missing_values * 100 / data.dimension(0) << "%)" << endl;
}


void Dataset::print_input_target_raw_variables_correlations() const
{
    const Index inputs_number = get_variables_number("Input");
    const Index targets_number = get_raw_variables_number("Target");

    const vector<string> input_names = get_raw_variable_names("Input");
    const vector<string> targets_name = get_raw_variable_names("Target");

    const Tensor<Correlation, 2> correlations = calculate_input_target_raw_variable_pearson_correlations();

    for (Index j = 0; j < targets_number; j++)
        for (Index i = 0; i < inputs_number; i++)
            cout << targets_name[j] << " - " << input_names[i] << ": " << correlations(i, j).r << endl;
}


void Dataset::print_top_input_target_raw_variables_correlations() const
{
    const Index inputs_number = get_raw_variables_number("Input");
    const Index targets_number = get_raw_variables_number("Target");

    const vector<string> input_names = get_variable_names("Input");
    const vector<string> targets_name = get_variable_names("Target");

    const Tensor<type, 2> correlations = get_correlation_values(calculate_input_target_raw_variable_pearson_correlations());

    Tensor<type, 1> target_correlations(inputs_number);

    Tensor<string, 2> top_correlations(inputs_number, 2);

    map<type, string> top_correlation;

    for (Index i = 0; i < inputs_number; i++)
        for (Index j = 0; j < targets_number; j++)
            top_correlation.insert(pair<type, string>(correlations(i, j), input_names[i] + " - " + targets_name[j]));

    map<type, string>::iterator it;

    for (it = top_correlation.begin(); it != top_correlation.end(); it++)
        cout << "Correlation: " << (*it).first << "  between  " << (*it).second << endl;
}


Tensor<Correlation, 2> Dataset::calculate_input_raw_variable_pearson_correlations() const
{
    // list to return
    cout << "Calculating pearson inputs correlations..." << endl;

    const vector<Index> input_raw_variable_indices = get_raw_variable_indices("Input");

    const Index input_raw_variables_number = input_raw_variable_indices.size();

    Tensor<Correlation, 2> correlations_pearson(input_raw_variables_number, input_raw_variables_number);

    for (Index i = 0; i < input_raw_variables_number; i++)
    {
        cout << "Correlation " << i + 1<< " of " << input_raw_variables_number << endl;

        const Index current_input_index_i = input_raw_variable_indices[i];

        const Tensor<type, 2> input_i = get_raw_variable_data(current_input_index_i);

        if (is_constant(input_i)) continue;

        correlations_pearson(i, i).set_perfect();
        correlations_pearson(i, i).method = Correlation::Method::Pearson;

        for (Index j = i + 1; j < input_raw_variables_number; j++)
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


Tensor<Correlation, 2> Dataset::calculate_input_raw_variable_spearman_correlations() const
{
    cout << "Calculating spearman inputs correlations..." << endl;

    const vector<Index> input_raw_variable_indices = get_raw_variable_indices("Input");

    const Index input_raw_variables_number = get_raw_variables_number("Input");

    Tensor<Correlation, 2> correlations_spearman(input_raw_variables_number, input_raw_variables_number);

    for (Index i = 0; i < input_raw_variables_number; i++)
    {
        cout << "Correlation " << i + 1 << " of " << input_raw_variables_number << endl;

        const Index input_raw_variable_index_i = input_raw_variable_indices[i];

        const Tensor<type, 2> input_i = get_raw_variable_data(input_raw_variable_index_i);

        if (is_constant(input_i)) continue;

        correlations_spearman(i, i).set_perfect();
        correlations_spearman(i, i).method = Correlation::Method::Spearman;

        for (Index j = i + 1; j < input_raw_variables_number; j++)
        {
            const Index input_raw_variable_index_j = input_raw_variable_indices[j];

            const Tensor<type, 2> input_j = get_raw_variable_data(input_raw_variable_index_j);

            correlations_spearman(i, j) = correlation_spearman(thread_pool_device.get(), input_i, input_j);

            if (correlations_spearman(i, j).r > type(1) - NUMERIC_LIMITS_MIN)
                correlations_spearman(i, j).r = type(1);

            correlations_spearman(j, i) = correlations_spearman(i, j);
        }
    }

    return correlations_spearman;
}

void Dataset::print_inputs_correlations() const
{
    const Tensor<type, 2> inputs_correlations
        = get_correlation_values(calculate_input_raw_variable_pearson_correlations());

    cout << inputs_correlations << endl;
}


void Dataset::print_data_file_preview() const
{
    const Index size = data_file_preview.size();

    for (Index i = 0; i < size; i++)
    {
        for (size_t j = 0; j < data_file_preview[i].size(); j++)
            cout << data_file_preview[i][j] << " ";

        cout << endl;
    }
}


void Dataset::print_top_inputs_correlations() const
{
    const Index variables_number = get_variables_number("Input");

    const vector<string> variables_name = get_variable_names("Input");

    const Tensor<type, 2> variables_correlations = get_correlation_values(calculate_input_raw_variable_pearson_correlations());

    const Index correlations_number = variables_number * (variables_number - 1) / 2;

    Tensor<string, 2> top_correlations(correlations_number, 3);

    map<type, string> top_correlation;

    for (Index i = 0; i < variables_number; i++)
    {
        for (Index j = i; j < variables_number; j++)
        {
            if (i == j) continue;

            top_correlation.insert(pair<type, string>(variables_correlations(i, j), variables_name[i] + " - " + variables_name[j]));
        }
    }

    map<type, string> ::iterator it;

    for (it = top_correlation.begin(); it != top_correlation.end(); it++)
        cout << "Correlation: " << (*it).first << "  between  " << (*it).second << endl;
}


void Dataset::set_default_raw_variables_scalers()
{
    for (Dataset::RawVariable& raw_variable : raw_variables)
        raw_variable.scaler = (raw_variable.type == RawVariableType::Numeric)
        ? Scaler::MeanStandardDeviation
        : Scaler::MinimumMaximum;
}


vector<Descriptives> Dataset::scale_data()
{
    const Index variables_number = get_variables_number();

    const vector<Descriptives> variable_descriptives = calculate_variable_descriptives();

    Index raw_variable_index;

    for (Index i = 0; i < variables_number; i++)
    {
        raw_variable_index = get_raw_variable_index(i);

        switch (raw_variables[raw_variable_index].scaler)
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


vector<Descriptives> Dataset::scale_variables(const string& variable_use)
{
    const Index input_variables_number = get_variables_number(variable_use);

    const vector<Index> input_variable_indices = get_variable_indices(variable_use);
    const vector<Scaler> input_variable_scalers = get_variable_scalers(variable_use);

    const vector<Descriptives> input_variable_descriptives = calculate_variable_descriptives(variable_use);

    for (Index i = 0; i < input_variables_number; i++)
    {
        switch (input_variable_scalers[i])
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


void Dataset::unscale_variables(const string& variable_use,
    const vector<Descriptives>& input_variable_descriptives)
{
    const Index input_variables_number = get_variables_number(variable_use);

    const vector<Index> input_variable_indices = get_variable_indices(variable_use);

    const vector<Scaler> input_variable_scalers = get_variable_scalers("Input");

    for (Index i = 0; i < input_variables_number; i++)
    {
        switch (input_variable_scalers[i])
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


void Dataset::set_data_constant(const type& new_value)
{
    const vector<Index> input_indices = get_variable_indices("Input");

    const Index samples_number = get_samples_number();

    for (Index i = 0; i < samples_number; ++i)
        for (Index index : input_indices)
            data(i, index) = new_value;
}


void Dataset::set_data_random()
{
    set_random(data);
}


void Dataset::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Dataset");

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

    const string separator_string = get_separator_string();

    if (has_sample_ids)
        add_xml_element(printer, "SamplesId", vector_to_string(sample_ids, separator_string));

    add_xml_element(printer, "SampleUses", vector_to_string(get_sample_uses_vector()));
    printer.CloseElement();

    printer.OpenElement("MissingValues");
    add_xml_element(printer, "MissingValuesNumber", to_string(missing_values_number));

    if (missing_values_number > 0)
    {
        add_xml_element(printer, "MissingValuesMethod", get_missing_values_method_string());
        add_xml_element(printer, "RawVariablesMissingValuesNumber", tensor_to_string<Index, 1>(raw_variables_missing_values_number));
        add_xml_element(printer, "RowsMissingValuesNumber", to_string(rows_missing_values_number));
    }

    printer.CloseElement();

    printer.OpenElement("PreviewData");

    add_xml_element(printer, "PreviewSize", to_string(data_file_preview.size()));

    vector<string> vector_data_file_preview = convert_string_vector(data_file_preview,",");
    for(size_t i = 0; i < data_file_preview.size(); i++){
        printer.OpenElement("Row");
        printer.PushAttribute("Item", to_string(i + 1).c_str());
        printer.PushText(vector_data_file_preview[i].data());
        printer.CloseElement();
    }

    printer.CloseElement();

    add_xml_element(printer, "Display", to_string(display));

    printer.CloseElement();
}


void Dataset::from_XML(const XMLDocument& data_set_document)
{
    const XMLElement* data_set_element = data_set_document.FirstChildElement("Dataset");
    if (!data_set_element)
        throw runtime_error("Data set element is nullptr.\n");

    // Data Source
    const XMLElement* data_source_element = data_set_element->FirstChildElement("DataSource");

    if (!data_source_element)
        throw runtime_error("Data source element is nullptr.\n");

    set_data_path(read_xml_string(data_source_element, "Path"));
    set_separator_name(read_xml_string(data_source_element, "Separator"));
    set_has_header(read_xml_bool(data_source_element, "HasHeader"));
    set_has_ids(read_xml_bool(data_source_element, "HasSamplesId"));
    set_missing_values_label(read_xml_string(data_source_element, "MissingValuesLabel"));
    set_codification(read_xml_string(data_source_element, "Codification"));

    // Raw Variables
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

        if (raw_variable_element->Attribute("Item") != to_string(i + 1))
            throw runtime_error("Raw variable item number (" + to_string(i + 1) + ") does not match (" + raw_variable_element->Attribute("Item") + ").\n");

        raw_variable.name = read_xml_string(raw_variable_element, "Name");
        raw_variable.set_scaler(read_xml_string(raw_variable_element, "Scaler"));
        raw_variable.set_use(read_xml_string(raw_variable_element, "Use"));
        raw_variable.set_type(read_xml_string(raw_variable_element, "Type"));

        if (raw_variable.type == RawVariableType::Categorical || raw_variable.type == RawVariableType::Binary)
        {
            const XMLElement* categories_element = raw_variable_element->FirstChildElement("Categories");

            if (categories_element)
                raw_variable.categories = get_tokens(read_xml_string(raw_variable_element, "Categories"), ";");
            else if (raw_variable.type == RawVariableType::Binary)
                raw_variable.categories = { "0", "1" };
            else
                throw runtime_error("Categorical RawVariable Element is nullptr: Categories");
        }
    }

    // Samples
    const XMLElement* samples_element = data_set_element->FirstChildElement("Samples");

    if (!samples_element)
        throw runtime_error("Samples element is nullptr.\n");

    const Index samples_number = read_xml_index(samples_element, "SamplesNumber");

    if (has_sample_ids)
    {
        const string separator_string = get_separator_string();
        sample_ids = get_tokens(read_xml_string(samples_element, "SamplesId"), separator_string);
    }

    if (raw_variables.size() != 0)
    {
        const vector<vector<Index>> all_variable_indices = get_variable_indices();

        data.resize(samples_number, all_variable_indices[all_variable_indices.size() - 1][all_variable_indices[all_variable_indices.size() - 1].size() - 1] + 1);
        data.setZero();

        sample_uses.resize(samples_number);
        set_sample_uses(get_tokens(read_xml_string(samples_element, "SampleUses"), " "));
    }
    else
        data.resize(0, 0);

    // Missing values
    const XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");

    if (!missing_values_element)
        throw runtime_error("Missing values element is nullptr.\n");

    missing_values_number = read_xml_index(missing_values_element, "MissingValuesNumber");

    if (missing_values_number > 0)
    {
        set_missing_values_method(read_xml_string(missing_values_element, "MissingValuesMethod"));

        raw_variables_missing_values_number.resize(get_tokens(read_xml_string(missing_values_element, "RawVariablesMissingValuesNumber"), " ").size());

        for (Index i = 0; i < raw_variables_missing_values_number.size(); i++)
            raw_variables_missing_values_number(i) = stoi(get_tokens(read_xml_string(missing_values_element, "RawVariablesMissingValuesNumber"), " ")[i]);

        rows_missing_values_number = read_xml_index(missing_values_element, "RowsMissingValuesNumber");
    }

    //preview data
    const XMLElement* preview_data_element = data_set_element->FirstChildElement("PreviewData");

    if (!preview_data_element)
        throw runtime_error("Preview data element is nullptr. \n ");

    const XMLElement* preview_size_element = preview_data_element->FirstChildElement("PreviewSize");

    if (!preview_size_element)
        throw runtime_error("Preview size element is nullptr. \n ");

    Index preview_size = 0;
    if (preview_size_element->GetText())
        preview_size = static_cast<Index>(atoi(preview_size_element->GetText()));

    start_element = preview_size_element;
    if(preview_size > 0){
        data_file_preview.resize(preview_size);

        for (Index i = 0; i < preview_size; ++i) {
            const XMLElement* row_data = start_element->NextSiblingElement("Row");
            start_element = row_data;

            if (row_data->Attribute("Item") != to_string(i + 1))
                throw runtime_error("Row item number (" + to_string(i + 1) + ") does not match (" + row_data->Attribute("Item") + ").\n");

            if(row_data->GetText())
                data_file_preview[i] = get_tokens(row_data->GetText(), ",");
        }
    }


    set_display(read_xml_bool(data_set_element, "Display"));

    input_dimensions = { get_variables_number("Input") };
    target_dimensions = { get_variables_number("Target") };
}


void Dataset::print() const
{
    if (!display) return;

    const Index variables_number = get_variables_number();
    const Index input_variables_number = get_variables_number("Input");
    const Index samples_number = get_samples_number();
    const Index target_variables_bumber = get_variables_number("Target");
    const Index training_samples_number = get_samples_number("Training");
    const Index selection_samples_number = get_samples_number("Selection");
    const Index testing_samples_number = get_samples_number("Testing");
    const Index unused_samples_number = get_samples_number("None");

    cout << "Data set object summary:\n"
        << "Number of samples: " << samples_number << "\n"
        << "Number of variables: " << variables_number << "\n"
        << "Number of input variables: " << input_variables_number << "\n"
        << "Number of target variables: " << target_variables_bumber << "\n"
        << "Input dimensions: ";

    print_vector(get_dimensions("Input"));

    cout << "Target dimensions: ";

    print_vector(get_dimensions("Target"));

    cout << "Number of training samples: " << training_samples_number << endl
        << "Number of selection samples: " << selection_samples_number << endl
        << "Number of testing samples: " << testing_samples_number << endl
        << "Number of unused samples: " << unused_samples_number << endl;

    //for (const Dataset::RawVariable& raw_variable : raw_variables)
    //    raw_variable.print();
}


void Dataset::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        return;

    XMLPrinter document;

    to_XML(document);

    file << document.CStr();
}


void Dataset::load(const filesystem::path& file_name)
{
    XMLDocument document;

    if (document.LoadFile(file_name.string().c_str()))
        throw runtime_error("Cannot load XML file " + file_name.string() + ".\n");

    from_XML(document);
}


void Dataset::print_raw_variables() const
{
    for (const Dataset::RawVariable& raw_variable : raw_variables)
        raw_variable.print();

    cout << endl;
}


void Dataset::print_data() const
{
    cout << data << endl;
}


void Dataset::print_data_preview() const
{
    if (!display) return;

    const Index samples_number = get_samples_number();

    if (samples_number > 0)
    {
        const Tensor<type, 1> first_sample = data.chip(0, 0);

        cout << "First sample: \n";

        for (int i = 0; i < first_sample.dimension(0); i++)
            cout << first_sample(i) << "  ";
    }

    if (samples_number > 1)
    {
        const Tensor<type, 1> second_sample = data.chip(1, 0);

        cout << "Second sample: \n";

        for (int i = 0; i < second_sample.dimension(0); i++)
            cout << second_sample(i) << "  ";
    }

    if (samples_number > 2)
    {
        const Tensor<type, 1> last_sample = data.chip(samples_number - 1, 0);

        cout << "Last sample: \n";

        for (int i = 0; i < last_sample.dimension(0); i++)
            cout << last_sample(i) << "  ";
    }

    cout << endl;
}


void Dataset::save_data() const
{
    ofstream file(data_path.c_str());

    if (!file.is_open())
        throw runtime_error("Cannot open matrix data file: " + data_path.string() + "\n");

    file.precision(20);

    const Index samples_number = get_samples_number();
    const Index variables_number = get_variables_number();

    const vector<string> variable_names = get_variable_names();

    const string separator_string = get_separator_string();

    if (has_sample_ids)
        file << "id" << separator_string;

    for (Index j = 0; j < variables_number; j++)
    {
        file << variable_names[j];

        if (j != variables_number - 1)
            file << separator_string;
    }

    file << endl;

    for (Index i = 0; i < samples_number; i++)
    {
        if (has_sample_ids)
            file << sample_ids[i] << separator_string;

        for (Index j = 0; j < variables_number; j++)
        {
            file << data(i, j);

            if (j != variables_number - 1)
                file << separator_string;
        }

        file << endl;
    }

    file.close();
}


void Dataset::save_data_binary(const filesystem::path& binary_data_file_name) const
{
    ofstream file(binary_data_file_name, ios::binary);

    if (!file.is_open())
        throw runtime_error("Cannot open data binary file.");

    // Write data

    streamsize size = sizeof(Index);

    Index columns_number = data.dimension(1);
    Index rows_number = data.dimension(0);

    file.write(reinterpret_cast<char*>(&columns_number), size);
    file.write(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(type);

    const Index total_elements = columns_number * rows_number;

    file.write(reinterpret_cast<const char*>(data.data()), total_elements * size);

    file.close();
}


void Dataset::load_data_binary()
{
    ifstream file(data_path, ios::binary);

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


Tensor<Index, 1> Dataset::calculate_target_distribution() const
{
    const Index samples_number = get_samples_number();
    const Index targets_number = get_variables_number("Target");
    const vector<Index> target_variable_indices = get_variable_indices("Target");

    Tensor<Index, 1> class_distribution;

    if (targets_number == 1)
    {
        class_distribution.resize(2);

        Index target_index = target_variable_indices[0];

        Index positives = 0;
        Index negatives = 0;

        for (Index sample_index = 0; sample_index < samples_number; sample_index++){
            if (!isnan(data(sample_index, target_index)))
                (data(sample_index, target_index) < type(0.5))
                ? negatives++
                : positives++;
        }

        class_distribution(0) = negatives;
        class_distribution(1) = positives;
    }
    else // More than two classes
    {
        class_distribution.resize(targets_number);

        class_distribution.setZero();

        for (Index i = 0; i < samples_number; i++)
        {
            if (get_sample_use(i) == "None")
                continue;

            for (Index j = 0; j < targets_number; j++)
            {
                if (isnan(data(i, target_variable_indices[j])))
                    continue;

                if (data(i, target_variable_indices[j]) > type(0.5))
                    class_distribution(j)++;
            }
        }
    }
    return class_distribution;
}


vector<vector<Index>> Dataset::calculate_Tukey_outliers(const type& cleaning_parameter) const
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

//#pragma omp parallel for

    for (Index i = 0; i < raw_variables_number; i++)
    {
        const RawVariable& raw_variable = raw_variables[i];

        if (raw_variable.use == "None"
            && raw_variable.type == RawVariableType::Categorical)
        {
            variable_index += raw_variable.get_categories_number();
            continue;
        }
        else if (raw_variable.use == "None") // Numeric, Binary or DateTime
        {
            variable_index++;
            continue;
        }

        if (raw_variable.type == RawVariableType::Categorical)
        {
            variable_index += raw_variable.get_categories_number();
            used_variable_index++;
            continue;
        }
        else if (raw_variable.type == RawVariableType::Binary
            || raw_variable.type == RawVariableType::DateTime)
        {
            variable_index++;
            used_variable_index++;
            continue;
        }
        else // Numeric
        {
            const type interquartile_range = box_plots[i].third_quartile - box_plots[i].first_quartile;

            if (interquartile_range < numeric_limits<type>::epsilon())
            {
                variable_index++;
                used_variable_index++;
                continue;
            }

            Index raw_variables_outliers = 0;

            for (Index j = 0; j < samples_number; j++)
            {
                const Tensor<type, 1> sample = get_sample_data(sample_indices[Index(j)]);

                if (sample(variable_index) < box_plots[i].first_quartile - cleaning_parameter * interquartile_range
                    || sample(variable_index) > box_plots[i].third_quartile + cleaning_parameter * interquartile_range)
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


vector<vector<Index>> Dataset::replace_Tukey_outliers_with_NaN(const type& cleaning_parameter)
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

    for (Index i = 0; i < raw_variables_number; i++)
    {
        const RawVariable& raw_variable = raw_variables[i];

        if (raw_variable.use == "None"
            && raw_variable.type == RawVariableType::Categorical)
        {
            variable_index += raw_variable.get_categories_number();
            continue;
        }
        else if (raw_variable.use == "None") // Numeric, Binary or DateTime
        {
            variable_index++;
            continue;
        }

        if (raw_variable.type == RawVariableType::Categorical)
        {
            variable_index += raw_variable.get_categories_number();
            used_variable_index++;
            continue;
        }
        else if (raw_variable.type == RawVariableType::Binary
            || raw_variable.type == RawVariableType::DateTime)
        {
            variable_index++;
            used_variable_index++;
            continue;
        }
        else // Numeric
        {
            const type interquartile_range = box_plots[i].third_quartile - box_plots[i].first_quartile;

            if (interquartile_range < numeric_limits<type>::epsilon())
            {
                variable_index++;
                used_variable_index++;
                continue;
            }

            Index raw_variables_outliers = 0;

            for (Index j = 0; j < samples_number; j++)
            {
                const Tensor<type, 1> sample = get_sample_data(sample_indices[Index(j)]);

                if (sample[variable_index] < (box_plots[i].first_quartile - cleaning_parameter * interquartile_range)
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


void Dataset::unuse_Tukey_outliers(const type& cleaning_parameter)
{
    const vector<vector<Index>> outliers_indices = calculate_Tukey_outliers(cleaning_parameter);

    const vector<Index> outliers_samples = get_elements_greater_than(outliers_indices, 0);

    set_sample_uses(outliers_samples, "None");
}


void Dataset::set_data_rosenbrock()
{
    const Index samples_number = get_samples_number();
    const Index variables_number = get_variables_number();

    set_data_random();

#pragma omp parallel for

    for (Index i = 0; i < samples_number; i++)
    {
        type rosenbrock(0);

        for (Index j = 0; j < variables_number - 1; j++)
        {
            const type value = data(i, j);
            const type next_value = data(i, j + 1);

            rosenbrock += (type(1) - value) * (type(1) - value) + type(100) * (next_value - value * value) * (next_value - value * value);
        }

        data(i, variables_number - 1) = rosenbrock;
    }
}


void Dataset::set_data_binary_classification()
{
    const Index samples_number = get_samples_number();
    const Index variables_number = get_variables_number();

    set_data_random();

    #pragma omp parallel for
    for (Index i = 0; i < samples_number; i++)
        data(i, variables_number - 1) = type(get_random_bool());
}


Tensor<Index, 1> Dataset::filter_data(const Tensor<type, 1>& minimums,
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

    for (Index i = 0; i < used_variables_number; i++)
    {
        for (Index j = 0; j < used_samples_number; j++)
        {
            sample_index = used_sample_indices[j];

            const type value = data(sample_index, used_variable_indices[i]);

            if (get_sample_use(sample_index) == "None"
                || isnan(value))
                continue;

            if (abs(value - minimums(i)) <= NUMERIC_LIMITS_MIN
                || abs(value - maximums(i)) <= NUMERIC_LIMITS_MIN)
                continue;

            if (minimums(i) == maximums(i))
            {
                if (value != minimums(i))
                {
                    filtered_indices(sample_index) = type(1);
                    set_sample_use(sample_index, "None");
                }
            }
            else if (value < minimums(i) || value > maximums(i))
            {
                filtered_indices(sample_index) = type(1);
                set_sample_use(sample_index, "None");
            }
        }
    }

    const Index filtered_samples_number =
        Index(count_if(filtered_indices.data(),
            filtered_indices.data() + filtered_indices.size(),
            [](type value)
            {
                return value > type(0.5);
            }));

    Tensor<Index, 1> filtered_samples_indices(filtered_samples_number);

    Index index = 0;

    for (Index i = 0; i < samples_number; i++)
        if (filtered_indices(i) > type(0.5))
            filtered_samples_indices(index++) = i;

    return filtered_samples_indices;
}


void Dataset::impute_missing_values_unuse()
{
    const Index samples_number = get_samples_number();

#pragma omp parallel for

    for (Index i = 0; i < samples_number; i++)
        if (has_nan_row(i))
            set_sample_use(i, "None");
}


void Dataset::impute_missing_values_mean()
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_variable_indices = get_used_variable_indices();
    const vector<Index> input_variable_indices = get_variable_indices("Input");
    const vector<Index> target_variable_indices = get_variable_indices("Target");

    const Tensor<type, 1> means = mean(data, used_sample_indices, used_variable_indices);

    const Index samples_number = used_sample_indices.size();
    const Index variables_number = used_variable_indices.size();
    const Index target_variables_number = target_variable_indices.size();

    Index current_variable;
    Index current_sample;

#pragma omp parallel for schedule(dynamic)

    for (Index j = 0; j < variables_number - target_variables_number; j++)
    {
        current_variable = input_variable_indices[j];

        for (Index i = 0; i < samples_number; i++)
        {
            current_sample = used_sample_indices[i];

            if (isnan(data(current_sample, current_variable)))
                data(current_sample, current_variable) = means(j);
        }
    }

#pragma omp parallel for schedule(dynamic)

    for (Index j = 0; j < target_variables_number; j++)
    {
        current_variable = target_variable_indices[j];

        for (Index i = 0; i < samples_number; i++)
        {
            current_sample = used_sample_indices[i];

            if (isnan(data(current_sample, current_variable)))
                set_sample_use(i, "None");
        }
    }
}


void Dataset::impute_missing_values_median()
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_variable_indices = get_used_variable_indices();
    const vector<Index> input_variable_indices = get_variable_indices("Input");
    const vector<Index> target_variable_indices = get_variable_indices("Target");

    const Tensor<type, 1> medians = median(data, used_sample_indices, used_variable_indices);

    const Index samples_number = used_sample_indices.size();
    const Index variables_number = used_variable_indices.size();
    const Index target_variables_number = target_variable_indices.size();

#pragma omp parallel for schedule(dynamic)

    for (Index j = 0; j < variables_number - target_variables_number; j++)
    {
        const Index current_variable = input_variable_indices[j];

        for (Index i = 0; i < samples_number; i++)
        {
            const Index current_sample = used_sample_indices[i];

            if (isnan(data(current_sample, current_variable)))
                data(current_sample, current_variable) = medians(j);
        }
    }

#pragma omp parallel for schedule(dynamic)

    for (Index j = 0; j < target_variables_number; j++)
    {
        const Index current_variable = target_variable_indices[j];

        for (Index i = 0; i < samples_number; i++)
        {
            const Index current_sample = used_sample_indices[i];

            if (isnan(data(current_sample, current_variable)))
                set_sample_use(i, "None");
        }
    }
}


void Dataset::impute_missing_values_interpolate()
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_variable_indices = get_used_variable_indices();
    const vector<Index> input_variable_indices = get_variable_indices("Input");
    const vector<Index> target_variable_indices = get_variable_indices("Target");

    const Index samples_number = used_sample_indices.size();
    const Index variables_number = used_variable_indices.size();
    const Index target_variables_number = target_variable_indices.size();

    Index current_variable;
    Index current_sample;

#pragma omp parallel for schedule(dynamic)
    for (Index j = 0; j < variables_number - target_variables_number; j++)
    {
        current_variable = input_variable_indices[j];

        for (Index i = 0; i < samples_number; i++)
        {
            current_sample = used_sample_indices[i];

            if (isnan(data(current_sample, current_variable)))
            {
                type x1 = type(0);
                type x2 = type(0);
                type y1 = type(0);
                type y2 = type(0);
                type x = type(0);
                type y = type(0);

                for (Index k = i - 1; k >= 0; k--)
                {
                    if (isnan(data(used_sample_indices[k], current_variable))) continue;

                    x1 = type(used_sample_indices[k]);
                    y1 = data(x1, current_variable);
                    break;
                }

                for (Index k = i + 1; k < samples_number; k++)
                {
                    if (isnan(data(used_sample_indices[k], current_variable))) continue;

                    x2 = type(used_sample_indices[k]);
                    y2 = data(x2, current_variable);
                    break;
                }

                if (x2 != x1)
                {
                    x = type(current_sample);
                    y = y1 + (x - x1) * (y2 - y1) / (x2 - x1);
                }
                else
                {
                    y = y1;
                }

                data(current_sample, current_variable) = y;
            }
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (Index j = 0; j < target_variables_number; j++)
    {
        current_variable = target_variable_indices[j];

        for (Index i = 0; i < samples_number; i++)
        {
            current_sample = used_sample_indices[i];

            if (isnan(data(current_sample, current_variable)))
                set_sample_use(i, "None");
        }
    }
}


void Dataset::scrub_missing_values()
{
    switch (missing_values_method)
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


void Dataset::prepare_line(string& line) const
{
    decode(line);
    trim(line);
    erase(line, '"');
}


void Dataset::infer_column_types(const vector<vector<string>>& sample_rows)
{
    const Index raw_variables_number = raw_variables.size();
    const size_t rows_to_check = std::min(size_t(100), sample_rows.size());

    for (Index col_idx = 0; col_idx < raw_variables_number; ++col_idx) {
        RawVariable& raw_variable = raw_variables[col_idx];
        raw_variable.type = RawVariableType::None;

        for (size_t row_idx = 0; row_idx < rows_to_check; ++row_idx) {
            const size_t token_idx = has_sample_ids ? col_idx + 1 : col_idx;
            if (token_idx >= sample_rows[row_idx].size()) continue;

            const string& token = sample_rows[row_idx][token_idx];
            if (token.empty() || token == missing_values_label) continue;

            if (raw_variable.type == RawVariableType::Categorical) break;

            if (is_numeric_string(token)) {
                if (raw_variable.type == RawVariableType::None) raw_variable.type = RawVariableType::Numeric;
            } else if (is_date_time_string(token)) {
                if (raw_variable.type == RawVariableType::None) raw_variable.type = RawVariableType::DateTime;
                else raw_variable.type = RawVariableType::Categorical;
            } else {
                raw_variable.type = RawVariableType::Categorical;
            }
        }

        if (raw_variable.type == RawVariableType::None) {
            raw_variable.type = RawVariableType::Numeric;
        }
    }

    for (Index col_idx = 0; col_idx < raw_variables_number; ++col_idx) {
        if (raw_variables[col_idx].type == RawVariableType::Categorical) {
            std::set<string> unique_categories;
            for (const auto& row : sample_rows) {
                const size_t token_idx = has_sample_ids ? col_idx + 1 : col_idx;
                if (token_idx < row.size() && !row[token_idx].empty() && row[token_idx] != missing_values_label) {
                    unique_categories.insert(row[token_idx]);
                }
            }
            raw_variables[col_idx].categories.assign(unique_categories.begin(), unique_categories.end());
        }
    }
}


void Dataset::read_csv()
{
    if (data_path.empty())
        throw runtime_error("Data path is empty.\n");

    ifstream file(data_path, ios::binary);
    if (!file.is_open())
        throw runtime_error("Error: Cannot open file " + data_path.string() + "\n");

    char bom[3] = {0};
    file.read(bom, 3);
    if (static_cast<unsigned char>(bom[0]) != 0xEF || static_cast<unsigned char>(bom[1]) != 0xBB || static_cast<unsigned char>(bom[2]) != 0xBF) {
        file.seekg(0);
    }

    vector<vector<string>> raw_file_content;
    string line;
    const string separator_string = get_separator_string();
    while (getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        prepare_line(line);
        if (line.empty()) continue;
        check_separators(line);
        raw_file_content.push_back(get_tokens(line, separator_string));
    }

    file.close();

    if (raw_file_content.empty())
        throw runtime_error("File " + data_path.string() + " is empty or contains no valid data rows.");

    read_data_file_preview(raw_file_content);

    vector<string> header_tokens = raw_file_content[0];
    if (has_header) {
        if (has_numbers(header_tokens)) throw runtime_error("Error: Some header names are numeric.");
        raw_file_content.erase(raw_file_content.begin());
    }

    if (raw_file_content.empty()) throw runtime_error("Data file only contains a header.");

    const Index samples_number = raw_file_content.size();
    const size_t columns_number = header_tokens.size();
    const Index raw_variables_number = has_sample_ids ? columns_number - 1 : columns_number;
    raw_variables.resize(raw_variables_number);

    if (has_header) {
        if (has_sample_ids) for (Index i = 0; i < raw_variables_number; i++) raw_variables[i].name = header_tokens[i + 1];
        else set_raw_variable_names(header_tokens);
    } else {
        set_default_raw_variable_names();
    }

    infer_column_types(raw_file_content);

    for (Dataset::RawVariable& raw_variable : raw_variables)
        if (raw_variable.type == RawVariableType::Categorical && raw_variable.get_categories_number() == 2)
            raw_variable.type = RawVariableType::Binary;

    sample_uses.resize(samples_number);
    sample_ids.resize(samples_number);
    const vector<vector<Index>> all_variable_indices = get_variable_indices();
    const Index total_numeric_columns = all_variable_indices.empty() ? 0 : all_variable_indices.back().back() + 1;
    data.resize(samples_number, total_numeric_columns);
    data.setZero();

    rows_missing_values_number = 0;
    missing_values_number = 0;
    raw_variables_missing_values_number.resize(raw_variables_number);
    raw_variables_missing_values_number.setZero();

    // #pragma omp parallel for
    for (Index sample_index = 0; sample_index < samples_number; ++sample_index)
    {
        const vector<string>& tokens = raw_file_content[sample_index];

        if (has_missing_values(tokens)) {
            rows_missing_values_number++;
            for (size_t i = (has_sample_ids ? 1 : 0); i < tokens.size(); i++) {
                if (tokens[i].empty() || tokens[i] == missing_values_label) {
                    missing_values_number++;
                    raw_variables_missing_values_number(has_sample_ids ? i - 1 : i)++;
                }
            }
        }

        if (has_sample_ids) {
            sample_ids[sample_index] = tokens[0];
        }

        for (Index raw_variable_index = 0; raw_variable_index < raw_variables_number; raw_variable_index++)
        {
            const RawVariable& raw_variable = raw_variables[raw_variable_index];
            const string& token = has_sample_ids ? tokens[raw_variable_index + 1] : tokens[raw_variable_index];
            const vector<Index>& variable_indices = all_variable_indices[raw_variable_index];

            switch(raw_variable.type) {
            case RawVariableType::Numeric:
                data(sample_index, variable_indices[0]) = (token.empty() || token == missing_values_label) ? NAN : stof(token);
                break;
            case RawVariableType::DateTime:
                data(sample_index, variable_indices[0]) = time_t(date_to_timestamp(token));
                break;
            case RawVariableType::Categorical:
                if (token.empty() || token == missing_values_label) {
                    for (Index cat_idx : variable_indices) data(sample_index, cat_idx) = NAN;
                } else {
                    auto it = std::find(raw_variable.categories.begin(), raw_variable.categories.end(), token);
                    if (it != raw_variable.categories.end()) {
                        Index category_index = std::distance(raw_variable.categories.begin(), it);
                        data(sample_index, variable_indices[category_index]) = 1;
                    }
                }
                break;
            case RawVariableType::Binary:
                if (contains(positive_words, token) || contains(negative_words, token)) {
                    data(sample_index, variable_indices[0]) = contains(positive_words, token) ? 1 : 0;
                } else {
                    const vector<string>& categories = raw_variable.categories;
                    if (token.empty() || token == missing_values_label)
                        data(sample_index, variable_indices[0]) = NAN;
                    else if (categories.size() > 0 && token == categories[0])
                        data(sample_index, variable_indices[0]) = 1;
                    else if (categories.size() > 1 && token == categories[1])
                        data(sample_index, variable_indices[0]) = 0;
                    else
                        data(sample_index, variable_indices[0]) = stof(token);
                }
                break;
            default: break;
            }
        }
    }

    unuse_constant_raw_variables();
    set_binary_raw_variables();
    split_samples_random();
}


string Dataset::RawVariable::get_type_string() const
{
    switch (type)
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


void Dataset::read_data_file_preview(const vector<vector<string>>& all_rows)
{
    if (all_rows.empty())
        return;

    const Index num_first_rows_to_show = 3;

    data_file_preview.clear();

    for (Index i = 0; i < min((size_t)num_first_rows_to_show, all_rows.size()); ++i)
        data_file_preview.push_back(all_rows[i]);

    if (all_rows.size() > num_first_rows_to_show)
        data_file_preview.push_back(all_rows.back());
    else if (all_rows.empty() && data_file_preview.size() < num_first_rows_to_show +1 )
        while(data_file_preview.size() < num_first_rows_to_show +1)
            data_file_preview.push_back(vector<string>());
}


void Dataset::check_separators(const string& line) const
{
    if (line.find(',') == string::npos
        && line.find(';') == string::npos
        && line.find(' ') == string::npos
        && line.find('\t') == string::npos) return;

    const string separator_string = get_separator_string();

    if (line.find(separator_string) == string::npos)
        throw runtime_error("Error: Separator '" + separator_string + "' not found in line " + line + ".\n");

    if (separator == Separator::Space)
    {
        if (line.find(',') != string::npos)
            throw runtime_error("Error: Found comma (',') in data file " + data_path.string() + ", but separator is space (' ').");
        else if (line.find(';') != string::npos)
            throw runtime_error("Error: Found semicolon (';') in data file " + data_path.string() + ", but separator is space (' ').");
    }
    else if (separator == Separator::Tab)
    {
        if (line.find(',') != string::npos)
            throw runtime_error("Error: Found comma (',') in data file " + data_path.string() + ", but separator is tab ('   ').");
        else if (line.find(';') != string::npos)
            throw runtime_error("Error: Found semicolon (';') in data file " + data_path.string() + ", but separator is tab ('   ').");
    }
    else if (separator == Separator::Comma)
    {
        if (line.find(";") != string::npos)
            throw runtime_error("Error: Found semicolon (';') in data file " + data_path.string() + ", but separator is comma (',').");
    }
    else if (separator == Separator::Semicolon)
    {
        if (line.find(",") != string::npos)
            throw runtime_error("Error: Found comma (',') in data file " + data_path.string() + ", but separator is semicolon (';').");
    }
}


void Dataset::fill_input_tensor(const vector<Index>& sample_indices, const vector<Index>& input_indices, type* input_tensor_data) const
{
    fill_tensor_data(data, sample_indices, input_indices, input_tensor_data);
}


void Dataset::fill_input_tensor_row_major(const vector<Index>& sample_indices, const vector<Index>& input_indices, type* input_tensor_data) const
{
    fill_tensor_data_row_major(data, sample_indices, input_indices, input_tensor_data);
}


// void Dataset::fill_decoder_tensor(const vector<Index>& sample_indices, const vector<Index>& decoder_indices, type* decoder_tensor_data) const
// {
//     fill_tensor_data(data, sample_indices, decoder_indices, decoder_tensor_data);
// }


void Dataset::fill_target_tensor(const vector<Index>& sample_indices, const vector<Index>& target_indices, type* target_tensor_data) const
{
    fill_tensor_data(data, sample_indices, target_indices, target_tensor_data);
}


bool Dataset::has_binary_raw_variables() const
{
    return any_of(raw_variables.begin(), raw_variables.end(),
        [](const RawVariable& raw_variable) { return raw_variable.type == RawVariableType::Binary; });
}


bool Dataset::has_categorical_raw_variables() const
{
    return any_of(raw_variables.begin(), raw_variables.end(),
        [](const RawVariable& raw_variable) { return raw_variable.type == RawVariableType::Categorical; });
}


bool Dataset::has_binary_or_categorical_raw_variables() const
{
    for (const Dataset::RawVariable& raw_variable : raw_variables)
        if (raw_variable.type == RawVariableType::Binary || raw_variable.type == RawVariableType::Categorical)
            return true;

    return false;
}


bool Dataset::has_time_raw_variable() const
{
    return any_of(raw_variables.begin(), raw_variables.end(),
                  [](const RawVariable& raw_variable) { return raw_variable.type == RawVariableType::DateTime; });
}


bool Dataset::has_selection() const
{
    return get_samples_number("Selection") != 0;
}


bool Dataset::has_missing_values(const vector<string>& row) const
{
    for (size_t i = 0; i < row.size(); i++)
        if (row[i].empty() || row[i] == missing_values_label)
            return true;

    return false;
}


Tensor<Index, 1> Dataset::count_nans_per_raw_variable() const
{
    const Index raw_variables_number = get_raw_variables_number();
    const Index rows_number = get_samples_number();

    Tensor<Index, 1> raw_variables_with_nan(raw_variables_number);
    raw_variables_with_nan.setZero();

    #pragma omp parallel for
    for (Index raw_variable_index = 0; raw_variable_index < raw_variables_number; raw_variable_index++)
    {
        const Index current_variable_index = get_variable_indices(raw_variable_index)[0];

        Index counter = 0;

        for (Index row_index = 0; row_index < rows_number; row_index++)
            if (isnan(data(row_index, current_variable_index)))
                counter++;

        raw_variables_with_nan(raw_variable_index) = counter;
    }

    return raw_variables_with_nan;
}


Index Dataset::count_raw_variables_with_nan() const
{
    Tensor<Index, 1> raw_variables_with_nan = count_nans_per_raw_variable();

    Index missing_raw_variables_number = 0;

    for (Index i = 0; i < raw_variables_with_nan.dimension(0); i++)
        if (raw_variables_with_nan(i) > 0)
            missing_raw_variables_number++;

    return missing_raw_variables_number;
}


Index Dataset::count_rows_with_nan() const
{
    Index rows_with_nan = 0;

    const Index rows_number = data.dimension(0);
    const Index raw_variables_number = data.dimension(1);

    bool has_nan = true;

    for (Index row_index = 0; row_index < rows_number; row_index++)
    {
        has_nan = false;

        for (Index raw_variable_index = 0; raw_variable_index < raw_variables_number; raw_variable_index++)
        {
            if (isnan(data(row_index, raw_variable_index)))
            {
                has_nan = true;
                break;
            }
        }

        if (has_nan)
            rows_with_nan++;
    }

    return rows_with_nan;
}


Index Dataset::count_nan() const
{
    return count_NAN(data);
}


void Dataset::fix_repeated_names()
{
    map<string, Index> raw_variables_count_map;

    for (const Dataset::RawVariable& raw_variable : raw_variables)
    {
        auto result = raw_variables_count_map.insert(pair<string, Index>(raw_variable.name, 1));

        if (!result.second)
            result.first->second++;
    }

    for (const auto& element : raw_variables_count_map)
    {
        if (element.second > 1)
        {
            const string repeated_name = element.first;

            Index repeated_index = 1;

            for (Dataset::RawVariable& raw_variable : raw_variables)
                if (raw_variable.name == repeated_name)
                    raw_variable.name = raw_variable.name + "_" + to_string(repeated_index++);
        }
    }

    // Fix variables names

    if (has_categorical_raw_variables() || has_binary_raw_variables())
    {
        vector<string> variable_names = get_variable_names();

        const Index variables_number = variable_names.size();

        map<string, Index> variables_count_map;

        for (Index i = 0; i < variables_number; i++)
        {
            auto result = variables_count_map.insert(pair<string, Index>(variable_names[i], 1));

            if (!result.second) result.first->second++;
        }

        for (const auto& element : variables_count_map)
        {
            if (element.second > 1)
            {
                const string repeated_name = element.first;

                for (Index i = 0; i < variables_number; i++)
                {
                    if (variable_names[i] == repeated_name)
                    {
                        const Index raw_variable_index = get_raw_variable_index(i);

                        if (raw_variables[raw_variable_index].type != RawVariableType::Categorical)
                            continue;

                        variable_names[i] += "_" + raw_variables[raw_variable_index].name;
                    }
                }
            }
        }

        set_variable_names(variable_names);
    }
}


vector<vector<Index>> Dataset::split_samples(const vector<Index>& sample_indices, const Index& new_batch_size) const
{
    const Index samples_number = sample_indices.size();
    Index batch_size = new_batch_size;
    Index batches_number;

    if (samples_number < batch_size)
    {
        batches_number = 1;
        batch_size = samples_number;
    }
    else
        batches_number = (samples_number + batch_size - 1) / batch_size;

    vector<vector<Index>> batches(batches_number);

#pragma omp parallel for
    for (Index i = 0; i < batches_number; i++)
    {
        const Index start_index = i * batch_size;
        const Index end_index = std::min(start_index + batch_size, samples_number);

        batches[i].assign(sample_indices.begin() + start_index,
            sample_indices.begin() + end_index);
    }
    return batches;
}


bool Dataset::get_has_rows_labels() const
{
    return has_sample_ids;
}


void Dataset::decode(string&) const
{
    switch (codification)
    {
    case Dataset::Codification::SHIFT_JIS:
        //        input_string = sj2utf8(input_string);
        break;
    default:
        break;
    }
}


void Batch::fill(const vector<Index>& sample_indices,
                 const vector<Index>& input_indices,
                 // const vector<Index>& decoder_indices,
                 const vector<Index>& target_indices)
{
    dataset->fill_input_tensor(sample_indices, input_indices, input_tensor.data());

    if (dynamic_cast<opennn::TimeSeriesDataset*>(dataset))
    {
        input_dimensions.clear();
        input_dimensions.push_back(sample_indices.size());
        input_dimensions.push_back(input_indices.size());
        input_dimensions.push_back(input_indices.size());
    }

    dataset->fill_target_tensor(sample_indices, target_indices, target_tensor.data());

    // dataset->fill_decoder_tensor(sample_indices, decoder_indices, decoder_tensor.data());
}


Batch::Batch(const Index& new_samples_number, Dataset* new_dataset)
{
    const unsigned int threads_number = thread::hardware_concurrency();
    thread_pool = make_unique<ThreadPool>(threads_number);
    thread_pool_device = make_unique<ThreadPoolDevice>(thread_pool.get(), threads_number);

    set(new_samples_number, new_dataset);
}


void Batch::set(const Index& new_samples_number, Dataset* new_dataset)
{
    if (!new_dataset) return;

    samples_number = new_samples_number;
    dataset = new_dataset;

    const dimensions& data_set_input_dimensions = dataset->get_dimensions("Input");
    // const dimensions& data_set_decoder_dimensions = dataset->get_dimensions("Decoder");
    const dimensions& data_set_target_dimensions = dataset->get_dimensions("Target");

    if (!data_set_input_dimensions.empty())
    {
        input_dimensions = prepend(samples_number, data_set_input_dimensions);
        input_tensor.resize(get_size(input_dimensions));
    }

    // if (!data_set_decoder_dimensions.empty())
    // {
    //     decoder_dimensions = prepend(samples_number, data_set_decoder_dimensions);
    //     decoder_tensor.resize(get_size(decoder_dimensions));
    // }

    if (!data_set_target_dimensions.empty())
    {
        target_dimensions = prepend(samples_number, data_set_target_dimensions);
        target_tensor.resize(get_size(target_dimensions));
    }
}


Index Batch::get_samples_number() const
{
    return samples_number;
}


void Batch::print() const
{
    // cout << "Batch" << endl
    //      << "Inputs:" << endl
    //      << "Input dimensions:" << endl;

    // print_vector(input_dimensions);

    // input_dimensions.size() == 4
    //     ? cout << TensorMap<Tensor<type, 4>>((type*)input_tensor.data(),
    //                                          input_dimensions[0],
    //                                          input_dimensions[1],
    //                                          input_dimensions[2],
    //                                          input_dimensions[3]) << endl
    //     : cout << TensorMap<Tensor<type, 2>>((type*)input_tensor.data(),
    //                                          input_dimensions[0],
    //                                          input_dimensions[1]) << endl;

    // cout << "Decoder:" << endl
    //      << "Decoder dimensions:" << endl;

    // print_vector(decoder_dimensions);

    // cout << "Targets:" << endl
    //      << "Target dimensions:" << endl;

    // print_vector(target_dimensions);

    // cout << TensorMap<Tensor<type, 2>>((type*)target_tensor.data(),
    //                                    target_dimensions[0],
    //                                    target_dimensions[1]) << endl;
}


bool Batch::is_empty() const
{
    return input_tensor.size() == 0;
}



vector<pair<type*, dimensions>> Batch::get_input_pairs() const
{
    vector<pair<type*, dimensions>> input_pairs = {{(type*)input_tensor.data(), input_dimensions}};

    // @todo DECODER VARIABLES
    // if (!decoder_dimensions.empty())
    //     input_pairs.insert(input_pairs.begin(), {(type*)decoder_tensor.data(), decoder_dimensions});

    return input_pairs;
}


pair<type*, dimensions> Batch::get_target_pair() const
{
    return { (type*)target_tensor.data() , target_dimensions};
}


#ifdef OPENNN_CUDA

void BatchCuda::fill(const vector<Index>& sample_indices,
                     const vector<Index>& input_indices,
                     const vector<Index>& decoder_indices,
                     const vector<Index>& target_indices)
{
    dataset->fill_input_tensor_row_major(sample_indices, input_indices, inputs_host);

    //dataset->fill_decoder_tensor(sample_indices, decoder_indices, decoder_host);

    dataset->fill_target_tensor(sample_indices, target_indices, targets_host);

    copy_device();
}


BatchCuda::BatchCuda(const Index& new_samples_number, Dataset* new_dataset)
{
    set(new_samples_number, new_dataset);
}


void BatchCuda::set(const Index& new_samples_number, Dataset* new_dataset)
{
    if (!new_dataset) return;

    samples_number = new_samples_number;
    dataset = new_dataset;

    const dimensions& data_set_input_dimensions = dataset->get_dimensions("Input");
    const dimensions& data_set_decoder_dimensions = dataset->get_dimensions("Decoder");
    const dimensions& data_set_target_dimensions = dataset->get_dimensions("Target");

    if (!data_set_input_dimensions.empty())
    {
        input_dimensions = { samples_number };
        input_dimensions.insert(input_dimensions.end(), data_set_input_dimensions.begin(), data_set_input_dimensions.end());

        const Index input_size = accumulate(input_dimensions.begin(), input_dimensions.end(), 1, multiplies<Index>());

        CHECK_CUDA(cudaMallocHost(&inputs_host, input_size * sizeof(float)));
        //CHECK_CUDA(cudaMalloc(&inputs_device, input_size * sizeof(float)));
        CUDA_MALLOC_AND_REPORT(inputs_device, input_size * sizeof(float));
    }

    if (!data_set_decoder_dimensions.empty())
    {
        decoder_dimensions = { samples_number };
        decoder_dimensions.insert(decoder_dimensions.end(), data_set_decoder_dimensions.begin(), data_set_decoder_dimensions.end());

        const Index decoder_size = accumulate(decoder_dimensions.begin(), decoder_dimensions.end(), 1, multiplies<Index>());

        CHECK_CUDA(cudaMallocHost(&decoder_host, decoder_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&decoder_device, decoder_size * sizeof(float)));
    }

    if (!data_set_target_dimensions.empty())
    {
        target_dimensions = { samples_number };
        target_dimensions.insert(target_dimensions.end(), data_set_target_dimensions.begin(), data_set_target_dimensions.end());

        const Index target_size = accumulate(target_dimensions.begin(), target_dimensions.end(), 1, multiplies<Index>());

        CHECK_CUDA(cudaMallocHost(&targets_host, target_size * sizeof(float)));
        //CHECK_CUDA(cudaMalloc(&targets_device, target_size * sizeof(float)));
        CUDA_MALLOC_AND_REPORT(targets_device, target_size * sizeof(float));
    }
}


void BatchCuda::copy_device()
{
    const Index input_size = accumulate(input_dimensions.begin(), input_dimensions.end(), 1, multiplies<Index>());
    const Index decoder_size = accumulate(decoder_dimensions.begin(), decoder_dimensions.end(), 1, multiplies<Index>());
    const Index target_size = accumulate(target_dimensions.begin(), target_dimensions.end(), 1, multiplies<Index>());

    CHECK_CUDA(cudaMemcpy(inputs_device, inputs_host, input_size * sizeof(float), cudaMemcpyHostToDevice));

    if (!decoder_dimensions.empty())
        CHECK_CUDA(cudaMemcpy(decoder_device, decoder_host, decoder_size * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(targets_device, targets_host, target_size * sizeof(float), cudaMemcpyHostToDevice));
}


Tensor<type, 2> BatchCuda::get_inputs_device() const
{
    const Index inputs_number = dataset->get_raw_variables_number("Input");

    Tensor<type, 2> inputs(samples_number, inputs_number);

    inputs.setZero();

    CHECK_CUDA(cudaMemcpy(inputs.data(), inputs_device, samples_number * inputs_number * sizeof(type), cudaMemcpyDeviceToHost));

    return inputs;
}


Tensor<type, 2> BatchCuda::get_decoder_device() const
{
    const Index decoder_number = dataset->get_raw_variables_number("Decoder");

    Tensor<type, 2> decoder(samples_number, decoder_number);

    decoder.setZero();

    CHECK_CUDA(cudaMemcpy(decoder.data(), inputs_device, samples_number * decoder_number * sizeof(type), cudaMemcpyDeviceToHost));

    return decoder;
}


Tensor<type, 2> BatchCuda::get_targets_device() const
{
    const Index targets_number = target_dimensions[1];

    Tensor<type, 2> targets(samples_number, targets_number);

    targets.setZero();

    CHECK_CUDA(cudaMemcpy(targets.data(), targets_device, samples_number * targets_number * sizeof(type), cudaMemcpyDeviceToHost));

    return targets;
}


vector<float*> BatchCuda::get_input_device() const
{
    vector<float*> inputs = { inputs_device };

    if (!decoder_dimensions.empty())
        inputs.insert(inputs.begin(), decoder_device );

    return inputs;
}


pair<type*, dimensions> BatchCuda::get_target_pair_device() const
{
    pair<type*, dimensions> target_pair = {targets_device , target_dimensions};

    return target_pair;
}


Index BatchCuda::get_samples_number() const
{
    return samples_number;
}


void BatchCuda::print() const
{
    if (!input_dimensions.empty())
        cout << "get_inputs_device:" << endl << get_inputs_device() << endl;

    if (!decoder_dimensions.empty())
        cout << "get_decoder_device:" << endl << get_decoder_device() << endl;

    if (!target_dimensions.empty())
        cout << "get_targets_device:" << endl << get_targets_device() << endl;
}


bool BatchCuda::is_empty() const
{
    return input_dimensions.empty();
}


void BatchCuda::free()
{
    cudaFreeHost(inputs_host);
    cudaFreeHost(decoder_host);
    cudaFreeHost(targets_host);
    cudaFree(inputs_device);
    cudaFree(decoder_device);
    cudaFree(targets_device);

    inputs_device = nullptr;
    decoder_device = nullptr;
    targets_device = nullptr;
    inputs_host = nullptr;
    decoder_host = nullptr;
    targets_host = nullptr;
}

#endif

} // namespace opennn


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
