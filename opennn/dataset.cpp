//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "dataset.h"
#include <regex>
#include "time_series_dataset.h"
#include "statistics.h"
#include "scaling.h"
#include "correlations.h"
#include "tensor_utilities.h"
#include "string_utilities.h"
#include "random_utilities.h"
#include "variable.h"

namespace opennn
{

Dataset::Dataset(const Index new_samples_number,
                 const Shape& new_input_shape,
                 const Shape& new_target_shape)
{
    set(new_samples_number, new_input_shape, new_target_shape);
}

Dataset::Dataset(const filesystem::path& data_path,
                 const string& separator,
                 bool has_header,
                 bool has_sample_ids,
                 const Codification& data_codification)
{
    set(data_path, separator, has_header, has_sample_ids, data_codification);
}

VectorI Dataset::get_sample_role_numbers() const
{
    VectorI count(4);
    count.setZero();

    const Index samples_number = get_samples_number();

    for(Index i = 0; i < samples_number; i++)
    {
        const string& role = sample_roles[i];

        if (role == "Training")         count[0]++;
        else if (role == "Validation")  count[1]++;
        else if (role == "Testing")     count[2]++;
        else if (role == "None")        count[3]++;
        else throw runtime_error("Unknown sample role: " + role);
    }

    return count;
}

vector<Index> Dataset::get_sample_indices(const string& sample_role) const
{
    const Index samples_number = get_samples_number();

    vector<Index> indices;
    indices.reserve(get_samples_number(sample_role));

    for(Index i = 0; i < samples_number; ++i)
        if(sample_roles[i] == sample_role)
            indices.push_back(i);

    return indices;
}

vector<Index> Dataset::get_used_sample_indices() const
{
    const Index samples_number = get_samples_number();

    vector<Index> used_indices;
    used_indices.reserve(samples_number - get_samples_number("None"));

    for(Index i = 0; i < samples_number; i++)
        if (sample_roles[i] != "None")
            used_indices.push_back(i);

    return used_indices;
}

vector<Index> Dataset::get_sample_roles_vector() const
{
    const Index samples_number = get_samples_number();

    vector<Index> sample_roles_vector(samples_number);

#pragma omp parallel for

    for(Index i = 0; i < samples_number; i++)
    {
        const string& role = sample_roles[i];

        if(role == "Training")
            sample_roles_vector[i] = 0;
        else if(role == "Validation")
            sample_roles_vector[i] = 1;
        else if(role == "Testing")
            sample_roles_vector[i] = 2;
        else if(role == "None")
            sample_roles_vector[i] = 3;
        else
            throw runtime_error("Unknown sample role: " + role);
    }

    return sample_roles_vector;
}

vector<vector<Index>> Dataset::get_batches(const vector<Index>& sample_indices,
                                           Index batch_size,
                                           bool shuffle) const
{
    const Index samples_number = sample_indices.size();

    if(samples_number == 0) return {};

    if(batch_size <= 0 || batch_size > samples_number)
        batch_size = samples_number;

    const Index batches_number = (samples_number + batch_size - 1) / batch_size;

    vector<Index> shuffled_indices;

    if(shuffle)
    {
        shuffled_indices = sample_indices;
        shuffle_vector_blocks(shuffled_indices);
    }

    const vector<Index>& indices = shuffle ? shuffled_indices : sample_indices;

    vector<vector<Index>> batches(batches_number);

    #pragma omp parallel for if(batches_number > 64)
    for(Index i = 0; i < batches_number; i++)
    {
        const auto start_it = indices.begin() + (i * batch_size);
        const auto end_it = indices.begin() + min((i + 1) * batch_size, samples_number);

        batches[i].assign(start_it, end_it);
    }

    return batches;
}

Index Dataset::get_samples_number(const string& sample_role) const
{
    return count(sample_roles.begin(), sample_roles.end(), sample_role);
}

Index Dataset::get_used_samples_number() const
{
    const Index samples_number = get_samples_number();
    const Index unused_samples_number = get_samples_number("None");

    return samples_number - unused_samples_number;
}

void Dataset::set_sample_roles(const string& sample_role)
{
    fill(sample_roles.begin(), sample_roles.end(), sample_role);
}

void Dataset::set_sample_role(const Index index, const string& new_role)
{
    static const unordered_set<string> valid_roles = {"Training", "Validation", "Testing", "None"};

    if (valid_roles.count(new_role))
        sample_roles[index] = new_role;
    else
        throw runtime_error("Unknown sample role: " + new_role + "\n");
}

void Dataset::set_sample_roles(const vector<string>& new_roles)
{
    const Index samples_number = new_roles.size();

    for(Index i = 0; i < samples_number; i++)
        if (new_roles[i] == "Training" || new_roles[i] == "0")
            sample_roles[i] = "Training";
        else if (new_roles[i] == "Validation" || new_roles[i] == "1")
            sample_roles[i] = "Validation";
        else if (new_roles[i] == "Testing" || new_roles[i] == "2")
            sample_roles[i] = "Testing";
        else if (new_roles[i] == "None" || new_roles[i] == "3")
            sample_roles[i] = "None";
        else
            throw runtime_error("Unknown sample role: " + new_roles[i] + ".\n");
}

void Dataset::set_sample_roles(const vector<Index>& indices, const string& sample_role)
{
    for(const auto& i : indices)
        set_sample_role(i, sample_role);
}

void Dataset::split_samples(const type training_samples_ratio,
                            type validation_samples_ratio,
                            type testing_samples_ratio,
                            bool shuffle)
{
    const Index used_samples_number = get_used_samples_number();

    if (used_samples_number == 0) return;

    const type total_ratio = training_samples_ratio + validation_samples_ratio + testing_samples_ratio;

    const Index validation_samples_number = Index((validation_samples_ratio * used_samples_number) / total_ratio);
    const Index testing_samples_number = Index((testing_samples_ratio * used_samples_number) / total_ratio);
    const Index training_samples_number = used_samples_number - validation_samples_number - testing_samples_number;

    if (training_samples_number + validation_samples_number + testing_samples_number != used_samples_number)
        throw runtime_error("Sum of numbers of training, selection and testing samples is not equal to number of used samples.\n");

    const Index samples_number = get_samples_number();

    vector<Index> indices(samples_number);
    iota(indices.begin(), indices.end(), 0);

    if(shuffle)
        shuffle_vector(indices);

    auto assign_role = [this, &indices](const string& role, Index count, Index& i)
    {
        Index assigned = 0;

        while (assigned < count)
        {
            const Index idx = indices[i++];

            if (sample_roles[idx] != "None")
            {
                sample_roles[idx] = role;
                assigned++;
            }
        }
    };

    Index index = 0;

    assign_role("Training", training_samples_number, index);
    assign_role("Validation", validation_samples_number, index);
    assign_role("Testing", testing_samples_number, index);
}

void Dataset::split_samples_random(const type training_ratio,
                                   type validation_ratio,
                                   type testing_ratio)
{
    split_samples(training_ratio, validation_ratio, testing_ratio, true);
}

void Dataset::split_samples_sequential(const type training_ratio,
                                       type validation_ratio,
                                       type testing_ratio)
{
    split_samples(training_ratio, validation_ratio, testing_ratio, false);
}

void Dataset::set_default_variable_roles()
{
    const Index variables_number = variables.size();

    if (variables_number == 0)
        return;

    if (variables_number == 1)
    {
        variables[0].set_role("None");
        return;
    }

    set_variable_roles("Input");

    for(Index i = variables_number - 1; i >= 0; i--)
    {
        Variable& variable = variables[i];

        if (variable.type == VariableType::Constant
        ||  variable.type == VariableType::DateTime)
        {
            variable.set_role("None");
        }
        else
        {
            variable.set_role("Target");
            break;
        }
    }
}

vector<Index> Dataset::get_feature_dimensions() const
{
    const Index used_variables_number = get_used_variables_number();

    vector<Index> feature_dimensions(used_variables_number);

    Index i = 0;

    for(const Variable& variable : variables)
    {
        if(!variable.is_used())
            continue;

        variable.is_categorical()
            ? feature_dimensions[i] = variable.get_categories_number()
            : feature_dimensions[i] = 1;

        i++;
    }

    return feature_dimensions;
}

void Dataset::set_default_variable_roles_forecasting()
{
    const Index variables_number = variables.size();

    bool target = false;
    bool time_variable = false;

    if (variables_number == 0)
        return;

    if (variables_number == 1)
    {
        variables[0].set_role("None");
        return;
    }

    set_variable_roles("Input");

    for(Index i = variables_number - 1; i >= 0; i--)
    {
        if (variables[i].type == VariableType::DateTime && !time_variable)
        {
            variables[i].set_role("Time");

            time_variable = true;
            continue;
        }

        if (variables[i].type == VariableType::Constant)
        {
            variables[i].set_role("None");
            continue;
        }

        if(!target)
        {
            variables[i].set_role("Target");

            target = true;

            continue;
        }
    }
}

vector<VariableType> Dataset::get_variable_types(const vector<Index> indices) const
{
    vector<VariableType> variable_types(indices.size());

    transform(indices.begin(), indices.end(), variable_types.begin(),
              [this](Index i) { return get_variable_type(i); });

    return variable_types;
}

void Dataset::set_default_variable_names()
{
    const Index variables_number = variables.size();

    for(Index i = 0; i < variables_number; i++)
        variables[i].name = "variable_" + to_string(1 + i);
}

vector<string> Dataset::get_feature_names() const
{
    vector<string> feature_names;

    for (const auto& variable : variables)
    {
        const vector<string> names = variable.get_names();

        feature_names.insert(feature_names.end(), names.begin(), names.end());
    }
    return feature_names;
}

vector<string> Dataset::get_feature_names(const string& variable_role) const
{
    vector<string> feature_names;

    for (const auto& variable : get_variables(variable_role))
    {
        const vector<string> names = variable.get_names();

        feature_names.insert(feature_names.end(), names.begin(), names.end());
    }

    return feature_names;
}

Shape Dataset::get_shape(const string& variable_role) const
{
    if (variable_role == "Input")
        return input_shape;
    else if (variable_role == "Target")
        return target_shape;
    else if (variable_role == "Decoder")
         return decoder_shape;
    else
        throw invalid_argument("get_shape: Invalid variable role string: " + variable_role);
}

void Dataset::set_shape(const string& variable_role, const Shape& new_shape)
{
    if (variable_role == "Input")
        input_shape = new_shape;
    else if (variable_role == "Target")
        target_shape = new_shape;
     else if (variable_role == "Decoder")
         decoder_shape = new_shape;
    else
        throw invalid_argument("set_shape: Invalid variable role string: " + variable_role);
}

Index Dataset::get_used_features_number() const
{
    const Index features_number = get_features_number();

    const Index unused_variables_number = get_features_number("None");

    const Index time_variables_number = get_features_number("Time");

    return features_number - unused_variables_number - time_variables_number;
}

vector<Index> Dataset::get_feature_indices(const string& variable_role) const
{
    const Index this_features_number = get_features_number(variable_role);
    vector<Index> this_feature_indices(this_features_number);

    Index feature_index = 0;
    Index this_feature_index = 0;

    for(const Variable& variable : variables)
    {
        const Index count = variable.is_categorical() ? variable.get_categories_number() : 1;

        if (variable.role.find(variable_role) == string::npos)
        {
            feature_index += count;
            continue;
        }

        for(Index j = 0; j < count; j++)
            this_feature_indices[this_feature_index++] = feature_index++;
    }

    return this_feature_indices;
}

vector<Index> Dataset::get_variable_indices(const string& variable_role) const
{
    vector<Index> indices;
    indices.reserve(get_variables_number(variable_role));

    for(size_t i = 0; i < variables.size(); i++)
        if (variables[i].role.find(variable_role) != string::npos)
            indices.push_back(i);

    return indices;
}

vector<Index> Dataset::get_used_variables_indices() const
{
    vector<Index> used_indices;
    used_indices.reserve(get_used_variables_number());

    for(size_t i = 0; i < variables.size(); i++)
        if (variables[i].is_used())
            used_indices.push_back(i);

    return used_indices;
}

vector<string> Dataset::get_feature_scalers(const string& variable_role) const
{
    const vector<Variable> role_variables = get_variables(variable_role);

    vector<string> scalers;
    scalers.reserve(get_features_number(variable_role));

    for(const Variable& var : role_variables)
    {
        const Index count = var.is_categorical() ? var.get_categories_number() : 1;

        for(Index j = 0; j < count; j++)
            scalers.push_back(var.scaler);
    }

    return scalers;
}

vector<string> Dataset::get_variable_names() const
{
    vector<string> variable_names(variables.size());

    transform(variables.begin(), variables.end(), variable_names.begin(),
              [](const Variable& var) { return var.name; });

    return variable_names;
}

vector<string> Dataset::get_variable_names(const string& variable_role) const
{
    vector<string> names;
    names.reserve(get_variables_number(variable_role));

    for(const Variable& variable : variables)
    {
        if(variable.role == variable_role
        || ((variable_role == "Input" || variable_role == "Target") && variable.role == "InputTarget"))
            names.push_back(variable.name);
    }

    return names;
}

Index Dataset::get_variables_number(const string& variable_role) const
{
    Index count = 0;

    for(const Variable& variable : variables)
        if (variable.role.find(variable_role) != string::npos)
            count++;

    return count;
}

Index Dataset::get_used_variables_number() const
{
    return count_if(variables.begin(), variables.end(),
                    [](const Variable& var) {
                          return var.is_used();
                    });
}

vector<Variable> Dataset::get_variables(const string& variable_role) const
{
    vector<Variable> this_variables;
    this_variables.reserve(get_variables_number(variable_role));

    copy_if(variables.begin(), variables.end(), back_inserter(this_variables),
            [&variable_role](const Variable& var) { return var.role.find(variable_role) != string::npos; });

    return this_variables;
}

Index Dataset::get_features_number() const
{
    return accumulate(variables.begin(), variables.end(), 0,
                      [](Index sum, const Variable& var) {
                      return sum + (var.type == VariableType::Categorical ? var.get_categories_number() : 1);
                      });
}

Index Dataset::get_features_number(const string& variable_role) const
{
    Index count = 0;

    for(const Variable& variable : variables)
        if (variable.role.find(variable_role) != string::npos)
            count += (variable.type == VariableType::Categorical)
                         ? variable.get_categories_number()
                         : 1;

    return count;
}

vector<Index> Dataset::get_used_feature_indices() const
{
    const Index used_features_number = get_used_features_number();
    vector<Index> used_feature_indices(used_features_number);

    Index feature_index = 0;
    Index used_feature_index = 0;

    for(const Variable& variable : variables)
    {
        const Index count = variable.is_categorical() ? variable.get_categories_number() : 1;

        if(variable.role == "None" || variable.role == "Time")
        {
            feature_index += count;
            continue;
        }

        for(Index j = 0; j < count; j++)
            used_feature_indices[used_feature_index++] = feature_index++;
    }

    return used_feature_indices;
}

void Dataset::set_variable_roles(const vector<string>& new_variables_roles)
{
    const size_t new_variables_roles_size = new_variables_roles.size();

    if (new_variables_roles_size != variables.size())
        throw runtime_error("Size of variables uses (" + to_string(new_variables_roles_size) + ") "
                                                                                                      "must be equal to variables size (" + to_string(variables.size()) + ").\n");

    for(size_t i = 0; i < new_variables_roles.size(); i++)
        variables[i].set_role(new_variables_roles[i]);
}

void Dataset::set_variables(const string& variable_role)
{
    const Index variables_number = get_variables_number();

    for(Index i = 0; i < variables_number; i++)
        set_variable_role(i, variable_role);
}

void Dataset::set_variable_indices(const vector<Index>& input_variables,
                                       const vector<Index>& target_variables)
{
    set_variables("None");

    for(const Index index : input_variables)
        set_variable_role(index, "Input");

    for(const Index index : target_variables)
        variables[index].role == "Input"
            ? set_variable_role(index, "InputTarget")
            : set_variable_role(index, "Target");

    const Index input_dimensions_num = get_features_number("Input");
    const Index target_shape_num = get_features_number("Target");

    if (auto* ts = dynamic_cast<TimeSeriesDataset*>(this))
        set_shape("Input", {ts->get_past_time_steps(), input_dimensions_num});
    else
        set_shape("Input", {input_dimensions_num});

    set_shape("Target", {target_shape_num});
}

void Dataset::set_input_variables_unused()
{
    const Index variables_number = get_variables_number();

    for(Index i = 0; i < variables_number; i++)
        if (variables[i].role == "Input")
            set_variable_role(i, "None");
}

void Dataset::set_variable_role(const Index index, const string& new_role)
{
    static const unordered_set<string> valid_strings
        = {"Id", "Input", "Target", "InputTarget", "Time", "None", "Decoder"};

    if(valid_strings.count(new_role) <= 0)
        throw runtime_error("Invalid variable role: " + new_role);

    variables[index].role = new_role;
}

void Dataset::set_variable_role(const string& name, const string& new_role)
{
    const Index index = get_variable_index(name);

    set_variable_role(index, new_role);
}

void Dataset::set_variable_type(const Index index, const VariableType& new_type)
{
    variables[index].type = new_type;
}

void Dataset::set_variable_type(const string& name, const VariableType& new_type)
{
    const Index index = get_variable_index(name);

    set_variable_type(index, new_type);
}

void Dataset::set_variable_types(const VariableType& new_type)
{
    for(auto& variable : variables)
        variable.type = new_type;
}

void Dataset::set_feature_names(const vector<string>& new_variables_names)
{
    Index index = 0;

    for(Variable& variable : variables)
        if (variable.type == VariableType::Categorical)
            for(Index j = 0; j < variable.get_categories_number(); j++)
                variable.categories[j] = new_variables_names[index++];
        else
            variable.name = new_variables_names[index++];
}

void Dataset::set_variable_names(const vector<string>& new_names)
{
    const Index new_names_size = new_names.size();
    const Index variables_number = get_variables_number();

    if (new_names_size != variables_number)
        throw runtime_error("Size of names (" + to_string(new_names.size()) + ") "
                            "is not equal to variables number (" + to_string(variables_number) + ").\n");

    for(Index i = 0; i < variables_number; i++)
        variables[i].name = get_trimmed(new_names[i]);
}

void Dataset::set_variable_roles(const string& variable_role)
{
    for(Variable& variable : variables)
        variable.type == VariableType::Constant || variable.type == VariableType::DateTime
            ? variable.set_role("None")
            : variable.set_role(variable_role);
}

void Dataset::set_variable_scalers(const string& scalers)
{
    for(Variable& variable : variables)
        variable.scaler = scalers;
}

void Dataset::set_variable_scalers(const vector<string>& new_scalers)
{
    const size_t variables_number = get_variables_number();

    if (new_scalers.size() != variables_number)
        throw runtime_error("Size of variable scalers(" + to_string(new_scalers.size()) + ") "
                            "has to be the same as variables numbers(" + to_string(variables_number) + ").\n");

    for(size_t i = 0; i < variables_number; i++)
        variables[i].scaler = new_scalers[i];
}

void Dataset::infer_variable_types_from_data()
{
    Index feature_index = 0;

    const Index variables_number = get_variables_number();

    for(Index variable_index = 0; variable_index < variables_number; variable_index++)
    {
        Variable& variable = variables[variable_index];

        if (variable.type == VariableType::Numeric)
        {
            const VectorR data_column = data.col(feature_index);

            if (is_constant(data_column))
                variable.set(variable.name, "None", VariableType::Constant);
            else if (is_binary(data_column))
            {
                variable.type = VariableType::Binary;
                variable.categories = { "0", "1" };
            }

            feature_index++;
        }
        else if (variable.type == VariableType::Binary)
        {
            if (variable.get_categories_number() == 1)
                variable.set(variable.name, "None", VariableType::Constant);

            feature_index++;
        }
        else if (variable.type == VariableType::Categorical)
        {
            if (variable.get_categories_number() == 1)
                variable.set(variable.name, "None", VariableType::Constant);

            feature_index += variable.get_categories_number();
        }
        else
        {
            feature_index++;
        }
    }
}

void Dataset::set_binary_variables()
{
    infer_variable_types_from_data();
}

void Dataset::unuse_constant_variables()
{
    infer_variable_types_from_data();
}

static const vector<pair<Dataset::MissingValuesMethod, string>> missing_values_method_map = {
    {Dataset::MissingValuesMethod::Unuse,         "Unuse"},
    {Dataset::MissingValuesMethod::Mean,          "Mean"},
    {Dataset::MissingValuesMethod::Median,        "Median"},
    {Dataset::MissingValuesMethod::Interpolation, "Interpolation"}
};

string Dataset::get_missing_values_method_string() const
{
    for(const auto& [method, name] : missing_values_method_map)
        if (method == missing_values_method) return name;

    throw runtime_error("Unknown missing values method");
}

static const vector<tuple<Dataset::Separator, string, string>> separator_map = {
    {Dataset::Separator::Space,     " ",  "Space"},
    {Dataset::Separator::Tab,       "\t", "Tab"},
    {Dataset::Separator::Comma,     ",",  "Comma"},
    {Dataset::Separator::Semicolon, ";",  "Semicolon"}
};

string Dataset::get_separator_string() const
{
    for(const auto& [sep, str, name] : separator_map)
        if (sep == separator) return str;

    return string();
}

string Dataset::get_separator_name() const
{
    for(const auto& [sep, str, name] : separator_map)
        if (sep == separator) return name;

    return string();
}

static const vector<pair<Dataset::Codification, string>> codification_map = {
    {Dataset::Codification::UTF8,      "UTF-8"},
    {Dataset::Codification::SHIFT_JIS, "SHIFT_JIS"}
};

const string Dataset::get_codification_string() const
{
    for(const auto& [cod, name] : codification_map)
        if (cod == codification) return name;

    return "UTF-8";
}

MatrixR Dataset::get_feature_data(const string& variable_role) const
{
    const Index samples_number = get_samples_number();

    vector<Index> indices(samples_number);
    iota(indices.begin(), indices.end(), 0);

    return get_data_from_indices(indices, get_feature_indices(variable_role));
}

MatrixR Dataset::get_data(const string& sample_role, const string& variable_role) const
{
    return get_data_from_indices(get_sample_indices(sample_role), get_feature_indices(variable_role));
}

MatrixR Dataset::get_data_from_indices(const vector<Index>& sample_indices, const vector<Index>& feature_indices) const
{
    MatrixR this_data(sample_indices.size(), feature_indices.size());

    fill_tensor_data(data, sample_indices, feature_indices, this_data.data());

    return this_data;
}

VectorR Dataset::get_sample_data(const Index index) const
{
    return data.row(index);
}

Index Dataset::get_variable_index(const string& variable_name) const
{
    const Index variables_number = get_variables_number();

    for(Index i = 0; i < variables_number; i++)
        if (variables[i].name == variable_name)
            return i;

    throw runtime_error("Cannot find " + variable_name + "\n");
}

Index Dataset::get_variable_index(const Index feature_index) const
{
    const Index variables_number = get_variables_number();

    Index total_variables_number = 0;

    for(Index i = 0; i < variables_number; i++)
    {
        total_variables_number += (variables[i].type == VariableType::Categorical)
            ? variables[i].get_categories_number()
            : 1;

        if (feature_index + 1 <= total_variables_number)
            return i;
    }

    throw runtime_error("Cannot find variable index: " + to_string(feature_index) + ".\n");
}

vector<vector<Index>> Dataset::get_feature_indices() const
{
    const Index variables_number = get_variables_number();

    vector<vector<Index>> indices(variables_number);

    for(Index i = 0; i < variables_number; i++)
        indices[i] = get_feature_indices(i);

    return indices;
}

vector<Index> Dataset::get_feature_indices(const Index variable_index) const
{
    Index index = 0;

    for(Index i = 0; i < variable_index; i++)
        index += (variables[i].type == VariableType::Categorical)
                     ? variables[i].categories.size()
                     : 1;

    const Variable& variable = variables[variable_index];

    if (variable.type == VariableType::Categorical)
    {
        vector<Index> indices(variable.categories.size());
        iota(indices.begin(), indices.end(), index);

        return indices;
    }

    return vector<Index>(1, index);
}

MatrixR Dataset::get_variable_data(const Index variable_index) const
{
    Index variables_number = 1;
    const Index rows_number = data.rows();

    if (variables[variable_index].type == VariableType::Categorical)
        variables_number = variables[variable_index].get_categories_number();

    const Index start_column = get_feature_indices(variable_index)[0];

    return data.block(0, start_column, rows_number, variables_number);
}

MatrixR Dataset::get_variable_data(const Index variable_index, const vector<Index>& row_indices) const
{
    MatrixR variable_data(row_indices.size(), get_feature_indices(variable_index).size());

    fill_tensor_data(data, row_indices, get_feature_indices(variable_index), variable_data.data());

    return variable_data;
}

MatrixR Dataset::get_variable_data(const string& column_name) const
{
    const Index variable_index = get_variable_index(column_name);

    return get_variable_data(variable_index);
}

void Dataset::set(const filesystem::path& new_data_path,
                  const string& new_separator,
                  bool new_has_header,
                  bool new_has_ids,
                  const Dataset::Codification& new_codification)
{
    set_default();

    set_data_path(new_data_path);

    set_separator_string(new_separator);

    set_has_header(new_has_header);

    set_has_ids(new_has_ids);

    set_codification(new_codification);

    read_csv();

    set_default_variable_scalers();

    set_default_variable_roles();

    missing_values_method = MissingValuesMethod::Unuse;

    input_shape = { get_features_number("Input") };
    target_shape = { get_features_number("Target") };
}

void Dataset::set(const Index new_samples_number,
                  const Shape& new_input_shape,
                  const Shape& new_target_shape)
{
    if (new_samples_number == 0
        || new_input_shape.empty()
        || new_target_shape.empty())
        return;

    input_shape = new_input_shape;

    const Index new_inputs_number = new_input_shape.size();

    Index new_targets_number = new_target_shape.size();

    new_targets_number = (new_targets_number == 2) ? 1 : new_targets_number;

    target_shape = { new_targets_number };

    const Index new_features_number = new_inputs_number + new_targets_number;

    data.resize(new_samples_number, new_features_number);

    variables.resize(new_features_number);

    set_default();

    for(Index i = 0; i < new_features_number; i++)
    {
        Variable& variable = variables[i];

        variable.type = VariableType::Numeric;
        variable.name = "variable_" + to_string(i + 1);

        variable.role = (i < new_inputs_number)
                               ? "Input"
                               : "Target";
    }

    sample_roles.resize(new_samples_number);

    split_samples_random();
}

void Dataset::set(const filesystem::path& file_name)
{
    load(file_name);
}

void Dataset::set_default()
{
    has_header = false;

    has_sample_ids = false;

    separator = Separator::Semicolon;

    missing_values_label = "NA";

    set_default_variable_names();
}

void Dataset::set_data(const MatrixR& new_data)
{
    if (new_data.rows() != get_samples_number())
        throw runtime_error("Rows number is not equal to samples number");

    if (new_data.cols() != get_features_number())
        throw runtime_error("Columns number is not equal to variables number");

    data = new_data;
}

void Dataset::set_separator_string(const string& new_separator_string)
{
    for(const auto& [sep, str, name] : separator_map)
        if (str == new_separator_string) { separator = sep; return; }

    throw runtime_error("Unknown separator: " + new_separator_string);
}

void Dataset::set_separator_name(const string& new_separator_name)
{
    for(const auto& [sep, str, name] : separator_map)
        if (name == new_separator_name) { separator = sep; return; }

    throw runtime_error("Unknown separator: " + new_separator_name + ".\n");
}

void Dataset::set_codification(const string& new_codification_string)
{
    for(const auto& [cod, name] : codification_map)
        if (name == new_codification_string) { codification = cod; return; }

    throw runtime_error("Unknown codification: " + new_codification_string + ".\n");
}

void Dataset::set_missing_values_method(const string& new_missing_values_method)
{
    for(const auto& [method, name] : missing_values_method_map)
        if (name == new_missing_values_method) { missing_values_method = method; return; }

    throw runtime_error("Unknown method type.\n");
}

vector<string> Dataset::unuse_uncorrelated_variables(const type minimum_correlation)
{
    vector<string> unused_variables;

    const Tensor<Correlation, 2> correlations = calculate_input_target_variable_pearson_correlations();

    const Index input_variables_number = get_variables_number("Input");
    const Index target_variables_number = get_variables_number("Target");

    const vector<Index> input_variable_indices = get_variable_indices("Input");

    for(Index i = 0; i < input_variables_number; i++)
    {
        const Index input_variable_index = input_variable_indices[i];

        bool has_significant_correlation = false;

        for(Index j = 0; j < target_variables_number; j++)
        {
            const type r = correlations(i, j).r;

            if(!isnan(r) && abs(r) >= minimum_correlation)
            {
                has_significant_correlation = true;
                break;
            }
        }

        if(!has_significant_correlation && variables[input_variable_index].role != "None")
        {
            variables[input_variable_index].set_role("None");
            unused_variables.push_back(variables[input_variable_index].name);
        }
    }

    const Index new_input_variables_number = get_features_number("Input");
    const Index new_target_variables_number = get_features_number("Target");

    if(const TimeSeriesDataset* time_series_dataset = dynamic_cast<TimeSeriesDataset*>(this))
        set_shape("Input", {time_series_dataset->get_past_time_steps(), new_input_variables_number});
    else
        set_shape("Input", {new_input_variables_number});

    set_shape("Target", {new_target_variables_number});

    return unused_variables;
}

vector<string> Dataset::unuse_collinear_variables(const type maximum_correlation)
{
    const Tensor<Correlation, 2> correlations = calculate_input_variable_pearson_correlations();
    const vector<Index> input_variable_indices = get_variable_indices("Input");
    const Index input_variables_number = input_variable_indices.size();

    vector<Index> high_corr_counts(input_variables_number, 0);
    vector<type> mean_abs_corr(input_variables_number, 0.0);
    vector<bool> to_be_removed(input_variables_number, false);

    for(Index i = 0; i < input_variables_number; ++i)
    {
        type sum_of_abs_corr = 0.0;
        for(Index j = 0; j < input_variables_number; ++j)
        {
            if (i == j) continue;

            const type abs_r = abs(correlations(i, j).r);
            if(!isnan(abs_r))
            {
                if (abs_r >= maximum_correlation)
                    high_corr_counts[i]++;

                sum_of_abs_corr += abs_r;
            }
        }

        if (input_variables_number > 1)
            mean_abs_corr[i] = sum_of_abs_corr / (input_variables_number - 1);
    }

    for(Index i = 0; i < input_variables_number; ++i)
    {
        for(Index j = i + 1; j < input_variables_number; ++j)
        {

            if (to_be_removed[i] || to_be_removed[j])
                continue;

            if(!isnan(correlations(i, j).r) && abs(correlations(i, j).r) >= maximum_correlation)
            {
                const Index index_to_flag_for_removal =
                    (high_corr_counts[i] > high_corr_counts[j]) ? i :
                        (high_corr_counts[j] > high_corr_counts[i]) ? j :
                        (mean_abs_corr[i] >= mean_abs_corr[j]) ? i : j;

                to_be_removed[index_to_flag_for_removal] = true;
            }
        }
    }

    vector<string> unused_variables;
    for(Index i = 0; i < input_variables_number; ++i)
    {
        if (to_be_removed[i])
        {
            const Index global_variable_index = input_variable_indices[i];

            if (variables[global_variable_index].role != "None")
            {
                variables[global_variable_index].set_role("None");
                unused_variables.push_back(variables[global_variable_index].name);
            }
        }
    }

    return unused_variables;
}

vector<Histogram> Dataset::calculate_variable_distributions(const Index bins_number) const
{
    const Index used_variables_number = get_used_variables_number();
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const Index used_samples_number = used_sample_indices.size();

    vector<Histogram> histograms(used_variables_number);

    Index feature_index = 0;
    Index used_variable_index = 0;

    for(const Variable& variable : variables)
    {
        if (variable.role == "None")
        {
            feature_index += (variable.type == VariableType::Categorical)
            ? variable.get_categories_number()
            : 1;
            continue;
        }

        switch (variable.type)
        {

        case VariableType::Numeric:
        {
            VectorR variable_data(used_samples_number);

            for(Index j = 0; j < used_samples_number; j++)
                variable_data(j) = data(used_sample_indices[j], feature_index);

            histograms[used_variable_index++] = histogram(variable_data, bins_number);

            feature_index++;
        }
        break;

        case VariableType::Categorical:
        {
            const Index categories_number = variable.get_categories_number();

            VectorR categories_frequencies(categories_number);
            categories_frequencies.setZero();
            VectorR centers(categories_number);

            for(Index j = 0; j < categories_number; j++)
            {
                for(Index k = 0; k < used_samples_number; k++)
                    if (abs(data(used_sample_indices[k], feature_index) - type(1)) < EPSILON)
                        categories_frequencies(j)++;

                centers(j) = type(j);

                feature_index++;
            }

            histograms[used_variable_index].frequencies = categories_frequencies;
            histograms[used_variable_index].centers = centers;

            used_variable_index++;
        }
        break;

        case VariableType::Binary:
        {
            VectorR binary_frequencies(2);
            binary_frequencies.setZero();

            for(Index j = 0; j < used_samples_number; j++)
                binary_frequencies(abs(data(used_sample_indices[j], feature_index) - type(1)) < EPSILON
                   ? 1
                   : 0)++;

            histograms[used_variable_index].frequencies = binary_frequencies;
            feature_index++;
            used_variable_index++;
        }
        break;

        case VariableType::DateTime:

            feature_index++;

            break;

        default:

            throw runtime_error("Unknown variable type.");
        }
    }

    return histograms;
}

vector<BoxPlot> Dataset::calculate_variables_box_plots() const
{
    const Index variables_number = get_variables_number();

    const vector<Index> used_sample_indices = get_used_sample_indices();

    vector<BoxPlot> box_plots(variables_number);

    Index feature_index = 0;

    for(Index i = 0; i < variables_number; i++)
    {
        const Variable& variable = variables[i];

        if (variable.type == VariableType::Numeric
        || variable.type == VariableType::Binary)
        {
            if (variable.role != "None")
                box_plots[i] = box_plot(data.col(feature_index), used_sample_indices);

            feature_index++;
        }
        else if (variable.type == VariableType::Categorical)
        {
            feature_index += variable.get_categories_number();
        }
        else
        {
            feature_index++;
        }
    }

    return box_plots;
}

vector<Descriptives> Dataset::calculate_feature_descriptives() const
{
    return descriptives(data);
}

vector<Index> Dataset::filter_used_samples_by_column(Index column_index, bool positive) const
{
    const vector<Index> used_sample_indices = get_used_sample_indices();

    vector<Index> filtered;
    filtered.reserve(used_sample_indices.size());

    for(const Index sample_index : used_sample_indices)
    {
        const bool match = positive
            ? abs(data(sample_index, column_index) - type(1)) < EPSILON
            : data(sample_index, column_index) < EPSILON;

        if(match)
            filtered.push_back(sample_index);
    }

    return filtered;
}

vector<Descriptives> Dataset::calculate_variable_descriptives_positive_samples() const
{
    const Index target_index = get_feature_indices("Target")[0];

    return descriptives(data, filter_used_samples_by_column(target_index, true), get_feature_indices("Input"));
}

vector<Descriptives> Dataset::calculate_variable_descriptives_negative_samples() const
{
    const Index target_index = get_feature_indices("Target")[0];

    return descriptives(data, filter_used_samples_by_column(target_index, false), get_feature_indices("Input"));
}

vector<Descriptives> Dataset::calculate_variable_descriptives_categories(const Index class_index) const
{
    return descriptives(data, filter_used_samples_by_column(class_index, true), get_feature_indices("Input"));
}

vector<Descriptives> Dataset::calculate_feature_descriptives(const string& variable_role) const
{
    const vector<Index> used_sample_indices = get_used_sample_indices();

    const vector<Index> input_feature_indices = get_feature_indices(variable_role);

    return descriptives(data, used_sample_indices, input_feature_indices);
}

Tensor<Correlation, 2> Dataset::calculate_input_target_variable_correlations(
    Correlation (*correlation_function)(const MatrixR&, const MatrixR&),
    const string& method_name) const
{
    if (display) cout << "Calculating " << method_name << " correlations..." << endl;

    const Index input_variables_number = get_variables_number("Input");
    const Index target_variables_number = get_variables_number("Target");

    const vector<Index> input_variable_indices = get_variable_indices("Input");
    const vector<Index> target_variable_indices = get_variable_indices("Target");

    const vector<Index> used_sample_indices = get_used_sample_indices();

    Tensor<Correlation, 2> correlations(input_variables_number, target_variables_number);

    #pragma omp parallel for schedule(dynamic)
    for(Index i = 0; i < input_variables_number; i++)
    {
        const Index input_variable_index = input_variable_indices[i];
        const MatrixR input_variable_data = get_variable_data(input_variable_index, used_sample_indices);

        for(Index j = 0; j < target_variables_number; j++)
        {
            const Index target_variable_index = target_variable_indices[j];
            const MatrixR target_variable_data = get_variable_data(target_variable_index, used_sample_indices);
            correlations(i, j) = correlation_function(input_variable_data, target_variable_data);
        }
    }

    return correlations;
}

Tensor<Correlation, 2> Dataset::calculate_input_target_variable_pearson_correlations() const
{
    return calculate_input_target_variable_correlations(correlation, "pearson");
}

Tensor<Correlation, 2> Dataset::calculate_input_target_variable_spearman_correlations() const
{
    return calculate_input_target_variable_correlations(correlation_spearman, "spearman");
}

bool Dataset::has_nan() const
{
    const Index rows_number = data.rows();

    for(Index i = 0; i < rows_number; i++)
        if (sample_roles[i] != "None")
            if (has_nan_row(i))
                return true;

    return false;
}

bool Dataset::has_nan_row(const Index row_index) const
{
    const Index features_number = get_features_number();

    for(Index j = 0; j < features_number; j++)
        if (isnan(data(row_index, j)))
            return true;

    return false;
}

Tensor<Correlation, 2> Dataset::calculate_input_variable_correlations(
    Correlation (*correlation_function)(const MatrixR&, const MatrixR&),
    Correlation::Method method,
    const string& method_name) const
{
    if (display) cout << "Calculating " << method_name << " inputs correlations..." << endl;

    const vector<Index> input_variable_indices = get_variable_indices("Input");

    const Index input_variables_number = input_variable_indices.size();

    Tensor<Correlation, 2> correlations(input_variables_number, input_variables_number);

    for(Index i = 0; i < input_variables_number; i++)
    {
        if (display) cout << "Correlation " << i + 1 << " of " << input_variables_number << endl;

        const MatrixR input_i = get_variable_data(input_variable_indices[i]);

        if (is_constant(input_i)) continue;

        correlations(i, i).set_perfect();
        correlations(i, i).method = method;

        for(Index j = i + 1; j < input_variables_number; j++)
        {
            const MatrixR input_j = get_variable_data(input_variable_indices[j]);

            correlations(i, j) = correlation_function(input_i, input_j);

            if (correlations(i, j).r > type(1) - EPSILON)
                correlations(i, j).r = type(1);

            correlations(j, i) = correlations(i, j);
        }
    }

    return correlations;
}

Tensor<Correlation, 2> Dataset::calculate_input_variable_pearson_correlations() const
{
    return calculate_input_variable_correlations(correlation, Correlation::Method::Pearson, "pearson");
}

Tensor<Correlation, 2> Dataset::calculate_input_variable_spearman_correlations() const
{
    return calculate_input_variable_correlations(correlation_spearman, Correlation::Method::Spearman, "spearman");
}

VectorI Dataset::calculate_correlations_rank() const
{
    const Tensor<Correlation, 2> correlations = calculate_input_target_variable_pearson_correlations();

    const MatrixR absolute_correlations = get_correlation_values(correlations).array().abs();

    const VectorR absolute_mean_correlations = absolute_correlations.rowwise().mean();

    return calculate_rank(absolute_mean_correlations);
}

void Dataset::set_default_variable_scalers()
{
    for(Variable& variable : variables)
        variable.scaler = (variable.type == VariableType::Numeric)
                                  ? "MeanStandardDeviation"
                                  : "MinimumMaximum";
}

void Dataset::apply_scaler(Index feature_index, const string& scaler, const Descriptives& desc, bool unscale)
{
    if(scaler == "None")
        return;
    else if(scaler == "MinimumMaximum")
        unscale ? unscale_minimum_maximum(data, feature_index, desc)
                : scale_minimum_maximum(data, feature_index, desc);
    else if(scaler == "MeanStandardDeviation")
        unscale ? unscale_mean_standard_deviation(data, feature_index, desc)
                : scale_mean_standard_deviation(data, feature_index, desc);
    else if(scaler == "StandardDeviation")
        unscale ? unscale_standard_deviation(data, feature_index, desc)
                : scale_standard_deviation(data, feature_index, desc);
    else if(scaler == "Logarithm")
        unscale ? unscale_logarithmic(data, feature_index)
                : scale_logarithmic(data, feature_index);
    else if(unscale && scaler == "ImageMinMax")
        unscale_image_minimum_maximum(data, feature_index);
    else
        throw runtime_error("Unknown scaler: " + scaler + "\n");
}

vector<Descriptives> Dataset::scale_data()
{
    const Index features_number = get_features_number();

    const vector<Descriptives> feature_descriptives = calculate_feature_descriptives();

    for(Index i = 0; i < features_number; i++)
        apply_scaler(i, variables[get_variable_index(i)].scaler, feature_descriptives[i], false);

    return feature_descriptives;
}

vector<Descriptives> Dataset::scale_features(const string& variable_role)
{
    const vector<Index> feature_indices = get_feature_indices(variable_role);
    const vector<string> scalers = get_feature_scalers(variable_role);
    const vector<Descriptives> feature_descriptives = calculate_feature_descriptives(variable_role);

    for(size_t i = 0; i < feature_indices.size(); i++)
        apply_scaler(feature_indices[i], scalers[i], feature_descriptives[i], false);

    return feature_descriptives;
}

void Dataset::unscale_features(const string& variable_role,
                                const vector<Descriptives>& feature_descriptives)
{
    const vector<Index> feature_indices = get_feature_indices(variable_role);
    const vector<string> scalers = get_feature_scalers(variable_role);

    for(size_t i = 0; i < feature_indices.size(); i++)
        apply_scaler(feature_indices[i], scalers[i], feature_descriptives[i], true);
}

void Dataset::set_data_constant(const type new_value)
{
    data.setConstant(new_value);
}

void Dataset::set_data_random()
{
    set_random_uniform(data);
}

void Dataset::set_data_integer(const Index vocabulary_size)
{
    set_random_integer(data, 0, vocabulary_size - 1);
}

void Dataset::to_XML(XMLPrinter& printer) const
{
    printer.OpenElement("Dataset");

    printer.OpenElement("DataSource");
    write_xml_properties(printer, {
        {"FileType", "csv"},
        {"Path", data_path.string()},
        {"Separator", get_separator_name()},
        {"HasHeader", to_string(has_header)},
        {"HasSamplesId", to_string(has_sample_ids)},
        {"MissingValuesLabel", missing_values_label},
        {"Codification", get_codification_string()}
    });
    printer.CloseElement();

    variables_to_XML(printer);
    samples_to_XML(printer);
    missing_values_to_XML(printer);
    preview_data_to_XML(printer);

    add_xml_element(printer, "Display", to_string(display));

    printer.CloseElement();
}

void Dataset::variables_to_XML(XMLPrinter &printer) const
{
    printer.OpenElement("Variables");
    add_xml_element(printer, "VariablesNumber", to_string(get_variables_number()));

    for(Index i = 0; i < get_variables_number(); i++)
    {
        printer.OpenElement("Variable");
        printer.PushAttribute("Item", to_string(i + 1).c_str());
        variables[i].to_XML(printer);
        printer.CloseElement();
    }

    printer.CloseElement();
}

void Dataset::samples_to_XML(XMLPrinter &printer) const
{
    printer.OpenElement("Samples");

    add_xml_element(printer, "SamplesNumber", to_string(get_samples_number()));

    const string separator_string = get_separator_string();

    if (has_sample_ids)
        add_xml_element(printer, "SamplesId", vector_to_string(sample_ids, separator_string));

    add_xml_element(printer, "SampleRoles", vector_to_string(get_sample_roles_vector()));
    printer.CloseElement();
}

void Dataset::missing_values_to_XML(XMLPrinter &printer) const
{
    printer.OpenElement("MissingValues");
    add_xml_element(printer, "MissingValuesNumber", to_string(missing_values_number));

    if (missing_values_number > 0)
    {
        add_xml_element(printer, "MissingValuesMethod", get_missing_values_method_string());
        add_xml_element(printer, "VariablesMissingValuesNumber", vector_to_string(variables_missing_values_number));
        add_xml_element(printer, "SamplesMissingValuesNumber", to_string(rows_missing_values_number));
    }

    printer.CloseElement();
}

void Dataset::preview_data_to_XML(XMLPrinter &printer) const
{
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
}

void Dataset::variables_from_XML(const XMLElement *variables_element)
{
    if(!variables_element)
        throw runtime_error("Variables element is nullptr.\n");

    set_variables_number(read_xml_index(variables_element, "VariablesNumber"));

    for_xml_items(variables_element, "Variable", variables.size(), [&](Index i, const XMLElement* el)
    {
        Variable& variable = variables[i];

        variable.name = read_xml_string(el, "Name");
        variable.set_scaler(read_xml_string(el, "Scaler"));
        variable.set_role(read_xml_string(el, "Role"));
        variable.set_type(read_xml_string(el, "Type"));

        if (variable.type == VariableType::Categorical || variable.type == VariableType::Binary)
        {
            const XMLElement* categories_element = el->FirstChildElement("Categories");

            if (categories_element)
                variable.categories = get_tokens(read_xml_string(el, "Categories"), ";");
            else if (variable.type == VariableType::Binary)
                variable.categories = { "0", "1" };
            else
                throw runtime_error("Categorical Variable Element is nullptr: Categories");
        }
    });
}

void Dataset::samples_from_XML(const XMLElement *samples_element)
{
    if(!samples_element)
        throw runtime_error("Samples element is nullptr.\n");

    const Index samples_number = read_xml_index(samples_element, "SamplesNumber");

    if (has_sample_ids)
    {
        const string separator_string = get_separator_string();
        sample_ids = get_tokens(read_xml_string(samples_element, "SamplesId"), separator_string);
    }

    if (variables.size() != 0)
    {
        const vector<vector<Index>> all_feature_indices = get_feature_indices();

        const auto& last_indices = all_feature_indices.back();
        data.resize(samples_number, last_indices.back() + 1);
        data.setZero();

        sample_roles.resize(samples_number);
        set_sample_roles(get_tokens(read_xml_string(samples_element, "SampleRoles"), " "));
    }
    else
        data.resize(0, 0);
}

void Dataset::missing_values_from_XML(const XMLElement *missing_values_element)
{
    if(!missing_values_element)
        throw runtime_error("Missing values element is nullptr.\n");

    missing_values_number = read_xml_index(missing_values_element, "MissingValuesNumber");

    if(missing_values_number > 0)
    {
        set_missing_values_method(read_xml_string(missing_values_element, "MissingValuesMethod"));

        const string variables_string = read_xml_string_fallback(missing_values_element,
            {"VariablesMissingValuesNumber", "RawVariablesMissingValuesNumber"});

        const vector<string> tokens = get_tokens(variables_string, " ");

        variables_missing_values_number.resize(tokens.size());
        for(size_t i = 0; i < tokens.size(); ++i)
            if(!tokens[i].empty())
                variables_missing_values_number(i) = stoi(tokens[i]);

        rows_missing_values_number = stol(read_xml_string_fallback(missing_values_element,
            {"SamplesMissingValuesNumber", "RowsMissingValuesNumber"}));
    }
}
void Dataset::preview_data_from_XML(const XMLElement *preview_data_element)
{
    if(!preview_data_element)
        throw runtime_error("Preview data element is nullptr.\n ");

    const Index preview_size = read_xml_index(preview_data_element, "PreviewSize");

    if(preview_size > 0)
    {
        data_file_preview.resize(preview_size);

        for_xml_items(preview_data_element, "Row", preview_size, [&](Index i, const XMLElement* row)
        {
            if(row->GetText())
                data_file_preview[i] = get_tokens(row->GetText(), ",");
        });
    }
}

void Dataset::from_XML(const XMLDocument& data_set_document)
{
    const XMLElement* root = get_xml_root(data_set_document, "Dataset");

    const XMLElement* src = require_xml_element(root, "DataSource");

    set_data_path(read_xml_string(src, "Path"));
    set_separator_name(read_xml_string(src, "Separator"));
    set_has_header(read_xml_bool(src, "HasHeader"));
    set_has_ids(read_xml_bool(src, "HasSamplesId"));
    set_missing_values_label(read_xml_string(src, "MissingValuesLabel"));
    set_codification(read_xml_string(src, "Codification"));

    variables_from_XML(require_xml_element(root, "Variables"));
    samples_from_XML(require_xml_element(root, "Samples"));
    missing_values_from_XML(require_xml_element(root, "MissingValues"));
    preview_data_from_XML(require_xml_element(root, "PreviewData"));

    set_display(read_xml_bool(root, "Display"));

    input_shape = { get_features_number("Input") };
    target_shape = { get_features_number("Target") };
}

void Dataset::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if(!file.is_open())
        return;

    XMLPrinter document;

    to_XML(document);

    file << document.CStr();
}

void Dataset::load(const filesystem::path& file_name)
{
    from_XML(load_xml_file(file_name));
}

void Dataset::save_data() const
{
    ofstream file(data_path);

    if(!file.is_open())
        throw runtime_error("Cannot open matrix data file: " + data_path.string() + "\n");

    file.precision(20);

    const Index samples_number = get_samples_number();
    const Index features_number = get_features_number();

    const vector<string> feature_names = get_feature_names();

    const string separator_string = get_separator_string();

    if (has_sample_ids)
        file << "id" << separator_string;

    for(Index j = 0; j < features_number; j++)
    {
        file << feature_names[j];

        if (j != features_number - 1)
            file << separator_string;
    }

    file << endl;

    for(Index i = 0; i < samples_number; i++)
    {
        if (has_sample_ids)
            file << sample_ids[i] << separator_string;

        for(Index j = 0; j < features_number; j++)
        {
            file << data(i, j);

            if (j != features_number - 1)
                file << separator_string;
        }

        file << endl;
    }

    file.close();
}

void Dataset::save_data_binary(const filesystem::path& binary_data_file_name) const
{
    ofstream file(binary_data_file_name, ios::binary);

    if(!file.is_open())
        throw runtime_error("Cannot open data binary file.");

    // Write data

    streamsize size = sizeof(Index);

    Index columns_number = data.cols();
    Index rows_number = data.rows();

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

    if(!file.is_open())
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

VectorI Dataset::calculate_target_distribution() const
{
    const Index samples_number = get_samples_number();
    const Index targets_number = get_features_number("Target");
    const vector<Index> target_feature_indices = get_feature_indices("Target");

    VectorI class_distribution;

    if (targets_number == 1)
    {
        class_distribution.resize(2);

        const Index target_index = target_feature_indices[0];

        Index positives = 0;
        Index negatives = 0;

        for(Index sample_index = 0; sample_index < samples_number; sample_index++){
            if(!isnan(data(sample_index, target_index)))
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

        for(Index i = 0; i < samples_number; i++)
        {
            if (get_sample_role(i) == "None")
                continue;

            for(Index j = 0; j < targets_number; j++)
            {
                if (isnan(data(i, target_feature_indices[j])))
                    continue;

                if (data(i, target_feature_indices[j]) > type(0.5))
                    class_distribution(j)++;
            }
        }
    }

    return class_distribution;
}

vector<vector<Index>> Dataset::calculate_Tukey_outliers(const type cleaning_parameter, bool replace_with_nan)
{
    const Index samples_number = get_used_samples_number();
    const vector<Index> sample_indices = get_used_sample_indices();

    const Index variables_number = get_variables_number();
    const Index used_variables_number = get_used_variables_number();

    vector<vector<Index>> return_values(2);

    return_values[0].resize(samples_number, 0);
    return_values[1].resize(used_variables_number, 0);

    const vector<BoxPlot> box_plots = calculate_variables_box_plots();

    Index feature_index = 0;
    Index used_feature_index = 0;

    for(Index i = 0; i < variables_number; i++)
    {
        const Variable& variable = variables[i];

        if (variable.role == "None"
        && variable.type == VariableType::Categorical)
        {
            feature_index += variable.get_categories_number();
            continue;
        }
        else if (variable.role == "None")
        {
            feature_index++;
            continue;
        }

        if (variable.type == VariableType::Categorical)
        {
            feature_index += variable.get_categories_number();
            used_feature_index++;
            continue;
        }
        else if (variable.type == VariableType::Binary
                 || variable.type == VariableType::DateTime)
        {
            feature_index++;
            used_feature_index++;
            continue;
        }
        else // Numeric
        {
            const type interquartile_range = box_plots[i].third_quartile - box_plots[i].first_quartile;

            if (interquartile_range < EPSILON)
            {
                feature_index++;
                used_feature_index++;
                continue;
            }

            Index variables_outliers = 0;

            for(Index j = 0; j < samples_number; j++)
            {
                const VectorR sample = get_sample_data(sample_indices[Index(j)]);

                if (sample(feature_index) < box_plots[i].first_quartile - cleaning_parameter * interquartile_range
                || sample(feature_index) > box_plots[i].third_quartile + cleaning_parameter * interquartile_range)
                {
                    return_values[0][j] = 1;

                    variables_outliers++;

                    if (replace_with_nan)
                        data(sample_indices[Index(j)], feature_index) = numeric_limits<type>::quiet_NaN();
                }
            }

            return_values[1][used_feature_index] = variables_outliers;

            feature_index++;
            used_feature_index++;
        }
    }

    return return_values;
}

vector<vector<Index>> Dataset::replace_Tukey_outliers_with_NaN(const type cleaning_parameter)
{
    return calculate_Tukey_outliers(cleaning_parameter, true);
}

void Dataset::unuse_Tukey_outliers(const type cleaning_parameter)
{
    const vector<vector<Index>> outliers_indices = calculate_Tukey_outliers(cleaning_parameter);

    const vector<Index> outliers_samples = get_elements_greater_than(outliers_indices, 0);

    set_sample_roles(outliers_samples, "None");
}

void Dataset::set_data_rosenbrock()
{
    const Index samples_number = get_samples_number();
    const Index features_number = get_features_number();

    set_data_random();

#pragma omp parallel for

    for(Index i = 0; i < samples_number; i++)
    {
        type rosenbrock(0);

        for(Index j = 0; j < features_number - 1; j++)
        {
            const type value = data(i, j);
            const type next_value = data(i, j + 1);

            rosenbrock += (type(1) - value) * (type(1) - value) + type(100) * (next_value - value * value) * (next_value - value * value);
        }

        data(i, features_number - 1) = rosenbrock;
    }
}

void Dataset::set_data_binary_classification()
{
    const Index samples_number = get_samples_number();
    const Index features_number = get_features_number();

    set_data_random();

#pragma omp parallel for
    for(Index i = 0; i < samples_number; i++)
        data(i, features_number - 1) = type(random_bool());
}

void Dataset::impute_missing_values_unuse()
{
    const Index samples_number = get_samples_number();

#pragma omp parallel for

    for(Index i = 0; i < samples_number; i++)
        if (has_nan_row(i))
            set_sample_role(i, "None");
}

void Dataset::impute_missing_values_statistic(const MissingValuesMethod& method)
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_feature_indices = get_used_feature_indices();
    const vector<Index> input_feature_indices = get_feature_indices("Input");
    const vector<Index> target_feature_indices = get_feature_indices("Target");

    if (used_sample_indices.empty() || used_feature_indices.empty())
        return;

    const VectorR replacements = (method == MissingValuesMethod::Mean)
        ? mean(data, used_sample_indices, used_feature_indices)
        : median(data, used_sample_indices, used_feature_indices);

    const Index samples_number = used_sample_indices.size();
    const Index features_number = used_feature_indices.size();
    const Index target_features_number = target_feature_indices.size();

    for(Index j = 0; j < features_number - target_features_number; j++)
    {
        const Index current_variable = input_feature_indices[j];

        for(Index i = 0; i < samples_number; i++)
        {
            const Index current_sample = used_sample_indices[i];

            if (isnan(data(current_sample, current_variable)))
                data(current_sample, current_variable) = replacements(j);
        }
    }

    for(Index j = 0; j < target_features_number; j++)
    {
        const Index current_variable = target_feature_indices[j];

        for(Index i = 0; i < samples_number; i++)
        {
            const Index current_sample = used_sample_indices[i];

            if (isnan(data(current_sample, current_variable)))
                set_sample_role(i, "None");
        }
    }
}

void Dataset::impute_missing_values_interpolate()
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_feature_indices = get_used_feature_indices();
    const vector<Index> input_feature_indices = get_feature_indices("Input");
    const vector<Index> target_feature_indices = get_feature_indices("Target");

    const Index samples_number = used_sample_indices.size();
    const Index features_number = used_feature_indices.size();
    const Index target_features_number = target_feature_indices.size();

    Index current_variable;
    Index current_sample;

    for(Index j = 0; j < features_number - target_features_number; j++)
    {
        current_variable = input_feature_indices[j];

        for(Index i = 0; i < samples_number; i++)
        {
            current_sample = used_sample_indices[i];

            if (isnan(data(current_sample, current_variable)))
            {
                Index x1 = 0;
                Index x2 = 0;
                Index y1 = 0;
                Index y2 = 0;
                type x = type(0);
                type y = type(0);

                for(Index k = i - 1; k >= 0; k--)
                {
                    if (isnan(data(used_sample_indices[k], current_variable))) continue;

                    x1 = used_sample_indices[k];
                    y1 = data(x1, current_variable);
                    break;
                }

                for(Index k = i + 1; k < samples_number; k++)
                {
                    if (isnan(data(used_sample_indices[k], current_variable))) continue;

                    x2 = used_sample_indices[k];
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

    for(Index j = 0; j < target_features_number; j++)
    {
        current_variable = target_feature_indices[j];

        for(Index i = 0; i < samples_number; i++)
        {
            current_sample = used_sample_indices[i];

            if (isnan(data(current_sample, current_variable)))
                set_sample_role(i, "None");
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
    case MissingValuesMethod::Median:
        impute_missing_values_statistic(missing_values_method);
        break;

    case MissingValuesMethod::Interpolation:
        impute_missing_values_interpolate();
        break;
    }

    missing_values_number = count_nan();
}

void Dataset::calculate_missing_values_statistics()
{
    missing_values_number = count_nan();
    variables_missing_values_number = count_nans_per_variable();
    rows_missing_values_number = count_rows_with_nan();
}

void Dataset::infer_column_types(const vector<vector<string>>& sample_rows)
{
    const Index variables_number = variables.size();
    const size_t total_rows = sample_rows.size();

    if(total_rows == 0) return;

    vector<size_t> row_indices(total_rows);
    iota(row_indices.begin(), row_indices.end(), 0);

    shuffle_vector(row_indices);

    const size_t rows_to_check = min(size_t(100), total_rows);

    for(Index col_idx = 0; col_idx < variables_number; ++col_idx)
    {
        Variable& variable = variables[col_idx];
        variable.type = VariableType::None;

        for(size_t i = 0; i < rows_to_check; ++i)
        {
            const size_t row_idx = row_indices[i];

            const size_t token_idx = has_sample_ids ? col_idx + 1 : col_idx;
            if(token_idx >= sample_rows[row_idx].size()) continue;

            const string& token = sample_rows[row_idx][token_idx];

            if(token.empty() || token == missing_values_label) continue;

            if(variable.type == VariableType::Categorical) break;

            if(is_numeric_string(token))
            {
                if(variable.type == VariableType::None)
                    variable.type = VariableType::Numeric;
            }
            else if (is_date_time_string(token))
            {
                if(variable.type == VariableType::None)
                    variable.type = VariableType::DateTime;
            }
            else
                variable.type = VariableType::Categorical;
        }

        if(variable.type == VariableType::None)
            variable.type = VariableType::Numeric;
    }

    for(Index col_idx = 0; col_idx < variables_number; ++col_idx)
    {
        if(variables[col_idx].type == VariableType::Categorical)
        {
            std::set<string> unique_categories;
            for(const vector<string>& row : sample_rows)
            {
                const size_t token_idx = has_sample_ids ? col_idx + 1 : col_idx;
                if(token_idx < row.size() && !row[token_idx].empty() && row[token_idx] != missing_values_label)
                    unique_categories.insert(row[token_idx]);
            }
            variables[col_idx].categories.assign(unique_categories.begin(), unique_categories.end());
        }
    }
}

DateFormat Dataset::infer_dataset_date_format(const vector<Variable>& variables,
                                              const vector<vector<string>>& sample_rows,
                                              bool has_sample_ids,
                                              const string& missing_values_label)
{
    for(size_t col_idx = 0; col_idx < variables.size(); ++col_idx)
    {
        if(variables[col_idx].type != VariableType::DateTime)
            continue;

        for(const vector<string>& row : sample_rows)
        {
            const size_t token_idx = has_sample_ids ? col_idx + 1 : col_idx;

            if(token_idx >= row.size())
                continue;

            const string& token = row[token_idx];

            if(token.empty() || token == missing_values_label)
                continue;

            smatch date_parts;
            if(regex_match(token, date_parts, regex(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}).*)")))
            {
                const int part1 = stoi(date_parts[1].str());
                const int part2 = stoi(date_parts[2].str());

                if(part1 > 12)
                    return DMY;
                if(part2 > 12)
                    return MDY;
            }
        }
    }

    return AUTO;
}

void Dataset::read_csv()
{
    if(data_path.empty())
        throw runtime_error("Data path is empty.\n");

    ifstream file(data_path, ios::binary);

    if(!file.is_open())
        throw runtime_error("Error: Cannot open file " + data_path.string() + "\n");

    // BOM

    char bom[3] = {0};
    file.read(bom, 3);

    if(static_cast<unsigned char>(bom[0]) != 0xEF
        || static_cast<unsigned char>(bom[1]) != 0xBB
        || static_cast<unsigned char>(bom[2]) != 0xBF)
        file.seekg(0);

    // Read file

    vector<vector<string>> raw_file_content;
    string line;
    const string separator_string = get_separator_string();

    while(getline(file, line))
    {
        if(!line.empty() && line.back() == '\r')
            line.pop_back();

        prepare_line(line);

        if (line.empty())
            continue;

        check_separators(line);

        raw_file_content.push_back(get_tokens(line, separator_string));
    }

    file.close();

    if(raw_file_content.empty())
        throw runtime_error("File " + data_path.string() + " is empty or contains no valid data rows.");

    read_data_file_preview(raw_file_content);

    vector<string> header_tokens = raw_file_content[0];
    if(has_header)
    {
        if(has_numbers(header_tokens))
            throw runtime_error("Error: Some header names are numeric.");

        raw_file_content.erase(raw_file_content.begin());
    }

    if(raw_file_content.empty())
        throw runtime_error("Data file only contains a header.");

    // Id detection

    const Index samples_number = raw_file_content.size();

    if(!has_sample_ids && samples_number > 0)
    {
        std::unordered_set<string> unique_elements;

        bool possible_id = true;
        bool is_numeric_column = true;
        bool is_date_column = true;
        Index date_check_count = 0;
        const Index max_date_checks = 20;

        for(const vector<string>& row : raw_file_content)
        {
            if(row.empty())
                continue;

            const string& token = row[0];

            if(!unique_elements.insert(token).second)
            {
                possible_id = false;
                break;
            }

            if(is_numeric_column && !token.empty() && token != missing_values_label)
                if(!is_numeric_string(token))
                    is_numeric_column = false;

            if(is_date_column && date_check_count < max_date_checks && !token.empty() && token != missing_values_label)
            {
                if(!is_date_time_string(token))
                    is_date_column = false;

                date_check_count++;
            }
        }

        if(is_date_column && date_check_count > 0)
            possible_id = false;

        if(possible_id && !is_numeric_column && unique_elements.size() == static_cast<size_t>(samples_number))
            has_sample_ids = true;
    }

    // Variables

    const size_t columns_number = header_tokens.size();
    const Index variables_number = has_sample_ids ? columns_number - 1 : columns_number;
    variables.resize(variables_number);

    if(has_header)
        if(has_sample_ids)
            for(Index i = 0; i < variables_number; i++) variables[i].name = header_tokens[i + 1];
        else
            set_variable_names(header_tokens);
    else
        set_default_variable_names();

    infer_column_types(raw_file_content);

    const DateFormat date_format = infer_dataset_date_format(variables, raw_file_content, has_sample_ids, missing_values_label);

    for(Variable& variable : variables)
        if(variable.type == VariableType::Categorical && variable.get_categories_number() == 2)
            variable.type = VariableType::Binary;

    // Samples data

    sample_roles.resize(samples_number);
    sample_ids.resize(samples_number);

    const vector<vector<Index>> all_feature_indices = get_feature_indices();
    const Index total_numeric_columns = all_feature_indices.empty() ? 0 : all_feature_indices.back().back() + 1;

    data.resize(samples_number, total_numeric_columns);
    data.setZero();

    rows_missing_values_number = 0;
    missing_values_number = 0;

    variables_missing_values_number.resize(variables_number);
    variables_missing_values_number.setZero();

    // Build category lookup maps for O(1) access during data loading
    vector<unordered_map<string, Index>> category_maps(variables_number);
    for(Index v = 0; v < variables_number; v++)
        if(variables[v].type == VariableType::Categorical)
            for(Index c = 0; c < static_cast<Index>(variables[v].categories.size()); c++)
                category_maps[v][variables[v].categories[c]] = c;

    for(Index sample_index = 0; sample_index < samples_number; ++sample_index)
    {
        const vector<string>& tokens = raw_file_content[sample_index];

        if(has_missing_values(tokens))
        {
            rows_missing_values_number++;
            for(size_t i = (has_sample_ids ? 1 : 0); i < tokens.size(); i++)
            {
                if(tokens[i].empty() || tokens[i] == missing_values_label)
                {
                    missing_values_number++;
                    variables_missing_values_number(has_sample_ids ? i - 1 : i)++;
                }
            }
        }

        if(has_sample_ids)
            sample_ids[sample_index] = tokens[0];

        for(Index variable_index = 0; variable_index < variables_number; variable_index++)
        {
            const Variable& variable = variables[variable_index];
            const string& token = has_sample_ids ? tokens[variable_index + 1] : tokens[variable_index];
            const vector<Index>& feature_indices = all_feature_indices[variable_index];

            switch(variable.type)
            {
            case VariableType::Numeric:
                data(sample_index, feature_indices[0]) = (token.empty() || token == missing_values_label) ? NAN : stof(token);
                break;
            case VariableType::DateTime:
                if(token.empty() || token == missing_values_label)
                    data(sample_index, feature_indices[0]) = NAN;
                else
                {
                    const time_t timestamp = date_to_timestamp(token, gmt, date_format);

                    if(timestamp == -1)
                        throw runtime_error("Date format is unsupported or date is prior to 1970.");
                    else
                        data(sample_index, feature_indices[0]) = timestamp;
                }
                break;
            case VariableType::Categorical:
                if(token.empty() || token == missing_values_label)
                    for(const Index cat_idx : feature_indices)
                        data(sample_index, cat_idx) = NAN;
                else
                {
                    auto it = category_maps[variable_index].find(token);
                    if(it != category_maps[variable_index].end())
                        data(sample_index, feature_indices[it->second]) = 1;
                }
                break;
            case VariableType::Binary:
                if(const bool is_positive = contains(positive_words, token); is_positive || contains(negative_words, token))
                    data(sample_index, feature_indices[0]) = is_positive ? 1 : 0;
                else
                {
                    const vector<string>& categories = variable.categories;

                    if(token.empty() || token == missing_values_label)
                        data(sample_index, feature_indices[0]) = NAN;
                    else if(!categories.empty() && token == categories[0])
                        data(sample_index, feature_indices[0]) = 0;
                    else if(categories.size() > 1 && token == categories[1])
                        data(sample_index, feature_indices[0]) = 1;
                    else
                        //from_chars(token.data(), token.data() + token.size(), data(sample_index, feature_indices[0])); // AFTER
                        data(sample_index, feature_indices[0]) = stof(token);
                }
                break;
            default:
                break;
            }
        }
    }

    infer_variable_types_from_data();
    split_samples_random();
}

void Dataset::read_data_file_preview(const vector<vector<string>>& all_rows)
{
    if (all_rows.empty())
        return;

    const Index num_first_rows_to_show = 3;

    data_file_preview.clear();

    for(Index i = 0; i < Index(min(static_cast<size_t>(num_first_rows_to_show), all_rows.size())); ++i)
        data_file_preview.push_back(all_rows[i]);

    if (all_rows.size() > num_first_rows_to_show)
        data_file_preview.push_back(all_rows.back());
    else if (all_rows.empty() && data_file_preview.size() < num_first_rows_to_show +1 )
        while(data_file_preview.size() < num_first_rows_to_show +1)
            data_file_preview.push_back(vector<string>());
}

void Dataset::check_separators(const string& line) const
{
    const string separator_string = get_separator_string();
    const string separator_name = get_separator_name();

    if (line.find(separator_string) == string::npos)
    {
        bool has_any_separator = false;

        for(const auto& [sep, str, name] : separator_map)
            if (line.find(str) != string::npos) { has_any_separator = true; break; }

        if (has_any_separator)
            throw runtime_error("Error: Separator '" + separator_string + "' not found in line " + line + ".\n");

        return;
    }

    for(const auto& [sep, str, name] : separator_map)
        if (sep != separator && line.find(str) != string::npos)
            throw runtime_error("Error: Found " + name + " ('" + str + "') in data file "
                                + data_path.string() + ", but separator is " + separator_name + " ('" + separator_string + "').");
}

void Dataset::fill_inputs(const vector<Index>& sample_indices,
                          const vector<Index>& input_indices,
                          type* input_data, bool parallelize) const
{
    fill_tensor_data(data, sample_indices, input_indices, input_data, parallelize);
}

void Dataset::fill_decoder(const vector<Index>& sample_indices,
                           const vector<Index>& decoder_indices,
                           type* decoder_data, bool parallelize) const
{
    fill_tensor_data(data, sample_indices, decoder_indices, decoder_data, parallelize);
}

void Dataset::fill_targets(const vector<Index>& sample_indices,
                           const vector<Index>& target_indices,
                           type* target_data, bool parallelize) const
{
    fill_tensor_data(data, sample_indices, target_indices, target_data, parallelize);
}

bool Dataset::has_binary_variables() const
{
    return any_of(variables.begin(), variables.end(),
                  [](const Variable& variable) { return variable.type == VariableType::Binary; });
}

bool Dataset::has_categorical_variables() const
{
    return any_of(variables.begin(), variables.end(),
                  [](const Variable& variable) { return variable.type == VariableType::Categorical; });
}

bool Dataset::has_binary_or_categorical_variables() const
{
    return any_of(variables.begin(), variables.end(),[](const Variable& v) {
        return v.type == VariableType::Binary || v.type == VariableType::Categorical;
    });
}

bool Dataset::has_time_variable() const
{
    return any_of(variables.begin(), variables.end(),
                  [](const Variable& variable) { return variable.role == "Time"; });
}

bool Dataset::has_validation() const
{
    return get_samples_number("Validation") != 0;
}

bool Dataset::has_missing_values(const vector<string>& row) const
{
    for(size_t i = 0; i < row.size(); i++)
        if (row[i].empty() || row[i] == missing_values_label)
            return true;

    return false;
}

VectorI Dataset::count_nans_per_variable() const
{
    return data.array().isNaN().cast<Index>().colwise().sum();
}

Index Dataset::count_variables_with_nan() const
{
    return (count_nans_per_variable().array() > 0).count();
}

Index Dataset::count_rows_with_nan() const
{
    return data.array().isNaN().rowwise().any().count();
}

Index Dataset::count_nan() const
{
    return data.array().isNaN().count();
}

vector<vector<Index>> Dataset::split_samples(const vector<Index>& sample_indices, Index new_batch_size) const
{
    const Index samples_number = sample_indices.size();

    if(samples_number == 0) return {};

    Index batch_size = new_batch_size;
    Index batches_number;

    if(batch_size <= 0 || samples_number < batch_size)
    {
        batches_number = 1;
        batch_size = samples_number;
    }
    else
        batches_number = (samples_number + batch_size - 1) / batch_size;

    vector<vector<Index>> batches(batches_number);

    #pragma omp parallel for
    for(Index i = 0; i < batches_number; i++)
    {
        const Index start_index = i * batch_size;
        const Index end_index = min(start_index + batch_size, samples_number);

        batches[i].assign(sample_indices.begin() + start_index,
                          sample_indices.begin() + end_index);
    }

    return batches;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
