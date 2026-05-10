//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "dataset.h"
#include "random_utilities.h"
#include <regex>

namespace opennn
{

VectorI Dataset::get_sample_role_numbers() const
{
    VectorI count = VectorI::Zero(4);

    const Index samples_number = get_samples_number();

    for (Index i = 0; i < samples_number; ++i)
    {
        switch (sample_roles[i])
        {
        case SampleRole::Training:   count[0]++; break;
        case SampleRole::Validation: count[1]++; break;
        case SampleRole::Testing:    count[2]++; break;
        case SampleRole::None:       count[3]++; break;
        }
    }

    return count;
}

vector<Index> Dataset::get_sample_indices(const string& sample_role) const
{
    const SampleRole role_type = string_to_sample_role(sample_role);
    const Index samples_number = get_samples_number();

    vector<Index> indices;
    indices.reserve(get_samples_number(sample_role));

    for (Index i = 0; i < samples_number; ++i)
        if (sample_roles[i] == role_type)
            indices.push_back(i);

    return indices;
}

vector<Index> Dataset::get_used_sample_indices() const
{
    const Index samples_number = get_samples_number();

    vector<Index> used_indices;
    used_indices.reserve(samples_number - get_samples_number("None"));

    for (Index i = 0; i < samples_number; ++i)
        if (sample_roles[i] != SampleRole::None)
            used_indices.push_back(i);

    return used_indices;
}

vector<Index> Dataset::get_sample_roles_vector() const
{
    const Index samples_number = get_samples_number();

    vector<Index> sample_roles_vector(samples_number);

#pragma omp parallel for
    for (Index i = 0; i < samples_number; ++i)
        sample_roles_vector[i] = static_cast<Index>(sample_roles[i]);

    return sample_roles_vector;
}

void Dataset::get_batches(const vector<Index>& sample_indices,
                          Index batch_size,
                          bool shuffle,
                          vector<vector<Index>>& batches) const
{
    const Index samples_number = sample_indices.size();

    if (samples_number == 0) { batches.clear(); return; }

    if (batch_size <= 0 || batch_size > samples_number)
        batch_size = samples_number;

    const Index batches_number = samples_number / batch_size;

    if (ssize(batches) != batches_number)
        batches.resize(batches_number);

    vector<Index> shuffled_indices;

    if (shuffle)
    {
        shuffled_indices = sample_indices;
        shuffle_vector_blocks(shuffled_indices);
    }

    const vector<Index>& indices = shuffle ? shuffled_indices : sample_indices;

    #pragma omp parallel for if (batches_number > 64)
    for (Index i = 0; i < batches_number; ++i)
    {
        const Index start = i * batch_size;
        const Index end = min(start + batch_size, samples_number);

        batches[i].assign(indices.begin() + start, indices.begin() + end);
    }
}

Index Dataset::get_samples_number(const string& sample_role) const
{
    const SampleRole role_type = string_to_sample_role(sample_role);
    return count(sample_roles.begin(), sample_roles.end(), role_type);
}

Index Dataset::get_used_samples_number() const
{
    const Index samples_number = get_samples_number();
    const Index unused_samples_number = get_samples_number("None");

    return samples_number - unused_samples_number;
}

void Dataset::set_sample_roles(const string& sample_role)
{
    const SampleRole role_type = string_to_sample_role(sample_role);
    fill(sample_roles.begin(), sample_roles.end(), role_type);
}

void Dataset::set_sample_role(const Index index, const string& new_role)
{
    sample_roles[index] = string_to_sample_role(new_role);
}

void Dataset::set_sample_roles(const vector<string>& new_roles)
{
    transform(new_roles.begin(), new_roles.end(), sample_roles.begin(), string_to_sample_role);
}

void Dataset::set_sample_roles(const vector<Index>& indices, const string& sample_role)
{
    const SampleRole role_type = string_to_sample_role(sample_role);
    for (const auto& i : indices)
        sample_roles[i] = role_type;
}

void Dataset::split_samples(const float training_samples_ratio,
                            float validation_samples_ratio,
                            float testing_samples_ratio,
                            bool shuffle)
{
    const Index used_samples_number = get_used_samples_number();

    if (used_samples_number == 0) return;

    const float total_ratio = training_samples_ratio + validation_samples_ratio + testing_samples_ratio;

    const Index validation_samples_number = Index((validation_samples_ratio * used_samples_number) / total_ratio);
    const Index testing_samples_number = Index((testing_samples_ratio * used_samples_number) / total_ratio);
    const Index training_samples_number = used_samples_number - validation_samples_number - testing_samples_number;

    if (training_samples_number + validation_samples_number + testing_samples_number != used_samples_number)
        throw runtime_error("Sum of numbers of training, selection and testing samples is not equal to number of used samples.\n");

    const Index samples_number = get_samples_number();

    vector<Index> indices(samples_number);
    iota(indices.begin(), indices.end(), 0);

    if (shuffle)
        shuffle_vector(indices);

    auto assign_role = [this, &indices](SampleRole role, Index count, Index& i)
    {
        Index assigned = 0;

        while (assigned < count)
        {
            const Index idx = indices[i++];

            if (sample_roles[idx] != SampleRole::None)
            {
                sample_roles[idx] = role;
                ++assigned;
            }
        }
    };

    Index index = 0;

    assign_role(SampleRole::Training, training_samples_number, index);
    assign_role(SampleRole::Validation, validation_samples_number, index);
    assign_role(SampleRole::Testing, testing_samples_number, index);
}

void Dataset::split_samples_random(const float training_ratio,
                                   float validation_ratio,
                                   float testing_ratio)
{
    split_samples(training_ratio, validation_ratio, testing_ratio, true);
}

void Dataset::split_samples_sequential(const float training_ratio,
                                       float validation_ratio,
                                       float testing_ratio)
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

    for (Index i = variables_number - 1; i >= 0; i--)
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

    for (const Variable& variable : variables)
    {
        if (!variable.is_used())
            continue;

        feature_dimensions[i] = variable.feature_count();

        ++i;
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

    for (Index i = variables_number - 1; i >= 0; i--)
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

        if (!target)
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

    for (Index i = 0; i < variables_number; ++i)
        variables[i].name = "variable_" + to_string(1 + i);
}

vector<string> Dataset::get_feature_names() const
{
    vector<string> feature_names;
    feature_names.reserve(variables.size());

    for (const auto& variable : variables)
    {
        const vector<string> names = variable.get_names();

        feature_names.insert(feature_names.end(), names.begin(), names.end());
    }
    return feature_names;
}

vector<string> Dataset::get_feature_names(const string& variable_role) const
{
    const auto vars = get_variables(variable_role);

    vector<string> feature_names;
    feature_names.reserve(vars.size());

    for (const auto& variable : vars)
    {
        const vector<string> names = variable.get_names();

        feature_names.insert(feature_names.end(), names.begin(), names.end());
    }

    return feature_names;
}

Shape Dataset::get_shape(const string& variable_role) const
{
    const VariableRole role = string_to_variable_role(variable_role);

    if (role == VariableRole::Input)   return input_shape;
    if (role == VariableRole::Target)  return target_shape;
    if (role == VariableRole::Decoder) return decoder_shape;

    throw invalid_argument("get_shape: Invalid variable role string: " + variable_role);
}

void Dataset::set_shape(const string& variable_role, const Shape& new_shape)
{
    const VariableRole role = string_to_variable_role(variable_role);

    if (role == VariableRole::Input)
        input_shape = new_shape;
    else if (role == VariableRole::Target)
        target_shape = new_shape;
    else if (role == VariableRole::Decoder)
        decoder_shape = new_shape;
    else
        throw invalid_argument("set_shape: Invalid variable role string: " + variable_role);
}

Index Dataset::get_used_features_number() const
{
    return get_features_number() - get_features_number("None") - get_features_number("Time");
}

vector<Index> Dataset::get_feature_indices(const string& variable_role) const
{
    const VariableRole role_type = string_to_variable_role(variable_role);
    const Index this_features_number = get_features_number(variable_role);
    vector<Index> this_feature_indices(this_features_number);

    Index feature_index = 0;
    Index this_feature_index = 0;

    for (const Variable& variable : variables)
    {
        const Index count = variable.feature_count();

        if (!role_matches(variable.role, role_type))
        {
            feature_index += count;
            continue;
        }

        for (Index j = 0; j < count; ++j)
            this_feature_indices[this_feature_index++] = feature_index++;
    }

    return this_feature_indices;
}

vector<Index> Dataset::get_variable_indices(const string& variable_role) const
{
    const VariableRole role_type = string_to_variable_role(variable_role);

    vector<Index> indices;
    indices.reserve(get_variables_number(variable_role));

    for (size_t i = 0; i < variables.size(); ++i)
        if (role_matches(variables[i].role, role_type))
            indices.push_back(i);

    return indices;
}

vector<Index> Dataset::get_used_variables_indices() const
{
    vector<Index> used_indices;
    used_indices.reserve(get_used_variables_number());

    for (size_t i = 0; i < variables.size(); ++i)
        if (variables[i].is_used())
            used_indices.push_back(i);

    return used_indices;
}

vector<string> Dataset::get_feature_scalers(const string& variable_role) const
{
    const vector<Variable> role_variables = get_variables(variable_role);

    vector<string> scalers;
    scalers.reserve(get_features_number(variable_role));

    for (const Variable& var : role_variables)
    {
        const Index count = var.feature_count();

        for (Index j = 0; j < count; ++j)
            scalers.push_back(scaler_method_to_string(var.scaler));
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
    const VariableRole role_type = string_to_variable_role(variable_role);

    vector<string> names;
    names.reserve(get_variables_number(variable_role));

    for (const Variable& variable : variables)
        if (role_matches(variable.role, role_type))
            names.push_back(variable.name);

    return names;
}

Index Dataset::get_variables_number(const string& variable_role) const
{
    const VariableRole role_type = string_to_variable_role(variable_role);

    Index count = 0;

    for (const Variable& variable : variables)
        if (role_matches(variable.role, role_type))
            ++count;

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

    const VariableRole role_type = string_to_variable_role(variable_role);

    copy_if(variables.begin(), variables.end(), back_inserter(this_variables),
            [role_type](const Variable& var) { return role_matches(var.role, role_type); });

    return this_variables;
}

Index Dataset::get_features_number() const
{
    return accumulate(variables.begin(), variables.end(), 0,
                      [](Index sum, const Variable& var) { return sum + var.feature_count(); });
}

Index Dataset::get_features_number(const string& variable_role) const
{
    const VariableRole role_type = string_to_variable_role(variable_role);

    Index count = 0;

    for (const Variable& variable : variables)
        if (role_matches(variable.role, role_type))
            count += variable.feature_count();

    return count;
}

vector<Index> Dataset::get_used_feature_indices() const
{
    const Index used_features_number = get_used_features_number();
    vector<Index> used_feature_indices(used_features_number);

    Index feature_index = 0;
    Index used_feature_index = 0;

    for (const Variable& variable : variables)
    {
        const Index count = variable.feature_count();

        if (variable.role == VariableRole::None || variable.role == VariableRole::Time)
        {
            feature_index += count;
            continue;
        }

        for (Index j = 0; j < count; ++j)
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

    for (size_t i = 0; i < new_variables_roles.size(); ++i)
        variables[i].set_role(new_variables_roles[i]);
}

void Dataset::set_variables(const string& variable_role)
{
    const Index variables_number = get_variables_number();

    for (Index i = 0; i < variables_number; ++i)
        set_variable_role(i, variable_role);
}

void Dataset::set_variable_indices(const vector<Index>& input_variables,
                                       const vector<Index>& target_variables)
{
    set_variables("None");

    for (const Index index : input_variables)
        set_variable_role(index, "Input");

    for (const Index index : target_variables)
        set_variable_role(index,
            variables[index].role == VariableRole::Input ? "InputTarget" : "Target");

    const Index input_dimensions_num = get_features_number("Input");
    const Index target_shape_num = get_features_number("Target");

    set_shape("Input", {input_dimensions_num});
    set_shape("Target", {target_shape_num});
}

void Dataset::set_input_variables_unused()
{
    const Index variables_number = get_variables_number();

    for (Index i = 0; i < variables_number; ++i)
        if (variables[i].role == VariableRole::Input)
            set_variable_role(i, "None");
}

void Dataset::set_variable_role(const Index index, const string& new_role)
{
    variables[index].set_role(new_role);
}

void Dataset::set_variable_role(const string& name, const string& new_role)
{
    set_variable_role(get_variable_index(name), new_role);
}

void Dataset::set_variable_type(const Index index, const VariableType& new_type)
{
    variables[index].type = new_type;
}

void Dataset::set_variable_type(const string& name, const VariableType& new_type)
{
    set_variable_type(get_variable_index(name), new_type);
}

void Dataset::set_variable_types(const VariableType& new_type)
{
    for (auto& variable : variables)
        variable.type = new_type;
}

void Dataset::set_feature_names(const vector<string>& new_variables_names)
{
    Index index = 0;

    for (Variable& variable : variables)
        if (variable.is_categorical())
            for (Index j = 0; j < variable.get_categories_number(); ++j)
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

    for (Index i = 0; i < variables_number; ++i)
        variables[i].name = get_trimmed(new_names[i]);
}

void Dataset::set_variable_roles(const string& variable_role)
{
    for (Variable& variable : variables)
        variable.set_role(
            (variable.type == VariableType::Constant || variable.type == VariableType::DateTime)
                ? "None"
                : variable_role);
}

void Dataset::set_variable_scalers(const string& scalers)
{
    const ScalerMethod method = string_to_scaler_method(scalers);
    for (Variable& variable : variables)
        variable.scaler = method;
}

void Dataset::set_variable_scalers(const vector<string>& new_scalers)
{
    const size_t variables_number = get_variables_number();

    if (new_scalers.size() != variables_number)
        throw runtime_error("Size of variable scalers(" + to_string(new_scalers.size()) + ") "
                            "has to be the same as variables numbers(" + to_string(variables_number) + ").\n");

    for (size_t i = 0; i < variables_number; ++i)
        variables[i].set_scaler(new_scalers[i]);
}

static const vector<pair<Dataset::MissingValuesMethod, string>> missing_values_method_map = {
    {Dataset::MissingValuesMethod::Unuse,         "Unuse"},
    {Dataset::MissingValuesMethod::Mean,          "Mean"},
    {Dataset::MissingValuesMethod::Median,        "Median"},
    {Dataset::MissingValuesMethod::Interpolation, "Interpolation"}
};

string Dataset::get_missing_values_method_string() const
{
    for (const auto& [method, name] : missing_values_method_map)
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
    for (const auto& [sep, str, name] : separator_map)
        if (sep == separator) return str;

    return string();
}

string Dataset::get_separator_name() const
{
    for (const auto& [sep, str, name] : separator_map)
        if (sep == separator) return name;

    return string();
}

static const vector<pair<Dataset::Codification, string>> codification_map = {
    {Dataset::Codification::UTF8,      "UTF-8"},
    {Dataset::Codification::SHIFT_JIS, "SHIFT_JIS"}
};

const string Dataset::get_codification_string() const
{
    for (const auto& [cod, name] : codification_map)
        if (cod == codification) return name;

    return "UTF-8";
}

Index Dataset::get_variable_index(const string& variable_name) const
{
    const Index variables_number = get_variables_number();

    for (Index i = 0; i < variables_number; ++i)
        if (variables[i].name == variable_name)
            return i;

    throw runtime_error("Cannot find " + variable_name + "\n");
}

Index Dataset::get_variable_index(const Index feature_index) const
{
    const Index variables_number = get_variables_number();

    Index total_variables_number = 0;

    for (Index i = 0; i < variables_number; ++i)
    {
        total_variables_number += variables[i].feature_count();

        if (feature_index + 1 <= total_variables_number)
            return i;
    }

    throw runtime_error("Cannot find variable index: " + to_string(feature_index) + ".\n");
}

vector<vector<Index>> Dataset::get_feature_indices() const
{
    const Index variables_number = get_variables_number();

    vector<vector<Index>> indices(variables_number);

    for (Index i = 0; i < variables_number; ++i)
        indices[i] = get_feature_indices(i);

    return indices;
}

vector<Index> Dataset::get_feature_indices(const Index variable_index) const
{
    Index index = 0;

    for (Index i = 0; i < variable_index; ++i)
        index += variables[i].feature_count();

    const Variable& variable = variables[variable_index];

    if (variable.type == VariableType::Categorical)
    {
        vector<Index> indices(variable.categories.size());
        iota(indices.begin(), indices.end(), index);

        return indices;
    }

    return vector<Index>(1, index);
}

void Dataset::set_default()
{
    has_header = false;

    has_sample_ids = false;

    separator = Separator::Semicolon;

    missing_values_label = "NA";

    set_default_variable_names();
}

void Dataset::set_separator_string(const string& new_separator_string)
{
    for (const auto& [sep, str, name] : separator_map)
        if (str == new_separator_string) { separator = sep; return; }

    throw runtime_error("Unknown separator: " + new_separator_string);
}

void Dataset::set_separator_name(const string& new_separator_name)
{
    for (const auto& [sep, str, name] : separator_map)
        if (name == new_separator_name) { separator = sep; return; }

    throw runtime_error("Unknown separator: " + new_separator_name + ".\n");
}

void Dataset::set_codification(const string& new_codification_string)
{
    for (const auto& [cod, name] : codification_map)
        if (name == new_codification_string) { codification = cod; return; }

    throw runtime_error("Unknown codification: " + new_codification_string + ".\n");
}

void Dataset::set_missing_values_method(const string& new_missing_values_method)
{
    for (const auto& [method, name] : missing_values_method_map)
        if (name == new_missing_values_method) { missing_values_method = method; return; }

    throw runtime_error("Unknown method type.\n");
}

void Dataset::set_default_variable_scalers()
{
    for (Variable& variable : variables)
        variable.scaler = (variable.type == VariableType::Numeric)
                                  ? ScalerMethod::MeanStandardDeviation
                                  : ScalerMethod::MinimumMaximum;
}

void Dataset::to_JSON(JsonWriter& printer) const
{
    printer.open_element("Dataset");

    printer.open_element("DataSource");
    write_json(printer, {
        {"FileType", "csv"},
        {"Path", data_path.string()},
        {"Separator", get_separator_name()},
        {"HasHeader", to_string(has_header)},
        {"HasSamplesId", to_string(has_sample_ids)},
        {"MissingValuesLabel", missing_values_label},
        {"Codification", get_codification_string()}
    });
    printer.close_element();

    variables_to_JSON(printer);
    samples_to_JSON(printer);
    missing_values_to_JSON(printer);
    preview_data_to_JSON(printer);

    add_json_field(printer, "Display", to_string(display));

    printer.close_element();
}

void Dataset::variables_to_JSON(JsonWriter &printer) const
{
    printer.open_element("Variables");
    add_json_field(printer, "VariablesNumber", to_string(get_variables_number()));

    printer.begin_array("Variable");
    for (Index i = 0; i < get_variables_number(); ++i)
    {
        printer.begin_array_object();
        variables[i].to_JSON(printer);
        printer.end_array_object();
    }
    printer.end_array();

    printer.close_element();
}

void Dataset::samples_to_JSON(JsonWriter &printer) const
{
    printer.open_element("Samples");

    add_json_field(printer, "SamplesNumber", to_string(get_samples_number()));

    const string separator_string = get_separator_string();

    if (has_sample_ids)
        add_json_field(printer, "SamplesId", vector_to_string(sample_ids, separator_string));

    add_json_field(printer, "SampleRoles", vector_to_string(get_sample_roles_vector()));
    printer.close_element();
}

void Dataset::missing_values_to_JSON(JsonWriter &printer) const
{
    printer.open_element("MissingValues");

    if (missing_values_number > 0)
        write_json(printer, {
            {"MissingValuesNumber", to_string(missing_values_number)},
            {"MissingValuesMethod", get_missing_values_method_string()},
            {"VariablesMissingValuesNumber", vector_to_string(variables_missing_values_number)},
            {"SamplesMissingValuesNumber", to_string(rows_missing_values_number)}
        });
    else
        add_json_field(printer, "MissingValuesNumber", to_string(missing_values_number));

    printer.close_element();
}

void Dataset::preview_data_to_JSON(JsonWriter &printer) const
{
    printer.open_element("PreviewData");

    add_json_field(printer, "PreviewSize", to_string(data_file_preview.size()));

    vector<string> vector_data_file_preview = convert_string_vector(data_file_preview, ",");

    printer.begin_array("Row");
    for (size_t i = 0; i < data_file_preview.size(); ++i)
    {
        printer.begin_array_object();
        add_json_field(printer, "Text", vector_data_file_preview[i]);
        printer.end_array_object();
    }
    printer.end_array();

    printer.close_element();
}

void Dataset::variables_from_JSON(const Json *variables_element)
{
    if (!variables_element)
        throw runtime_error("Variables element is nullptr.\n");

    set_variables_number(read_json_index(variables_element, "VariablesNumber"));

    for_json_items(variables_element, "Variable", variables.size(), [&](Index i, const Json* el)
    {
        Variable& variable = variables[i];

        variable.name = read_json_string(el, "Name");
        variable.set_scaler(read_json_string(el, "Scaler"));
        variable.set_role(read_json_string(el, "Role"));
        variable.set_type(read_json_string(el, "Type"));

        if (variable.is_categorical() || variable.is_binary())
        {
            const Json* categories_element = el->first_child("Categories");

            if (categories_element)
                variable.categories = get_tokens(read_json_string(el, "Categories"), ";");
            else if (variable.is_binary())
                variable.categories = { "0", "1" };
            else
                throw runtime_error("Categorical Variable Element is nullptr: Categories");
        }
    });
}

void Dataset::missing_values_from_JSON(const Json *missing_values_element)
{
    if (!missing_values_element)
        throw runtime_error("Missing values element is nullptr.\n");

    missing_values_number = read_json_index(missing_values_element, "MissingValuesNumber");

    if (missing_values_number > 0)
    {
        set_missing_values_method(read_json_string(missing_values_element, "MissingValuesMethod"));

        const string variables_string = read_json_string_fallback(missing_values_element,
            {"VariablesMissingValuesNumber", "RawVariablesMissingValuesNumber"});

        const vector<string> tokens = get_tokens(variables_string, " ");

        variables_missing_values_number.resize(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i)
            if (!tokens[i].empty())
                variables_missing_values_number(i) = stoi(tokens[i]);

        rows_missing_values_number = stol(read_json_string_fallback(missing_values_element,
            {"SamplesMissingValuesNumber", "RowsMissingValuesNumber"}));
    }
}

void Dataset::preview_data_from_JSON(const Json *preview_data_element)
{
    if (!preview_data_element)
        throw runtime_error("Preview data element is nullptr.\n ");

    const Index preview_size = read_json_index(preview_data_element, "PreviewSize");

    if (preview_size > 0)
    {
        data_file_preview.resize(preview_size);

        for_json_items(preview_data_element, "Row", preview_size, [&](Index i, const Json* row)
        {
            const string text = read_json_string(row, "Text");
            if (!text.empty())
                data_file_preview[i] = get_tokens(text, ",");
        });
    }
}

void Dataset::save(const filesystem::path& file_name) const
{
    ofstream file(file_name);

    if (!file.is_open())
        throw runtime_error("Cannot open file: " + file_name.string());

    JsonWriter document;

    to_JSON(document);

    file << document.c_str();
}

void Dataset::load(const filesystem::path& file_name)
{
    from_JSON(load_json_file(file_name));
}

void Dataset::infer_column_types(const vector<vector<string_view>>& sample_rows)
{
    const Index variables_number = variables.size();
    const size_t total_rows = sample_rows.size();

    if (total_rows == 0) return;

    vector<size_t> row_indices(total_rows);
    iota(row_indices.begin(), row_indices.end(), 0);

    shuffle_vector(row_indices);

    const size_t rows_to_check = min(size_t(100), total_rows);
    const size_t id_offset = has_sample_ids ? 1 : 0;

    for (Index col_index = 0; col_index < variables_number; ++col_index)
    {
        Variable& variable = variables[col_index];
        variable.type = VariableType::None;

        const size_t token_index = col_index + id_offset;

        for (size_t i = 0; i < rows_to_check; ++i)
        {
            const size_t row_index = row_indices[i];

            if (token_index >= sample_rows[row_index].size()) continue;

            const string_view token = sample_rows[row_index][token_index];

            if (token.empty() || token == missing_values_label) continue;

            if (variable.is_categorical()) break;

            if (is_numeric_string(token))
            {
                if (variable.type == VariableType::None)
                    variable.type = VariableType::Numeric;
            }
            else if (is_date_time_string(token))
            {
                if (variable.type == VariableType::None)
                    variable.type = VariableType::DateTime;
            }
            else
                variable.type = VariableType::Categorical;
        }

        if (variable.type == VariableType::None)
            variable.type = VariableType::Numeric;
    }

    for (Index col_index = 0; col_index < variables_number; ++col_index)
    {
        if (!variables[col_index].is_categorical()) continue;

        const size_t token_index = col_index + id_offset;

        std::set<string> unique_categories;
        for (const vector<string_view>& row : sample_rows)
            if (token_index < row.size() && !row[token_index].empty() && row[token_index] != missing_values_label)
                unique_categories.emplace(row[token_index]);

        variables[col_index].categories.assign(unique_categories.begin(), unique_categories.end());
    }
}

DateFormat Dataset::infer_dataset_date_format(const vector<Variable>& variables,
                                              const vector<vector<string_view>>& sample_rows,
                                              bool has_sample_ids,
                                              const string& missing_values_label)
{
    static const regex date_re(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}).*)");

    const size_t id_offset = has_sample_ids ? 1 : 0;

    for (size_t col_index = 0; col_index < variables.size(); ++col_index)
    {
        if (variables[col_index].type != VariableType::DateTime)
            continue;

        const size_t token_index = col_index + id_offset;

        for (const vector<string_view>& row : sample_rows)
        {
            if (token_index >= row.size())
                continue;

            const string_view token = row[token_index];

            if (token.empty() || token == missing_values_label)
                continue;

            cmatch date_parts;
            if (regex_match(token.data(), token.data() + token.size(), date_parts, date_re))
            {
                const int part1 = stoi(date_parts[1].str());
                const int part2 = stoi(date_parts[2].str());

                if (part1 > 12)
                    return DMY;
                if (part2 > 12)
                    return MDY;
            }
        }
    }

    return AUTO;
}

void Dataset::read_data_file_preview(const vector<vector<string_view>>& all_rows)
{
    if (all_rows.empty())
        return;

    const Index num_first_rows_to_show = 3;

    data_file_preview.clear();

    auto copy_row = [](const vector<string_view>& src) {
        vector<string> dst;
        dst.reserve(src.size());
        for (string_view sv : src) dst.emplace_back(sv);
        return dst;
    };

    const Index first_rows = Index(min(static_cast<size_t>(num_first_rows_to_show), all_rows.size()));

    for (Index i = 0; i < first_rows; ++i)
        data_file_preview.push_back(copy_row(all_rows[i]));

    if (all_rows.size() > num_first_rows_to_show)
        data_file_preview.push_back(copy_row(all_rows.back()));
}

void Dataset::check_separators(string_view line) const
{
    const string separator_string = get_separator_string();
    const string separator_name = get_separator_name();

    if (line.find(separator_string) == string_view::npos)
    {
        bool has_any_separator = false;

        for (const auto& [sep, str, name] : separator_map)
            if (line.find(str) != string_view::npos) { has_any_separator = true; break; }

        if (has_any_separator)
            throw runtime_error("Separator '" + separator_string + "' not found in line " + string(line) + ".\n");

        return;
    }

    for (const auto& [sep, str, name] : separator_map)
        if (sep != separator && line.find(str) != string_view::npos)
            throw runtime_error("Found " + name + " ('" + str + "') in data file "
                                + data_path.string() + ", but separator is " + separator_name + " ('" + separator_string + "').");
}

bool Dataset::has_binary_variables() const
{
    return any_of(variables.begin(), variables.end(),
                  [](const Variable& variable) { return variable.is_binary(); });
}

bool Dataset::has_categorical_variables() const
{
    return any_of(variables.begin(), variables.end(),
                  [](const Variable& variable) { return variable.is_categorical(); });
}

bool Dataset::has_binary_or_categorical_variables() const
{
    return any_of(variables.begin(), variables.end(), [](const Variable& v) {
        return v.is_binary() || v.is_categorical();
    });
}

bool Dataset::has_time_variable() const
{
    return any_of(variables.begin(), variables.end(),
                  [](const Variable& variable) { return variable.role == VariableRole::Time; });
}

bool Dataset::has_validation() const
{
    return get_samples_number("Validation") != 0;
}

bool Dataset::has_missing_values(const vector<string_view>& row) const
{
    for (size_t i = 0; i < row.size(); ++i)
        if (row[i].empty() || row[i] == missing_values_label)
            return true;

    return false;
}

vector<vector<Index>> Dataset::split_samples(const vector<Index>& sample_indices, Index new_batch_size) const
{
    vector<vector<Index>> batches;
    get_batches(sample_indices, new_batch_size, false, batches);
    return batches;
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
