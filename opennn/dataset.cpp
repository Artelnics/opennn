//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "dataset.h"
#include "batch.h"
#include "random_utilities.h"
#include <cstdlib>
#include <regex>

namespace opennn
{

VectorI Dataset::get_sample_role_numbers() const
{
    VectorI count = VectorI::Zero(4);

    const Index samples_number = get_samples_number();

    for (Index i = 0; i < samples_number; ++i)
    {
        using enum SampleRole;
        switch (sample_roles[i])
        {
        case Training:   count[0]++; break;
        case Validation: count[1]++; break;
        case Testing:    count[2]++; break;
        case None:       count[3]++; break;
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

    transform(execution::par, sample_roles.begin(), sample_roles.end(),
              sample_roles_vector.begin(),
              [](SampleRole r) { return static_cast<Index>(r); });

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
        shuffle_vector(shuffled_indices);
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

void Dataset::set_data_path(const filesystem::path& new_data_path)
{
    data_path = new_data_path;
    binary_rows_number = 0;
    binary_columns_number = 0;
    invalidate_data_buffer();
}

void Dataset::set_storage_mode(StorageMode new_storage_mode)
{
    storage_mode = new_storage_mode;

    if (storage_mode == StorageMode::BinaryFile)
        read_binary_header();

    invalidate_data_buffer();
}

void Dataset::use_binary_data_file(const filesystem::path& binary_data_file_name)
{
    set_data_path(binary_data_file_name);
    storage_mode = StorageMode::BinaryFile;
    read_binary_header();

    if (sample_roles.empty())
        sample_roles.resize(binary_rows_number, SampleRole::Training);

    resize_data_from_JSON(0);
}

Index Dataset::get_samples_number(const string& sample_role) const
{
    const SampleRole role_type = string_to_sample_role(sample_role);
    return ranges::count(sample_roles, role_type);
}

Index Dataset::get_used_samples_number() const
{
    return get_samples_number() - get_samples_number("None");
}

void Dataset::set_sample_roles(const string& sample_role)
{
    const SampleRole role_type = string_to_sample_role(sample_role);
    ranges::fill(sample_roles, role_type);
}

void Dataset::set_sample_role(const Index index, const string& new_role)
{
    sample_roles[index] = string_to_sample_role(new_role);
}

void Dataset::set_sample_roles(const vector<string>& new_roles)
{
    ranges::transform(new_roles, sample_roles.begin(), string_to_sample_role);
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
            if (const Index idx = indices[i++]; sample_roles[idx] != SampleRole::None)
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

    if (variables_number == 0)
        return;

    if (variables_number == 1)
    {
        variables[0].set_role("None");
        return;
    }

    set_variable_roles("Input");

    bool target = false;
    bool time_variable = false;

    for (Index i = variables_number - 1; i >= 0; i--)
    {
        if (variables[i].type == VariableType::DateTime && !time_variable)
        {
            variables[i].set_role("Time");
            time_variable = true;
        }
        else if (variables[i].type == VariableType::Constant)
        {
            variables[i].set_role("None");
        }
        else if (!target)
        {
            variables[i].set_role("Target");
            target = true;
        }
    }
}

vector<VariableType> Dataset::get_variable_types(const vector<Index>& indices) const
{
    vector<VariableType> variable_types(indices.size());

    ranges::transform(indices, variable_types.begin(),
                      [this](Index i) { return get_variable_type(i); });

    return variable_types;
}

void Dataset::set_default_variable_names()
{
    const Index variables_number = variables.size();

    for (Index i = 0; i < variables_number; ++i)
        variables[i].name = format("variable_{}", 1 + i);
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

    invalidate_data_buffer();
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

vector<string> Dataset::get_variable_names() const
{
    vector<string> variable_names(variables.size());

    ranges::transform(variables, variable_names.begin(),
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

    return ranges::count_if(variables,
                            [role_type](const Variable& v) { return role_matches(v.role, role_type); });
}

Index Dataset::get_used_variables_number() const
{
    return ranges::count_if(variables,
                            [](const Variable& var) {
                                  return var.is_used();
                            });
}

vector<Variable> Dataset::get_variables(const string& variable_role) const
{
    vector<Variable> this_variables;
    this_variables.reserve(get_variables_number(variable_role));

    const VariableRole role_type = string_to_variable_role(variable_role);

    ranges::copy_if(variables, back_inserter(this_variables),
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

    return transform_reduce(variables.begin(), variables.end(), Index(0), plus<>{},
        [&](const Variable& v) { return role_matches(v.role, role_type) ? v.feature_count() : Index(0); });
}

vector<Index> Dataset::get_used_feature_indices() const
{
    const Index used_features_number =
        get_features_number() - get_features_number("None") - get_features_number("Time");
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
        throw runtime_error(format("Size of variables uses ({}) must be equal to variables size ({}).\n",
                                   new_variables_roles_size, variables.size()));

    for (size_t i = 0; i < new_variables_roles.size(); ++i)
        variables[i].set_role(new_variables_roles[i]);

    invalidate_data_buffer();
}

void Dataset::set_variable_indices(const vector<Index>& input_variables,
                                       const vector<Index>& target_variables)
{
    set_variable_roles("None");

    for (const Index index : input_variables)
        set_variable_role(index, "Input");

    for (const Index index : target_variables)
        set_variable_role(index,
            variables[index].role == VariableRole::Input ? "InputTarget" : "Target");

    set_shape("Input", { get_features_number("Input") });
    set_shape("Target", { get_features_number("Target") });
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
    invalidate_data_buffer();
}

void Dataset::set_variable_role(const string& name, const string& new_role)
{
    set_variable_role(get_variable_index(name), new_role);
}

void Dataset::set_variable_type(const Index index, const VariableType& new_type)
{
    variables[index].type = new_type;
    invalidate_data_buffer();
}

void Dataset::set_variable_type(const string& name, const VariableType& new_type)
{
    set_variable_type(get_variable_index(name), new_type);
}

void Dataset::set_variable_types(const VariableType& new_type)
{
    for (auto& variable : variables)
        variable.type = new_type;

    invalidate_data_buffer();
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
        throw runtime_error(format("Size of names ({}) is not equal to variables number ({}).\n",
                                   new_names.size(), variables_number));

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

    invalidate_data_buffer();
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

    return {};
}

string Dataset::get_separator_name() const
{
    for (const auto& [sep, str, name] : separator_map)
        if (sep == separator) return name;

    return {};
}

static const vector<pair<Dataset::Codification, string>> codification_map = {
    {Dataset::Codification::UTF8,      "UTF-8"},
    {Dataset::Codification::SHIFT_JIS, "SHIFT_JIS"}
};

string Dataset::get_codification_string() const
{
    for (const auto& [cod, name] : codification_map)
        if (cod == codification) return name;

    return "UTF-8";
}

Index Dataset::get_variable_index(const string& variable_name) const
{
    auto it = ranges::find_if(variables,
                              [&](const Variable& v) { return v.name == variable_name; });

    if (it == variables.end())
        throw runtime_error(format("Cannot find {}\n", variable_name));

    return distance(variables.begin(), it);
}

Index Dataset::get_variable_index(const Index feature_index) const
{
    const Index variables_number = get_variables_number();

    Index features_seen = 0;

    for (Index i = 0; i < variables_number; ++i)
    {
        features_seen += variables[i].feature_count();

        if (feature_index < features_seen)
            return i;
    }

    throw runtime_error(format("Cannot find variable index: {}.\n", feature_index));
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
    const Index index = transform_reduce(variables.begin(), variables.begin() + variable_index,
        Index(0), plus<>{}, [](const Variable& v) { return v.feature_count(); });

    vector<Index> indices(variables[variable_index].feature_count());
    iota(indices.begin(), indices.end(), index);

    return indices;
}

void Dataset::set_default()
{
    has_header = false;

    has_sample_ids = false;

    separator = Separator::Semicolon;

    set_default_variable_names();
}

void Dataset::set_separator_string(const string& new_separator_string)
{
    for (const auto& [sep, str, name] : separator_map)
        if (str == new_separator_string) { separator = sep; return; }

    throw runtime_error(format("Unknown separator: {}", new_separator_string));
}

void Dataset::set_separator_name(const string& new_separator_name)
{
    for (const auto& [sep, str, name] : separator_map)
        if (name == new_separator_name) { separator = sep; return; }

    throw runtime_error(format("Unknown separator: {}.\n", new_separator_name));
}

void Dataset::set_codification(const string& new_codification_string)
{
    for (const auto& [cod, name] : codification_map)
        if (name == new_codification_string) { codification = cod; return; }

    throw runtime_error(format("Unknown codification: {}.\n", new_codification_string));
}

void Dataset::variables_to_JSON(JsonWriter &printer) const
{
    const Index variables_number = get_variables_number();

    printer.open_element("Variables");
    add_json_field(printer, "VariablesNumber", to_string(variables_number));

    printer.begin_array("Variable");
    for (Index i = 0; i < variables_number; ++i)
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

    if (has_sample_ids)
        add_json_field(printer, "SamplesId", vector_to_string(sample_ids, get_separator_string()));

    add_json_field(printer, "SampleRoles", vector_to_string(get_sample_roles_vector()));
    printer.close_element();
}

void Dataset::preview_data_to_JSON(JsonWriter &printer) const
{
    printer.open_element("PreviewData");

    add_json_field(printer, "PreviewSize", to_string(data_file_preview.size()));

    vector<string> vector_data_file_preview = convert_string_vector(data_file_preview, ",");

    printer.begin_array("Row");
    for (const string& row_text : vector_data_file_preview)
    {
        printer.begin_array_object();
        add_json_field(printer, "Text", row_text);
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

void Dataset::preview_data_from_JSON(const Json *preview_data_element)
{
    if (!preview_data_element)
        throw runtime_error("Preview data element is nullptr.\n ");

    if (const Index preview_size = read_json_index(preview_data_element, "PreviewSize"); preview_size > 0)
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
        throw runtime_error(format("Cannot open file: {}", file_name.string()));

    JsonWriter document;

    to_JSON(document);

    file << document.c_str();
}

void Dataset::load(const filesystem::path& file_name)
{
    from_JSON(load_json_file(file_name));
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
        const bool has_any_separator = ranges::any_of(separator_map,
            [&](const auto& entry) { return line.find(get<1>(entry)) != string_view::npos; });

        if (has_any_separator)
            throw runtime_error(format("Separator '{}' not found in line {}.\n", separator_string, line));

        return;
    }

    for (const auto& [sep, str, name] : separator_map)
        if (sep != separator && line.find(str) != string_view::npos)
            throw runtime_error(format("Found {} ('{}') in data file {}, but separator is {} ('{}').",
                                       name, str, data_path.string(), separator_name, separator_string));
}

bool Dataset::has_validation() const
{
    return get_samples_number("Validation") != 0;
}

vector<vector<Index>> Dataset::split_samples(const vector<Index>& sample_indices, Index new_batch_size) const
{
    vector<vector<Index>> batches;
    get_batches(sample_indices, new_batch_size, false, batches);
    return batches;
}

void Dataset::fill_inputs(const vector<Index>&, const vector<Index>&, float*, bool, int) const
{
    throw runtime_error("Dataset::fill_inputs must be implemented by a concrete dataset.");
}

void Dataset::fill_decoder(const vector<Index>&, const vector<Index>&, float*, bool, int) const
{
    throw runtime_error("Dataset::fill_decoder must be implemented by a concrete dataset.");
}

void Dataset::fill_targets(const vector<Index>&, const vector<Index>&, float*, bool, int) const
{
    throw runtime_error("Dataset::fill_targets must be implemented by a concrete dataset.");
}

void Dataset::fill_batch(Batch& batch,
                         const vector<Index>& sample_indices,
                         const vector<Index>& input_indices,
                         const vector<Index>& decoder_indices,
                         const vector<Index>& target_indices,
                         bool is_training,
                         bool allow_device_data_buffer) const
{
#ifdef OPENNN_HAS_CUDA
    if (allow_device_data_buffer
        && batch.uses_cuda()
        && supports_device_data_buffer()
        && try_fill_from_device_data_buffer(batch,
                                            sample_indices,
                                            input_indices,
                                            decoder_indices,
                                            target_indices))
    {
        batch.placement = BatchPlacement::Device;
        return;
    }
#endif

    batch.current_sample_count = ssize(sample_indices);

    const bool on_gpu = batch.uses_cuda();

    float* const input_buffer   = on_gpu ? batch.inputs_host  : batch.input.as<float>();
    float* const decoder_buffer = on_gpu ? batch.decoder_host : batch.decoder.as<float>();
    float* const target_buffer  = on_gpu ? batch.targets_host : batch.target.as<float>();

    if (batch.input_contiguous < 0 && !input_indices.empty())
        batch.input_contiguous = is_contiguous(input_indices) ? 1 : 0;
    if (batch.decoder_contiguous < 0 && !decoder_indices.empty())
        batch.decoder_contiguous = is_contiguous(decoder_indices) ? 1 : 0;
    if (batch.target_contiguous < 0 && !target_indices.empty())
        batch.target_contiguous = is_contiguous(target_indices) ? 1 : 0;

    fill_inputs(sample_indices,
                input_indices,
                input_buffer,
                is_training,
                batch.input_contiguous);

    if (!batch.decoder_shape.empty())
        fill_decoder(sample_indices,
                     decoder_indices,
                     decoder_buffer,
                     is_training,
                     batch.decoder_contiguous);

    fill_targets(sample_indices,
                 target_indices,
                 target_buffer,
                 is_training,
                 batch.target_contiguous);

    batch.placement = BatchPlacement::Host;
}

void Dataset::read_binary_header() const
{
    ifstream file(data_path, ios::binary);

    if (!file.is_open())
        throw runtime_error(format("Failed to open binary data file: {}", data_path.string()));

    file.read(reinterpret_cast<char*>(&binary_columns_number), sizeof(Index));
    file.read(reinterpret_cast<char*>(&binary_rows_number), sizeof(Index));

    if (!file)
        throw runtime_error(format("Failed to read binary data header: {}", data_path.string()));
}

bool Dataset::try_fill_binary_tensor(const vector<Index>& sample_indices,
                                     const vector<Index>& feature_indices,
                                     float* output,
                                     int contiguous_hint) const
{
    if (storage_mode != StorageMode::BinaryFile) return false;
    if (sample_indices.empty() || feature_indices.empty()) return true;

    if (binary_rows_number == 0 || binary_columns_number == 0)
        read_binary_header();

    ifstream file(data_path, ios::binary);
    if (!file.is_open())
        throw runtime_error(format("Failed to open binary data file: {}", data_path.string()));

    const Index columns_number = binary_columns_number;
    const Index features_number = ssize(feature_indices);
    const bool contiguous = contiguous_hint >= 0
                          ? static_cast<bool>(contiguous_hint)
                          : is_contiguous(feature_indices);

    const Index first_column = feature_indices.front();
    const streamoff header_bytes = streamoff(2 * sizeof(Index));

    for (Index i = 0; i < ssize(sample_indices); ++i)
    {
        const Index row = sample_indices[size_t(i)];
        if (row < 0 || row >= binary_rows_number)
            throw runtime_error("Binary data row index is out of range.");

        float* const dst = output + i * features_number;

        if (contiguous)
        {
            const streamoff offset = header_bytes
                + streamoff(row * columns_number + first_column) * streamoff(sizeof(float));
            file.seekg(offset, ios::beg);
            file.read(reinterpret_cast<char*>(dst), features_number * sizeof(float));
        }
        else
        {
            for (Index j = 0; j < features_number; ++j)
            {
                const Index column = feature_indices[size_t(j)];
                const streamoff offset = header_bytes
                    + streamoff(row * columns_number + column) * streamoff(sizeof(float));
                file.seekg(offset, ios::beg);
                file.read(reinterpret_cast<char*>(dst + j), sizeof(float));
            }
        }

        if (!file)
            throw runtime_error(format("Failed to read binary data file: {}", data_path.string()));
    }

    return true;
}

void Dataset::set_matrix_storage()
{
    storage_mode = StorageMode::Matrix;
    binary_rows_number = 0;
    binary_columns_number = 0;
    invalidate_data_buffer();
}

void Dataset::invalidate_data_buffer() const
{
    data_buffer_shape.clear();
}

#ifdef OPENNN_HAS_CUDA

bool Dataset::prepare_device_data_buffer() const
{
    lock_guard<mutex> lock(data_buffer_mutex);

    const Index rows  = get_samples_number();
    const Index features = get_features_number();
    if (rows == 0 || features == 0) return false;

    if (data_buffer_shape == Shape{rows, features}
        && data_buffer.device_type == Device::CUDA
        && !data_buffer.empty())
        return true;

    const Index buffer_bytes = rows * features * Index(sizeof(float));

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CHECK_CUDA(cudaMemGetInfo(&free_bytes, &total_bytes));
    const Index margin = Index(512) << 20;
    if (buffer_bytes + margin > Index(free_bytes)) return false;

    vector<Index> sample_indices(size_t(rows));
    iota(sample_indices.begin(), sample_indices.end(), Index(0));

    vector<Index> feature_indices(size_t(features));
    iota(feature_indices.begin(), feature_indices.end(), Index(0));

    vector<float> host_data(size_t(rows) * size_t(features));
    fill_inputs(sample_indices, feature_indices, host_data.data(),
                /*is_training=*/false);

    data_buffer.resize_bytes(Index(host_data.size()) * Index(sizeof(float)), Device::CUDA);

    cudaStream_t stream = Backend::get_compute_stream();
    CHECK_CUDA(cudaMemcpyAsync(data_buffer.data, host_data.data(),
                               host_data.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    data_buffer_shape = {rows, features};
    return true;
}

bool Dataset::try_fill_from_device_data_buffer(Batch& batch,
                                               const vector<Index>& sample_indices,
                                               const vector<Index>& input_indices,
                                               const vector<Index>& decoder_indices,
                                               const vector<Index>& target_indices) const
{
    if (!decoder_indices.empty()) return false;

    if (const char* disable_buffer = getenv("OPENNN_DISABLE_DATA_BUFFER");
        disable_buffer && disable_buffer[0] == '1')
    {
        return false;
    }

    if (!prepare_device_data_buffer())
        return false;

    for (const Index sample_index : sample_indices)
    {
        if (sample_index < 0 || sample_index >= data_buffer_shape[0])
            return false;
    }

    batch.gather_device_async(sample_indices,
                              data_buffer.as<float>(),
                              data_buffer_shape[1],
                              input_indices,
                              target_indices);

    return true;
}

#endif

void Dataset::samples_from_JSON(const Json *samples_element)
{
    if (!samples_element)
        throw runtime_error("Samples element is nullptr.\n");

    const Index samples_number = read_json_index(samples_element, "SamplesNumber");

    if (has_sample_ids)
        sample_ids = get_tokens(read_json_string(samples_element, "SamplesId"), get_separator_string());

    sample_roles.resize(samples_number, SampleRole::Training);

    if (!variables.empty())
        set_sample_roles(get_tokens(read_json_string(samples_element, "SampleRoles"), " "));

    resize_data_from_JSON(samples_number);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
