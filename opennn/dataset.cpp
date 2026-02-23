//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   D A T A   S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "dataset.h"
#include "time_series_dataset.h"
#include "statistics.h"
#include "scaling.h"
#include "correlations.h"
#include "tensors.h"
#include "strings_utilities.h"
#include "random_utilities.h"
#include "image_dataset.h"

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


bool Dataset::get_display() const
{
    return display;
}


bool Dataset::is_empty() const
{
    return data.size() == 0;
}

Shape Dataset::get_input_shape() const
{
    return input_shape;
}


Shape Dataset::get_target_shape() const
{
    return target_shape;
}


Dataset::Variable::Variable(const string& new_name,
                                  const string& new_variable_role,
                                  const VariableType& new_type,
                                  const string& new_scaler,
                                  const vector<string>& new_categories)
{
    set(new_name, new_variable_role, new_type, new_scaler, new_categories);
}


void Dataset::Variable::set(const string& new_name,
                               const string& new_variable_role,
                               const VariableType& new_type,
                               const string& new_scaler,
                               const vector<string>& new_categories)
{
    name = new_name;
    role = new_variable_role;
    type = new_type;
    scaler = new_scaler;
    categories = new_categories;
}


void Dataset::Variable::set_scaler(const string& new_scaler)
{
    scaler = new_scaler;
}


void Dataset::Variable::set_role(const string& new_variable_role)
{
    role = new_variable_role;
}


void Dataset::Variable::set_type(const string& new_variable_type)
{
    if (new_variable_type == "Numeric")
        type = VariableType::Numeric;
    else if (new_variable_type == "Binary")
        type = VariableType::Binary;
    else if (new_variable_type == "Categorical")
        type = VariableType::Categorical;
    else if (new_variable_type == "DateTime")
        type = VariableType::DateTime;
    else if (new_variable_type == "Constant")
        type = VariableType::Constant;
    else
        throw runtime_error("Variable type is not valid (" + new_variable_type + ").\n");
}


void Dataset::Variable::set_categories(const vector<string>& new_categories)
{
    categories = new_categories;
}


void Dataset::Variable::from_XML(const XMLDocument& document)
{
    name = read_xml_string(document.FirstChildElement(), "Name");
    set_scaler(read_xml_string(document.FirstChildElement(), "Scaler"));
    set_role(read_xml_string(document.FirstChildElement(), "Role"));
    set_type(read_xml_string(document.FirstChildElement(), "Type"));

    if (type == VariableType::Categorical)
    {
        const string categories_text = read_xml_string(document.FirstChildElement(), "Categories");
        categories = get_tokens(categories_text, ";");
    }
}


void Dataset::Variable::to_XML(XMLPrinter& printer) const
{
    add_xml_element(printer, "Name", name);
    add_xml_element(printer, "Scaler", scaler);
    add_xml_element(printer, "Role", get_role());
    add_xml_element(printer, "Type", get_type_string());

    if (type == VariableType::Categorical || type == VariableType::Binary)
        add_xml_element(printer, "Categories", vector_to_string(categories,";"));
}


void Dataset::Variable::print() const
{
    cout << "Variable" << endl
         << "Name: " << name << endl
         << "Role: " << get_role() << endl
         << "Type: " << get_type_string() << endl
         << "string: " << scaler << endl;

    if (categories.size() != 0)
    {
        cout << "Categories: " << endl
             << categories;
    }

    cout << endl;
}


string Dataset::Variable::get_role() const
{
    return role;
}


Index Dataset::Variable::get_categories_number() const
{
    return categories.size();
}


void Dataset::get_categorical_info(const string& variable_role, vector<Index>& variable_indices, vector<Index>& categories_number) const
{
    variable_indices.clear();
    categories_number.clear();

    const vector<Variable> variables = get_variables(variable_role);

    const Index variables_number = variables.size();

    for(Index i = 0; i < variables_number; ++i)
    {
        if(variables[i].is_categorical())
        {
            variable_indices.push_back(i);

            categories_number.push_back(variables[i].get_categories_number());
        }
    }
}


bool Dataset::is_sample_used(const Index index) const
{
    return sample_roles[index] != "None";
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
        else if (role == "Validation")   count[1]++;
        else if (role == "Testing")     count[2]++;
        else if (role == "None")        count[3]++;
        else throw runtime_error("Unknown sample role: " + role);
    }

    return count;
}


vector<Index> Dataset::get_sample_indices(const string& sample_role) const
{
    const Index samples_number = get_samples_number();

    const Index count = get_samples_number(sample_role);

    vector<Index> indices(count);

    Index index = 0;

    for(Index i = 0; i < samples_number; i++)
        if (sample_roles[i] == sample_role)
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

    for(Index i = 0; i < samples_number; i++)
        if (sample_roles[i] != "None")
            used_indices[index++] = i;

    return used_indices;
}


string Dataset::get_sample_role(const Index index) const
{
    return sample_roles[index];
}


const vector<string>& Dataset::get_sample_roles() const
{
    return sample_roles;
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
    const Index batches_number = (samples_number + batch_size - 1) / batch_size;

    vector<Index> indices = sample_indices;

    if (shuffle)
        shuffle_vector_blocks(indices);

    vector<vector<Index>> batches(batches_number);

    #pragma omp parallel for if(batches_number > 64)
    for (Index i = 0; i < batches_number; i++)
    {
        const auto start_it = indices.begin() + (i * batch_size);
        const auto end_it = indices.begin() + min((i + 1) * batch_size, samples_number);

        batches[i].assign(start_it, end_it);
    }

    return batches;
}


Index Dataset::get_samples_number(const string& sample_role) const
{
    return count_if(sample_roles.begin(), sample_roles.end(),
                    [&sample_role](const string& new_sample_role) { return new_sample_role == sample_role; });
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
    if (new_role == "Training")
        sample_roles[index] = "Training";
    else if (new_role == "Validation")
        sample_roles[index] = "Validation";
    else if (new_role == "Testing")
        sample_roles[index] = "Testing";
    else if (new_role == "None")
        sample_roles[index] = "None";
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


void Dataset::split_samples_random(const type training_samples_ratio,
                                   type validation_samples_ratio,
                                   type testing_samples_ratio)
{
    const Index used_samples_number = get_used_samples_number();

    if (used_samples_number == 0) return;

    const type total_ratio = training_samples_ratio + validation_samples_ratio + testing_samples_ratio;

    const Index validation_samples_number = Index((validation_samples_ratio * used_samples_number) / total_ratio);
    const Index testing_samples_number = Index((testing_samples_ratio * used_samples_number) / total_ratio);

    const Index training_samples_number = used_samples_number - validation_samples_number - testing_samples_number;

    const Index sum_samples_number = training_samples_number + validation_samples_number + testing_samples_number;

    if (sum_samples_number != used_samples_number)
        throw runtime_error("Sum of numbers of training, selection and testing samples is not equal to number of used samples.\n");

    const Index samples_number = get_samples_number();

    vector<Index> indices(samples_number);
    iota(indices.begin(), indices.end(), 0);

    shuffle_vector(indices);

    auto assign_sample_role = [this, &indices](string role, Index count, Index& i)
    {
        Index assigned_count = 0;

        while (assigned_count < count)
        {
            const Index index = indices[i++];

            if (sample_roles[index] != "None")
            {
                sample_roles[index] = role;
                assigned_count++;
            }
        }
    };

    Index index = 0;

    assign_sample_role("Training", training_samples_number, index);
    assign_sample_role("Validation", validation_samples_number, index);
    assign_sample_role("Testing", testing_samples_number, index);
}


void Dataset::split_samples_sequential(const type training_samples_ratio,
                                       type validation_samples_ratio,
                                       type testing_samples_ratio)
{
    const Index used_samples_number = get_used_samples_number();

    if (used_samples_number == 0) return;

    const type total_ratio = training_samples_ratio + validation_samples_ratio + testing_samples_ratio;

    const Index validation_samples_number = Index(validation_samples_ratio * type(used_samples_number) / type(total_ratio));
    const Index testing_samples_number = Index(testing_samples_ratio * type(used_samples_number) / type(total_ratio));
    const Index training_samples_number = used_samples_number - validation_samples_number - testing_samples_number;

    const Index sum_samples_number = training_samples_number + validation_samples_number + testing_samples_number;

    if (sum_samples_number != used_samples_number)
        throw runtime_error("Sum of numbers of training, selection and testing samples is not equal to number of used samples.\n");

    auto set_sample_roles = [this](string role, Index count, Index& i)
    {
        Index current_count = 0;

        while (current_count < count)
        {
            if (sample_roles[i] != "None")
            {
                sample_roles[i] = role;
                current_count++;
            }

            i++;
        }
    };

    Index index = 0;

    set_sample_roles("Training", training_samples_number, index);
    set_sample_roles("Validation", validation_samples_number, index);
    set_sample_roles("Testing", testing_samples_number, index);
}


void Dataset::set_variables(const vector<Variable>& new_variables)
{
    variables = new_variables;
}


void Dataset::set_default_variables_roles()
{
    const Index variables_number = variables.size();

    if (variables_number == 0)
        return;

    if (variables_number == 1)
    {
        variables[0].set_role("None");
        return;
    }

    set_feature_roles("Input");

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

    for(const Dataset::Variable& variable : variables)
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

void Dataset::set_default_variables_roles_forecasting()
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

    set_feature_roles("Input");

    for(Index i = variables.size() - 1; i >= 0; i--)
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

vector<Dataset::VariableType> Dataset::get_variable_types(const vector<Index> indices) const
{
    vector<Dataset::VariableType> variable_types(indices.size());

    for (Index i = 0; i < static_cast<Index>(indices.size()) ; i++)
        variable_types[i] = get_variable_type(indices[i]);

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
    const Index features_number = get_features_number();

    vector<string> feature_names(features_number);

    Index index = 0;

    for(const Dataset::Variable& variable : variables)
        if (variable.type == VariableType::Categorical)
            for(size_t j = 0; j < variable.categories.size(); j++)
                feature_names[index++] = variable.categories[j];
        else
            feature_names[index++] = variable.name;

    return feature_names;
}


vector<string> Dataset::get_feature_names(const string& variable_role) const
{
    const Index features_number = get_features_number(variable_role);

    vector<string> feature_names(features_number);

    Index index = 0;

    for(const Dataset::Variable& variable : variables)
    {
        if(!((variable.role == variable_role) ||
              ((variable_role == "Input" || variable_role == "Target") && variable.role == "InputTarget")))
            continue;

        if (variable.type == VariableType::Categorical)
            for(Index j = 0; j < variable.get_categories_number(); j++)
                feature_names[index++] = variable.categories[j];
        else
            feature_names[index++] = variable.name;
    }

    return feature_names;
}


Shape Dataset::get_shape(const string& variable_role) const
{
    if (variable_role == "Input")
        return input_shape;
    else if (variable_role == "Target")
        return target_shape;
    // else if (variable_role == "Decoder")
    //     return decoder_shape;
    else
        throw invalid_argument("get_shape: Invalid variable role string: " + variable_role);
}


void Dataset::set_shape(const string& variable_role, const Shape& new_dimensions)
{
    if (variable_role == "Input")
        input_shape = new_dimensions;
    else if (variable_role == "Target")
        target_shape = new_dimensions;
    // else if (variable_role == "Decoder")
    //     decoder_shape = new_dimensions;
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

    for(const Dataset::Variable& variable : variables)
    {
        if (variable.role.find(variable_role) == string::npos)
        {
            variable.type == VariableType::Categorical
                ? feature_index += variable.get_categories_number()
                : feature_index++;

            continue;
        }

        if (variable.type == VariableType::Categorical)
            for(Index j = 0; j < variable.get_categories_number(); j++)
                this_feature_indices[this_feature_index++] = feature_index++;
        else
            this_feature_indices[this_feature_index++] = feature_index++;
    }

    return this_feature_indices;
}


vector<Index> Dataset::get_variable_indices(const string& variable_role) const
{
    const Index count = get_variables_number(variable_role);

    vector<Index> indices(count);

    const Index variables_number = get_variables_number();

    Index index = 0;

    for(Index i = 0; i < variables_number; i++)
        if (variables[i].role.find(variable_role) != string::npos)
            indices[index++] = i;

    return indices;
}


vector<Index> Dataset::get_used_variables_indices() const
{
    const Index variables_number = get_variables_number();

    const Index used_variables_number = get_used_variables_number();

    vector<Index> used_indices(used_variables_number);

    Index index = 0;

    for(Index i = 0; i < variables_number; i++)
        if (variables[i].role == "Input"
            || variables[i].role == "Target"
            || variables[i].role == "Time"
            || variables[i].role == "InputTarget")
            used_indices[index++] = i;

    return used_indices;
}


vector<string> Dataset::get_feature_scalers(const string& variable_role) const
{
    const Index input_variables_number = get_variables_number(variable_role);
    const Index input_features_number = get_features_number(variable_role);

    const vector<Variable> input_variables = get_variables(variable_role);

    vector<string> input_variable_scalers(input_features_number);

    Index index = 0;

    for(Index i = 0; i < input_variables_number; i++)
        if (input_variables[i].type == VariableType::Categorical)
            for(Index j = 0; j < input_variables[i].get_categories_number(); j++)
                input_variable_scalers[index++] = input_variables[i].scaler;
        else
            input_variable_scalers[index++] = input_variables[i].scaler;

    return input_variable_scalers;
}


vector<string> Dataset::get_variable_names() const
{
    const Index variables_number = get_variables_number();

    vector<string> variable_names(variables_number);

    for(Index i = 0; i < variables_number; i++)
        variable_names[i] = variables[i].name;

    return variable_names;
}


vector<string> Dataset::get_variable_names(const string& variable_role) const
{
    const Index count = get_variables_number(variable_role);

    vector<string> names(count);

    Index index = 0;

    for(const Dataset::Variable& variable : variables)
    {
        if(!((variable.role == variable_role) ||
              ((variable_role == "Input" || variable_role == "Target") && variable.role == "InputTarget")))
            continue;

        names[index++] = variable.name;
    }

    return names;
}


Index Dataset::get_variables_number(const string& variable_role) const
{
    Index count = 0;

    for(const Dataset::Variable& variable : variables)
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


const vector<Dataset::Variable>& Dataset::get_variables() const
{
    return variables;
}


vector<Dataset::Variable> Dataset::get_variables(const string& variable_role) const
{
    const Index count = get_variables_number(variable_role);

    vector<Variable> this_variables(count);
    Index index = 0;

    for(const Dataset::Variable& variable : variables)
        if (variable.role.find(variable_role) != string::npos)
            this_variables[index++] = variable;

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

    for(const Dataset::Variable& variable : variables)
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

    for(const Dataset::Variable& variable : variables)
    {
        const Index categories_number = variable.get_categories_number();

        if(variable.role == "None" || variable.role == "Time")
        {
            feature_index += (variable.type == VariableType::Categorical)
            ? variable.get_categories_number()
            : 1;
            continue;
        }

        if(variable.type == VariableType::Categorical)
            for(Index j = 0; j < categories_number; j++)
                used_feature_indices[used_feature_index++] = feature_index++;
        else
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
    {
        if(variables[index].role == "Input")
            set_variable_role(index, "InputTarget");
        else
            set_variable_role(index, "Target");
    }

    const Index input_dimensions_num = get_features_number("Input");
    const Index target_shape_num = get_features_number("Target");

    TimeSeriesDataset* ts_dataset = dynamic_cast<TimeSeriesDataset*>(this);

    if(ts_dataset)
        set_shape("Input", {ts_dataset->get_past_time_steps(), input_dimensions_num});
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

    for(Dataset::Variable& variable : variables)
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


void Dataset::set_feature_roles(const string& variable_role)
{
    const Index variables_number = get_variables_number();

    for(Index i = 0; i < variables_number; i++)
    {
        if (variables[i].type == VariableType::Constant || variables[i].type == VariableType::DateTime)
            variables[i].set_role("None");
        else
            variables[i].set_role(variable_role);
    }
}


void Dataset::set_variables_number(const Index new_variables_number)
{
    variables.resize(new_variables_number);
}


void Dataset::set_variable_scalers(const string& scalers)
{
    for(Dataset::Variable& variable : variables)
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


void Dataset::set_binary_variables()
{
    Index feature_index = 0;

    const Index variables_number = get_variables_number();

    for(Index variable_index = 0; variable_index < variables_number; variable_index++)
    {
        Variable& variable = variables[variable_index];

        if (variable.type == VariableType::Numeric)
        {
            const VectorR data_column = data.col(feature_index);

            if (is_binary(data_column))
            {
                variable.type = VariableType::Binary;
                variable.categories = { "0","1" };
            }

            feature_index++;
        }
        else if (variable.type == VariableType::Categorical)
            feature_index += variable.get_categories_number();
        else if (variable.type == VariableType::DateTime
                 || variable.type == VariableType::Constant
                 || variable.type == VariableType::Binary)
            feature_index++;
    }
}


void Dataset::unuse_constant_variables()
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

            feature_index++;
        }
        else if (variable.type == VariableType::DateTime || variable.type == VariableType::Constant)
        {
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
    }
}


const MatrixR& Dataset::get_data() const
{
    return data;
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


bool Dataset::get_header_line() const
{
    return has_header;
}


bool Dataset::get_has_sample_ids() const
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


MatrixR Dataset::get_data_samples(const string& sample_role) const
{
    const vector<Index> feature_indices = get_used_feature_indices();

    const vector<Index> sample_indices = get_sample_indices(sample_role);

    MatrixR this_data(sample_indices.size(), feature_indices.size());

    fill_tensor_data(data, sample_indices, feature_indices, this_data.data());

    return this_data;
}


MatrixR Dataset::get_feature_data(const string& variable_role) const
{
    const Index samples_number = get_samples_number();

    vector<Index> indices(samples_number);
    iota(indices.begin(), indices.end(), 0);

    const vector<Index> feature_indices = get_feature_indices(variable_role);

    MatrixR this_data(indices.size(), feature_indices.size());

    fill_tensor_data(data, indices, feature_indices, this_data.data());

    return this_data;
}


MatrixR Dataset::get_data(const string& sample_role, const string& variable_role) const
{
    const vector<Index> sample_indices = get_sample_indices(sample_role);

    const vector<Index> feature_indices = get_feature_indices(variable_role);

    MatrixR this_data(sample_indices.size(), feature_indices.size());

    fill_tensor_data(data, sample_indices, feature_indices, this_data.data());

    return this_data;
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


VectorR Dataset::get_sample_data(const Index sample_index, const vector<Index>& feature_indices) const
{
    const Index features_number = feature_indices.size();

    VectorR row(features_number);

#pragma omp parallel for
    for(Index i = 0; i < features_number; i++)
        row(i) = data(sample_index, feature_indices[i]);

    return row;
}


MatrixR Dataset::get_sample_input_data(const Index sample_index) const
{
    const Index input_features_number = get_features_number("Input");

    const vector<Index> input_feature_indices = get_feature_indices("Input");

    MatrixR inputs(1, input_features_number);

    for(Index i = 0; i < input_features_number; i++)
        inputs(0, i) = data(sample_index, input_feature_indices[i]);

    return inputs;
}


MatrixR Dataset::get_sample_target_data(const Index sample_index) const
{
    const vector<Index> target_feature_indices = get_feature_indices("Target");

    MatrixR sample_target_data(1, target_feature_indices.size());

    fill_tensor_data(data, vector<Index>(sample_index), target_feature_indices, sample_target_data.data());

    return sample_target_data;
}


Index Dataset::get_variable_index(const string& column_name) const
{
    const Index variables_number = get_variables_number();

    for(Index i = 0; i < variables_number; i++)
        if (variables[i].name == column_name)
            return i;

    throw runtime_error("Cannot find " + column_name + "\n");
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


VectorR Dataset::get_sample(const Index sample_index) const
{
    if (sample_index >= data.rows())
        throw runtime_error("Sample index out of bounds.");

    return data.row(sample_index);
}


string Dataset::get_sample_category(const Index sample_index, Index column_index_start) const
{
    if (variables[column_index_start].type != VariableType::Categorical)
        throw runtime_error("The specified variable is not of categorical type.");

    for(size_t variable_index = column_index_start; variable_index < variables.size(); variable_index++)
        if (data(sample_index, variable_index) == 1)
            return variables[column_index_start].categories[variable_index - column_index_start];

    throw runtime_error("Sample does not have a valid one-hot encoded category.");
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


const vector<vector<string>>& Dataset::get_data_file_preview() const
{
    return data_file_preview;
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

    set_default_variables_scalers();

    set_default_variables_roles();

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

    const Index new_inputs_number = new_input_shape.count();

    Index new_targets_number = new_target_shape.count();

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


void Dataset::set_display(bool new_display)
{
    display = new_display;
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


void Dataset::set_data_path(const filesystem::path& new_data_path)
{
    data_path = new_data_path;
}


void Dataset::set_has_header(bool new_has_header)
{
    has_header = new_has_header;
}


void Dataset::set_has_ids(bool new_has_ids)
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

    TimeSeriesDataset* ts_dataset = dynamic_cast<TimeSeriesDataset*>(this);
    if(ts_dataset)
        set_shape("Input", {ts_dataset->get_past_time_steps(), new_input_variables_number});
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

    for(const Dataset::Variable& variable : variables)
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
                    if (abs(data(used_sample_indices[k], feature_index) - type(1)) < NUMERIC_LIMITS_MIN)
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
                binary_frequencies(abs(data(used_sample_indices[j], feature_index) - type(1)) < NUMERIC_LIMITS_MIN
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
            {
                box_plots[i] = box_plot(data.col(feature_index), used_sample_indices);

                //used_variable_index++;
            }

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


Index Dataset::calculate_used_negatives(const Index target_index) const
{
    Index negatives = 0;

    const vector<Index> used_indices = get_used_sample_indices();

    const Index used_samples_number = used_indices.size();

    for(Index i = 0; i < used_samples_number; i++)
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


Index Dataset::calculate_negatives(const Index target_index, const string& sample_role) const
{
    Index negatives = 0;
    const vector<Index> indices = get_sample_indices(sample_role);
    const Index samples_number = get_samples_number(sample_role);

    for(Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = indices[i];
        type sample_value = data(sample_index, target_index);

        if (sample_role == "Testing")
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
                type threshold = (sample_role == "Training") ? type(1.0e-3) : type(NUMERIC_LIMITS_MIN);
                if (abs(sample_value - type(1)) > threshold)
                    throw runtime_error("Sample is neither a positive nor a negative: "
                                        + to_string(sample_value) + "-" + to_string(target_index) + "-" + to_string(sample_value));
            }
        }
    }

    return negatives;
}


vector<Descriptives> Dataset::calculate_feature_descriptives() const
{
    return descriptives(data);
}


vector<Descriptives> Dataset::calculate_variable_descriptives_positive_samples() const
{
    const Index target_index = get_feature_indices("Target")[0];

    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> input_feature_indices = get_feature_indices("Input");

    const Index samples_number = used_sample_indices.size();

    Index positive_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
        if (abs(data(used_sample_indices[i], target_index) - type(1)) < NUMERIC_LIMITS_MIN)
            positive_samples_number++;

    vector<Index> positive_used_sample_indices(positive_samples_number);
    Index positive_sample_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = used_sample_indices[i];

        if (abs(data(sample_index, target_index) - type(1)) < NUMERIC_LIMITS_MIN)
            positive_used_sample_indices[positive_sample_index++] = sample_index;
    }

    return descriptives(data, positive_used_sample_indices, input_feature_indices);
}


vector<Descriptives> Dataset::calculate_variable_descriptives_negative_samples() const
{
    const Index target_index = get_feature_indices("Target")[0];

    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> input_feature_indices = get_feature_indices("Input");

    const Index samples_number = used_sample_indices.size();

    Index negative_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
        if (data(used_sample_indices[i], target_index) < NUMERIC_LIMITS_MIN)
            negative_samples_number++;

    vector<Index> negative_used_sample_indices(negative_samples_number);
    Index negative_sample_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = used_sample_indices[i];

        if (data(sample_index, target_index) < NUMERIC_LIMITS_MIN)
            negative_used_sample_indices[negative_sample_index++] = sample_index;
    }

    return descriptives(data, negative_used_sample_indices, input_feature_indices);
}


vector<Descriptives> Dataset::calculate_variable_descriptives_categories(const Index class_index) const
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> input_feature_indices = get_feature_indices("Input");

    const Index samples_number = used_sample_indices.size();

    // Count used class samples

    Index class_samples_number = 0;

    for(Index i = 0; i < samples_number; i++)
        if (abs(data(used_sample_indices[i], class_index) - type(1)) < NUMERIC_LIMITS_MIN)
            class_samples_number++;

    vector<Index> class_used_sample_indices(class_samples_number, 0);

    Index class_sample_index = 0;

    for(Index i = 0; i < samples_number; i++)
    {
        const Index sample_index = used_sample_indices[i];

        if (abs(data(sample_index, class_index) - type(1)) < NUMERIC_LIMITS_MIN)
            class_used_sample_indices[class_sample_index++] = sample_index;
    }

    return descriptives(data, class_used_sample_indices, input_feature_indices);
}


vector<Descriptives> Dataset::calculate_feature_descriptives(const string& variable_role) const
{
    const vector<Index> used_sample_indices = get_used_sample_indices();

    const vector<Index> input_feature_indices = get_feature_indices(variable_role);

    return descriptives(data, used_sample_indices, input_feature_indices);
}


vector<Descriptives> Dataset::calculate_testing_target_variable_descriptives() const
{
    const vector<Index> testing_indices = get_sample_indices("Testing");

    const vector<Index> target_feature_indices = get_feature_indices("Target");

    return descriptives(data, testing_indices, target_feature_indices);
}


// VectorR Dataset::calculate_used_variables_minimums() const
// {
//     return column_minimums(data, get_used_sample_indices(), get_used_feature_indices());
// }


VectorR Dataset::calculate_means(const string& sample_role,
                                 const string& variable_role) const
{
    const vector<Index> sample_indices = get_sample_indices(sample_role);

    const vector<Index> feature_indices = get_feature_indices(variable_role);

    return mean(data, sample_indices, feature_indices);
}


Index Dataset::get_gmt() const
{
    return gmt;
}


void Dataset::set_gmt(const Index new_gmt)
{
    gmt = new_gmt;
}


Tensor<Correlation, 2> Dataset::calculate_input_target_variable_pearson_correlations() const
{
    if (display) cout << "Calculating pearson correlations..." << endl;

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
            correlations(i, j) = correlation(input_variable_data, target_variable_data);
        }
    }

    return correlations;
}


Tensor<Correlation, 2> Dataset::calculate_input_target_variable_spearman_correlations() const
{
    if (display) cout << "Calculating spearman correlations..." << endl;

    const Index input_variables_number = get_variables_number("Input");
    const Index target_variables_number = get_variables_number("Target");

    const vector<Index> input_variable_indices = get_variable_indices("Input");
    const vector<Index> target_variable_indices = get_variable_indices("Target");

    const vector<Index> used_sample_indices = get_used_sample_indices();

    Tensor<Correlation, 2> correlations(input_variables_number, target_variables_number);

    #pragma omp parallel for schedule(dynamic)
    for(Index i = 0; i < input_variables_number; i++)
    {
        const Index input_index = input_variable_indices[i];
        const MatrixR input_variable_data = get_variable_data(input_index, used_sample_indices);

        for(Index j = 0; j < target_variables_number; j++)
        {
            const Index target_index = target_variable_indices[j];
            const MatrixR target_variable_data = get_variable_data(target_index, used_sample_indices);
            correlations(i, j) = correlation_spearman(input_variable_data, target_variable_data);
        }
    }

    return correlations;
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


void Dataset::print_missing_values_information() const
{
    const Index missing_variables_number = count_variables_with_nan();
    const Index samples_with_missing_values = count_rows_with_nan();

    cout << "Missing values number: " << missing_values_number << " (" << missing_values_number * 100 / data.size() << "%)" << endl
         << "Variables with missing values: " << missing_variables_number
         << " (" << missing_variables_number * 100 / data.cols() << "%)" << endl
         << "Samples with missing values: "
         << samples_with_missing_values << " (" << samples_with_missing_values * 100 / data.rows() << "%)" << endl;
}


void Dataset::print_input_target_variables_correlations() const
{
    const Index inputs_number = get_features_number("Input");
    const Index targets_number = get_variables_number("Target");

    const vector<string> input_names = get_variable_names("Input");
    const vector<string> targets_name = get_variable_names("Target");

    const Tensor<Correlation, 2> correlations = calculate_input_target_variable_pearson_correlations();

    for(Index j = 0; j < targets_number; j++)
        for(Index i = 0; i < inputs_number; i++)
            cout << targets_name[j] << " - " << input_names[i] << ": " << correlations(i, j).r << endl;
}


void Dataset::print_top_input_target_variables_correlations() const
{
    const Index inputs_number = get_variables_number("Input");
    const Index targets_number = get_variables_number("Target");

    const vector<string> input_names = get_feature_names("Input");
    const vector<string> targets_name = get_feature_names("Target");

    const MatrixR correlations = get_correlation_values(calculate_input_target_variable_pearson_correlations());

    VectorR target_correlations(inputs_number);

    Tensor<string, 2> top_correlations(inputs_number, 2);

    map<type, string> top_correlation;

    for(Index i = 0; i < inputs_number; i++)
        for(Index j = 0; j < targets_number; j++)
            top_correlation.insert(pair<type, string>(correlations(i, j), input_names[i] + " - " + targets_name[j]));

    map<type, string>::iterator it;

    for(it = top_correlation.begin(); it != top_correlation.end(); it++)
        if (display) cout << "Correlation: " << (*it).first << "  between  " << (*it).second << endl;
}


Tensor<Correlation, 2> Dataset::calculate_input_variable_pearson_correlations() const
{
    if (display) cout << "Calculating pearson inputs correlations..." << endl;

    const vector<Index> input_variable_indices = get_variable_indices("Input");

    const Index input_variables_number = input_variable_indices.size();

    Tensor<Correlation, 2> correlations_pearson(input_variables_number, input_variables_number);

    for(Index i = 0; i < input_variables_number; i++)
    {
        if (display) cout << "Correlation " << i + 1<< " of " << input_variables_number << endl;

        const Index current_input_index_i = input_variable_indices[i];

        const MatrixR input_i = get_variable_data(current_input_index_i);

        if (is_constant(input_i)) continue;

        correlations_pearson(i, i).set_perfect();
        correlations_pearson(i, i).method = Correlation::Method::Pearson;

        for(Index j = i + 1; j < input_variables_number; j++)
        {
            const Index current_input_index_j = input_variable_indices[j];

            const MatrixR input_j = get_variable_data(current_input_index_j);
            correlations_pearson(i, j) = correlation(input_i, input_j);

            if (correlations_pearson(i, j).r > type(1) - NUMERIC_LIMITS_MIN)
                correlations_pearson(i, j).r = type(1);

            correlations_pearson(j, i) = correlations_pearson(i, j);
        }
    }

    return correlations_pearson;
}


Tensor<Correlation, 2> Dataset::calculate_input_variable_spearman_correlations() const
{
    if (display) cout << "Calculating spearman inputs correlations..." << endl;

    const vector<Index> input_variable_indices = get_variable_indices("Input");

    const Index input_variables_number = get_variables_number("Input");

    Tensor<Correlation, 2> correlations_spearman(input_variables_number, input_variables_number);

    for(Index i = 0; i < input_variables_number; i++)
    {
        if (display) cout << "Correlation " << i + 1 << " of " << input_variables_number << endl;

        const Index input_variable_index_i = input_variable_indices[i];

        const MatrixR input_i = get_variable_data(input_variable_index_i);

        if (is_constant(input_i)) continue;

        correlations_spearman(i, i).set_perfect();
        correlations_spearman(i, i).method = Correlation::Method::Spearman;

        for(Index j = i + 1; j < input_variables_number; j++)
        {
            const Index input_variable_index_j = input_variable_indices[j];

            const MatrixR input_j = get_variable_data(input_variable_index_j);

            correlations_spearman(i, j) = correlation_spearman(input_i, input_j);

            if (correlations_spearman(i, j).r > type(1) - NUMERIC_LIMITS_MIN)
                correlations_spearman(i, j).r = type(1);

            correlations_spearman(j, i) = correlations_spearman(i, j);
        }
    }

    return correlations_spearman;
}

void Dataset::print_inputs_correlations() const
{
    const MatrixR input_correlations = get_correlation_values(calculate_input_variable_pearson_correlations());

    cout << input_correlations << endl;
}


void Dataset::print_data_file_preview() const
{
    const Index size = data_file_preview.size();

    for(Index i = 0; i < size; i++)
    {
        for(size_t j = 0; j < data_file_preview[i].size(); j++)
            cout << data_file_preview[i][j] << " ";

        cout << endl;
    }
}


void Dataset::print_top_inputs_correlations() const
{
    const Index features_number = get_features_number("Input");

    const vector<string> variables_name = get_feature_names("Input");

    const MatrixR variables_correlations = get_correlation_values(calculate_input_variable_pearson_correlations());

    const Index correlations_number = features_number * (features_number - 1) / 2;

    Tensor<string, 2> top_correlations(correlations_number, 3);

    map<type, string> top_correlation;

    for(Index i = 0; i < features_number; i++)
    {
        for(Index j = i; j < features_number; j++)
        {
            if (i == j) continue;

            top_correlation.insert(pair<type, string>(variables_correlations(i, j), variables_name[i] + " - " + variables_name[j]));
        }
    }

    map<type, string> ::iterator it;

    for(it = top_correlation.begin(); it != top_correlation.end(); it++)
        if (display) cout << "Correlation: " << (*it).first << "  between  " << (*it).second << endl;
}


VectorI Dataset::calculate_correlations_rank() const
{
    const Tensor<Correlation, 2> correlations = calculate_input_target_variable_pearson_correlations();

    const MatrixR absolute_correlations = get_correlation_values(correlations).array().abs();

    const VectorR absolute_mean_correlations = absolute_correlations.rowwise().mean();

    return calculate_rank_less(absolute_mean_correlations);
}




void Dataset::set_default_variables_scalers()
{
    for(Dataset::Variable& variable : variables)
        variable.scaler = (variable.type == VariableType::Numeric)
                                  ? "MeanStandardDeviation"
                                  : "MinimumMaximum";
}


vector<Descriptives> Dataset::scale_data()
{
    const Index features_number = get_features_number();

    const vector<Descriptives> variable_descriptives = calculate_feature_descriptives();

    Index variable_index;

    for(Index i = 0; i < features_number; i++)
    {
        variable_index = get_variable_index(i);

        const string& scaler = variables[variable_index].scaler;

        if(scaler == "None")
            continue;
        else if(scaler == "MinimumMaximum")
            scale_minimum_maximum(data, i, variable_descriptives[i]);
        else if(scaler == "MeanStandardDeviation")
            scale_mean_standard_deviation(data, i, variable_descriptives[i]);
        else if(scaler == "StandardDeviation")
            scale_standard_deviation(data, i, variable_descriptives[i]);
        else if(scaler == "Logarithm")
            scale_logarithmic(data, i);
        else
            throw runtime_error("Unknown scaler: " + scaler + "\n");
    }

    return variable_descriptives;
}


vector<Descriptives> Dataset::scale_features(const string& variable_role)
{
    const Index input_features_number = get_features_number(variable_role);

    const vector<Index> input_feature_indices = get_feature_indices(variable_role);
    const vector<string> input_variable_scalers = get_feature_scalers(variable_role);

    const vector<Descriptives> input_variable_descriptives = calculate_feature_descriptives(variable_role);

    for(Index i = 0; i < input_features_number; i++)
    {
        const string& scaler = input_variable_scalers[i];

        if(scaler == "None")
            continue;
        else if(scaler == "MinimumMaximum")
            scale_minimum_maximum(data, input_feature_indices[i], input_variable_descriptives[i]);
        else if(scaler == "MeanStandardDeviation")
            scale_mean_standard_deviation(data, input_feature_indices[i], input_variable_descriptives[i]);
        else if(scaler == "StandardDeviation")
            scale_standard_deviation(data, input_feature_indices[i], input_variable_descriptives[i]);
        else if(scaler == "Logarithm")
            scale_logarithmic(data, input_feature_indices[i]);
        else
            throw runtime_error("Unknown scaling inputs method: " + scaler + "\n");
    }

    return input_variable_descriptives;
}


void Dataset::unscale_features(const string& variable_role,
                                const vector<Descriptives>& input_variable_descriptives)
{
    const Index features_number = get_features_number(variable_role);

    const vector<Index> variables_indices = get_feature_indices(variable_role);

    const vector<string> variables_scalers = get_feature_scalers(variable_role);

    for(Index i = 0; i < features_number; i++)
    {
        const string& scaler = variables_scalers[i];

        if(scaler == "None")
            continue;
        else if(scaler == "MinimumMaximum")
            unscale_minimum_maximum(data, variables_indices[i], input_variable_descriptives[i]);
        else if(scaler == "MeanStandardDeviation")
            unscale_mean_standard_deviation(data, variables_indices[i], input_variable_descriptives[i]);
        else if(scaler == "StandardDeviation")
            unscale_standard_deviation(data, variables_indices[i], input_variable_descriptives[i]);
        else if(scaler == "Logarithm")
            unscale_logarithmic(data, variables_indices[i]);
        else if(scaler == "ImageMinMax")
            unscale_image_minimum_maximum(data, variables_indices[i]);
        else
            throw runtime_error("Unknown unscaling and unscaling method: " + scaler + "\n");
    }
}


void Dataset::set_data_constant(const type new_value)
{
    const vector<Index> input_indices = get_feature_indices("Input");

    const Index samples_number = get_samples_number();

    for(Index i = 0; i < samples_number; ++i)
        for(Index index : input_indices)
            data(i, index) = new_value;
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
    add_xml_element(printer, "FileType", "csv");

    add_xml_element(printer, "Path", data_path.string());

    add_xml_element(printer, "Separator", get_separator_name());
    add_xml_element(printer, "HasHeader", to_string(has_header));
    add_xml_element(printer, "HasSamplesId", to_string(has_sample_ids));
    add_xml_element(printer, "MissingValuesLabel", missing_values_label);
    add_xml_element(printer, "Codification", get_codification_string());
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

    const XMLElement* start_element = variables_element->FirstChildElement("VariablesNumber");

    for(size_t i = 0; i < variables.size(); i++)
    {
        Variable& variable = variables[i];
        const XMLElement* variable_element = start_element->NextSiblingElement("Variable");
        start_element = variable_element;

        if (variable_element->Attribute("Item") != to_string(i + 1))
            throw runtime_error("Variable item number (" + to_string(i + 1) + ") does not match (" + variable_element->Attribute("Item") + ").\n");

        variable.name = read_xml_string(variable_element, "Name");
        variable.set_scaler(read_xml_string(variable_element, "Scaler"));
        variable.set_role(read_xml_string(variable_element, "Role"));
        variable.set_type(read_xml_string(variable_element, "Type"));

        if (variable.type == VariableType::Categorical || variable.type == VariableType::Binary)
        {
            const XMLElement* categories_element = variable_element->FirstChildElement("Categories");

            if (categories_element)
                variable.categories = get_tokens(read_xml_string(variable_element, "Categories"), ";");
            else if (variable.type == VariableType::Binary)
                variable.categories = { "0", "1" };
            else
                throw runtime_error("Categorical Variable Element is nullptr: Categories");
        }
    }
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

        data.resize(samples_number, all_feature_indices[all_feature_indices.size() - 1][all_feature_indices[all_feature_indices.size() - 1].size() - 1] + 1);
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

    if (missing_values_number > 0)
    {
        set_missing_values_method(read_xml_string(missing_values_element, "MissingValuesMethod"));

        const string variables_string = read_xml_string(missing_values_element, "VariablesMissingValuesNumber");
        const vector<string> tokens = get_tokens(variables_string, " ");

        vector<Index> valid_numbers;
        valid_numbers.reserve(tokens.size());

        for(const string& token : tokens)
            if(!token.empty())
                valid_numbers.push_back(stoi(token));

        variables_missing_values_number.resize(valid_numbers.size());
        for(size_t i = 0; i < valid_numbers.size(); ++i)
            variables_missing_values_number(i) = valid_numbers[i];

        rows_missing_values_number = read_xml_index(missing_values_element, "SamplesMissingValuesNumber");
    }
}

void Dataset::preview_data_from_XML(const XMLElement *preview_data_element)
{
    if(!preview_data_element)
        throw runtime_error("Preview data element is nullptr.\n ");

    const XMLElement* preview_size_element = preview_data_element->FirstChildElement("PreviewSize");

    if(!preview_size_element)
        throw runtime_error("Preview size element is nullptr.\n ");

    Index preview_size = 0;
    if (preview_size_element->GetText())
        preview_size = static_cast<Index>(atoi(preview_size_element->GetText()));

    const XMLElement* start_element = preview_size_element;

    if(preview_size > 0){
        data_file_preview.resize(preview_size);

        for(Index i = 0; i < preview_size; ++i) {
            const XMLElement* row_data = start_element->NextSiblingElement("Row");
            start_element = row_data;

            if (row_data->Attribute("Item") != to_string(i + 1))
                throw runtime_error("Row item number (" + to_string(i + 1) + ") does not match (" + row_data->Attribute("Item") + ").\n");

            if(row_data->GetText())
                data_file_preview[i] = get_tokens(row_data->GetText(), ",");
        }
    }
}

void Dataset::from_XML(const XMLDocument& data_set_document)
{
    const XMLElement* data_set_element = data_set_document.FirstChildElement("Dataset");
    if(!data_set_element)
        throw runtime_error("Dataset element is nullptr.\n");

    // Data Source
    const XMLElement* data_source_element = data_set_element->FirstChildElement("DataSource");

    if(!data_source_element)
        throw runtime_error("Data source element is nullptr.\n");

    set_data_path(read_xml_string(data_source_element, "Path"));
    set_separator_name(read_xml_string(data_source_element, "Separator"));
    set_has_header(read_xml_bool(data_source_element, "HasHeader"));
    set_has_ids(read_xml_bool(data_source_element, "HasSamplesId"));
    set_missing_values_label(read_xml_string(data_source_element, "MissingValuesLabel"));
    set_codification(read_xml_string(data_source_element, "Codification"));

    // Variables
    const XMLElement* variables_element = data_set_element->FirstChildElement("Variables");

    variables_from_XML(variables_element);

    // Samples
    const XMLElement* samples_element = data_set_element->FirstChildElement("Samples");

    samples_from_XML(samples_element);


    const XMLElement* missing_values_element = data_set_element->FirstChildElement("MissingValues");

    missing_values_from_XML(missing_values_element);

    const XMLElement* preview_data_element = data_set_element->FirstChildElement("PreviewData");

    preview_data_from_XML(preview_data_element);

    set_display(read_xml_bool(data_set_element, "Display"));

    input_shape = { get_features_number("Input") };
    target_shape = { get_features_number("Target") };
}


void Dataset::print() const
{
    if(!display) return;

    const Index features_number = get_features_number();
    const Index input_features_number = get_features_number("Input");
    const Index samples_number = get_samples_number();
    const Index target_variables_number = get_features_number("Target");
    const Index training_samples_number = get_samples_number("Training");
    const Index validation_samples_number = get_samples_number("Validation");
    const Index testing_samples_number = get_samples_number("Testing");
    const Index unused_samples_number = get_samples_number("None");

    cout << "Dataset object summary:\n"
         << "Number of samples: " << samples_number << "\n"
         << "Number of variables: " << features_number << "\n"
         << "Number of input variables: " << input_features_number << "\n"
         << "Number of target variables: " << target_variables_number << "\n"
         << "Input shape: " << get_shape("Input") << "\n"
         << "Target shape: " << get_shape("Target");

    cout << "Number of training samples: " << training_samples_number << endl
         << "Number of selection samples: " << validation_samples_number << endl
         << "Number of testing samples: " << testing_samples_number << endl
         << "Number of unused samples: " << unused_samples_number << endl;

    //for(const Dataset::Variable& variable : variables)
    //    variable.print();
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
    XMLDocument document;

    if (document.LoadFile(file_name.string().c_str()))
        throw runtime_error("Cannot load XML file " + file_name.string() + ".\n");

    from_XML(document);
}


void Dataset::print_variables() const
{
    for(const Dataset::Variable& variable : variables)
        variable.print();

    cout << endl;
}


void Dataset::print_data() const
{
    cout << data << endl;
}


void Dataset::print_data_preview() const
{
    if(!display) return;

    const Index samples_number = get_samples_number();

    if (samples_number > 0)
    {
        const VectorR first_sample = data.row(0);

        cout << "First sample: \n";

        for(int i = 0; i < first_sample.size(); i++)
            cout << first_sample(i) << "  ";
    }

    if (samples_number > 1)
    {
        const VectorR second_sample = data.row(1);

        cout << "Second sample: \n";

        for(int i = 0; i < second_sample.size(); i++)
            cout << second_sample(i) << "  ";
    }

    if (samples_number > 2)
    {
        const VectorR last_sample = data.row(samples_number - 1);

        cout << "Last sample: \n";

        for(int i = 0; i < last_sample.size(); i++)
            cout << last_sample(i) << "  ";
    }

    cout << endl;
}


void Dataset::save_data() const
{
    ofstream file(data_path.c_str());

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

        Index target_index = target_feature_indices[0];

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


vector<vector<Index>> Dataset::calculate_Tukey_outliers(const type cleaning_parameter) const
{
    const Index samples_number = get_used_samples_number();
    const vector<Index> sample_indices = get_used_sample_indices();

    const Index variables_number = get_variables_number();
    const Index used_variables_number = get_used_variables_number();
    const vector<Index> used_variables_indices = get_used_variables_indices();

    vector<vector<Index>> return_values(2);

    return_values[0].resize(samples_number, 0);
    return_values[1].resize(used_variables_number, 0);

    const vector<BoxPlot> box_plots = calculate_variables_box_plots();

    Index feature_index = 0;
    Index used_feature_index = 0;

    //#pragma omp parallel for

    for(Index i = 0; i < variables_number; i++)
    {
        const Variable& variable = variables[i];

        if (variable.role == "None"
            && variable.type == VariableType::Categorical)
        {
            feature_index += variable.get_categories_number();
            continue;
        }
        else if (variable.role == "None") // Numeric, Binary or DateTime
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

            if (interquartile_range < numeric_limits<type>::epsilon())
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
    const Index samples_number = get_used_samples_number();
    const vector<Index> sample_indices = get_used_sample_indices();

    const Index variables_number = get_variables_number();
    const Index used_variables_number = get_used_variables_number();
    const vector<Index> used_variables_indices = get_used_variables_indices();

    vector<vector<Index>> return_values(2);

    return_values[0].resize(samples_number, 0);
    return_values[1].resize(used_variables_number, 0);

    const vector<BoxPlot> box_plots = calculate_variables_box_plots();

    Index feature_index = 0;
    Index used_feature_index = 0;

#pragma omp parallel for

    for(Index i = 0; i < variables_number; i++)
    {
        const Variable& variable = variables[i];

        if (variable.role == "None"
            && variable.type == VariableType::Categorical)
        {
            feature_index += variable.get_categories_number();
            continue;
        }
        else if (variable.role == "None") // Numeric, Binary or DateTime
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

            if (interquartile_range < numeric_limits<type>::epsilon())
            {
                feature_index++;
                used_feature_index++;
                continue;
            }

            Index variables_outliers = 0;

            for(Index j = 0; j < samples_number; j++)
            {
                const VectorR sample = get_sample_data(sample_indices[Index(j)]);

                if (sample[feature_index] < (box_plots[i].first_quartile - cleaning_parameter * interquartile_range)
                    || sample[feature_index] > (box_plots[i].third_quartile + cleaning_parameter * interquartile_range))
                {
                    return_values[0][Index(j)] = 1;

                    variables_outliers++;

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


VectorI Dataset::filter_data(const VectorR& minimums,
                                      const VectorR& maximums)
{
    const vector<Index> used_feature_indices = get_used_feature_indices();

    const Index used_features_number = used_feature_indices.size();

    const Index samples_number = get_samples_number();

    VectorR filtered_indices(samples_number);
    filtered_indices.setZero();

    const vector<Index> used_sample_indices = get_used_sample_indices();
    const Index used_samples_number = used_sample_indices.size();

    Index sample_index = 0;

    for(Index i = 0; i < used_features_number; i++)
    {
        for(Index j = 0; j < used_samples_number; j++)
        {
            sample_index = used_sample_indices[j];

            const type value = data(sample_index, used_feature_indices[i]);

            if (get_sample_role(sample_index) == "None"
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
                    set_sample_role(sample_index, "None");
                }
            }
            else if (value < minimums(i) || value > maximums(i))
            {
                filtered_indices(sample_index) = type(1);
                set_sample_role(sample_index, "None");
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

    VectorI filtered_samples_indices(filtered_samples_number);

    Index index = 0;

    for(Index i = 0; i < samples_number; i++)
        if (filtered_indices(i) > type(0.5))
            filtered_samples_indices(index++) = i;

    return filtered_samples_indices;
}


void Dataset::impute_missing_values_unuse()
{
    const Index samples_number = get_samples_number();

#pragma omp parallel for

    for(Index i = 0; i < samples_number; i++)
        if (has_nan_row(i))
            set_sample_role(i, "None");
}


void Dataset::impute_missing_values_mean()
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_feature_indices = get_used_feature_indices();
    const vector<Index> input_feature_indices = get_feature_indices("Input");
    const vector<Index> target_feature_indices = get_feature_indices("Target");

    if (used_sample_indices.empty() || used_feature_indices.empty())
        return;

    const VectorR means = mean(data, used_sample_indices, used_feature_indices);

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
                data(current_sample, current_variable) = means(j);
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


void Dataset::impute_missing_values_median()
{
    const vector<Index> used_sample_indices = get_used_sample_indices();
    const vector<Index> used_feature_indices = get_used_feature_indices();
    const vector<Index> input_feature_indices = get_feature_indices("Input");
    const vector<Index> target_feature_indices = get_feature_indices("Target");

    const VectorR medians = median(data, used_sample_indices, used_feature_indices);

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
                data(current_sample, current_variable) = medians(j);
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
        impute_missing_values_mean();
        break;

    case MissingValuesMethod::Median:
        impute_missing_values_median();
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


DateFormat Dataset::infer_dataset_date_format(const vector<Dataset::Variable>& variables,
                                              const vector<vector<string>>& sample_rows,
                                              bool has_sample_ids,
                                              const string& missing_values_label)
{
    for(size_t col_idx = 0; col_idx < variables.size(); ++col_idx)
    {
        if(variables[col_idx].type != Dataset::VariableType::DateTime)
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
                int part1 = stoi(date_parts[1].str());
                int part2 = stoi(date_parts[2].str());

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
        std::set<string> unique_elements;

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
    {
        if(has_sample_ids)
            for(Index i = 0; i < variables_number; i++) variables[i].name = header_tokens[i + 1];
        else
            set_variable_names(header_tokens);
    }
    else
    {
        set_default_variable_names();
    }

    infer_column_types(raw_file_content);

    const DateFormat date_format = infer_dataset_date_format(variables, raw_file_content, has_sample_ids, missing_values_label);

    for(Dataset::Variable& variable : variables)
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
                    time_t timestamp = date_to_timestamp(token, gmt, date_format);

                    if(timestamp == -1)
                        throw runtime_error("Date format is unsupported or date is prior to 1970.");
                    else
                        data(sample_index, feature_indices[0]) = timestamp;
                }
                break;
            case VariableType::Categorical:
                if(token.empty() || token == missing_values_label)
                    for(Index cat_idx : feature_indices) data(sample_index, cat_idx) = NAN;
                else
                {
                    auto it = find(variable.categories.begin(), variable.categories.end(), token);
                    if(it != variable.categories.end())
                    {
                        Index category_index = distance(variable.categories.begin(), it);
                        data(sample_index, feature_indices[category_index]) = 1;
                    }
                }
                break;
            case VariableType::Binary:
                if(contains(positive_words, token) || contains(negative_words, token))
                    data(sample_index, feature_indices[0]) = contains(positive_words, token) ? 1 : 0;
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
                        data(sample_index, feature_indices[0]) = stof(token);
                }
                break;
            default:
                break;
            }
        }
    }

    unuse_constant_variables();
    set_binary_variables();
    split_samples_random();
}


string Dataset::Variable::get_type_string() const
{
    switch (type)
    {
    case VariableType::None:
        return "None";
    case VariableType::Numeric:
        return "Numeric";
    case VariableType::Constant:
        return "Constant";
    case VariableType::Binary:
        return "Binary";
    case VariableType::Categorical:
        return "Categorical";
    case VariableType::DateTime:
        return "DateTime";
    default:
        throw runtime_error("Unknown variable type");
    }
}


void Dataset::read_data_file_preview(const vector<vector<string>>& all_rows)
{
    if (all_rows.empty())
        return;

    const Index num_first_rows_to_show = 3;

    data_file_preview.clear();

    for(Index i = 0; i < Index(min((size_t)num_first_rows_to_show, all_rows.size())); ++i)
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


void Dataset::fill_input_tensor(const vector<Index>& sample_indices, const vector<Index>& input_indices, type* input_data) const
{
    fill_tensor_data(data, sample_indices, input_indices, input_data);
}


void Dataset::fill_input_tensor_row_major(const vector<Index>& sample_indices, const vector<Index>& input_indices, type* input_data) const
{
    fill_tensor_data_row_major(data, sample_indices, input_indices, input_data);
}


// void Dataset::fill_decoder_tensor(const vector<Index>& sample_indices, const vector<Index>& decoder_indices, type* decoder_tensor_data) const
// {
//     fill_tensor_data(data, sample_indices, decoder_indices, decoder_tensor_data);
// }


void Dataset::fill_target_tensor(const vector<Index>& sample_indices, const vector<Index>& target_indices, type* target_tensor_data) const
{
    fill_tensor_data(data, sample_indices, target_indices, target_tensor_data);
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
    for(const Dataset::Variable& variable : variables)
        if (variable.type == VariableType::Binary || variable.type == VariableType::Categorical)
            return true;

    return false;
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
    const Index variables_number = get_variables_number();
    const Index rows_number = get_samples_number();

    VectorI variables_with_nan(variables_number);
    variables_with_nan.setZero();

#pragma omp parallel for
    for(Index variable_index = 0; variable_index < variables_number; variable_index++)
    {
        const Index current_variable_index = get_feature_indices(variable_index)[0];

        Index counter = 0;

        for(Index row_index = 0; row_index < rows_number; row_index++)
            if (isnan(data(row_index, current_variable_index)))
                counter++;

        variables_with_nan(variable_index) = counter;
    }

    return variables_with_nan;
}


Index Dataset::count_variables_with_nan() const
{
    VectorI variables_with_nan = count_nans_per_variable();

    Index missing_variables_number = 0;

    for(Index i = 0; i < variables_with_nan.size(); i++)
        if (variables_with_nan(i) > 0)
            missing_variables_number++;

    return missing_variables_number;
}


Index Dataset::count_rows_with_nan() const
{
    Index rows_with_nan = 0;

    const Index rows_number = data.rows();
    const Index variables_number = data.cols();

    bool has_nan = true;

    for(Index row_index = 0; row_index < rows_number; row_index++)
    {
        has_nan = false;

        for(Index variable_index = 0; variable_index < variables_number; variable_index++)
        {
            if (isnan(data(row_index, variable_index)))
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
    map<string, Index> variables_count_map;

    for(const Dataset::Variable& variable : variables)
    {
        auto result = variables_count_map.insert(pair<string, Index>(variable.name, 1));
        if(!result.second)
            result.first->second++;
    }

    for(const auto& element : variables_count_map)
    {
        if (element.second > 1)
        {
            const string repeated_name = element.first;
            Index repeated_index = 1;

            for(Dataset::Variable& variable : variables)
                if (variable.name == repeated_name)
                    variable.name = variable.name + "_" + to_string(repeated_index++);
        }
    }

    if (has_categorical_variables() || has_binary_variables())
    {
        const vector<string> all_feature_names = get_feature_names();

        map<string, Index> global_names_count;

        for(const string& name : all_feature_names)
            global_names_count[name]++;

        for(Dataset::Variable& variable : variables)
        {
            bool needs_disambiguation = false;

            if (variable.type == VariableType::Categorical)
            {
                for(const string& category : variable.categories)
                {
                    if (global_names_count[category] > 1)
                    {
                        needs_disambiguation = true;
                        break;
                    }
                }

                if (needs_disambiguation)
                    for(string& category : variable.categories)
                        category += "_" + variable.name;

            }
            else if (variable.type == VariableType::Binary)
            {
                for(const string& category : variable.categories)
                {
                    if (global_names_count[category] > 1)
                    {
                        needs_disambiguation = true;
                        break;
                    }
                }

                if (needs_disambiguation)
                    for(string& category : variable.categories)
                        category += "_" + variable.name;
            }
        }
    }
}


vector<vector<Index>> Dataset::split_samples(const vector<Index>& sample_indices, Index new_batch_size) const
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
    for(Index i = 0; i < batches_number; i++)
    {
        const Index start_index = i * batch_size;
        const Index end_index = min(start_index + batch_size, samples_number);

        batches[i].assign(sample_indices.begin() + start_index,
                          sample_indices.begin() + end_index);
    }

    return batches;
}


bool Dataset::get_has_rows_labels() const
{
    return has_sample_ids;
}


// void Dataset::decode(string&) const
// {
//     switch (codification)
//     {
//     case Dataset::Codification::SHIFT_JIS:
//         //        input_string = sj2utf8(input_string);
//         break;
//     default:
//         break;
//     }
// }


void Batch::fill(const vector<Index>& sample_indices,
                 const vector<Index>& input_indices,
                 // const vector<Index>& decoder_indices,
                 const vector<Index>& target_indices)
{
    dataset->fill_input_tensor(sample_indices, input_indices, input_tensor.data());

    // if (dynamic_cast<TimeSeriesDataset*>(dataset))
    // {
    //    input_shape.clear();
    //    input_shape.push_back(sample_indices.size());
    //    input_shape.push_back(input_indices.size());
    //    input_shape.push_back(input_indices.size());
    // }

    dataset->fill_target_tensor(sample_indices, target_indices, target_tensor.data());

    // dataset->fill_decoder_tensor(sample_indices, decoder_indices, decoder_tensor.data());
}


Batch::Batch(const Index new_samples_number, const Dataset* new_dataset)
{
    set(new_samples_number, new_dataset);
}


void Batch::set(const Index new_samples_number, const Dataset* new_dataset)
{
    if(!new_dataset) return;

    samples_number = new_samples_number;

    dataset = const_cast<Dataset*>(new_dataset);

    // Inputs

    const Shape& data_set_input_dimensions = dataset->get_shape("Input");

    if(!data_set_input_dimensions.empty())
    {
        input_shape = prepend(samples_number, data_set_input_dimensions);
        input_tensor.resize(get_size(input_shape));
    }

    // Targets

    const Shape& data_set_target_shape = dataset->get_shape("Target");

    if(!data_set_target_shape.empty())
    {
        target_shape = prepend(samples_number, data_set_target_shape);
        target_tensor.resize(get_size(target_shape));
    }

    // Decoder

    // const Shape& data_set_decoder_dimensions = dataset->get_shape("Decoder");

    // if(!data_set_decoder_dimensions.empty())
    // {
    //     decoder_shape = prepend(samples_number, data_set_decoder_dimensions);
    //     decoder_tensor.resize(get_size(decoder_shape));
    // }
}


Index Batch::get_samples_number() const
{
    return samples_number;
}


void Batch::print() const
{
    cout << "Batch" << endl
         << "Inputs:" << endl
         << "Input shape:" << input_shape << endl;

    if (input_shape.size() == 4)
        cout << TensorMap4((type*)input_tensor.data(),
                                           input_shape[0],
                                           input_shape[1],
                                           input_shape[2],
                                           input_shape[3]);
    else if (input_shape.size() == 3)
        cout << TensorMap3((type*)input_tensor.data(),
                                           input_shape[0],
                                           input_shape[1],
                                           input_shape[2]);
    else if (input_shape.size() == 2)
        cout << MatrixMap((type*)input_tensor.data(),
                                           input_shape[0],
                                           input_shape[1]);

    cout << endl;

    // cout << "Decoder:" << endl
    //      << "Decoder shape:" << decoder_shape << endl;
/*
    cout << "Targets:" << endl
         << "Target shape:" << target_shape << endl;
*/
    cout << MatrixMap((type*)target_tensor.data(),
                                       target_shape[0],
                                       target_shape[1]) << endl;
}


bool Batch::is_empty() const
{
    return input_tensor.size() == 0;
}



vector<TensorView> Batch::get_inputs() const
{
    vector<TensorView> input_views = {{(type*)input_tensor.data(), input_shape}};

    // @todo DECODER VARIABLES
    // if(!decoder_shape.empty())
    //     input_views.insert(input_views.begin(), {(type*)decoder_tensor.data(), decoder_shape});

    return input_views;
}


TensorView Batch::get_targets() const
{
    return {(type*)target_tensor.data() , target_shape};
}


#ifdef OPENNN_CUDA
    
void BatchCuda::fill(const vector<Index>& sample_indices,
                     const vector<Index>& input_indices,
                     //const vector<Index>& decoder_indices,
                     const vector<Index>& target_indices)
{
    fill_host(sample_indices, input_indices, target_indices);

    const Index batch_size = sample_indices.size();

    copy_device(batch_size);
}


void BatchCuda::fill_host(const vector<Index>& sample_indices,
                          const vector<Index>& input_indices,
                          //const vector<Index>& decoder_indices,
                          const vector<Index>& target_indices)
{
    if (const ImageDataset* image_dataset = dynamic_cast<ImageDataset*>(dataset))
        image_dataset->fill_input_tensor_row_major(sample_indices, input_indices, inputs_host);
    else
        dataset->fill_input_tensor(sample_indices, input_indices, inputs_host);

    //dataset->fill_decoder_tensor(sample_indices, decoder_indices, decoder_host);

    dataset->fill_target_tensor(sample_indices, target_indices, targets_host);
}



BatchCuda::BatchCuda(const Index new_samples_number, Dataset* new_dataset)
{
    set(new_samples_number, new_dataset);
}


BatchCuda::~BatchCuda()
{
    if (inputs_host)
    {
        cudaFreeHost(inputs_host);
        inputs_host = nullptr;
    }

    if (decoder_host)
    {
        cudaFreeHost(decoder_host);
        decoder_host = nullptr;
    }

    if (targets_host)
    {
        cudaFreeHost(targets_host);
        targets_host = nullptr;
    }
}


void BatchCuda::set(const Index new_samples_number, Dataset* new_dataset)
{
    if(!new_dataset) return;

    samples_number = new_samples_number;
    dataset = new_dataset;

    const Shape& data_set_input_dimensions = dataset->get_shape("Input");
    //const Shape& data_set_decoder_dimensions = dataset->get_shape("Decoder");
    const Shape& data_set_target_shape = dataset->get_shape("Target");

    if(!data_set_input_dimensions.empty())
    {
        num_input_features = dataset->get_features_number("Input");
        const Index input_size = samples_number * num_input_features;

        input_shape = { samples_number };
        input_shape.insert(input_shape.end(), data_set_input_dimensions.begin(), data_set_input_dimensions.end());

        if (input_size > inputs_host_allocated_size)
        {
            if (inputs_host) cudaFreeHost(inputs_host);
            CHECK_CUDA(cudaMallocHost(&inputs_host, input_size * sizeof(float)));
            inputs_host_allocated_size = input_size;
        }

        inputs_device.resize({samples_number, num_input_features});
    }
    /*
    if(!data_set_decoder_dimensions.empty())
    {
        decoder_shape = { samples_number };
        decoder_shape.insert(decoder_shape.end(), data_set_decoder_dimensions.begin(), data_set_decoder_dimensions.end());

        const Index decoder_size = decoder_shape.count();

        if (decoder_size > decoder_host_allocated_size)
        {
            if (decoder_host) cudaFreeHost(decoder_host);
            CHECK_CUDA(cudaMallocHost(&decoder_host, decoder_size * sizeof(float)));
            decoder_host_allocated_size = decoder_size;
        }

        CHECK_CUDA(cudaMalloc(&decoder_device, decoder_size * sizeof(float)));
    }
    */
    if(!data_set_target_shape.empty())
    {
        num_target_features = dataset->get_features_number("Target");
        const Index target_size = samples_number * num_target_features;

        target_shape = { samples_number };
        target_shape.insert(target_shape.end(), data_set_target_shape.begin(), data_set_target_shape.end());

        if (target_size > targets_host_allocated_size)
        {
            if (targets_host) cudaFreeHost(targets_host);
            CHECK_CUDA(cudaMallocHost(&targets_host, target_size * sizeof(float)));
            targets_host_allocated_size = target_size;
        }

        targets_device.resize({samples_number, num_target_features});
    }
}


void BatchCuda::copy_device(const Index current_batch_size)
{
    const Index input_size = current_batch_size * num_input_features;
    const Index target_size = current_batch_size * num_target_features;

    CHECK_CUDA(cudaMemcpy(inputs_device.data, inputs_host, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(targets_device.data, targets_host, target_size * sizeof(float), cudaMemcpyHostToDevice));
}


void BatchCuda::copy_device_async(const Index current_batch_size, cudaStream_t stream)
{
    const Index input_size = current_batch_size * num_input_features;
    const Index target_size = current_batch_size * num_target_features;

    CHECK_CUDA(cudaMemcpyAsync(inputs_device.data, inputs_host, input_size * sizeof(float), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(targets_device.data, targets_host, target_size * sizeof(float), cudaMemcpyHostToDevice, stream));
}


Tensor2 BatchCuda::get_inputs_from_device() const
{
    const Index inputs_number = dataset->get_variables_number("Input");

    Tensor2 inputs(samples_number, inputs_number);

    inputs.setZero();

    CHECK_CUDA(cudaMemcpy(inputs.data(), inputs_device.data, samples_number * inputs_number * sizeof(type), cudaMemcpyDeviceToHost));

    return inputs;
}


Tensor2 BatchCuda::get_decoder_from_device() const
{
    const Index decoder_number = dataset->get_variables_number("Decoder");

    Tensor2 decoder(samples_number, decoder_number);

    decoder.setZero();

    CHECK_CUDA(cudaMemcpy(decoder.data(), inputs_device.data, samples_number * decoder_number * sizeof(type), cudaMemcpyDeviceToHost));

    return decoder;
}


Tensor2 BatchCuda::get_targets_from_device() const
{
    const Index targets_number = target_shape[1];

    Tensor2 targets(samples_number, targets_number);

    targets.setZero();

    CHECK_CUDA(cudaMemcpy(targets.data(), targets_device.data, samples_number * targets_number * sizeof(type), cudaMemcpyDeviceToHost));

    return targets;
}


vector<TensorViewCuda> BatchCuda::get_inputs_device() const
{
    return {{inputs_device.data, nullptr}};
}


TensorViewCuda BatchCuda::get_targets_device() const
{
    return {targets_device.data , nullptr};
}


Index BatchCuda::get_samples_number() const
{
    return samples_number;
}


void BatchCuda::print() const
{
    // @todo
}


bool BatchCuda::is_empty() const
{
    return input_shape.empty();
}

#endif

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
