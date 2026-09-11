// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#include "opennn/dataset/tabular_dataset.h"
#include "opennn/core/log.h"
#include "opennn/core/io_utilities.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/tensor_types.h"

#include <ranges>

namespace opennn
{

namespace
{

constexpr size_t maximum_expanded_data_bytes = size_t(2) * 1024 * 1024 * 1024;

bool looks_like_id_variable(const Variable& variable, const Index samples_number)
{
    if (!variable.is_categorical() || samples_number == 0)
        return false;

    const Index categories_number = ssize(variable.categories);

    if (categories_number * 20 >= samples_number * 19)
        return true;

    string name = variable.name;
    ranges::transform(name, name.begin(), [](unsigned char c) { return char(tolower(c)); });

    const bool id_name = name == "id"
                      || name.ends_with(" id")
                      || name.ends_with("_id")
                      || name.ends_with("-id")
                      || name.ends_with(".id");

    return id_name && categories_number * 10 >= samples_number * 9;
}

}

static float parse_float_or_nan(string_view token, const NumberFormat& number_format)
{
    float value;
    return parse_real(token, value, number_format) ? value : QUIET_NAN;
}

static bool is_missing_token(string_view token, string_view missing_label)
{
    return token.empty() || token == missing_label;
}

static void parse_numeric_token(float* row, Index feature_index,
                         string_view token, string_view missing_label,
                         const NumberFormat& number_format)
{
    row[feature_index] = is_missing_token(token, missing_label)
                       ? QUIET_NAN
                       : parse_float_or_nan(token, number_format);
}

static void parse_datetime_token(float* row, Index feature_index,
                          string_view token, string_view missing_label,
                          const DateFormat& date_format)
{
    if (is_missing_token(token, missing_label))
    {
        row[feature_index] = QUIET_NAN;
        return;
    }

    const time_t timestamp = date_to_timestamp(token, 0, date_format);
    throw_if(timestamp == -1, "Date format is unsupported or date is prior to 1970.");
    row[feature_index] = timestamp;
}

static void parse_categorical_token(float* row, const vector<Index>& feature_indices,
                             string_view token, string_view missing_label,
                             const unordered_map<string_view, Index>& category_map)
{
    if (is_missing_token(token, missing_label))
        for (const Index cat_index : feature_indices)
            row[cat_index] = QUIET_NAN;
    else
    {
        const auto it = category_map.find(token);
        if (it != category_map.end())
            row[feature_indices[it->second]] = 1;
    }
}

static void parse_binary_token(float* row, Index feature_index,
                        string_view token, string_view missing_label,
                        const vector<string>& categories,
                        const NumberFormat& number_format)
{
    row[feature_index] =
        contains(positive_words, token) ? 1.0f :
        contains(negative_words, token) ? 0.0f :
        is_missing_token(token, missing_label) ? QUIET_NAN :
        !categories.empty() && token == categories[0] ? 0.0f :
        categories.size() > 1 && token == categories[1] ? 1.0f :
        parse_float_or_nan(token, number_format);
}

using CategoryMaps = vector<unordered_map<string_view, Index>>;

struct NumericColumnValues
{
    bool has_value = false;
    float first_value = 0.0f;
    bool constant = true;
    bool zero_one = true;
};

static CategoryMaps make_category_maps(const vector<Variable>& variables)
{
    CategoryMaps maps(variables.size());
    for(size_t i = 0; i < variables.size(); ++i)
    {
        const Variable& variable = variables[i];
        if(!variable.is_categorical()) continue;
        for(Index category = 0; category < ssize(variable.categories); ++category)
            maps[i].emplace(variable.categories[size_t(category)], category);
    }
    return maps;
}

static void check_expanded_data_size(const vector<Variable>& variables,
                                     Index samples_number,
                                     Index feature_columns_number)
{
    const size_t bytes = size_t(samples_number) * size_t(feature_columns_number) * sizeof(float);
    if(feature_columns_number <= 0 || bytes <= maximum_expanded_data_bytes) return;

    const auto largest = ranges::max_element(variables, {}, [](const Variable& variable) {
        return variable.is_categorical() ? variable.categories.size() : size_t(0);
    });
    const Index categories = largest == variables.end() ? 0 : ssize(largest->categories);
    const string name = largest == variables.end() ? string() : largest->name;
    throw runtime_error(format(
        "Expanding the categorical variables of this file would produce {} feature "
        "columns for {} samples ({:.1f} GB). The largest contributor is '{}' with {} "
        "categories. If that column is meant to be numeric, check its number format; "
        "if it is an identifier, remove it from the file.",
        feature_columns_number, samples_number,
        double(bytes) / (1024.0 * 1024.0 * 1024.0), name, categories));
}

class CsvDataParser
{
public:
    CsvDataParser(const vector<string_view>& lines,
                  const vector<Variable>& variables,
                  const vector<Index>& token_indices,
                  const vector<vector<Index>>& feature_indices,
                  const CategoryMaps& category_maps,
                  vector<string>& sample_ids,
                  vector<SampleRole>& sample_roles,
                  VectorI& variables_missing,
                  Index& rows_missing,
                  Index& missing,
                  string_view missing_label,
                  const NumberFormat& number_format,
                  DateFormat date_format,
                  char separator,
                  bool has_quotes,
                  bool has_sample_ids,
                  bool binary_storage,
                  Index required_tokens,
                  Index feature_columns)
        : lines(lines), variables(variables), token_indices(token_indices),
          feature_indices(feature_indices), category_maps(category_maps),
          sample_ids(sample_ids), sample_roles(sample_roles),
          variables_missing(variables_missing), rows_missing(rows_missing), missing(missing),
          missing_label(missing_label), number_format(number_format), date_format(date_format),
          separator(separator), has_quotes(has_quotes), has_sample_ids(has_sample_ids),
          binary_storage(binary_storage), required_tokens(required_tokens),
          feature_columns(feature_columns), numeric_values(variables.size()),
          bad_row_index(ssize(lines)), parse_error_index(ssize(lines))
    {
    }

    void parse_rows(Index base, Index end, float* destination)
    {
        Index range_rows_missing = 0;
        Index range_missing = 0;
        vector<Index> range_variables_missing(variables.size(), 0);

#pragma omp parallel
        {
            string scratch;
            vector<string_view> tokens;
            vector<Index> thread_variables_missing(variables.size(), 0);
            Index thread_rows_missing = 0;
            Index thread_missing = 0;

#pragma omp for schedule(static) nowait
            for(Index row_index = base; row_index < end; ++row_index)
            {
                get_token_views_maybe_quoted(lines[size_t(row_index)], separator,
                                             has_quotes, scratch, tokens);
                float* row = destination
                           + size_t(row_index - base) * size_t(feature_columns);
                const bool row_has_missing = count_missing(
                    tokens, thread_rows_missing, thread_missing, thread_variables_missing);

                if(ssize(tokens) < required_tokens)
                {
#pragma omp critical
                    record_bad_row(row_index, ssize(tokens));
                    continue;
                }

                if(has_sample_ids) sample_ids[size_t(row_index)] = string(tokens[0]);
                if(binary_storage && row_has_missing)
                    sample_roles[size_t(row_index)] = SampleRole::None;

                try { parse_row(row, tokens); }
                catch(const exception& error)
                {
#pragma omp critical
                    record_parse_error(row_index, error.what());
                }
            }

#pragma omp critical
            {
                range_rows_missing += thread_rows_missing;
                range_missing += thread_missing;
                for(size_t i = 0; i < variables.size(); ++i)
                    range_variables_missing[i] += thread_variables_missing[i];
            }
        }

        rows_missing += range_rows_missing;
        missing += range_missing;
        for(Index i = 0; i < ssize(variables); ++i)
            variables_missing(i) += range_variables_missing[size_t(i)];
        for(Index i = base; i < end; ++i)
            refine_numeric(destination + size_t(i - base) * size_t(feature_columns));
    }

    void throw_parse_error() const
    {
        if(bad_row_index < ssize(lines) && bad_row_index <= parse_error_index)
            throw runtime_error(format("Row {} has fewer columns than expected ({}).",
                                       bad_row_index, bad_row_columns));
        if(parse_error_index < ssize(lines))
            throw runtime_error(format("Row {}: {}", parse_error_index, parse_error_message));
    }

    void refine_variable_types(vector<Variable>& mutable_variables) const
    {
        for(size_t i = 0; i < mutable_variables.size(); ++i)
        {
            Variable& variable = mutable_variables[i];
            if(variable.type == VariableType::Numeric)
            {
                if(numeric_values[i].constant)
                    variable.set(variable.name, "None", VariableType::Constant);
                else if(numeric_values[i].zero_one)
                {
                    variable.type = VariableType::Binary;
                    variable.categories = {"0", "1"};
                }
            }
            else if(is_one_of(variable.type, VariableType::Binary, VariableType::Categorical)
                    && variable.get_categories_number() == 1)
                variable.set(variable.name, "None", VariableType::Constant);
        }
    }

private:
    void parse_row(float* row, const vector<string_view>& tokens) const
    {
        for(size_t i = 0; i < variables.size(); ++i)
        {
            const Variable& variable = variables[i];
            const string_view token = tokens[size_t(token_indices[i])];
            const vector<Index>& features = feature_indices[i];
            switch(variable.type)
            {
                case VariableType::None:
                case VariableType::Constant: break;
                case VariableType::Numeric:
                case VariableType::Integer:
                    parse_numeric_token(row, features[0], token, missing_label, number_format);
                    break;
                case VariableType::DateTime:
                    parse_datetime_token(row, features[0], token, missing_label, date_format);
                    break;
                case VariableType::Categorical:
                    parse_categorical_token(row, features, token, missing_label, category_maps[i]);
                    break;
                case VariableType::Binary:
                    parse_binary_token(row, features[0], token, missing_label,
                                       variable.categories, number_format);
                    break;
            }
        }
    }

    bool count_missing(const vector<string_view>& tokens,
                       Index& rows, Index& count, vector<Index>& columns) const
    {
        bool row_has_missing = false;
        for(size_t i = 0; i < variables.size(); ++i)
        {
            const size_t token = size_t(token_indices[i]);
            if(token >= tokens.size()) break;
            if(!is_missing_token(tokens[token], missing_label)) continue;
            row_has_missing = true;
            ++count;
            ++columns[i];
        }
        if(row_has_missing) ++rows;
        return row_has_missing;
    }

    void refine_numeric(const float* row)
    {
        for(size_t i = 0; i < variables.size(); ++i)
        {
            if(variables[i].type != VariableType::Numeric) continue;
            NumericColumnValues& column = numeric_values[i];
            const float value = row[size_t(feature_indices[i][0])];
            if(isnan(value)) continue;
            if(!column.has_value)
            {
                column.has_value = true;
                column.first_value = value;
            }
            else if(abs(value - column.first_value) > numeric_limits<float>::min())
                column.constant = false;
            if(value != 0.0f && value != 1.0f) column.zero_one = false;
        }
    }

    void record_bad_row(Index row, Index columns)
    {
        if(row >= bad_row_index) return;
        bad_row_index = row;
        bad_row_columns = columns;
    }

    void record_parse_error(Index row, const char* message)
    {
        if(row >= parse_error_index) return;
        parse_error_index = row;
        parse_error_message = message;
    }

    const vector<string_view>& lines;
    const vector<Variable>& variables;
    const vector<Index>& token_indices;
    const vector<vector<Index>>& feature_indices;
    const CategoryMaps& category_maps;
    vector<string>& sample_ids;
    vector<SampleRole>& sample_roles;
    VectorI& variables_missing;
    Index& rows_missing;
    Index& missing;
    string_view missing_label;
    const NumberFormat& number_format;
    DateFormat date_format;
    char separator;
    bool has_quotes;
    bool has_sample_ids;
    bool binary_storage;
    Index required_tokens;
    Index feature_columns;
    vector<NumericColumnValues> numeric_values;
    Index bad_row_index;
    Index bad_row_columns = 0;
    Index parse_error_index;
    string parse_error_message;
};

static DateFormat infer_dataset_date_format(const vector<Variable>& variables,
                                     const vector<string_view>& sample_lines,
                                     char file_separator,
                                     bool has_sample_ids,
                                     const string& missing_values_label,
                                     bool has_quotes)
{
    const bool any_datetime = ranges::any_of(variables,
        [](const Variable& v) { return v.type == VariableType::DateTime; });

    if (!any_datetime)
        return Auto;

    const size_t id_offset = has_sample_ids ? 1 : 0;

    string scratch;
    vector<string_view> row;
    for (const string_view line : sample_lines)
    {
        get_token_views_maybe_quoted(line, file_separator, has_quotes, scratch, row);

        for (size_t col_index = 0; col_index < variables.size(); ++col_index)
        {
            if (variables[col_index].type != VariableType::DateTime)
                continue;

            const size_t token_index = col_index + id_offset;
            if (token_index >= row.size())
                continue;

            const string_view token = row[token_index];

            if (is_missing_token(token, missing_values_label))
                continue;

            const DateFormat detected = detect_date_format(token);
            if (detected != Auto) return detected;
        }
    }

    return Auto;
}

static NumberFormat detect_number_format(const vector<string_view>& lines,
                                         const char file_separator,
                                         const bool has_quotes)
{
    constexpr size_t maximum_rows_to_check = 100;

    const size_t total_rows = lines.size();

    if (total_rows == 0) return {};

    const size_t rows_to_check = min(maximum_rows_to_check, total_rows);

    NumberFormatVotes votes;

    string scratch;
    vector<string_view> tokens;

    for (size_t i = 0; i < rows_to_check; ++i)
    {
        get_token_views_maybe_quoted(
            lines[i * total_rows / rows_to_check],
            file_separator,
            has_quotes,
            scratch,
            tokens);

        for (const string_view token : tokens)
            vote_number_format(token, votes);
    }

    return decide_number_format(votes);
}

void TabularDataset::read_csv()
{
    const string separator_string = get_separator_string();
    const char file_separator = separator_string.empty() ? ',' : separator_string[0];

    CsvReader::Result parsed =
        CsvReader(
            [this](const string_view line)
            {
                check_separators(line);
            })
        .read(data_path);

    const bool has_quotes = parsed.has_quotes;
    vector<string_view>& lines = parsed.lines;

    throw_if(lines.empty(),
             "File {} is empty or contains no valid data rows.",
             data_path.string());

    read_data_file_preview(lines, file_separator, has_quotes);

    const DateFormat date_format =
        configure_csv_columns(lines, file_separator, has_quotes);

    load_csv_data(lines, file_separator, has_quotes, date_format);
}

DateFormat TabularDataset::configure_csv_columns(vector<string_view>& lines,
                                                  const char file_separator,
                                                  const bool has_quotes)
{
    string header_scratch;

    const vector<string_view> header_tokens =
        get_token_views_maybe_quoted(
            lines[0],
            file_separator,
            has_quotes,
            header_scratch);

    if(has_header)
    {
        const auto is_number = [](const string_view token)
        {
            return is_numeric_string(token);
        };

        throw_if(ranges::any_of(header_tokens, is_number),
                 "Some header names are numeric.");

        lines.erase(lines.begin());
    }

    throw_if(lines.empty(),
             "Data file only contains a header.");

    const Index samples_number = ssize(lines);

    if(number_format_automatic)
        number_format = detect_number_format(lines, file_separator, has_quotes);

    if(display && !number_format.is_default())
    {
        logging::info() << "Reading numbers in " << data_path.string()
             << " with decimal separator "
             << number_format_name(number_format.decimal_separator)
             << " and thousands separator "
             << number_format_name(number_format.group_separator)
             << ".\n";
    }

    if(!has_sample_ids)
    {
        unordered_set<string> unique_elements;
        string id_scratch;

        bool possible_id = true;
        bool is_numeric_column = true;
        bool is_date_column = true;

        Index date_check_count = 0;
        constexpr Index max_date_checks = 20;

        for(const string_view line : lines)
        {
            const string_view token =
                first_token_maybe_quoted(
                    line,
                    file_separator,
                    has_quotes,
                    id_scratch);

            if(!unique_elements.emplace(token).second)
            {
                possible_id = false;
                break;
            }

            if(is_numeric_column
               && !is_missing_token(token, missing_values_label)
               && !is_numeric_string(token, number_format))
            {
                is_numeric_column = false;
            }

            if(is_date_column
               && date_check_count < max_date_checks
               && !is_missing_token(token, missing_values_label))
            {
                if(!is_date_time_string(token))
                    is_date_column = false;

                ++date_check_count;
            }
        }

        if(is_date_column && date_check_count > 0)
            possible_id = false;

        has_sample_ids =
            possible_id
            && !is_numeric_column
            && unique_elements.size() == size_t(samples_number);
    }

    const Index columns_number = ssize(header_tokens);
    const Index id_offset = has_sample_ids ? 1 : 0;

    throw_if(columns_number <= id_offset,
             "Data file contains no variables.");

    const vector<Variable> previous_variables = variables;

    const Index variables_number = columns_number - id_offset;

    variables.assign(size_t(variables_number), Variable{});

    if(has_header)
    {
        set_variable_names(
            vector<string>(
                header_tokens.begin() + id_offset,
                header_tokens.end()));
    }
    else
    {
        set_default_variable_names();
    }

    const DateFormat date_format =
        infer_column_types(lines, file_separator, has_quotes);

    for(Variable& variable : variables)
    {
        if(variable.is_categorical()
           && variable.get_categories_number() == 2)
        {
            variable.type = VariableType::Binary;
        }
    }

    for(Variable& variable : variables)
    {
        const vector<Variable>::const_iterator previous =
            ranges::find_if(
                previous_variables,
                [&](const Variable& candidate)
                {
                    return candidate.name == variable.name;
                });

        if(previous == previous_variables.end())
            continue;

        variable.role = previous->role;
        variable.scaler = previous->scaler;
    }

    return date_format;
}

void TabularDataset::load_csv_data(const vector<string_view>& lines,
                                   const char file_separator,
                                   const bool has_quotes,
                                   const DateFormat date_format)
{
    const Index samples_number = ssize(lines);
    const Index id_offset = has_sample_ids ? 1 : 0;
    vector<Index> variable_token_indices(variables.size());
    iota(variable_token_indices.begin(), variable_token_indices.end(), id_offset);

    for(const size_t i : views::iota(size_t(0), variables.size()) | views::reverse)
    {
        if(!looks_like_id_variable(variables[i], samples_number)) continue;
        logging::info() << "Excluding identifier column: " << variables[i].name << endl;
        variables.erase(variables.begin() + Index(i));
        variable_token_indices.erase(variable_token_indices.begin() + Index(i));
    }

    throw_if(variables.empty(),
             "Data file contains no variables (all columns are identifiers).");
    const Index variables_number = ssize(variables);
    const Index required_tokens = variable_token_indices.back() + 1;

    sample_roles.assign(size_t(samples_number), SampleRole::Training);
    if(has_sample_ids) sample_ids.assign(size_t(samples_number), {});
    else sample_ids.clear();

    const vector<vector<Index>> feature_indices = get_feature_indices();
    const Index feature_columns = feature_indices.empty()
                                ? 0 : feature_indices.back().back() + 1;
    check_expanded_data_size(variables, samples_number, feature_columns);

    const bool binary_storage = storage_mode == StorageMode::BinaryFile;
    FileWriter cache_writer;
    if(binary_storage)
    {
        cache_reader.close();
        clear_cache_derived_state();
        cache_path = cache_file_path();
        filesystem::create_directories(cache_path.parent_path());
        cache_writer.open(cache_path.string() + ".tmp");
        cache_columns_number = feature_columns;
    }
    else
        data = MatrixR::Zero(samples_number, feature_columns);

    rows_missing_values_number = 0;
    missing_values_number = 0;
    variables_missing_values_number = VectorI::Zero(variables_number);
    const CategoryMaps category_maps = make_category_maps(variables);
    CsvDataParser parser(lines, variables, variable_token_indices, feature_indices,
                         category_maps, sample_ids, sample_roles,
                         variables_missing_values_number, rows_missing_values_number,
                         missing_values_number, missing_values_label, number_format,
                         date_format, file_separator, has_quotes, has_sample_ids,
                         binary_storage, required_tokens, feature_columns);

    if(binary_storage)
    {
        constexpr Index chunk_size = 16384;
        vector<float> chunk;
        for(Index base = 0; base < samples_number; base += chunk_size)
        {
            const Index end = min(base + chunk_size, samples_number);
            chunk.assign(size_t(end - base) * size_t(feature_columns), 0.0f);
            parser.parse_rows(base, end, chunk.data());
            cache_writer.write(span(chunk));
        }
    }
    else
        parser.parse_rows(0, samples_number, data.data());

    parser.throw_parse_error();
    if(binary_storage)
    {
        cache_writer.finish_with_rename(cache_path);
        cache_reader.open(cache_path);
    }

    parser.refine_variable_types(variables);
    split_samples_random();
    if(binary_storage) refresh_cache_statistics();
}
DateFormat TabularDataset::infer_column_types(
    const vector<string_view>& sample_lines,
    const char file_separator,
    const bool has_quotes)
{
    const Index variables_number = ssize(variables);
    const size_t total_rows = sample_lines.size();

    if(total_rows == 0) return Auto;

    constexpr size_t max_rows_to_check = 100;

    const size_t rows_to_check = min(max_rows_to_check, total_rows);
    const size_t id_offset = has_sample_ids ? 1 : 0;

    vector<vector<string_view>> sampled_tokens(rows_to_check);
    vector<string> sampled_scratch(rows_to_check);

    for(size_t i = 0; i < rows_to_check; ++i)
    {
        const size_t row = i * total_rows / rows_to_check;

        sampled_tokens[i] = get_token_views_maybe_quoted(
            sample_lines[row],
            file_separator,
            has_quotes,
            sampled_scratch[i]);
    }

    for(Index col_index = 0; col_index < variables_number; ++col_index)
    {
        Variable& variable = variables[size_t(col_index)];
        variable.type = VariableType::None;

        const size_t token_index = size_t(col_index) + id_offset;

        size_t checked_tokens = 0;
        size_t numeric_tokens = 0;
        string first_unparseable;

        for(const vector<string_view>& tokens : sampled_tokens)
        {
            if(token_index >= tokens.size()) continue;

            const string_view token = tokens[token_index];

            if(is_missing_token(token, missing_values_label))
                continue;

            ++checked_tokens;

            const bool numeric = is_numeric_string(token, number_format);

            if(numeric)
            {
                ++numeric_tokens;
            }
            else if(first_unparseable.empty())
            {
                first_unparseable = token;
            }

            if(variable.is_categorical())
                continue;

            if(numeric)
            {
                if(variable.type == VariableType::None)
                    variable.type = VariableType::Numeric;

                continue;
            }

            if(is_date_time_string(token))
            {
                if(variable.type == VariableType::None)
                    variable.type = VariableType::DateTime;

                continue;
            }

            variable.type = VariableType::Categorical;
        }

        if(variable.type == VariableType::None)
            variable.type = VariableType::Numeric;

        if(variable.type == VariableType::Categorical
           && checked_tokens > 0
           && numeric_tokens * 10 >= checked_tokens * 9)
        {
            logging::warning() << "Warning: variable '" << variable.name
                 << "' was classified as categorical, but "
                 << numeric_tokens << " of its " << checked_tokens
                 << " sampled values are numeric. First value that failed to parse: '"
                 << first_unparseable
                 << "'. Check the number format (thousands separators, decimal commas); "
                    "otherwise this column expands into one column per distinct value.\n";
        }
    }

    const DateFormat date_format = infer_dataset_date_format(variables,
                                                             sample_lines,
                                                             file_separator,
                                                             has_sample_ids,
                                                             missing_values_label,
                                                             has_quotes);

    if(ranges::none_of(
           variables,
           [](const Variable& variable)
           {
               return variable.is_categorical();
           }))
    {
        return date_format;
    }

    vector<unordered_set<string>>
        unique_categories(static_cast<size_t>(variables_number));

    const Index lines_number = ssize(sample_lines);

#pragma omp parallel
    {
        vector<unordered_set<string>>
            local_categories(static_cast<size_t>(variables_number));

        string scratch;
        vector<string_view> tokens;

#pragma omp for schedule(static) nowait
        for(Index row = 0; row < lines_number; ++row)
        {
            get_token_views_maybe_quoted(
                sample_lines[size_t(row)],
                file_separator,
                has_quotes,
                scratch,
                tokens);

            for(Index col_index = 0;
                col_index < variables_number;
                ++col_index)
            {
                const size_t index = size_t(col_index);

                if(!variables[index].is_categorical())
                    continue;

                const size_t token_index =
                    index + id_offset;

                if(token_index >= tokens.size())
                    continue;

                const string_view token =
                    tokens[token_index];

                if(is_missing_token(token, missing_values_label))
                    continue;

                local_categories[index].emplace(token);
            }
        }

#pragma omp critical
        {
            for(Index col_index = 0;
                col_index < variables_number;
                ++col_index)
            {
                const size_t index = size_t(col_index);

                unique_categories[index].insert(
                    local_categories[index].begin(),
                    local_categories[index].end());
            }
        }
    }

    for(Index col_index = 0;
        col_index < variables_number;
        ++col_index)
    {
        Variable& variable =
            variables[size_t(col_index)];

        if(!variable.is_categorical())
            continue;

        const unordered_set<string>& unique =
            unique_categories[size_t(col_index)];

        variable.categories.assign(
            unique.begin(),
            unique.end());

        ranges::sort(variable.categories);
    }

    return date_format;
}

}
