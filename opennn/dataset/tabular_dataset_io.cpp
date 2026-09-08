//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T A B U L A R   D A T A S E T   I M P O R T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/dataset/tabular_dataset.h"
#include "opennn/core/io_utilities.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/string_utilities.h"
#include "opennn/core/tensor_types.h"

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
        cout << "Reading numbers in " << data_path.string()
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

        cout << "Excluding identifier column: " << variables[i].name << endl;

        variables.erase(variables.begin() + Index(i));
        variable_token_indices.erase(variable_token_indices.begin() + Index(i));
    }

    throw_if(variables.empty(),
             "Data file contains no variables (all columns are identifiers).");

    const Index variables_number = ssize(variables);
    const Index required_tokens = variable_token_indices.back() + 1;

    sample_roles.assign(size_t(samples_number), SampleRole::Training);

    // Only when the file actually carries ids. std::string is 32 bytes here
    // even when empty, so one per sample costs 32 bytes a row before holding
    // a single character -- 15.3 MiB on a 500,000-row file whose ids are never
    // written and never read. Every use of sample_ids is already guarded by
    // has_sample_ids, so an empty vector is the correct representation of
    // "this file has none".
    if (has_sample_ids)
        sample_ids.assign(size_t(samples_number), {});
    else
        sample_ids.clear();

    const vector<vector<Index>> all_feature_indices =
        get_feature_indices();

    const Index feature_columns_number =
        all_feature_indices.empty()
        ? 0
        : all_feature_indices.back().back() + 1;

    if(feature_columns_number > 0)
    {
        const size_t projected_bytes =
            size_t(samples_number)
            * size_t(feature_columns_number)
            * sizeof(float);

        if(projected_bytes > maximum_expanded_data_bytes)
        {
            Index worst_index = 0;
            Index worst_categories = 0;

            for(Index i = 0; i < variables_number; ++i)
            {
                if(variables[size_t(i)].is_categorical()
                   && ssize(variables[size_t(i)].categories) > worst_categories)
                {
                    worst_categories =
                        ssize(variables[size_t(i)].categories);

                    worst_index = i;
                }
            }

            throw runtime_error(
                format(
                    "Expanding the categorical variables of this file would produce {} feature "
                    "columns for {} samples ({:.1f} GB). The largest contributor is '{}' with {} "
                    "categories. If that column is meant to be numeric, check its number format; "
                    "if it is an identifier, remove it from the file.",
                    feature_columns_number,
                    samples_number,
                    double(projected_bytes)
                        / (1024.0 * 1024.0 * 1024.0),
                    variables[size_t(worst_index)].name,
                    worst_categories));
        }
    }

    const bool binary_storage =
        storage_mode == StorageMode::BinaryFile;

    FileWriter cache_writer;
    vector<float> row_values;

    if(binary_storage)
    {
        cache_reader.close();
        clear_cache_derived_state();
        cache_path = cache_file_path();

        filesystem::create_directories(
            cache_path.parent_path());

        cache_writer.open(
            cache_path.string() + ".tmp");

        cache_columns_number = feature_columns_number;

        row_values.resize(size_t(feature_columns_number));
    }
    else
    {
        data = MatrixR::Zero(
            samples_number,
            feature_columns_number);
    }

    rows_missing_values_number = 0;
    missing_values_number = 0;

    variables_missing_values_number =
        VectorI::Zero(variables_number);

    vector<unordered_map<string_view, Index>>
        category_maps(static_cast<size_t>(variables_number));

    for(Index variable_index = 0;
        variable_index < variables_number;
        ++variable_index)
    {
        const Variable& variable =
            variables[size_t(variable_index)];

        if(!variable.is_categorical())
            continue;

        unordered_map<string_view, Index>& category_map =
            category_maps[size_t(variable_index)];

        for(Index category = 0;
            category < ssize(variable.categories);
            ++category)
        {
            category_map.emplace(
                string_view(variable.categories[size_t(category)]),
                category);
        }
    }

    struct NumericColumnValues
    {
        bool has_value = false;
        float first_value = 0.0f;
        bool constant = true;
        bool zero_one = true;
    };

    vector<NumericColumnValues>
        numeric_column_values(static_cast<size_t>(variables_number));

    const auto parse_row =
        [&](float* row,
            const vector<string_view>& row_tokens)
    {
        for(Index variable_index = 0;
            variable_index < variables_number;
            ++variable_index)
        {
            const size_t index = size_t(variable_index);

            const Variable& variable = variables[index];

            const string_view token =
                row_tokens[size_t(variable_token_indices[index])];

            const vector<Index>& feature_indices =
                all_feature_indices[index];

            using enum VariableType;

            switch(variable.type)
            {
                case None:
                case Constant:
                    break;

                case Numeric:
                case Integer:
                    parse_numeric_token(
                        row,
                        feature_indices[0],
                        token,
                        missing_values_label,
                        number_format);
                    break;

                case DateTime:
                    parse_datetime_token(
                        row,
                        feature_indices[0],
                        token,
                        missing_values_label,
                        date_format);
                    break;

                case Categorical:
                    parse_categorical_token(
                        row,
                        feature_indices,
                        token,
                        missing_values_label,
                        category_maps[index]);
                    break;

                case Binary:
                    parse_binary_token(
                        row,
                        feature_indices[0],
                        token,
                        missing_values_label,
                        variable.categories,
                        number_format);
                    break;
            }
        }
    };

    const auto refine_numeric =
        [&](const float* row)
    {
        for(Index variable_index = 0;
            variable_index < variables_number;
            ++variable_index)
        {
            const size_t index = size_t(variable_index);

            if(variables[index].type != VariableType::Numeric)
                continue;

            NumericColumnValues& column =
                numeric_column_values[index];

            const float value =
                row[size_t(all_feature_indices[index][0])];

            if(isnan(value))
                continue;

            if(!column.has_value)
            {
                column.has_value = true;
                column.first_value = value;
            }
            else if(abs(value - column.first_value)
                    > numeric_limits<float>::min())
            {
                column.constant = false;
            }

            if(value != 0.0f && value != 1.0f)
                column.zero_one = false;
        }
    };

    const auto count_missing =
        [&](const vector<string_view>& row_tokens,
            Index& thread_rows_missing,
            Index& thread_missing,
            vector<Index>& thread_variables_missing)
    {
        bool row_has_missing = false;

        for(Index variable_index = 0;
            variable_index < variables_number;
            ++variable_index)
        {
            const size_t index = size_t(variable_index);

            const size_t token_index =
                size_t(variable_token_indices[index]);

            if(token_index >= row_tokens.size())
                break;

            if(!is_missing_token(row_tokens[token_index], missing_values_label))
                continue;

            row_has_missing = true;
            ++thread_missing;
            ++thread_variables_missing[index];
        }

        if(row_has_missing)
            ++thread_rows_missing;

        return row_has_missing;
    };

    bool bad_row = false;
    Index bad_row_index = samples_number;
    Index bad_row_columns = 0;

    bool parse_error = false;
    Index parse_error_index = samples_number;
    string parse_error_message;

    const auto parse_rows =
        [&](const Index base,
            const Index end,
            float* destination)
    {
        Index range_rows_missing = 0;
        Index range_missing = 0;

        vector<Index> range_variables_missing(
            size_t(variables_number),
            0);

#pragma omp parallel
        {
            string thread_scratch;
            vector<string_view> thread_tokens;

            vector<Index> thread_variables_missing(
                size_t(variables_number),
                0);

            Index thread_rows_missing = 0;
            Index thread_missing = 0;

#pragma omp for schedule(static) nowait
            for(Index i = base; i < end; ++i)
            {
                get_token_views_maybe_quoted(
                    lines[size_t(i)],
                    file_separator,
                    has_quotes,
                    thread_scratch,
                    thread_tokens);

                float* row =
                    destination
                    + size_t(i - base)
                        * size_t(feature_columns_number);

                const bool row_has_missing =
                    count_missing(
                        thread_tokens,
                        thread_rows_missing,
                        thread_missing,
                        thread_variables_missing);

                if(ssize(thread_tokens) < required_tokens)
                {
#pragma omp critical
                    {
                        if(i < bad_row_index)
                        {
                            bad_row = true;
                            bad_row_index = i;
                            bad_row_columns =
                                ssize(thread_tokens);
                        }
                    }

                    continue;
                }

                if(has_sample_ids)
                {
                    sample_ids[size_t(i)] =
                        string(thread_tokens[0]);
                }

                if(binary_storage && row_has_missing)
                {
                    sample_roles[size_t(i)] =
                        SampleRole::None;
                }

                try
                {
                    parse_row(row, thread_tokens);
                }
                catch(const exception& e)
                {
#pragma omp critical
                    {
                        if(i < parse_error_index)
                        {
                            parse_error = true;
                            parse_error_index = i;
                            parse_error_message = e.what();
                        }
                    }
                }
            }

#pragma omp critical
            {
                range_rows_missing += thread_rows_missing;
                range_missing += thread_missing;

                for(Index variable_index = 0;
                    variable_index < variables_number;
                    ++variable_index)
                {
                    range_variables_missing[
                        size_t(variable_index)]
                        += thread_variables_missing[
                            size_t(variable_index)];
                }
            }
        }

        rows_missing_values_number +=
            range_rows_missing;

        missing_values_number +=
            range_missing;

        for(Index variable_index = 0;
            variable_index < variables_number;
            ++variable_index)
        {
            variables_missing_values_number(variable_index)
                += range_variables_missing[
                    size_t(variable_index)];
        }

        for(Index i = base; i < end; ++i)
        {
            refine_numeric(
                destination
                + size_t(i - base)
                    * size_t(feature_columns_number));
        }
    };

    if(binary_storage)
    {
        constexpr Index chunk_size = 16384;

        vector<float> chunk_buffer;

        for(Index base = 0;
            base < samples_number;
            base += chunk_size)
        {
            const Index end =
                min(base + chunk_size, samples_number);

            const Index rows_number =
                end - base;

            chunk_buffer.assign(
                size_t(rows_number)
                    * size_t(feature_columns_number),
                0.0f);

            parse_rows(
                base,
                end,
                chunk_buffer.data());

            cache_writer.write(
                span(chunk_buffer));
        }
    }
    else
    {
        parse_rows(
            0,
            samples_number,
            data.data());
    }

    if(bad_row
       && (!parse_error
           || bad_row_index <= parse_error_index))
    {
        throw runtime_error(
            format(
                "Row {} has fewer columns than expected ({}).",
                bad_row_index,
                bad_row_columns));
    }

    if(parse_error)
    {
        throw runtime_error(
            format(
                "Row {}: {}",
                parse_error_index,
                parse_error_message));
    }

    if(binary_storage)
    {
        cache_writer.finish_with_rename(cache_path);
        cache_reader.open(cache_path);
    }

    for(Index variable_index = 0;
        variable_index < variables_number;
        ++variable_index)
    {
        Variable& variable =
            variables[size_t(variable_index)];

        if(variable.type == VariableType::Numeric)
        {
            const NumericColumnValues& column =
                numeric_column_values[
                    size_t(variable_index)];

            if(column.constant)
            {
                variable.set(
                    variable.name,
                    "None",
                    VariableType::Constant);
            }
            else if(column.zero_one)
            {
                variable.type = VariableType::Binary;
                variable.categories = {"0", "1"};
            }
        }
        else if(is_one_of(
                    variable.type,
                    VariableType::Binary,
                    VariableType::Categorical)
                && variable.get_categories_number() == 1)
        {
            variable.set(
                variable.name,
                "None",
                VariableType::Constant);
        }
    }

    split_samples_random();

    if (binary_storage)
        refresh_cache_statistics();
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
            cout << "Warning: variable '" << variable.name
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

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
