// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence Techniques, SL.

#include "opennn/core/string_utilities.h"

#include <cctype>

namespace opennn
{

namespace
{

bool equal_ignoring_case(string_view left, string_view right) noexcept
{
    return left.size() == right.size()
        && ranges::equal(left, right,
                         [](unsigned char left_char, unsigned char right_char)
                         {
                             return tolower(left_char) == tolower(right_char);
                         });
}

template <typename Emit>
void for_each_token(string_view document, Emit&& emit)
{
    size_t i = 0;
    while (i < document.size())
    {
        const unsigned char character = static_cast<unsigned char>(document[i]);
        const size_t start = i++;

        if (isalnum(character))
        {
            while (i < document.size() && isalnum(static_cast<unsigned char>(document[i])))
                ++i;
        }
        else if (!ispunct(character))
            continue;

        emit(document.substr(start, i - start));
    }
}

template <typename Emit>
void for_each_quoted_field(string_view line, char separator, string& scratch, Emit&& emit)
{
    enum class Phase { Leading, Bare, Quoted };

    Phase phase = Phase::Leading;
    size_t closed_bytes = 0;

    scratch.clear();
    scratch.reserve(line.size());

    const auto pending_field = [&scratch, &closed_bytes]
    {
        return string_view(scratch).substr(closed_bytes);
    };

    const auto close_field = [&]
    {
        if (!emit(pending_field())) return false;
        closed_bytes = scratch.size();
        phase = Phase::Leading;
        return true;
    };

    for (size_t i = 0; i < line.size(); ++i)
    {
        const char character = line[i];

        switch (phase)
        {
        case Phase::Leading:
            if (character == '"')
            {
                phase = Phase::Quoted;
            }
            else if (character != separator)
            {
                scratch.push_back(character);
                phase = Phase::Bare;
            }
            else if (!close_field())
            {
                return;
            }
            break;

        case Phase::Bare:
            // A quote inside an unquoted field does not flip the mode, so it is literal
            // unless it is the separator, which still ends the field here.
            if (character != separator)
                scratch.push_back(character);
            else if (!close_field())
                return;
            break;

        case Phase::Quoted:
            if (character != '"')
            {
                scratch.push_back(character);
            }
            else if (i + 1 < line.size() && line[i + 1] == '"')
            {
                // A doubled quote inside a quoted field stands for one literal quote.
                scratch.push_back('"');
                ++i;
            }
            else
            {
                phase = scratch.size() == closed_bytes ? Phase::Leading : Phase::Bare;
            }
            break;
        }
    }

    emit(pending_field());
}

}

float parse_float(string_view text, string_view context)
{
    return parse_number<float>(text, context);
}

int parse_int(string_view text, string_view context)
{
    return parse_number<int>(text, context, "integer");
}

long parse_long(string_view text, string_view context)
{
    return parse_number<long>(text, context, "integer");
}

vector<string> tokenize(const string& document)
{
    vector<string> tokens;
    for_each_token(document, [&](string_view token)
    {
        string& value = tokens.emplace_back(token);
        for (char& character : value)
            character = static_cast<char>(tolower(static_cast<unsigned char>(character)));
    });

    return tokens;
}

vector<string_view> tokenize_views(string_view document)
{
    vector<string_view> tokens;
    for_each_token(document, [&](string_view token) { tokens.push_back(token); });
    return tokens;
}

vector<string> get_tokens(string_view text, string_view separator)
{
    if (separator.empty()) return {string(text)};

    vector<string> tokens;

    for (const auto token : text | views::split(separator))
        tokens.emplace_back(token.begin(), token.end());

    return tokens;
}

vector<string_view> get_token_views(string_view text, char separator)
{
    vector<string_view> tokens;
    split_views(text, separator, tokens);
    return tokens;
}

void split_views(string_view text, char separator, vector<string_view>& tokens)
{
    tokens.clear();
    size_t start = 0;
    while (true)
    {
        const size_t end = text.find(separator, start);

        if (end == string_view::npos)
        {
            tokens.emplace_back(text.substr(start));
            break;
        }

        tokens.emplace_back(text.substr(start, end - start));
        start = end + 1;
    }
}

void get_token_views_maybe_quoted(string_view line, char separator, bool file_has_quotes,
                                  string& scratch, vector<string_view>& out)
{
    if (!file_has_quotes || line.find('"') == string_view::npos)
    {
        split_views(line, separator, out);
        return;
    }

    out.clear();

    for_each_quoted_field(line, separator, scratch, [&](string_view field)
    {
        out.push_back(field);
        return true;
    });
}

vector<string_view> get_token_views_maybe_quoted(string_view line, char separator,
                                                 bool file_has_quotes, string& scratch)
{
    vector<string_view> fields;
    get_token_views_maybe_quoted(line, separator, file_has_quotes, scratch, fields);
    return fields;
}

string_view first_token_maybe_quoted(string_view line, char separator, bool file_has_quotes, string& scratch)
{
    if (!file_has_quotes || line.find('"') == string_view::npos)
        return line.substr(0, line.find(separator));

    string_view first;
    for_each_quoted_field(line, separator, scratch, [&](string_view field)
    {
        first = field;
        return false;
    });
    return first;
}

string_view trim_view(string_view text)
{
    constexpr string_view whitespace = " \t\n\r\f\v\b";

    const size_t start = text.find_first_not_of(whitespace);
    if (start == string_view::npos) return {};

    const size_t end = text.find_last_not_of(whitespace);
    return text.substr(start, end - start + 1);
}

void ascii_lowercase_in_place(string& text) noexcept
{
    for (char& character : text)
        if (character >= 'A' && character <= 'Z')
            character = static_cast<char>(character + ('a' - 'A'));
}

string ascii_lowercase(string_view text)
{
    string result(text);
    ascii_lowercase_in_place(result);
    return result;
}

void append_utf8(string& output, uint32_t codepoint)
{
    if (codepoint <= 0x7F)
    {
        output.push_back(static_cast<char>(codepoint));
    }
    else if (codepoint <= 0x7FF)
    {
        output.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
    else if (codepoint <= 0xFFFF)
    {
        output.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
    else
    {
        output.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
    }
}

vector<string> convert_string_vector(const vector<vector<string>>& input_vector, const string& separator)
{
    vector<string> vector_result(input_vector.size());
    ranges::transform(input_vector, vector_result.begin(),
                      [&separator](const vector<string>& values) { return join_strings(values, separator); });

    return vector_result;
}

void replace_all_word_appearances(string& text, const string& to_replace, const string& replace_with)
{
    if (to_replace.empty()) return;

    const auto is_word_character = [](char character)
    {
        const unsigned char byte = static_cast<unsigned char>(character);
        return isalnum(byte) != 0 || byte == '_';
    };

    const size_t length = to_replace.length();
    size_t position = 0;

    while ((position = text.find(to_replace, position)) != string::npos)
    {
        const size_t end = position + length;

        const bool starts_word = position == 0 || !is_word_character(text[position - 1]);
        const bool ends_word = end == text.length() || !is_word_character(text[end]);

        if (starts_word && ends_word)
        {
            text.replace(position, length, replace_with);
            position += replace_with.length();
        }
        else
        {
            ++position;
        }
    }
}

string join_strings(span<const string> values, string_view separator)
{
    if (values.empty()) return {};

    size_t size = separator.size() * (values.size() - 1);
    for (const string& value : values) size += value.size();

    string result;
    result.reserve(size);

    for (size_t i = 0; i < values.size(); ++i)
    {
        if (i != 0) result.append(separator);
        result.append(values[i]);
    }

    return result;
}

void replace(string& source, const string& find_what, const string& replace_with)
{
    if (find_what.empty()) return;

    size_t position = 0;

    while ((position = source.find(find_what, position)) != string::npos)
    {
        source.replace(position, find_what.length(), replace_with);

        position += replace_with.length();
    }
}

string get_first_word(const string& line)
{
    return line.substr(0, line.find_first_of(" ="));
}

float get_elapsed_time(const time_t& beginning_time)
{
    return float(difftime(time(nullptr), beginning_time));
}

string get_time(float time)
{
    const int total_seconds = static_cast<int>(time);
    const int hours = total_seconds / 3600;
    const int minutes = (total_seconds % 3600) / 60;
    const int seconds = total_seconds % 60;

    return format("{:02}:{:02}:{:02}", hours, minutes, seconds);
}

void display_progress_bar(Index completed, Index total)
{
    const int width = 50;
    const float progress = total > 0 ? static_cast<float>(completed) / total : 0.0f;
    const int position = min(static_cast<int>(width * progress), width);

    logging::info() << "\r[" << string(position, '=')
                    << (position < width ? ">" + string(width - position - 1, ' ') : "")
                    << "] " << int(progress * 100.0) << " %   ";
}

void string_to_vector(const string& input, VectorR& values)
{
    const vector<float> parsed = parse_number_list<float>(input, "Vector");
    values.resize(static_cast<Index>(parsed.size()));
    ranges::copy(parsed, values.data());
}

bool contains(const vector<string>& data, string_view value)
{
    return ranges::any_of(data,
                          [&](const string& s) { return s == value; });
}

bool contains(initializer_list<string_view> data, string_view value)
{
    return ranges::find(data, value) != data.end();
}

bool starts_with_any(string_view text, initializer_list<string_view> prefixes)
{
    return ranges::any_of(prefixes,
                          [text](string_view prefix)
                          {
                              return text.starts_with(prefix);
                          });
}

bool env_flag_enabled(const char* name) noexcept
{
    return env_flag_enabled(name, false);
}

bool env_flag_enabled(const char* name, bool default_value) noexcept
{
    const char* const value = getenv(name);
    if (!value || !*value) return default_value;
    const string_view text(value);
    for (string_view on : {"1", "true", "on", "yes"})
        if (equal_ignoring_case(text, on)) return true;
    for (string_view off : {"0", "false", "off", "no"})
        if (equal_ignoring_case(text, off)) return false;
    return default_value;
}

long long env_int_or(const char* name, long long default_value) noexcept
{
    const char* const value = getenv(name);
    if (!value || !*value) return default_value;
    char* end = nullptr;
    const long long parsed = strtoll(value, &end, 10);
    return end == value ? default_value : parsed;
}

}
