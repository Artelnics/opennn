//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T R I N G   U T I L I T I E S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

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
    string current_token;

    for (const char character : document)
    {
        const unsigned char unsigned_character = static_cast<unsigned char>(character);

        if (isalnum(unsigned_character))
        {
            current_token += static_cast<char>(tolower(unsigned_character));
        }
        else
        {
            if (!current_token.empty())
            {
                tokens.emplace_back(move(current_token));
                current_token.clear();
            }

            if (ispunct(unsigned_character))
                tokens.emplace_back(1, character);
        }
    }

    if (!current_token.empty())
        tokens.emplace_back(move(current_token));

    return tokens;
}

vector<string_view> tokenize_views(string_view document)
{
    vector<string_view> tokens;

    size_t i = 0;
    while (i < document.size())
    {
        const unsigned char c = static_cast<unsigned char>(document[i]);

        if (isalnum(c))
        {
            const size_t start = i;
            while (i < document.size() && isalnum(static_cast<unsigned char>(document[i])))
                ++i;
            tokens.emplace_back(document.substr(start, i - start));
        }
        else if (ispunct(c))
        {
            tokens.emplace_back(document.substr(i, 1));
            ++i;
        }
        else
        {
            ++i;
        }
    }

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
    out.clear();

    if (!file_has_quotes || line.find('"') == string_view::npos)
    {
        split_views(line, separator, out);
        return;
    }

    scratch.clear();
    scratch.reserve(line.size());
    const char* const base = scratch.data();

    bool in_quote = false;
    size_t field_start = 0;

    for (const char c : line)
    {

        if (c == '"' && (in_quote || scratch.size() == field_start))
        {
            in_quote = !in_quote;
            continue;
        }

        if (!in_quote && c == separator)
        {
            out.emplace_back(base + field_start, scratch.size() - field_start);
            field_start = scratch.size();
            continue;
        }

        if (in_quote && (c == ',' || c == ';')) continue;

        scratch.push_back(c);
    }

    out.emplace_back(base + field_start, scratch.size() - field_start);
}

vector<string_view> get_token_views_maybe_quoted(string_view line, char separator,
                                                 bool file_has_quotes, string& scratch)
{
    vector<string_view> out;
    get_token_views_maybe_quoted(line, separator, file_has_quotes, scratch, out);
    return out;
}

string_view first_token_maybe_quoted(string_view line, char separator, bool file_has_quotes, string& scratch)
{
    if (!file_has_quotes || line.find('"') == string_view::npos)
    {
        const size_t pos = line.find(separator);
        return pos == string_view::npos ? line : line.substr(0, pos);
    }

    scratch.clear();
    scratch.reserve(line.size());

    bool in_quote = false;

    for (const char c : line)
    {
        if (c == '"' && (in_quote || scratch.empty())) { in_quote = !in_quote; continue; }
        if (!in_quote && c == separator) break;
        if (in_quote && (c == ',' || c == ';')) continue;
        scratch.push_back(c);
    }

    return string_view(scratch.data(), scratch.size());
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

    size_t start_pos = 0;

    while ((start_pos = text.find(to_replace, start_pos)) != string::npos)
    {
        const bool is_prefix_valid = (start_pos == 0)
            || (!isalnum(static_cast<unsigned char>(text[start_pos - 1])) && text[start_pos - 1] != '_');

        const size_t end_pos = start_pos + to_replace.length();
        const bool is_suffix_valid = (end_pos == text.length())
            || (!isalnum(static_cast<unsigned char>(text[end_pos])) && text[end_pos] != '_');

        if (is_prefix_valid && is_suffix_valid)
        {
            text.replace(start_pos, to_replace.length(), replace_with);
            start_pos += replace_with.length();
        }
        else
            start_pos += 1;
    }
}

void replace_all_appearances(string& text, const string& to_replace, const string& replace_with)
{
    if (to_replace.empty()) return;

    string buffer;
    size_t position = 0;
    size_t previous_position;

    buffer.reserve(text.size());

    while (true)
    {
        previous_position = position;

        position = text.find(to_replace, position);

        if (position == string::npos) break;

        buffer.append(text, previous_position, position - previous_position);

        buffer += (!buffer.empty() && buffer.back() == '_') ? to_replace : replace_with;

        position += to_replace.size();
    }

    buffer.append(text, previous_position, text.size() - previous_position);

    text.swap(buffer);
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

void display_progress_bar(int completed, int total)
{
    const int width = 50;
    const float progress = total > 0 ? static_cast<float>(completed) / total : 0.0f;
    const int position = min(static_cast<int>(width * progress), width);

    cout << "\r[" << string(position, '=');

    if (position < width)
        cout << ">" << string(width - position - 1, ' ');

    cout << "] " << int(progress * 100.0) << " %   ";
    cout.flush();
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
    const char* const value = getenv(name);
    if (!value) return false;

    const string_view text(value);

    const initializer_list<string_view> enabled_values{"1", "true", "on", "yes"};
    return ranges::any_of(enabled_values,
                          [text](string_view enabled_value)
                          {
                              return equal_ignoring_case(text, enabled_value);
                          });
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
