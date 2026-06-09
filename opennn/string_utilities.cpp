//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T R I N G   U T I L I T I E S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "string_utilities.h"
#include <regex>
#include <cctype>
#include <cerrno>
#include <cstdlib>

namespace opennn
{

float parse_float(const string& text, const string& context)
{
    try { return stof(text); }
    catch (const exception&)
    {
        throw runtime_error(format("{}: invalid numeric value \"{}\".", context, text));
    }
}

int parse_int(const string& text, const string& context)
{
    try { return stoi(text); }
    catch (const exception&)
    {
        throw runtime_error(format("{}: invalid integer value \"{}\".", context, text));
    }
}

long parse_long(const string& text, const string& context)
{
    try { return stol(text); }
    catch (const exception&)
    {
        throw runtime_error(format("{}: invalid integer value \"{}\".", context, text));
    }
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

vector<string> get_tokens(const string& text, const string& separator)
{
    vector<string> tokens;

    const size_t sep_len = separator.length();
    size_t start = 0;

    while (start <= text.length())
    {
        const size_t end = text.find(separator, start);

        tokens.emplace_back((end == string_view::npos)
                                ? text.substr(start)
                                : text.substr(start, end - start));

        if (end == string_view::npos)
            break;

        start = end + sep_len;
    }

    return tokens;
}

vector<string_view> get_token_views(string_view text, char separator)
{
    vector<string_view> tokens;

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

    return tokens;
}

string_view trim_view(string_view text)
{
    constexpr string_view whitespace = " \t\n\r\f\v\b";

    const size_t start = text.find_first_not_of(whitespace);
    if (start == string_view::npos) return {};

    const size_t end = text.find_last_not_of(whitespace);
    return text.substr(start, end - start + 1);
}

vector<string> convert_string_vector(const vector<vector<string>>& input_vector, const string& separator)
{
    vector<string> vector_result;
    vector_result.reserve(input_vector.size());

    for (const auto& subvec : input_vector)
    {
        string joined;
        for (size_t i = 0; i < subvec.size(); ++i)
        {
            if (i) joined += separator;
            joined += subvec[i];
        }
        vector_result.push_back(move(joined));
    }

    return vector_result;
}

bool is_numeric_string(string_view text)
{
    if (text.empty()) return false;

    double value;
    const char* first = text.data();
    const char* last  = first + text.size();
    auto [ptr, ec] = from_chars(first, last, value);

    if (ec != errc{} || ptr == first) return false;

    const size_t consumed = static_cast<size_t>(ptr - first);

    return consumed == text.size()
        || (text.find('%') != string_view::npos && consumed + 1 == text.size());
}

bool is_date_time_string(string_view text)
{
    if (is_numeric_string(text))
        return false;

    const static vector<regex> date_regexes = {
        regex(R"((\d{4})[-/.](\d{1,2})[-/.](\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2})\.(\d+))"),
        regex(R"((\d{4})[-/.](\d{1,2})[-/.](\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2}))"),
        regex(R"((\d{4})[-/.](\d{1,2})[-/.](\d{1,2}) (\d{1,2}):(\d{1,2}))"),
        regex(R"((\d{4})[-/.](\d{1,2})[-/.](\d{1,2}))"),
        regex(R"((\d{4})[-/.](\d{1,2}))"),
        regex(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}) (\d{1,2}):(\d{2}):(\d{2}))"),
        regex(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}) (\d{1,2}):(\d{2}))"),
        regex(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}))"),
        regex(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}) (\d{1,2}):(\d{2}):(\d{2}) ([AP]M))"),
        regex(R"((\d{1,2}):(\d{1,2}):(\d{1,2}))")
    };

    return ranges::any_of(date_regexes,
                          [&](const regex& date_regex) { return regex_match(text.begin(), text.end(), date_regex); });
}

time_t date_to_timestamp(const string& date, Index gmt, const DateFormat& format)
{
    static const regex re_ymd_hms_ms(R"((\d{4})[-/.](\d{1,2})[-/.](\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2})\.(\d+))");
    static const regex re_ymd_hms(R"((\d{4})[-/.](\d{1,2})[-/.](\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2}))");
    static const regex re_ymd_hm(R"((\d{4})[-/.](\d{1,2})[-/.](\d{1,2}) (\d{1,2}):(\d{1,2}))");
    static const regex re_ymd(R"((\d{4})[-/.](\d{1,2})[-/.](\d{1,2}))");
    static const regex re_ym(R"((\d{4})[-/.](\d{1,2}))");
    static const regex re_dmy_hms(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}) (\d{1,2}):(\d{1,2}):(\d{1,2})((?: ([AP]M))?)?)");
    static const regex re_dmy_hm(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}) (\d{1,2}):(\d{1,2}))");
    static const regex re_dmy(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}))");
    static const regex re_hms(R"((\d{1,2}):(\d{1,2}):(\d{1,2}))");

    struct tm time_components = {};
    smatch match;

    const bool try_ymd = (format == Ymd || format == Auto);
    const bool try_dmy = (format == Dmy || format == Mdy || format == Auto);

    auto fill_dmy = [&](int part1, int part2)
    {
        const bool mdy = (format == Mdy) || (format == Auto && part1 <= 12 && part2 > 12);
        if (mdy) { time_components.tm_mon = part1 - 1; time_components.tm_mday = part2; }
        else    { time_components.tm_mday = part1;    time_components.tm_mon = part2 - 1; }
    };

    if (try_ymd && regex_match(date, match, re_ymd_hms_ms))
    {
        time_components.tm_year = stoi(match[1]) - 1900;
        time_components.tm_mon  = stoi(match[2]) - 1;
        time_components.tm_mday = stoi(match[3]);
        time_components.tm_hour = stoi(match[4]) - gmt;
        time_components.tm_min  = stoi(match[5]);
        time_components.tm_sec  = stoi(match[6]);
        return mktime(&time_components);
    }
    if (try_ymd && regex_match(date, match, re_ymd_hms))
    {
        time_components.tm_year = stoi(match[1]) - 1900;
        time_components.tm_mon  = stoi(match[2]) - 1;
        time_components.tm_mday = stoi(match[3]);
        time_components.tm_hour = stoi(match[4]) - gmt;
        time_components.tm_min  = stoi(match[5]);
        time_components.tm_sec  = stoi(match[6]);
        return mktime(&time_components);
    }
    if (try_ymd && regex_match(date, match, re_ymd_hm))
    {
        time_components.tm_year = stoi(match[1]) - 1900;
        time_components.tm_mon  = stoi(match[2]) - 1;
        time_components.tm_mday = stoi(match[3]);
        time_components.tm_hour = stoi(match[4]) - gmt;
        time_components.tm_min  = stoi(match[5]);
        return mktime(&time_components);
    }
    if (try_ymd && regex_match(date, match, re_ymd))
    {
        time_components.tm_year = stoi(match[1]) - 1900;
        time_components.tm_mon  = stoi(match[2]) - 1;
        time_components.tm_mday = stoi(match[3]);
        return mktime(&time_components);
    }
    if (try_ymd && regex_match(date, match, re_ym))
    {
        time_components.tm_year = stoi(match[1]) - 1900;
        time_components.tm_mon  = stoi(match[2]) - 1;
        time_components.tm_mday = 1;
        return mktime(&time_components);
    }
    if (try_dmy && regex_match(date, match, re_dmy_hms))
    {
        fill_dmy(stoi(match[1]), stoi(match[2]));
        time_components.tm_year = stoi(match[3]) - 1900;

        int hour = stoi(match[4]);
        if (match[8].matched)
        {
            const string ampm = match[8].str();
            if (ampm == "PM" && hour < 12) hour += 12;
            if (ampm == "AM" && hour == 12) hour = 0;
        }

        time_components.tm_hour = hour - gmt;
        time_components.tm_min  = stoi(match[5]);
        time_components.tm_sec  = stoi(match[6]);
        return mktime(&time_components);
    }
    if (try_dmy && regex_match(date, match, re_dmy_hm))
    {
        fill_dmy(stoi(match[1]), stoi(match[2]));
        time_components.tm_year = stoi(match[3]) - 1900;
        time_components.tm_hour = stoi(match[4]) - gmt;
        time_components.tm_min  = stoi(match[5]);
        return mktime(&time_components);
    }
    if (try_dmy && regex_match(date, match, re_dmy))
    {
        fill_dmy(stoi(match[1]), stoi(match[2]));
        time_components.tm_year = stoi(match[3]) - 1900;
        return mktime(&time_components);
    }
    if (format == Auto && regex_match(date, match, re_hms))
    {
        time_components.tm_hour = stoi(match[1]) - gmt;
        time_components.tm_min  = stoi(match[2]);
        time_components.tm_sec  = stoi(match[3]);
        return mktime(&time_components);
    }

    return -1;
}

void replace_all_word_appearances(string& text, const string& to_replace, const string& replace_with)
{
    size_t start_pos = 0;

    while ((start_pos = text.find(to_replace, start_pos)) != string::npos)
    {
        const bool is_prefix_valid = (start_pos == 0) || (!isalnum(text[start_pos - 1]) && text[start_pos - 1] != '_');

        const size_t end_pos = start_pos + to_replace.length();
        const bool is_suffix_valid = (end_pos == text.length()) || (!isalnum(text[end_pos]) && text[end_pos] != '_');

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

string get_trimmed(const string& text)
{
    const auto is_space = [](char character) { return isspace(static_cast<unsigned char>(character)); };

    const auto start = ranges::find_if_not(text, is_space);
    const auto end = find_if_not(text.rbegin(), text.rend(), is_space).base();

    return (start < end) ? string(start, end) : string();
}

bool has_numbers(const vector<string>& string_list)
{
    return ranges::any_of(string_list, is_numeric_string);
}

bool has_numbers(const vector<string_view>& string_list)
{
    return ranges::any_of(string_list, is_numeric_string);
}

void replace(string& source, const string& find_what, const string& replace_with)
{
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
    istringstream stream(input);
    float value;
    vector<float> buffer;

    while (stream >> value)
        buffer.push_back(value);

    values.resize(static_cast<Index>(buffer.size()));
    ranges::copy(buffer, values.data());
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

static bool equal_ignoring_case(string_view left, string_view right) noexcept
{
    return left.size() == right.size()
        && ranges::equal(left, right,
                         [](unsigned char left_char, unsigned char right_char)
                         {
                             return tolower(left_char) == tolower(right_char);
                         });
}

bool env_flag_enabled(const char* name) noexcept
{
    const char* value = getenv(name);
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
