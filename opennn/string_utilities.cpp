//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S T R I N G   U T I L I T I E S
//
//   Artificial Intelligence Techniques, SL
//   artelnics@artelnics.com

#include "string_utilities.h"
#include <regex>
#include <cerrno>
#include <cstdlib>

namespace opennn
{

void prepare_line(string& line)
{
    //decode(line);
    trim(line);
    normalize_csv_line(line);
    erase(line, '"');
}

Index count_non_empty_lines(const filesystem::path& data_path)
{
    ifstream file(data_path);

    if(!file.is_open())
        throw runtime_error("Cannot open file: " + data_path.string() + "\n");

    Index count = 0;

    string line;

    while (getline(file, line))
    {
        prepare_line(line);

        if(!line.empty())
            count++;
    }

    return count;
}

Index count_tokens(const string& text, const string& separator)
{
    Index tokens_number = 0;

    string::size_type position = 0;

    while((position = text.find(separator, position)) != string::npos)
    {
        tokens_number++;
        position += separator.length();
    }

    if(text.find(separator, 0) == 0)
        tokens_number--;

    if(position == text.length())
        tokens_number--;

    return tokens_number + 1;
}

vector<string> tokenize(const string& document)
{
    vector<string> tokens;
    string current_token;

    for(const char c : document)
    {
        const unsigned char uc = static_cast<unsigned char>(c);

        if(isalnum(uc))
        {
            current_token += static_cast<char>(tolower(uc));
        }
        else
        {
            if(!current_token.empty())
            {
                tokens.emplace_back(std::move(current_token));
                current_token.clear();
            }

            if(ispunct(uc))
                tokens.emplace_back(1, c);
        }
    }

    if(!current_token.empty())
        tokens.emplace_back(std::move(current_token));

    // @todo -> this is only for encoder-decoder
    // if(!tokens.empty())
    // {
    //     tokens.insert(tokens.begin(), START_TOKEN);
    //     tokens.emplace_back(END_TOKEN);
    // }

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

vector<string_view> get_tokens_fast(string_view text, string_view separator)
{
    vector<string_view> tokens;

    size_t start = 0;

    while (start < text.length())
    {
        const size_t end = text.find(separator, start);

        if (end == string_view::npos)
        {
            tokens.push_back(text.substr(start));
            break;
        }

        tokens.push_back(text.substr(start, end - start));
        start = end + separator.length();
    }

    return tokens;
}

vector<string> convert_string_vector(const vector<vector<string>>& input_vector, const string& separator)
{
    vector<string> vector_result;
    vector_result.reserve(input_vector.size());

    for(const auto& subvec : input_vector)
    {
        string joined;
        for(size_t i = 0; i < subvec.size(); ++i)
        {
            if(i) joined += separator;
            joined += subvec[i];
        }
        vector_result.push_back(std::move(joined));
    }

    return vector_result;
}

VectorR to_type_vector(const string& text, const string& separator)
{
    const vector<string> tokens = get_tokens(text, separator);

    const Index tokens_size = tokens.size();

    VectorR type_vector(tokens_size);

    for(Index i = 0; i < tokens_size; i++)
    {
        const char* begin = tokens[i].c_str();
        char* end = nullptr;
        errno = 0;
        const float value = strtof(begin, &end);

        // NaN if parse failed, overflow, or trailing garbage.
        type_vector(i) = (end == begin || errno == ERANGE || *end != '\0')
            ? type(nan(""))
            : type(value);
    }

    return type_vector;
}

bool is_numeric_string(const string& text)
{
    if(text.empty()) return false;

    const char* begin = text.c_str();
    char* end = nullptr;
    errno = 0;
    strtod(begin, &end);

    if(end == begin || errno == ERANGE) return false;

    const size_t consumed = static_cast<size_t>(end - begin);

    return consumed == text.size()
        || (text.find('%') != string::npos && consumed + 1 == text.size());
}

bool is_date_time_string(const string& text)
{
    if(is_numeric_string(text))
        return false;

    const static vector<regex> date_regexes = {
        // YMD
        regex(R"((\d{4})[-/.](\d{1,2})[-/.](\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2})\.(\d+))"), // yyyy/mm/dd hh:mm:ss.sss
        regex(R"((\d{4})[-/.](\d{1,2})[-/.](\d{1,2}) (\d{1,2}):(\d{1,2}):(\d{1,2}))"), // yyyy/mm/dd hh:mm:ss
        regex(R"((\d{4})[-/.](\d{1,2})[-/.](\d{1,2}) (\d{1,2}):(\d{1,2}))"), // yyyy/mm/dd hh:mm
        regex(R"((\d{4})[-/.](\d{1,2})[-/.](\d{1,2}))"), // yyyy/mm/dd
        regex(R"((\d{4})[-/.](\d{1,2}))"), // yyyy/mm
        // DMY/MDY
        regex(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}) (\d{1,2}):(\d{2}):(\d{2}))"), // dd/mm/yyyy hh:mm:ss
        regex(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}) (\d{1,2}):(\d{2}))"), // dd/mm/yyyy hh:mm
        regex(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}))"), // dd/mm/yyyy
        regex(R"((\d{1,2})[-/.](\d{1,2})[-/.](\d{4}) (\d{1,2}):(\d{2}):(\d{2}) ([AP]M))"), // dd/mm/yyyy hh:mm:ss AM/PM
        // HMS
        regex(R"((\d{1,2}):(\d{1,2}):(\d{1,2}))") // hh:mm:ss
    };

    return any_of(date_regexes.begin(), date_regexes.end(),
                  [&](const regex& r) { return regex_match(text, r); });
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

    struct tm t = {};
    smatch m;

    const bool try_ymd = (format == YMD || format == AUTO);
    const bool try_dmy = (format == DMY || format == MDY || format == AUTO);

    auto fill_dmy = [&](int part1, int part2)
    {
        const bool mdy = (format == MDY) || (format == AUTO && part1 <= 12 && part2 > 12);
        if(mdy) { t.tm_mon = part1 - 1; t.tm_mday = part2; }
        else    { t.tm_mday = part1;    t.tm_mon = part2 - 1; }
    };

    if(try_ymd && regex_match(date, m, re_ymd_hms_ms))
    {
        t.tm_year = stoi(m[1]) - 1900;
        t.tm_mon  = stoi(m[2]) - 1;
        t.tm_mday = stoi(m[3]);
        t.tm_hour = stoi(m[4]) - gmt;
        t.tm_min  = stoi(m[5]);
        t.tm_sec  = stoi(m[6]);
        return mktime(&t);
    }
    if(try_ymd && regex_match(date, m, re_ymd_hms))
    {
        t.tm_year = stoi(m[1]) - 1900;
        t.tm_mon  = stoi(m[2]) - 1;
        t.tm_mday = stoi(m[3]);
        t.tm_hour = stoi(m[4]) - gmt;
        t.tm_min  = stoi(m[5]);
        t.tm_sec  = stoi(m[6]);
        return mktime(&t);
    }
    if(try_ymd && regex_match(date, m, re_ymd_hm))
    {
        t.tm_year = stoi(m[1]) - 1900;
        t.tm_mon  = stoi(m[2]) - 1;
        t.tm_mday = stoi(m[3]);
        t.tm_hour = stoi(m[4]) - gmt;
        t.tm_min  = stoi(m[5]);
        return mktime(&t);
    }
    if(try_ymd && regex_match(date, m, re_ymd))
    {
        t.tm_year = stoi(m[1]) - 1900;
        t.tm_mon  = stoi(m[2]) - 1;
        t.tm_mday = stoi(m[3]);
        return mktime(&t);
    }
    if(try_ymd && regex_match(date, m, re_ym))
    {
        t.tm_year = stoi(m[1]) - 1900;
        t.tm_mon  = stoi(m[2]) - 1;
        t.tm_mday = 1;
        return mktime(&t);
    }
    if(try_dmy && regex_match(date, m, re_dmy_hms))
    {
        fill_dmy(stoi(m[1]), stoi(m[2]));
        t.tm_year = stoi(m[3]) - 1900;

        int hour = stoi(m[4]);
        if(m[7].matched)
        {
            const string ampm = m[7].str();
            if(ampm == "PM" && hour < 12) hour += 12;
            if(ampm == "AM" && hour == 12) hour = 0;
        }

        t.tm_hour = hour - gmt;
        t.tm_min  = stoi(m[5]);
        t.tm_sec  = stoi(m[6]);
        return mktime(&t);
    }
    if(try_dmy && regex_match(date, m, re_dmy_hm))
    {
        fill_dmy(stoi(m[1]), stoi(m[2]));
        t.tm_year = stoi(m[3]) - 1900;
        t.tm_hour = stoi(m[4]) - gmt;
        t.tm_min  = stoi(m[5]);
        return mktime(&t);
    }
    if(try_dmy && regex_match(date, m, re_dmy))
    {
        fill_dmy(stoi(m[1]), stoi(m[2]));
        t.tm_year = stoi(m[3]) - 1900;
        return mktime(&t);
    }
    if(format == AUTO && regex_match(date, m, re_hms))
    {
        t.tm_hour = stoi(m[1]) - gmt;
        t.tm_min  = stoi(m[2]);
        t.tm_sec  = stoi(m[3]);
        return mktime(&t);
    }

    return -1;
}

void replace_all_word_appearances(string& text, const string& to_replace, const string& replace_with)
{
    size_t start_pos = 0;

    while((start_pos = text.find(to_replace, start_pos)) != string::npos)
    {
        // Verify that there are no letters or underscores before to_replace

        const bool is_prefix_valid = (start_pos == 0) || (!isalnum(text[start_pos - 1]) && text[start_pos - 1] != '_');

        // Verify that there are no letters or underscores after to_replace

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

    // Reserves rough estimate of override size of string
    buffer.reserve(text.size());

    while(true)
    {
        previous_position = position;

        position = text.find(to_replace, position);

        if(position == string::npos) break;

        buffer.append(text, previous_position, position - previous_position);

        buffer += (!buffer.empty() && buffer.back() == '_') ? to_replace : replace_with;

        position += to_replace.size();
    }

    buffer.append(text, previous_position, text.size() - previous_position);

    text.swap(buffer);
}

void trim(string& text)
{
    text.erase(0, text.find_first_not_of(" \t\n\r\f\v\b"));
    text.erase(text.find_last_not_of(" \t\n\r\f\v\b") + 1);
}

void normalize_csv_line(string& text)
{
    replace_first_and_last_char_with_missing_label(text, ';', "NA", "");
    replace_first_and_last_char_with_missing_label(text, ',', "NA", "");

    replace_double_char_with_label(text, ";", "NA");
    replace_double_char_with_label(text, ",", "NA");

    replace_substring_within_quotes(text, ",", "");
    replace_substring_within_quotes(text, ";", "");
}

void replace_first_and_last_char_with_missing_label(string &str, char target_char, const string &first_missing_label, const string &last_missing_label)
{
    if(str.empty()) return;

    if(str.front() == target_char)
        str.insert(0, first_missing_label);

    if(str.back() == target_char)
        str.append(last_missing_label);
}

void replace_double_char_with_label(string &str, const string &target_char, const string &missing_label)
{
    replace(str, target_char + target_char, target_char + missing_label + target_char);
}

void replace_substring_within_quotes(string &str, const string &target, const string &replacement)
{
    const regex r("\"([^\"]*)\"");
    string result;
    string::const_iterator search_start(str.begin());
    smatch match;

    while (regex_search(search_start, str.cend(), match, r))
    {
        result += string(search_start, match[0].first);

        string quoted_content = match[1].str();
        replace(quoted_content, target, replacement);

        result += "\"" + quoted_content + "\"";
        search_start = match[0].second;
    }

    result += string(search_start, str.cend()); 
    str = result;
}

void erase(string& text, char character)
{
    text.erase(remove(text.begin(), text.end(), character), text.end());
}

string get_trimmed(const string& text)
{
    const auto is_space = [](char c) { return isspace(static_cast<unsigned char>(c)); };

    const auto start = find_if_not(text.begin(), text.end(), is_space);
    const auto end = find_if_not(text.rbegin(), text.rend(), is_space).base();

    return (start < end) ? string(start, end) : string();
}

bool has_numbers(const vector<string>& string_list)
{
    return any_of(string_list.begin(), string_list.end(), is_numeric_string);
}

void replace(string& source, const string& find_what, const string& replace_with)
{
    size_t position = 0;

    while((position = source.find(find_what, position)) != string::npos)
    {
        source.replace(position, find_what.length(), replace_with);

        position += replace_with.length();
    }
}

string get_first_word(const string& line)
{
    return line.substr(0, line.find_first_of(" ="));
}

string get_time(type time)
{
    const int total_seconds = static_cast<int>(time);
    const int hours = total_seconds / 3600;
    const int minutes = (total_seconds % 3600) / 60;
    const int seconds = total_seconds % 60;

    ostringstream elapsed_time;
    elapsed_time << setfill('0')
                 << setw(2) << hours << ":"
                 << setw(2) << minutes << ":"
                 << setw(2) << seconds;

    return elapsed_time.str();
}

void replace_substring_in_string (vector<string>& tokens, string& expression, const string& keyword)
{
    string::size_type previous_pos = 0;

    for(const string& token : tokens)
    {
        //const string to_replace(token);

        const string new_word = keyword + " " + token;

        string::size_type position = 0;

        while((position = expression.find(token, position)) != string::npos)
        {
            if(position > previous_pos)
            {
                expression.replace(position, token.length(), new_word);
                position += new_word.length();
                previous_pos = position;
                break;
            }
            else
            {
                position += new_word.length();
            }
        }
    }
}

void display_progress_bar(const int& completed, const int& total)
{
    const int width = 100;
    const float progress = static_cast<float>(completed) / total;
    const int position = min(static_cast<int>(width * progress), width);

    cout << "[" << string(position, '=');

    if(position < width)
        cout << ">" << string(width - position - 1, ' ');

    cout << "] " << int(progress * 100.0) << " %\r";
    cout.flush();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
