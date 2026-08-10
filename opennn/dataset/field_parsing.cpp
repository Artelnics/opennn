//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F I E L D   P A R S I N G   S O U R C E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/dataset/field_parsing.h"
#include "opennn/core/string_utilities.h"

#include <array>
#include <cctype>

namespace opennn
{

void CsvReader::parse(Result& out, const string_view content) const
{
    out.lines.reserve(ranges::count(content, '\n') + 1);

    size_t line_start = 0;

    while (line_start < content.size())
    {
        size_t line_end = content.find('\n', line_start);
        if (line_end == string_view::npos) line_end = content.size();

        string_view line = content.substr(line_start, line_end - line_start);
        line_start = line_end + 1;

        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        line = trim_view(line);

        if (line.empty()) continue;

        if (line_validator) line_validator(line);

        out.lines.push_back(line);
    }
}

CsvReader::Result CsvReader::read(const filesystem::path& path) const
{
    throw_if(path.empty(),
             "Data path is empty.\n");

    Result result;
    string_view content;
    const bool mapped = result.mapping.map(path);

    if (mapped)
    {
        content = string_view(result.mapping.data(), result.mapping.size());
    }
    else
    {
        result.buffer = read_text_file(path);
        content = result.buffer;
    }

    constexpr string_view bom = "\xEF\xBB\xBF";
    if (content.starts_with(bom))
    {
        if (mapped)
            content.remove_prefix(bom.size());
        else
        {
            result.buffer.erase(0, bom.size());
            content = result.buffer;
        }
    }

    result.has_quotes = content.find('"') != string_view::npos;
    parse(result, content);
    return result;
}

const vector<string> positive_words = {"1", "yes", "positive", "+", "true", "good", "si", "sí", "Sí"};
const vector<string> negative_words = {"0", "no", "negative", "-", "false", "bad", "not", "No"};

bool is_numeric_string(string_view text)
{
    if (text.empty()) return false;

    double value;
    const char* const first = text.data();
    const char* const last  = first + text.size();
    const auto [end, error] = from_chars(first, last, value);

    return error == errc{}
        && (end == last || (end + 1 == last && *end == '%'));
}

enum class Meridiem {None, Am, Pm};

struct ParsedDateTime
{
    std::array<int, 4> date{};
    std::array<int, 4> time{};
    size_t date_count = 0;
    size_t time_count = 0;
    int year_index = -1;
    Meridiem meridiem = Meridiem::None;
};

static size_t parse_fields(string_view text,
                           string_view separators,
                           std::array<int, 4>& values,
                           int* year_index = nullptr)
{
    if (year_index) *year_index = -1;

    size_t count = 0;
    const char* current = text.data();
    const char* const text_end = current + text.size();

    while (current < text_end)
    {
        if (count == values.size()
            || !isdigit(static_cast<unsigned char>(*current)))
            return 0;

        const auto [field_end, error] = from_chars(current, text_end, values[count]);
        if (error != errc{}) return 0;

        if (year_index && *year_index < 0 && field_end - current == 4)
            *year_index = int(count);

        ++count;
        if (field_end == text_end) break;
        if (separators.find(*field_end) == string_view::npos) return 0;
        current = field_end + 1;
    }

    return count;
}

static optional<ParsedDateTime> parse_date_time(string_view text)
{
    text = trim_view(text);
    if (text.empty()) return nullopt;

    ParsedDateTime parsed;

    if (text.ends_with(" AM"))
        parsed.meridiem = Meridiem::Am;
    else if (text.ends_with(" PM"))
        parsed.meridiem = Meridiem::Pm;

    if (parsed.meridiem != Meridiem::None)
        text.remove_suffix(3);

    const size_t space = text.find(' ');
    if (space == string_view::npos)
    {
        if (text.find(':') != string_view::npos)
        {
            parsed.time_count = parse_fields(text, ":", parsed.time);
            if (parsed.time_count != 3) return nullopt;
        }
        else
        {
            parsed.date_count = parse_fields(text, "-/.", parsed.date, &parsed.year_index);
            if (parsed.date_count < 2 || parsed.date_count > 3) return nullopt;
        }

        return parsed;
    }

    parsed.date_count =
        parse_fields(text.substr(0, space), "-/.", parsed.date, &parsed.year_index);
    if (parsed.date_count < 2 || parsed.date_count > 3)
        return nullopt;

    parsed.time_count = parse_fields(text.substr(space + 1), ":.", parsed.time);
    if (parsed.time_count < 2 || parsed.time_count > 4)
        return nullopt;

    return parsed;
}

bool is_date_time_string(string_view text)
{
    if (is_numeric_string(text)) return false;
    return parse_date_time(text).has_value();
}

DateFormat detect_date_format(string_view text)
{
    const optional<ParsedDateTime> parsed = parse_date_time(text);
    if (!parsed || parsed->date_count != 3) return Auto;
    if (parsed->year_index == 0) return Ymd;
    if (parsed->date[0] > 12) return Dmy;
    if (parsed->date[1] > 12) return Mdy;
    return Auto;
}

time_t date_to_timestamp(string_view text, Index gmt, DateFormat format)
{
    const optional<ParsedDateTime> parsed = parse_date_time(text);
    if (!parsed) return -1;
    if (parsed->date_count == 0 && format != Auto) return -1;

    tm time_components{};

    if (parsed->date_count > 0 && parsed->year_index == 0)
    {
        if (format != Auto && format != Ymd) return -1;
        time_components.tm_year = parsed->date[0] - 1900;
        time_components.tm_mon = parsed->date[1] - 1;
        time_components.tm_mday = parsed->date_count == 3 ? parsed->date[2] : 1;
    }
    else if (parsed->date_count > 0)
    {
        if (parsed->date_count != 3
            || format == Ymd
            || parsed->year_index != int(parsed->date_count - 1))
            return -1;

        const bool month_first = format == Mdy
            || (format == Auto && parsed->date[0] <= 12 && parsed->date[1] > 12);
        time_components.tm_mday = month_first ? parsed->date[1] : parsed->date[0];
        time_components.tm_mon = (month_first ? parsed->date[0] : parsed->date[1]) - 1;
        time_components.tm_year = parsed->date[2] - 1900;
    }

    if (parsed->time_count > 0)
    {
        int hour = parsed->time[0];
        if (parsed->meridiem == Meridiem::Pm && hour < 12) hour += 12;
        if (parsed->meridiem == Meridiem::Am && hour == 12) hour = 0;

        time_components.tm_hour = hour - int(gmt);
        time_components.tm_min = parsed->time[1];
        time_components.tm_sec = parsed->time_count >= 3 ? parsed->time[2] : 0;
    }

    return mktime(&time_components);
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
