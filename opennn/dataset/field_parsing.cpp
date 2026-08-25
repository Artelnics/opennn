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
        if (trim_view(line).empty()) continue;

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

constexpr size_t maximum_number_length = 64;

template <typename T>
static bool parse_point_decimal(const string_view text, T& value)
{
    const char* const first = text.data();
    const char* const last  = first + text.size();
    const auto [end, error] = from_chars(first, last, value);

    return error == errc{} && end == last;
}

static bool has_digit_groups(const string_view integer_part, const char group_separator)
{
    if (integer_part.empty()) return false;

    size_t group_start = 0;

    for (bool first_group = true; ; first_group = false)
    {
        const size_t separator = integer_part.find(group_separator, group_start);

        const size_t group_end = separator == string_view::npos
                               ? integer_part.size()
                               : separator;

        const size_t group_size = group_end - group_start;

        if (first_group ? (group_size < 1 || group_size > 3) : group_size != 3)
            return false;

        for (size_t i = group_start; i < group_end; ++i)
            if (!isdigit(static_cast<unsigned char>(integer_part[i])))
                return false;

        if (separator == string_view::npos) return true;

        group_start = separator + 1;
    }
}

template <typename T>
static bool parse_real_value(const string_view text, T& value, const NumberFormat& format)
{
    if (text.empty()) return false;

    const bool has_groups = format.group_separator != '\0'
                         && format.group_separator != format.decimal_separator
                         && text.find(format.group_separator) != string_view::npos;

    const bool has_foreign_mark = format.decimal_separator != '.'
                               && text.find(format.decimal_separator) != string_view::npos;

    if (!has_groups && !has_foreign_mark)
        return parse_point_decimal(text, value);

    if (text.size() > maximum_number_length) return false;

    const size_t decimal_mark = text.find(format.decimal_separator);

    if (has_groups
        && decimal_mark != string_view::npos
        && text.find(format.group_separator, decimal_mark) != string_view::npos)
        return false;

    if (has_groups)
    {
        const size_t sign_length = text.front() == '+' || text.front() == '-' ? 1 : 0;

        const size_t integer_end = decimal_mark == string_view::npos
                                 ? text.size()
                                 : decimal_mark;

        if (integer_end < sign_length
            || !has_digit_groups(text.substr(sign_length, integer_end - sign_length),
                                 format.group_separator))
            return false;
    }

    char buffer[maximum_number_length];
    size_t length = 0;

    for (const char character : text)
    {
        if (has_groups && character == format.group_separator) continue;

        buffer[length++] = character == format.decimal_separator
                         ? '.'
                         : character;
    }

    return parse_point_decimal(string_view(buffer, length), value);
}

bool parse_real(const string_view text, float& value, const NumberFormat& format)
{
    return parse_real_value(text, value, format);
}

bool parse_real(const string_view text, double& value, const NumberFormat& format)
{
    return parse_real_value(text, value, format);
}

bool is_numeric_string(const string_view text, const NumberFormat& format)
{
    if (text.empty()) return false;

    const string_view number = text.back() == '%'
                             ? text.substr(0, text.size() - 1)
                             : text;

    double value;

    return parse_real_value(number, value, format);
}

void vote_number_format(const string_view text, NumberFormatVotes& votes)
{
    if (text.empty() || text.size() > maximum_number_length) return;

    size_t digits = 0;
    size_t commas = 0;
    size_t points = 0;

    size_t last_comma = 0;
    size_t last_point = 0;

    for (size_t position = 0; position < text.size(); ++position)
    {
        const char character = text[position];

        if (isdigit(static_cast<unsigned char>(character)))
            ++digits;
        else if (character == ',')
            ++commas, last_comma = position;
        else if (character == '.')
            ++points, last_point = position;
        else if (position > 0 || (character != '+' && character != '-'))
            return;
    }

    if (digits == 0 || commas + points == 0) return;

    const auto marks_a_decimal = [&](const size_t mark)
    {
        return text.size() - mark - 1 != 3;
    };

    if (commas > 0 && points > 0)
    {
        if (last_comma > last_point && commas == 1)
            ++votes.comma_decimal, ++votes.point_group;
        else if (last_point > last_comma && points == 1)
            ++votes.point_decimal, ++votes.comma_group;
    }
    else if (commas > 1)
        ++votes.comma_group;
    else if (points > 1)
        ++votes.point_group;
    else if (commas == 1 && marks_a_decimal(last_comma))
        ++votes.comma_decimal;
    else if (points == 1 && marks_a_decimal(last_point))
        ++votes.point_decimal;
}

NumberFormat decide_number_format(const NumberFormatVotes& votes)
{
    if (votes.point_decimal == 0 && (votes.comma_decimal > 0 || votes.point_group > 0))
        return {',', '.'};

    if (votes.comma_decimal == 0 && votes.comma_group > 0)
        return {'.', ','};

    return {};
}

string number_format_name(const char separator)
{
    return separator == ','  ? "Comma"
         : separator == '.'  ? "Point"
         : separator == '\0' ? "None"
                             : string(1, separator);
}

char number_format_separator(const string& name, const string_view context)
{
    if (name == "Comma") return ',';
    if (name == "Point") return '.';
    if (name == "None")  return '\0';

    throw runtime_error(format("{}: unknown number separator \"{}\".", context, name));
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
