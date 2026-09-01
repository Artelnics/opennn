#include "tests/pch.h"

#include <array>

#include "opennn/dataset/field_parsing.h"

using namespace opennn;

namespace
{

filesystem::path make_temp_path(const string& name)
{
    return filesystem::temp_directory_path()
         / ("opennn_field_parsing_test_" + to_string(::getpid()) + "_" + name);
}

void write_text_file(const filesystem::path& path, const string& content)
{
    ofstream stream(path, ios::binary | ios::trunc);
    stream.write(content.data(), streamsize(content.size()));
    stream.close();
}

void remove_quietly(const filesystem::path& path)
{
    error_code ec;
    filesystem::remove(path, ec);
}

}

TEST(FieldParsingTest, IsNumericStringBasic)
{
    EXPECT_TRUE(is_numeric_string("0"));
    EXPECT_TRUE(is_numeric_string("123"));
    EXPECT_TRUE(is_numeric_string("-42"));
    EXPECT_TRUE(is_numeric_string("3.14159"));
    EXPECT_TRUE(is_numeric_string("1e9"));
    EXPECT_TRUE(is_numeric_string("-2.5e-3"));
}

TEST(FieldParsingTest, IsNumericStringNonNumeric)
{
    EXPECT_FALSE(is_numeric_string(""));
    EXPECT_FALSE(is_numeric_string("abc"));
    EXPECT_FALSE(is_numeric_string("12abc"));
    EXPECT_FALSE(is_numeric_string("1.2.3"));
    EXPECT_FALSE(is_numeric_string("nan_text"));
}

TEST(FieldParsingTest, IsNumericStringPercent)
{
    EXPECT_TRUE(is_numeric_string("50%"));
    EXPECT_TRUE(is_numeric_string("12.5%"));
    EXPECT_FALSE(is_numeric_string("%"));
    EXPECT_FALSE(is_numeric_string("5%5"));
}

TEST(FieldParsingTest, ParseRealDefaultFormat)
{
    float value = 0.0f;

    EXPECT_TRUE(parse_real("3.14159", value));
    EXPECT_NEAR(value, 3.14159f, 1e-5f);

    EXPECT_TRUE(parse_real("-2.5e-3", value));
    EXPECT_NEAR(value, -0.0025f, 1e-9f);

    EXPECT_FALSE(parse_real("24,44", value));
    EXPECT_FALSE(parse_real("50%", value));
    EXPECT_FALSE(parse_real("", value));
}

TEST(FieldParsingTest, ParseRealDecimalComma)
{
    const NumberFormat european{',', '.'};

    float value = 0.0f;

    EXPECT_TRUE(parse_real("24,44", value, european));
    EXPECT_NEAR(value, 24.44f, 1e-4f);

    EXPECT_TRUE(parse_real("-0,5", value, european));
    EXPECT_NEAR(value, -0.5f, 1e-6f);

    EXPECT_TRUE(parse_real("1.234,56", value, european));
    EXPECT_NEAR(value, 1234.56f, 1e-2f);

    EXPECT_TRUE(parse_real("1.234.567,89", value, european));
    EXPECT_NEAR(value, 1234567.89f, 1.0f);

    EXPECT_TRUE(parse_real("81", value, european));
    EXPECT_NEAR(value, 81.0f, 1e-6f);

    EXPECT_FALSE(parse_real("1.23,4", value, european));
    EXPECT_FALSE(parse_real("1.2345,6", value, european));
    EXPECT_FALSE(parse_real("1,234.5", value, european));
    EXPECT_FALSE(parse_real("24,4,4", value, european));
}

TEST(FieldParsingTest, ParseRealThousandsComma)
{
    const NumberFormat grouped{'.', ','};

    float value = 0.0f;

    EXPECT_TRUE(parse_real("1,234.56", value, grouped));
    EXPECT_NEAR(value, 1234.56f, 1e-2f);

    EXPECT_TRUE(parse_real("1,234,567", value, grouped));
    EXPECT_NEAR(value, 1234567.0f, 1.0f);

    EXPECT_TRUE(parse_real("12.5", value, grouped));
    EXPECT_NEAR(value, 12.5f, 1e-6f);

    EXPECT_FALSE(parse_real("1,23.4", value, grouped));
}

TEST(FieldParsingTest, IsNumericStringHonoursNumberFormat)
{
    const NumberFormat european{',', '.'};

    EXPECT_FALSE(is_numeric_string("24,44"));
    EXPECT_TRUE(is_numeric_string("24,44", european));
    EXPECT_TRUE(is_numeric_string("24,4%", european));
    EXPECT_TRUE(is_numeric_string("1.699", european));
    EXPECT_FALSE(is_numeric_string("abc", european));
}

TEST(FieldParsingTest, DetectNumberFormat)
{
    const auto detect = [](const vector<string>& fields)
    {
        NumberFormatVotes votes;
        for (const string& field : fields) vote_number_format(field, votes);
        return decide_number_format(votes);
    };

    const NumberFormat european = detect({"Female", "24,443011", "1,699998", "81,66995", "no"});
    EXPECT_EQ(european.decimal_separator, ',');
    EXPECT_EQ(european.group_separator, '.');

    EXPECT_TRUE(detect({"Female", "24.443011", "1.699998", "81.66995"}).is_default());
    EXPECT_TRUE(detect({"a", "1", "2", "3"}).is_default());
    EXPECT_TRUE(detect({}).is_default());
    EXPECT_TRUE(detect({"1.5", "2.5"}).is_default());
    EXPECT_TRUE(detect({"2020-01-15", "12:30:45"}).is_default());
    EXPECT_TRUE(detect({"1,234", "5,678"}).is_default());
    EXPECT_TRUE(detect({"24,44", "3.14"}).is_default());

    EXPECT_EQ(detect({"1.234.567", "2.345.678"}).decimal_separator, ',');
    EXPECT_EQ(detect({"1,234,567", "2,345,678"}).group_separator, ',');
    EXPECT_EQ(detect({"1,234.5", "9,876.5"}).group_separator, ',');
}

TEST(FieldParsingTest, NumberFormatNames)
{
    EXPECT_EQ(number_format_name(','), "Comma");
    EXPECT_EQ(number_format_name('.'), "Point");
    EXPECT_EQ(number_format_name('\0'), "None");

    EXPECT_EQ(number_format_separator("Comma", "test"), ',');
    EXPECT_EQ(number_format_separator("Point", "test"), '.');
    EXPECT_EQ(number_format_separator("None", "test"), '\0');

    EXPECT_THROW(number_format_separator("Dot", "test"), runtime_error);
}

TEST(FieldParsingTest, IsDateTimeString)
{
    EXPECT_TRUE(is_date_time_string("2020-01-15"));
    EXPECT_TRUE(is_date_time_string("2020/01/15"));
    EXPECT_TRUE(is_date_time_string("2020-01-15 13:45:30"));
    EXPECT_TRUE(is_date_time_string("15/01/2020"));
    EXPECT_TRUE(is_date_time_string("13:45:30"));

    EXPECT_FALSE(is_date_time_string("not a date"));
    EXPECT_FALSE(is_date_time_string("123"));
    EXPECT_FALSE(is_date_time_string(""));
}

TEST(FieldParsingTest, DateToTimestampRoundTrip)
{
    const time_t timestamp = date_to_timestamp("2020-06-15 12:30:45", 0, Ymd);

    ASSERT_NE(timestamp, time_t(-1));

    struct tm expected = {};
    expected.tm_year = 2020 - 1900;
    expected.tm_mon  = 6 - 1;
    expected.tm_mday = 15;
    expected.tm_hour = 12;
    expected.tm_min  = 30;
    expected.tm_sec  = 45;
    expected.tm_isdst = 0;

    EXPECT_EQ(timestamp, mktime(&expected));

    struct tm decoded = {};
#if defined(_WIN32)
    localtime_s(&decoded, &timestamp);
#else
    localtime_r(&timestamp, &decoded);
#endif

    EXPECT_EQ(decoded.tm_year + 1900, 2020);
    EXPECT_EQ(decoded.tm_mon + 1, 6);
    EXPECT_EQ(decoded.tm_mday, 15);
    EXPECT_EQ(decoded.tm_min, 30);
    EXPECT_EQ(decoded.tm_sec, 45);
}

TEST(FieldParsingTest, DateToTimestampInvalid)
{
    EXPECT_EQ(date_to_timestamp("not a date", 0, Auto), time_t(-1));
}

TEST(FieldParsingTest, DateToTimestampMeridiem)
{
    const auto expected_timestamp = [](int hour)
    {
        struct tm expected = {};
        expected.tm_year = 2020 - 1900;
        expected.tm_mon = 6 - 1;
        expected.tm_mday = 15;
        expected.tm_hour = hour;
        expected.tm_isdst = 0;
        return mktime(&expected);
    };

    EXPECT_EQ(date_to_timestamp("2020-06-15 12:00 AM", 0, Ymd), expected_timestamp(0));
    EXPECT_EQ(date_to_timestamp("2020-06-15 12:00 PM", 0, Ymd), expected_timestamp(12));
    EXPECT_EQ(date_to_timestamp("2020-06-15 1:00 PM", 0, Ymd), expected_timestamp(13));
}

TEST(FieldParsingTest, CsvReaderCommaSeparated)
{
    const filesystem::path path = make_temp_path("comma.csv");
    remove_quietly(path);

    write_text_file(path, "a,b,c\n1,2,3\n4,5,6\n");

    CsvReader reader;
    const CsvReader::Result result = reader.read(path);

    ASSERT_EQ(result.lines.size(), size_t(3));
    EXPECT_EQ(string(result.lines[0]), "a,b,c");
    EXPECT_EQ(string(result.lines[1]), "1,2,3");
    EXPECT_EQ(string(result.lines[2]), "4,5,6");

    remove_quietly(path);
}

TEST(FieldParsingTest, CsvReaderEmptyFile)
{
    const filesystem::path path = make_temp_path("empty.csv");
    remove_quietly(path);
    write_text_file(path, "");

    const CsvReader::Result result = CsvReader().read(path);
    EXPECT_TRUE(result.lines.empty());

    remove_quietly(path);
}

TEST(FieldParsingTest, CsvReaderSkipsBlankLinesAndCarriageReturns)
{
    const filesystem::path path = make_temp_path("blanks.csv");
    remove_quietly(path);

    write_text_file(path, "x,y\r\n\r\n1,2\r\n   \n3,4\r\n");

    CsvReader reader;
    const CsvReader::Result result = reader.read(path);

    ASSERT_EQ(result.lines.size(), size_t(3));
    EXPECT_EQ(string(result.lines[0]), "x,y");
    EXPECT_EQ(string(result.lines[1]), "1,2");
    EXPECT_EQ(string(result.lines[2]), "3,4");

    remove_quietly(path);
}

TEST(FieldParsingTest, CsvReaderPreservesQuotedFieldsForTokenizer)
{
    const filesystem::path path = make_temp_path("quoted.csv");
    remove_quietly(path);

    write_text_file(path, "name,note\n\"hello, world\",ok\n\"a;b\",plain\n");

    CsvReader reader;
    const CsvReader::Result result = reader.read(path);

    ASSERT_EQ(result.lines.size(), size_t(3));
    EXPECT_EQ(string(result.lines[0]), "name,note");
    EXPECT_EQ(string(result.lines[1]), "\"hello, world\",ok");
    EXPECT_EQ(string(result.lines[2]), "\"a;b\",plain");

    remove_quietly(path);
}

TEST(FieldParsingTest, CsvReaderStripsBom)
{
    const filesystem::path path = make_temp_path("bom.csv");
    remove_quietly(path);

    write_text_file(path, "\xEF\xBB\xBF" "h1,h2\n10,20\n");

    CsvReader reader;
    const CsvReader::Result result = reader.read(path);

    ASSERT_EQ(result.lines.size(), size_t(2));
    EXPECT_EQ(string(result.lines[0]), "h1,h2");
    EXPECT_EQ(string(result.lines[1]), "10,20");

    remove_quietly(path);
}

TEST(FieldParsingTest, CsvReaderPreservesSemicolonLines)
{
    const filesystem::path path = make_temp_path("semicolon.csv");
    remove_quietly(path);

    write_text_file(path, "a;b;c\n7;8;9\n");

    CsvReader reader;
    const CsvReader::Result result = reader.read(path);

    ASSERT_EQ(result.lines.size(), size_t(2));
    EXPECT_EQ(string(result.lines[0]), "a;b;c");
    EXPECT_EQ(string(result.lines[1]), "7;8;9");

    remove_quietly(path);
}

TEST(FieldParsingTest, CsvReaderLineValidatorRuns)
{
    const filesystem::path path = make_temp_path("validator.csv");
    remove_quietly(path);

    write_text_file(path, "row1\nrow2\nrow3\n");

    Index validated_count = 0;

    CsvReader reader([&validated_count](string_view) { ++validated_count; });
    const CsvReader::Result result = reader.read(path);

    ASSERT_EQ(result.lines.size(), size_t(3));
    EXPECT_EQ(validated_count, Index(3));

    remove_quietly(path);
}

TEST(FieldParsingTest, CsvReaderEmptyPathThrows)
{
    CsvReader reader;

    EXPECT_THROW((void)reader.read(filesystem::path()), runtime_error);
}
