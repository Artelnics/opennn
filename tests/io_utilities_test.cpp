#include "pch.h"

#include <array>

#include "../opennn/io_utilities.h"

using namespace opennn;

namespace
{

filesystem::path make_temp_path(const string& name)
{
    return filesystem::temp_directory_path()
         / ("opennn_io_utilities_test_" + to_string(::getpid()) + "_" + name);
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

TEST(IoUtilitiesTest, IsNumericStringBasic)
{
    EXPECT_TRUE(is_numeric_string("0"));
    EXPECT_TRUE(is_numeric_string("123"));
    EXPECT_TRUE(is_numeric_string("-42"));
    EXPECT_TRUE(is_numeric_string("3.14159"));
    EXPECT_TRUE(is_numeric_string("1e9"));
    EXPECT_TRUE(is_numeric_string("-2.5e-3"));
}

TEST(IoUtilitiesTest, IsNumericStringNonNumeric)
{
    EXPECT_FALSE(is_numeric_string(""));
    EXPECT_FALSE(is_numeric_string("abc"));
    EXPECT_FALSE(is_numeric_string("12abc"));
    EXPECT_FALSE(is_numeric_string("1.2.3"));
    EXPECT_FALSE(is_numeric_string("nan_text"));
}

TEST(IoUtilitiesTest, IsNumericStringPercent)
{
    EXPECT_TRUE(is_numeric_string("50%"));
    EXPECT_TRUE(is_numeric_string("12.5%"));
    EXPECT_FALSE(is_numeric_string("%"));
    EXPECT_FALSE(is_numeric_string("5%5"));
}

TEST(IoUtilitiesTest, IsDateTimeString)
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

TEST(IoUtilitiesTest, HasNumbers)
{
    const vector<string> with_number = {"alpha", "beta", "7", "gamma"};
    const vector<string> without_number = {"alpha", "beta", "gamma"};

    EXPECT_TRUE(has_numbers(with_number));
    EXPECT_FALSE(has_numbers(without_number));

    const vector<string_view> views_with = {"x", "9.5"};
    const vector<string_view> views_without = {"x", "y"};

    EXPECT_TRUE(has_numbers(views_with));
    EXPECT_FALSE(has_numbers(views_without));
}

TEST(IoUtilitiesTest, DateToTimestampRoundTrip)
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

TEST(IoUtilitiesTest, DateToTimestampInvalid)
{
    EXPECT_EQ(date_to_timestamp("not a date", 0, Auto), time_t(-1));
}

TEST(IoUtilitiesTest, FileWriterReaderRoundTrip)
{
    const filesystem::path tmp = make_temp_path("rw.tmp");
    const filesystem::path final = make_temp_path("rw.bin");

    remove_quietly(tmp);
    remove_quietly(final);

    const vector<uint8_t> payload = {1, 2, 3, 4, 5, 250, 128, 0, 255};

    {
        FileWriter writer;
        writer.open(tmp);
        writer.write(payload.data(), payload.size());
        writer.finish_with_rename(final);
    }

    EXPECT_FALSE(filesystem::exists(tmp));
    ASSERT_TRUE(filesystem::exists(final));

    {
        FileReader reader;
        reader.open(final);
        ASSERT_TRUE(reader.is_open());
        EXPECT_EQ(reader.file_size(), uint64_t(payload.size()));

        vector<uint8_t> read_back(payload.size(), 0);
        reader.read_at(read_back.data(), read_back.size(), 0);
        EXPECT_EQ(read_back, payload);

        reader.close();
        EXPECT_FALSE(reader.is_open());
    }

    remove_quietly(final);
}

TEST(IoUtilitiesTest, FileReaderReadAtOffset)
{
    const filesystem::path path = make_temp_path("offset.bin");
    remove_quietly(path);

    const string content = "ABCDEFGHIJ";
    write_text_file(path, content);

    FileReader reader;
    reader.open(path);
    ASSERT_TRUE(reader.is_open());
    EXPECT_EQ(reader.file_size(), uint64_t(content.size()));

    std::array<char, 3> chunk = {0, 0, 0};
    reader.read_at(chunk.data(), chunk.size(), 4);
    EXPECT_EQ(string(chunk.data(), chunk.size()), "EFG");

    reader.close();
    remove_quietly(path);
}

TEST(IoUtilitiesTest, FileWriterDiscardsTmpWhenNotFinalized)
{
    const filesystem::path tmp = make_temp_path("discard.tmp");
    remove_quietly(tmp);

    {
        FileWriter writer;
        writer.open(tmp);
        const char data[] = "partial";
        writer.write(data, sizeof(data));
    }

    EXPECT_FALSE(filesystem::exists(tmp));
    remove_quietly(tmp);
}

TEST(IoUtilitiesTest, FileMappingMapsContent)
{
    const filesystem::path path = make_temp_path("mapping.txt");
    remove_quietly(path);

    const string content = "mapped content here";
    write_text_file(path, content);

    FileMapping mapping;
    ASSERT_TRUE(mapping.map(path));
    ASSERT_EQ(mapping.size(), content.size());

    const string mapped(mapping.data(), mapping.size());
    EXPECT_EQ(mapped, content);

    mapping.reset();
    EXPECT_EQ(mapping.size(), size_t(0));

    remove_quietly(path);
}

TEST(IoUtilitiesTest, FileMappingFailsOnMissing)
{
    const filesystem::path path = make_temp_path("does_not_exist.txt");
    remove_quietly(path);

    FileMapping mapping;
    EXPECT_FALSE(mapping.map(path));
}

TEST(IoUtilitiesTest, CsvReaderCommaSeparated)
{
    const filesystem::path path = make_temp_path("comma.csv");
    remove_quietly(path);

    write_text_file(path, "a,b,c\n1,2,3\n4,5,6\n");

    CsvReader::Configuration configuration;
    configuration.separator = ',';

    CsvReader reader(configuration);
    const CsvReader::Result result = reader.read(path);

    EXPECT_EQ(result.separator, ',');
    ASSERT_EQ(result.lines.size(), size_t(3));
    EXPECT_EQ(string(result.lines[0]), "a,b,c");
    EXPECT_EQ(string(result.lines[1]), "1,2,3");
    EXPECT_EQ(string(result.lines[2]), "4,5,6");

    remove_quietly(path);
}

TEST(IoUtilitiesTest, CsvReaderSkipsBlankLinesAndCarriageReturns)
{
    const filesystem::path path = make_temp_path("blanks.csv");
    remove_quietly(path);

    write_text_file(path, "x,y\r\n\r\n1,2\r\n   \n3,4\r\n");

    CsvReader::Configuration configuration;
    configuration.separator = ',';

    CsvReader reader(configuration);
    const CsvReader::Result result = reader.read(path);

    ASSERT_EQ(result.lines.size(), size_t(3));
    EXPECT_EQ(string(result.lines[0]), "x,y");
    EXPECT_EQ(string(result.lines[1]), "1,2");
    EXPECT_EQ(string(result.lines[2]), "3,4");

    remove_quietly(path);
}

TEST(IoUtilitiesTest, CsvReaderQuotedFieldsStripped)
{
    const filesystem::path path = make_temp_path("quoted.csv");
    remove_quietly(path);

    write_text_file(path, "name,note\n\"hello, world\",ok\n\"a;b\",plain\n");

    CsvReader::Configuration configuration;
    configuration.separator = ',';

    CsvReader reader(configuration);
    const CsvReader::Result result = reader.read(path);

    ASSERT_EQ(result.lines.size(), size_t(3));
    EXPECT_EQ(string(result.lines[0]), "name,note");
    EXPECT_EQ(string(result.lines[1]), "hello world,ok");
    EXPECT_EQ(string(result.lines[2]), "ab,plain");

    remove_quietly(path);
}

TEST(IoUtilitiesTest, CsvReaderStripsBom)
{
    const filesystem::path path = make_temp_path("bom.csv");
    remove_quietly(path);

    write_text_file(path, "\xEF\xBB\xBF" "h1,h2\n10,20\n");

    CsvReader::Configuration configuration;
    configuration.separator = ',';

    CsvReader reader(configuration);
    const CsvReader::Result result = reader.read(path);

    ASSERT_EQ(result.lines.size(), size_t(2));
    EXPECT_EQ(string(result.lines[0]), "h1,h2");
    EXPECT_EQ(string(result.lines[1]), "10,20");

    remove_quietly(path);
}

TEST(IoUtilitiesTest, CsvReaderSemicolonSeparator)
{
    const filesystem::path path = make_temp_path("semicolon.csv");
    remove_quietly(path);

    write_text_file(path, "a;b;c\n7;8;9\n");

    CsvReader::Configuration configuration;
    configuration.separator = ';';

    CsvReader reader(configuration);
    const CsvReader::Result result = reader.read(path);

    EXPECT_EQ(result.separator, ';');
    ASSERT_EQ(result.lines.size(), size_t(2));
    EXPECT_EQ(string(result.lines[0]), "a;b;c");
    EXPECT_EQ(string(result.lines[1]), "7;8;9");

    remove_quietly(path);
}

TEST(IoUtilitiesTest, CsvReaderLineValidatorRuns)
{
    const filesystem::path path = make_temp_path("validator.csv");
    remove_quietly(path);

    write_text_file(path, "row1\nrow2\nrow3\n");

    Index validated_count = 0;

    CsvReader::Configuration configuration;
    configuration.separator = ',';
    configuration.line_validator = [&validated_count](string_view) { ++validated_count; };

    CsvReader reader(configuration);
    const CsvReader::Result result = reader.read(path);

    ASSERT_EQ(result.lines.size(), size_t(3));
    EXPECT_EQ(validated_count, Index(3));

    remove_quietly(path);
}

TEST(IoUtilitiesTest, CsvReaderEmptyPathThrows)
{
    CsvReader::Configuration configuration;
    CsvReader reader(configuration);

    EXPECT_THROW((void)reader.read(filesystem::path()), runtime_error);
}
