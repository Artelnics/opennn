#include "tests/pch.h"

#include <array>

#include "opennn/core/io_utilities.h"

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
        writer.write(span(payload));
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
        reader.read_at(span(read_back), 0);
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
    reader.read_at(span(chunk), 4);
    EXPECT_EQ(string(chunk.data(), chunk.size()), "EFG");

    reader.close();
    remove_quietly(path);
}

TEST(IoUtilitiesTest, ReadTextFileAndValidateCurrentCache)
{
    const filesystem::path source = make_temp_path("source.txt");
    const filesystem::path cache = make_temp_path("cache.bin");
    remove_quietly(source);
    remove_quietly(cache);

    write_text_file(source, "source");
    write_text_file(cache, "cache");

    const auto cache_time = filesystem::last_write_time(cache);
    filesystem::last_write_time(source, cache_time - chrono::seconds(1));

    EXPECT_EQ(read_text_file(source), "source");
    EXPECT_TRUE(is_file_current(cache, {source}, 5));
    EXPECT_FALSE(is_file_current(cache, {source}, 6));

    filesystem::last_write_time(source, cache_time + chrono::seconds(1));
    EXPECT_FALSE(is_file_current(cache, {source}, 5));

    remove_quietly(source);
    remove_quietly(cache);
}

TEST(IoUtilitiesTest, FileWriterDiscardsTmpWhenNotFinalized)
{
    const filesystem::path tmp = make_temp_path("discard.tmp");
    remove_quietly(tmp);

    {
        FileWriter writer;
        writer.open(tmp);
        const char data[] = "partial";
        writer.write(span(data));
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
