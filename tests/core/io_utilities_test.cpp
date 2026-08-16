#include "tests/pch.h"

#include <array>

#include "opennn/core/io_utilities.h"
#include "opennn/core/json.h"

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

template<typename T>
concept HasPublicJsonKind = requires(T value) { value.kind; };

template<typename T>
concept HasPublicJsonDocumentRoot = requires(T value) { value.root; };

}

TEST(JsonTest, PayloadDeterminesKind)
{
    static_assert(!HasPublicJsonKind<Json>);
    static_assert(!HasPublicJsonDocumentRoot<JsonDocument>);

    Json value(3);
    EXPECT_EQ(value.get_kind(), Json::Kind::Number);
    EXPECT_THROW(value.as_array(), runtime_error);
    EXPECT_THROW(value.as_object(), runtime_error);

    value["name"] = Json("OpenNN");
    ASSERT_TRUE(value.is_object());
    EXPECT_EQ(value.at("name").as_string(), "OpenNN");
    EXPECT_THROW(value.as_array(), runtime_error);

    value.push_back(Json(true));
    ASSERT_TRUE(value.is_array());
    ASSERT_EQ(value.as_array().size(), 1u);
    EXPECT_TRUE(value.as_array().front().as_bool());
    EXPECT_THROW(value.as_object(), runtime_error);
}

TEST(JsonTest, NestedValueRoundTrips)
{
    const string text = R"({"number":3,"items":[true,"OpenNN",null]})";
    const Json value = Json::parse(text);

    ASSERT_TRUE(value.is_object());
    EXPECT_EQ(value.at("number").as_long(), 3);

    const Json::Array& items = value.at("items").as_array();
    ASSERT_EQ(items.size(), 3u);
    EXPECT_TRUE(items[0].as_bool());
    EXPECT_EQ(items[1].as_string(), "OpenNN");
    EXPECT_TRUE(items[2].is_null());
    EXPECT_EQ(value.dump(0), text);
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
