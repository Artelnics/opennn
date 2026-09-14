#include "tests/pch.h"

#include <array>
#include <future>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

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

class DownloadFixture
{
public:
#ifdef _WIN32
    using Socket = SOCKET;
    static constexpr Socket invalid_socket = INVALID_SOCKET;
    static void close_socket(Socket socket) { closesocket(socket); }
#else
    using Socket = int;
    static constexpr Socket invalid_socket = -1;
    static void close_socket(Socket socket) { close(socket); }
#endif

    DownloadFixture()
    {
#ifdef _WIN32
        WSADATA data{};
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0)
            throw runtime_error("Cannot initialize the local download fixture.");
#endif
        ScopeExit rollback([this] { close_all(); });
        listener = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listener == invalid_socket)
            throw runtime_error("Cannot create the local download fixture socket.");

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (::bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0
            || ::listen(listener, 2) != 0)
            throw runtime_error("Cannot bind the local download fixture.");
#ifdef _WIN32
        int size = sizeof(address);
#else
        socklen_t size = sizeof(address);
#endif
        if (::getsockname(listener, reinterpret_cast<sockaddr*>(&address), &size) != 0)
            throw runtime_error("Cannot read the local download fixture port.");
        url = "http://127.0.0.1:" + to_string(ntohs(address.sin_port)) + "/model";

        worker = async(launch::async, [this]
        {
            const ScopeExit stop_listening([this]
            {
                close_socket(listener);
                listener = invalid_socket;
            });
            for (int attempt = 0; attempt < 2; ++attempt)
            {
                fd_set ready;
                FD_ZERO(&ready);
                FD_SET(listener, &ready);
                timeval timeout{5, 0};
                if (::select(int(listener + 1), &ready, nullptr, nullptr, &timeout) <= 0)
                    throw runtime_error("Timed out waiting for the download request.");
                const Socket client = ::accept(listener, nullptr, nullptr);
                if (client == invalid_socket)
                    throw runtime_error("Cannot accept the download request.");
                const ScopeExit cleanup([&] { close_socket(client); });
                FD_ZERO(&ready);
                FD_SET(client, &ready);
                timeout = {5, 0};
                if (::select(int(client + 1), &ready, nullptr, nullptr, &timeout) <= 0)
                    throw runtime_error("Timed out reading the download request.");
                std::array<char, 2048> request{};
                if (::recv(client, request.data(), int(request.size()), 0) <= 0)
                    throw runtime_error("Cannot read the download request.");
                const string response = "HTTP/1.1 200 OK\r\nContent-Length: "
                    + to_string(attempt == 0 ? 32 : 7)
                    + "\r\nConnection: close\r\n\r\npayload";
                if (::send(client, response.data(), int(response.size()), 0) != int(response.size()))
                    throw runtime_error("Cannot send the download fixture response.");
            }
        });
        rollback.release();
    }

    ~DownloadFixture()
    {
        if (worker.valid()) worker.wait();
        close_all();
    }

    string url;
    future<void> worker;

private:
    void close_all()
    {
        if (listener != invalid_socket) close_socket(listener);
#ifdef _WIN32
        WSACleanup();
#endif
    }

    Socket listener = invalid_socket;
};

}

TEST(IoUtilitiesTest, InterruptedDownloadRetriesAndPublishesOnlyCompleteFiles)
{
#ifdef _WIN32
    if (system("curl.exe --version >NUL 2>&1") != 0) GTEST_SKIP() << "curl is unavailable.";
#else
    if (system("curl --version >/dev/null 2>&1") != 0) GTEST_SKIP() << "curl is unavailable.";
#endif
    const filesystem::path folder = make_temp_path("download with spaces");
    filesystem::create_directory(folder);
    const filesystem::path path = folder / "model.bin";
    const ScopeExit cleanup([&]
    {
        remove_quietly(path);
        remove_quietly(folder);
    });
    ASSERT_TRUE(filesystem::is_empty(folder));

    DownloadFixture server;
    EXPECT_THROW(download_if_missing(path, server.url), runtime_error);
    EXPECT_FALSE(filesystem::exists(path));
    EXPECT_TRUE(filesystem::is_empty(folder));

    ASSERT_NO_THROW(download_if_missing(path, server.url));
    EXPECT_EQ(read_text_file(path), "payload");
    EXPECT_EQ(distance(filesystem::directory_iterator(folder), filesystem::directory_iterator{}), 1);
    ASSERT_NO_THROW(server.worker.get());

    // A completed local file is reusable even without an available remote.
    EXPECT_NO_THROW(download_if_missing(path, "http://127.0.0.1:0/unavailable"));
    EXPECT_EQ(read_text_file(path), "payload");
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

TEST(JsonTest, IntegerBoundaryRoundTrips)
{
    for (const long long value : {numeric_limits<long long>::min(),
                                 numeric_limits<long long>::max()})
    {
        EXPECT_EQ(Json(value).as_long(), value);
        EXPECT_EQ(Json::parse(Json(value).dump()).as_long(), value);
        EXPECT_EQ(Json(to_string(value)).as_long(), value);
    }
    EXPECT_EQ(Json(3.75).as_long(), 3);
    EXPECT_EQ(Json(-3.75).as_long(), -3);
}

TEST(JsonTest, IntegerConversionRejectsNonfiniteAndOutOfRangeNumbers)
{
    const double infinity = numeric_limits<double>::infinity();
    for (const double value : {infinity, -infinity, numeric_limits<double>::quiet_NaN(),
                              nextafter(double(numeric_limits<long long>::max()), infinity),
                              nextafter(double(numeric_limits<long long>::min()), -infinity)})
        EXPECT_THROW(Json(value).as_long(), runtime_error);
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

TEST(IoUtilitiesTest, ReadInt32BatchSupportsConcurrentMixedWidths)
{
    const filesystem::path tmp = make_temp_path("int32_batch.tmp");
    const filesystem::path path = make_temp_path("int32_batch.bin");
    remove_quietly(tmp);
    remove_quietly(path);

    constexpr Index rows = 5;
    constexpr Index columns = 6;
    vector<int32_t> records(size_t(rows * columns));
    for (Index row = 0; row < rows; ++row)
        for (Index column = 0; column < columns; ++column)
            records[size_t(row * columns + column)] = int32_t(row * 100 + column);

    FileWriter writer;
    writer.open(tmp);
    writer.write(span(records));
    writer.finish_with_rename(path);

    FileReader reader;
    reader.open(path);
    const vector<Index> samples = {4, 1, 3};

    const auto read_columns = [&](Index offset, Index count)
    {
        vector<float> output(size_t(ssize(samples) * count));
        read_int32_batch(reader, samples, rows, columns, offset, count,
                         output, count, 0, "IoUtilitiesTest");
        return output;
    };

    future<vector<float>> narrow = async(launch::async, read_columns, 1, 2);
    future<vector<float>> wide = async(launch::async, read_columns, 0, 5);

    const vector<float> narrow_values = narrow.get();
    const vector<float> wide_values = wide.get();
    for (Index row = 0; row < ssize(samples); ++row)
    {
        for (Index column = 0; column < 2; ++column)
            EXPECT_FLOAT_EQ(narrow_values[size_t(row * 2 + column)],
                            float(samples[size_t(row)] * 100 + column + 1));
        for (Index column = 0; column < 5; ++column)
            EXPECT_FLOAT_EQ(wide_values[size_t(row * 5 + column)],
                            float(samples[size_t(row)] * 100 + column));
    }

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
