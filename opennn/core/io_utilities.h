//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "opennn/core/opennn_types.h"

#include <type_traits>
#include <functional>

namespace opennn
{

void download_if_missing(const filesystem::path&, const string&);
void download_files_if_missing(const filesystem::path& directory,
                               string_view base_url,
                               const vector<string_view>& filenames);

string read_text_file(const filesystem::path&);

vector<filesystem::path> list_files(const filesystem::path& directory,
                                    std::function<bool(const filesystem::path&)> predicate);
vector<filesystem::path> list_directories(const filesystem::path& directory,
                                          std::function<bool(const filesystem::path&)> predicate);
bool is_file_current(const filesystem::path& file,
                     const vector<filesystem::path>& sources,
                     uintmax_t expected_size = 0);

class FileReader
{
public:
    FileReader() = default;
    ~FileReader();

    FileReader(const FileReader&)            = delete;
    FileReader& operator=(const FileReader&) = delete;
    FileReader(FileReader&&)                 = delete;
    FileReader& operator=(FileReader&&)      = delete;

    void open(const filesystem::path&);
    void close();
    bool is_open() const;

    void read_at(span<std::byte>, uint64_t offset) const;

    template <typename T, size_t Extent>
        requires (!is_const_v<T> && is_trivially_copyable_v<T>)
    void read_at(const span<T, Extent> buffer, const uint64_t offset) const
    {
        read_at(span<std::byte>(as_writable_bytes(buffer)), offset);
    }

    uint64_t file_size() const;

private:
#if defined(_WIN32)
    void* handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

void read_int32_batch(const FileReader&,
                      const vector<Index>& sample_indices,
                      Index samples_number,
                      uint64_t record_values,
                      Index source_offset,
                      Index values_number,
                      span<float> output,
                      Index output_stride,
                      Index output_offset,
                      string_view context);

class FileWriter
{
public:
    FileWriter() = default;
    ~FileWriter();

    FileWriter(const FileWriter&)            = delete;
    FileWriter& operator=(const FileWriter&) = delete;
    FileWriter(FileWriter&&)                 = delete;
    FileWriter& operator=(FileWriter&&)      = delete;

    void open(const filesystem::path&);

    void write(span<const std::byte>);

    template <typename T, size_t Extent>
        requires is_trivially_copyable_v<remove_const_t<T>>
    void write(const span<T, Extent> buffer)
    {
        write(span<const std::byte>(as_bytes(buffer)));
    }

    void finish_with_rename(const filesystem::path&);

private:
    filesystem::path tmp_path_;
    ofstream stream_;
};

template <typename T>
concept RawStorable = is_trivially_copyable_v<T>;

bool read_binary_value(istream& stream, RawStorable auto& value)
{
    return bool(stream.read(reinterpret_cast<char*>(&value), sizeof(value)));
}

bool read_binary_values(istream& stream, RawStorable auto&... values)
{
    return (read_binary_value(stream, values) && ...);
}

void write_binary_value(FileWriter& writer, const RawStorable auto& value)
{
    writer.write(span(&value, 1));
}

inline bool read_binary_string(istream& stream, string& value)
{
    uint64_t size = 0;
    if (!read_binary_value(stream, size)
        || size > uint64_t(numeric_limits<streamsize>::max()))
        return false;

    value.resize(size_t(size));
    return size == 0 || bool(stream.read(value.data(), streamsize(size)));
}

inline void write_binary_string(FileWriter& writer, const string& value)
{
    write_binary_value(writer, uint64_t(value.size()));
    writer.write(span(value));
}

class FileMapping
{
public:
    FileMapping() = default;
    ~FileMapping();

    FileMapping(const FileMapping&)            = delete;
    FileMapping& operator=(const FileMapping&) = delete;
    FileMapping(FileMapping&&) noexcept;
    FileMapping& operator=(FileMapping&&) noexcept;

    bool map(const filesystem::path&);
    void reset();

    const char* data() const { return data_; }
    size_t      size() const { return size_; }

private:
    const char* data_ = nullptr;
    size_t      size_ = 0;
#if defined(_WIN32)
    void* file_handle_    = nullptr;
    void* mapping_handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
