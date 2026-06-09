//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

class FileReader
{
public:
    FileReader() = default;
    ~FileReader();

    FileReader(const FileReader&)            = delete;
    FileReader& operator=(const FileReader&) = delete;
    FileReader(FileReader&&)                 = delete;
    FileReader& operator=(FileReader&&)      = delete;

    void open(const filesystem::path& path);
    void close();
    bool is_open() const;

    void read_at(void* buffer, size_t bytes, uint64_t offset) const;

    uint64_t file_size() const;

private:
#if defined(_WIN32)
    void* handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

class FileWriter
{
public:
    FileWriter() = default;
    ~FileWriter();

    FileWriter(const FileWriter&)            = delete;
    FileWriter& operator=(const FileWriter&) = delete;
    FileWriter(FileWriter&&)                 = delete;
    FileWriter& operator=(FileWriter&&)      = delete;

    void open(const filesystem::path& tmp_path);
    bool is_open() const;

    void write(const void* buffer, size_t bytes);

    void finish_with_rename(const filesystem::path& final_path);

private:
    filesystem::path tmp_path_;
    ofstream stream_;
    bool finalized_ = false;
};


void atomic_rename(const filesystem::path& from, const filesystem::path& to);

class FileMapping
{
public:
    FileMapping() = default;
    ~FileMapping();

    FileMapping(const FileMapping&)            = delete;
    FileMapping& operator=(const FileMapping&) = delete;
    FileMapping(FileMapping&&) noexcept;
    FileMapping& operator=(FileMapping&&) noexcept;

    bool map(const filesystem::path& path);
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

class CsvReader
{
public:

    struct Config
    {
        char separator = ',';
        function<void(string_view)> line_validator;
    };

    struct Result
    {
        FileMapping         mapping;
        string              buffer;
        string_view         content;
        vector<string_view> lines;
        char                separator = ',';
    };

    explicit CsvReader(Config c) : config(std::move(c)) {}

    Result read(const filesystem::path& path) const;

private:

    Config config;

    void parse(Result& out) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
