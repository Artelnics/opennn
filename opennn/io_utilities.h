//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I / O   U T I L I T I E S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "pch.h"

namespace opennn
{

/// @brief Thread-safe positional file reader (pread on POSIX, overlapped ReadFile on Windows).
// Thread-safe positional reader. On POSIX uses pread(); on Windows uses
// ReadFile with OVERLAPPED so the offset is per-call and not shared state.
class FileReader
{
public:
    FileReader() = default;
    ~FileReader();

    FileReader(const FileReader&)            = delete;
    FileReader& operator=(const FileReader&) = delete;
    FileReader(FileReader&&)                 = delete;
    FileReader& operator=(FileReader&&)      = delete;

    /// @brief Opens the file at the given path for reading.
    void open(const filesystem::path& path);

    /// @brief Closes the underlying file handle, if any.
    void close();

    /// @brief Returns true while the underlying handle is open.
    [[nodiscard]] bool is_open() const;

    /// @brief Reads bytes from a specific offset into the provided buffer (thread-safe).
    /// @param buffer Destination buffer (must hold at least bytes).
    /// @param bytes Number of bytes to read.
    /// @param offset Byte offset within the file to read from.
    void read_at(void* buffer, size_t bytes, uint64_t offset) const;

    /// @brief Returns the total size of the open file in bytes.
    [[nodiscard]] uint64_t file_size() const;

private:
#if defined(_WIN32)
    void* handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};


/// @brief Streaming writer that finalises by atomic-renaming a .tmp file to its final path.
// Streaming writer that lands on a .tmp file and renames atomically when
// finish_with_rename() succeeds. Drops the .tmp on destruction otherwise.
class FileWriter
{
public:
    FileWriter() = default;
    ~FileWriter();

    FileWriter(const FileWriter&)            = delete;
    FileWriter& operator=(const FileWriter&) = delete;
    FileWriter(FileWriter&&)                 = delete;
    FileWriter& operator=(FileWriter&&)      = delete;

    /// @brief Opens a temporary file to which subsequent write() calls are appended.
    void open(const filesystem::path& tmp_path);

    /// @brief Returns true while the underlying stream is open.
    [[nodiscard]] bool is_open() const;

    /// @brief Appends the given byte range to the open file.
    void write(const void* buffer, size_t bytes);

    /// @brief Closes the stream and atomically renames the tmp file to final_path.
    void finish_with_rename(const filesystem::path& final_path);

    /// @brief Closes and deletes the tmp file, discarding any written data.
    void abort();

private:
    filesystem::path tmp_path_;
    ofstream stream_;
    bool finalized_ = false;
};


/// @brief Atomically renames a file, replacing the destination if needed.
void atomic_rename(const filesystem::path& from, const filesystem::path& to);


/// @brief Tokenising CSV reader that returns string_views into a single backing buffer.
// CSV reader. Loads a file (or string) into a single buffer and tokenizes
// each non-empty line into a vector of string_views into that buffer.
// The buffer lives in Result, so the views stay valid as long as Result does.
// Handles UTF-8 BOM, quoted fields (the separator inside quotes is ignored),
// and CRLF line endings.
class CsvReader
{
public:

    /// @brief Reader configuration: field separator and an optional per-line validator.
    struct Config
    {
        char separator = ',';
        function<void(string_view)> line_validator;
    };

    /// @brief Parsed CSV result; owns the source buffer that backs all row views.
    struct Result
    {
        string                      buffer;
        vector<vector<string_view>> rows;
    };

    /// @brief Constructs a reader with the given configuration.
    explicit CsvReader(Config c) : config(std::move(c)) {}

    /// @brief Reads and parses the CSV file at the given path.
    [[nodiscard]] Result read(const filesystem::path& path) const;

    /// @brief Parses a CSV string already held in memory.
    [[nodiscard]] Result read_string(string_view csv_text) const;

private:

    Config config;

    void parse(Result& out) const;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
