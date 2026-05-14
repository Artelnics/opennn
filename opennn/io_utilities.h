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

    void open(const filesystem::path& tmp_path);
    bool is_open() const;

    void write(const void* buffer, size_t bytes);

    void finish_with_rename(const filesystem::path& final_path);

    void abort();

private:
    filesystem::path tmp_path_;
    std::ofstream stream_;
    bool finalized_ = false;
};


void atomic_rename(const filesystem::path& from, const filesystem::path& to);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
