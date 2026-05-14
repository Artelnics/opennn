//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I / O   U T I L I T I E S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "io_utilities.h"

#if defined(_WIN32)
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <unistd.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <cerrno>
#endif

namespace opennn
{

#if defined(_WIN32)

FileReader::~FileReader() { close(); }

void FileReader::open(const filesystem::path& path)
{
    close();

    handle_ = ::CreateFileW(path.wstring().c_str(),
                            GENERIC_READ,
                            FILE_SHARE_READ,
                            nullptr,
                            OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL,
                            nullptr);

    if (handle_ == INVALID_HANDLE_VALUE)
    {
        handle_ = nullptr;
        throw runtime_error("FileReader: cannot open " + path.string());
    }
}

void FileReader::close()
{
    if (handle_ && handle_ != INVALID_HANDLE_VALUE)
    {
        ::CloseHandle(handle_);
        handle_ = nullptr;
    }
}

bool FileReader::is_open() const
{
    return handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE;
}

void FileReader::read_at(void* buffer, size_t bytes, uint64_t offset) const
{
    if (!is_open()) throw runtime_error("FileReader::read_at: file not open.");

    size_t total = 0;
    auto* dst = static_cast<uint8_t*>(buffer);
    while (total < bytes)
    {
        const size_t remaining = bytes - total;
        const DWORD chunk = (remaining > 0x7FFFFFFFULL) ? 0x7FFFFFFFu : DWORD(remaining);

        OVERLAPPED ov{};
        const uint64_t current = offset + total;
        ov.Offset     = DWORD(current & 0xFFFFFFFFULL);
        ov.OffsetHigh = DWORD((current >> 32) & 0xFFFFFFFFULL);

        DWORD read_bytes = 0;
        const BOOL ok = ::ReadFile(handle_, dst + total, chunk, &read_bytes, &ov);
        if (!ok || read_bytes == 0)
            throw runtime_error("FileReader::read_at: ReadFile failed (offset="
                                + to_string(current) + ", n=" + to_string(chunk) + ").");
        total += read_bytes;
    }
}

uint64_t FileReader::file_size() const
{
    if (!is_open()) throw runtime_error("FileReader::file_size: file not open.");
    LARGE_INTEGER size{};
    if (!::GetFileSizeEx(handle_, &size))
        throw runtime_error("FileReader::file_size: GetFileSizeEx failed.");
    return uint64_t(size.QuadPart);
}

#else  // POSIX

FileReader::~FileReader() { close(); }

void FileReader::open(const filesystem::path& path)
{
    close();

    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
        throw runtime_error("FileReader: cannot open " + path.string()
                            + " (errno=" + to_string(errno) + ").");
}

void FileReader::close()
{
    if (fd_ >= 0)
    {
        ::close(fd_);
        fd_ = -1;
    }
}

bool FileReader::is_open() const { return fd_ >= 0; }

void FileReader::read_at(void* buffer, size_t bytes, uint64_t offset) const
{
    if (!is_open()) throw runtime_error("FileReader::read_at: file not open.");

    size_t total = 0;
    auto* dst = static_cast<uint8_t*>(buffer);
    while (total < bytes)
    {
        const ssize_t n = ::pread(fd_, dst + total, bytes - total, off_t(offset + total));
        if (n < 0)
        {
            if (errno == EINTR) continue;
            throw runtime_error("FileReader::read_at: pread failed (errno="
                                + to_string(errno) + ", offset="
                                + to_string(offset + total) + ").");
        }
        if (n == 0)
            throw runtime_error("FileReader::read_at: unexpected EOF at offset "
                                + to_string(offset + total) + ".");
        total += size_t(n);
    }
}

uint64_t FileReader::file_size() const
{
    if (!is_open()) throw runtime_error("FileReader::file_size: file not open.");
    struct stat st{};
    if (::fstat(fd_, &st) != 0)
        throw runtime_error("FileReader::file_size: fstat failed.");
    return uint64_t(st.st_size);
}

#endif


FileWriter::~FileWriter()
{
    if (stream_.is_open()) stream_.close();
    if (!finalized_ && !tmp_path_.empty())
    {
        std::error_code ec;
        filesystem::remove(tmp_path_, ec);
    }
}

void FileWriter::open(const filesystem::path& tmp_path)
{
    tmp_path_ = tmp_path;
    finalized_ = false;

    filesystem::create_directories(tmp_path.parent_path());

    stream_.open(tmp_path, ios::binary | ios::trunc);
    if (!stream_.is_open())
        throw runtime_error("FileWriter: cannot open " + tmp_path.string());
}

bool FileWriter::is_open() const { return stream_.is_open(); }

void FileWriter::write(const void* buffer, size_t bytes)
{
    if (!stream_.is_open()) throw runtime_error("FileWriter::write: not open.");
    stream_.write(reinterpret_cast<const char*>(buffer), std::streamsize(bytes));
    if (!stream_.good()) throw runtime_error("FileWriter::write: stream error.");
}

void FileWriter::finish_with_rename(const filesystem::path& final_path)
{
    if (!stream_.is_open()) throw runtime_error("FileWriter::finish: not open.");
    stream_.flush();
    stream_.close();
    if (!stream_.good() && stream_.eof() == false && stream_.fail())
        throw runtime_error("FileWriter::finish: flush/close failed.");

    atomic_rename(tmp_path_, final_path);
    finalized_ = true;
    tmp_path_.clear();
}

void FileWriter::abort()
{
    if (stream_.is_open()) stream_.close();
    if (!tmp_path_.empty())
    {
        std::error_code ec;
        filesystem::remove(tmp_path_, ec);
        tmp_path_.clear();
    }
    finalized_ = true;
}


void atomic_rename(const filesystem::path& from, const filesystem::path& to)
{
#if defined(_WIN32)
    if (!::MoveFileExW(from.wstring().c_str(),
                       to.wstring().c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw runtime_error("atomic_rename: MoveFileExW failed for " + from.string()
                            + " -> " + to.string());
#else
    if (::rename(from.c_str(), to.c_str()) != 0)
        throw runtime_error("atomic_rename: rename failed for " + from.string()
                            + " -> " + to.string() + " (errno=" + to_string(errno) + ").");
#endif
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
