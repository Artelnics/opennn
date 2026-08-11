//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/io_utilities.h"
#include "opennn/core/string_utilities.h"

#include <cstdlib>
#include <iostream>

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
    #include <sys/mman.h>
    #include <cerrno>
#endif

namespace opennn
{

void download_if_missing(const filesystem::path& path, const string& url)
{
    if (filesystem::exists(path)) return;

    if (path.has_parent_path())
        filesystem::create_directories(path.parent_path());

    cout << "Downloading " << url << " -> " << path.string() << " ..." << endl;

#if defined(_WIN32)
    const string curl = "curl.exe";
#else
    const string curl = "curl";
#endif

    const string command =
        curl + " -L --fail -o \"" + path.string() + "\" \"" + url + "\"";

    if (system(command.c_str()) != 0 || !filesystem::exists(path))
        throw runtime_error("Download failed. Get it manually from:\n  " + url);
}

void download_files_if_missing(const filesystem::path& directory,
                               const string_view base_url,
                               const vector<string_view>& filenames)
{
    for (const string_view filename : filenames)
        download_if_missing(directory / filename, string(base_url) + string(filename));
}

string read_text_file(const filesystem::path& path)
{
    ifstream file(path, ios::binary | ios::ate);

    throw_if(!file.is_open(), "Cannot open file {}", path.string());

    const streamoff byte_count = file.tellg();
    throw_if(byte_count < 0, "Cannot determine size for file {}", path.string());
    throw_if(byte_count > streamoff(numeric_limits<streamsize>::max()),
             "File {} is too large to read into memory", path.string());

    file.seekg(0);

    string contents(size_t(byte_count), '\0');
    if (byte_count > 0)
        file.read(contents.data(), streamsize(byte_count));

    throw_if(!file, "Cannot read file {}", path.string());
    return contents;
}

namespace
{

template <typename Kind>
vector<filesystem::path> list_entries(const filesystem::path& directory,
                                      Kind is_wanted_kind,
                                      bool (*predicate)(const filesystem::path&))
{
    vector<filesystem::path> paths;

    for (const filesystem::directory_entry& entry : filesystem::directory_iterator(directory))
        if (is_wanted_kind(entry) && predicate(entry.path()))
            paths.push_back(entry.path());

    ranges::sort(paths);
    return paths;
}

}

vector<filesystem::path> list_files(const filesystem::path& directory,
                                    bool (*predicate)(const filesystem::path&))
{
    return list_entries(directory,
                        [](const filesystem::directory_entry& e) { return e.is_regular_file(); },
                        predicate);
}

vector<filesystem::path> list_directories(const filesystem::path& directory,
                                          bool (*predicate)(const filesystem::path&))
{
    return list_entries(directory,
                        [](const filesystem::directory_entry& e) { return e.is_directory(); },
                        predicate);
}

bool is_file_current(const filesystem::path& file,
                     const vector<filesystem::path>& sources,
                     const uintmax_t expected_size)
{
    error_code error;

    if (!filesystem::exists(file, error) || error) return false;

    if (expected_size > 0
     && (filesystem::file_size(file, error) != expected_size || error))
        return false;

    const auto file_time = filesystem::last_write_time(file, error);
    if (error) return false;

    for (const filesystem::path& source : sources)
    {
        const auto source_time = filesystem::last_write_time(source, error);
        if (error || file_time < source_time) return false;
    }

    return true;
}

FileMapping::~FileMapping() { reset(); }

FileMapping::FileMapping(FileMapping&& other) noexcept { *this = move(other); }

FileMapping& FileMapping::operator=(FileMapping&& other) noexcept
{
    if (this != &other)
    {
        reset();
        data_ = other.data_;
        size_ = other.size_;
#if defined(_WIN32)
        file_handle_    = other.file_handle_;
        mapping_handle_ = other.mapping_handle_;
        other.file_handle_    = nullptr;
        other.mapping_handle_ = nullptr;
#else
        fd_ = other.fd_;
        other.fd_ = -1;
#endif
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

#if defined(_WIN32)

bool FileMapping::map(const filesystem::path& path)
{
    reset();

    file_handle_ = ::CreateFileW(path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ,
                                 nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file_handle_ == INVALID_HANDLE_VALUE) { file_handle_ = nullptr; return false; }

    LARGE_INTEGER file_size;
    if (!::GetFileSizeEx(file_handle_, &file_size) || file_size.QuadPart == 0)
    {
        reset();
        return false;
    }

    mapping_handle_ = ::CreateFileMappingW(file_handle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapping_handle_) { reset(); return false; }

    void* const view = ::MapViewOfFile(mapping_handle_, FILE_MAP_READ, 0, 0, 0);
    if (!view) { reset(); return false; }

    data_ = static_cast<const char*>(view);
    size_ = static_cast<size_t>(file_size.QuadPart);
    return true;
}

void FileMapping::reset()
{
    if (data_) { ::UnmapViewOfFile(const_cast<char*>(data_)); data_ = nullptr; }
    if (mapping_handle_) { ::CloseHandle(mapping_handle_); mapping_handle_ = nullptr; }
    if (file_handle_ && file_handle_ != INVALID_HANDLE_VALUE) { ::CloseHandle(file_handle_); }
    file_handle_ = nullptr;
    size_ = 0;
}

#else

bool FileMapping::map(const filesystem::path& path)
{
    reset();

    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) { fd_ = -1; return false; }

    struct stat st;
    if (::fstat(fd_, &st) != 0 || st.st_size == 0) { reset(); return false; }

    void* const addr = ::mmap(nullptr, size_t(st.st_size), PROT_READ, MAP_PRIVATE, fd_, 0);
    if (addr == MAP_FAILED) { reset(); return false; }

    data_ = static_cast<const char*>(addr);
    size_ = size_t(st.st_size);
    return true;
}

void FileMapping::reset()
{
    if (data_) { ::munmap(const_cast<char*>(data_), size_); data_ = nullptr; }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    size_ = 0;
}

#endif

FileReader::~FileReader() { close(); }

void FileReader::close()
{
#if defined(_WIN32)
    if (handle_ && handle_ != INVALID_HANDLE_VALUE)
    {
        ::CloseHandle(handle_);
        handle_ = nullptr;
    }
#else
    if (fd_ >= 0)
    {
        ::close(fd_);
        fd_ = -1;
    }
#endif
}

bool FileReader::is_open() const
{
#if defined(_WIN32)
    return handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE;
#else
    return fd_ >= 0;
#endif
}

#if defined(_WIN32)

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
        throw runtime_error(format("FileReader: cannot open {}", path.string()));
    }
}

void FileReader::read_at(const span<std::byte> buffer, const uint64_t offset) const
{
    throw_if(!is_open(), "FileReader::read_at: file not open.");

    const size_t bytes = buffer.size_bytes();
    size_t total = 0;
    std::byte* const dst = buffer.data();
    while (total < bytes)
    {
        const uint64_t current_offset = offset + total;
        const DWORD bytes_to_read = DWORD(min(bytes - total, size_t(0x7FFFFFFF)));

        OVERLAPPED overlapped{};
        overlapped.Offset     = DWORD(current_offset);
        overlapped.OffsetHigh = DWORD(current_offset >> 32);

        DWORD read_bytes = 0;
        const BOOL ok = ::ReadFile(handle_, dst + total, bytes_to_read, &read_bytes, &overlapped);
        throw_if(!ok || read_bytes == 0,
                 "FileReader::read_at: ReadFile failed (offset={}, n={}).",
                        current_offset, bytes_to_read);
        total += read_bytes;
    }
}

#else

void FileReader::open(const filesystem::path& path)
{
    close();

    fd_ = ::open(path.c_str(), O_RDONLY);
    throw_if(fd_ < 0,
             "FileReader: cannot open {} (errno={}).",
                    path.string(), errno);
}

void FileReader::read_at(const span<std::byte> buffer, const uint64_t offset) const
{
    throw_if(!is_open(), "FileReader::read_at: file not open.");

    const size_t bytes = buffer.size_bytes();
    size_t total = 0;
    std::byte* const dst = buffer.data();
    while (total < bytes)
    {
        const ssize_t n = ::pread(fd_, dst + total, bytes - total, off_t(offset + total));
        if (n < 0)
        {
            if (errno == EINTR) continue;
            throw runtime_error(format("FileReader::read_at: pread failed (errno={}, offset={}).",
                                       errno, offset + total));
        }
        throw_if(n == 0,
                 "FileReader::read_at: unexpected EOF at offset {}.",
                        offset + total);
        total += size_t(n);
    }
}

#endif

uint64_t FileReader::file_size() const
{
    throw_if(!is_open(), "FileReader::file_size: file not open.");
#if defined(_WIN32)
    LARGE_INTEGER size{};
    throw_if(!::GetFileSizeEx(handle_, &size),
             "FileReader::file_size: GetFileSizeEx failed.");
    return uint64_t(size.QuadPart);
#else
    struct stat st{};
    throw_if(::fstat(fd_, &st) != 0,
             "FileReader::file_size: fstat failed.");
    return uint64_t(st.st_size);
#endif
}

void read_int32_batch(const FileReader& reader,
                      const vector<Index>& sample_indices,
                      const Index samples_number,
                      const uint64_t record_values,
                      const Index source_offset,
                      const Index values_number,
                      const span<float> output,
                      const Index output_stride,
                      const Index output_offset,
                      const string_view context)
{
    throw_if(record_values == 0,
             "{} record width must be greater than zero.", context);
    throw_if(source_offset < 0 || values_number < 0
          || uint64_t(source_offset) + uint64_t(values_number) > record_values,
             "{} record range is invalid.", context);
    throw_if(output_offset < 0 || output_stride < 0
          || uint64_t(output_stride) < uint64_t(output_offset) + uint64_t(values_number),
             "{} output range is invalid.", context);
    if (sample_indices.empty()) return;

    const Index required_size = (ssize(sample_indices) - 1) * output_stride
                              + output_offset + values_number;

    throw_if(ssize(output) < required_size,
             "{} output buffer holds {} values but {} are required.",
             context, ssize(output), required_size);

    float* const output_data = output.data();

    string omp_error;

    #pragma omp parallel for
    for (Index i = 0; i < ssize(sample_indices); ++i)
    {
        try
        {
            const Index sample_index = sample_indices[size_t(i)];
            throw_if(sample_index < 0 || sample_index >= samples_number,
                     "{} sample index is out of range.", context);

            thread_local vector<int32_t> buffer;
            buffer.resize(size_t(values_number));

            reader.read_at(span(buffer),
                           (uint64_t(sample_index) * record_values
                            + uint64_t(source_offset)) * sizeof(int32_t));

            for (Index j = 0; j < values_number; ++j)
                output_data[i * output_stride + output_offset + j] = float(buffer[size_t(j)]);
        }
        catch (const exception& exception)
        {
            #pragma omp critical
            {
                if (omp_error.empty()) omp_error = exception.what();
            }
        }
    }

    throw_if(!omp_error.empty(), omp_error);
}

FileWriter::~FileWriter()
{
    if (stream_.is_open()) stream_.close();
    if (!tmp_path_.empty())
    {
        error_code ec;
        filesystem::remove(tmp_path_, ec);
    }
}

void FileWriter::open(const filesystem::path& tmp_path)
{
    tmp_path_ = tmp_path;

    filesystem::create_directories(tmp_path.parent_path());

    stream_.open(tmp_path, ios::binary | ios::trunc);
    throw_if(!stream_.is_open(),
             "FileWriter: cannot open {}", tmp_path.string());
}

void FileWriter::write(const span<const std::byte> buffer)
{
    throw_if(!stream_.is_open(), "FileWriter::write: not open.");
    if (buffer.empty()) return;
    stream_.write(reinterpret_cast<const char*>(buffer.data()), streamsize(buffer.size_bytes()));
    throw_if(!stream_.good(), "FileWriter::write: stream error.");
}

void FileWriter::finish_with_rename(const filesystem::path& final_path)
{
    throw_if(!stream_.is_open(), "FileWriter::finish: not open.");
    stream_.flush();
    stream_.close();
    throw_if(stream_.fail(), "FileWriter::finish: flush/close failed.");

#if defined(_WIN32)
    if (!::MoveFileExW(tmp_path_.wstring().c_str(),
                       final_path.wstring().c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw runtime_error(format("FileWriter::finish_with_rename: MoveFileExW failed for {} -> {} (GetLastError={}).",
                                   tmp_path_.string(), final_path.string(), ::GetLastError()));
#else
    throw_if(::rename(tmp_path_.c_str(), final_path.c_str()) != 0,
             "FileWriter::finish_with_rename: rename failed for {} -> {} (errno={}).",
             tmp_path_.string(), final_path.string(), errno);
#endif

    tmp_path_.clear();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
