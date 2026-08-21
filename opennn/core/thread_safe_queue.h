//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T H R E A D   S A F E   Q U E U E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <utility>

namespace opennn
{

template <typename T>
class ThreadSafeQueue
{
public:

    void push(T item)
    {
        { std::lock_guard<std::mutex> lock(mutex_); queue_.push(std::move(item)); }
        cond_.notify_one();
    }

    bool wait_pop(T& item)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty() || closed_; });
        if (queue_.empty()) return false;

        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    void close()
    {
        { std::lock_guard<std::mutex> lock(mutex_); closed_ = true; }
        cond_.notify_all();
    }

    void reopen()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = false;
    }

private:

    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    bool closed_ = false;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
