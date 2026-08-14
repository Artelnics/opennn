//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T H R E A D   S A F E   Q U E U E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
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

    T pop()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty() || closed_; });
        if (queue_.empty()) return T{};
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
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
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    bool closed_ = false;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
