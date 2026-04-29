//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T H R E A D   S A F E   Q U E U E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

// Bounded-blocking-style concurrent queue used by the optimizer family
// (background batch prefetch, async data loaders). Standard mutex + condvar
// pattern; nothing tensor-specific.

#include <queue>
#include <mutex>
#include <condition_variable>

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
        cond_.wait(lock, [this] { return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:

    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
