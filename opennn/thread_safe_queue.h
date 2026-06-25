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

namespace opennn
{

template <typename T>
class ThreadSafeQueue
{
public:

    void push(T)
    {
        { lock_guard<mutex> lock(mutex_); queue_.push(move(item)); }
        cond_.notify_one();
    }

    T pop()
    {
        unique_lock<mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty() || closed_; });
        if (queue_.empty()) return T{};
        T item = move(queue_.front());
        queue_.pop();
        return item;
    }

    bool empty() const
    {
        lock_guard<mutex> lock(mutex_);
        return queue_.empty();
    }

    void close()
    {
        { lock_guard<mutex> lock(mutex_); closed_ = true; }
        cond_.notify_all();
    }

    void reopen()
    {
        lock_guard<mutex> lock(mutex_);
        closed_ = false;
    }

private:

    queue<T> queue_;
    mutable mutex mutex_;
    condition_variable cond_;
    bool closed_ = false;
};

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
