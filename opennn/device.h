#include "omp.h"

#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif

#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "../eigen/unsupported/Eigen/CXX11/ThreadPool"

using namespace Eigen;

class Device
{
    public:

        enum device_type{SimpleThreadPoolType, NonBlockingThreadPoolType};

        explicit Device() {}

        virtual ~Device() {}

    private:

        SimpleThreadPool* simple_thread_pool;
        NonBlockingThreadPool* non_blocking_thread_pool;

        ThreadPoolDevice* thread_pool_device;
/*
        static set_device(const string&)
        {
            non_blocking_thread_pool = new NonBlockingThreadPool(1);
        }
*/
};

