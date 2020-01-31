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

        enum Type{EigenDefault, EigenThreadPool, EigenGpu, IntelMkl};

        explicit Device() {}

        virtual ~Device() {}

        Type get_type() const {return type;}

        DefaultDevice* get_eigen_default_device() const {return default_device;}

        ThreadPoolDevice* get_eigen_thread_pool_device() const {return thread_pool_device;}

        GpuDevice* get_eigen_gpu_device() const {return gpu_device;}


    private:

        Type type = EigenThreadPool;

        DefaultDevice* default_device = nullptr;

        SimpleThreadPool* simple_thread_pool = nullptr;
        NonBlockingThreadPool* non_blocking_thread_pool = nullptr;

        ThreadPoolDevice* thread_pool_device = nullptr;

        GpuDevice* gpu_device = nullptr;



};

