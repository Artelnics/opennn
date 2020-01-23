#include "omp.h"

#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif

#include <../eigen/unsupported/Eigen/CXX11/Tensor>

#include <../../eigen/unsupported/Eigen/CXX11/ThreadPool>

using namespace Eigen;

class Device
{
    public:

        enum device_type{SimpleThreadPoolType, NonBlockingThreadPoolType};



    private:


        // Private Constructor

        Device();

        // Stop the compiler generating methods of copy the object

        Device(Device const& copy);            // Not Implemented

        Device& operator=(Device const& copy); // Not Implemented

        NonBlockingThreadPool* non_blocking_thread_pool;

        ThreadPoolDevice* thread_pool_device;


        static Device& getInstance()
        {
            // The only instance
            // Guaranteed to be lazy initialized
            // Guaranteed that it will be destroyed correctly
            static Device instance;
            return instance;
        }
/*
        static set_device(const string&)
        {
            non_blocking_thread_pool = new NonBlockingThreadPool(1);
        }
*/
};

