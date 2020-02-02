
#ifndef DEVICE_H
#define DEVICE_H


#include "config.h"

#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "../eigen/unsupported/Eigen/CXX11/ThreadPool"

using namespace Eigen;

namespace OpenNN {

class Device
{
    public:

        enum Type{EigenDefault, EigenSimpleThreadPool, EigenGpu, IntelMkl};

        explicit Device() {}

        explicit Device(const Type& new_type)
        {
            set_type(new_type);
        }

        virtual ~Device() {}

        Type get_type() const {return type;}

        void set_type(const Type& new_type)
        {
            type = new_type;

            switch(type)
            {
                case EigenDefault:

                    default_device = new DefaultDevice;

                break;

                case EigenSimpleThreadPool:

                    //cint n = omp_get_max_threads();

                    simple_thread_pool = new SimpleThreadPool(16);

                    thread_pool_device = new ThreadPoolDevice(simple_thread_pool, 16);

                break;

                case EigenGpu:

                    //gpu_device = new GpuDevice();

                break;

                default:

                break;

            }
        }

        DefaultDevice* get_eigen_default_device() const {return default_device;}

        ThreadPoolDevice* get_eigen_thread_pool_device() const {return thread_pool_device;}

        GpuDevice* get_eigen_gpu_device() const {return gpu_device;}


    private:

        Type type = EigenSimpleThreadPool;

        DefaultDevice* default_device = nullptr;

        SimpleThreadPool* simple_thread_pool = nullptr;
        NonBlockingThreadPool* non_blocking_thread_pool = nullptr;

        ThreadPoolDevice* thread_pool_device = nullptr;

        //CudaStreamDevice cuda_stream_device;

        GpuDevice* gpu_device = nullptr;

};

}


#endif

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
