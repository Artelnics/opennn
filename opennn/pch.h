#ifndef PCH_H
#define PCH_H

#pragma once

#define NUMERIC_LIMITS_MIN type(0.000001)

#define NOMINMAX
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING
#define _CRT_SECURE_NO_WARNINGS
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include <algorithm>
#include <string>
#include <cassert>
#include <cmath>
#include <ctime>
#include <codecvt>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <map>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <exception>
#include <memory>
#include <random>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <stdlib.h>
#include <set>
#include <regex>
#include <sstream>

#include <omp.h>

#define EIGEN_USE_THREADS

#include "../eigen/Eigen/Core"
#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "../eigen/Eigen/src/Core/util/DisableStupidWarnings.h"

#define OPENNN_CUDA_test // Comment this line to disable cuda files

#ifdef OPENNN_CUDA_test

#include "../opennn/kernel.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>
#include <curand.h>
#include <cudnn.h>

#endif

//#include "tinyxml2.h"

using namespace std;
using namespace Eigen;
//using namespace tinyxml2;

using type = float;

using dimensions = vector<Index>;

template<typename Base, typename T>
inline bool is_instance_of(const T* ptr)
{
    return dynamic_cast<const Base*>(ptr);
}


#endif // PCH_H
