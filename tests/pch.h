//
// pch.h
//

#ifndef PCH_TESTS_H
#define PCH_TESTS_H

#define NDEBUG
#define EIGEN_MAX_ALIGN_BYTES 32
#define EIGEN_NO_DEBUG
#define EIGEN_USE_THREADS

#include <algorithm>
#include <vector>
#include <iostream>

#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "gtest/gtest.h"

#endif
