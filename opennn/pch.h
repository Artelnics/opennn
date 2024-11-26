#ifndef PCH_H
#define PCH_H

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <exception>
#include <memory>
#include <functional>
#include <cmath>
#include <random>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <codecvt>
#include <fstream>
#include <stdexcept>
#include <stdlib.h>
#include <set>
#include <regex>
#include <sstream>
#include <iomanip>

using namespace std;

#define EIGEN_USE_THREADS

#include "../eigen/Eigen/Core"
#include "../eigen/unsupported/Eigen/CXX11/Tensor"
#include "../eigen/Eigen/src/Core/util/DisableStupidWarnings.h"

using namespace Eigen;

using type = float;

using dimensions = vector<Index>;


#endif // PCH_H
