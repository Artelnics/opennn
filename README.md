<div align="center">
  <img src="http://www.opennn.net/images/opennn_git_logo.svg">
</div>

[![Build Status](https://travis-ci.org/{ORG-or-USERNAME}/{REPO-NAME}.png?branch=master)](https://travis-ci.org/Artelnics/opennn)

OpenNN is a software library written in C++ for advanced analytics. It implements neural networks, the most successful machine learning method. 

The main advantage of OpenNN is its high performance. 

This library outstands in terms of execution speed and memory allocation. It is constantly optimized and parallelized in order to maximize its efficiency.

Some typical applications of OpenNN are business intelligence (customer segmentation, churn prevention...), health care (early diagnosis, microarray analysis,...) and engineering (performance optimization, predictive maitenance...).

The documentation is composed by tutorials and examples to offer a complete overview about the library. 

The documentation can be found at the official <a href="http://opennn.net" target="_blank">OpenNN site</a>.

OpenNN is built with CMake. Qt Creator, CLion, and Visual Studio all open the top-level `CMakeLists.txt` directly. Note that OpenNN does not make use of the Qt library.

## Prerequisites

- A C++20 compiler (MSVC 2019+, GCC 11+, or Clang 14+).
- CMake >= 3.18.
- For CUDA builds: NVIDIA CUDA Toolkit and cuDNN >= 9.0 installed on the system (via the NVIDIA installer, `apt install nvidia-cudnn`, or `vcpkg install cudnn`).
- Eigen, googletest, and cudnn-frontend are fetched automatically on first `cmake` configure.

OpenNN is developed by <a href="http://artelnics.com" target="_blank">Artelnics</a>, a company specialized in artificial intelligence. 
