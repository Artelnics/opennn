<div align="center">
  <img src="http://www.opennn.net/images/opennn_git_logo.svg">
</div>

[![Build Status](https://travis-ci.org/{ORG-or-USERNAME}/{REPO-NAME}.png?branch=master)](https://travis-ci.org/Artelnics/opennn)

# What is OpenNN ?

OpenNN is a software library written in C++ for advanced analytics. It implements neural networks, the most successful machine learning method.

The main advantage of OpenNN is its high performance.

This library outstands in terms of execution speed and memory allocation. It is constantly optimized and parallelized in order to maximize its efficiency.

Some typical applications of OpenNN are business intelligence (customer segmentation, churn prevention...), health care (early diagnosis, microarray analysis,...) and engineering (performance optimization, predictive maitenance...).

The documentation is composed by tutorials and examples to offer a complete overview about the library.

The documentation can be found at the official <a href="http://opennn.net" target="_blank">OpenNN site</a>.

# How to build ?

CMakeLists.txt are build files for CMake.

```
git clone git@github.com:Artelnics/opennn.git
cd opennn
mkdir build
cd build
cmake ..
make
```

# How to install

Once the project is build, you can run the following line to install the project (possibly in sudo)

```
make install
```

You can also uninstall the project if you dont require it any more

```
make uninstall
```

# Perform unit testing

You also have to possibility to do unit testing of the library. To do so, activate the option to build test in the top level cmake list. Then, from the build directory perform the following operations:

```
cmake ..
make tests
cd tests
./tests
```

# Notes

The .pro files are project files for the Qt Creator IDE, which can be downloaded from its <a href="http://www.qt.io" target="_blank">site</a>. Note that OpenNN does not make use of the Qt library.



OpenNN is developed by <a href="http://artelnics.com" target="_blank">Artelnics</a>, a company specialized in artificial intelligence.
