#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   O P E N N N   E X A M P L E S
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com


# Project Configuration
TEMPLATE = app
CONFIG += console c++17


# Project Name
TARGET = run_tests


TEMPLATE = app


DESTDIR = "$$PWD/bin"


GTEST_DIR = ../googletest


SOURCES += $$GTEST_DIR/src/gtest-all.cc
INCLUDEPATH += $$GTEST_DIR/include
INCLUDEPATH += $$GTEST_DIR


SOURCES += test.cpp \
           adaptive_moment_estimation_test.cpp \
           bounding_layer_test.cpp \
           conjugate_gradient_test.cpp \
           convolutional_layer_test.cpp \
           correlations_test.cpp \
           cross_entropy_error_3d_test.cpp \
           cross_entropy_error_test.cpp \
           data_set_test.cpp \
           flatten_layer_test.cpp \
           genetic_algorithm_test.cpp \
           growing_inputs_test.cpp \
           growing_neurons_test.cpp \
           image_data_set_test.cpp \
           inputs_selection_test.cpp \
           learning_rate_algorithm_test.cpp \
           levenberg_marquardt_algorithm_test.cpp \
           long_short_term_memory_layer_test.cpp \
           mean_squared_error_test.cpp \
           minkowski_error_test.cpp \
           model_selection_test.cpp \
           neural_network_test.cpp \
           neurons_selection_test.cpp \
           normalized_squared_error_test.cpp \
           performance_test.cpp \
           pooling_layer_test.cpp \
           pch.cpp  # Precompiled header source


# Include more test files if needed by uncommenting
# SOURCES += perceptron_layer_test.cpp \
#            pooling_layer_test.cpp \
#            probabilistic_layer_3d_test.cpp \
#            ...


# Precompiled Header
HEADERS += pch.h


# Enable precompiled headers in Qt
QMAKE_CXXFLAGS += -include pch.h


# OpenNN library


win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../opennn/release/ -lopennn
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../opennn/debug/ -lopennn
else:unix: LIBS += -L$$OUT_PWD/../opennn/ -lopennn


INCLUDEPATH += $$PWD/../opennn
DEPENDPATH += $$PWD/../opennn


win32:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/release/opennn.lib
else:win32:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/debug/opennn.lib
else:unix: PRE_TARGETDEPS += $$OUT_PWD/../opennn/libopennn.a


# OpenMP library


#win32:{
#QMAKE_CXXFLAGS += -openmp
#QMAKE_LFLAGS  += -openmp
#}


#unix:!macx{
#QMAKE_CXXFLAGS+= -fopenmp
#QMAKE_LFLAGS += -fopenmp


#QMAKE_CXXFLAGS+= -std=c++17
#QMAKE_LFLAGS += -std=c++17
#}
# OpenMP library


include(../opennmp.pri)

