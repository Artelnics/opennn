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
           convolutional_layer_test.cpp \
           correlations_test.cpp \
           cross_entropy_error_3d_test.cpp \
           cross_entropy_error_test.cpp \
           data_set_test.cpp \
           embedding_layer_test.cpp \
           flatten_layer_test.cpp \
           genetic_algorithm_test.cpp \
           growing_inputs_test.cpp \
           growing_neurons_test.cpp \
           image_data_set_test.cpp \
           learning_rate_algorithm_test.cpp \
           levenberg_marquardt_algorithm_test.cpp \
           mean_squared_error_test.cpp \
           minkowski_error_test.cpp \
           model_selection_test.cpp \
           multihead_attention_layer_test.cpp \
           neural_network_test.cpp \
           normalized_squared_error_test.cpp \
           perceptron_layer_test.cpp \
           performance_test.cpp \
           pooling_layer_test.cpp \
           probabilistic_layer_3d_test.cpp \
           probabilistic_layer_test.cpp \
           quasi_newton_method_test.cpp \
           recurrent_layer_test.cpp \
           response_optimization_test.cpp \
           scaling_layer_test.cpp \
           scaling_test.cpp \
           statistics_test.cpp \
           stochastic_gradient_descent_test.cpp \
           tensors_test.cpp \
           testing_analysis_test.cpp \
           time_series_data_set_test.cpp \
           training_strategy_test.cpp \
           transformer_test.cpp \
           unscaling_layer_test.cpp \
           weighted_squared_error_test.cpp \
           pch.cpp \
            # Include more test files if needed by uncommenting

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
