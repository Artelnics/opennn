#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   O P E N N N   Q T   C R E A T O R   P R O J E C T
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

QT = # Do not use qt

TARGET = opennn

TEMPLATE = lib

CONFIG += staticlib
#CONFIG += c++11

CONFIG(debug, debug|release) {
    DEFINES += __OPENNN_DEBUG__
}

#DEFINES += __Cpp11__


# OpenMP library

#win32:!win32-g++{
#QMAKE_CXXFLAGS += -openmp
#QMAKE_LFLAGS   += -openmp

#QMAKE_CXXFLAGS += -std=c++11 -fopenmp -pthread -lgomp
#QMAKE_LFLAGS += -fopenmp -pthread -lgomp
#LIBS += -fopenmp -pthread -lgomp
#}else:!macx{
#QMAKE_CXXFLAGS+= -fopenmp -lgomp
#QMAKE_LFLAGS += -fopenmp -lgomp
#}

#macx{
#INCLUDEPATH += /usr/local/opt/libiomp/include/libiomp
#}

QMAKE_CXXFLAGS+= -fopenmp
QMAKE_LFLAGS +=  -fopenmp

# Eigen library

INCLUDEPATH += eigen

HEADERS += \
    tinyxml2.h \
    vector.h \
    matrix.h \
    tensor.h \
    functions.h \
    statistics.h \
    correlations.h \
    transformations.h \
    opennn_strings.h \
    metrics.h \
    k_means.h \
    numerical_differentiation.h \
    scaling_layer.h \
    unscaling_layer.h \
    bounding_layer.h \
    long_short_term_memory_layer.h \
    recurrent_layer.h \
    perceptron_layer.h \
    probabilistic_layer.h \
    layer.h \
    pooling_layer.h \
    convolutional_layer.h \
    principal_components_layer.h \
    loss_index.h \
    data_set.h \
    neural_network.h \
    sum_squared_error.h\
    normalized_squared_error.h\
    minkowski_error.h \
    mean_squared_error.h \
    weighted_squared_error.h\
    cross_entropy_error.h \
    training_strategy.h \
    optimization_algorithm.h \
    learning_rate_algorithm.h \
    quasi_newton_method.h \
    levenberg_marquardt_algorithm.h\
    gradient_descent.h \
    stochastic_gradient_descent.h\
    adaptive_moment_estimation.h\
    conjugate_gradient.h\
    model_selection.h \
    neurons_selection.h \
    incremental_neurons.h \
    inputs_selection.h \
    growing_inputs.h \
    pruning_inputs.h \
    genetic_algorithm.h \
    testing_analysis.h \
    response_optimization.h \
    unit_testing.h
    opennn.h \

SOURCES += \
    tinyxml2.cpp \
    functions.cpp \
    statistics.cpp \
    opennn_strings.cpp \
    metrics.cpp \
    correlations.cpp \
    transformations.cpp \
    k_means.cpp \
    numerical_differentiation.cpp \
    unscaling_layer.cpp \
    scaling_layer.cpp \
    bounding_layer.cpp \
    long_short_term_memory_layer.cpp \
    recurrent_layer.cpp \
    probabilistic_layer.cpp \
    perceptron_layer.cpp \
    layer.cpp \
    pooling_layer.cpp \
    convolutional_layer.cpp \
    principal_components_layer.cpp \
    loss_index.cpp \
    data_set.cpp \
    neural_network.cpp \
    sum_squared_error.cpp \
    normalized_squared_error.cpp \
    minkowski_error.cpp \
    mean_squared_error.cpp \
    weighted_squared_error.cpp \
    cross_entropy_error.cpp \
    training_strategy.cpp \
    optimization_algorithm.cpp \
    learning_rate_algorithm.cpp \
    quasi_newton_method.cpp \
    levenberg_marquardt_algorithm.cpp \
    gradient_descent.cpp \
    stochastic_gradient_descent.cpp\
    adaptive_moment_estimation.cpp\
    conjugate_gradient.cpp \
    model_selection.cpp \
    neurons_selection.cpp \
    incremental_neurons.cpp \
    inputs_selection.cpp \
    growing_inputs.cpp \
    pruning_inputs.cpp \
    genetic_algorithm.cpp \
    testing_analysis.cpp \
    response_optimization.cpp \
    unit_testing.cpp
