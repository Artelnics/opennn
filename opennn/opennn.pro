#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   O P E N N N   Q T   C R E A T O R   P R O J E C T
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

TARGET = opennn

TEMPLATE = lib

CONFIG += staticlib
CONFIG += c++17

CONFIG(debug, debug|release){DEFINES += OPENNN_DEBUG}

# OpenMP library

win32{
QMAKE_CXXFLAGS += -bigobj

QMAKE_CXXFLAGS += -openmp
QMAKE_LFLAGS += -openmp
}

unix:macx{
INCLUDEPATH += /usr/local/opt/libomp/include
LIBS += /usr/local/opt/libomp/lib/libomp.dylib
}

unix:!macx{
QMAKE_CXXFLAGS+= -fopenmp
QMAKE_LFLAGS += -fopenmp
}

# Eigen library

INCLUDEPATH += ../eigen

HEADERS += \
    numerical_differentiation.h \
    config.h \
    opennn_strings.h \
    statistics.h \
    scaling.h \
    correlations.h \
    tinyxml2.h \
    data_set.h \
    layer.h \
    scaling_layer.h \
    unscaling_layer.h \
    perceptron_layer.h \
    probabilistic_layer.h \
    pooling_layer.h \
    convolutional_layer.h \
    bounding_layer.h \
    long_short_term_memory_layer.h \
    recurrent_layer.h \
    neural_network.h \
    loss_index.h \
    mean_squared_error.h \
    optimization_algorithm.h \
    stochastic_gradient_descent.h\
    training_strategy.h \
    neural_network.h \
    sum_squared_error.h\
    normalized_squared_error.h\
    minkowski_error.h \
    weighted_squared_error.h\
    cross_entropy_error.h \
    training_strategy.h \
    learning_rate_algorithm.h \
    quasi_newton_method.h \
    levenberg_marquardt_algorithm.h\
    gradient_descent.h \
    adaptive_moment_estimation.h\
    conjugate_gradient.h\
    model_selection.h \
    neurons_selection.h \
    growing_neurons.h \
    inputs_selection.h \
    growing_inputs.h \
    pruning_inputs.h \
    genetic_algorithm.h \
    testing_analysis.h \
    response_optimization.h \
    tensor_utilities.h \
    unit_testing.h \
    opennn.h

SOURCES += \
    numerical_differentiation.cpp \
    opennn_strings.cpp \
    tensor_utilities.cpp \
    statistics.cpp \
    scaling.cpp \
    correlations.cpp \
    tinyxml2.cpp \
    data_set.cpp \
    layer.cpp \
    scaling_layer.cpp \
    unscaling_layer.cpp \
    perceptron_layer.cpp \
    probabilistic_layer.cpp \
    pooling_layer.cpp \
    bounding_layer.cpp \
    convolutional_layer.cpp \
    long_short_term_memory_layer.cpp \
    recurrent_layer.cpp \
    neural_network.cpp \
    loss_index.cpp \
    mean_squared_error.cpp \
    stochastic_gradient_descent.cpp \
    training_strategy.cpp \
    optimization_algorithm.cpp \
    sum_squared_error.cpp \
    normalized_squared_error.cpp \
    minkowski_error.cpp \
    weighted_squared_error.cpp \
    cross_entropy_error.cpp \
    learning_rate_algorithm.cpp \
    quasi_newton_method.cpp \
    levenberg_marquardt_algorithm.cpp \
    gradient_descent.cpp \
    adaptive_moment_estimation.cpp\
    conjugate_gradient.cpp \
    model_selection.cpp \
    neurons_selection.cpp \
    growing_neurons.cpp \
    inputs_selection.cpp \
    growing_inputs.cpp \
    pruning_inputs.cpp \
    genetic_algorithm.cpp \
    testing_analysis.cpp \
    response_optimization.cpp \
    unit_testing.cpp
