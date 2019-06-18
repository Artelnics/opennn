###################################################################################################
#                                                                                                 #
#   OpenNN: Open Neural Networks Library                                                          #
#   www.opennn.net                                                                                #
#                                                                                                 #
#   O P E N N N   Q T   C R E A T O R   P R O J E C T                                             #
#                                                                                                 #
#   Artificial Intelligence Techniques SL (Artelnics)                                             #
#   artelnics@artelnics.com                                                                       #
#                                                                                                 #
###################################################################################################

QT = # Do not use qt

TARGET = opennn

TEMPLATE = lib

CONFIG += staticlib
CONFIG += c++11

CONFIG(debug, debug|release) {
    DEFINES += __OPENNN_DEBUG__
}

#DEFINES += __Cpp11__

# Eigen library

INCLUDEPATH += eigen

# OpenMP library

win32:!win32-g++{
QMAKE_CXXFLAGS += -openmp
QMAKE_LFLAGS   += -openmp

QMAKE_CXXFLAGS += -std=c++11 -fopenmp -pthread -lgomp
QMAKE_LFLAGS += -fopenmp -pthread -lgomp
LIBS += -fopenmp -pthread -lgomp
}else:!macx{
QMAKE_CXXFLAGS+= -fopenmp -lgomp
QMAKE_LFLAGS +=  -fopenmp -lgomp
}

macx{

#INCLUDEPATH += /usr/local/opt/libiomp/include/libiomp

}

HEADERS += \
    variables.h \
    instances.h \
    missing_values.h \
    data_set.h \
    inputs.h \
    outputs.h \
    unscaling_layer.h \
    scaling_layer.h \
    inputs_trending_layer.h \
    outputs_trending_layer.h \
    probabilistic_layer.h \
    perceptron_layer.h \
    neural_network.h \
    multilayer_perceptron.h \
    bounding_layer.h \
    sum_squared_error.h \
    loss_index.h \
    normalized_squared_error.h \
    minkowski_error.h \
    mean_squared_error.h \
    weighted_squared_error.h \
    cross_entropy_error.h \
    training_strategy.h \
    optimization_algorithm.h \
    learning_rate_algorithm.h \
    quasi_newton_method.h \
    levenberg_marquardt_algorithm.h \
    gradient_descent.h \
    stochastic_gradient_descent.h\
    adaptive_moment_estimation.h\
    conjugate_gradient.h \
    model_selection.h \
    order_selection_algorithm.h \
    incremental_order.h \
    golden_section_order.h \
    simulated_annealing_order.h \
    inputs_selection_algorithm.h \
    growing_inputs.h \
    pruning_inputs.h \
    genetic_algorithm.h \
    testing_analysis.h \
    vector.h \
    matrix.h \
    sparse_matrix.h \
    numerical_differentiation.h \
    opennn.h \
    principal_components_layer.h \
    selective_pruning.h \
    file_utilities.h \
    association_rules.h \
    text_analytics.h \
    k_nearest_neighbors.h \
    tinyxml2.h \
    correlation_analysis.h \

SOURCES += \
    variables.cpp \
    instances.cpp \
    missing_values.cpp \
    data_set.cpp \
    inputs.cpp \
    outputs.cpp \
    unscaling_layer.cpp \
    scaling_layer.cpp \
    inputs_trending_layer.cpp \
    outputs_trending_layer.cpp \
    probabilistic_layer.cpp \
    perceptron_layer.cpp \
    neural_network.cpp \
    multilayer_perceptron.cpp \
    bounding_layer.cpp \
    sum_squared_error.cpp \
    loss_index.cpp \
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
    order_selection_algorithm.cpp \
    incremental_order.cpp \
    golden_section_order.cpp \
    simulated_annealing_order.cpp \
    inputs_selection_algorithm.cpp \
    growing_inputs.cpp \
    pruning_inputs.cpp \
    genetic_algorithm.cpp \
    testing_analysis.cpp \
    numerical_differentiation.cpp \
    principal_components_layer.cpp \
    selective_pruning.cpp \
    file_utilities.cpp \
    association_rules.cpp \
    text_analytics.cpp \
    k_nearest_neighbors.cpp \
    tinyxml2.cpp \
    correlation_analysis.cpp \

# MPI libraries
#DEFINES += __OPENNN_MPI__

contains(DEFINES, __OPENNN_MPI__){
include(../mpi.pri)

}
