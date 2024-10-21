#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   O P E N N N   Q T   C R E A T O R   P R O J E C T
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

QT = # Do not use qt

TARGET = opennn
#DESTDIR = "$$PWD/bin"

TEMPLATE = lib

CONFIG += staticlib
CONFIG += c++17

CONFIG(debug, debug|release) {
    DEFINES += OPENNN_DEBUG
}

DEFINES += __Cpp17__

win32:{
QMAKE_CXXFLAGS += -bigobj
}

# OpenMP library

win32:{
#QMAKE_CXXFLAGS += -std=c++17 -fopenmp -pthread #-lgomp -openmp
#QMAKE_LFLAGS += -fopenmp -pthread #-lgomp -openmp
#LIBS += -fopenmp -pthread #-lgomp
}
else:!macx{QMAKE_CXXFLAGS+= -fopenmp -lgomp -std=c++17
QMAKE_LFLAGS += -fopenmp -lgomp
LIBS += -fopenmp -pthread -lgomp
}
else: macx{
INCLUDEPATH += /usr/local/opt/libomp/include
LIBS += /usr/local/opt/libomp/lib/libomp.dylib}

#win32:!win32-g++{
##QMAKE_CXXFLAGS+= -arch:AVX
##QMAKE_CFLAGS+= -arch:AVX
#}

#macx{
#INCLUDEPATH += /usr/local/opt/libiomp/include/libiomp
#}

# Eigen library

INCLUDEPATH += ../eigen

HEADERS += \
    codification.h \
    cross_entropy_error_3d.h \
    embedding_layer.h \
    multihead_attention_layer.h \
    kmeans.h \
    config.h \
    strings_utilities.h \
    images.h \
    statistics.h \
    descriptives.h \
    box_plot.h \
    histogram.h \
    scaling.h \
    correlations.h \
    correlation.h \
    codification.h \
    tinyxml2.h \
    data_set.h \
    batch.h \
    time_series_data_set.h \
    auto_association_data_set.h \
    image_data_set.h \
    text_data_set.h \
    language_data_set.h \
    layer.h \
    layer_forward_propagation.h \
    layer_back_propagation.h \
    layer_back_propagation_lm.h \
    scaling_layer_2d.h \
    scaling_layer_4d.h \
    transformer.h \
    unscaling_layer.h \
    perceptron_layer.h \
    perceptron_layer_3d.h \
    probabilistic_layer.h \
    probabilistic_layer_3d.h \
    pooling_layer.h \
    convolutional_layer.h \
    bounding_layer.h \
    long_short_term_memory_layer.h \
    recurrent_layer.h \
    neural_network.h \
    model_expression.h \
    neural_network_forward_propagation.h \
    neural_network_back_propagation.h \
    neural_network_back_propagation_lm.h \
    auto_associative_neural_network.h \
    loss_index.h \
    loss_index_forward_propagation.h \
    back_propagation.h \
    mean_squared_error.h \
    optimization_algorithm.h \
    stochastic_gradient_descent.h\
    training_strategy.h \
    sum_squared_error.h\
    normalized_squared_error.h\
    minkowski_error.h \
    mean_squared_error.h \
    weighted_squared_error.h\
    cross_entropy_error.h \
    training_strategy.h \
    learning_rate_algorithm.h \
    quasi_newton_method.h \
    levenberg_marquardt_algorithm.h\
    gradient_descent.h \
    stochastic_gradient_descent.h\
    adaptive_moment_estimation.h\
    conjugate_gradient.h\
    model_selection.h \
    neurons_selection.h \
    growing_neurons.h \
    inputs_selection.h \
    growing_inputs.h \
    genetic_algorithm.h \
    testing_analysis.h \
    response_optimization.h \
    tensors.h \
    unit_testing.h \
    flatten_layer.h \
    bounding_box_regressor_layer.h \
    bounding_box.h \
    unscaling_layer_forward_propagation.h \
    word_bag.h \
    addition_layer_3d.h \
    normalization_layer_3d.h \
    opennn.h

SOURCES += \
    cross_entropy_error_3d.cpp \
    embedding_layer.cpp \
    multihead_attention_layer.cpp \
    kmeans.cpp \
    strings_utilities.cpp \
    images.cpp \
    tensors.cpp \
    statistics.cpp \
    scaling.cpp \
    correlations.cpp \
    codification.cpp \
    tinyxml2.cpp \
    data_set.cpp \
    batch.cpp \
    time_series_data_set.cpp \
    auto_association_data_set.cpp \
    image_data_set.cpp \
    text_data_set.cpp \
    language_data_set.cpp \
    layer.cpp \
    scaling_layer_2d.cpp \
    scaling_layer_4d.cpp \
    transformer.cpp \
    unscaling_layer.cpp \
    perceptron_layer.cpp \
    perceptron_layer_3d.cpp \
    probabilistic_layer.cpp \
    probabilistic_layer_3d.cpp \
    pooling_layer.cpp \
    bounding_layer.cpp \
    convolutional_layer.cpp \
    long_short_term_memory_layer.cpp \
    recurrent_layer.cpp \
    neural_network.cpp \
    auto_associative_neural_network.cpp \
    loss_index.cpp \
    mean_squared_error.cpp \
    stochastic_gradient_descent.cpp \
    training_strategy.cpp \
    optimization_algorithm.cpp \
    sum_squared_error.cpp \
    normalized_squared_error.cpp \
    minkowski_error.cpp \
    mean_squared_error.cpp \
    weighted_squared_error.cpp \
    cross_entropy_error.cpp \
    learning_rate_algorithm.cpp \
    quasi_newton_method.cpp \
    levenberg_marquardt_algorithm.cpp \
    gradient_descent.cpp \
    stochastic_gradient_descent.cpp\
    adaptive_moment_estimation.cpp\
    conjugate_gradient.cpp \
    model_selection.cpp \
    neurons_selection.cpp \
    growing_neurons.cpp \
    inputs_selection.cpp \
    growing_inputs.cpp \
    genetic_algorithm.cpp \
    testing_analysis.cpp \
    response_optimization.cpp \
    flatten_layer.cpp \
    addition_layer_3d.cpp \
    normalization_layer_3d.cpp \
    unit_testing.cpp
