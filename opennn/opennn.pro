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
#QMAKE_CXXFLAGS += -bigobj
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
    flatten_layer_3d.h \
    pch.h \
    codification.h \
    cross_entropy_error_3d.h \
    embedding_layer.h \
    multihead_attention_layer.h \
    kmeans.h \
    strings_utilities.h \
    images.h \
    statistics.h \
    scaling.h \
    correlations.h \
    tinyxml2.h \
    dataset.h \
    batch.h \
    time_series_data_set.h \
    image_dataset.h \
    language_data_set.h \
    layer.h \
    scaling_layer_2d.h \
    scaling_layer_4d.h \
    transformer.h \
    unscaling_layer.h \
    perceptron_layer.h \
    perceptron_layer_3d.h \
    probabilistic_layer_3d.h \
    pooling_layer.h \
    convolutional_layer.h \
    bounding_layer.h \
    recurrent_layer.h \
    neural_network.h \
    model_expression.h \
    loss_index.h \
    optimization_algorithm.h \
    normalized_squared_error.h\
    minkowski_error.h \
    mean_squared_error.h \
    weighted_squared_error.h\
    cross_entropy_error_2d.h \
    training_strategy.h \
    learning_rate_algorithm.h \
    quasi_newton_method.h \
    levenberg_marquardt_algorithm.h\
    stochastic_gradient_descent.h\
    adaptive_moment_estimation.h\
    model_selection.h \
    neurons_selection.h \
    growing_neurons.h \
    inputs_selection.h \
    growing_inputs.h \
    genetic_algorithm.h \
    testing_analysis.h \
    response_optimization.h \
    tensors.h \
    flatten_layer.h \
    bounding_box_regressor_layer.h \
    bounding_box.h \
    word_bag.h \
    addition_layer_3d.h \
    normalization_layer_3d.h \
    flatten_layer_3d.h \
    opennn.h

SOURCES += \
    flatten_layer_3d.cpp \
    model_expression.cpp \
    pch.cpp \
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
    tinyxml2.cpp \
    data_set.cpp \
    batch.cpp \
    time_series_data_set.cpp \
    image_data_set.cpp \
    language_data_set.cpp \
    layer.cpp \
    scaling_layer_2d.cpp \
    scaling_layer_4d.cpp \
    transformer.cpp \
    unscaling_layer.cpp \
    perceptron_layer.cpp \
    perceptron_layer_3d.cpp \
    probabilistic_layer_3d.cpp \
    pooling_layer.cpp \
    bounding_layer.cpp \
    convolutional_layer.cpp \
    recurrent_layer.cpp \
    neural_network.cpp \
    loss_index.cpp \
    stochastic_gradient_descent.cpp \
    training_strategy.cpp \
    optimization_algorithm.cpp \
    normalized_squared_error.cpp \
    minkowski_error.cpp \
    mean_squared_error.cpp \
    weighted_squared_error.cpp \
    cross_entropy_error.cpp \
    learning_rate_algorithm.cpp \
    quasi_newton_method.cpp \
    levenberg_marquardt_algorithm.cpp \
    adaptive_moment_estimation.cpp\
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
    flatten_layer_3d.cpp \
