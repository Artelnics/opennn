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

DEFINES += __Cpp17__

# OpenMP library

win32-msvc* {
    QMAKE_CXXFLAGS += /std:c++17 /openmp /EHsc
    QMAKE_CXXFLAGS += /bigobj
    DEFINES += "EIGEN_THREAD_LOCAL=thread_local"
}
else:macx {
    QMAKE_CXXFLAGS += -std=c++17
    INCLUDEPATH += /usr/local/opt/libomp/include
    LIBS += -L/usr/local/opt/libomp/lib -lomp
    TEMPLATE_IS_APP = $$find(TEMPLATE, "app")
    !isEmpty(TEMPLATE_IS_APP): LIBS += -lpthread
}
else {
    QMAKE_CXXFLAGS += -std=c++17 -fopenmp
    QMAKE_LFLAGS   += -fopenmp
    TEMPLATE_IS_APP = $$find(TEMPLATE, "app")
    !isEmpty(TEMPLATE_IS_APP): LIBS += -lpthread -lgomp
}

# --- CUDA 12.5 ---
MY_CUDA_VER_MAJOR = 12
MY_CUDA_VER_MINOR = 5

MY_CUDA_FULL_VERSION_STR = $$sprintf("v%1.%2", $$MY_CUDA_VER_MAJOR, $$MY_CUDA_VER_MINOR)
MY_CUDA_SHORT_VERSION_STR = $$sprintf("%1.%2", $$MY_CUDA_VER_MAJOR, $$MY_CUDA_VER_MINOR)

win32 {
    CUDA_PATH_ATTEMPT1 = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/$$MY_CUDA_FULL_VERSION_STR"
    CUDA_PATH_ATTEMPT2 = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/$$MY_CUDA_SHORT_VERSION_STR"

    CUDA_PATH = $$CUDA_PATH_ATTEMPT1
    !exists($$CUDA_PATH) {
        CUDA_PATH = $$CUDA_PATH_ATTEMPT2
    }

    CUDA_LIB_DIR = $$CUDA_PATH/lib/x64
    CUDA_BIN_DIR = $$CUDA_PATH/bin
} else:unix {
    CUDA_PATH_ATTEMPT1_UNIX = /usr/local/cuda-$$MY_CUDA_SHORT_VERSION_STR
    CUDA_PATH_ATTEMPT2_UNIX = /usr/local/cuda-$$MY_CUDA_FULL_VERSION_STR
    CUDA_PATH_ATTEMPT3_UNIX = /usr/local/cuda

    CUDA_PATH = $$CUDA_PATH_ATTEMPT1_UNIX
    !exists($$CUDA_PATH) {
        CUDA_PATH = $$CUDA_PATH_ATTEMPT2_UNIX
        !exists($$CUDA_PATH) {
            CUDA_PATH = $$CUDA_PATH_ATTEMPT3_UNIX
        }
    }
    CUDA_LIB_DIR = $$CUDA_PATH/lib
    CUDA_BIN_DIR = $$CUDA_PATH/bin
}

!exists($$CUDA_PATH) {
    warning("CUDA path not found: $$CUDA_PATH. Please check your CUDA installation and CUDA_PATH in the .pro file.")
    warning("Attempted versions based on MY_VARS: $$MY_CUDA_FULL_VERSION_STR and $$MY_CUDA_SHORT_VERSION_STR")
} else {
    message("Using CUDA from: $$CUDA_PATH")
}

INCLUDEPATH += $$CUDA_PATH/include
DEPENDPATH += $$CUDA_PATH/include

LIBS += -L$$CUDA_LIB_DIR -lcudart_static

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
    time_seriesdataset.h \
    image_dataset.h \
    language_dataset.h \
    layer.h \
    scaling_layer_2d.h \
    scaling_layer_4d.h \
    transformer.h \
    unscaling_layer.h \
    perceptron_layer.h \
    perceptron_layer_3d.cpp \
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
    vgg16.h \
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
    dataset.cpp \
    time_series_dataset.cpp \
    image_dataset.cpp \
    language_dataset.cpp \
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
    vgg16.cpp \
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
