#   OpenNN: Open Neural Networks Library
#   www.artelnics.com/opennn
#
#   B L A N K _ C U D A   P R O J E C T
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

QT += \
    core \
    widgets

TARGET = blank_cuda

TEMPLATE = app

CONFIG += console
CONFIG += c++17

mac{
    CONFIG-=app_bundle
}

DESTDIR = "$$PWD/bin"

SOURCES += \
    ../opennn/adaptive_moment_estimation.cpp \
    ../opennn/addition_layer_3d.cpp \
    ../opennn/bounding_layer.cpp \
    ../opennn/convolutional_layer.cpp \
    ../opennn/correlations.cpp \
    ../opennn/cross_entropy_error.cpp \
    ../opennn/cross_entropy_error_3d.cpp \
    ../opennn/dataset.cpp \
    ../opennn/embedding_layer.cpp \
    ../opennn/flatten_layer.cpp \
    ../opennn/flatten_layer_3d.cpp \
    ../opennn/genetic_algorithm.cpp \
    ../opennn/growing_inputs.cpp \
    ../opennn/growing_neurons.cpp \
    ../opennn/image_dataset.cpp \
    ../opennn/images.cpp \
    ../opennn/inputs_selection.cpp \
    ../opennn/kmeans.cpp \
    ../opennn/language_dataset.cpp \
    ../opennn/layer.cpp \
    ../opennn/learning_rate_algorithm.cpp \
    ../opennn/levenberg_marquardt_algorithm.cpp \
    ../opennn/loss_index.cpp \
    ../opennn/mean_squared_error.cpp \
    ../opennn/minkowski_error.cpp \
    ../opennn/model_expression.cpp \
    ../opennn/model_selection.cpp \
    ../opennn/multihead_attention_layer.cpp \
    ../opennn/neural_network.cpp \
    ../opennn/neurons_selection.cpp \
    ../opennn/normalization_layer_3d.cpp \
    ../opennn/normalized_squared_error.cpp \
    ../opennn/optimization_algorithm.cpp \
    ../opennn/perceptron_layer.cpp \
    ../opennn/perceptron_layer_3d.cpp \
    ../opennn/pooling_layer.cpp \
    ../opennn/probabilistic_layer_3d.cpp \
    ../opennn/quasi_newton_method.cpp \
    ../opennn/recurrent_layer.cpp \
    ../opennn/response_optimization.cpp \
    ../opennn/scaling.cpp \
    ../opennn/scaling_layer_2d.cpp \
    ../opennn/scaling_layer_4d.cpp \
    ../opennn/statistics.cpp \
    ../opennn/stochastic_gradient_descent.cpp \
    ../opennn/strings_utilities.cpp \
    ../opennn/tensors.cpp \
    ../opennn/testing_analysis.cpp \
    ../opennn/time_series_dataset.cpp \
    ../opennn/tinyxml2.cpp \
    ../opennn/training_strategy.cpp \
    ../opennn/transformer.cpp \
    ../opennn/unscaling_layer.cpp \
    ../opennn/vgg16.cpp \
    ../opennn/weighted_squared_error.cpp \
    main.cpp 

CUDA_SOURCES += \
    ../opennn/kernel.cu

HEADERS += \
    ../opennn/adaptive_moment_estimation.h \
    ../opennn/addition_layer_3d.h \
    ../opennn/bounding_layer.h \
    ../opennn/convolutional_layer.h \
    ../opennn/correlations.h \
    ../opennn/cross_entropy_error.h \
    ../opennn/cross_entropy_error_3d.h \
    ../opennn/dataset.h \
    ../opennn/embedding_layer.h \
    ../opennn/flatten_layer.h \
    ../opennn/flatten_layer_3d.h \
    ../opennn/genetic_algorithm.h \
    ../opennn/growing_inputs.h \
    ../opennn/growing_neurons.h \
    ../opennn/image_dataset.h \
    ../opennn/images.h \
    ../opennn/inputs_selection.h \
    ../opennn/kernel.cuh \
    ../opennn/kmeans.h \
    ../opennn/language_dataset.h \
    ../opennn/layer.h \
    ../opennn/learning_rate_algorithm.h \
    ../opennn/levenberg_marquardt_algorithm.h \
    ../opennn/loss_index.h \
    ../opennn/mean_squared_error.h \
    ../opennn/minkowski_error.h \
    ../opennn/model_expression.h \
    ../opennn/model_selection.h \
    ../opennn/multihead_attention_layer.h \
    ../opennn/neural_network.h \
    ../opennn/neurons_selection.h \
    ../opennn/normalization_layer_3d.h \
    ../opennn/normalized_squared_error.h \
    ../opennn/optimization_algorithm.h \
    ../opennn/pch.h \
    ../opennn/perceptron_layer.h \
    ../opennn/dense_3d.h \
    ../opennn/pooling_layer.h \
    ../opennn/probabilistic_layer_3d.h \
    ../opennn/quasi_newton_method.h \
    ../opennn/recurrent_layer.h \
    ../opennn/response_optimization.h \
    ../opennn/scaling.h \
    ../opennn/scaling_layer_2d.h \
    ../opennn/scaling_layer_4d.h \
    ../opennn/statistics.h \
    ../opennn/stochastic_gradient_descent.h \
    ../opennn/strings_utilities.h \
    ../opennn/tensors.h \
    ../opennn/testing_analysis.h \
    ../opennn/time_series_dataset.h \
    ../opennn/tinyxml2.h \
    ../opennn/training_strategy.h \
    ../opennn/transformer.h \
    ../opennn/unscaling_layer.h \
    ../opennn/vgg16.h \
    ../opennn/weighted_squared_error.h

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
include(../opennmp.pri)

#win32-g++{
#QMAKE_LFLAGS += -static-libgcc
#QMAKE_LFLAGS += -static-libstdc++
#QMAKE_LFLAGS += -static
#}

#win32:!win32-g++{
##QMAKE_CXXFLAGS+= -arch:AVX
##QMAKE_CFLAGS+= -arch:AVX
#}
