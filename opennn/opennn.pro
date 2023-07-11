#   OpenNN: Open Neural Networks Library
#   www.opennn.net
#
#   O P E N N N   Q T   C R E A T O R   P R O J E C T
#
#   Artificial Intelligence Techniques SL (Artelnics)
#   artelnics@artelnics.com

QT = \ # Do not use qt
    widgets

TARGET = opennn
#DESTDIR = "$$PWD/bin"

TEMPLATE = lib

CONFIG += staticlib
#CONFIG += c++17

CONFIG(debug, debug|release) {
    DEFINES += OPENNN_DEBUG
}

DEFINES += __Cpp17__


QMAKE_CXXFLAGS += -bigobj

# OpenMP library

win32:{
#QMAKE_CXXFLAGS += -std=c++17 -fopenmp -pthread #-lgomp -openmp
#QMAKE_LFLAGS += -fopenmp -pthread #-lgomp -openmp
#LIBS += -fopenmp -pthread #-lgomp
}else:!macx{QMAKE_CXXFLAGS+= -fopenmp -lgomp -std=c++17
QMAKE_LFLAGS += -fopenmp -lgomp
LIBS += -fopenmp -pthread -lgomp
}else: macx{
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
    numerical_differentiation.h \
    config.h \
    opennn_strings.h \
    opennn_images.h \
    statistics.h \
    scaling.h \
    correlations.h \
    codification.h \
    tinyxml2.h \
    filesystem.h \
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
    mean_squared_error.h \
    vgg16.h \
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
    tensor_utilities.h \
    unit_testing.h \
    flatten_layer.h \
    text_analytics.h \
#    region_based_object_detector.h \
#    json_to_xml.h \
    region_proposal_layer.h \
    non_max_suppression_layer.h \
    bounding_box_regressor_layer.h \
    opennn.h

SOURCES += \
    numerical_differentiation.cpp \
    opennn_strings.cpp \
    opennn_images.cpp \
    tensor_utilities.cpp \
    statistics.cpp \
    scaling.cpp \
    correlations.cpp \
    codification.cpp \
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
    data_set.cpp \
    sum_squared_error.cpp \
    normalized_squared_error.cpp \
    minkowski_error.cpp \
    mean_squared_error.cpp \
    vgg16.cpp \
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
    text_analytics.cpp \
#    region_based_object_detector.cpp \
#    json_to_xml.cpp \
    region_proposal_layer.cpp \
    non_max_suppression_layer.cpp \
    bounding_box_regressor_layer.cpp \
    unit_testing.cpp


#contains(DEFINES, OPENNN_MKL)
#{
#    win32:{

#    INTEL_HOME = "C:\\Program Files (x86)\\IntelSWTools\\compilers_and_libraries_2020.1.216\\windows"

#    INCLUDEPATH += "$$INTEL_HOME/mkl/include"

#    LIBS += -L"$$INTEL_HOME/redist\intel64_win\mkl" -lmkl_rt
#    LIBS += -L"$$INTEL_HOME/redist\intel64_win\mkl" -lmkl_core

#    win32:CONFIG(release, debug|release): LIBS += -L'$$INTEL_HOME/mkl/lib/intel64_win/' -lmkl_rt
#    else:win32:CONFIG(debug, debug|release): LIBS += -L'$$INTEL_HOME/mkl/lib/intel64_win/' -lmkl_rt
#    }

#    unix:{

#    ENV_HOME = $$(HOME)

#    INCLUDEPATH += $$ENV_HOME/intel/compilers_and_libraries_2020.4.304/linux/mkl/include/
#    INCLUDEPATH += $$ENV_HOME/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64/

#    #QMAKE_PRE_LINK += $$ENV_HOME/intel/compilers_and_libraries_2020.4.304/linux/mkl/bin/mklvars.sh intel64

#    QMAKE_PRE_LINK=/bin/sh $$ENV_HOME/intel/compilers_and_libraries_2020.4.304/linux/mkl/bin/mklvars.sh intel64

#    #LIBS += -L$$ENV_HOME/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64/ -lmkl_rt

#    LIBS += -L$$ENV_HOME/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64/ -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core
#    LIBS += -L$$ENV_HOME/intel/lib/intel64 -liomp5 -lpthread -ldl -lm

#    #LIBS += -L$$ENV_HOME/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64/ -lmkl_rt
#    #LIBS += -L$$ENV_HOME/intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64/  \
#    #    -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
#    #    -L/opt/intel/lib/intel64 \
#    #    -liomp5 -lpthread -ldl -lm
#    }
#}

