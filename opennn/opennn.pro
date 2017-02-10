###################################################################################################
#                                                                                                 #
#   OpenNN: Open Neural Networks Library                                                          #
#   www.opennn.net                                                                                #
#                                                                                                 #
#   O P E N N N   Q T   C R E A T O R   P R O J E C T                                             #
#                                                                                                 #
#   Roberto Lopez                                                                                 #
#   Artelnics - Making intelligent use of data                                                    #
#   robertolopez@artelnics.com                                                                    #
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
}else{
QMAKE_CXXFLAGS+= -fopenmp
QMAKE_LFLAGS +=  -fopenmp
}

mac{

INCLUDEPATH += /usr/local/opt/libiomp/include/libiomp

}

HEADERS += \
    variables.h \
    instances.h \
    missing_values.h \
    data_set.h \
    plug_in.h \
    ordinary_differential_equations.h \
    mathematical_model.h \
        inputs.h \
        outputs.h \
        unscaling_layer.h \
        scaling_layer.h \
        probabilistic_layer.h \
        perceptron_layer.h \
        perceptron.h \
        neural_network.h \
    multilayer_perceptron.h \
    independent_parameters.h \
    conditions_layer.h \
    bounding_layer.h \
    sum_squared_error.h \
    root_mean_squared_error.h \
    error_term.h \
    regularization_term.h \
    loss_index.h \
    outputs_integrals.h \
    normalized_squared_error.h \
    neural_parameters_norm.h \
    minkowski_error.h \
    mean_squared_error.h \
    weighted_squared_error.h \
    roc_area_error.h \
    cross_entropy_error.h \
    training_strategy.h \
    training_algorithm.h \
    training_rate_algorithm.h \
    random_search.h \
    quasi_newton_method.h \
    newton_method.h \
    levenberg_marquardt_algorithm.h \
    gradient_descent.h \
    evolutionary_algorithm.h \
    conjugate_gradient.h \
    model_selection.h \
    order_selection_algorithm.h \
    incremental_order.h \
    golden_section_order.h \
    simulated_annealing_order.h \
    ant_colony_optimization.h \
    inputs_selection_algorithm.h \
    growing_inputs.h \
    pruning_inputs.h \
    genetic_algorithm.h \
    testing_analysis.h \
    vector.h \
    matrix.h \
    numerical_integration.h \
    numerical_differentiation.h \
    opennn.h \
    principal_components_layer.h \
    threshold_selection_algorithm.h \
    f1_score_optimization_threshold.h \
    matthew_correlation_optimization_threshold.h \
    youden_index_optimization_threshold.h \
    kappa_coefficient_optimization_threshold.h \
    roc_curve_optimization_threshold.h \
    selective_pruning.h

SOURCES += \
    variables.cpp \
    instances.cpp \
    missing_values.cpp \
    data_set.cpp \
    plug_in.cpp \
    ordinary_differential_equations.cpp \
    mathematical_model.cpp \
    inputs.cpp \
    outputs.cpp \
    unscaling_layer.cpp \
    scaling_layer.cpp \
    probabilistic_layer.cpp \
    perceptron_layer.cpp \
    perceptron.cpp \
    neural_network.cpp \
    multilayer_perceptron.cpp \
    independent_parameters.cpp \
    conditions_layer.cpp \
    bounding_layer.cpp \
    sum_squared_error.cpp \
    root_mean_squared_error.cpp \
    error_term.cpp \
    regularization_term.cpp \
    loss_index.cpp \
    outputs_integrals.cpp \
    normalized_squared_error.cpp \
    neural_parameters_norm.cpp \
    minkowski_error.cpp \
    mean_squared_error.cpp \
    weighted_squared_error.cpp \
    roc_area_error.cpp \
    cross_entropy_error.cpp \
    training_strategy.cpp \
    training_algorithm.cpp \
    training_rate_algorithm.cpp \
    random_search.cpp \
    quasi_newton_method.cpp \
    newton_method.cpp \
    levenberg_marquardt_algorithm.cpp \
    gradient_descent.cpp \
    evolutionary_algorithm.cpp \
    conjugate_gradient.cpp \
    model_selection.cpp \
    order_selection_algorithm.cpp \
    incremental_order.cpp \
    golden_section_order.cpp \
    simulated_annealing_order.cpp \
    ant_colony_optimization.cpp \
    inputs_selection_algorithm.cpp \
    growing_inputs.cpp \
    pruning_inputs.cpp \
    genetic_algorithm.cpp \
    testing_analysis.cpp \
    numerical_integration.cpp \
    numerical_differentiation.cpp \
    principal_components_layer.cpp \
    threshold_selection_algorithm.cpp \
    f1_score_optimization_threshold.cpp \
    matthew_correlation_optimization_threshold.cpp \
    youden_index_optimization_threshold.cpp \
    kappa_coefficient_optimization_threshold.cpp \
    roc_curve_optimization_threshold.cpp \
    selective_pruning.cpp

# TinyXML2 library

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../tinyxml2/release/ -ltinyxml2
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../tinyxml2/debug/ -ltinyxml2
else:unix: LIBS += -L$$OUT_PWD/../tinyxml2/ -ltinyxml2

INCLUDEPATH += $$PWD/../tinyxml2
DEPENDPATH += $$PWD/../tinyxml2

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../tinyxml2/release/libtinyxml2.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../tinyxml2/debug/libtinyxml2.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../tinyxml2/release/tinyxml2.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../tinyxml2/debug/tinyxml2.lib
else:unix: PRE_TARGETDEPS += $$OUT_PWD/../tinyxml2/libtinyxml2.a

# MPI libraries
#DEFINES += __OPENNN_MPI__

contains(DEFINES, __OPENNN_MPI__){
include(../mpi.pri)
}

# CUDA - To uncomment:
    #the following DEFINES
    #include(../cuda.pri) at the end of this file
    #neuralengine.pro > include(../opennn/cuda.pri)
    #neuralreader < include(../cuda.pri)
    #tinyxml2.pro > #include(../cuda.pri)
    #test.pro > #include(../cuda.pri)
    #don't use this include in neuraleditor.pro for the Qt 5.4.0 version




#DEFINES += __OPENNN_CUDA__

contains(DEFINES, __OPENNN_CUDA__){

OTHER_FILES +=  utilities.cu

windows{
CUDA_DIR = C:/"Program Files"/"NVIDIA GPU Computing Toolkit"/CUDA/v8.0            # Path to cuda toolkit install
}else:mac{
CUDA_DIR = /Developer/NVIDIA/CUDA-7.5
}else:unix{
CUDA_DIR = /usr/local/cuda-7.5
}

INCLUDEPATH += $$CUDA_DIR/lib/x64
DEPENDPATH += $$CUDA_DIR/lib/x64

win32:!win32-g++: PRE_TARGETDEPS += $$CUDA_DIR/lib/x64/cuda.lib
else:win32-g++: PRE_TARGETDEPS += $$CUDA_DIR/lib/x64/libcuda.a

INCLUDEPATH += $$CUDA_DIR/lib/x64
DEPENDPATH += $$CUDA_DIR/lib/x64

win32:!win32-g++: PRE_TARGETDEPS += $$CUDA_DIR/lib/x64/cudart.lib
else:win32-g++: PRE_TARGETDEPS += $$CUDA_DIR/lib/x64/libcudart.a

win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$CUDA_DIR/lib/x64/libcublas.a
else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$CUDA_DIR/lib/x64/libcublas.a
else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$CUDA_DIR/lib/x64/cublas.lib
else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$CUDA_DIR/lib/x64/cublas.lib

INCLUDEPATH += $$CUDA_DIR/include

# CUDA settings <-- may change depending on your system

CUDA_SOURCES += utilities.cu
SYSTEM_NAME = x64           # Depending on your system either 'Win32', 'x64', or 'Win64'
SYSTEM_TYPE = 64            # '32' or '64', depending on your system
CUDA_ARCH = sm_35           # Type of CUDA architecture, for example 'compute_10', 'compute_11', 'sm_10'
NVCC_OPTIONS = --use_fast_math

CUDA_LIBS += -lcuda -lcudart -lcublas

# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

CUDA_OBJECTS_DIR = $$OBJECTS_DIR

# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = debug/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$CUDA_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = release/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$CUDA_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}

#include(../cuda.pri)

}


