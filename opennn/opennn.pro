###################################################################################################
#                                                                                                 #
#   OpenNN: Open Neural Networks Library                                                          #
#   www.opennn.net                                                                      #
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

CONFIG(debug, debug|release) {
    DEFINES += __OPENNN_DEBUG__
}

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

# C++11 flags
unix: !mac{
QMAKE_CXXFLAGS+= -std=c++11
QMAKE_LFLAGS +=  -std=c++11
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
    solutions_error.h \
    root_mean_squared_error.h \
    performance_term.h \
    performance_functional.h \
    outputs_integrals.h \
    normalized_squared_error.h \
    neural_parameters_norm.h \
    minkowski_error.h \
    mean_squared_error.h \
    inverse_sum_squared_error.h \
    independent_parameters_error.h \
    final_solutions_error.h \
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
    order_selection_algorithm.h\
    incremental_order.h\
    golden_section_order.h\
    simulated_annealing_order.h\
    inputs_selection_algorithm.h\
    growing_inputs.h\
    pruning_inputs.h\
    genetic_algorithm.h\
    testing_analysis.h \
    vector.h \
    matrix.h \
    numerical_integration.h \
    numerical_differentiation.h \
    opennn.h

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
    solutions_error.cpp \
    root_mean_squared_error.cpp \
    performance_term.cpp \
    performance_functional.cpp \
    outputs_integrals.cpp \
    normalized_squared_error.cpp \
    neural_parameters_norm.cpp \
    minkowski_error.cpp \
    mean_squared_error.cpp \
    inverse_sum_squared_error.cpp \
    independent_parameters_error.cpp \
    final_solutions_error.cpp \
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
    order_selection_algorithm.cpp\
    incremental_order.cpp\
    golden_section_order.cpp\
    simulated_annealing_order.cpp\
    inputs_selection_algorithm.cpp\
    growing_inputs.cpp\
    pruning_inputs.cpp\
    genetic_algorithm.cpp\
    testing_analysis.cpp \
    numerical_integration.cpp \
    numerical_differentiation.cpp
