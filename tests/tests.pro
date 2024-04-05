#   OpenNN: Open Neural Networks Library                                                          
#   www.opennn.net                                                                                
#                                                                                                 
#   T E S T S   P R O J E C T                                                                     
#                                                                                                 
#   Artificial Intelligence Techniques SL (Artelnics)                                             
#   artelnics@artelnics.com                                                                       

CONFIG += console

mac{CONFIG-=app_bundle}

TARGET = tests

TEMPLATE = app

DESTDIR = "$$PWD/bin"

SOURCES += \
    adaptive_moment_estimation_test.cpp \
    cross_entropy_error_3d_test.cpp \
    tensor_utilities_test.cpp \
    data_set_test.cpp \
    growing_neurons_test.cpp \
    unscaling_layer_test.cpp \
    scaling_layer_test.cpp \
    probabilistic_layer_test.cpp \
    probabilistic_layer_3d_test.cpp \
    perceptron_layer_test.cpp \
    long_short_term_memory_layer_test.cpp \
    recurrent_layer_test.cpp \
    neural_network_test.cpp \
    bounding_layer_test.cpp \
    sum_squared_error_test.cpp \
    weighted_squared_error_test.cpp \
    minkowski_error_test.cpp \
    mean_squared_error_test.cpp \
    normalized_squared_error_test.cpp \
    cross_entropy_error_test.cpp \
    training_strategy_test.cpp \
    learning_rate_algorithm_test.cpp \
    quasi_newton_method_test.cpp \
    levenberg_marquardt_algorithm_test.cpp \
    gradient_descent_test.cpp \
    conjugate_gradient_test.cpp \
    model_selection_test.cpp \
    neurons_selection_test.cpp \
    inputs_selection_test.cpp \
    growing_inputs_test.cpp \
    genetic_algorithm_test.cpp \
    testing_analysis_test.cpp \
    numerical_differentiation_test.cpp \
    correlations_test.cpp \
    stochastic_gradient_descent_test.cpp \
    statistics_test.cpp \
    scaling_test.cpp \
    transformer_test.cpp \
    convolutional_layer_test.cpp \
    pooling_layer_test.cpp \
    response_optimization_test.cpp \
    flatten_layer_test.cpp \
    main.cpp

HEADERS += \
    adaptive_moment_estimation_test.h \
    cross_entropy_error_3d_test.h \
    tensor_utilities_test.h \
    growing_neurons_test.h \
    growing_neurons_test.h \
    unit_testing.h \
    data_set_test.h \
    unscaling_layer_test.h \
    scaling_layer_test.h \
    probabilistic_layer_test.h \
    probabilistic_layer_3d_test.h \
    perceptron_layer_test.h \
    long_short_term_memory_layer_test.h \
    recurrent_layer_test.h \
    neural_network_test.h \
    bounding_layer_test.h \
    sum_squared_error_test.h \
    weighted_squared_error_test.h \
    minkowski_error_test.h \
    mean_squared_error_test.h \
    normalized_squared_error_test.h \
    cross_entropy_error_test.h \
    training_strategy_test.h \
    learning_rate_algorithm_test.h \
    quasi_newton_method_test.h \
    levenberg_marquardt_algorithm_test.h \
    gradient_descent_test.h \
    conjugate_gradient_test.h \
    model_selection_test.h \
    neurons_selection_test.h \
    inputs_selection_test.h \
    growing_inputs_test.h \
    genetic_algorithm_test.h \
    testing_analysis_test.h  \
    numerical_differentiation_test.h \
    opennn_tests.h \
    stochastic_gradient_descent_test.h \
    correlations_test.h \
    statistics_test.h \
    scaling_test.h \
    transformer_test.h \
    convolutional_layer_test.h \
    pooling_layer_test.h \
    flatten_layer_test.h \
    response_optimization_test.h

# OpenMP library

include(../opennmp.pri)


# OpenNN library

win32:CONFIG(release, debug|release): LIBS += -L$$OUT_PWD/../opennn/release/ -lopennn
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OUT_PWD/../opennn/debug/ -lopennn
else:unix: LIBS += -L$$OUT_PWD/../opennn/ -lopennn

INCLUDEPATH += $$PWD/../opennn
DEPENDPATH += $$PWD/../opennn

win32:CONFIG(release, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/release/opennn.lib
else:win32:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$OUT_PWD/../opennn/debug/opennn.lib
else:unix: PRE_TARGETDEPS += $$OUT_PWD/../opennn/libopennn.a

