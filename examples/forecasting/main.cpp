//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   F O R E C A S T I N G   P R O J E C T
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

// OpenNN includes

#include "../../opennn/time_series_dataset.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/loss.h"
#include "../../opennn/training_strategy.h"
#include "adaptive_moment_estimation.h"
#include "quasi_newton_method.h"
#include "stochastic_gradient_descent.h"
#include "testing_analysis.h"
#include "recurrent_layer.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Forecasting Example." << endl;

        //throw runtime_error("Forecasting Example is not yet implemented. Please check back in a future version.");

        // TimeSeriesDataset time_series_dataset("../data/funcion_seno_inputTarget.csv", ",", false, false);
        TimeSeriesDataset time_series_dataset("../data/funcion_seno_missing.csv", ",", false, false);
        // TimeSeriesDataset time_series_dataset("../data/madridNO2forecasting_copy.csv", ",", true, false);
        // TimeSeriesDataset time_series_dataset("../data/madridNO2forecasting.csv", ",", true, false);
        // TimeSeriesDataset time_series_dataset("../data/Pendulum.csv", ",", false, false);
        // TimeSeriesDataset time_series_dataset("../data/twopendulum.csv", ";", false, false);


        cout << "dataset leido" << endl;
        const Index lags = 2;
        time_series_dataset.set_multi_target(false);
        time_series_dataset.set_past_time_steps(lags);
        time_series_dataset.set_future_time_steps(1);
        //time_series_dataset.print();

        // //time_series_dataset.scale_data();

        cout << "-----------------------------------" << endl;

        ///===
        ///===   P R U E B A S   I N T E R P O L A T E    =====
        ///===




        ///===
        ///==   P R U E B A S   T R A I N    ====
        ///===

        // ForecastingNetwork nn({time_series_dataset.get_input_shape()},
        //                        {45},
        //                        {time_series_dataset.get_target_shape()});

        // nn.print();
        // cout << "-----------------------------------" << endl;

        // TrainingStrategy training_strategy(&nn, &time_series_dataset);

        // AdaptiveMomentEstimation* adam = static_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        // adam->set_learning_rate(0.0001);
        // // adam->set_batch_size(1000);
        // adam->set_batch_size(64);
        // adam->set_maximum_epochs(5000);

        // nn.set_parameters_glorot();

        // cout << "Iniciando entrenamiento..." << endl;
        // TrainingResults results = training_strategy.train();

        // cout << "Entrenamiento finalizado." << endl;
        // cout << "Perdida final (MSE): " << results.loss << endl;

        /// Pruebas para t+1
        // cout << "-----------------------------------" << endl;
        // cout << "VALIDACIÓN CON LAG 10" << endl;
        // cout << "-----------------------------------" << endl;

        // const MatrixR& raw_data = time_series_dataset.Dataset::get_data();
        // const Index target_col = time_series_dataset.get_feature_indices("Target")[0];

        // // 1. Validación paso a paso
        // for(Index i = 0; i < 15; ++i) {
        //     Tensor3 input_sample(1, lags, 1);
        //     for(Index l = 0; l < lags; ++l) {
        //         input_sample(0, l, 0) = raw_data(i + l, 0);
        //     }

        //     MatrixR output = nn.calculate_outputs(input_sample);
        //     type real_val = raw_data(i + lags, target_col);
        //     cout << "Real: " << real_val << "\tPred: " << output(0,0) << endl;
        // }

        // // 2. Generación recursiva (Aquí verás la forma del seno)
        // cout << "-----------------------------------" << endl;
        // cout << "PREDICCIÓN RECURSIVA (GENERANDO ONDA)" << endl;
        // Tensor3 rolling_input(1, lags, 1);

        // // Llenamos la semilla con los primeros 10 datos del CSV
        // for(Index l = 0; l < lags; ++l) {
        //     rolling_input(0, l, 0) = raw_data(l, 0);
        // }

        // for(int j = 0; j < 100; ++j) {
        //     MatrixR prediction = nn.calculate_outputs(rolling_input);
        //     type next_val = prediction(0,0);
        //     cout << "Paso t+" << j+1 << ": " << next_val << endl;

        //     // Desplazar ventana: movemos todos una posición atrás y metemos el nuevo al final
        //     for(Index l = 0; l < lags - 1; ++l) {
        //         rolling_input(0, l, 0) = rolling_input(0, l+1, 0);
        //     }
        //     rolling_input(0, lags - 1, 0) = next_val;
        // }

        ///pruebas para t+3
        // cout << "-----------------------------------" << endl;
        // cout << "VALIDACIÓN: PREDICCIÓN DIRECTA A t+3" << endl;
        // cout << "-----------------------------------" << endl;

        // const MatrixR& raw_data = time_series_dataset.Dataset::get_data();
        // const Index target_col = time_series_dataset.get_feature_indices("Target")[0];

        // for(Index i = 0; i < 15; ++i) {
        //     Tensor3 input_sample(1, lags, 1);
        //     for(Index l = 0; l < lags; ++l) {
        //         input_sample(0, l, 0) = raw_data(i + l, 0);
        //     }

        //     MatrixR output = nn.NeuralNetwork::calculate_outputs(input_sample);

        //     // El valor real de t+3 está en: i + lags + (3 - 1)
        //     type real_val = raw_data(i + lags + 2, target_col);
        //     type pred_val = output(0, 0);

        //     cout << "Muestra " << i << " -> Real(t+3): " << real_val << "\tPred(t+3): " << pred_val << endl;
        // }

        // cout << "-----------------------------------" << endl;
        // cout << "GENERACIÓN RECURSIVA (SALTOS DE 3 PASOS)" << endl;
        // Tensor3 rolling_input(1, lags, 1);

        // for(Index l = 0; l < lags; ++l) {
        //     rolling_input(0, l, 0) = raw_data(l, 0);
        // }

        // for(int j = 0; j < 10; ++j) {
        //     MatrixR prediction = nn.NeuralNetwork::calculate_outputs(rolling_input);
        //     type next_val = prediction(0, 0);

        //     cout << "Paso t+" << (j+1)*3 << ": " << next_val << endl;
        //     for(Index l = 0; l < lags - 1; ++l) {
        //         rolling_input(0, l, 0) = rolling_input(0, l+1, 0);
        //     }
        //     rolling_input(0, lags - 1, 0) = next_val;
        // }

        /// Pruebas para varios t
        // cout << "-----------------------------------" << endl;
        // cout << "EJECUTANDO PRUEBA DE 3 SALIDAS" << endl;
        // cout << "-----------------------------------" << endl;

        // // Accedemos a los datos crudos del dataset para comparar
        // const MatrixR& raw_data = time_series_dataset.Dataset::get_data();

        // // Vamos a probar con la Muestra 0
        // Index i = 0;
        // Tensor3 input_sample(1, lags, 1);
        // for(Index l = 0; l < lags; ++l) {
        //     input_sample(0, l, 0) = raw_data(i + l, 0);
        // }

        // // Ejecutar la red: devuelve una fila con 3 columnas
        // MatrixR output = nn.NeuralNetwork::calculate_outputs(input_sample);

        // cout << "Predicción para los próximos 3 pasos:" << endl;
        // for(int s = 0; s < 5; ++s) {
        //     type real_val = raw_data(i + lags + s, 0); // t+1, t+2, t+3
        //     type pred_val = output(0, s);
        //     cout << "  t+" << s+1 << " -> Real: " << real_val << " | Pred: " << pred_val << endl;
        // }
        // cout << "-----------------------------------" << endl;

        // Tensor3 rolling_input(1, lags, 1);

        // // Llenamos el buffer inicial con la "semilla" (los primeros 15 datos reales del CSV)
        // // Usamos raw_data que es la matriz que cargaste antes
        // for(Index l = 0; l < lags; ++l) {
        //     rolling_input(0, l, 0) = raw_data(l, 0);
        // }

        // // --- 2. EL BUCLE DE GENERACIÓN RECURSIVA ---
        // cout << "-----------------------------------" << endl;
        // cout << "GENERACIÓN RECURSIVA (BLOQUES DE 5)" << endl;
        // cout << "-----------------------------------" << endl;

        // for(int j = 0; j < 20; ++j) { // Generamos 20 bloques (100 pasos en total)

        //     // Predicción: la red recibe 15 datos y devuelve un vector de 5
        //     MatrixR prediction = nn.NeuralNetwork::calculate_outputs(rolling_input);

        //     // Mostramos el valor final de este bloque de 5 para ver la evolución
        //     cout << "Bloque t+" << (j+1)*5 << " finaliza en: " << prediction(0, 4) << endl;

        //     // --- 3. ACTUALIZACIÓN DEL BUFFER (Mecanismo de Memoria) ---
        //     // Para que la red siga prediciendo, debemos meter los 5 valores que acaba
        //     // de inventar dentro del buffer de "pasado" (rolling_input).
        //     for(int s = 0; s < 5; ++s) {
        //         // Desplazar todos los valores una posición a la izquierda
        //         for(Index l = 0; l < lags - 1; ++l) {
        //             rolling_input(0, l, 0) = rolling_input(0, l+1, 0);
        //         }
        //         // Insertar el nuevo valor predicho en la última posición
        //         rolling_input(0, lags - 1, 0) = prediction(0, s);
        //     }
        // }
        // cout << "-----------------------------------" << endl;

        ///===
        ///==  I N I C I O   P R U E B A S   ====
        ///===

        // cout << "=== TEST 1: ESTRUCTURA BASICA ===" << endl;

        // // Definimos: 5 pasos de tiempo, 3 entradas por paso, 10 neuronas de salida
        // Index time_steps = 5;
        // Index inputs = 3;
        // Index outputs = 10;

        // Recurrent rnn({time_steps, inputs}, {outputs});

        // // 1. Comprobar formas (Shapes)
        // cout << "Input Shape: " << rnn.get_input_shape()[0] << "x" << rnn.get_input_shape()[1]
        //      << " (Esperado: 5x3)" << endl;

        // cout << "Output Shape: " << rnn.get_output_shape()[0]
        //      << " (Esperado: 10)" << endl;

        // // 2. Comprobar que los tensores de pesos existen y tienen el tamaño correcto
        // vector<TensorView*> params = rnn.get_parameter_views();

        // // params[0] son los Biases (deben ser = outputs)
        // cout << "Biases size: " << params[0]->shape.count()
        //      << " (Esperado: 10)" << endl;

        // // params[1] son los Input Weights (deben ser inputs * outputs)
        // cout << "Input Weights size: " << params[1]->shape.count()
        //      << " (Esperado: 30)" << endl;

        // // params[2] son los Recurrent Weights (ESTE ES EL CORAZON DE LA RECURRENCIA)
        // // Debe ser outputs * outputs (porque cada neurona se conecta con todas las del paso anterior)
        // cout << "Recurrent Weights size: " << params[2]->shape.count()
        //      << " (Esperado: 100)" << endl;

        // if(params[2]->shape.count() != (outputs * outputs)) {
        //     cout << "❌ ERROR: Los pesos recurrentes no estan bien dimensionados." << endl;
        // } else {
        //     cout << "✅ Pesos correctamente dimensionados." << endl;
        // }

        // cout << "\n=== TEST 2: LOGICA DE REDIMENSIONADO ===" << endl;
        // Recurrent rnn; // Capa vacia

        // rnn.set({2, 2}, {4}); // 2 steps, 2 inputs, 4 neurons

        // auto params = rnn.get_parameter_views();
        // cout << "Recurrent Weights despues de set: " << params[2]->shape.count()
        //      << " (Esperado: 16)" << endl;

        // // Intentamos cambiar solo el numero de neuronas de salida
        // rnn.set_output_shape({8});
        // params = rnn.get_parameter_views();
        // cout << "Recurrent Weights despues de cambiar salida a 8: " << params[2]->shape.count()
        //      << " (Esperado: 64)" << endl;

        // cout << "\n=== TEST 3: FLUJO DE MEMORIA (RECURRENCIA) ===" << endl;

        // Recurrent rnn({2, 1}, {1});
        // rnn.set_activation_function("Linear");

        // // Pesos manuales (Biases=0, In=1, Rec=1)
        // vector<TensorView*> params = rnn.get_parameter_views();
        // MatrixR b(1, 1); b.setZero();
        // MatrixR w_in(1, 1); w_in.fill(1.0f);
        // MatrixR w_rec(1, 1); w_rec.fill(1.0f);

        // params[0]->data = b.data();
        // params[1]->data = w_in.data();
        // params[2]->data = w_rec.data();

        // // Entrada: t0=1.0, t1=0.0
        // MatrixR input_data(2, 1);
        // input_data(0, 0) = 1.0f;
        // input_data(1, 0) = 0.0f;

        // TensorView input_view;
        // input_view.data = input_data.data();
        // input_view.shape = {1, 2, 1}; // {Batch=1, Steps=2, Inputs=1}

        // // Preparar propagación
        // unique_ptr<LayerForwardPropagation> fp = make_unique<RecurrentForwardPropagation>(1, &rnn);

        // // --- CORRECCIÓN CRUCIAL: Añadir la entrada ---
        // fp->inputs.push_back(input_view);

        // fp->initialize();

        // // Gestión de memoria de workspace (como ya lo tenías)
        // vector<TensorView*> ws = fp->get_workspace_views();
        // vector<unique_ptr<MatrixR>> ws_mem;
        // for(size_t i = 0; i < ws.size(); ++i) {
        //     auto aligned_buffer = make_unique<MatrixR>((Index)ws[i]->shape.count(), 1);
        //     aligned_buffer->setZero();
        //     ws[i]->data = aligned_buffer->data();
        //     ws_mem.push_back(std::move(aligned_buffer));
        // }

        // try {
        //     rnn.forward_propagate(fp, false);

        //     RecurrentForwardPropagation* rfp = static_cast<RecurrentForwardPropagation*>(fp.get());
        //     MatrixMap output_map = matrix_map(rfp->outputs);

        //     cout << "Entrada en t=0: 1.0, Entrada en t=1: 0.0" << endl;
        //     cout << "Resultado final en t=1: " << output_map(0, 0) << endl;

        //     // Lógica esperada:
        //     // t=0: h = linear(1*1 + 0*1 + 0) = 1.0
        //     // t=1: h = linear(0*1 + 1.0*1 + 0) = 1.0 (el 1.0 viene del paso anterior)
        //     if (std::abs(output_map(0, 0) - 1.0f) < 0.001f) {
        //         cout << "✅ EXITO: La capa mantiene el estado del paso anterior." << endl;
        //     } else {
        //         cout << "❌ FALLO: Resultado inesperado: " << output_map(0, 0) << endl;
        //     }
        // } catch (const std::exception& e) {
        //     cout << "Error durante la ejecución: " << e.what() << endl;
        // }

        // cout << "\n=== TEST 4: BATCH (2) Y NEURONAS (2) ===" << endl;

        // Index batch_size = 2;
        // Index time_steps = 3;
        // Index inputs = 2;
        // Index outputs = 2;

        // Recurrent rnn({time_steps, inputs}, {outputs});
        // rnn.set_activation_function("Linear");

        // // 1. Configurar Pesos (Matrices Identidad)
        // // Usamos MatrixR locales para que gestionen la memoria durante el test
        // vector<TensorView*> params = rnn.get_parameter_views();

        // MatrixR b(outputs, 1); b.setZero();
        // MatrixR w_in(inputs, outputs); w_in.setIdentity();    // Identidad: in0->out0, in1->out1
        // MatrixR w_rec(outputs, outputs); w_rec.setIdentity(); // Identidad: memoria simple

        // params[0]->data = b.data();
        // params[1]->data = w_in.data();
        // params[2]->data = w_rec.data();

        // // 2. Crear entrada para 2 ejemplos (Batch)
        // MatrixR input_buffer(batch_size * time_steps * inputs, 1);
        // input_buffer.setZero();

        // TensorView input_view;
        // input_view.shape = {batch_size, time_steps, inputs};
        // input_view.data = input_buffer.data();

        // // Mapeamos para asignar valores fácilmente
        // TensorMap3 input_map(input_buffer.data(), batch_size, time_steps, inputs);

        // // Batch 0: Solo activamos la neurona 0 en t=0.
        // // Esperamos que el "1" viaje por la memoria hasta el final.
        // input_map(0, 0, 0) = 1.0f;

        // // Batch 1: Activamos la neurona 1 en todos los pasos (t=0, 1, 2).
        // // Esperamos que se acumulen: 1 (t0) + 1 (t1) + 1 (t2) = 3.0
        // input_map(1, 0, 1) = 1.0f;
        // input_map(1, 1, 1) = 1.0f;
        // input_map(1, 2, 1) = 1.0f;

        // // 3. Preparar propagación
        // unique_ptr<LayerForwardPropagation> fp = make_unique<RecurrentForwardPropagation>(batch_size, &rnn);

        // // --- CORRECCIÓN: AÑADIR LA ENTRADA AL VECTOR ---
        // fp->inputs.push_back(input_view);

        // fp->initialize();

        // // Workspace alineado
        // vector<TensorView*> ws = fp->get_workspace_views();
        // vector<unique_ptr<MatrixR>> ws_mem;
        // for(size_t i = 0; i < ws.size(); ++i) {
        //     auto aligned_buffer = make_unique<MatrixR>((Index)ws[i]->shape.count(), 1);
        //     aligned_buffer->setZero();
        //     ws[i]->data = aligned_buffer->data();
        //     ws_mem.push_back(std::move(aligned_buffer));
        // }

        // // 4. Ejecutar
        // try {
        //     rnn.forward_propagate(fp, false);

        //     // 5. Analizar Resultados
        //     RecurrentForwardPropagation* rfp = static_cast<RecurrentForwardPropagation*>(fp.get());

        //     // rfp->outputs tiene forma {batch_size, outputs}
        //     MatrixMap output_map(rfp->outputs.data, batch_size, outputs);

        //     cout << "Batch 0 - Neurona 0 (Esperado 1.0): " << output_map(0, 0) << endl;
        //     cout << "Batch 0 - Neurona 1 (Esperado 0.0): " << output_map(0, 1) << endl;
        //     cout << "Batch 1 - Neurona 0 (Esperado 0.0): " << output_map(1, 0) << endl;
        //     cout << "Batch 1 - Neurona 1 (Esperado 3.0): " << output_map(1, 1) << endl;

        //     // Verificación
        //     bool b0_ok = (std::abs(output_map(0,0) - 1.0f) < 1e-5);
        //     bool b1_ok = (std::abs(output_map(1,1) - 3.0f) < 1e-5);

        //     if (b0_ok && b1_ok) {
        //         cout << "✅ EXITO: El procesamiento por lotes (Batch) y la acumulacion funcionan." << endl;
        //     } else {
        //         cout << "❌ FALLO: Los resultados no coinciden con la logica recurrente." << endl;
        //     }
        // }
        // catch (const std::exception& e) {
        //     cout << "Error: " << e.what() << endl;
        // }
        // cout << "\n=== TEST 5: GRADIENTES (BACKPROPAGATION) ===" << endl;

        // Index batch_size = 1;
        // Index time_steps = 2;
        // Index inputs = 1;
        // Index outputs = 1;

        // Recurrent rnn({time_steps, inputs}, {outputs});
        // rnn.set_activation_function("Linear");

        // // 1. Configurar Pesos a 1.0
        // vector<TensorView*> params = rnn.get_parameter_views();
        // MatrixR b_mat(outputs, 1); b_mat.setZero();
        // MatrixR win_mat(inputs, outputs); win_mat.fill(1.0f);
        // MatrixR wrec_mat(outputs, outputs); wrec_mat.fill(1.0f);

        // params[0]->data = b_mat.data();
        // params[1]->data = win_mat.data();
        // params[2]->data = wrec_mat.data();

        // // 2. Input [1.0] en t=0 y [1.0] en t=1
        // MatrixR input_buf(batch_size * time_steps * inputs, 1);
        // input_buf.fill(1.0f);
        // TensorView input_view;
        // input_view.shape = {batch_size, time_steps, inputs};
        // input_view.data = input_buf.data();

        // // 3. Preparar FORWARD
        // unique_ptr<LayerForwardPropagation> fp = make_unique<RecurrentForwardPropagation>(batch_size, &rnn);
        // fp->inputs.push_back(input_view);
        // fp->initialize();

        // // Workspace del Forward
        // vector<TensorView*> ws = fp->get_workspace_views();
        // vector<unique_ptr<MatrixR>> ws_mem;
        // for(auto* v : ws) {
        //     auto buf = make_unique<MatrixR>((Index)v->shape.count(), 1);
        //     buf->setZero();
        //     v->data = buf->data();
        //     ws_mem.push_back(std::move(buf));
        // }
        // rnn.forward_propagate(fp, true);

        // // 4. Preparar BACKWARD
        // unique_ptr<LayerBackPropagation> bp = make_unique<RecurrentBackPropagation>(batch_size, &rnn);
        // bp->initialize();

        // // --- PARCHE DE MEMORIA PARA GRADIENTES DE PARÁMETROS ---
        // vector<TensorView*> gv = bp->get_gradient_views();
        // vector<unique_ptr<MatrixR>> gv_mem;
        // for(auto* v : gv) {
        //     auto buf = make_unique<MatrixR>((Index)v->shape.count(), 1);
        //     buf->setZero();
        //     v->data = buf->data();
        //     gv_mem.push_back(std::move(buf));
        // }

        // // --- SOLUCIÓN DEFINITIVA AL CRASH (Asignar memoria al gradiente de entrada) ---
        // // Esto evita que rnn.back_propagate escriba en un puntero nulo (nullptr)
        // MatrixR input_grad_mem(batch_size * time_steps * inputs, 1);
        // input_grad_mem.setZero();
        // bp->input_gradients[0].data = input_grad_mem.data();

        // // 5. Gradiente de salida = 1.0
        // MatrixR grad_out_mat(batch_size, outputs);
        // grad_out_mat.fill(1.0f);
        // TensorView grad_out_view;
        // grad_out_view.shape = {batch_size, outputs};
        // grad_out_view.data = grad_out_mat.data();

        // // 6. Ejecutar
        // try {
        //     rnn.back_propagate({input_view}, {grad_out_view}, fp, bp);

        //     RecurrentBackPropagation* rbp = static_cast<RecurrentBackPropagation*>(bp.get());
        //     MatrixMap dW_in = matrix_map(rbp->input_weight_gradients);
        //     MatrixMap dW_rec = matrix_map(rbp->recurrent_weight_gradients);

        //     cout << "Gradiente dW_in  (Esperado: 2.0): " << dW_in(0, 0) << endl;
        //     cout << "Gradiente dW_rec (Esperado: 1.0): " << dW_rec(0, 0) << endl;

        //     if (std::abs(dW_in(0,0) - 2.0f) < 0.01f && std::abs(dW_rec(0,0) - 1.0f) < 0.01f) {
        //         cout << "✅ EXITO: Gradientes calculados correctamente." << endl;
        //     } else {
        //         cout << "❌ FALLO: Gradientes incorrectos." << endl;
        //     }
        // }
        // catch (const std::exception& e) {
        //     cout << "Error: " << e.what() << endl;
        // }

        // cout << "\n=== TEST 6: PERSISTENCIA (XML) ===" << endl;

        // Recurrent rnn_original({10, 5}, {8});
        // rnn_original.set_activation_function("Sigmoid");
        // rnn_original.set_label("capa_recurrente_test");

        // XMLDocument doc;
        // XMLPrinter printer;
        // rnn_original.to_XML(printer);
        // doc.Parse(printer.CStr());

        // Recurrent rnn_loaded;
        // try {
        //     rnn_loaded.from_XML(doc);

        //     cout << "Label cargado: " << rnn_loaded.get_label() << " (Esperado: capa_recurrente_test)" << endl;
        //     cout << "Activacion cargada: " << rnn_loaded.get_activation_function() << " (Esperado: Sigmoid)" << endl;
        //     cout << "Neurons Number cargado: " << rnn_loaded.get_output_shape()[0] << " (Esperado: 8)" << endl;

        //     if(rnn_loaded.get_activation_function() == "Sigmoid" && rnn_loaded.get_output_shape()[0] == 8) {
        //         cout << "✅ EXITO: Serializacion XML correcta." << endl;
        //     } else {
        //         cout << "❌ FALLO: Los datos cargados no coinciden." << endl;
        //     }
        // } catch (const std::exception& e) {
        //     cout << "Error en XML: " << e.what() << endl;
        // }

        ///===
        ///  P R U E B A S    R E D  =====
        ///===
        // cout << "\n=== TEST: FORECASTING NETWORK MINIMO ===" << endl;

        // try {
        //     ForecastingNetwork net({1, 1}, {1}, {1});

        //     cout << "Red creada y compilada con exito." << endl;

        //     // 2. Crear entrada minimalista: Tensor 3D [Batch=1, Time=1, Feature=1]
        //     Tensor3 input_data(1, 1, 1);
        //     input_data.setConstant(1.0f);

        //     cout << "Ejecutando calculate_outputs..." << endl;

        //     // 3. Calcular salida
        //     // Esto disparará la propagación a través de la RNN y luego la Dense
        //     MatrixR result = net.calculate_outputs(input_data);

        //     cout << "Resultado obtenido: " << result(0, 0) << endl;
        //     cout << "✅ EXITO: La red ha completado el flujo sin errores." << endl;
        // }
        // catch (const std::exception& e) {
        //     cout << "❌ FALLO: " << e.what() << endl;
        // }

        // cout << "\n=== TEST 2: BATCH (2) Y SECUENCIA (3 PASOS) ===" << endl;

        // try {
        //     Index batch_size = 2;
        //     Index steps = 3;
        //     Index inputs = 1;
        //     Index hidden = 5;
        //     Index outputs = 1;

        //     ForecastingNetwork net({steps, inputs}, {hidden}, {outputs});

        //     // 1. Crear entrada Tensor 3D [2, 3, 1]
        //     Tensor3 input_data(batch_size, steps, inputs);
        //     input_data.setZero();

        //     // Ejemplo 0: Una serie de 1.0s
        //     for(Index t = 0; t < steps; t++) input_data(0, t, 0) = 1.0f;

        //     // Ejemplo 1: Una serie de 0.0s
        //     for(Index t = 0; t < steps; t++) input_data(1, t, 0) = 0.0f;

        //     cout << "Ejecutando prediccion para Batch de 2..." << endl;

        //     // 2. Calcular salida
        //     // Debería devolver una MatrixR de [2 x 1] (una salida por cada ejemplo del batch)
        //     MatrixR result = net.calculate_outputs(input_data);

        //     cout << "Resultado Ejemplo 0 (Serie 1.0): " << result(0, 0) << endl;
        //     cout << "Resultado Ejemplo 1 (Serie 0.0): " << result(1, 0) << endl;

        //     // 3. Verificación lógica básica
        //     if (result.rows() == 2 && result(0, 0) != result(1, 0)) {
        //         cout << "✅ EXITO: La red procesa ejemplos independientes en el mismo batch." << endl;
        //     } else {
        //         cout << "❌ FALLO: El batch no se proceso correctamente o los resultados son identicos." << endl;
        //     }
        // }
        // catch (const std::exception& e) {
        //     cout << "❌ FALLO: " << e.what() << endl;
        // }

        // cout << "\n=== DIAGNOSTICO DE CULPABLE ===" << endl;

        // // 2 steps, 1 input, 1 neuron -> Debe tener 3 parámetros (1 bias, 1 win, 1 wrec)
        // ForecastingNetwork net({2, 1}, {1}, {2});

        // cout << "Parametros reportados por la RED: " << net.get_parameters_number() << endl;

        // auto* r = static_cast<opennn::Recurrent*>(net.get_layer(0).get());
        // auto views = r->get_parameter_views();

        // cout << "Size del Bias View: " << views[0]->size() << endl;
        // cout << "Size del Win View: " << views[1]->size() << endl;
        // cout << "Size del Wrec View: " << views[2]->size() << endl;

        // if (net.get_parameters_number() == 0) {
        //     cout << "❌ CULPABLE ENCONTRADO: La red cree que la capa tiene 0 parametros." << endl;
        // }
        // cout << "\n=== ESCANEO EXHAUSTIVO DE ALINEACION (Grid Search) ===" << endl;
        // cout << "Inputs | Hidden | Output | Total Params | Alignment (Bias Size) | W_in(0,0) | W_rec(0,0) | Estado" << endl;
        // cout << "-------|--------|--------|--------------|-----------------------|-----------|------------|--------" << endl;

        // int limit = 5; // Prueba hasta 5 en cada dimension para ver los patrones iniciales

        // for (Index in = 1; in <= limit; ++in)
        // {
        //     for (Index hid = 1; hid <= limit; ++hid)
        //     {
        //         for (Index out = 1; out <= limit; ++out)
        //         {
        //             Index steps = 2; // El tiempo no afecta al numero de pesos, solo al workspace

        //             // Creamos la red con la combinacion actual
        //             ForecastingNetwork net({steps, in}, {hid}, {out});

        //             auto* r = static_cast<opennn::Recurrent*>(net.get_layer(0).get());
        //             auto views = r->get_parameter_views();

        //             Index logical = views[0]->size() + views[1]->size() + views[2]->size();
        //             Index with_padding = net.get_parameters_number();
        //             if (in == 1 && hid == 1 && out == 1)
        //                 printf("Logico: %d, con padding: %d\n", (int)logical, (int)with_padding);

        //             // Llenamos el vector global con 7.0
        //             VectorR p(net.get_parameters_number());
        //             p.setConstant(7.0f);
        //             net.set_parameters(p);

        //             // Mapeamos usando las funciones NATIVAS de OpenNN (las que fallan con desalineacion)
        //             // b_map empieza en vector[0] -> Siempre alineado
        //             // win_map empieza en vector[size_bias] -> ¡Aqui esta el peligro!
        //             Eigen::Map<VectorR, Eigen::Unaligned> b_check(views[0]->data, views[0]->size());
        //             Eigen::Map<MatrixR, Eigen::Unaligned> win_check(views[1]->data, views[1]->shape[0], views[1]->shape[1]);
        //             Eigen::Map<MatrixR, Eigen::Unaligned> wrec_check(views[2]->data, views[2]->shape[0], views[2]->shape[1]);

        //             float v1 = win_check(0, 0);
        //             float v2 = wrec_check(0, 0);

        //             // Verificamos si los datos son correctos
        //             bool is_ok = (std::abs(v1 - 7.0f) < 1e-4 && std::abs(v2 - 7.0f) < 1e-4);

        //             // Imprimir fila
        //             printf("  %2d   |   %2d   |   %2d   |      %3d     |          %2d           | %9.2f | %10.2f | %s\n",
        //                    (int)in, (int)hid, (int)out,
        //                    (int)net.get_parameters_number(),
        //                    (int)views[0]->size(),
        //                    v1, v2,
        //                    is_ok ? "✅" : "❌");
        //         }
        //     }
        // }
        // cout << "-------------------------------------------------------------------------------------------------" << endl;

        // opennn::ForecastingNetwork net({2, 1}, {1}, {1});

        // auto* r = static_cast<opennn::Recurrent*>(net.get_first("Recurrent"));
        // r->set_activation_function("Linear");
        // cout << "a71 -0" <<endl;

        // // Forzar parámetros: [bias, win, wrec] -> [0, 1, 1]
        // VectorR p(3);
        // p(0) = 0.0f; // bias
        // p(1) = 1.0f; // win
        // p(2) = 1.0f; // wrec
        // net.set_parameters(p);

        // // Entrada: 1 batch, 2 steps, 1 feature (todos a 1.0)
        // Tensor3 input(1, 2, 1);
        // input.setConstant(1.0f);

        // MatrixR output = net.calculate_outputs(input);

        // cout << "Resultado Forward: " << output(0,0) << " (Esperado: 2.0)" << endl;
        // if (abs(output(0,0) - 2.0f) < 1e-5) cout << "✅ Forward OK" << endl;
        // else cout << "❌ Forward FALLÓ" << endl;

        // cout << "--- Iniciando Pruebas de Capa Recurrente ---" << endl;

        // // 1. Configuración de dimensiones
        // const Index time_steps = 5;
        // const Index input_features = 3;
        // const Index neurons = 10;
        // const Index batch_size = 2;

        // // 2. Crear la Red Neuronal
        // NeuralNetwork nn;

        // // Crear la capa recurrente
        // // Input shape: {time_steps, input_features}
        // // Output shape: {neurons}
        // unique_ptr<Recurrent> recurrent_layer(new Recurrent({time_steps, input_features}, {neurons}));
        // recurrent_layer->set_activation_function("HyperbolicTangent");

        // nn.add_layer(std::move(recurrent_layer));

        // // 3. Comprobar parámetros
        // // Biases (10) + Input Weights (3*10) + Recurrent Weights (10*10) = 10 + 30 + 100 = 140
        // Index expected_parameters = neurons + (input_features * neurons) + (neurons * neurons);
        // cout << "Parametros esperados: " << expected_parameters << endl;
        // cout << "Parametros calculados por la red: " << nn.get_parameters_number() << endl;

        // if(nn.get_parameters_number() == expected_parameters) {
        //     cout << "[OK] Conteo de parametros correcto." << endl;
        // } else {
        //     cout << "[ERROR] Discrepancia en parametros." << endl;
        // }

        // // 4. Inicializar parámetros (importante para que no sean basura)
        // nn.set_parameters_random();

        // // 5. Probar Propagación hacia delante (Forward Propagation)
        // // Las capas recurrentes esperan un Tensor3: [batch_size, time_steps, input_features]
        // Tensor3 input_data(batch_size, time_steps, input_features);
        // input_data.setRandom();

        // cout << "Calculando salidas para un batch de " << batch_size << "..." << endl;
        // MatrixR outputs = nn.calculate_outputs(input_data);

        // cout << "Dimensiones de la salida: " << outputs.rows() << "x" << outputs.cols() << endl;
        // // La salida debería ser [batch_size, neurons] porque el Recurrent devuelve el último estado oculto
        // if(outputs.rows() == batch_size && outputs.cols() == neurons) {
        //     cout << "[OK] Dimensiones de salida correctas." << endl;
        // } else {
        //     cout << "[ERROR] Dimensiones de salida incorrectas." << endl;
        // }

        // // 6. Probar Serialización XML
        // cout << "Probando guardado y carga XML..." << endl;
        // nn.save("test_recurrent_nn.xml");

        // NeuralNetwork nn_loaded;
        // nn_loaded.load("test_recurrent_nn.xml");

        // if(nn_loaded.get_layers_number() == 1 && nn_loaded.get_parameters_number() == expected_parameters) {
        //     cout << "[OK] Carga de XML exitosa y coherente." << endl;
        // }

        // // 7. Mostrar expresión matemática (opcional)
        // // Esto sirve para verificar que la lógica de la secuencia es correcta
        // // cout << "\nExpresion de la red:" << endl;
        // // cout << nn.get_expression() << endl;

        // cout << "--- Pruebas Finalizadas con Exito ---" << endl;



        ///===
        ///  F I N   P R U E B A S   =====
        ///===

        // if(time_series_dataset.has_nan())
        //     time_series_dataset.impute_missing_values_interpolate();

        // if(time_series_dataset.has_nan())
        //     cout << "sigue habiendo nans" << endl;
        // else
        //     cout << "nans arreglados" << endl;

        // cout << "-----------------------------------" << endl;

        // cout << "input_shape:  " << time_series_dataset.get_input_shape() << endl;
        // cout << "target_shape: " << time_series_dataset.get_target_shape() << endl;

        // ForecastingNetwork forecasting_network({time_series_dataset.get_input_shape()},
        //                                        {1},
        //                                        {time_series_dataset.get_target_shape()});

        // Layer* layer_ptr = forecasting_network.get_first("Recurrent");
        // Recurrent* recurrent_layer = dynamic_cast<Recurrent*>(layer_ptr);
        // recurrent_layer->set_activation_function("HyperbolicTangent");

        // Layer* layer_ptr_scaling = forecasting_network.get_first("Scaling3d");
        // Scaling3d* scaling_layer = dynamic_cast<Scaling3d*>(layer_ptr_scaling);
        // scaling_layer->set_scalers("None");
        // recurrent_layer->set_timesteps(1);

        // forecasting_network.print();

        // cout << "------------------------------------------" << endl;

        /// Pruebas de la red neuronal

        // cout << "=== TEST RED NEURONAL 1 ===" << endl;

        // ForecastingNetwork forecasting_network({time_series_dataset.get_input_shape()},
        //                                        {4},
        //                                        {time_series_dataset.get_target_shape()});

        // VectorR ceros(3);
        // ceros.setZero();
        // forecasting_network.set_parameters(ceros);

        // Tensor3 input_test(1, 2, 1);
        // input_test.setConstant(1.0); // No importa el valor si los pesos son 0

        // MatrixR output = forecasting_network.NeuralNetwork::calculate_outputs(input_test);

        // cout << "Salida con pesos a cero: " << output(0, 0) << endl;
        // // Si usas Tanh, debe dar 0.
        // // Si usas Sigmoid, debe dar 0.5.

        // Index n_params = forecasting_network.get_parameters_number();
        // cout << "La red necesita " << n_params << " parametros." << endl;

        // // 2. Crear un vector de ese tamaño exacto
        // VectorR pesos_uno(n_params);
        // pesos_uno.setConstant(1.0); // Ahora TODOS los pesos (incluido el recurrente) son 1.0

        // forecasting_network.set_parameters(pesos_uno);

        // Tensor3 seqA(1, 2, 1);
        // seqA(0, 0, 0) = 1.0; // t=0
        // seqA(0, 1, 0) = 0.0; // t=1

        // Tensor3 seqB(1, 2, 1);
        // seqB(0, 0, 0) = 0.0; // t=0
        // seqB(0, 1, 0) = 0.0; // t=1

        // auto* r = static_cast<opennn::Recurrent*>(forecasting_network.get_layer(0).get());

        // cout << "Layer input_shape = {"
        //      << r->get_input_shape()[0] << ", "
        //      << r->get_input_shape()[1] << "}\n";

        // cout << "seqA shape = {"
        //      << seqA.dimension(1) << ", "
        //      << seqA.dimension(2) << "}\n";

        // type outA = forecasting_network.NeuralNetwork::calculate_outputs(seqA)(0, 0);
        // type outB = forecasting_network.NeuralNetwork::calculate_outputs(seqB)(0, 0);

        // cout << "Salida A (pasado activo): " << outA << endl;
        // cout << "Salida B (pasado cero): " << outB << endl;

        // if (abs(outA - outB) < 1e-10) {
        //     cout << "ERROR: La recurrencia NO funciona. t=0 no influye en la salida final." << endl;
        // } else {
        //     cout << "OK: La red tiene memoria. El valor pasado se ha propagado al presente." << endl;
        // }

        // cout << "=== TEST AUTOMATICO === 017-a " << endl;

        // // Ya no creamos un vector de tamaño 3 a mano.
        // // Le preguntamos a la red cuánto necesita.
        // Index n = forecasting_network.get_parameters_number();
        // VectorR nuevos_pesos(n);
        // nuevos_pesos.setConstant(1.0);

        // // Esto ahora funcionará perfectamente porque set_parameters llama a link()
        // forecasting_network.set_parameters(nuevos_pesos);

        // // Los cálculos ahora tendrán memoria automáticamente
        // type outA = forecasting_network.calculate_outputs(seqA)(0, 0);
        // type outB = forecasting_network.calculate_outputs(seqB)(0, 0);

        // cout << "Salida A (pasado activo): " << outA << endl;
        // cout << "Salida B (pasado cero): " << outB << endl;

        // cout << "Diferencia: " << abs(outA - outB) << endl;

        /// Pruebas carga de datos multicolumna

        // cout << "=== AUDITORÍA TOTAL: INPUTS + LAGS + AHEAD (SIN DUPLICADOS) ===" << endl;

        // Index mis_lags = 2;
        // Index mis_ahead = 1;

        // time_series_dataset.set_past_time_steps(mis_lags);
        // time_series_dataset.set_future_time_steps(mis_ahead);

        // vector<Index> input_indices = time_series_dataset.get_feature_indices("Input");
        // vector<Index> input_target_indices = time_series_dataset.get_feature_indices("InputTarget");

        // vector<Index> all_input_indices = input_indices;
        // all_input_indices.insert(all_input_indices.end(), input_target_indices.begin(), input_target_indices.end());

        // sort(all_input_indices.begin(), all_input_indices.end());
        // all_input_indices.erase(unique(all_input_indices.begin(), all_input_indices.end()), all_input_indices.end());

        // vector<Index> target_indices = time_series_dataset.get_feature_indices("Target");
        // if(target_indices.empty()) target_indices = input_target_indices;

        // vector<Index> sample_indices = time_series_dataset.get_sample_indices("Training");

        // Tensor3 inputs_tensor(sample_indices.size(), mis_lags, all_input_indices.size());
        // time_series_dataset.fill_inputs(sample_indices, all_input_indices, inputs_tensor.data());

        // MatrixR targets_matrix(sample_indices.size(), target_indices.size());
        // time_series_dataset.fill_targets(sample_indices, target_indices, targets_matrix.data());

        // cout << "Columnas únicas de entrada detectadas: " << all_input_indices.size() << endl;

        // for(Index i = 0; i < 10; i++) {
        //     cout << "\n------------------------------------------------------------------" << endl;
        //     cout << "MUESTRA " << i << endl;
        //     cout << "------------------------------------------------------------------" << endl;

        //     for(Index t = 0; t < mis_lags; t++) {
        //         cout << "  [t-" << (mis_lags - t) << "] -> ";
        //         for(Index v = 0; v < all_input_indices.size(); v++) {
        //             cout << inputs_tensor(i, t, v) << " | ";
        //         }
        //         cout << endl;
        //     }

        //     cout << "  ----------------------------------------------------------------" << endl;
        //     cout << "  [TARGET t+" << mis_ahead << "] -> ";
        //     for(Index v = 0; v < target_indices.size(); v++) {
        //         cout << targets_matrix(i, v) << " | ";
        //     }
        //     cout << endl;
        // }


        /// Pruebas carga de datos funcion_seno_inputTarget

        // cout << "=== PASO 1: AUDITORÍA DE VALORES (CSV vs DATASET) ===" << endl;

        // Tensor3 training_inputs = time_series_dataset.get_data("Training", "InputTarget");

        // vector<Index> training_indices = time_series_dataset.get_sample_indices("Training");
        // vector<Index> target_indices = time_series_dataset.get_feature_indices("Target");
        // MatrixR training_targets(training_indices.size(), target_indices.size());

        // time_series_dataset.fill_targets(training_indices, target_indices, training_targets.data());

        // if (training_inputs.size() == 0 || training_targets.size() == 0) {
        //     cout << "ERROR: Los tensores están vacíos. ¿Llamaste a read_csv()?" << endl;
        // } else {
        //     cout << "Muestras de entrenamiento detectadas: " << training_indices.size() << endl;
        //     cout << fixed << setprecision(6);

        //     for(Index i = 0; i < 20; i++) {
        //         cout << "\n[Ventana " << i << "]" << endl;

        //         type t_minus_2 = training_inputs(i, 0, 0);
        //         type t_minus_1 = training_inputs(i, 1, 0);

        //         type t_future = training_targets(i, 0);

        //         cout << "  Inputs (Pasado): [ " << t_minus_2 << ", " << t_minus_1 << " ]" << endl;
        //         cout << "  Target (Futuro): " << t_future << endl;
        //     }
        // }
        // cout << "===\n" << endl;
        // cout << "=== PRUEBA DE TRAZABILIDAD: LAGS vs STEPS AHEAD ===" << endl;

        // // 1. Definimos los parámetros que queremos probar
        // Index mis_lags = 2;       // Cuántos pasos atrás miramos
        // Index mis_steps = 4;      // A qué distancia está el objetivo (1 = el siguiente inmediato)

        // time_series_dataset.set_past_time_steps(mis_lags);
        // time_series_dataset.set_future_time_steps(mis_steps);

        // vector<Index> training_indices0 = time_series_dataset.get_sample_indices("Training");
        // vector<Index> target_indices0 = time_series_dataset.get_feature_indices("Target");

        // Tensor3 inputs = time_series_dataset.get_data("Training", "InputTarget");
        // MatrixR targets(training_indices0.size(), target_indices0.size());
        // time_series_dataset.fill_targets(training_indices0, target_indices0, targets.data());

        // cout << "Configuración actual -> Lags: " << mis_lags << ", Steps Ahead: " << mis_steps << endl;

        // for(Index i = 0; i < 10; i++) {
        //     cout << "\n[Muestra " << i << "]" << endl;

        //     cout << "  Inputs (Pasado): ";
        //     for(Index l = 0; l < mis_lags; l++) {
        //         cout << "t-" << (mis_lags - l) << ":[" << inputs(i, l, 0) << "] ";
        //     }
        //     cout << endl;
        //     cout << "  Target (Futuro t+" << mis_steps << "): " << targets(i, 0) << endl;

        //     if (mis_steps == 1 && i < 1) {
        //         if (abs(targets(i, 0) - inputs(i+1, mis_lags-1, 0)) < 1e-7) {
        //             cout << "  [OK] El target es el paso inmediatamente siguiente (1-step ahead)." << endl;
        //         }
        //     }
        // }


        /// Calcular gradiente
        // NormalizedSquaredError normalized_squared_error(&forecasting_network, &time_series_dataset);

        // const VectorR gradient = normalized_squared_error.calculate_gradient();
        // const VectorR numerical_gradient = normalized_squared_error.calculate_numerical_gradient();

        // cout << "Gradient" << endl;
        // cout << gradient << endl;
        // cout << "Numerical Gradient" << endl;
        // cout << numerical_gradient << endl;
        // cout << "diferencia" << endl;
        // cout << gradient.abs() - numerical_gradient.abs() << endl;

        // cout << "------------------------------------------" << endl;

        /// Entrenamiento
        // TrainingStrategy training_strategy(&forecasting_network, &time_series_dataset);
        // training_strategy.set_loss("MeanSquaredError");

        // // training_strategy.set_optimization_algorithm("QuasiNewtonMethod");
        // // training_strategy.set_optimization_algorithm("StochasticGradientDescent");

        // AdaptiveMomentEstimation* adam = static_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        // // adam->set_learning_rate(0.0001);
        // adam->set_batch_size(1000);
        // adam->set_maximum_epochs(10000);

        // QuasiNewtonMethod* quasi = static_cast<QuasiNewtonMethod*>(training_strategy.get_optimization_algorithm());
        // quasi->set_loss_goal(0.001);

        // StochasticGradientDescent* stochastic = static_cast<StochasticGradientDescent*>(training_strategy.get_optimization_algorithm());
        // stochastic->set_batch_size(10000);

        // training_strategy.train();

        // cout << "Error: " << normalized_squared_error.calculate_numerical_error() << endl;

        // cout << "------------------------------------------" << endl;

        /// Testing analysis
        // TestingAnalysis testing_analysis(&forecasting_network, &time_series_dataset);
        // cout << "Goodness of fit analysis: " << endl;
        // testing_analysis.print_goodness_of_fit_analysis();

        /// Pruebas output funcion seno 1 variable
        // Tensor3 test_input(1, 2, 1);
        // test_input(0, 0, 0) = 0.0;
        // test_input(0, 1, 0) = 0.099833417;

        // // Ejecutar predicción
        // MatrixR prediction = forecasting_network.calculate_outputs(static_cast<const Tensor3&>(test_input));

        // cout << "--- COMPROBACIÓN ---" << endl;
        // cout << "adri71" << endl;
        // cout << "Entrada pasada: " << test_input << endl;
        // cout << "Predicción de la red: " << prediction(0, 0) << endl;
        // cout << "Valor real esperado: 0.1986" << endl;
        // cout << "Error absoluto: " << abs(prediction(0, 0) - 0.198669331) << endl;


        /// Pruebas output
        // Tensor3 inputs(1,2,2);
        // inputs.setValues({
        //     {
        //         {1.76624, 1.41520},
        //         {2.11640, 1.80730},
        //        // {1.2405,  2.1018}
        //     }
        // });
        // cout << "Inputs: \n" << inputs << endl;
        // const MatrixR outputs = forecasting_network.calculate_outputs(inputs);
        // cout << "outputs: " << outputs << endl;

        /// Pruebas output funcion seno
        // const vector<std::pair<type, type>> input_views = {
        //     {0.0,          0.0998334166},
        //     {0.0998334166, 0.1986693308},
        //     {0.198669331,  0.295520207},
        //     {0.295520207,  0.389418342},
        //     {0.389418342,  0.479425539},
        //     {0.479425539,  0.564642473},
        //     {0.564642473,  0.644217687},
        //     {0.644217687,  0.717356091},
        //     {0.717356091,  0.78332691},
        //     {0.78332691,   0.841470985},
        //     {0.841470985,  0.89120736}
        // };

        // for(size_t i = 0; i < input_views.size() - 1; ++i)
        // {
        //     const auto& current_pair = input_views[i];
        //     const type input_val_1 = current_pair.first;
        //     const type input_val_2 = current_pair.second;

        //     cout << "\n--- Prueba " << i + 1 << " ---" << endl;
        //     cout << "Inputs: [ " << input_val_1 << ", " << input_val_2 << " ]" << endl;

        //     Tensor3 inputs(1, 2, 1); //batch, time, input
        //     inputs.setValues({{{input_val_1}, {input_val_2}}});
        //     const Tensor2 outputs = forecasting_network.calculate_outputs<3,2>(inputs);

        //     cout << "Output: " << outputs << endl;
        //     cout << "Target: " << input_views[i+1].second << endl;
        // }

        cout << "Good bye!" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
