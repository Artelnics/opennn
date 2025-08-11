//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A M A Z O N   R E V I E W S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

#include "../../opennn/language_dataset.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/normalized_squared_error.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"

#include "../../opennn/adaptive_moment_estimation.h"
#include "mean_squared_error.h"
#include "multihead_attention_layer.h"

template<typename T>
void matmul_por_lotes_bucles(const Tensor<T, 4>& A, const Tensor<T, 4>& B, Tensor<T, 4>& C) {
    const Index batch_size = A.dimension(0);
    const Index num_heads = A.dimension(1);
    const Index seq_len_a = A.dimension(2);
    const Index depth = A.dimension(3);
    const Index seq_len_b = B.dimension(2);

    C.setZero();

#pragma omp parallel for collapse(2)
    for (Index b = 0; b < batch_size; ++b) {
        for (Index h = 0; h < num_heads; ++h) {
            for (Index i = 0; i < seq_len_a; ++i) {
                for (Index j = 0; j < seq_len_b; ++j) {
                    T sum = static_cast<T>(0);
                    for (Index d = 0; d < depth; ++d) {
                        sum += A(b, h, i, d) * B(b, h, j, d);
                    }
                    C(b, h, i, j) = sum;
                }
            }
        }
    }
}

// template<typename T>
// void matmul_por_lotes_eigen(const Tensor<T, 4>& Q, const Tensor<T, 4>& K, Tensor<T, 4>& C) {
//     const Index batch_size = Q.dimension(0);
//     const Index num_heads = Q.dimension(1);
//     const Index seq_len_q = Q.dimension(2);
//     const Index depth = Q.dimension(3);
//     const Index seq_len_k = K.dimension(2);

//     Tensor<T, 4> K_permuted = K.shuffle(Eigen::array<int, 4>({0, 1, 3, 2}));

//     Eigen::array<Eigen::IndexPair<int>, 2> contraction_dims;
//     contraction_dims[0] = Eigen::IndexPair<int>(3, 2);
//     contraction_dims[1] = Eigen::IndexPair<int>(4, 5);

//     Tensor<T, 6> Q_reshaped = Q.reshape(Eigen::array<Index, 6>({batch_size, num_heads, seq_len_q, 1, depth, 1}));
//     Tensor<T, 6> K_reshaped = K_permuted.reshape(Eigen::array<Index, 6>({batch_size, num_heads, 1, depth, 1, seq_len_k}));

//     Eigen::array<Index, 6> bcast_Q = {1, 1, 1, 1, 1, seq_len_k};
//     Eigen::array<Index, 6> bcast_K = {1, 1, seq_len_q, 1, 1, 1};

//     Tensor<T, 6> Q_broadcast = Q_reshaped.broadcast(bcast_Q);
//     Tensor<T, 6> K_broadcast = K_reshaped.broadcast(bcast_K);

//     Tensor<T, 6> product = Q_broadcast * K_broadcast;

//     Eigen::array<Index, 1> reduction_dims = {4};
//     C = product.sum(reduction_dims).reshape(Eigen::array<Index, 4>({batch_size, num_heads, seq_len_q, seq_len_k}));
// }

// template<typename T>
// void matmul_por_lotes_tensor(const Tensor<T, 4>& Q, const Tensor<T, 4>& K, Tensor<T, 4>& C) {
//     using Eigen::array;
//     using Eigen::Index;
//     using Eigen::IndexPair;
//     using Eigen::TensorMap;

//     const Index batch_size = Q.dimension(0);
//     const Index num_heads = Q.dimension(1);
//     const Index seq_len_q = Q.dimension(2);
//     const Index depth = Q.dimension(3);
//     const Index seq_len_k = K.dimension(2);

//     C.resize(batch_size, num_heads, seq_len_q, seq_len_k);

//     array<IndexPair<Index>, 1> contract_dims = {IndexPair<Index>(1, 0)};
//     array<Index, 2> shuffle_dims = {1, 0};

// #pragma omp parallel for collapse(2)
//     for (Index b = 0; b < batch_size; ++b) {
//         for (Index h = 0; h < num_heads; ++h) {
//             const Index q_offset = ((b * num_heads + h) * seq_len_q) * depth;
//             const Index k_offset = ((b * num_heads + h) * seq_len_k) * depth;
//             const Index c_offset = ((b * num_heads + h) * seq_len_q) * seq_len_k;

//             TensorMap<const Tensor<T, 2>> q_slice(Q.data() + q_offset, seq_len_q, depth);
//             TensorMap<const Tensor<T, 2>> k_slice(K.data() + k_offset, seq_len_k, depth);
//             TensorMap<Tensor<T, 2>> c_slice(C.data() + c_offset, seq_len_q, seq_len_k);

//             c_slice = q_slice.contract(k_slice.shuffle(shuffle_dims), contract_dims);
//         }
//     }
// }

// template<typename T>
// void matmul_por_lotes_tensor(const Tensor<T, 4>& Q, const Tensor<T, 4>& K, Tensor<T, 4>& C) {
//     using Eigen::array;
//     using Eigen::Index;
//     using Eigen::IndexPair;
//     using Eigen::TensorMap;

//     const Index batch_size = Q.dimension(0);
//     const Index num_heads = Q.dimension(1);
//     const Index seq_len_q = Q.dimension(2);
//     const Index depth = Q.dimension(3);
//     const Index seq_len_k = K.dimension(2);

//     C.resize(batch_size, num_heads, seq_len_q, seq_len_k);

//     array<IndexPair<Index>, 1> contract_dims = {IndexPair<Index>(1, 0)};
//     array<Index, 2> shuffle_dims = {1, 0};

// #pragma omp parallel for collapse(2)
//     for (Index b = 0; b < batch_size; ++b) {
//         for (Index h = 0; h < num_heads; ++h) {
//             const Index q_offset = b * (num_heads * seq_len_q * depth) + h * (seq_len_q * depth);
//             const Index k_offset = b * (num_heads * seq_len_k * depth) + h * (seq_len_k * depth);
//             const Index c_offset = b * (num_heads * seq_len_q * seq_len_k) + h * (seq_len_q * seq_len_k);

//             TensorMap<const Tensor<T, 2>> q_slice(Q.data() + q_offset, seq_len_q, depth);
//             TensorMap<const Tensor<T, 2>> k_slice(K.data() + k_offset, seq_len_k, depth);
//             TensorMap<Tensor<T, 2>> c_slice(C.data() + c_offset, seq_len_q, seq_len_k);

//             c_slice = q_slice.contract(k_slice.shuffle(shuffle_dims), contract_dims);
//         }
//     }
// }

template<typename T>
void matmul_por_lotes_tensor(const Tensor<T, 4>& Q, const Tensor<T, 4>& K, Tensor<T, 4>& C) {
    using Eigen::array;
    using Eigen::Index;

    const Index batch_size = Q.dimension(0);
    const Index num_heads = Q.dimension(1);
    const Index seq_len_q = Q.dimension(2);
    const Index depth = Q.dimension(3);
    const Index seq_len_k = K.dimension(2);

    C.resize(batch_size, num_heads, seq_len_q, seq_len_k);

    array<Index, 5> q_reshape_dims = {batch_size, num_heads, seq_len_q, 1, depth};
    array<Index, 5> k_reshape_dims = {batch_size, num_heads, 1, seq_len_k, depth};

    array<Index, 5> q_broadcast_dims = {1, 1, 1, seq_len_k, 1};
    array<Index, 5> k_broadcast_dims = {1, 1, seq_len_q, 1, 1};

    auto Q_expanded = Q.reshape(q_reshape_dims).broadcast(q_broadcast_dims);
    auto K_expanded = K.reshape(k_reshape_dims).broadcast(k_broadcast_dims);

    auto product = Q_expanded * K_expanded;

    array<Index, 1> reduction_dims = {4};
    C = product.sum(reduction_dims);
}


template<typename T>
void print_20_elementos(const Eigen::Tensor<T, 4>& tensor) {
    using Eigen::Index;

    const Index dim0 = tensor.dimension(0);
    const Index dim1 = tensor.dimension(1);
    const Index dim2 = tensor.dimension(2);
    const Index dim3 = tensor.dimension(3);

    Index count = 0;

    for (Index i = 1; i < dim0; ++i) {
        for (Index j = 0; j < dim1; ++j) {
            for (Index k = 0; k < dim2; ++k) {
                for (Index l = 0; l < dim3; ++l) {
                    std::cout << "tensor(" << i << "," << j << "," << k << "," << l << ") = "
                              << tensor(i, j, k, l) << std::endl;
                    if (++count >= 20) return;
                }
            }
        }
    }
}

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Amazon reviews example." << endl;

        //         // const Index batch_size = 4;
        //         // const Index num_heads  = 2;
        //         // const Index seq_len_a  = 8;
        //         // const Index seq_len_b  = 8;
        //         // const Index depth      = 16;

        //         const Index batch_size = 2;
        //         const Index num_heads  = 2;
        //         const Index seq_len_a  = 4;
        //         const Index seq_len_b  = 4;
        //         const Index depth      = 10;


        //         Tensor<type, 4> A(batch_size, num_heads, seq_len_a, depth);
        //         Tensor<type, 4> B(batch_size, num_heads, seq_len_b, depth);
        //         Tensor<type, 4> C(batch_size, num_heads, seq_len_a, seq_len_b);
        //         Tensor<type, 4> E(batch_size, num_heads, seq_len_a, seq_len_b);
        //         Tensor<type, 4> D(batch_size, num_heads, seq_len_a, seq_len_b);

        //         Tensor<type, 4> V(batch_size, num_heads, seq_len_b, depth);

        //         for (Index b = 0; b < batch_size; ++b) {
        //             for (Index h = 0; h < num_heads; ++h) {
        //                 for (Index s = 0; s < seq_len_a; ++s) {
        //                     for (Index d = 0; d < depth; ++d) {
        //                         Index flat_index = ((b * num_heads + h) * seq_len_a + s) * depth + d;
        //                         A(b, h, s, d) = static_cast<type>(flat_index + 1);
        //                     }
        //                 }
        //             }
        //         }

        //         for (Index b = 0; b < batch_size; ++b) {
        //             for (Index h = 0; h < num_heads; ++h) {
        //                 for (Index s = 0; s < seq_len_b; ++s) {
        //                     for (Index d = 0; d < depth; ++d) {
        //                         Index flat_index = ((b * num_heads + h) * seq_len_b + s) * depth + d;
        //                         B(b, h, s, d) = static_cast<type>(flat_index + 1);
        //                     }
        //                 }
        //             }
        //         }

        //         for (Index b = 0; b < batch_size; ++b) {
        //             for (Index h = 0; h < num_heads; ++h) {
        //                 for (Index s = 0; s < seq_len_b; ++s) {
        //                     for (Index d = 0; d < depth; ++d) {
        //                         Index flat_index = ((b * num_heads + h) * seq_len_b + s) * depth + d;
        //                         V(b, h, s, d) = static_cast<type>(flat_index + 1);
        //                     }
        //                 }
        //             }
        //         }

        //         matmul_por_lotes_bucles(A, B, C);

        //         cout << "C(0, 0, 0, 0)" << C(0, 0, 0, 0) << endl;
        //         cout << "C(1, 2, 3, 3)" << C(1, 1, 2, 3) << endl;

        //         cout << C.dimensions() << endl;
        //         cout << V.dimensions() << endl;

        //         Tensor<type, 4> F(batch_size, num_heads, seq_len_a, depth);
        //         // matmul_por_lotes_tensor(C, V, F);

        //         cout << "F.dimensions" << F.dimensions() << endl;

        //         F = (C.reshape(array_5(batch_size, num_heads, seq_len_a, seq_len_b, 1)).broadcast(array_5(1, 1, 1, 1, depth))
        //              * V.reshape(array_5(batch_size, num_heads, 1, seq_len_b, depth)).broadcast(array_5(1, 1, seq_len_a, 1, 1)))
        //                 .sum(array_1(3));

        //         cout << "F.dimensions" << F.dimensions() << endl;
        //         cout << "F(0, 0, 0, 0)" << F(0, 0, 0, 0) << endl;
        //         cout << "F(1, 2, 3, 3)" << F(1, 1, 2, 3) << endl;

        //         // Tensor<type, 3> result(batch_size, seq_len_a, num_heads * depth);
        //         // cout << "result: " << result.dimensions() <<  endl;

        //         // auto transposed = F.shuffle(array_4(0, 2, 1, 3)).eval();
        //         // result = transposed.reshape(array_3(batch_size, seq_len_a, num_heads * depth));

        //         // result.resize(batch_size, seq_len_a, num_heads * depth);
        //         // for(int b = 0; b < batch_size; ++b) {
        //         //     for(int s = 0; s < seq_len_a; ++s) {
        //         //         for(int h = 0; h < num_heads; ++h) {
        //         //             for(int d = 0; d < depth; ++d) {
        //         //                 result(b, s, h * depth + d) = F(b, h, s, d);
        //         //             }
        //         //         }
        //         //     }
        //         // }

        //         Tensor<type, 3> result(batch_size, seq_len_a, num_heads * depth);

        //         // for(int h = 0; h < num_heads; ++h) {
        //         //     // chip() es muy eficiente para extraer una "rebanada"
        //         //     auto head_slice = F.chip(h, 1); // Extraer head h: (batch, seq, depth)

        //         //     // Hacer transpose de seq y depth usando shuffle
        //         //     auto transposed_head = head_slice.shuffle(array_3(0, 2, 1)); // (batch, depth, seq)

        //         //     // Colocar en la posición correcta
        //         //     Eigen::array<Index, 3> offsets = {0, 0, h * depth};
        //         //     Eigen::array<Index, 3> extents = {batch_size, seq_len_a, depth};
        //         //     result.slice(offsets, extents) = transposed_head.shuffle(array_3(0, 2, 1));
        //         // }
        //         // auto F_shuffled = F.shuffle(array_4(0, 2, 1, 3));  // (B, L, H, D)
        //         // auto F_flat = F_shuffled.reshape(array_3(batch_size, seq_len_a, num_heads * depth));
        //         // result = F_flat;


        // // #pragma omp parallel for
        // //         for(int h = 0; h < num_heads; ++h)
        // //             result.slice(array_3(0, 0, h * depth), array_3(batch_size, seq_len_a, depth)) = F.chip(h, 1);

        //         // result.resize(batch_size, seq_len_a, num_heads * depth);
        //         // for(int h = 0; h < num_heads; ++h)
        //         //     result.slice(array_3(0, 0, h * depth), array_3(batch_size, seq_len_a, depth)) = F.chip(h, 1);

        //         result = F.shuffle(array_4(0, 2, 1, 3))  // (B, S, H, D/H)
        //                      .reshape(array_3(batch_size, seq_len_a, depth));  // (B, S, D)


        //         cout << "result.dimensions" << result.dimensions() << endl;
        //         cout << "result(0, 0, 0, 0)" << result(0, 0, 0) << endl;
        //         cout << "result(1, 2, 3, 3)" << result(1, 1, 1) << endl;


        // return 0;

        LanguageDataset language_dataset("../data/amazon_cells_labelled.txt");

        const Index embedding_dimension = 64;
        const Index neurons_number = 8;

        const Index input_vocabulary_size = language_dataset.get_input_vocabulary_size();
        const Index target_vocabulary_size = language_dataset.get_target_vocabulary_size();

        const Index input_sequence_length = language_dataset.get_input_sequence_length();
        const Index targets_number = language_dataset.get_target_sequence_length();

        dimensions input_dimensions = {input_vocabulary_size, input_sequence_length, embedding_dimension};
        dimensions complexity_dimensions = {neurons_number};
        dimensions output_dimensions = {targets_number};

        TextClassificationNetwork text_classification_network(
            input_dimensions,
            complexity_dimensions,
            output_dimensions
            );

        TrainingStrategy training_strategy(&text_classification_network, &language_dataset);
        AdaptiveMomentEstimation* adam = static_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_display_period(5);
        adam->set_maximum_epochs_number(100);
        // adam->set_batch_size(10);

        training_strategy.train();

        // NormalizedSquaredError normalized_squared_error(&text_classification_network, &language_dataset);

        // Tensor<type, 1> gradient = normalized_squared_error.calculate_gradient();
        // Tensor<type, 1> numerical_gradient  = normalized_squared_error.calculate_numerical_gradient();

        // Tensor<type, 1> difference = gradient - numerical_gradient;
        // Tensor<type, 1> abs_difference = difference.abs();
        // cout << abs_difference.maximum();

        // cout << normalized_squared_error.calculate_numerical_error() << endl;

        // cout << normalized_squared_error.calculate_gradient().abs() - normalized_squared_error.calculate_numerical_gradient().abs()  << endl;


        // const Index batch_size = 1;
        // Tensor<type, 2> input(batch_size, input_sequence_length);
        // input.setRandom();
        // const Tensor<type, 2> outputs = text_classification_network.calculate_outputs<2, 2>(input);

        const TestingAnalysis testing_analysis(&text_classification_network, &language_dataset);

        cout << "Confusion matrix:\n"
             << testing_analysis.calculate_confusion() << endl;

        TestingAnalysis::RocAnalysis roc_analysis = testing_analysis.perform_roc_analysis();

        cout << "perform_roc_analysis:\n"
             << "  AUC: " << roc_analysis.area_under_curve << "\n"
             << "  Confidence Limit: " << roc_analysis.confidence_limit << "\n"
             << "  Optimal Threshold: " << roc_analysis.optimal_threshold << "\n";

        return 0;

        cout << "Good bye!" << endl;
        return 0;
    }
    catch(const exception& e)
    {
        cout << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software Foundation.
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA), language_dataset.get_target_dimensions(), Dense2d::Activation::Logistic));
