// OpenNN GPU HIGGS dense training benchmark: one main function, written the
// way a user writes an OpenNN training application, plus a timer and the
// test-set quality gate. The measurement protocol lives in run_higgs_dense.py.
//
//   opennn_speed <train_csv> <epochs> <batch> <fp32|bf16> <hidden> <activation> <hidden_layers> <test_csv>
//                <min_accuracy> <max_log_loss> <min_auc>
//
// The canonical HIGGS dense classifier (28 -> hidden -> hidden -> 1, ReLU
// hidden, sigmoid output, binary cross-entropy) trains with Adam on the GPU:
// the training split is device-resident, the step runs as a captured CUDA
// graph, and two warmup epochs precede the timed ones. bf16 is OpenNN's
// mixed-precision training path, matching the autocast / mixed_bfloat16 cells
// of the PyTorch and TensorFlow drivers. After training the test CSV is scored
// on the same network and accuracy / log-loss / ROC-AUC are computed the way
// those drivers compute them (../higgs/metrics.py): whole batches only,
// probabilities clamped to [1e-7, 1 - 1e-7], AUC from average ranks with ties.
//
// A threshold argument is "none" when unset; a threshold is enforced, and
// quality_gate=PASS|FAIL printed, only when a number is given.
//
// OPENNN_SPEED_KEEP_TAIL=1 trains the remainder batch too (tail_kept=1),
// OPENNN_SPEED_NO_GRAPH=1 runs the step without the CUDA graph, and
// OPENNN_BENCH_SCALERS=1 keeps the dataset's default scalers, which are off
// because prepare_higgs.py already normalized the CSV.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef OPENNN_HAS_CUDA
#include <cuda_runtime.h>
#include <cudnn.h>
#endif

#include "opennn/core/configuration.h"
#include "opennn/core/random_utilities.h"
#include "opennn/core/tensor_types.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/neural_network.h"
#include "opennn/training_strategy/adaptive_moment_estimation.h"
#include "opennn/training_strategy/training_strategy.h"

using namespace opennn;
using clock_type = chrono::steady_clock;

int main(int argc, char* argv[])
{
    cout << unitbuf;

    try
    {
        if (argc < 5)
        {
            cerr << "usage: opennn_speed <train_csv> <epochs> <batch> <fp32|bf16>"
                    " <hidden> <activation> <hidden_layers> <test_csv>"
                    " <min_accuracy> <max_log_loss> <min_auc>\n";
            return 2;
        }

        const string train_path = argv[1];
        const Index epochs = Index(stoll(argv[2]));
        const Index batch = Index(stoll(argv[3]));
        const string precision = argv[4];
        const Index hidden = argc > 5 ? Index(stoll(argv[5])) : 1024;
        const string activation = argc > 6 ? argv[6] : "relu";
        const Index hidden_layers = argc > 7 ? Index(stoll(argv[7])) : 2;
        const string test_path = argc > 8 ? argv[8] : "";
        const string min_accuracy_arg = argc > 9 ? argv[9] : "none";
        const string max_log_loss_arg = argc > 10 ? argv[10] : "none";
        const string min_auc_arg = argc > 11 ? argv[11] : "none";

        if (test_path.empty())
            throw runtime_error("test CSV path is required for the quality gate");

        const auto has_threshold = [](const string& value)
        {
            return !(value.empty() || value == "none" || value == "None" || value == "nan");
        };

        set_seed(42);
        const Type training_type = (precision == "bf16") ? Type::BF16 : Type::FP32;
        Configuration::instance().set(Device::CUDA, training_type);

        TabularDataset dataset(train_path, ",", false, false);
        dataset.set_storage_mode(Dataset::StorageMode::GPUPersistantData);
        dataset.set_sample_roles("Training");
        if (!getenv("OPENNN_BENCH_SCALERS"))
            dataset.set_variable_scalers("None");
        const Index samples = dataset.get_samples_number();

        // Whole batches only, like the PyTorch and TensorFlow drivers
        // (range(0, n - batch + 1, batch)): the remainder is left out of the
        // epoch instead of trained as a smaller tail batch. The tail is real
        // work the throughput figure does not count (up to 6.5% at 896,000),
        // and the library keeps a second set of activation contexts for it,
        // which at 6 GB is the difference between fitting and paging at
        // 448,000. OPENNN_SPEED_KEEP_TAIL=1 trains it anyway.
        const bool keep_tail = getenv("OPENNN_SPEED_KEEP_TAIL") != nullptr;
        const Index whole_samples = (batch > 0 && !keep_tail) ? (samples / batch) * batch : samples;
        for (Index sample = whole_samples; sample < samples; ++sample)
            dataset.set_sample_role(sample, SampleRole::None);

        cout << "engine=opennn\n";
        cout << "mode=train\n";
#ifdef OPENNN_HAS_CUDA
        // Machine identity for the speed gate: throughput and kernel choice are
        // a property of (GPU, cuDNN), so baselines are keyed by both.
        cudaDeviceProp properties{};
        cout << "device="
             << (cudaGetDeviceProperties(&properties, 0) == cudaSuccess ? properties.name : "cuda")
             << "\n";
        cout << "cudnn=" << cudnnGetVersion() << "\n";
#else
        cout << "device=cuda\n";
#endif
        cout << "samples=" << samples << "\n";
        cout << "tail_kept=" << (keep_tail ? 1 : 0) << "\n";
        cout << "batch=" << batch << "\n";
        cout << "epochs=" << epochs << "\n";
        cout << "hidden=" << hidden << "\n";
        cout << "hidden_layers=" << hidden_layers << "\n";
        cout << "activation=" << activation << "\n";
        cout << "precision=" << precision << "\n";

        NeuralNetwork network;
        const string hidden_activation = (activation == "relu" || activation == "ReLU") ? "ReLU" : "Tanh";
        Shape current = dataset.get_input_shape();

        for (Index i = 0; i < hidden_layers; ++i)
        {
            network.add_layer(make_unique<opennn::Dense>(current, Shape{hidden}, hidden_activation, false,
                                                         "higgs_dense_" + to_string(i + 1)));
            current = network.get_output_shape();
        }

        network.add_layer(make_unique<opennn::Dense>(current, dataset.get_target_shape(), "Sigmoid", false,
                                                     "higgs_output"));
        network.compile();
        network.set_parameters_glorot();
        cout << "parameters=" << network.get_parameters_number() << "\n";

        TrainingStrategy training_strategy(&network, &dataset);
        training_strategy.set_loss("CrossEntropy");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_batch_size(batch);
        adam->set_cuda_graph(getenv("OPENNN_SPEED_NO_GRAPH") == nullptr);
        adam->set_display_period(1000000);
        adam->set_gradient_clip_norm(0.0f);

        const Index warmup_epochs = 2;
        adam->set_maximum_epochs(warmup_epochs + epochs);

        // The strategy owns the epoch loop, so the epochs are timed from its
        // post-epoch callback: the warmup epochs are dropped, and the wall
        // clock at the edges of the timed window is printed for the energy
        // runner, which integrates power between the two marks.
        const auto unix_now = []
        {
            return chrono::duration<double>(chrono::system_clock::now().time_since_epoch()).count();
        };

        vector<double> epoch_seconds;
        auto previous_mark = clock_type::now();
        adam->post_epoch_callback = [&](Index epoch, float, float, NeuralNetwork*)
        {
            const auto now = clock_type::now();
            const double elapsed = chrono::duration<double>(now - previous_mark).count();
            previous_mark = now;

            if (epoch == warmup_epochs - 1)
                cout << "TRAIN_START_UNIX=" << fixed << setprecision(3) << unix_now() << "\n" << defaultfloat;
            else if (epoch >= warmup_epochs)
                epoch_seconds.push_back(elapsed);

            if (epoch == warmup_epochs + epochs - 1)
                cout << "TRAIN_END_UNIX=" << fixed << setprecision(3) << unix_now() << "\n" << defaultfloat;
        };

        training_strategy.train();

        throw_if(Index(epoch_seconds.size()) != epochs, "epoch timing marks missing");
        sort(epoch_seconds.begin(), epoch_seconds.end());
        const double median_epoch_s = epoch_seconds[epoch_seconds.size() / 2];

        // An epoch runs whole batches only; the remainder is dropped. Dividing the
        // full split by the epoch time overstates throughput by up to the size of
        // one batch, which is 6.5% at batch 896,000.
        const Index samples_per_epoch = (samples / batch) * batch;
        const double samples_per_sec = double(samples_per_epoch) / median_epoch_s;

        // What the speed gate asserts besides throughput: the step was
        // captured, and the output layer still takes the one-pass backward
        // that folds its producer's ReLU. Both are invariants a refactor can
        // drop without changing a single result.
        const auto* output_dense = dynamic_cast<const opennn::Dense*>(network.get_layers().back().get());
        cout << "cuda_graph="
             << (getenv("OPENNN_SPEED_NO_GRAPH") ? "off"
                 : adam->get_cuda_graph_capture_failed() ? "failed" : "captured") << "\n";
        cout << "single_output_fold="
             << (output_dense && output_dense->single_output_relu_fusion_wired() ? 1 : 0) << "\n";

        // The test split is scored on the trained network, whole batches only.
        TabularDataset test_dataset(test_path, ",", false, false);
        test_dataset.set_sample_roles("Testing");
        const MatrixR& test_data = test_dataset.get_data();
        const Index test_samples = test_data.rows();
        const Index inputs_number = test_dataset.get_input_shape()[0];
        const Index test_processed = (test_samples / batch) * batch;
        const MatrixR test_inputs = test_data.leftCols(inputs_number);

        vector<pair<float, int>> scored;
        scored.reserve(size_t(test_processed));
        double log_loss_sum = 0.0;
        Index correct = 0;

        for (Index i = 0; i + batch <= test_samples; i += batch)
        {
            const TensorView view(const_cast<float*>(test_inputs.data()) + i * inputs_number,
                                  Shape{batch, inputs_number}, Type::FP32);
            const MatrixR outputs = network.calculate_outputs(vector<TensorView>{view});

            for (Index r = 0; r < batch; ++r)
            {
                float probability = float(outputs(r, 0));
                if (probability < 1.0e-7f) probability = 1.0e-7f;
                if (probability > 1.0f - 1.0e-7f) probability = 1.0f - 1.0e-7f;

                const int label = test_data(i + r, inputs_number) >= 0.5f ? 1 : 0;
                const int predicted = probability >= 0.5f ? 1 : 0;
                correct += predicted == label ? 1 : 0;
                log_loss_sum += label ? -log(double(probability)) : -log(double(1.0f - probability));
                scored.emplace_back(probability, label);
            }
        }

        double accuracy = 0.0;
        double log_loss = 0.0;
        double auc = 0.0;

        if (test_processed > 0)
        {
            accuracy = double(correct) / double(test_processed);
            log_loss = log_loss_sum / double(test_processed);

            // ROC-AUC as the rank statistic with average ranks for ties; 0 when
            // the split holds a single class.
            Index positives = 0;
            for (const auto& item : scored)
                positives += item.second ? 1 : 0;
            const Index negatives = test_processed - positives;

            if (positives > 0 && negatives > 0)
            {
                sort(scored.begin(), scored.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });

                double positive_rank_sum = 0.0;
                for (Index i = 0; i < test_processed;)
                {
                    Index j = i + 1;
                    while (j < test_processed && scored[j].first == scored[i].first) ++j;
                    const double average_rank = (double(i + 1) + double(j)) * 0.5;
                    for (Index k = i; k < j; ++k)
                        if (scored[k].second) positive_rank_sum += average_rank;
                    i = j;
                }

                auc = (positive_rank_sum - double(positives) * double(positives + 1) * 0.5)
                    / (double(positives) * double(negatives));
            }
        }

        cout << "samples_per_epoch=" << samples_per_epoch << "\n";
        cout << "median_epoch_s=" << median_epoch_s << "\n";
        cout << "samples_per_sec=" << long(samples_per_sec) << "\n";
        cout << "test_samples=" << test_processed << "\n";
        cout << "test_accuracy=" << accuracy << "\n";
        cout << "test_log_loss=" << log_loss << "\n";
        cout << "test_roc_auc=" << auc << "\n";

        bool gate_pass = true;
        if (has_threshold(min_accuracy_arg) && accuracy < stod(min_accuracy_arg))
            gate_pass = false;
        if (has_threshold(max_log_loss_arg) && log_loss > stod(max_log_loss_arg))
            gate_pass = false;
        if (has_threshold(min_auc_arg) && (!isfinite(auc) || auc < stod(min_auc_arg)))
            gate_pass = false;

        if (has_threshold(min_accuracy_arg) || has_threshold(max_log_loss_arg) || has_threshold(min_auc_arg))
            cout << "quality_gate=" << (gate_pass ? "PASS" : "FAIL") << "\n";

        cout << "RESULT=OK\n";
        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << "\n";
        cout << "RESULT=ERROR\n";
        return 1;
    }
}
