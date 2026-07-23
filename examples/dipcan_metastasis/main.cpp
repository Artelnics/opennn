//   DIPCAN - Primary tumor location (binary classification), OpenNN engine.
//   Architecture: NO hidden layer -> Scaling + single logistic output (logistic regression).
//   Loss: WeightedSquaredError, L2. Optimizer: QuasiNewtonMethod.
//   Honest evaluation: repeated random 60/20/20 splits (reproducible seeds); report MEAN test AUC.
//
//   Selection modes ([selector] arg):
//     * GrowingInputs (default): greedy one-at-a-time forward selection, correlation-ordered.
//     * GeneticAlgorithm: population search over input subsets.
//     * Correlation: univariate filter, top max_inputs candidates by |Pearson correlation|, no wrapper training.
//     * CorrelationGolden: like Correlation, but N (how many top-correlated vars to keep) is
//       chosen by golden-section search over validation error instead of fixed at max_inputs.
//       Assumes validation error is roughly unimodal in N; tracks the best N actually observed
//       as a safety net.
//     * Fixed inputs: pass a file (one variable name per line) as [inputs_file] -> use exactly those
//       variables (intersected with the dataset), no selection. Replicates a given feature set.
//
//   Usage:
//     dipcan_primary_tumor <csv> <target> <unused_csv> <out_dir> <max_inputs> <max_time_s> [trials] [sel_epochs] [repeats] [inputs_file]

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <filesystem>

#include "opennn/tabular_dataset.h"
#include "opennn/neural_network.h"
#include "opennn/standard_networks.h"
#include "opennn/dense_layer.h"
#include "opennn/training_strategy.h"
#include "opennn/optimizer.h"
#include "opennn/adaptive_moment_estimation.h"
#include "opennn/growing_inputs.h"
#include "opennn/genetic_algorithm.h"
#include "opennn/inputs_selection.h"
#include "opennn/testing_analysis.h"
#include "opennn/configuration.h"
#include "opennn/random_utilities.h"

using namespace opennn;
using namespace std;

static vector<string> split_csv(const string& s)
{
    vector<string> out;
    stringstream ss(s);
    string item;
    while (getline(ss, item, ','))
    {
        size_t a = item.find_first_not_of(" \t\r\n");
        size_t b = item.find_last_not_of(" \t\r\n");
        if (a != string::npos) out.push_back(item.substr(a, b - a + 1));
    }
    return out;
}

static vector<string> read_lines(const string& path)
{
    vector<string> out;
    ifstream f(path);
    string line;
    while (getline(f, line))
    {
        size_t a = line.find_first_not_of(" \t\r\n");
        size_t b = line.find_last_not_of(" \t\r\n");
        if (a == string::npos) continue;
        string s = line.substr(a, b - a + 1);
        if (!s.empty() && s[0] != '#') out.push_back(s);
    }
    return out;
}

static void configure(TrainingStrategy& ts, const string& loss, float reg_weight, const string& optimizer, float lr)
{
    ts.set_loss(loss);
    ts.get_loss()->set_regularization("L2");
    ts.get_loss()->set_regularization_weight(reg_weight);
    ts.set_optimization_algorithm(optimizer);
    Optimizer* opt = ts.get_optimization_algorithm();
    if (opt)
    {
        opt->set_display(false);
        if (AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(opt))
        {
            adam->set_learning_rate(lr);
            adam->set_batch_size(512);
        }
    }
}

int main(int argc, char* argv[])
{
    try
    {
        if (argc < 7)
        {
            cerr << "Usage: dipcan_primary_tumor <csv> <target> <unused_csv> <out_dir> <max_inputs> <max_time_s> [trials] [sel_epochs] [repeats] [inputs_file]" << endl;
            return 1;
        }

        const string csv         = argv[1];
        const string target      = argv[2];
        const string unused_csv   = argv[3];
        const string out_dir      = argv[4];
        const Index  max_inputs   = stol(argv[5]);
        const float  max_time     = stof(argv[6]);
        const Index  trials       = (argc > 7) ? stol(argv[7]) : 2;
        const Index  sel_epochs   = (argc > 8) ? stol(argv[8]) : 200;
        const int    repeats      = (argc > 9) ? (int)stol(argv[9]) : 8;
        string       inputs_file  = (argc > 10) ? argv[10] : "";
        if (inputs_file == "-" || inputs_file == "none" || inputs_file == "NONE") inputs_file.clear();
        const bool   fixed_mode   = !inputs_file.empty();
        const float  reg_weight   = (argc > 11) ? stof(argv[11]) : 0.001f;
        const string optimizer    = (argc > 12) ? argv[12] : "QuasiNewtonMethod";
        const float  learning_rate = (argc > 13) ? stof(argv[13]) : 0.01f;
        const string loss          = (argc > 14) ? argv[14] : "WeightedSquaredError";
        const string selector      = (argc > 15) ? argv[15] : "GrowingInputs";  // or GeneticAlgorithm
        const Index  individuals   = (argc > 16) ? stol(argv[16]) : 40;
        const Index  max_val_fail  = (argc > 17) ? stol(argv[17]) : 100;  // stopping strictness
        const Index  folds         = (argc > 18) ? stol(argv[18]) : 1;    // inner k-fold CV (1 = off)
        const Index  hidden        = (argc > 19) ? stol(argv[19]) : 0;    // hidden-layer neurons (0 = logistic)
        const float  dropout_rate  = (argc > 20) ? stof(argv[20]) : 0.0f; // dropout on the hidden layer

        filesystem::create_directories(out_dir);
        cout << "=== DIPCAN primary tumor: target=" << target
             << " | no hidden layer (logistic) | "
             << (fixed_mode ? "FIXED inputs" : "GrowingInputs")
             << " | repeats=" << repeats << " ===" << endl;

        Configuration::instance().set(Device::CPU, Type::FP32);

        TabularDataset dataset(csv, ",", true, true);
        // Impute missing values once (mean). GrowingInputs/GA do this inside perform_input_selection,
        // but the Correlation/fixed paths skip it -> training on NaN gives a degenerate AUC~0.5 model.
        if (dataset.has_nan()) dataset.scrub_missing_values();
        const vector<string> all_names = dataset.get_variable_names();
        const set<string> all_set(all_names.begin(), all_names.end());
        const vector<string> unused_v = split_csv(unused_csv);
        const set<string> unused_set(unused_v.begin(), unused_v.end());

        // Fixed-inputs set (intersect requested list with dataset columns)
        set<string> fixed_set;
        vector<string> missing;
        if (fixed_mode)
        {
            for (const string& n : read_lines(inputs_file))
            {
                if (all_set.count(n)) fixed_set.insert(n);
                else missing.push_back(n);
            }
            cout << "Fixed inputs: requested with " << fixed_set.size() << " present, "
                 << missing.size() << " missing in corrected CSV." << endl;
        }

        auto set_full_candidates = [&]()   // GrowingInputs starting point
        {
            for (const string& name : all_names)
                dataset.set_variable_role(name, (name == target) ? "Target" : (unused_set.count(name) ? "None" : "Input"));
        };
        auto set_fixed_roles = [&]()       // exactly the fixed set as Input
        {
            for (const string& name : all_names)
                dataset.set_variable_role(name, (name == target) ? "Target" : (fixed_set.count(name) ? "Input" : "None"));
        };
        // set_variable_role does NOT recompute the cached input/target shapes; do it explicitly.
        auto recompute_shapes = [&]()
        {
            dataset.set_shape("Input",  Shape{ dataset.get_features_number("Input") });
            dataset.set_shape("Target", Shape{ dataset.get_features_number("Target") });
        };

        // rank 0 -> Scaling + logistic output (no hidden layer); {hidden} -> one Tanh hidden layer (MLP).
        Shape hidden_shape;
        if (hidden > 0) hidden_shape = Shape{ hidden };

        vector<double> aucs;
        vector<double> val_errors;   // selection validation error (what GI/GA optimize)
        vector<double> thresholds;
        vector<int> input_counts;
        vector<vector<std::array<double,3>>> roc_curves;
        map<string,int> freq;
        Index candidates = 0;

        for (int r = 0; r < repeats; ++r)
        {
            set_seed(1000u + (unsigned)r);

            vector<string> selected;
            double sel_val_error = -1.0;   // validation error achieved by the selector

            if (fixed_mode)
            {
                set_fixed_roles();
                dataset.split_samples_random(0.6f, 0.2f, 0.2f);
                selected.assign(fixed_set.begin(), fixed_set.end());
                if (r == 0) candidates = (Index)selected.size();
            }
            else
            {
                set_full_candidates();
                dataset.split_samples_random(0.6f, 0.2f, 0.2f);
                if (r == 0) candidates = (Index)dataset.get_variable_names("Input").size();

                if (selector == "Correlation")
                {
                    // Univariate filter: keep the top-N inputs by |Pearson correlation| with the
                    // target. NOT GrowingInputs/GA (no wrapper training). Honest: exclude Testing
                    // from the correlation (calculate_correlations_rank uses "used" samples), so the
                    // ranking never sees the test set.
                    const vector<Index> test_idx = dataset.get_sample_indices("Testing");
                    for (Index i : test_idx) dataset.set_sample_role(i, "None");
                    const VectorI rank = dataset.calculate_correlations_rank();  // ascending |corr|
                    for (Index i : test_idx) dataset.set_sample_role(i, "Testing");

                    const vector<string> in_names = dataset.get_variable_names("Input");
                    const Index n = (Index)in_names.size();
                    const Index take = min<Index>(max_inputs, n);
                    set<string> keep;
                    for (Index j = n - take; j < n; ++j) keep.insert(in_names[(size_t)rank(j)]);

                    for (const string& name : in_names)
                        if (!keep.count(name)) dataset.set_variable_role(name, "None");
                    dataset.set_variable_role(target, "Target");
                    selected.assign(keep.begin(), keep.end());
                    sel_val_error = -1.0;
                }
                else if (selector == "CorrelationGolden")
                {
                    // Golden-section search over N (top-N candidates by |correlation|),
                    // instead of GrowingInputs' one-at-a-time greedy walk: assumes
                    // validation error is roughly unimodal in N. O(log N) evaluations
                    // instead of up to max_inputs + max_val_fail. Tracks the best N
                    // actually observed as a safety net in case the curve has more than
                    // one local minimum.
                    const vector<Index> test_idx = dataset.get_sample_indices("Testing");
                    for (Index i : test_idx) dataset.set_sample_role(i, "None");
                    const VectorI rank = dataset.calculate_correlations_rank();  // ascending |corr|
                    for (Index i : test_idx) dataset.set_sample_role(i, "Testing");

                    const vector<string> in_names = dataset.get_variable_names("Input");
                    const Index n_candidates = (Index)in_names.size();

                    ClassificationNetwork nn(dataset.get_input_shape(), hidden_shape, dataset.get_target_shape());
                    TrainingStrategy ts(&nn, &dataset);
                    // Some top-N subsets are numerically ill-conditioned (highly collinear
                    // correlation-ranked variables). QuasiNewtonMethod's line search can hang
                    // on those well past its own maximum_time (observed: stuck >3 min despite
                    // a 30s cap -- looks like a real degenerate-Hessian edge case inside a
                    // single step, not something a time/epoch limit catches). Adam has no line
                    // search, so it can't hang the same way; use it only for these per-N scoring
                    // passes. The final model below still uses the requested optimizer/loss.
                    configure(ts, loss, reg_weight, "AdaptiveMomentEstimation", learning_rate);
                    if (ts.get_optimization_algorithm()) ts.get_optimization_algorithm()->set_maximum_epochs(300);

                    auto apply_top_n = [&](Index N) -> set<string>
                    {
                        const Index take = min<Index>(max<Index>(N, 1), n_candidates);
                        set<string> keep;
                        for (Index j = n_candidates - take; j < n_candidates; ++j)
                            keep.insert(in_names[(size_t)rank(j)]);

                        for (const string& name : in_names)
                            dataset.set_variable_role(name, keep.count(name) ? "Input" : "None");
                        dataset.set_variable_role(target, "Target");

                        const Index input_features_number = dataset.get_features_number("Input");
                        const Shape input_shape{ input_features_number };
                        nn.set_input_shape(input_shape);
                        dataset.set_shape("Input", input_shape);
                        nn.set_input_variables(dataset.get_variables("Input"));
                        nn.compile();
                        return keep;
                    };

                    map<Index, float> cache;
                    auto eval_n = [&](Index N) -> float
                    {
                        if (const auto it = cache.find(N); it != cache.end()) return it->second;
                        cout << "  [Golden] evaluating N=" << N << "..." << flush;
                        const set<string> chosen = apply_top_n(N);
                        cout << " [shapes set, features=" << dataset.get_features_number("Input")
                             << ", vars=" << chosen.size() << "]\n    vars: ";
                        for (const string& v : chosen) cout << v << " | ";
                        cout << "\n    training..." << flush;
                        const float verr = evaluate_selection_error(&ts, &nn, &dataset, folds).validation_error;
                        cache[N] = verr;
                        cout << " val_error=" << verr << "\n";
                        return verr;
                    };

                    Index lo = 1, hi = min<Index>(max_inputs, n_candidates);
                    Index best_n = hi;
                    float best_err = MAX;
                    auto note_best = [&](Index N, float err) { if (err < best_err) { best_err = err; best_n = N; } };

                    if (hi <= lo)
                    {
                        note_best(hi, eval_n(hi));
                    }
                    else
                    {
                        constexpr double invphi = 0.6180339887498949;
                        Index c = clamp<Index>((Index)llround(hi - invphi * (double)(hi - lo)), lo, hi);
                        Index d = clamp<Index>((Index)llround(lo + invphi * (double)(hi - lo)), lo, hi);
                        float fc = eval_n(c); note_best(c, fc);
                        float fd = eval_n(d); note_best(d, fd);

                        while (hi - lo > 2)
                        {
                            if (fc < fd)
                            {
                                hi = d; d = c; fd = fc;
                                c = clamp<Index>((Index)llround(hi - invphi * (double)(hi - lo)), lo, hi);
                                fc = eval_n(c); note_best(c, fc);
                            }
                            else
                            {
                                lo = c; c = d; fc = fd;
                                d = clamp<Index>((Index)llround(lo + invphi * (double)(hi - lo)), lo, hi);
                                fd = eval_n(d); note_best(d, fd);
                            }
                        }
                        for (Index N = lo; N <= hi; ++N) note_best(N, eval_n(N));
                    }

                    selected.assign(apply_top_n(best_n).begin(), apply_top_n(best_n).end());
                    sel_val_error = best_err;

                    cout << "  CorrelationGolden: best N=" << best_n << " (evaluated "
                         << cache.size() << " distinct N), val_error=" << best_err << "\n";
                }
                else
                {
                    ClassificationNetwork nn(dataset.get_input_shape(), hidden_shape, dataset.get_target_shape());
                    TrainingStrategy ts(&nn, &dataset);
                    configure(ts, loss, reg_weight, optimizer, learning_rate);

                    InputsSelectionResult res;
                    if (selector == "GeneticAlgorithm")
                    {
                        GeneticAlgorithm ga(&ts);
                        ga.set_minimum_inputs_number(1);
                        ga.set_maximum_inputs_number(max_inputs);
                        ga.set_trials_number(trials);
                        ga.set_maximum_validation_failures(max_val_fail);
                        ga.set_folds_number(folds);
                        ga.set_individuals_number(individuals);
                        ga.set_initialization_method("Correlations");
                        ga.set_maximum_epochs(sel_epochs);   // generations
                        ga.set_maximum_time(max_time);
                        ga.set_display(false);
                        res = ga.perform_input_selection();
                    }
                    else
                    {
                        GrowingInputs gi(&ts);
                        gi.set_minimum_inputs_number(1);
                        gi.set_maximum_inputs_number(max_inputs);
                        gi.set_trials_number(trials);
                        gi.set_maximum_validation_failures(max_val_fail);
                        gi.set_folds_number(folds);
                        gi.set_maximum_epochs(sel_epochs);
                        gi.set_maximum_time(max_time);
                        gi.set_display(false);
                        res = gi.perform_input_selection();
                    }
                    selected = res.optimal_input_variable_names;
                    sel_val_error = (double)res.optimum_validation_error;

                    const set<string> sel(selected.begin(), selected.end());
                    for (const string& name : dataset.get_variable_names("Input"))
                        if (!sel.count(name)) dataset.set_variable_role(name, "None");
                    for (const string& name : selected) dataset.set_variable_role(name, "Input");
                    dataset.set_variable_role(target, "Target");
                }
            }

            // --- Final model on the current Input set
            recompute_shapes();
            ClassificationNetwork fin(dataset.get_input_shape(), hidden_shape, dataset.get_target_shape());
            if (hidden > 0 && dropout_rate > 0.0f)
                if (opennn::Dense* d = dynamic_cast<opennn::Dense*>(fin.get_layer(1).get()))   // layer 1 = hidden Dense (0 = Scaling)
                    d->set_dropout_rate(dropout_rate);
            TrainingStrategy ts2(&fin, &dataset);
            configure(ts2, loss, reg_weight, optimizer, learning_rate);
            if (ts2.get_optimization_algorithm()) ts2.get_optimization_algorithm()->set_maximum_epochs(1000);
            ts2.train();

            TestingAnalysis ta(&fin, &dataset);
            TestingAnalysis::RocAnalysis roc = ta.perform_roc_analysis();

            aucs.push_back(roc.area_under_curve);
            val_errors.push_back(sel_val_error);
            thresholds.push_back(roc.optimal_threshold);
            input_counts.push_back((int)selected.size());
            for (const string& name : selected) freq[name]++;

            vector<std::array<double,3>> curve;
            const long cols = roc.roc_curve.cols();
            for (long i = 0; i < roc.roc_curve.rows(); ++i)
                curve.push_back({ (double)roc.roc_curve(i,0),
                                  (double)roc.roc_curve(i,1),
                                  cols > 2 ? (double)roc.roc_curve(i,2) : 0.0 });
            roc_curves.push_back(curve);

            cout << "  repeat " << r << ": AUC=" << roc.area_under_curve
                 << "  inputs=" << selected.size() << endl;
        }

        const int K = (int)aucs.size();
        const double mean = accumulate(aucs.begin(), aucs.end(), 0.0) / K;
        double var = 0; for (double a : aucs) var += (a - mean) * (a - mean);
        const double sd = K > 1 ? sqrt(var / (K - 1)) : 0.0;
        const double amin = *min_element(aucs.begin(), aucs.end());
        const double amax = *max_element(aucs.begin(), aucs.end());
        int ri = 0; double bestd = 1e9;
        for (int i = 0; i < K; ++i) { double d = fabs(aucs[i] - mean); if (d < bestd) { bestd = d; ri = i; } }

        // Selection validation error (the objective GrowingInputs/GeneticAlgorithm minimize)
        double vmean = 0; int vn = 0;
        for (double v : val_errors) if (v >= 0) { vmean += v; ++vn; }
        vmean = vn ? vmean / vn : -1.0;
        double vsd = 0; if (vn > 1) { for (double v : val_errors) if (v >= 0) vsd += (v - vmean)*(v - vmean); vsd = sqrt(vsd/(vn-1)); }

        cout << "AUC mean=" << mean << " sd=" << sd << " min=" << amin << " max=" << amax
             << " | ValError mean=" << vmean
             << " (representative repeat " << ri << ")" << endl;

        {
            ofstream f(out_dir + "/summary.txt");
            f << "target=" << target << "\n";
            f << "csv=" << csv << "\n";
            f << "mode=" << (fixed_mode ? "fixed_inputs" : "selection") << "\n";
            f << "samples=" << dataset.get_samples_number() << "\n";
            f << "architecture=" << (hidden > 0 ? ("MLP hidden=" + std::to_string(hidden)) : "logistic (no hidden layer)") << "\n";
            f << "selector=" << (fixed_mode ? "fixed" : selector) << "\n";
            f << "folds=" << folds << "\n";
            f << "hidden=" << hidden << "\n";
            f << "dropout=" << dropout_rate << "\n";
            f << "loss=" << loss << "\n";
            f << "optimizer=" << optimizer << "\n";
            f << "reg_weight=" << reg_weight << "\n";
            f << "candidate_inputs=" << candidates << "\n";
            if (fixed_mode) f << "fixed_inputs_missing=" << missing.size() << "\n";
            f << "repeats=" << K << "\n";
            f << "AUC_mean=" << mean << "\n";
            f << "AUC_std=" << sd << "\n";
            f << "AUC_min=" << amin << "\n";
            f << "AUC_max=" << amax << "\n";
            f << "AUC_representative=" << aucs[ri] << "\n";
            f << "ValError_mean=" << vmean << "\n";
            f << "ValError_std=" << vsd << "\n";
            f << "representative_threshold=" << thresholds[ri] << "\n";
            f << "representative_inputs=" << input_counts[ri] << "\n";
        }
        {
            ofstream f(out_dir + "/aucs_per_repeat.txt");
            for (int i = 0; i < K; ++i)
                f << "repeat " << i << " seed " << (1000 + i) << " AUC " << aucs[i]
                  << " val_error " << val_errors[i]
                  << " inputs " << input_counts[i] << "\n";
        }
        {
            vector<pair<string,int>> fv(freq.begin(), freq.end());
            sort(fv.begin(), fv.end(), [](const pair<string,int>&a, const pair<string,int>&b)
                 { return a.second != b.second ? a.second > b.second : a.first < b.first; });
            ofstream f(out_dir + "/selected_inputs.txt");
            f << "# " << (fixed_mode ? "Fixed input set" : ("Variables selected by " + selector))
              << " across " << K << " splits (" << (hidden > 0 ? "MLP" : "logistic") << " model)\n";
            f << "# format: <count>/" << K << "  <variable>\n";
            for (const pair<string,int>& p : fv) f << p.second << "/" << K << "\t" << p.first << "\n";
            if (fixed_mode && !missing.empty())
            {
                f << "# --- requested but MISSING in corrected CSV (" << missing.size() << ") ---\n";
                for (const string& m : missing) f << "0/" << K << "\t" << m << "\n";
            }
        }
        {
            ofstream f(out_dir + "/roc_curve.csv");
            f << "false_positive_rate,true_positive_rate,threshold\n";
            for (const std::array<double,3>& row : roc_curves[ri])
                f << row[0] << "," << row[1] << "," << row[2] << "\n";
        }

        cout << "Done. Outputs in " << out_dir << endl;
        return 0;
    }
    catch (const exception& e)
    {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }
}
