//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   C U D A   —   Translation benchmark (refactor)
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>

#include "../opennn/language_dataset.h"
#include "../opennn/standard_networks.h"
#include "../opennn/training_strategy.h"
#include "../opennn/adaptive_moment_estimation.h"
#include "../opennn/random_utilities.h"
#include "../opennn/transformer_decoder.h"

using namespace opennn;
using namespace std::chrono;

// tinychat.txt format: each line is one conversation with multiple
// `[INST] user [/INST] assistant` turns concatenated. We extract every turn as
// an independent (user, assistant) pair and write them tab-separated, which is
// the format LanguageDataset::read_csv expects (one input/target per line).
//
// `max_pairs == 0` means no limit. Output is cached at `dst`; subsequent runs
// skip preprocessing.
static void prepare_tinychat_pairs(const std::filesystem::path& src,
                                   const std::filesystem::path& dst,
                                   Index max_pairs = 0)
{
    if (std::filesystem::exists(dst))
    {
        cout << "Reusing cached pairs file at " << dst << endl;
        return;
    }

    cout << "Preprocessing " << src << " -> " << dst
         << (max_pairs > 0 ? " (limit " + to_string(max_pairs) + " pairs)" : "")
         << " ..." << endl;

    std::ifstream in(src);
    if (!in.is_open()) throw runtime_error("Cannot open " + src.string());

    std::ofstream out(dst);
    if (!out.is_open()) throw runtime_error("Cannot open " + dst.string() + " for writing");

    const string open_tag  = "[INST]";
    const string close_tag = "[/INST]";

    auto sanitize = [](string& s)
    {
        const size_t first = s.find_first_not_of(" \t\r\n");
        const size_t last  = s.find_last_not_of(" \t\r\n");
        if (first == string::npos) { s.clear(); return; }
        s = s.substr(first, last - first + 1);
        for (char& c : s) if (c == '\t' || c == '\n' || c == '\r') c = ' ';
    };

    Index pair_count = 0;
    string line;
    while (std::getline(in, line))
    {
        if (max_pairs > 0 && pair_count >= max_pairs) break;

        size_t cursor = 0;
        while (cursor < line.size())
        {
            if (max_pairs > 0 && pair_count >= max_pairs) break;

            const size_t open_i = line.find(open_tag, cursor);
            if (open_i == string::npos) break;
            const size_t close_i = line.find(close_tag, open_i + open_tag.size());
            if (close_i == string::npos) break;
            const size_t next_open = line.find(open_tag, close_i + close_tag.size());
            const size_t resp_end  = (next_open == string::npos) ? line.size() : next_open;

            string user      = line.substr(open_i  + open_tag.size(),  close_i - (open_i  + open_tag.size()));
            string assistant = line.substr(close_i + close_tag.size(), resp_end - (close_i + close_tag.size()));
            sanitize(user);
            sanitize(assistant);

            if (!user.empty() && !assistant.empty())
            {
                out << user << '\t' << assistant << '\n';
                ++pair_count;
            }

            cursor = (next_open == string::npos) ? line.size() : next_open;
        }
    }

    cout << "Wrote " << pair_count << " (user, assistant) pairs to " << dst << endl;
}

int main()
{
    try
    {
        cout << "OpenNN. Translation benchmark (refactor)." << endl;

#ifdef OPENNN_WITH_CUDA

/*
        Configuration::instance().set(Device::CUDA, Type::BF16, Type::BF16);

        set_seed(42);

        LanguageDataset dataset("/home/artelnics/Documents/openNN/opennn/temp/translation_en_es.txt");
        dataset.split_samples_random(0.8, 0.1, 0.1);

        const Index input_vocab  = dataset.get_input_vocabulary_size();
        const Index output_vocab = dataset.get_target_vocabulary_size();
        const Index input_seq    = dataset.get_shape("Input")[0];
        const Index decoder_seq  = dataset.get_shape("Decoder")[0];
        const Index target_seq   = dataset.get_shape("Target")[0];

        if(decoder_seq != target_seq)
            throw runtime_error("Decoder and target sequence lengths must match.");

        const Index embedding_dimension     = 512;
        const Index heads_number            = 8;
        const Index feed_forward_dimension  = 2048;
        const Index layers_number           = 4;

        Transformer transformer(input_seq,
                                decoder_seq,
                                input_vocab,
                                output_vocab,
                                embedding_dimension,
                                heads_number,
                                feed_forward_dimension,
                                layers_number);

        transformer.set_dropout_rate(float(0));

        TrainingStrategy training_strategy(&transformer, &dataset);

        training_strategy.set_loss("CrossEntropyError3d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        if(!adam) throw runtime_error("AdaptiveMomentEstimation optimizer not found.");

        adam->set_batch_size(128);
        adam->set_learning_rate(float(5e-4));
        adam->set_maximum_epochs(4);   // 5 epochs (loop runs 0..4 inclusive)
        adam->set_display_period(1);

        cout << "[PARITY] train="  << dataset.get_samples_number("Training")
             << " val="            << dataset.get_samples_number("Validation")
             << " test="           << dataset.get_samples_number("Testing")
             << " input_vocab="    << input_vocab
             << " target_vocab="   << output_vocab
             << " input_seq="      << input_seq
             << " decoder_seq="    << decoder_seq
             << " params="         << transformer.get_parameters_number()
             << " (buffer="        << transformer.get_parameters_size() << ")" << endl;

        const auto t0 = steady_clock::now();
        training_strategy.train();
        const auto t1 = steady_clock::now();

        const double training_seconds = duration_cast<milliseconds>(t1 - t0).count() / 1000.0;
        cout << "\nTotal training time (refactor): " << training_seconds << " s" << endl;

        cout << "\n================ TRANSFORMER PREDICTIONS ================\n";

        const vector<string> test_sources =
        {
            "the cat eats an apple",
            "a boy sings happily",
            "my friend writes a book",
            "the teacher speaks loudly"
        };

        TransformerDecoder translation_decoder(transformer, dataset);
        for(Index i = 0; i < Index(test_sources.size()); ++i)
        {
            const string prediction = translation_decoder.decode(test_sources[i]);

            cout << "Sample " << i << endl;
            cout << "  Source:    " << test_sources[i] << endl;
            cout << "  Predicted: " << prediction << endl;
            cout << endl;
        }

        cout << "=========================================================\n";

        */

        // ====================  TINYCHAT (instruction-tuning)  ====================
        /*
        Configuration::instance().set(Device::CUDA, Type::BF16, Type::BF16);

        const std::filesystem::path tinychat_src = "/home/artelnics/Documents/openNN/opennn/temp/tinychat.txt";
        // Half-sized cache (~2M pairs) — keeps the previous full cache around if it exists.
        const std::filesystem::path tinychat_pairs = "/tmp/tinychat_pairs_2M.tsv";
        const Index max_pairs = 2'000'000;

        prepare_tinychat_pairs(tinychat_src, tinychat_pairs, max_pairs);

        LanguageDataset dataset(tinychat_pairs);
        dataset.split_samples_random(0.9, 0.05, 0.05);

        const Index input_vocab  = dataset.get_input_vocabulary_size();
        const Index output_vocab = dataset.get_target_vocabulary_size();
        const Index input_seq    = dataset.get_shape("Input")[0];
        const Index decoder_seq  = dataset.get_shape("Decoder")[0];
        const Index target_seq   = dataset.get_shape("Target")[0];

        if(decoder_seq != target_seq)
            throw runtime_error("Decoder and target sequence lengths must match.");

        const Index embedding_dimension     = 512;
        const Index heads_number            = 8;
        const Index feed_forward_dimension  = 2048;
        const Index layers_number           = 4;

        Transformer transformer(input_seq,
                                decoder_seq,
                                input_vocab,
                                output_vocab,
                                embedding_dimension,
                                heads_number,
                                feed_forward_dimension,
                                layers_number);

        transformer.set_dropout_rate(float(0.1));

        cout << "[TINYCHAT] train="    << dataset.get_samples_number("Training")
             << " val="                << dataset.get_samples_number("Validation")
             << " test="               << dataset.get_samples_number("Testing")
             << " input_vocab="        << input_vocab
             << " target_vocab="       << output_vocab
             << " input_seq="          << input_seq
             << " decoder_seq="        << decoder_seq
             << " params="             << transformer.get_parameters_number() << endl;

        const std::filesystem::path weights_file = "/tmp/tinychat_weights.bin";

        if (std::filesystem::exists(weights_file))
        {
            cout << "Loading cached weights from " << weights_file
                 << " (skipping training)" << endl;
            transformer.load_parameters_binary(weights_file);
        }
        else
        {
            TrainingStrategy training_strategy(&transformer, &dataset);

            training_strategy.set_loss("CrossEntropyError3d");
            training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

            auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
            if(!adam) throw runtime_error("AdaptiveMomentEstimation optimizer not found.");

            adam->set_batch_size(128);
            adam->set_learning_rate(float(5e-4));
            adam->set_maximum_epochs(0);
            adam->set_display_period(1);

            const auto t0 = steady_clock::now();
            training_strategy.train();
            const auto t1 = steady_clock::now();

            cout << "\nTotal training time (tinychat): "
                 << duration_cast<milliseconds>(t1 - t0).count() / 1000.0 << " s" << endl;

            std::ofstream weights_out(weights_file, std::ios::binary);
            if (!weights_out)
                throw runtime_error("Cannot open " + weights_file.string() + " for writing");
            const Index parameters_floats = transformer.get_parameters_size();
            weights_out.write(reinterpret_cast<const char*>(transformer.get_parameters_data()),
                              parameters_floats * sizeof(float));
            cout << "Saved " << parameters_floats << " floats ("
                 << parameters_floats * sizeof(float) << " bytes) to "
                 << weights_file << endl;
        }

        cout << "\n================ TINYCHAT CHAT ================\n";

        TransformerDecoder::SamplingConfig sampling_config;
        sampling_config.temperature        = 0.8f;
        sampling_config.top_p              = 0.9f;
        sampling_config.top_k              = 50;
        sampling_config.repetition_penalty = 1.15f;

        TransformerDecoder decoder(transformer, dataset);
        decoder.chat(sampling_config);

        cout << "===============================================\n";
        */

        // ====================  OASST2 (instruction-tuning)  ====================
        // Pre-extracted .tsv produced by temp/OASST/prepare_oasst2_pairs.py from
        // the OpenAssistant ready_for_export trees (English subset, length-capped).
        /*
        Configuration::instance().set(Device::CUDA, Type::BF16, Type::BF16);

        set_seed(42);

        const std::filesystem::path oasst2_pairs = "/home/artelnics/Documents/openNN/opennn/temp/OASST/oasst2_pairs.tsv";

        LanguageDataset dataset(oasst2_pairs, std::numeric_limits<Index>::max(), 2);
        dataset.split_samples_random(0.9, 0.05, 0.05);

        const Index input_vocab  = dataset.get_input_vocabulary_size();
        const Index output_vocab = dataset.get_target_vocabulary_size();
        const Index input_seq    = dataset.get_shape("Input")[0];
        const Index decoder_seq  = dataset.get_shape("Decoder")[0];
        const Index target_seq   = dataset.get_shape("Target")[0];

        if(decoder_seq != target_seq)
            throw runtime_error("Decoder and target sequence lengths must match.");

        const Index embedding_dimension     = 768;
        const Index heads_number            = 12;
        const Index feed_forward_dimension  = 3072;
        const Index layers_number           = 6;

        Transformer transformer(input_seq,
                                decoder_seq,
                                input_vocab,
                                output_vocab,
                                embedding_dimension,
                                heads_number,
                                feed_forward_dimension,
                                layers_number);

        transformer.set_dropout_rate(float(0.1));

        cout << "[OASST2] train="     << dataset.get_samples_number("Training")
             << " val="               << dataset.get_samples_number("Validation")
             << " test="              << dataset.get_samples_number("Testing")
             << " input_vocab="       << input_vocab
             << " target_vocab="      << output_vocab
             << " input_seq="         << input_seq
             << " decoder_seq="       << decoder_seq
             << " params="            << transformer.get_parameters_number() << endl;

        const std::filesystem::path weights_file = "/tmp/oasst2_weights.bin";

        if (std::filesystem::exists(weights_file))
        {
            cout << "Loading cached weights from " << weights_file
                 << " (skipping training)" << endl;
            transformer.load_parameters_binary(weights_file);
        }
        else
        {
            TrainingStrategy training_strategy(&transformer, &dataset);

            training_strategy.set_loss("CrossEntropyError3d");
            training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

            auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
            if(!adam) throw runtime_error("AdaptiveMomentEstimation optimizer not found.");

            adam->set_batch_size(32);
            adam->set_learning_rate(float(5e-4));
            adam->set_maximum_epochs(29);
            adam->set_display_period(1);

            const auto t0 = steady_clock::now();
            training_strategy.train();
            const auto t1 = steady_clock::now();

            cout << "\nTotal training time (oasst2): "
                 << duration_cast<milliseconds>(t1 - t0).count() / 1000.0 << " s" << endl;

            std::ofstream weights_out(weights_file, std::ios::binary);
            if (!weights_out)
                throw runtime_error("Cannot open " + weights_file.string() + " for writing");
            const Index parameters_floats = transformer.get_parameters_size();
            weights_out.write(reinterpret_cast<const char*>(transformer.get_parameters_data()),
                              parameters_floats * sizeof(float));
            cout << "Saved " << parameters_floats << " floats ("
                 << parameters_floats * sizeof(float) << " bytes) to "
                 << weights_file << endl;
        }

        cout << "\n================ OASST2 CHAT ================\n";

        TransformerDecoder::SamplingConfig sampling_config;
        sampling_config.temperature        = 0.8f;
        sampling_config.top_p              = 0.9f;
        sampling_config.top_k              = 50;
        sampling_config.repetition_penalty = 1.15f;

        TransformerDecoder decoder(transformer, dataset);
        decoder.chat(sampling_config);

        cout << "=============================================\n";
        */

        // ====================  TINYCHAT + OASST2 + ULTRACHAT (combined3)  =======
        // Concatenation of all three datasets:
        //   tinychat   ~2.00M pairs   short Q&A volume
        //   OASST2       ~27K pairs   diverse human-written conversations
        //   UltraChat  ~274K pairs    high-quality synthetic multi-turn chat
        // Total ~2.30M pairs.
        //
        // Model bumped to ~110M params (d_model=640, layers=6, heads=10, ff=2560)
        // — bigger than the 57M proven config but conservative enough to fit in
        // 16 GB VRAM with batch=64 and vocab cap 25K. Designed for ~25-35h of
        // training on the 4080 (10 epochs). Won't OOM.

        Configuration::instance().set(Device::CUDA, Type::BF16, Type::BF16);

        set_seed(42);

        const std::filesystem::path combined3_pairs = "/home/artelnics/Documents/openNN/opennn/temp/combined3_pairs.tsv";

        // Vocabulary cap 25K; drop hapax legomena (≥2 occurrences required).
        // 25K keeps all of tinychat (17K) plus the most frequent OASST2 +
        // UltraChat additions. The rare long tail is dropped to keep the
        // output-projection tensor manageable.
        LanguageDataset dataset(combined3_pairs, 25000, 2);
        dataset.split_samples_random(0.95, 0.025, 0.025);

        const Index input_vocab  = dataset.get_input_vocabulary_size();
        const Index output_vocab = dataset.get_target_vocabulary_size();
        const Index input_seq    = dataset.get_shape("Input")[0];
        const Index decoder_seq  = dataset.get_shape("Decoder")[0];
        const Index target_seq   = dataset.get_shape("Target")[0];

        if(decoder_seq != target_seq)
            throw runtime_error("Decoder and target sequence lengths must match.");

        const Index embedding_dimension     = 640;
        const Index heads_number            = 10;
        const Index feed_forward_dimension  = 2560;
        const Index layers_number           = 6;

        Transformer transformer(input_seq,
                                decoder_seq,
                                input_vocab,
                                output_vocab,
                                embedding_dimension,
                                heads_number,
                                feed_forward_dimension,
                                layers_number);

        transformer.set_dropout_rate(float(0.1));

        cout << "[COMBINED3] train="  << dataset.get_samples_number("Training")
             << " val="               << dataset.get_samples_number("Validation")
             << " test="              << dataset.get_samples_number("Testing")
             << " input_vocab="       << input_vocab
             << " target_vocab="      << output_vocab
             << " input_seq="         << input_seq
             << " decoder_seq="       << decoder_seq
             << " params="            << transformer.get_parameters_number() << endl;

        const std::filesystem::path weights_file = "/tmp/combined3_weights.bin";

        if (std::filesystem::exists(weights_file))
        {
            cout << "Loading cached weights from " << weights_file
                 << " (skipping training)" << endl;
            transformer.load_parameters_binary(weights_file);
        }
        else
        {
            TrainingStrategy training_strategy(&transformer, &dataset);

            training_strategy.set_loss("CrossEntropyError3d");
            training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

            auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
            if(!adam) throw runtime_error("AdaptiveMomentEstimation optimizer not found.");

            adam->set_batch_size(64);     // safe headroom for 110M params + vocab 25K + decoder_seq=251
            adam->set_learning_rate(float(5e-4));
            adam->set_maximum_epochs(9);   // 10 epochs (loop runs 0..9 inclusive)
            adam->set_display_period(1);

            const auto t0 = steady_clock::now();
            training_strategy.train();
            const auto t1 = steady_clock::now();

            cout << "\nTotal training time (combined3): "
                 << duration_cast<milliseconds>(t1 - t0).count() / 1000.0 << " s" << endl;

            std::ofstream weights_out(weights_file, std::ios::binary);
            if (!weights_out)
                throw runtime_error("Cannot open " + weights_file.string() + " for writing");
            const Index parameters_floats = transformer.get_parameters_size();
            weights_out.write(reinterpret_cast<const char*>(transformer.get_parameters_data()),
                              parameters_floats * sizeof(float));
            cout << "Saved " << parameters_floats << " floats ("
                 << parameters_floats * sizeof(float) << " bytes) to "
                 << weights_file << endl;
        }

        cout << "\n================ COMBINED3 CHAT ================\n";

        TransformerDecoder::SamplingConfig sampling_config;
        sampling_config.temperature        = 0.8f;
        sampling_config.top_p              = 0.9f;
        sampling_config.top_k              = 50;
        sampling_config.repetition_penalty = 1.15f;

        TransformerDecoder decoder(transformer, dataset);
        decoder.chat(sampling_config);

        cout << "===============================================\n";

#endif

        cout << "Bye!" << endl;
        return 0;
    }
    catch(const exception& e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
