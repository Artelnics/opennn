//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   C U D A
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>

#include "../opennn/language_dataset.h"
#include "../opennn/standard_networks.h"
#include "../opennn/loss.h"
#include "../opennn/training_strategy.h"
#include "../opennn/adaptive_moment_estimation.h"

using namespace opennn;

static constexpr Index PAD_TOKEN = 0;
static constexpr Index UNK_TOKEN = 1;
static constexpr Index START_TOKEN = 2;
static constexpr Index END_TOKEN = 3;

static string decode_ids(const VectorR& row, const vector<string>& vocabulary, bool skip_start = false)
{
    ostringstream buffer;

    for(Index i = 0; i < row.size(); i++)
    {
        const Index token = static_cast<Index>(row(i));

        if(token == PAD_TOKEN) continue;
        if(skip_start && token == START_TOKEN) continue;

        if(token >= 0 && token < static_cast<Index>(vocabulary.size()))
            buffer << vocabulary[token];
        else
            buffer << "[OOB]";

        if(token == END_TOKEN) break;
        buffer << " ";
    }

    return buffer.str();
}

static VectorR argmax_decode_tokens(const Tensor3& outputs)
{
    const Index sequence_length = outputs.dimension(1);
    const Index vocabulary_size = outputs.dimension(2);

    VectorR predicted(sequence_length);
    predicted.setZero();

    for(Index t = 0; t < sequence_length; t++)
    {
        type best_value = outputs(0, t, 0);
        Index best_index = 0;

        for(Index k = 1; k < vocabulary_size; k++)
        {
            if(outputs(0, t, k) > best_value)
            {
                best_value = outputs(0, t, k);
                best_index = k;
            }
        }

        predicted(t) = static_cast<type>(best_index);
    }

    return predicted;
}

static void print_prediction_report(const string& title,
                                    const VectorR& context_row,
                                    const VectorR& decoder_row,
                                    const VectorR& target_row,
                                    const VectorR& predicted_row,
                                    const vector<string>& input_vocabulary,
                                    const vector<string>& target_vocabulary)
{
    cout << "\n== " << title << " ==\n";

    cout << "Context ids:   " << context_row.transpose() << endl;
    cout << "Decoder ids:   " << decoder_row.transpose() << endl;
    cout << "Target ids:    " << target_row.transpose() << endl;
    cout << "Predicted ids: " << predicted_row.transpose() << endl;

    cout << "Context text:   " << decode_ids(context_row, input_vocabulary, false) << endl;
    cout << "Decoder text:   " << decode_ids(decoder_row, target_vocabulary, false) << endl;
    cout << "Target text:    " << decode_ids(target_row, target_vocabulary, true) << endl;
    cout << "Predicted text: " << decode_ids(predicted_row, target_vocabulary, true) << endl;

    cout << "===\n";
}

int main()
{
    try
    {
        cout << "OpenNN. Blank Cuda." << endl;
#ifdef CUDA

        cout << "OpenNN. Transformer training on OPUS Books en-es." << endl;

        // ---------------------------------------------------------------------
        // Dataset
        // ---------------------------------------------------------------------

        const string data_path = "/home/artelnics/Documents/opus_books_es_en_small_transformer.txt";
        LanguageDataset language_dataset(data_path);

        // language_dataset.print(); // @todo

        const Index input_vocabulary_size  = language_dataset.get_input_vocabulary_size();
        const Index output_vocabulary_size = language_dataset.get_target_vocabulary_size();

        const Index input_sequence_length   = language_dataset.get_shape("Input")[0];
        const Index decoder_sequence_length = language_dataset.get_shape("Decoder")[0];
        const Index target_sequence_length  = language_dataset.get_shape("Target")[0];

        cout << "\nDataset summary" << endl;
        cout << "Input vocabulary size:  " << input_vocabulary_size << endl;
        cout << "Output vocabulary size: " << output_vocabulary_size << endl;
        cout << "Input sequence length:  " << input_sequence_length << endl;
        cout << "Decoder sequence length:" << decoder_sequence_length << endl;
        cout << "Target sequence length: " << target_sequence_length << endl;

        if(decoder_sequence_length != target_sequence_length)
            throw runtime_error("Decoder and target sequence lengths must match.");

        // ---------------------------------------------------------------------
        // Transformer
        // ---------------------------------------------------------------------

        const Index embedding_dimension = 256;
        const Index heads_number = 2;
        const Index feed_forward_dimension = 1024;
        const Index layers_number = 1;

        Transformer transformer(input_sequence_length,
                                decoder_sequence_length,
                                input_vocabulary_size,
                                output_vocabulary_size,
                                embedding_dimension,
                                heads_number,
                                feed_forward_dimension,
                                layers_number);
        cout << "parameters number: " << transformer.get_parameters_number() << endl;
        transformer.set_input_vocabulary(language_dataset.get_input_vocabulary());
        transformer.set_output_vocabulary(language_dataset.get_target_vocabulary());

        // ---------------------------------------------------------------------
        // Training strategy
        // ---------------------------------------------------------------------

        TrainingStrategy training_strategy(&transformer, &language_dataset);

        training_strategy.set_loss("CrossEntropy");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());

        if(!adam)
            throw runtime_error("AdaptiveMomentEstimation optimizer not found.");

        adam->set_batch_size(16);
        adam->set_learning_rate(0.0005);
        adam->set_maximum_epochs(10);
        adam->set_display_period(1);

#ifdef CUDA
        cout << "\nTraining on CPU..." << endl;
        training_strategy.train();
#else
        cout << "\nTraining on CPU..." << endl;
        training_strategy.train();
#endif

        // ---------------------------------------------------------------------
        // Predictions
        // ---------------------------------------------------------------------

        cout << "\n== TRANSFORMER PREDICTIONS ==\n";

        const vector<string> test_sources =
            {
                "Su madre era hermana de mi padre",
                "Ahora la riqueza no era ya un peso para mí",
                "Trabajar en lo que está a mi alcance"
            };

        for(Index i = 0; i < static_cast<Index>(test_sources.size()); i++)
        {
            const string prediction = transformer.calculate_outputs(test_sources[i]);

            cout << "Sample " << i << endl;
            cout << "  Source:    " << test_sources[i] << endl;
            cout << "  Predicted: " << prediction << endl;
            cout << endl;
        }

        cout << "=\n";
        cout << "\nDone." << endl;

#endif

        cout << "Bye!" << endl;

        return 0;
    }
    catch (const exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}
