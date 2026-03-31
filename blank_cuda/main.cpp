//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   C U D A
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "../opennn/opennn.h"
#include <iostream>

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Blank Cuda." << endl;
        WeightedSquaredError();
#ifdef OPENNN_CUDA

        cout << "OpenNN. Transformer test." << endl;

        // ---------------------------------------------------------------------
        // Dataset
        // ---------------------------------------------------------------------

        const string data_path = "/home/artelnics/Documents/wmt14_de_en.txt";
        LanguageDataset language_dataset(data_path);

        const Index input_vocabulary_size  = language_dataset.get_input_vocabulary_size();
        const Index output_vocabulary_size = language_dataset.get_target_vocabulary_size();

        const Index input_sequence_length   = language_dataset.get_shape("Input")[0];
        const Index decoder_sequence_length = language_dataset.get_shape("Decoder")[0];
        const Index target_sequence_length  = language_dataset.get_shape("Target")[0];

        if(decoder_sequence_length != target_sequence_length)
            throw runtime_error("Decoder and target sequence lengths must match.");

        // ---------------------------------------------------------------------
        // Transformer
        // ---------------------------------------------------------------------

        const Index embedding_dimension = 128;
        const Index heads_number = 8;
        const Index feed_forward_dimension = 256;
        const Index layers_number = 1;

        cout << "Creando transformer..." << endl;
        Transformer transformer(input_sequence_length,
                                decoder_sequence_length,
                                input_vocabulary_size,
                                output_vocabulary_size,
                                embedding_dimension,
                                heads_number,
                                feed_forward_dimension,
                                layers_number);

        transformer.set_input_vocabulary(language_dataset.get_input_vocabulary());
        transformer.set_output_vocabulary(language_dataset.get_target_vocabulary());

        // ---------------------------------------------------------------------
        // Training strategy
        // ---------------------------------------------------------------------

        TrainingStrategy training_strategy(&transformer, &language_dataset);

        training_strategy.set_loss("CrossEntropyError3d");
        training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");

        auto* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());

        if(!adam)
            throw runtime_error("AdaptiveMomentEstimation optimizer not found.");

        adam->set_batch_size(16);
        adam->set_learning_rate(0.0005);
        adam->set_maximum_epochs(20);
        adam->set_display_period(1);

#ifdef OPENNN_CUDA
        cout << "\nTraining..." << endl;
        training_strategy.train_cuda();
#else
        cout << "\nTraining on CPU..." << endl;
        training_strategy.train();
#endif

        // ---------------------------------------------------------------------
        // Predictions
        // ---------------------------------------------------------------------
    /*
        cout << "\n================ TRANSFORMER PREDICTIONS ================\n";

        const vector<string> test_sources =
            {
                "yo tengo hambre",
                "tu estas feliz",
                "el esta cansado",
                "el perro es grande",
                "yo veo el gato"
            };

        for(Index i = 0; i < static_cast<Index>(test_sources.size()); i++)
        {
            const string prediction = transformer.calculate_outputs(test_sources[i]);

            cout << "Sample " << i << endl;
            cout << "  Source:    " << test_sources[i] << endl;
            cout << "  Predicted: " << prediction << endl;
            cout << endl;
        }

        cout << "=========================================================\n";
        */
        cout << "\n================ TRANSFORMER PREDICTIONS (DE -> EN) ================\n";

        const vector<string> test_sources =
            {
                "ich bin müde",          // i am tired
                "das ist ein haus",      // that is a house
                "der hund ist groß",     // the dog is big
                "ich sehe die stadt",    // i see the city
                "wir gehen nach hause"   // we go home
            };

        for(Index i = 0; i < static_cast<Index>(test_sources.size()); i++)
        {
            const string prediction = transformer.calculate_outputs(test_sources[i]);

            cout << "Sample " << i << endl;
            cout << "  Source (DE):    " << test_sources[i] << endl;
            cout << "  Predicted (EN): " << prediction << endl;
            cout << endl;
        }

        cout << "====================================================================\n";

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
