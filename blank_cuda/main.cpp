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
        
#ifdef OPENNN_CUDA

        cout << "OpenNN. Amazon reviews example." << endl;

        // Settings

        const Index embedding_dimension = 64;
        const Index heads_number = 4;

        // Dataset

        LanguageDataset language_dataset("/home/artelnics/Documents/opennn/examples/amazon_reviews/data/amazon_cells_labelled.txt");
        const Index input_vocabulary_size = language_dataset.get_input_vocabulary_size();
        const Index input_sequence_length = language_dataset.get_maximum_input_sequence_length();
        const Index targets_number = language_dataset.get_maximum_target_sequence_length();
        const vector<string> input_vocabulary = language_dataset.get_input_vocabulary();

        // Neural Network

        TextClassificationNetwork text_classification_network(
            {input_vocabulary_size, input_sequence_length, embedding_dimension},
            {heads_number},
            {targets_number},
            input_vocabulary);

        // Training Strategy

        TrainingStrategy training_strategy(&text_classification_network, &language_dataset);

        AdaptiveMomentEstimation* adam = dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        adam->set_maximum_epochs(100);
        adam->set_display_period(10);

        WeightedSquaredError wse;

        training_strategy.set_loss("CrossEntropyError2d");

        cout << "Training network..." << endl;
        training_strategy.train_cuda();
        //training_strategy.train();

        // Testing Analysis

        TestingAnalysis testing_analysis(&text_classification_network, &language_dataset);
        cout << "Confusion Matrix:" << endl;
        cout << testing_analysis.calculate_confusion() << endl;

        Tensor<string, 1> documents(1);
        documents[0] = "This product is amazing and I love it!";
        MatrixR outputs = text_classification_network.calculate_text_outputs(documents);

        cout << "Prediction for 'This product is amazing': " << outputs(0,0) << endl;

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
