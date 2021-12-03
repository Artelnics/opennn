#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <omp.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main(void)
{
    try
    {
        cout << "Outliers test." << endl;

//         NAME               CONTAMINATION           N POINTS
//         arrhythmia               0.15                452
//         cover                    0.01                286048
//         cardio                   0.096               1831
//         ionosphere               0.35                351
//         lympho                   0.041               148
//         mnist                    0.092               7603
//         satellite                0.32                6435
//         vowels                   0.034               1456
//         SkDataSet0-4             0.15                500
//         Outlier2Dimensions       0.1                 300

        srand(static_cast<unsigned>(time(nullptr)));
        unsigned t0, t1;

        string name;

        cout <<
        "\nAvailable datasets:\n\n"

        "arrhythmia\n"
        "cover\n"
        "cardio\n"
        "ionosphere\n"
        "lympho\n"
        "mnist\n"
        "satellite\n"
        "vowels\n"
        "SkDataSet0\n"
        "SkDataSet1\n"
        "SkDataSet2\n"
        "SkDataSet3\n"
        "SkDataSet4\n"
        "Outlier2Dimensions\n"
             << endl;

        cin >> name;

        // Data set

        DataSet data_set("../data/"+name+".csv", ',', true);
        Index K = 20;
        type contamination = type(0);


        Tensor<type, 2> true_outlier = data_set.get_column_data("outlier");
        data_set.set_column_use("outlier", DataSet::VariableUse::UnusedVariable);

        const Index input_variables_number = data_set.get_input_variables_number();

        t0 = clock();

        Tensor<Index, 1> outliers = data_set.calculate_local_outlier_factor_outliers(K, data_set.get_used_samples_number(), contamination);

        t1 = clock();
        double time = (double(t1-t0)/CLOCKS_PER_SEC);

        type truePositives = type(0);
        type falseNegatives = type(0);
        type trueNegatives = type(0);
        type falsePositives = type(0);

        Index count = 0;

        for(Index i = 0; i < outliers.size(); i++)
        {
            if(outliers(i)==1)
            {
                count++;
            }
        }

        Index count2 = 0;

        for(Index i = 0; i < true_outlier.size(); i++)
        {
            if(true_outlier(i)==1)
            {
                count2++;
            }
        }

        for(Index i = 0; i < outliers.size(); i++)
        {
            if(true_outlier(i) == 1 && outliers(i) == 1) truePositives++;
            else if(true_outlier(i) == 1 && outliers(i) == 0) falseNegatives++;
            else if(true_outlier(i) == 0 && outliers(i) == 0) trueNegatives++;
            else falsePositives++;
        }

        cout << "\nRESULTS FOR: " << name<<" data_set\n" << endl;
        cout<<"Execution time:" << time << endl;
        cout << "Precision:" << truePositives/(truePositives+falseNegatives) << endl;
        cout << "Recall:" << truePositives/(truePositives+falsePositives) << endl;
        cout << "Detected Number of Outlier:" << count << endl;
        cout << "True Number of true outlier:" << count2 << endl;
        cout << "Total Number of Points:" <<data_set.get_used_samples_number() << endl;

        return 0;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;

        return 1;
    }
}
