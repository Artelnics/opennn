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
        /* NAME               CONTAMINATION           N POINTS
         *
         * arrhythmia               0.15                452
         * cover                    0.01                286048
         * cardio                   0.096               1831
         * ionosphere               0.35                351
         * lympho                   0.041               148
         * mnist                    0.092               7603
         * satellite                0.32                6435
         * vowels                   0.034               1456
         * shuttle                  0.07                49097
         * SkDataSet0-4             0.15                500
         * Outlier2Dimensions       0.1                 300
         */

        srand(static_cast<unsigned>(time(nullptr)));
        unsigned t0, t1;
        string name = "ionosphere";
        // Data set
        DataSet data_set("../data/"+name+".csv", ',', true);
        Index K = 20;
        //K = (data_set.get_used_samples_number()*3.5)/100;
        //K = K < 20 ? 20 : K;
        //K = K > 500 ? 500 : K;
        type contamination = 0;


        Tensor<type, 2> true_outlier = data_set.get_column_data("outlier");
        data_set.set_column_use("outlier", data_set.UnusedVariable);

        const Index input_variables_number = data_set.get_input_variables_number();

       /* Tensor<string, 1> scaling_inputs_methods(input_variables_number);
        scaling_inputs_methods.setConstant("MinimumMaximum");

        const Tensor<Descriptives, 1> inputs_descriptives = data_set.scale_input_variables(scaling_inputs_methods);*/

        t0 = clock();
        Tensor<Index, 1> outliers = data_set.calculate_local_outlier_factor_outliers(K, data_set.get_used_samples_number(), contamination);
        t1 = clock();

        double time = (double(t1-t0)/CLOCKS_PER_SEC);
        cout<<"Execution time:"<<time<<endl;


        type truePositives = 0;
        type falseNegatives = 0;
        type trueNegatives = 0;
        type falsePositives = 0;


        type count = 0;

        /*
        for(Index i = 0; i < outliers.size(); i++)
        {
            cout<<outliers(i)<<",";
        }
        cout<<endl;
    */
        for(Index i = 0; i < outliers.size(); i++)
        {
            if(outliers(i)==1)
            {
                count++;
            }
        }

        type count2 = 0;
        for(Index i = 0; i < true_outlier.size(); i++)
        {
            if(true_outlier(i)==1)
            {
                count2++;
            }
        }



        for(Index i = 0; i < outliers.size(); i++)
        {
            if(true_outlier(i) == 1 && outliers(i) == 1)
            {
                truePositives++;
            }
            else if(true_outlier(i) == 1 && outliers(i) == 0)
            {
                falseNegatives++;
            }
            else if(true_outlier(i) == 0 && outliers(i) == 0)
            {
                trueNegatives++;
            }
            else
            {
                falsePositives++;
            }
        }

        cout<<"RESULTS FOR: "<< name<<" data_set"<<endl;
        cout<<"Precision:"<<truePositives/(truePositives+falseNegatives)<<endl;
        cout<<"Recall:"<<truePositives/(truePositives+falsePositives)<<endl;
        cout<<"Detected Number of Outlier:"<< count<<endl;
        cout<<"True Number of true outlier:"<<count2<<endl;
        cout<<"Total Number of Points:"<<data_set.get_used_samples_number()<<endl;
        return 0;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;

        return 1;
    }
}
