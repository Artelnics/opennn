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
        cout << "Mi prueba." << endl;

        /* NAME               CONTAMINATION           N POINTS
         *
         * mnist                    0.092               7603
         * satellite                0.31                6435
         * Datos2Dimensiones        0.1                 300
         * ionosphere               0.35                351
         * vowels                   0.034               1456
         * shuttle                  0.07                49097
         */

        srand(static_cast<unsigned>(time(nullptr)));
        unsigned t0, t1;
        string name = "shuttle";
        // Data set
        DataSet data_set("../data/"+name+".csv", ',', true);
        Index K = 250;
        K = (data_set.get_used_samples_number()*3.5)/100;
        K = K < 20 ? 20 : K;
        K = K > 500 ? 500 : K;
        type treshold = 1.3;
        type contamination = 0;


        Tensor<type, 2> true_outlier = data_set.get_column_data("outlier");
        data_set.set_column_use("outlier", data_set.UnusedVariable);


        const Index input_variables_number = data_set.get_input_variables_number();


        Tensor<string, 1> scaling_inputs_methods(input_variables_number);
        scaling_inputs_methods.setConstant("MinimumMaximum");

        //const Tensor<Descriptives, 1> inputs_descriptives = data_set.scale_input_variables(scaling_inputs_methods);

        t0 = clock();
        //Tensor<Index, 1> outliers = data_set.calculate_LocalOutlierFactor_outliers(K, treshold, data_set.get_used_samples_number());
        Tensor<Index, 1> outliers = data_set.calculate_LocalOutlierFactor_outliers(K, treshold, contamination, 2000);
        t1 = clock();

        double time = (double(t1-t0)/CLOCKS_PER_SEC);
        cout<<"Execution time:"<<time<<endl;


        type truePositives = 0;
        type falseNegatives = 0;
        type trueNegatives = 0;
        type falsePositives = 0;

        //cout<<"Outlier detectados:"<<endl;
        type count = 0;
        for(Index i = 0; i < outliers.size(); i++)
        {
            if(outliers(i)==1)
            {
                count++;
                //cout<<i<<", ";
            }
        }
        //cout<<endl;
        //cout<<"Outlier originales:" <<endl;
        type count2 = 0;
        for(Index i = 0; i < true_outlier.size(); i++)
        {
            if(true_outlier(i)==1)
            {
                count2++;
                //cout<<i<<", ";
            }
        }
        //cout<<endl;

        /*
        for(Index i = 0; i < outliers.size(); i++)
        {
            cout<<outliers[i]<<",";
        }
        cout<<endl;
        */

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
        cout<<"Precision:"<<truePositives/(truePositives+falseNegatives)<<endl;
        cout<<"Recall:"<<truePositives/(truePositives+falsePositives)<<endl;
        cout<<"Detected Number of Outlier:"<< count<<endl;
        cout<<"True Number of true outlier:"<<count2<<endl;
        cout<<"Total Number of Points:"<<data_set.get_used_samples_number()<<endl;
        cout<<"Detected contamination:"<< count / data_set.get_used_samples_number()<<endl;
        cout<<"Real contamination:"<< count2 / data_set.get_used_samples_number()<<endl;
        return 0;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;

        return 1;
    }
}
