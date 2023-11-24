//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <chrono>

// OpenNN includes

#include "../opennn/opennn.h"

#include <iostream>

using namespace std;
using namespace OpenNN;


//Tensor<type, 2> box_plots_to_tensor(const Tensor<BoxPlot, 1>& box_plots)
//{
//    const Index columns_number = box_plots.dimension(0);

//    Tensor<type, 2> summary(5, columns_number);

//    for(Index i = 0; i < columns_number; i++)
//    {
//        const BoxPlot& box_plot = box_plots(i);
//        summary(0, i) = box_plot.minimum;
//        summary(1, i) = box_plot.first_quartile;
//        summary(2, i) = box_plot.median;
//        summary(3, i) = box_plot.third_quartile;
//        summary(4, i) = box_plot.maximum;
//    }

//    //todo
//    Eigen::array<Index, 2> new_shape = {1, 5 * columns_number};
//    Tensor<type, 2> reshaped_summary = summary.reshape(new_shape);

//    return reshaped_summary;
//}


int main()
{
   try
   {
        cout << "Blank\n";      

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set("/Users/alvaros/Desktop/PoC/data_100_samples.csv", ',', true);

        Tensor<type, 2> data = data_set.get_data();

        map<string, DataSet> groupedData = data_set.group_by(data_set, "block_id");

        Tensor<Index, 1> selected_column_indices(data_set.get_columns_number() - 1);

        for(Index i = 0; i < selected_column_indices.size(); i++)
        {
            selected_column_indices(i) = i + 1;
        }


        Tensor<type, 2> merged_data;
        Tensor<type, 2> current_box_plots;

        bool is_first_iteration =true;

        for(auto& pair : groupedData)
        {
            DataSet& subset = pair.second;

            subset.set_data(subset.get_columns_data(selected_column_indices));

            current_box_plots = box_plots_to_tensor(subset.calculate_columns_box_plots());

            if (is_first_iteration)
            {
                merged_data = std::move(current_box_plots);
                is_first_iteration = false;
            }
            else
            {

                Tensor<type, 2> temp = merged_data.concatenate(std::move(current_box_plots), 0);
                merged_data = temp;
            }


        }

        Index clusters = 10;

        KMeans model(clusters);
        Tensor<type, 1> eblow = model.elbow_method(merged_data, clusters);
        Index optimal_clusters = model.find_optimal_clusters(eblow);

        cout << "optimal cluster: " << optimal_clusters << endl;

        model.set_cluster_number(optimal_clusters);
        model.fit(merged_data);
        cout << "outputs: " << model.calculate_outputs(merged_data);

        cout << "Bye!" << endl;

        return 0;
   }
   catch (const exception& e)
   {
       cerr << e.what() << endl;

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
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
