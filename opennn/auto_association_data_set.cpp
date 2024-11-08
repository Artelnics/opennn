//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A U T O   A S S O C I A T I O N   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <fstream>
#include <iostream>

#include "auto_association_data_set.h"
#include "tensors.h"

namespace opennn
{

AutoAssociationDataSet::AutoAssociationDataSet() : DataSet()
{

}


void AutoAssociationDataSet::transform_associative_data()
{
    cout << "Transforming associative data..." << endl;

    const Index samples_number = data.dimension(0);

    const Index old_variables_number = data.dimension(1);
    const Index new_variables_number = 2 * old_variables_number;

    associative_data = data;

    data.resize(samples_number, new_variables_number);

    // Duplicate data

    Index index = 0;

    for(Index i = 0; i < old_variables_number; i++)
    {
        copy(associative_data.data() + (i - index) * samples_number,
             associative_data.data() + (i + 1 - index) *  samples_number,
             data.data() + (i - index) * samples_number);

        copy(associative_data.data() + (i - index) * samples_number,
             associative_data.data() + (i + 1 - index) *  samples_number,
             data.data() + samples_number * old_variables_number + (i - index) * samples_number);
    }
}


void AutoAssociationDataSet::transform_associative_raw_variables()
{
    cout << "Transforming associative raw variables..." << endl;

    associative_raw_variables = raw_variables;

    const Index raw_variables_number = get_raw_variables_number();

    vector<RawVariable> new_raw_variables(2 * raw_variables_number);

    Index raw_variable_index = 0;
    Index index = 0;

    for(Index i = 0; i < 2*raw_variables_number; i++)
    {
        raw_variable_index = i%raw_variables_number;

        if(i < raw_variables_number)
        {
            new_raw_variables[index].name = raw_variables[raw_variable_index].name;

            new_raw_variables[index].set_use(DataSet::VariableUse::Input);
            new_raw_variables[index].type = raw_variables[raw_variable_index].type;
            new_raw_variables[index].categories = raw_variables[raw_variable_index].categories;
        }
        else
        {
            new_raw_variables[index].name = raw_variables[raw_variable_index].name + "_output";
            new_raw_variables[index].set_use(DataSet::VariableUse::Target);
            new_raw_variables[index].type = raw_variables[raw_variable_index].type;
            new_raw_variables[index].categories = raw_variables[raw_variable_index].categories;
        }

        index++;
    }

    raw_variables = new_raw_variables;
}


void AutoAssociationDataSet::set_auto_associative_samples_uses()
{
    random_device rng;

    mt19937 urng(rng());

    const Index used_samples_number = get_used_samples_number();

    if(used_samples_number == 0) return;

    const type training_samples_ratio = type(0.8);
    const type testing_samples_ratio = type(0.2);

    const type total_ratio = training_samples_ratio + testing_samples_ratio;

    // Get number of samples for training and testing

    const Index testing_samples_number = Index(testing_samples_ratio* type(used_samples_number)/ type(total_ratio));
    const Index training_samples_number = used_samples_number - testing_samples_number;

    const Index sum_samples_number = training_samples_number + testing_samples_number;

    if(sum_samples_number != used_samples_number)
        throw runtime_error("Sum of numbers of training, selection and testing samples is not equal to number of used samples.\n");

    const Index samples_number = get_samples_number();

    Tensor<Index, 1> indices;

    initialize_sequential(indices, 0, 1, samples_number-1);

    std::shuffle(indices.data(), indices.data() + indices.size(), urng);

    Index i = 0;
    Index index;

    // Training

    Index count_training = 0;

    while(count_training != training_samples_number)
    {
        index = indices(i++);

        if(sample_uses[index] != SampleUse::None)
        {
            sample_uses[index] = SampleUse::Training;
            count_training++;
        }
    }

    // Testing

    Index count_testing = 0;

    while(count_testing != testing_samples_number)
    {
        index = indices(i++);

        if(sample_uses[index] != SampleUse::None)
        {
            sample_uses[index] = SampleUse::Testing;
            count_testing++;
        }
    }
}


vector<DataSet::RawVariable> AutoAssociationDataSet::get_associative_raw_variables() const
{
    return associative_raw_variables;
}


const Tensor<type, 2>& AutoAssociationDataSet::get_associative_data() const
{
    return associative_data;
}


Index AutoAssociationDataSet::get_associative_raw_variables_number() const
{
    return associative_raw_variables.size();
}


void AutoAssociationDataSet::set_associative_data(const Tensor<type, 2>& new_data)
{
    associative_data = new_data;
}


void AutoAssociationDataSet::set_associative_raw_variables_number(const Index& new_variables_number)
{
    associative_raw_variables.resize(new_variables_number);
}


void AutoAssociationDataSet::save_auto_associative_data_binary(const string& binary_data_file_name) const
{
    ofstream file(binary_data_file_name.c_str(), ios::binary);

    if(!file.is_open())
        throw runtime_error("Cannot open data binary file.");

    // Write data

    streamsize size = sizeof(Index);

    Index columns_number = associative_data.dimension(1);
    Index rows_number = associative_data.dimension(0);

    cout << "Saving binary associative data file..." << endl;

    file.write(reinterpret_cast<char*>(&columns_number), size);
    file.write(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(type);

    const Index total_elements = columns_number * rows_number;

    file.write(reinterpret_cast<const char*>(associative_data.data()), total_elements * size);

    file.close();

    cout << "Binary data file saved." << endl;
}


void AutoAssociationDataSet::transform_associative_dataset()
{
    transform_associative_data();

    transform_associative_raw_variables();

    unuse_constant_raw_variables();

    set_auto_associative_samples_uses();
}


void AutoAssociationDataSet::load_auto_associative_data_binary(const string& auto_associative_data_file_name)
{
    ifstream file;

    file.open(auto_associative_data_file_name.c_str(), ios::binary);

    if(!file.is_open())
        throw runtime_error("Cannot open binary file: " + auto_associative_data_file_name + "\n");

    streamsize size = sizeof(Index);

    Index columns_number = 0;
    Index rows_number = 0;

    file.read(reinterpret_cast<char*>(&columns_number), size);
    file.read(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(type);

    const Index total_elements = rows_number * columns_number;

    file.read(reinterpret_cast<char*>(associative_data.data()), total_elements * size);

    file.close();
}

}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2024 Artificial Intelligence Techniques, SL.
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
