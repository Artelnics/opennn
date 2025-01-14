//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A U T O   A S S O C I A T I O N   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "auto_associative_data_set.h"

namespace opennn
{

AutoAssociativeDataSet::AutoAssociativeDataSet() : DataSet()
{

}


void AutoAssociativeDataSet::transform_associative_data()
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
        memcpy(data.data() + (i - index) * samples_number,
               associative_data.data() + (i - index) * samples_number,
               samples_number * sizeof(type));

        memcpy(data.data() + samples_number * old_variables_number + (i - index) * samples_number,
               associative_data.data() + (i - index) * samples_number,
               samples_number * sizeof(type));
    }
}


void AutoAssociativeDataSet::transform_associative_raw_variables()
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
            new_raw_variables[index].set(raw_variables[raw_variable_index].name,
                                         DataSet::VariableUse::Input,
                                         raw_variables[raw_variable_index].type,
                                         raw_variables[raw_variable_index].scaler,
                                         raw_variables[raw_variable_index].categories);
        else
            new_raw_variables[index].set(raw_variables[raw_variable_index].name + "_output",
                                         DataSet::VariableUse::Target,
                                         raw_variables[raw_variable_index].type,
                                         raw_variables[raw_variable_index].scaler,
                                         raw_variables[raw_variable_index].categories);

        index++;
    }

    raw_variables = new_raw_variables;
}


void AutoAssociativeDataSet::set_auto_associative_samples_uses()
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

    vector<Index> indices;
    iota(indices.begin(), indices.end(), 0);

    shuffle(indices.data(), indices.data() + indices.size(), urng);

    Index i = 0;
    Index index;

    // Training

    Index count_training = 0;

    while(count_training != training_samples_number)
    {
        index = indices[i++];

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
        index = indices[i++];

        if(sample_uses[index] != SampleUse::None)
        {
            sample_uses[index] = SampleUse::Testing;
            count_testing++;
        }
    }
}


vector<DataSet::RawVariable> AutoAssociativeDataSet::get_associative_raw_variables() const
{
    return associative_raw_variables;
}


const Tensor<type, 2>& AutoAssociativeDataSet::get_associative_data() const
{
    return associative_data;
}


Index AutoAssociativeDataSet::get_associative_raw_variables_number() const
{
    return associative_raw_variables.size();
}


void AutoAssociativeDataSet::set_associative_data(const Tensor<type, 2>& new_data)
{
    associative_data = new_data;
}


void AutoAssociativeDataSet::set_associative_raw_variables_number(const Index& new_variables_number)
{
    associative_raw_variables.resize(new_variables_number);
}


void AutoAssociativeDataSet::save_auto_associative_data_binary(const filesystem::path& binary_data_file_name) const
{
    ofstream file(binary_data_file_name, ios::binary);

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


void AutoAssociativeDataSet::transform_associative_dataset()
{
    transform_associative_data();

    transform_associative_raw_variables();

    unuse_constant_raw_variables();

    set_auto_associative_samples_uses();
}


void AutoAssociativeDataSet::load_auto_associative_data_binary(const filesystem::path& auto_associative_data_file_name)
{

    ifstream file(auto_associative_data_file_name, ios::binary);

    if(!file.is_open())
        throw runtime_error("Cannot open binary file: " + auto_associative_data_file_name.string() + "\n");

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
