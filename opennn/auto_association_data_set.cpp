//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A U T O   A S S O C I A T I O N   D A T A S E T   C L A S S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "auto_association_data_set.h"
#include "tensor_utilities.h"

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


/// This method duplicates the columns for association problems.

void AutoAssociationDataSet::transform_associative_columns()
{
    cout << "Transforming associative columns..." << endl;

    associative_columns = columns;

    const Index columns_number = get_columns_number();

    Tensor<RawVariable, 1> new_columns;

    new_columns.resize(2*columns_number);

    Index column_index = 0;
    Index index = 0;

    for(Index i = 0; i < 2*columns_number; i++)
    {
        column_index = i%columns_number;

        if(i < columns_number)
        {
            new_columns(index).name = columns(column_index).name;

            new_columns(index).categories_uses.resize(columns(column_index).get_categories_number());
            new_columns(index).set_use(DataSet::VariableUse::Input);
            new_columns(index).type = columns(column_index).type;
            new_columns(index).categories = columns(column_index).categories;
            index++;
        }
        else
        {
            new_columns(index).name = columns(column_index).name + "_output";

            new_columns(index).categories_uses.resize(columns(column_index).get_categories_number());
            new_columns(index).set_use(DataSet::VariableUse::Target);
            new_columns(index).type = columns(column_index).type;
            new_columns(index).categories = columns(column_index).categories;
            index++;
        }
    }

    columns = new_columns;
}


/// Sets the samples' uses in the auto associative data set.

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
    {
        ostringstream buffer;

        buffer << "OpenNN Warning: DataSet class.\n"
               << "void split_samples_random(const type&, const type&, const type&) method.\n"
               << "Sum of numbers of training, selection and testing samples is not equal to number of used samples.\n";

        throw invalid_argument(buffer.str());
    }

    const Index samples_number = get_samples_number();

    Tensor<Index, 1> indices;

    initialize_sequential(indices, 0, 1, samples_number-1);

    std::shuffle(indices.data(), indices.data() + indices.size(), urng);

    Index count = 0;

    for(Index i = 0; i < samples_uses.size(); i++)
    {
        if(samples_uses(i) == SampleUse::Unused) count++;
    }

    Index i = 0;
    Index index;

    // Training

    Index count_training = 0;

    while(count_training != training_samples_number)
    {
        index = indices(i);

        if(samples_uses(index) != SampleUse::Unused)
        {
            samples_uses(index)= SampleUse::Training;
            count_training++;
        }

        i++;
    }

    // Testing

    Index count_testing = 0;

    while(count_testing != testing_samples_number)
    {
        index = indices(i);

        if(samples_uses(index) != SampleUse::Unused)
        {
            samples_uses(index) = SampleUse::Testing;
            count_testing++;
        }

        i++;
    }
}

Tensor<DataSet::RawVariable, 1> AutoAssociationDataSet::get_associative_columns() const
{
    return associative_columns;
}


const Tensor<type, 2>& AutoAssociationDataSet::get_associative_data() const
{
    return associative_data;
}


Index AutoAssociationDataSet::get_associative_columns_number() const
{
    return associative_columns.size();
}


void AutoAssociationDataSet::set_associative_data(const Tensor<type, 2>& new_data)
{
    associative_data = new_data;
}


void AutoAssociationDataSet::set_associative_columns_number(const Index& new_variables_number)
{
    associative_columns.resize(new_variables_number);
}


/// Saves to the data file the values of the auto associative data matrix in binary format.

void AutoAssociationDataSet::save_auto_associative_data_binary(const string& binary_data_file_name) const
{
    std::ofstream file(binary_data_file_name.c_str(), ios::binary);

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class." << endl
               << "void save_auto_associative_data_binary(const string) method." << endl
               << "Cannot open data binary file." << endl;

        throw invalid_argument(buffer.str());
    }

    // Write data

    streamsize size = sizeof(Index);

    Index columns_number = associative_data.dimension(1);
    Index rows_number = associative_data.dimension(0);

    cout << "Saving binary associative data file..." << endl;

    file.write(reinterpret_cast<char*>(&columns_number), size);
    file.write(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(type);

    type value;

    for(int i = 0; i < columns_number; i++)
    {
        for(int j = 0; j < rows_number; j++)
        {
            value = associative_data(j,i);

            file.write(reinterpret_cast<char*>(&value), size);
        }
    }

    file.close();

    cout << "Binary data file saved." << endl;
}


/// Arranges an input-target DataSet from a time series matrix, according to the number of lags.

void AutoAssociationDataSet::transform_associative_dataset()
{
    transform_associative_data();

    transform_associative_columns();

    unuse_constant_columns();

    set_auto_associative_samples_uses();
}


/// This method loads associative data from a binary data file.

void AutoAssociationDataSet::load_auto_associative_data_binary(const string& auto_associative_data_file_name)
{
    std::ifstream file;

    file.open(auto_associative_data_file_name.c_str(), ios::binary);

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "void load_auto_associative_data_binary(const string&) method.\n"
               << "Cannot open binary file: " << auto_associative_data_file_name << "\n";

        throw invalid_argument(buffer.str());
    }

    streamsize size = sizeof(Index);

    Index columns_number;
    Index rows_number;

    file.read(reinterpret_cast<char*>(&columns_number), size);
    file.read(reinterpret_cast<char*>(&rows_number), size);

    size = sizeof(type);

    type value;

    associative_data.resize(rows_number, columns_number);

    for(Index i = 0; i < rows_number*columns_number; i++)
    {
        file.read(reinterpret_cast<char*>(&value), size);

        associative_data(i) = value;
    }

    file.close();
}



}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2023 Artificial Intelligence Techniques, SL.
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
