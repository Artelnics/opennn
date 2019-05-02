/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   F I L E   U T I L I T I E S    C L A S S                                                                   */
/*                                                                                                              */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   artelnics@artelnics.com                                                                                    */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "file_utilities.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor. 

FileUtilities::FileUtilities()
{
    display = true;
}


FileUtilities::FileUtilities(const string& new_file_name)
{
    file_name = new_file_name;
    display = true;
}


// DESTRUCTOR

/// Destructor.

FileUtilities::~FileUtilities()
{
}


void FileUtilities::set_file_name(const string& new_file_name)
{
    file_name = new_file_name;
}


size_t FileUtilities::count_lines_number() const
{
    ifstream file(file_name.c_str());

    if(!file.is_open())
    {
       ostringstream buffer;

       buffer << "OpenNN Exception: FileUtilities class.\n"
              << "size_t count_rows_number() const method.\n"
              << "Cannot open data file: " << file_name << "\n";

       throw logic_error(buffer.str());
    }

    if(file.peek() == ifstream::traits_type::eof())
    {
       return(0);
    }

    size_t count = 0;

    string line;

    while(file.good())
    {
        getline(file, line);

        count++;
    }

    return(count);
}


Vector<string> FileUtilities::get_output_file_names(const size_t& files_number) const
{
   Vector<string> names(files_number);

   Vector<ofstream> files(files_number);

   ostringstream buffer;

   for(size_t i = 0; i < files_number; i++)
   {
       buffer.str("");

       buffer << file_name << i;

       names[i] = buffer.str();
   }

   return(names);
}


string FileUtilities::read_header() const
{
    ifstream input_file(file_name.c_str());

    string header;

    getline(input_file, header);

    return(header);
}


Vector<string> FileUtilities::split_file(const size_t& output_files_number) const
{
    ifstream input_file(file_name.c_str());

    const size_t lines_number = count_lines_number();

    const Vector<string> output_file_names = get_output_file_names(output_files_number);

    Vector<ofstream> output_files(output_files_number);

    for(size_t i = 0; i < output_files_number; i++)
    {
        output_files[i].open(output_file_names[i].c_str());
    }

    Vector<size_t> minimums(output_files_number);
    Vector<size_t> maximums(output_files_number);

    const size_t minimum = 0;

    const double length = lines_number /static_cast<double>(output_files_number);

    minimums[0] = static_cast<size_t>(minimum);
    maximums[0] = static_cast<size_t>(minimum + length);

    // Calculate bins center

    for(size_t i = 1; i < output_files_number; i++)
    {
      minimums[i] = minimums[i - 1] + static_cast<size_t>(length);
      maximums[i] = maximums[i - 1] + static_cast<size_t>(length);
    }

    string line;

    getline(input_file, line);

    for(size_t i = 0; i < output_files_number; i++)
    {
        output_files[i] << line << endl;
    }

    for(size_t i = 1; i < lines_number; i++)
    {
        getline(input_file, line);

      for(size_t j = 0; j < output_files_number - 1; j++)
      {
        if(i >= minimums[j] && i < maximums[j])
        {
            output_files[j] << line << endl;
        }
      }

      if(i >= minimums[output_files_number - 1])
      {
          output_files[output_files_number-1] << line << endl;
      }
    }

    // Close files

    input_file.close();

    for(size_t i = 0; i < output_files_number; i++)
    {
        output_files[i].close();
    }

    return(output_file_names);
}


void FileUtilities::merge_files(const Vector<string>& input_file_names) const
{
    const size_t input_files_number = input_file_names.size();

    Vector<ifstream> input_files(input_files_number);

    for(size_t i = 0; i < input_files_number; i++)
    {
        input_files[i].open(input_file_names[i].c_str());
    }

    ofstream output_file(file_name.c_str());

    string line;

    for(size_t i = 0; i < input_files_number; i++)
    {
        // Header

        getline(input_files[i], line);

        if(i == 0)
        {
            output_file << line << endl;
        }

        // Contents

        while(input_files[i].good())
        {
            getline(input_files[i], line);

            output_file << line << endl;
        }
    }
}


void FileUtilities::replace(const string& find_what, const string& replace_with) const
{
    ifstream input_file(file_name.c_str());

    string output_file_name = file_name + "replace";

    ofstream output_file(output_file_name.c_str());

    string line;

    while(input_file.good())
    {
        getline(input_file, line);

        size_t position = 0;

        while((position = line.find(find_what, position)) != string::npos)
        {
             line.replace(position, find_what.length(), replace_with);

             position += replace_with.length();
        }

        output_file << line << endl;
    }

    // Close files

    input_file.close();
    output_file.close();

    // Clean files

    remove(file_name.c_str());
    rename(output_file_name.c_str(), file_name.c_str());
}


void FileUtilities::sample_file(const size_t& sample_lines_number) const
{
    const size_t lines_number = count_lines_number();

    ifstream input_file(file_name.c_str());

    string output_file_name = file_name + "sample";

    ofstream output_file(output_file_name.c_str());

    string line;

    size_t line_number = 0;

    const Vector<size_t> sample_lines(0,static_cast<double>(lines_number/sample_lines_number), lines_number);

    while(input_file.good())
    {
        getline(input_file, line);

        if(sample_lines.contains(line_number))
        {
            output_file << line << endl;
        }

        line_number++;
    }

    // Close files

    input_file.close();
    output_file.close();
}

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2018 Artificial Intelligence Techniques, SL.
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
