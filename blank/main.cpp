//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes
#include <QApplication>
#include <QLabel>

#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

// OpenNN includes

#include "../opennn/opennn.h"
#include "../opennn/layer.h"

using namespace opennn;
using namespace std;
using namespace Eigen;

#include "data_set.h"

 Tensor<unsigned char, 1> remove_padding(Tensor<unsigned char, 1>& img, const int& rows_number,const int& cols_number, const int& padding)
{
    Tensor<unsigned char, 1> data_without_padding(img.size() - padding*rows_number);

    const int channels = 3;

    if (rows_number % 4 ==0)
    {
        memcpy(data_without_padding.data(), img.data(), static_cast<size_t>(cols_number*channels*rows_number)*sizeof(unsigned char));
    }
    else
    {
        for (int i = 0; i<rows_number; i++)
        {
            if(i==0)
            {
                memcpy(data_without_padding.data(), img.data(), static_cast<size_t>(cols_number*channels)*sizeof(unsigned char));
            }
            else
            {
                memcpy(data_without_padding.data() + channels*cols_number*i, img.data() + channels*cols_number*i + padding*i, static_cast<size_t>(cols_number*channels)*sizeof(unsigned char));
             }
        }
     }
    return data_without_padding;
}


void sort_channel(Tensor<unsigned char,1>& original, Tensor<unsigned char,1>&sorted, const int& cols_number)
{
    unsigned char* aux_row = nullptr;

    aux_row = (unsigned char*)malloc(static_cast<size_t>(cols_number*sizeof(unsigned char)));

    const int rows_number = static_cast<int>(original.size()/cols_number);

    for(int i = 0; i <rows_number; i++)
    {
        memcpy(aux_row, original.data() + cols_number*rows_number - (i+1)*cols_number , static_cast<size_t>(cols_number)*sizeof(unsigned char));

//        reverse(aux_row, aux_row + cols_number); //uncomment this if the lower right corner px should be in the upper left corner.

        memcpy(sorted.data() + cols_number*i , aux_row, static_cast<size_t>(cols_number)*sizeof(unsigned char));
    }

}

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    try
    {
        cout<<"hello_world"<<endl;

        DataSet dataset;

        cout << "Reading data" << endl;

        const string filename = "C:/Users/alvaromartin/Documents/opennn/blank/data/test_padding.bmp";

        Tensor<unsigned char, 1> data = dataset.read_bmp_image(filename);

        const int channels = static_cast<int>(dataset.get_channels_number());
        const int rows_number = static_cast<int>(dataset.get_image_height());
        const int cols_number = static_cast<int>(dataset.get_image_width());

        const int padding = static_cast<int>(dataset.get_image_padding());

        cout << "Padding: " << padding << endl;

        Tensor<unsigned char, 1> data_without_padding = remove_padding(data, rows_number, cols_number, padding);

        cout << "channels: " << channels << endl;
        cout << "rows_number: " << rows_number << endl;
        cout << "cols_number: " << cols_number << endl;

        const Eigen::array<Eigen::Index, 3> dims_3D = {channels, rows_number, cols_number};
        const Eigen::array<Eigen::Index, 1> dims_1D = {rows_number*cols_number};

        Tensor<unsigned char,1> red_channel_flatted = data_without_padding.reshape(dims_3D).chip(2,0).reshape(dims_1D); // row_major
        Tensor<unsigned char,1> green_channel_flatted = data_without_padding.reshape(dims_3D).chip(1,0).reshape(dims_1D); // row_major
        Tensor<unsigned char,1> blue_channel_flatted = data_without_padding.reshape(dims_3D).chip(0,0).reshape(dims_1D); // row_major

        Tensor<unsigned char,1> red_channel_flatted_sorted(red_channel_flatted.size());
        Tensor<unsigned char,1> green_channel_flatted_sorted(green_channel_flatted.size());
        Tensor<unsigned char,1> blue_channel_flatted_sorted(blue_channel_flatted.size());

        red_channel_flatted_sorted.setZero();
        green_channel_flatted_sorted.setZero();
        blue_channel_flatted_sorted.setZero();

        sort_channel(red_channel_flatted, red_channel_flatted_sorted, cols_number);
        sort_channel(green_channel_flatted, green_channel_flatted_sorted, cols_number);
        sort_channel(blue_channel_flatted, blue_channel_flatted_sorted,cols_number);

        uint color = 0;
        QImage img(cols_number, rows_number , QImage::Format_RGB32);

        int row=0;
        int col=0;

        for(int i=0;i<rows_number*cols_number;i++) // move this to neuraldesigner for plotting images.
        {
            row = i/cols_number ;
            col = i%cols_number ;

            color = qRgb(static_cast<int>(red_channel_flatted_sorted(i)),
                         static_cast<int>(green_channel_flatted_sorted(i)),
                         static_cast<int>(blue_channel_flatted_sorted(i)));

            img.setPixel(col, row, color);

        }

        QLabel myLabel;
        myLabel.setPixmap(QPixmap::fromImage(img));
        myLabel.show();
        return a.exec();


    }
    catch(const exception& e)
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

