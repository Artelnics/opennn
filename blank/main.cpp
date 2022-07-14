//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

#include <any>

// OpenNN includes

#include "../opennn/opennn.h"

using namespace opennn;
using namespace std;
using namespace Eigen;

//template <typename T>
//class tensor_holder
//{
//private:
//    std::any _held;
//    Index _rank;
//public:
//    template <int N>
//    constexpr tensor_holder(Eigen::Tensor<T, N> tensor) :
//        _held{std::move(tensor)},
//        _rank{N}
//    {
//    }
//    constexpr tensor_holder(const tensor_holder&) = default;
//    constexpr tensor_holder(tensor_holder&&) = default;
//    template <Index N>
//    const Eigen::Tensor<T, N> get_values()
//    {
//        return std::any_cast<Eigen::Tensor<T, N>>(_held);
//        return static_cast<Eigen::Tensor<T, N>&>(_held);
//    }

//    template <Index N>
//    const Eigen::Tensor<T, N> get_values() const
//    {
//        return std::any_cast<Eigen::Tensor<T, N>>(_held);
//    }

//    template <Index N>
//    const Eigen::Tensor<T, N>& get()
//    {
//        return std::any_cast<Eigen::Tensor<T, N>&>(_held);
//        return static_cast<Eigen::Tensor<T, N>&>(_held);
//    }

//    template <Index N>
//    Eigen::Tensor<T, N>* get_pointer()
//    {
//        return std::any_cast<Eigen::Tensor<T, N>*>(_held);
//        return static_cast<Eigen::Tensor<T, N>&>(_held);
//    }

//    template <Index N>
//    const Eigen::Tensor<T, N>* get_pointer() const
//    {
//        return std::any_cast<Eigen::Tensor<T, N>*>(_held);
//        return static_cast<Eigen::Tensor<T, N>&>(_held);
//    }


//    template <Index N>
//    const Eigen::Tensor<T, N>& get() const
//    {
//        return std::any_cast<Eigen::Tensor<T, N>&>(_held);
//    }
//    constexpr int rank() const noexcept
//    {
//        return _rank;
//    }
//};


//void use_tensor(const tensor_holder<Index>& in)
//{
//    cout << "rank: " << in.rank() << endl;
//    if (in.rank() == 1)
//    {
//        //        const Tensor<Index, 1>& tensor = in.get<1>();
//        const Tensor<Index, 1>* tensor = in.get_pointer<1>();
//        cout << "copied data:" << tensor->data() << endl;
//        tensor = in.get<1>();
//        cout << "1: " << copy << endl;
//    }
//    else
//    {
//        auto& tensor = in.get<3>();
//        cout << tensor << endl;
//        // some other logic
//    }
//}



int main()
{
    try
    {
        cout << "Hello OpenNN!" << endl;

        Tensor<Index, 2> tensor(2,3);
        tensor.setRandom();

        cout << "first: " << tensor << endl;

        for(Index i = 0; i < 15; i++)
        {
            cout << *tensor.data() << "\t";

            if( i%5 == 0)
            {
                cout << "\n";
            }
        }

        cout << "Goodbye!" << endl;

        return 0;
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
