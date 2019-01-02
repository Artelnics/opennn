/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.opennn.net                                                                                             */
/*                                                                                                              */
/*   A S S O C I A T I O N   R U L E S   C L A S S                                                              */
/*                                                                                                              */
/*   Fernando Gomez                                                                                             */
/*   Artificial Intelligence Techniques SL                                                                      */
/*   fernandogomez@artelnics.com                                                                                */
/*                                                                                                              */
/****************************************************************************************************************/

// OpenNN includes

#include "association_rules.h"

namespace OpenNN
{

// DEFAULT CONSTRUCTOR

/// Default constructor.

AssociationRules::AssociationRules()
{
    display = true;

    sparse_matrix.set();
}


// DESTRUCTOR

/// Destructor.

AssociationRules::~AssociationRules()
{
}

// GET METHODS

SparseMatrix<int> AssociationRules::get_sparse_matrix() const
{
    return sparse_matrix;
}

double AssociationRules::get_minimum_support() const
{
    return minimum_support;
}

// Display messages

const bool& AssociationRules::get_display() const
{
    return display;
}

// SET METHODS

void AssociationRules::set_sparse_matrix(const SparseMatrix<int>& new_sparse_matrix)
{
    sparse_matrix = new_sparse_matrix;
}

void AssociationRules::set_minimum_support(const double& new_minimum_support)
{
    minimum_support = new_minimum_support;
}

// Display messages

void AssociationRules::set_display(const bool& new_display)
{
    display = new_display;
}

unsigned long long AssociationRules::calculate_combinations_number(const size_t& n, const size_t& r) const
{

    if(n == 0 || r > n)
    {
        return 0;
    }

    if(r == 1)
    {
        return n;
    }


    if(r == n || r == 0)
    {
        return 1;
    }

    unsigned long long result = 1;

    unsigned long long k = r;

    if( k > n - k )
    {
        k = n - k;
    }

    for(int i = 0; i < static_cast<int>(k); ++i)
    {
        result *= (n - static_cast<unsigned long long>(i));
        result /= (static_cast<unsigned long long>(i) + 1);
    }

    return result;
}



Matrix<size_t> AssociationRules::calculate_combinations(const Vector<size_t>& items_index, const size_t& r) const
{
    Matrix<size_t> combinations;

    if(r == 0)
    {
        return(combinations);
    }

    const size_t n = items_index.size();

    const unsigned long long  combinations_number = calculate_combinations_number(n, r);

    if(combinations_number == 0)
    {
        return(combinations);
    }

    combinations.set(combinations_number, r);

    Vector<bool> v(n);

    fill(v.end() - static_cast<unsigned>(r), v.end(), true);

    size_t row_index = 0;
    size_t column_index = 0;

    do {

        for(size_t i = 0; i < n; ++i)
        {
            if(v[i])
            {
                combinations(row_index, column_index) = items_index[i];
                column_index++;
            }
        }

        row_index++;
        column_index = 0;

    } while(next_permutation(v.begin(), v.end()));

    return(combinations);
}

Matrix<double> AssociationRules::calculate_support(const size_t& order_size, const Vector<size_t>& items_index) const
{
    Vector<size_t> this_items_index = items_index;

    if(this_items_index.empty())
    {
        this_items_index = Vector<size_t>(0,1,sparse_matrix.get_columns_number()-1);
    }

    Matrix<double> support_data;

    if(order_size == 0)
    {
        return(support_data);
    }

    const size_t items_number = this_items_index.size();

    const unsigned long long combinations_number = calculate_combinations_number(items_number, order_size);

    if(combinations_number == 0)
    {
        return(support_data);
    }

    const size_t orders_number = sparse_matrix.get_rows_number();

    const Matrix<size_t> combinations = calculate_combinations(this_items_index, order_size);

    Vector<size_t> frequencies(combinations_number);

    Vector<size_t> current_combination;

    Vector<size_t> previous_combination;

    Matrix<int> sub_matrix(orders_number,order_size);

    previous_combination = combinations.get_row(0);

    for(size_t i = 0; i < order_size; i++)
    {
        const Vector<int> current_column = sparse_matrix.get_column(previous_combination[i]);

        sub_matrix.set_column(i, current_column, "");
    }

    frequencies[0] = sub_matrix.count_rows_equal_to(1);

#pragma omp parallel for firstprivate(sub_matrix) private(current_combination,previous_combination)

    for(int i = 1; i < static_cast<int>(combinations_number); i++)
    {
        current_combination = combinations.get_row(static_cast<size_t>(i));

        const Vector<size_t> intersection = current_combination.get_intersection(previous_combination);

        for(size_t j = 0; j < order_size; j++)
        {
            if(!intersection.contains(current_combination[j]))
            {
                const Vector<int> current_column = sparse_matrix.get_column(current_combination[j]);

                sub_matrix.set_column(j, current_column, "");
            }
        }

        frequencies[static_cast<size_t>(i)] = sub_matrix.count_rows_equal_to(1);

        previous_combination = current_combination;
    }

    const Vector<double> supports = frequencies.to_double_vector()*(100.0/static_cast<double>(orders_number));
    const size_t columns_number = order_size + 2;

    support_data.set(combinations_number, columns_number);

    for(size_t i = 0;  i < order_size; i++)
    {
        support_data.set_column(i, combinations.get_column(i).to_double_vector(), "");
    }

    support_data.set_column(columns_number - 2, frequencies.to_double_vector(), "");
    support_data.set_column(columns_number - 1, supports, "");

    support_data = support_data.sort_descending(columns_number-2);

    return(support_data);
}

Matrix<double> AssociationRules::calculate_confidence(const size_t& left_order_size, const size_t& right_order_size, const Vector<size_t>& items_index) const
{
    Vector<size_t> this_items_index = items_index;

    if(this_items_index.empty())
    {
        this_items_index = Vector<size_t>(0,1,sparse_matrix.get_columns_number()-1);
    }

    Matrix<double> confidence_data;

    if(right_order_size == 0 || left_order_size == 0)
    {
        return confidence_data;
    }

    Matrix<double> total_support = calculate_support(left_order_size+right_order_size, this_items_index);

    if(total_support.get_rows_number() == 0)
    {
        return confidence_data;
    }

    Matrix<double> left_support = calculate_support(left_order_size, this_items_index);

    Matrix<size_t> right_combinations = calculate_combinations(this_items_index, right_order_size);

    Vector< Vector<double> > rows_combinations;

#pragma omp parallel for

    for(int i = 0; i < static_cast<int>(left_support.get_rows_number()); i++)
    {
        const Vector<double> current_left_indices = left_support.get_row(static_cast<size_t>(i)).get_first(left_order_size);

        for(size_t j = 0; j < static_cast<size_t>(right_combinations.get_rows_number()); j++)
        {
            const Vector<size_t> current_right_indices = right_combinations.get_row(j);

            if(!current_right_indices.contains(current_left_indices.to_size_t_vector()))
            {
                const Vector<double> current_row = current_left_indices.assemble(current_right_indices.to_double_vector());
                #pragma omp critical
                {
                    rows_combinations.push_back(current_row);
                }
            }
        }
    }

    const size_t rows_number = rows_combinations.size();
    const size_t columns_number = left_order_size + right_order_size + 1;

    confidence_data.set(rows_number, columns_number);

    const size_t total_support_rows = total_support.get_rows_number();
    Vector<size_t> combination_indices(0,1,left_order_size+right_order_size-1);

#pragma omp parallel for

    for(int i = 0; i < rows_number; i++)
    {
        double current_total_support = 0.0;

        for(size_t j = 0; j < total_support_rows; j++)
        {
            const Vector<double> current_combination = total_support.get_row(j,combination_indices);

            if(current_combination.has_same_elements(rows_combinations[i]))
            {
                current_total_support = total_support(j,left_order_size+right_order_size+1);

                break;
            }
        }

        confidence_data.set_row(i,rows_combinations[i].assemble(Vector<double>(1,current_total_support)));
    }

    Matrix<double> left_indices = confidence_data.get_submatrix_columns(Vector<size_t>(0,1,left_order_size-1));

#pragma omp parallel for

    for(int i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < left_support.get_rows_number(); j++)
        {
            size_t found = 0;

            for(size_t k = 0; k < left_order_size; k++)
            {
                if(fabs(left_indices(i,k) - left_support(j,k)) < std::numeric_limits<double>::min())
                {
                    found++;
                }
            }

            if(found == left_order_size)
            {
                if(left_support(j,left_order_size+1) != 0.0)
                {
                    confidence_data(i,left_order_size+right_order_size) /= left_support(j,left_order_size+1);
                }
                else
                {
                    confidence_data(i,left_order_size+right_order_size) = 0;
                }

                break;
            }
        }
    }

    confidence_data = confidence_data.sort_descending(left_order_size+right_order_size);

    return confidence_data;
}

Matrix<double> AssociationRules::calculate_lift(const size_t& left_order_size, const size_t& right_order_size, const Vector<size_t>& items_index) const
{
    Vector<size_t> this_items_index = items_index;

    if(this_items_index.empty())
    {
        this_items_index = Vector<size_t>(0,1,sparse_matrix.get_columns_number()-1);
    }

    Matrix<double> lift_data;

    if(right_order_size == 0 || left_order_size == 0)
    {
        return lift_data;
    }

    Matrix<double> total_support = calculate_support(left_order_size+right_order_size, this_items_index);

    if(total_support.get_rows_number() == 0)
    {
        return lift_data;
    }

    Matrix<double> left_support = calculate_support(left_order_size, this_items_index);

    Matrix<double> right_support;

    if(right_order_size == left_order_size)
    {
        right_support = left_support;
    }
    else
    {
        right_support = calculate_support(right_order_size, this_items_index);
    }

    Vector< Vector<double> > rows_combinations;
    Vector<size_t> right_combination_indices(0,1,right_order_size-1);
    Vector<size_t> left_combination_indices(0,1,left_order_size-1);

#pragma omp parallel for

    for(int i = 0; i < left_support.get_rows_number(); i++)
    {
        const Vector<double> current_left_indices = left_support.get_row(i,left_combination_indices);

        for(size_t j = 0; j < right_support.get_rows_number(); j++)
        {
            const Vector<double> current_right_indices = right_support.get_row(j,right_combination_indices);

            if(!current_right_indices.contains(current_left_indices))
            {
                const Vector<double> current_row = current_left_indices.assemble(current_right_indices);
                #pragma omp critical
                {
                    rows_combinations.push_back(current_row);
                }
            }
        }
    }

    const size_t rows_number = rows_combinations.size();
    const size_t columns_number = left_order_size + right_order_size + 1;

    lift_data.set(rows_number, columns_number);
    Vector<size_t> combination_indices(0,1,left_order_size+right_order_size-1);

#pragma omp parallel for

    for(int i = 0; i < rows_number; i++)
    {
        double current_total_support = 0.0;

        for(size_t j = 0; j < total_support.get_rows_number(); j++)
        {
            const Vector<double> current_combination = total_support.get_row(j,combination_indices);

            if(current_combination.has_same_elements(rows_combinations[i]))
            {
                current_total_support = total_support(j,left_order_size+right_order_size+1);

                break;
            }
        }

        lift_data.set_row(i,rows_combinations[i].assemble(Vector<double>(1,current_total_support)));
    }

    Matrix<double> left_indices = lift_data.get_submatrix_columns(Vector<size_t>(0,1,left_order_size-1));
    Matrix<double> right_indices = lift_data.get_submatrix_columns(Vector<size_t>(left_order_size,1,left_order_size+right_order_size-1));

#pragma omp parallel for

    for(int i = 0; i < rows_number; i++)
    {
        for(size_t j = 0; j < left_support.get_rows_number(); j++)
        {
            size_t found = 0;

            for(size_t k = 0; k < left_order_size; k++)
            {
                if(fabs(left_indices(i,k) - left_support(j,k)) < std::numeric_limits<double>::min())
                {
                    found++;
                }
            }

            if(found == left_order_size)
            {
                if(left_support(j,left_order_size+1) != 0.0)
                {
                    lift_data(i,left_order_size+right_order_size) /= left_support(j,left_order_size+1);
                }
                else
                {
                    lift_data(i,left_order_size+right_order_size) = 0;
                }

                break;
            }
        }

        for(size_t j = 0; j < right_support.get_rows_number(); j++)
        {
            size_t found = 0;

            for(size_t k = 0; k < right_order_size; k++)
            {
                if(fabs(right_indices(i,k) - right_support(j,k)) < std::numeric_limits<double>::min())
                {
                    found++;
                }
            }

            if(found == right_order_size)
            {
                if(right_support(j,right_order_size+1) != 0.0)
                {
                    lift_data(i,left_order_size+right_order_size) /= right_support(j,right_order_size+1);
                }
                else
                {
                    lift_data(i,left_order_size+right_order_size) = 0;
                }

                break;
            }
        }
    }

    lift_data = lift_data.sort_descending(left_order_size+right_order_size);

    return lift_data;
}

Vector< Matrix<double> > AssociationRules::perform_a_priori_algorithm(const size_t& maximum_order)
{
    size_t this_maximum_order = maximum_order;

    if(this_maximum_order == 0)
    {
        this_maximum_order = sparse_matrix.get_columns_number();
    }

    SparseMatrix<int> copy_sparse_matrix(sparse_matrix);

    Vector< Matrix<double> > most_frequent;

    time_t beginning_time, current_time;
    double elapsed_time;

    time(&beginning_time);

    Matrix<double> this_order_support = calculate_support(1);

    Matrix<double> this_unfrequent_items = this_order_support.filter_column_less_than(2,minimum_support);

    if(!this_unfrequent_items.empty())
    {
        this_unfrequent_items = this_unfrequent_items.get_submatrix_columns(Vector<size_t>(1,0));

        const size_t unfrequent_items_size = this_unfrequent_items.get_rows_number();

        Vector<size_t> rows_to_remove;

#pragma omp parallel for
        for(int i = 0; i < unfrequent_items_size; i++)
        {
            const Vector<size_t> this_rows_to_remove = sparse_matrix.get_row_indices_equal_to(this_unfrequent_items.get_row(i).to_size_t_vector(), 1);

#pragma omp critical
            {
                rows_to_remove = rows_to_remove.assemble(this_rows_to_remove);
            }
        }

        rows_to_remove = rows_to_remove.get_unique_elements();

        sparse_matrix = sparse_matrix.delete_rows(rows_to_remove);
    }

    this_order_support = this_order_support.filter_column_greater_than(2,minimum_support);

    Vector<size_t> first_valid_indices = this_order_support.get_column(0).to_size_t_vector();

    most_frequent.push_back(this_order_support);

    time(&current_time);
    elapsed_time = difftime(current_time, beginning_time);

    if(elapsed_time > maximum_time)
    {
        if(display)
        {
            cout << "Maximum time reached\n";
            cout << "Current order: 1\n";
            cout << "Current frequent combinations: " << this_order_support.get_rows_number() << endl;
            cout << "Elapsed time: " << elapsed_time << endl;
        }

        set_sparse_matrix(copy_sparse_matrix);

        return most_frequent;
    }

    if(display)
    {
        cout << "Current order: 1\n";
        cout << "Current frequent combinations: " << this_order_support.get_rows_number() << endl;
        cout << "Elapsed time: " << elapsed_time << endl;
    }


    for(size_t current_order = 2; current_order <= this_maximum_order; current_order++)
    {
        this_order_support = calculate_support(current_order, first_valid_indices);

        this_unfrequent_items = this_order_support.filter_column_less_than(current_order+1,minimum_support);

        if(!this_unfrequent_items.empty())
        {
            this_unfrequent_items = this_unfrequent_items.get_submatrix_columns(Vector<size_t>(0,1,current_order-1));

            const size_t unfrequent_items_size = this_unfrequent_items.get_rows_number();

            Vector<size_t> rows_to_remove;

#pragma omp parallel for
            for(int i = 0; i < unfrequent_items_size; i++)
            {
                Vector<size_t> this_rows_to_remove = sparse_matrix.get_row_indices_equal_to(this_unfrequent_items.get_row(i).to_size_t_vector(), 1);

#pragma omp critical
                {
                    rows_to_remove = rows_to_remove.assemble(this_rows_to_remove);
                }
            }

            rows_to_remove = rows_to_remove.get_unique_elements();

            sparse_matrix = sparse_matrix.delete_rows(rows_to_remove);
        }

        this_order_support = this_order_support.filter_column_greater_than(current_order+1,minimum_support);

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

        if(this_order_support.empty())
        {
            if(display)
            {
                cout << "No more frequent combinations." << endl;
                cout << "Algorithm finished." << endl;
                cout << "Elapsed time: " << elapsed_time << endl;
            }

            break;
        }

        most_frequent.push_back(this_order_support);

        if(elapsed_time > maximum_time)
        {
            if(display)
            {
                cout << "Maximum time reached\n";
                cout << "Current order: " << current_order << endl;
                cout << "Current frequent combinations: " << this_order_support.get_rows_number() << endl;
                cout << "Elapsed time: " << elapsed_time << endl;
            }

            set_sparse_matrix(copy_sparse_matrix);

            return most_frequent;
        }

        if(display)
        {
            cout << "Current order: " << current_order << endl;
            cout << "Current frequent combinations: " << this_order_support.get_rows_number() << endl;
            cout << "Elapsed time: " << elapsed_time << endl;
        }

    }

    set_sparse_matrix(copy_sparse_matrix);

    return most_frequent;
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
