/*
Tensor<Index, 1> DataSet::select_outliers_via_contamination(const Tensor<type, 1>& outlier_ranks,
                                                            const type& contamination,
                                                            bool higher) const
{
    const Index samples_number = get_used_samples_number();

    Tensor<Tensor<type, 1>, 1> ordered_ranks(samples_number);

    Tensor<Index, 1> outlier_indexes(samples_number);
    outlier_indexes.setZero();

    for(Index i = 0; i < samples_number; i++)
    {
        ordered_ranks(i) = Tensor<type, 1>(2);
        ordered_ranks(i)(0) = type(i);
        ordered_ranks(i)(1) = outlier_ranks(i);
    }

    sort(ordered_ranks.data(), ordered_ranks.data() + samples_number,
         [](Tensor<type, 1> & a, Tensor<type, 1> & b)
    {
        return a(1) < b(1);
    });

    if(higher)
    {
        for(Index i = Index((type(1) - contamination)*type(samples_number)); i < samples_number; i++)
            outlier_indexes(static_cast<Index>(ordered_ranks(i)(0))) = 1;
    }
    else
    {
        for(Index i = 0; i < Index(contamination*type(samples_number)); i++)
            outlier_indexes(static_cast<Index>(ordered_ranks(i)(0))) = 1;
    }

    return outlier_indexes;
}


Tensor<Index, 1> DataSet::select_outliers_via_standard_deviation(const Tensor<type, 1>& outlier_ranks,
                                                                 const type& deviation_factor,
                                                                 bool higher) const
{
    const Index samples_number = get_used_samples_number();
    const type mean_ranks = mean(outlier_ranks);
    const type std_ranks = standard_deviation(outlier_ranks);

    Tensor<Index, 1> outlier_indexes(samples_number);
    outlier_indexes.setZero();


    if(higher)
    {
        for(Index i = 0; i < samples_number; i++)
        {
            if(outlier_ranks(i) > mean_ranks + deviation_factor*std_ranks)
                outlier_indexes(i) = 1;
        }
    }
    else
    {
        for(Index i = 0; i < samples_number; i++)
        {
            if(outlier_ranks(i) < mean_ranks - deviation_factor*std_ranks)
                outlier_indexes(i) = 1;
        }
    }

    return outlier_indexes;
}


type DataSet::calculate_euclidean_distance(const Tensor<Index, 1>& variables_indices,
                                           const Index& sample_index,
                                           const Index& other_sample_index) const
{
    const Index input_variables_number = variables_indices.size();

    type distance = type(0);
    type error;

    for(Index i = 0; i < input_variables_number; i++)
    {
        error = data(sample_index, variables_indices(i)) - data(other_sample_index, variables_indices(i));

        distance += error*error;
    }

    return sqrt(distance);
}


Tensor<type, 2> DataSet::calculate_distance_matrix(const Tensor<Index,1>& indices)const
{
    const Index samples_number = indices.size();

    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    Tensor<type, 2> distance_matrix(samples_number, samples_number);

    distance_matrix.setZero();

#pragma omp parallel for

    for(Index i = 0; i < samples_number ; i++)
    {
        for(Index k = 0; k < i; k++)
        {
            distance_matrix(i,k)
                    = distance_matrix(k,i)
                    = calculate_euclidean_distance(input_variables_indices, indices(i), indices(k));
        }
    }

    return distance_matrix;
}


Tensor<list<Index>, 1> DataSet::calculate_k_nearest_neighbors(const Tensor<type, 2>& distance_matrix, const Index& k_neighbors) const
{
    const Index samples_number = distance_matrix.dimensions()[0];

    Tensor<list<Index>, 1> neighbors_indices(samples_number);

#pragma omp parallel for

    for(Index i = 0; i < samples_number; i++)
    {
        list<type> min_distances(k_neighbors, numeric_limits<type>::max());

        neighbors_indices(i) = list<Index>(k_neighbors, 0);

        for(Index j = 0; j < samples_number; j++)
        {
            if(j == i) continue;

            list<Index>::iterator neighbor_it = neighbors_indices(i).begin();
            list<type>::iterator current_min = min_distances.begin();

            for(Index k = 0; k < k_neighbors; k++, current_min++, neighbor_it++)
            {
                if(distance_matrix(i,j) < *current_min)
                {
                    neighbors_indices(i).insert(neighbor_it, j);

                    min_distances.insert(current_min, distance_matrix(i,j));

                    break;
                }
            }
        }

        neighbors_indices(i).resize(k_neighbors);
    }

    return neighbors_indices;
}


Tensor<Tensor<type, 1>, 1> DataSet::get_kd_tree_data() const
{
    const Index used_samples_number = get_used_samples_number();
    const Index input_variables_number = get_input_numeric_variables_number();

    const Tensor<Index, 1> used_samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    Tensor<Tensor<type, 1>, 1> kd_tree_data(used_samples_number);

    for(Index i = 0; i < used_samples_number; i++)
    {
        kd_tree_data(i) = Tensor<type, 1>(input_variables_number+1);

        kd_tree_data(i)(0) = type(used_samples_indices(i)); // Storing index

        for(Index j = 0; j < input_variables_number; j++)
            kd_tree_data(i)(j+1) = data(used_samples_indices(i), input_variables_indices(j));
    }

    return kd_tree_data;
}


Tensor<Tensor<Index, 1>, 1> DataSet::create_bounding_limits_kd_tree(const Index& depth) const
{
    Tensor<Tensor<Index, 1>, 1> bounding_limits(depth+1);

    bounding_limits(0) = Tensor<Index, 1>(2);
    bounding_limits(0)(0) = 0;
    bounding_limits(0)(1) = get_used_samples_number();

    for(Index i = 1; i <= depth; i++)
    {
        bounding_limits(i) = Tensor<Index, 1>(pow(2, i)+1);
        bounding_limits(i)(0) = 0;

        for(Index j = 1; j < bounding_limits(i).size()-1; j = j+2)
        {
            bounding_limits(i)(j) = (bounding_limits(i-1)(j/2+1) - bounding_limits(i-1)(j/2))/2
                    + bounding_limits(i-1)(j/2);

            bounding_limits(i)(j+1) = bounding_limits(i-1)(j/2+1);
        }
    }

    return bounding_limits;
}


void DataSet::create_kd_tree(Tensor<Tensor<type, 1>, 1>& tree, const Tensor<Tensor<Index, 1>, 1>& bounding_limits) const
{
    const Index depth = bounding_limits.size()-1;
    const Index input_variables = tree(0).size();

    auto specific_sort = [&tree](const Index & first, const Index & last, const Index & split_variable)
    {
        sort(tree.data() + first, tree.data() + last,
             [&split_variable](const Tensor<type, 1> & a, const Tensor<type, 1> & b)
        {
            return a(split_variable) > b(split_variable);
        });
    };

    specific_sort(bounding_limits(0)(0), bounding_limits(0)(1), 1);

    Index split_variable = 2;

    for(Index i = 1; i <= depth; i++, split_variable++)
    {
        split_variable = max(split_variable % input_variables, static_cast<Index>(1));

        specific_sort(bounding_limits(i)(0), bounding_limits(i)(1), split_variable);

#pragma omp parallel for

        for(Index j = 1; j < (Index)bounding_limits(i).size()-1; j++)
            specific_sort(bounding_limits(i)(j)+1, bounding_limits(i)(j+1), split_variable);
    }
}


Tensor<list<Index>, 1> DataSet::calculate_bounding_boxes_neighbors(const Tensor<Tensor<type, 1>, 1>& tree,
                                                                   const Tensor<Index, 1>& leaves_indices,
                                                                   const Index& depth,
                                                                   const Index& k_neighbors) const
{
    const Index used_samples_number = get_used_samples_number();
    const Index leaves_number = pow(2, depth);

    Tensor<type, 1> bounding_box;

    Tensor<type, 2> distance_matrix;
    Tensor<list<Index>, 1> k_nearest_neighbors(used_samples_number);

    for(Index i = 0; i < leaves_number; i++) // Each bounding box
    {

        const Index first = leaves_indices(i);
        const Index last = leaves_indices(i+1);
        bounding_box = Tensor<type, 1>(last-first);

        for(Index j = 0; j < last - first; j++)
            bounding_box(j) = tree(first+j)(0);

        Tensor<type, 2> distance_matrix = calculate_distance_matrix(bounding_box);

        Tensor<list<Index>, 1> box_nearest_neighbors = calculate_k_nearest_neighbors(distance_matrix, k_neighbors);

        for(Index j = 0; j < last - first; j++)
        {
            for(auto & element : box_nearest_neighbors(j))
                element = bounding_box(element);

            k_nearest_neighbors(bounding_box(j)) = move(box_nearest_neighbors(j));
        }

    }

    return k_nearest_neighbors;
}


Tensor<list<Index>, 1> DataSet::calculate_kd_tree_neighbors(const Index& k_neighbors, const Index& min_samples_leaf) const
{
    const Index used_samples_number = get_used_samples_number();

    Tensor<Tensor<type, 1>, 1> tree = get_kd_tree_data();

    const Index depth = max(floor(log(static_cast<type>(used_samples_number)/static_cast<type>(min_samples_leaf))),
                       static_cast<type>(0.0));

    Tensor<Tensor<Index, 1>, 1> bounding_limits = create_bounding_limits_kd_tree(depth);

    create_kd_tree(tree, bounding_limits);

    return calculate_bounding_boxes_neighbors(tree, bounding_limits(depth), depth, k_neighbors);

    return Tensor<list<Index>, 1>();
}


Tensor<type, 1> DataSet::calculate_average_reachability(Tensor<list<Index>, 1>& k_nearest_indexes,
                                                        const Index& k) const
{
    const Index samples_number = get_used_samples_number();
    const Tensor<Index, 1> samples_indices = get_used_samples_indices();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    Tensor<type, 1> average_reachability(samples_number);
    average_reachability.setZero();

#pragma omp parallel for

    for(Index i = 0; i < samples_number; i++)
    {
        list<Index>::iterator neighbor_it = k_nearest_indexes(i).begin();

        type distance_between_points;
        type distance_2_k_neighbor;

        for(Index j = 0; j < k; j++, neighbor_it++)
        {
            const Index neighbor_k_index = k_nearest_indexes(*neighbor_it).back();

            distance_between_points = calculate_euclidean_distance(input_variables_indices, i, *neighbor_it);
            distance_2_k_neighbor = calculate_euclidean_distance(input_variables_indices, *neighbor_it, neighbor_k_index);

            average_reachability(i) += max(distance_between_points, distance_2_k_neighbor);
        }

        average_reachability(i) /= type(k);
    }

    return average_reachability;
}


Tensor<type, 1> DataSet::calculate_local_outlier_factor(Tensor<list<Index>, 1>& k_nearest_indexes,
                                                        const Tensor<type, 1>& average_reachabilities,
                                                        const Index & k) const
{
    const Index samples_number = get_used_samples_number();

    if(average_reachabilities.size() > samples_number)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "    Tensor<type, 1> calculate_local_outlier_factor(Tensor<list<Index>, 1>&, const Tensor<type, 1>&, const Index &) const method.\n"
               << "Average reachibilities size must be less than samples number.\n";

        throw runtime_error(buffer.str());
    }


    Tensor<type, 1> LOF_value(samples_number);

    long double sum;

#pragma omp parallel for

    for(Index i = 0; i < samples_number; i++)
    {
        sum = 0.0;

        for(const auto & neighbor_index : k_nearest_indexes(i))
            sum += average_reachabilities(i) / average_reachabilities(neighbor_index);

        LOF_value(i) = type(sum/k) ;
    }
    return LOF_value;
}


type DataSet::calculate_tree_path(const Tensor<type, 2>& tree, const Index& sample_index,
                                  const Index& tree_depth) const
{
    Index current_index = 0;
    Index current_depth = 0;
    const Index tree_length = tree.dimensions()[0];

    type samples;
    type value;

    while(current_depth < tree_depth)
    {
        if(tree(current_index, 2) == type(1))
        {
            return type(current_depth);
        }
        else if(current_index*2 >= tree_length ||
                (tree(current_index*2+1, 2) == numeric_limits<type>::infinity())
                ) //Next node doesn't exist or node is leaf
        {
            samples = tree(current_index, 2);

            return type(log(samples- type(1)) - (type(2) *(samples- type(1)))/samples + type(0.5772) + type(current_depth));
        }


        value = data(sample_index, static_cast<Index>(tree(current_index, 1)));

        (value < tree(current_index, 0)) ? current_index = current_index*2 + 1
                : current_index = current_index*2 + 2;

        current_depth++;
    }

    samples = tree(current_index, 2);
    if(samples == type(1))
        return type(current_depth);
    else
        return type(log(samples- type(1))-(type(2.0) *(samples- type(1)))/samples + type(0.5772) + type(current_depth));
}


Tensor<type, 1> DataSet::calculate_average_forest_paths(const Tensor<Tensor<type, 2>, 1>& forest, const Index& tree_depth) const
{
    const Index samples_number = get_used_samples_number();
    const Index n_trees = forest.dimensions()[0];
    Tensor<type, 1> average_paths(samples_number);
    average_paths.setZero();

# pragma omp parallel for
    for(Index i = 0; i < samples_number; i++)
    {
        for(Index j = 0; j < n_trees; j++)
            average_paths(i) += calculate_tree_path(forest(j), i, tree_depth);

        average_paths(i) /= type(n_trees);
    }
    return average_paths;
}



Tensor<Index, 1> DataSet::calculate_isolation_forest_outliers(const Index& n_trees,
                                                              const Index& subs_set_samples,
                                                              const type& contamination) const
{
    const Index samples_number = get_used_samples_number();
    const Index fixed_subs_set_samples = min(samples_number, subs_set_samples);
    const Index max_depth = Index(ceil(log2(fixed_subs_set_samples))*2);
    const Tensor<Tensor<type, 2>, 1> forest = create_isolation_forest(n_trees, fixed_subs_set_samples, max_depth);

    const Tensor<type, 1> average_paths = calculate_average_forest_paths(forest, max_depth);

    Tensor<Index, 1> outlier_indexes;

    contamination > type(0)
            ? outlier_indexes = select_outliers_via_contamination(average_paths, contamination, false)
            : outlier_indexes = select_outliers_via_standard_deviation(average_paths, type(2.0), false);

    return outlier_indexes;
}



/// Calculate the outliers from the data set using the LocalOutlierFactor method.
/// @param k_neighbors Used to perform a k_nearest_algorithm to find the local density. Default is 20.
/// @param min_samples_leaf The minimum number of samples per leaf when building a KDTree.
/// If 0, automatically decide between using brute force aproach or KDTree.
/// If > samples_number/2, brute force aproach is performed. Default is 0.
/// @param contamination Percentage of outliers in the data_set to be selected. If 0.0, those paterns which deviates from the mean of LOF
/// more than 2 times are considered outlier. Default is 0.0.

Tensor<Index, 1> DataSet::calculate_local_outlier_factor_outliers(const Index& k_neighbors,
                                                                  const Index& min_samples_leaf,
                                                                  const type& contamination) const
{
    if(k_neighbors < 0)
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<Index, 1> DataSet::calculate_local_outlier_factor_outliers(const Index&, const Index&, const type&) const method.\n"
               << "k_neighbors(" << k_neighbors << ") should be a positive integer value\n";

        throw runtime_error(buffer.str());
    }

    if(contamination < type(0) && contamination > type(0.5))
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: DataSet class.\n"
               << "Tensor<Index, 1> DataSet::calculate_local_outlier_factor_outliers(const Index&, const Index&, const type&) const method.\n"
               << "Outlier contamination(" << contamination << ") should be a value between 0.0 and 0.5\n";

        throw runtime_error(buffer.str());
    }

    const Index samples_number = get_used_samples_number();

    bool kdtree = false;

    Index k = min(k_neighbors, samples_number-1);
    Index min_samples_leaf_fix = max(min_samples_leaf, static_cast<Index>(0));

    if(min_samples_leaf == 0 && samples_number > 5000)
    {
        min_samples_leaf_fix = 200;
        k = min(k, min_samples_leaf_fix-1);
        kdtree = true;
    }
    else if(min_samples_leaf!=0 && min_samples_leaf < samples_number/2)
    {
        k = min(k, min_samples_leaf_fix-1);
        kdtree = true;
    }

    Tensor<list<Index>, 1> k_nearest_indexes;

    kdtree ? k_nearest_indexes = calculate_kd_tree_neighbors(k, min_samples_leaf_fix)
            : k_nearest_indexes = calculate_k_nearest_neighbors(calculate_distance_matrix(get_used_samples_indices()), k);


    const Tensor<type, 1> average_reachabilities = calculate_average_reachability(k_nearest_indexes, k);


    const Tensor<type, 1> LOF_value = calculate_local_outlier_factor(k_nearest_indexes, average_reachabilities, k);


    Tensor<Index, 1> outlier_indexes;

    contamination > type(0)
            ? outlier_indexes = select_outliers_via_contamination(LOF_value, contamination, true)
            : outlier_indexes = select_outliers_via_standard_deviation(LOF_value, type(2.0), true);

    return outlier_indexes;
}


void DataSet::calculate_min_max_indices_list(list<Index>& elements, const Index& variable_index, type& min, type& max) const
{
    type value;
    min = max = data(elements.front(), variable_index);
    for(const auto & sample_index : elements)
    {
        value = data(sample_index, variable_index);
        if(min > value) min = value;
        else if(max < value) max = value;
    }
}


Index DataSet::split_isolation_tree(Tensor<type, 2> & tree, list<list<Index>>& tree_simulation, list<Index>& tree_index) const
{

    const Index current_tree_index = tree_index.front();
    const type current_variable = tree(current_tree_index, 1);
    const type division_value = tree(current_tree_index, 0);

    list<Index> current_node_samples  = tree_simulation.front();

    list<Index> one_side_samples;
    list<Index> other_side_samples;

    Index delta_next_depth_nodes = 0;
    Index one_side_count = 0;
    Index other_side_count = 0;

    for(auto & sample_index : current_node_samples)
    {
        if(data(sample_index, current_variable) < division_value)
        {
            one_side_count++;
            one_side_samples.emplace_back(sample_index);
        }
        else
        {
            other_side_count++;
            other_side_samples.emplace_back(sample_index);
        }
    }

    if(one_side_count != 0)
    {
        if(one_side_count != 1)
        {
            tree_simulation.emplace_back(one_side_samples);
            tree_index.emplace_back(current_tree_index*2+1);
            delta_next_depth_nodes++;
        }

        if(other_side_count != 1)
        {
            tree_simulation.emplace_back(other_side_samples);
            tree_index.emplace_back(current_tree_index*2+2);
            delta_next_depth_nodes++;
        }

        tree(current_tree_index*2+1, 2) = type(one_side_count);
        tree(current_tree_index*2+2, 2) = type(other_side_count);
    }


    return delta_next_depth_nodes;

//    return Index();
}


Tensor<type, 2> DataSet::create_isolation_tree(const Tensor<Index, 1>& indices, const Index& max_depth) const
{
    const Index used_samples_number = indices.size();

    const Index variables_number = get_input_numeric_variables_number();
    const Tensor<Index, 1> input_variables_indices = get_input_variables_indices();

    list<list<Index>> tree_simulation;
    list<Index> tree_index;
    list<Index> current_node_samples;

    Tensor<type, 2> tree(pow(2, max_depth+1) - 1, 3);
    tree.setConstant(numeric_limits<type>::infinity());

    for(Index i = 0; i < used_samples_number; i++)
        current_node_samples.emplace_back(indices(i));

    tree_simulation.emplace_back(current_node_samples);
    tree(0, 2) = type(used_samples_number);
    tree_index.emplace_back(0);

    current_node_samples.clear();

    Index current_depth_nodes = 1;
    Index next_depth_nodes = 0;
    Index current_variable_index = input_variables_indices(rand() % variables_number);
    Index current_depth = 0;

    Index current_index;

    type min;
    type max;

    while(current_depth < max_depth && !(tree_simulation.empty()))
    {
        current_node_samples = tree_simulation.front();
        current_index = tree_index.front();

        calculate_min_max_indices_list(current_node_samples, current_variable_index, min, max);

        tree(current_index, 0) = static_cast<type>((max-min)*(type(rand())/static_cast<type>(RAND_MAX))) + min;

        tree(current_index, 1) = type(current_variable_index);

        next_depth_nodes += split_isolation_tree(tree, tree_simulation, tree_index);

        tree_simulation.pop_front();
        tree_index.pop_front();

        current_depth_nodes--;

        if(current_depth_nodes == 0)
        {
            current_depth++;
            swap(current_depth_nodes, next_depth_nodes);
            current_variable_index = input_variables_indices(rand() % variables_number);
        }
    }

    return tree;
}


Tensor<Tensor<type, 2>, 1> DataSet::create_isolation_forest(const Index& trees_number, const Index& sub_set_size, const Index& max_depth) const
{
    const Tensor<Index, 1> indices = get_used_samples_indices();
    const Index samples_number = get_used_samples_number();
    Tensor<Tensor<type, 2>, 1> forest(trees_number);

    random_device rng;
    mt19937 urng(rng());

    for(Index i = 0; i < trees_number; i++)
    {
        Tensor<Index, 1> sub_set_indices(sub_set_size);
        Tensor<Index, 1> aux_indices = indices;
        std::shuffle(aux_indices.data(), aux_indices.data()+samples_number, urng);

        for(Index j = 0; j < sub_set_size; j++)
            sub_set_indices(j) = aux_indices(j);

        forest(i) = create_isolation_tree(sub_set_indices, max_depth);

    }
    return forest;
}
*/
