/*
Tensor<Index, 1> select_outliers_via_standard_deviation(const Tensor<type, 1>&, const type& = type(2.0), bool = true) const;

    Tensor<Index, 1> select_outliers_via_contamination(const Tensor<type, 1>&, const type& = type(0.05), bool = true) const;

    type calculate_euclidean_distance(const Tensor<Index, 1>&, const Index&, const Index&) const;

    Tensor<type, 2> calculate_distance_matrix(const Tensor<Index, 1>&) const;

    Tensor<list<Index>, 1> calculate_k_nearest_neighbors(const Tensor<type, 2>&, const Index& = 20) const;

    Tensor<Tensor<type, 1>, 1> get_kd_tree_data() const;

    Tensor<Tensor<Index, 1>, 1> create_bounding_limits_kd_tree(const Index&) const;

    void create_kd_tree(Tensor<Tensor<type, 1>, 1>&, const Tensor<Tensor<Index, 1>, 1>&) const;

    Tensor<list<Index>, 1> calculate_bounding_boxes_neighbors(const Tensor<Tensor<type, 1>, 1>&,
                                                              const Tensor<Index, 1>&,
                                                              const Index&, const Index&) const;

    Tensor<list<Index>, 1> calculate_kd_tree_neighbors(const Index& = 20, const Index& = 40) const;

    Tensor<type, 1> calculate_average_reachability(Tensor<list<Index>, 1>&, const Index&) const;

    Tensor<type, 1> calculate_local_outlier_factor(Tensor<list<Index>, 1>&, const Tensor<type, 1>&, const Index &) const;

    // Isolation Forest

    void calculate_min_max_indices_list(list<Index>&, const Index&, type&, type&) const;

    Index split_isolation_tree(Tensor<type, 2>&, list<list<Index>>&, list<Index>&) const;

    Tensor<type, 2> create_isolation_tree(const Tensor<Index, 1>&, const Index&) const;

    Tensor<Tensor<type, 2>, 1> create_isolation_forest(const Index&, const Index&, const Index&) const;

    type calculate_tree_path(const Tensor<type, 2>&, const Index&, const Index&) const;

    Tensor<type, 1> calculate_average_forest_paths(const Tensor<Tensor<type, 2>, 1>&, const Index&) const;

    // Local outlier factor

    Tensor<Index, 1> calculate_local_outlier_factor_outliers(const Index& = 20, const Index& = 0, const type& = type(0)) const;

    void unuse_local_outlier_factor_outliers(const Index& = 20, const type& = type(1.5));

    // Isolation Forest outlier

    Tensor<Index, 1> calculate_isolation_forest_outliers(const Index& = 100, const Index& = 256, const type& = type(0)) const;

    void unuse_isolation_forest_outliers(const Index& = 20, const type& = type(1.5));

    */