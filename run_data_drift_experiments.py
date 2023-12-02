import sys

import numpy as np
import plotly.express as px

from helpers import generator_helpers as gen_help

sys.path.append('../cluster_library')
sys.path.append('../')

import cluster_validation.davies_bouldin as db
import federated_clustering.local_learners as fcll
import federated_clustering.global_learner as fcgl


def generate_new_data(run_mode, n_points_per_generator, dict_init_generators_per_client, clients_global_drift=1,
                      mean_min_val=-5, mean_max_val=5, data_dim=2):
    """

    :param run_mode:
    :param dict_init_generators_per_client:
    :return:
    """

    if not run_mode in ['no_local_drift', 'local_but_no_global', 'global_drift']:
        raise NotImplementedError

    dict_new_data_per_client = {}

    if run_mode == 'no_local_drift':

        for client, client_generators in dict_init_generators_per_client.items():
            X_new_data_client = np.concatenate(
                [generator.generate_data_points(n_points_per_generator, seed=None) for generator in client_generators])
            dict_new_data_per_client[client] = X_new_data_client

    elif run_mode == 'local_but_no_global':

        n_clients = len(dict_init_generators_per_client.keys())
        # for each client randomly select another client's data generators
        for client, _ in dict_init_generators_per_client.items():

            other_client = client
            while other_client == client:
                other_client = np.random.choice(range(n_clients))
            # print(f'Client {client} chooses client {other_client}.')
            client_generators = dict_init_generators_per_client.get(other_client)

            dict_new_data_per_client[client] = X_new_data_client = np.concatenate(
                [generator.generate_data_points(n_points_per_generator, seed=None) for generator in client_generators])
            dict_new_data_per_client[client] = X_new_data_client

    elif run_mode == 'global_drift':
        n_clients = len(dict_init_generators_per_client.keys())
        drifted_clients = np.random.choice(range(n_clients), size=clients_global_drift, replace=False)

        for client, client_generators in dict_init_generators_per_client.items():

            if client in drifted_clients:
                new_data_generator = gen_help.make_gaussian_data_generator_random_mean_random_cov(dim=data_dim,
                                                                                                  mean_min_val=mean_min_val,
                                                                                                  mean_max_val=mean_max_val)

                X_new_data_client = new_data_generator.generate_data_points(n_points_per_generator, seed=None)
            else:
                X_new_data_client = np.concatenate(
                    [generator.generate_data_points(n_points_per_generator, seed=None) for generator in
                     client_generators])

            dict_new_data_per_client[client] = X_new_data_client

    return dict_new_data_per_client


if __name__ == '__main__':

    run_mode = 'global_drift'  # 'no_local_drift'  # 'local_but_no_global_drift' # global_drift
    n_repeats = 10
    n_clients = 2
    dim_data = 2
    n_generators_per_client = 2
    initial_points_per_generator = 1000
    next_timestep_new_points_per_generator = 1000
    mean_min_val = -5
    mean_max_val = 5
    cov_mat_initial = np.eye(dim_data) * 0.5  # we make sure the initial data is clusterable

    # fuzzy c-means parameters
    m = 2
    tol_local = 0.001
    max_iter_local = 50
    max_iter_global = 10
    tol_global = 0.01
    num_clusters = n_generators_per_client * n_clients  # we assume the number of (global) clusters to be give at this point

    # create initial datasets at t0
    dict_initial_data_generators_per_client = {}
    list_local_learners = []

    for client in range(n_clients):
        # each client has specified number of initial distributions
        client_generators = [gen_help.make_gaussian_data_generator_random_mean(
            min_val=mean_min_val,
            max_val=mean_max_val,
            cov_mat=cov_mat_initial
        ) for _ in
            range(n_generators_per_client)]
        dict_initial_data_generators_per_client[client] = client_generators

        X_initial_client = np.concatenate(
            [generator.generate_data_points(initial_points_per_generator, seed=43) for generator in
             client_generators])

        local_learner = fcll.FuzzyCMeansClient(client_data=X_initial_client,
                                               num_clusters=num_clusters,
                                               max_iter=max_iter_local,
                                               tol=tol_local, m=m
                                               )
        list_local_learners.append(local_learner)

        fig = px.scatter(x=X_initial_client[:, 0],
                         y=X_initial_client[:, 1],
                         range_x=[-10, 10],
                         range_y=[-10, 10],
                         title=f'Initial data of client {client}')
        fig.show()

    # learn federated fuzzy c-means model
    global_learner = fcgl.GlobalClusterer(local_learners=list_local_learners,
                                          num_clusters=num_clusters,
                                          data_dim=dim_data,
                                          max_rounds=max_iter_global,
                                          tol=tol_global,
                                          weighing_function=None,
                                          global_center_update='kmeans')

    global_learner.fit()
    global_cluster_centers = global_learner.cluster_centers

    print(global_cluster_centers)

    # calculate global fuzzy-DB and estimate acceptable range
    initial_global_db = db.calculate_federated_fuzzy_db(_local_learners=list_local_learners,
                                                        _num_clusters=num_clusters
                                                        )
    print(f'Initial federated fuzzy Davies-Bouldin: {initial_global_db}.')

    # depending on run_mode, get new data from generators for each client

    list_all_recalculated_davies_bouldin = []
    for repeat in range(n_repeats):
        dict_new_data_per_client = generate_new_data(run_mode=run_mode,
                                                     n_points_per_generator=next_timestep_new_points_per_generator,
                                                     dict_init_generators_per_client=dict_initial_data_generators_per_client)
        # ... and set new data
        for client, local_learner in enumerate(list_local_learners):
            local_learner.set_new_data(dict_new_data_per_client.get(client))

        # with new data, recalculate global Davies-Bouldin
        recalculated_global_db = db.calculate_federated_fuzzy_db(list_local_learners,
                                                                 num_clusters)
        list_all_recalculated_davies_bouldin.append(recalculated_global_db)
        print(f'Iteration {repeat+1} - Recalculated federated fuzzy Davies-Bouldin: {recalculated_global_db}')

    print(
        f'Min. recalculated DB: {np.min(list_all_recalculated_davies_bouldin)}.\n Max. recalculated DB: {np.max(list_all_recalculated_davies_bouldin)}')

    # Hypothesis testing
