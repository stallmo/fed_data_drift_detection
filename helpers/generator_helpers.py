import logging

import numpy as np
from sklearn.datasets import make_spd_matrix
from data_generators import GaussianDataGenerator


def generate_random_vector(length: int, min_val: float, max_val: float):
    """

    :param length:
    :param min_val:
    :param max_val:
    :return:
    """

    vec = np.random.uniform(low=min_val,
                            high=max_val,
                            size=length
                            )

    return vec


def generate_random_covariance_matrix(dim: int):
    """
    Creates a random covariance matrix (symmetric, positive semi-definite matrix) of shape <dim> x <dim>

    :param dim: Dimension of covariance matrix.
    :return: Covariance matrix.
    """

    sigma = make_spd_matrix(n_dim=dim, random_state=None)
    return sigma


# generate fixed gaussian generators
def make_gaussian_data_generator(mean_vec, cov_mat) -> GaussianDataGenerator:
    data_generator = GaussianDataGenerator(mean=mean_vec,
                                           standard_deviation=cov_mat)

    return data_generator


# generate fixed mean, random covariance
def make_gaussian_data_generator_random_covariance(mean_vec) -> GaussianDataGenerator:
    """
    Returns a Gaussian data generator with fixed mean, but random covariance matrix.

    :param mean_vec: Mean of (multivariate) Gaussian.
    :return: GaussianDataGenerator with specified mean and random covariance matrix.
    """
    dim = len(mean_vec)
    cov_mat = generate_random_covariance_matrix(dim=dim)

    data_generator = make_gaussian_data_generator(mean_vec=mean_vec,
                                                  cov_mat=cov_mat
                                                  )
    return data_generator


# generate random mean, fixed covariance generator
def make_gaussian_data_generator_random_mean(cov_mat, min_val: float, max_val: float) -> GaussianDataGenerator:
    """
    Returns a GaussianDataGenerator with random mean and fixed covariance matrix.

    :param cov_mat: Covariance matrix
    :param min_val: Minimum value in random mean vector
    :param max_val: Maximum value in random mean vector
    :return: GaussianDataGenerator
    """
    dim = len(cov_mat)
    mean_vec = generate_random_vector(length=dim,
                                      min_val=min_val,
                                      max_val=max_val
                                      )

    data_generator = make_gaussian_data_generator(mean_vec=mean_vec,
                                                  cov_mat=cov_mat
                                                  )

    return data_generator


# generate random mean, random covariance
def make_gaussian_data_generator_random_mean_random_cov(dim: int, mean_min_val: float,
                                                        mean_max_val: float) -> GaussianDataGenerator:
    """
    Returns a GaussianDataGenerator with random mean and random covariance matrix.

    :param dim: Dimension of the Gaussian data.
    :param mean_min_val: Minimum value in random mean vector.
    :param mean_max_val: Maximum value in random mean vector.
    :return: GaussianDataGenerator
    """
    mean_vec = generate_random_vector(length=dim,
                                      min_val=mean_min_val,
                                      max_val=mean_max_val
                                      )
    cov_mat = generate_random_covariance_matrix(dim=dim)

    data_generator = make_gaussian_data_generator(mean_vec=mean_vec,
                                                  cov_mat=cov_mat
                                                  )
    return data_generator

# function to generate new data for each client
def generate_data_from_generators(generators_per_client, n_points_per_generator):
    """
    Description: Generate data from generators for each client.

    :param generators_per_client: list of generators
    :param n_points_per_generator: number of points to generate per generator
    :return: np.array of shape (n_points_per_generator * n_generators_per_client, data_dim)
    """

    X_client = np.concatenate(
        [generator.generate_data_points(n_points_per_generator, seed=None) for generator in generators_per_client])

    return X_client


def generate_data_from_local_generators(dict_init_generators_per_client: dict, n_points_new_data: int, n_points_old_data: int,
                                        n_unused_client_generators: int):
    """
    Description: Generate new data for each client by using generators from other clients.

    :param dict_init_generators_per_client: Dictionary with client as key and list of generators as value
    :param n_points_new_data: integer, number of points to generate per generator of other client
    :param n_points_old_data: integer, number of points to generate per generator of own client
    :param n_unused_client_generators: integer, number of generators that are not used for generating any new data
    :return: dictionary with client as key and new data as value
    """

    dict_new_data_per_client = {}

    _n_clients = len(dict_init_generators_per_client.keys())
    random_constant = np.random.choice(range(1, _n_clients))
    # randomly choose clients of which generators are not used for generating new data
    unused_generators_clients = np.random.choice(range(_n_clients), size=n_unused_client_generators, replace=False)

    for _client, init_data_generators in dict_init_generators_per_client.items():

        # choose other_client by adding a random constant to _client and taking modulo _n_clients
        other_client = (_client + random_constant) % _n_clients
        # if other_client is in unused_generators_clients, choose another client
        while other_client in unused_generators_clients:
            other_client = (other_client + 1) % _n_clients

        logging.debug(f'Client {_client} chooses client {other_client}.')

        # generate new data with generators from other client
        _client_generators = dict_init_generators_per_client.get(other_client)
        X_new_data_client = generate_data_from_generators(generators_per_client=_client_generators,
                                                          n_points_per_generator=n_points_new_data)

        if n_points_old_data > 0:
            # init_data_generators = dict_init_generators_per_client.get(_client)

            X_old_data_client = generate_data_from_generators(generators_per_client=init_data_generators,
                                                              n_points_per_generator=n_points_old_data)
            X_new_data_client = np.concatenate([X_new_data_client, X_old_data_client])

        dict_new_data_per_client[_client] = X_new_data_client

    return dict_new_data_per_client


def generate_new_data(run_mode, n_points_per_generator, dict_init_generators_per_client, clients_global_drift=1,
                      mean_min_val=-5, mean_max_val=5, data_dim=2, ratio_new_data_distribution=1.0,
                      n_unused_client_generators=0):
    """
    Description: Depending on the run_mode, generate new data for each client.

    :param data_dim: Dimension of data
    :param mean_min_val: Minimum value for random mean of Gaussian data generators
    :param mean_max_val: Maximum value for random mean of Gaussian data generators
    :param clients_global_drift: number of clients that are affected by global drift
    :param n_points_per_generator: number of points to generate per generator
    :param dict_init_generators_per_client: dictionary with client as key and list of generators as value
    :param run_mode: 'no_local_drift' | 'local_but_no_global_drift' | 'global_drift' | 'global_drift_unused_generators'
    :param ratio_new_data_distribution: ratio of new data that is generated by new distributions
    :param n_unused_client_generators: number of generators that are not used for generating new data.
    :return: dictionary with client as key and new data as value
    """

    if run_mode not in ['no_local_drift', 'local_but_no_global_drift', 'global_drift', 'global_drift_unused_generators']:
        raise NotImplementedError

    dict_new_data_per_client = {}
    logging.debug(f'Creating new data for each client. Ratio of new data distribution: {ratio_new_data_distribution}.')
    logging.debug(f'Generating {n_points_per_generator} data points per generator.')
    n_points_new_data = int(n_points_per_generator * ratio_new_data_distribution)
    n_points_old_data = n_points_per_generator - n_points_new_data
    logging.debug(f'Generating {n_points_new_data} new data points and {n_points_old_data} old data points per client.')

    if run_mode == 'no_local_drift':
        # every client has data from its own generators
        for _client, _client_generators in dict_init_generators_per_client.items():
            X_new_data_client = np.concatenate(
                [generator.generate_data_points(n_points_per_generator, seed=None) for generator in _client_generators])
            dict_new_data_per_client[_client] = X_new_data_client

    elif run_mode == 'local_but_no_global_drift':

        dict_new_data_per_client = generate_data_from_local_generators(
            dict_init_generators_per_client=dict_init_generators_per_client,
            n_points_new_data=n_points_new_data, n_points_old_data=n_points_old_data,
            n_unused_client_generators=0) # all generators are used for generating new data to ensure no global drift

    elif run_mode == 'global_drift_unused_generators':
        dict_new_data_per_client = generate_data_from_local_generators(
            dict_init_generators_per_client=dict_init_generators_per_client,
            n_points_new_data=n_points_new_data, n_points_old_data=n_points_old_data,
            n_unused_client_generators=n_unused_client_generators) # some generators are not used for generating new data to ensure global drift

    elif run_mode == 'global_drift':
        _n_clients = len(dict_init_generators_per_client.keys())
        drifted_clients = np.random.choice(range(_n_clients), size=clients_global_drift, replace=False)

        for _client, _client_generators in dict_init_generators_per_client.items():
            if _client in drifted_clients:
                # generate new data with random mean and random covariance
                new_data_generator = make_gaussian_data_generator_random_mean_random_cov(dim=data_dim,
                                                                                         mean_min_val=mean_min_val,
                                                                                         mean_max_val=mean_max_val)

                X_new_data_client = new_data_generator.generate_data_points(n_points_new_data, seed=None)

                # add data from initial generators
                if n_points_old_data > 0:
                    init_data_generators = dict_init_generators_per_client.get(_client)

                    X_old_data_client = generate_data_from_generators(generators_per_client=init_data_generators,
                                                                      n_points_per_generator=n_points_old_data)
                    X_new_data_client = np.concatenate([X_new_data_client, X_old_data_client])

            else:
                X_new_data_client = generate_data_from_generators(generators_per_client=_client_generators,
                                                                  n_points_per_generator=n_points_per_generator)

            dict_new_data_per_client[_client] = X_new_data_client

    return dict_new_data_per_client