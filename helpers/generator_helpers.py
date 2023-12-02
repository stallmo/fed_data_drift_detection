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

    return
