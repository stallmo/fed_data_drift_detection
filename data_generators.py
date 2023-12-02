from abc import ABC, abstractmethod
import numpy as np


class BaseDataGenerator(ABC):

    def __int__(self, dist_name: str, dimension: int, params: dict):
        self.__name = dist_name
        self.__dim = dimension
        self.__params_dict = params

    @abstractmethod
    def generate_data_points(self, n_points: int):
        """

        :param n_points:
        :return:
        """
        pass


class GaussianDataGenerator(BaseDataGenerator):

    def __init__(self, mean, standard_deviation):
        """

        :param mean:
        :param standard_deviation:
        :return:
        """

        self.__mean = np.array(mean)
        self.__std = np.array(standard_deviation)

        assert self.__std.shape[0] == self.__std.shape[1], "Covariance matrix must be NxN."
        assert self.__mean.shape[0] == self.__std.shape[
            0], "Mean vector and covariance matrix must have matching dimension."

        self.__name = 'GaussianDataGenerator'
        self.__dim = len(mean)

        param_dict = {'mean': self.__mean,
                      'std': self.__std,
                      }
        super().__int__(dist_name=self.__name,
                        params=param_dict,
                        dimension=self.__dim
                        )

    @property
    def mean(self):
        return self.__mean

    @property
    def standard_deviation(self):
        return self.__std

    def generate_data_points(self, n_points: int, seed=43):
        """

        :param n_points:
        :param seed:
        :return:
        """
        rng = np.random.default_rng(seed=seed)

        X = rng.multivariate_normal(mean=self.__mean,
                                    cov=self.__std,
                                    size=n_points
                                    )
        return X
