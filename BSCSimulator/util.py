import json

import numpy as np
import pandas as pd
from numpy import ndarray


def dummy_population_phenotypes(config_file):
    """Create a dummy population phenotype table.

    :param config_file: Path to the TSV file containing the major blood group distribution in the population.
    :return: A pandas dataframe containing the population phenotype table.
    """
    df = pd.read_csv(config_file, sep='\t')
    df = df.drop(columns=['ABOD'])
    return df


def list_of_permutations(domain_list) -> list:
    prototype = []
    num_permutations = 1
    divisors = []
    len_dl = len(domain_list)
    for i in range(len_dl - 1, -1, -1):
        domain = domain_list[i]
        prototype = prototype + [domain[0]]
        len_d = len(domain)
        num_permutations = num_permutations * len_d
        divisors.append(num_permutations / len_d)
    permutation_list = []
    for i in range(num_permutations):
        permutation = []
        for j in range(len_dl):
            domain = domain_list[j]
            k = int(i / divisors[(len_dl - 1 - j)]) % len(domain)
            permutation.append(domain[k])
        permutation_list.append(permutation)
    return permutation_list


def _normalize(Y, normalization_type='stats'):
    """Normalize the vector Y using statistics or its range.

    :param Y: Row or column vector that you want to normalize.
    :param normalization_type: String specifying the kind of normalization
    to use. Options are 'stats' to use mean and standard deviation,
    or 'maxmin' to use the range of function values.
    :return Y_normalized: The normalized vector.
    """
    Y = np.asarray(Y, dtype=float)

    if np.max(Y.shape) != Y.size:
        raise NotImplementedError('Only 1-dimensional arrays are supported.')

    # Only normalize with non null sdev (divide by zero). For only one
    # data point both std and ptp return 0.
    if normalization_type == 'stats':
        Y_norm = Y - Y.mean()
        std = Y.std()
        if std > 0:
            Y_norm /= std
    elif normalization_type == 'maxmin':
        Y_norm = Y - Y.min()
        y_range = np.ptp(Y)
        if y_range > 0:
            Y_norm /= y_range
            # A range of [-1, 1] is more natural for a zero-mean GP
            Y_norm = 2 * (Y_norm - 0.5)
    else:
        raise ValueError(
            'Unknown normalization type: {}'.format(normalization_type))

    return Y_norm


def normalize(
        y: ndarray | list,
        norm_type='stats', *, upper=1, lower=-1, anchor_upper=None, anchor_lower=None) -> ndarray:
    """
    Normalizing the array using statistics of its range.
    Normalisation takes place along each column.

    :param y: 2-D array-like data that you want to normalize.
    :param norm_type: String specifying the kind of normalisation to use.
                        Options are `stats' to use mean and standard deviation, or
                        'maxmin' to use range of function values so that elements
                        lie in range [lower, ``upper].
    :param upper: Upper bound on normalised values when using norm_type='maxmin'
    :param lower: Lower bound on normalised values when using norm_type='maxmin'
    :param anchor_upper: Fixed upper bound on values in ``y``.
                            If specified, a row vector with values equal to ``anchor_upper`` is
                            appended to ``y`` before normalisation and removed after.
    :param anchor_lower: Fixed lower bound on values in ``y``.
                            If specified, a row vector with values equal to ``anchor_lower`` is
                            appended to ``y`` before normalisation and removed after.
    :return: A normalized numpy array.

    Example
    ---------
    >>> y = np.array([[1,3,5,5], [1,4,6,3], [1,6,9,0]])
    >>> y
    array([[1, 3, 5, 5],
           [1, 4, 6, 3],
           [1, 6, 9, 0]])
    >>> normalize(y, 'maxmin', upper=1, lower=0)
    array([[0.5       , 0.        , 0.        , 1.        ],
           [0.5       , 0.33333333, 0.25      , 0.6       ],
           [0.5       , 1.        , 1.        , 0.        ]])
    >>> normalize(y)
    array([[ 0.        , -1.06904497, -0.98058068,  1.13554995],
           [ 0.        , -0.26726124, -0.39223227,  0.16222142],
           [ 0.        ,  1.33630621,  1.37281295, -1.29777137]])
    >>> normalize(y, 'maxmin')
    array([[ 0.        , -1.        , -1.        ,  1.        ],
           [ 0.        , -0.33333333, -0.5       ,  0.2       ],
           [ 0.        ,  1.        ,  1.        , -1.        ]])
    """
    y_arr = np.asarray(y, dtype=float)

    anchors = 0
    if anchor_upper is not None:
        if len(y_arr.shape) == 1:
            y_arr = np.append(y_arr, anchor_upper)
        else:
            anchor_vec = np.ones((1, y_arr.shape[1])) * anchor_upper
            y_arr = np.vstack((y_arr, anchor_vec))
        anchors += 1
    if anchor_lower is not None:
        if len(y_arr.shape) == 1:
            y_arr = np.append(y_arr, anchor_lower)
        else:
            anchor_vec = np.ones((1, y_arr.shape[1])) * anchor_lower
            y_arr = np.vstack((y_arr, anchor_vec))
        anchors += 1

    if norm_type == 'maxmin' and upper <= lower:
        raise ValueError('Upper bound must be greater than lower bound.')

    if len(y_arr.shape) == 1:
        y_norm = _normalize(y_arr, norm_type)
        if norm_type == 'maxmin':
            _hr = (upper - lower)/2
            _mean = (upper + lower)/2
            y_norm = _hr * y_norm + _mean
        y_norm = y_norm[:len(y_norm)-anchors]
        return y_norm

    if norm_type == 'stats':
        y_mean = y.mean(axis=0)
        y_norm = y_arr - y_mean
        y_std = y_arr.std(axis=0)
        gt_zero = np.flatnonzero(y_std)

        y_norm[:, gt_zero] /= y_std[gt_zero]
    elif norm_type == 'maxmin':
        y_min = y_arr.min(axis=0)
        y_norm = y_arr - y_min
        y_range = y_arr.ptp(axis=0)
        gt_zero = np.flatnonzero(y_range)
        eq_zero = np.flatnonzero(np.sum(y_norm, axis=0) == 0)
        _range = upper - lower

        y_norm[:, gt_zero] /= y_range[gt_zero]
        y_norm[:, eq_zero] = 0.5
        y_norm = _range * y_norm + lower
    else:
        raise ValueError('Unknown normalization type: {}'.format(norm_type))

    y_norm = y_norm[:len(y_norm)-anchors]
    return y_norm


class UniqueIDTicker:
    def __init__(self, start: int = 0):
        self._id = start
        self._start = start

    def get(self):
        return self._id

    def tick(self, get: bool = False):
        self._id += 1
        if get:
            return self._id

    def increment(self, value: int):
        self._id += value

    def __add__(self, other: int):
        return self._id + other

    def __radd__(self, other: int):
        return self._id + other

    def __sub__(self, other: int):
        return self._id - other

    def __rsub__(self, other: int):
        return self._id - other
