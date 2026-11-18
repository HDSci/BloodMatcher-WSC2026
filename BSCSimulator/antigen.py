import os

import numpy as np
import pandas as pd

DEFAULT_ANTIGEN_ORDER = ('A', 'B', 'D', 'C', 'c', 'E', 'e', 'K', 'k', 'Fya', 'Fyb')

DEFAULT_VECTOR_LENGTH = 2 ** 5


class Antigens:
    population_frequencies = None
    population_abd_usabilities = None

    def __init__(self, antigens: list, antigen_order=None,
                 alloimmunisation_risk: pd.DataFrame=None, mismatch_weights: pd.DataFrame=None,
                 allo_Abs=None,) -> None:
        self.antigens = antigens
        self.antigen_order = DEFAULT_ANTIGEN_ORDER if antigen_order is None else antigen_order
        self.vector_length = len(antigens)
        assert set(self.antigen_order).issubset(set(self.antigens))
        self.mask = 2 ** self.vector_length - 1
        self.reference = self._setup_antigen_dict(antigens, self.antigen_order, self.vector_length)
        self.antigen_index = list(self.reference.keys())
        self.allo_risk = np.ones(self.vector_length - 3) if alloimmunisation_risk is None else alloimmunisation_risk
        self.major_mask = [a in ['A', 'B', 'D'] for a in self.antigen_index]
        self.all_minor_mask = [a not in ['A', 'B', 'D'] for a in self.antigen_index]
        rh_kell = ('C', 'c', 'E', 'e', 'K')
        self.rhkell_mask = [a in rh_kell for a in self.antigen_index]
        self.alloantibody_freqs = allo_Abs
        self._setup_convert_to_binarray()
        self.immuno_risks: pd.DataFrame = alloimmunisation_risk
        self.mismatch_weights: pd.DataFrame = mismatch_weights
        antigen_index = np.array(self.antigen_index)
        self.xA_rate = self.immuno_risks[antigen_index[self.all_minor_mask]].to_numpy().flatten()

    def _setup_antigen_dict(self, antigens, antigen_order=DEFAULT_ANTIGEN_ORDER, length=DEFAULT_VECTOR_LENGTH):
        power = int(length - 1)
        ant_dict = dict()
        _antigens = list(antigen_order) + antigens
        for antigen in _antigens:
            if antigen in ant_dict:
                continue
            ant_dict.update({antigen: 2 ** power})
            power -= 1
        return ant_dict

    def _setup_convert_to_binarray(self):
        self._binarrays = self._convert_to_binarray([i for i in range(2 ** self.vector_length)])

    def convert_to_binarray(self, a: int | list[int]) -> np.ndarray:
        return self._binarrays[a]
    
    def _convert_to_binarray(self, a):
        if isinstance(a, int):
            return np.array(binarray(a, self.vector_length))
        else:
            return np.array([binarray(i, self.vector_length) for i in a])

    def binarray_to_int(self, phen_array):
        pows = np.arange(len(self.antigen_index) - 1, -1, -1)
        bin_pows = 2 ** pows
        ints = phen_array.dot(bin_pows[:, None])
        return ints


def binarray(a: int, w: int = 8):
    b = [int(i) for i in f'{a:0{w}b}']
    return b


def bnot(a: int, m: int = 0):
    if m == 0:
        m = 2 ** a.bit_length() - 1
    return ~a & m


def not_compatible(a: int, b: int, m: int = 0):
    return bnot(a, m) & b


def not_exact_match_aborhd(a: int, b: int):
    return a != b


def not_exact_match_abo(a: int, b: int, remove_rhd: bool = False):
    if remove_rhd:
        a = a >> 1
        b = b >> 1
    return a != b


def not_exact_match_rhd(a: int, b: int, remove_abo: bool = False):
    if remove_abo:
        a = a & 0b001
        b = b & 0b001
    return a != b
