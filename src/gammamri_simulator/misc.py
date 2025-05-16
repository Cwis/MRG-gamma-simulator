"""Miscelleanous code for tests"""
import numpy as np
import numpy.typing as npt


def as_array(tab_in: npt.ArrayLike) -> np.ndarray:
    """

    :param tab_in:
    :return:
    """
    return np.array(tab_in)


def pop_array(val: int) -> np.ndarray:
    """

    :param val:
    :return:
    """
    tab = np.ndarray((1, 1))
    tab[0] = val
    return tab


tableau: np.ndarray = np.zeros((3, 2), dtype=int)
print(tableau)
print(tableau[0][0])
print(as_array(tableau))
print(pop_array(2))
