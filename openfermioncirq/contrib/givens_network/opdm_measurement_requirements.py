"""
Plot the measurement requirements necessary to simulate
restricted Hartree-Fock vs. number of spatial orbitals
"""
from itertools import product
import numpy
import os
from glob import glob

# plotting imports
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from openfermion import FermionOperator, QubitOperator
from openfermion.transforms import jordan_wigner


def generat_opdm_observables(n_orbitals, ghf=False):
    """
    Generate the QubitOperator terms for the 1-RDM

    use rhf by default

    :param n_orbitals: number of spatial orbitals
    :return:
    """
    opdm_measurement_ops = QubitOperator()
    for i, j in product(range(n_orbitals), repeat=2):
        if i <= j: # take upper triangle
            for sigma in range(2):  # alpha/beta
                opdm_sigma = jordan_wigner(FermionOperator(((2 * i + 1 * sigma, 1), (2 * j + 1 * sigma, 0))))
                opdm_measurement_ops += opdm_sigma

    print(opdm_measurement_ops)

if __name__ == "__main__":
    generat_opdm_observables(2)

