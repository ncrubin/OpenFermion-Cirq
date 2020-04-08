from hfvqe.molecular_data_construction import h_n_linear_molecule
import pytest


def test_negative_n_hydrogen_chain():
    with pytest.raises(ValueError):
        h_n_linear_molecule(1.3, 0)