from functools import reduce
import numpy
import scipy
from openfermion.ops._givens_rotations import (givens_matrix_elements, givens_rotate)
from openfermioncirq.contrib.givens_network.front_back_givens import front_and_back_givens_decomposition

import cirq


def test_givens_inverse():
    """
    The Givens rotation in OpenFermion is defined as

    .. math::

        \begin{pmatrix}
            \cos(\theta) & -e^{i \varphi} \sin(\theta) \\
            \sin(\theta) &     e^{i \varphi} \cos(\theta)
        \end{pmatrix}.

    confirm numerically its hermitian conjugate is it's inverse
    """
    a = numpy.random.random() + 1j * numpy.random.random()
    b = numpy.random.random() + 1j * numpy.random.random()
    ab_rotation = givens_matrix_elements(a, b, which='right')

    assert numpy.allclose(ab_rotation @ numpy.conj(ab_rotation).T, numpy.eye(2))
    assert numpy.allclose(numpy.conj(ab_rotation).T @ ab_rotation, numpy.eye(2))


def test_row_eliminate():
    """
    Test elemination of element in U[i, j] by rotating in i-1 and i.
    """
    dim = 3
    u_generator = numpy.random.random((dim, dim)) + 1j * numpy.random.random((dim, dim))
    u_generator = u_generator - numpy.conj(u_generator).T

    # make sure the generator is actually antihermitian
    assert numpy.allclose(-1 * u_generator, numpy.conj(u_generator).T)

    unitary = scipy.linalg.expm(u_generator)

    # eliminate U[2, 0] by rotating in 1, 2
    gmat = givens_matrix_elements(unitary[1, 0], unitary[2, 0], which='right')
    givens_rotate(unitary, gmat, 1, 2, which='row')
    assert numpy.isclose(unitary[2, 0], 0.0)

    # eliminate U[1, 0] by rotating in 0, 1
    gmat = givens_matrix_elements(unitary[0, 0], unitary[1, 0], which='right')
    givens_rotate(unitary, gmat, 0, 1, which='row')
    assert numpy.isclose(unitary[1, 0], 0.0)

    # eliminate U[2, 1] by rotating in 1, 2
    gmat = givens_matrix_elements(unitary[1, 1], unitary[2, 1], which='right')
    givens_rotate(unitary, gmat, 1, 2, which='row')
    assert numpy.isclose(unitary[2, 1], 0.0)


def create_givens(givens_mat, i, j, dim):
    """
    Create the givens matrix on the larger space

    :param givens_mat: 2x2 matrix with first column is real
    :param i: row index i
    :param j: row index i < j
    :param dim: dimension
    """
    gmat = numpy.eye(dim, dtype=complex)
    gmat[i, i] = givens_mat[0, 0]
    gmat[i, j] = givens_mat[0, 1]
    gmat[j, i] = givens_mat[1, 0]
    gmat[j, j] = givens_mat[1, 1]
    return gmat


def test_col_eliminate():
    """
    Test elimination by rotating in the column space.  Left multiplication of inverse givens
    """
    dim = 3
    u_generator = numpy.random.random((dim, dim)) + 1j * numpy.random.random((dim, dim))
    u_generator = u_generator - numpy.conj(u_generator).T
    # make sure the generator is actually antihermitian
    assert numpy.allclose(-1 * u_generator, numpy.conj(u_generator).T)
    unitary = scipy.linalg.expm(u_generator)

    # eliminate U[1, 0] by rotation in rows [0, 1] and mixing U[1, 0] and U[0, 0]
    unitary_original = unitary.copy()
    gmat = givens_matrix_elements(unitary[0, 0], unitary[1, 0], which='right')
    vec = numpy.array([[unitary[0, 0]], [unitary[1, 0]]])
    fullgmat = create_givens(gmat, 0, 1, 3)
    zeroed_unitary = fullgmat @ unitary

    givens_rotate(unitary, gmat, 0, 1)
    assert numpy.isclose(unitary[1, 0], 0.0)
    assert numpy.allclose(unitary.real, zeroed_unitary.real)
    assert numpy.allclose(unitary.imag, zeroed_unitary.imag)

    # eliminate U[2, 0] by rotating columns [0, 1] and mixing U[2, 0] and U[2, 1].
    unitary = unitary_original.copy()
    gmat = givens_matrix_elements(unitary[2, 0], unitary[2, 1], which='left')
    vec = numpy.array([[unitary[2, 0]], [unitary[2, 1]]])

    assert numpy.isclose((gmat @ vec)[0, 0], 0.0)
    assert numpy.isclose((vec.T @ gmat.T)[0, 0], 0.0)
    fullgmat = create_givens(gmat, 0, 1, 3)
    zeroed_unitary = unitary @ fullgmat.T
    # print(unitary @ fullgmat.T)

    # because col takes g[0, 0] * col_i + g[0, 1].conj() * col_j -> col_i
    # this is equivalent ot left multiplication by gmat.T
    givens_rotate(unitary, gmat.conj(), 0, 1, which='col')
    assert numpy.isclose(zeroed_unitary[2, 0], 0.0)
    assert numpy.allclose(unitary, zeroed_unitary)


def test_front_back_iteration():
    """
    Code demonstrating how we iterated over the matrix

    [[ 0.  0.  0.  0.  0.  0.]
     [15.  0.  0.  0.  0.  0.]
     [ 7. 14.  0.  0.  0.  0.]
     [ 6.  8. 13.  0.  0.  0.]
     [ 2.  5.  9. 12.  0.  0.]
     [ 1.  3.  4. 10. 11.  0.]]
    """
    N = 6
    unitary = numpy.zeros((N, N))
    unitary[-1, 0] = 1
    unitary[-2, 0] = 2
    unitary[-1, 1] = 3
    unitary[-1, 2] = 4
    unitary[-2, 1] = 5
    unitary[-3, 0] = 6
    unitary[-4, 0] = 7
    unitary[-3, 1] = 8
    unitary[-2, 2] = 9
    unitary[-1, 3] = 10
    unitary[-1, 4] = 11
    unitary[-2, 3] = 12
    unitary[-3, 2] = 13
    unitary[-4, 1] = 14
    unitary[-5, 0] = 15
    counter = 1
    for i in range(1, N):
        if i % 2 == 1:
            for j in range(0, i):
                print((N - j, i - j), i - j, i - j  + 1, "col rotation")
                assert numpy.isclose(unitary[N - j - 1, i - j - 1], counter)
                counter += 1
        else:
            for j in range(1, i + 1):
                print((N + j - i, j), N + j - i - 1, N + j - i, "row rotation")
                assert numpy.isclose(unitary[N + j - i - 1, j - 1], counter)
                counter += 1


def test_front_back_elimination():
    N = 3
    dim = N
    u_generator = numpy.random.random((dim, dim)) + 1j * numpy.random.random((dim, dim))
    u_generator = u_generator - numpy.conj(u_generator).T
    # make sure the generator is actually antihermitian
    assert numpy.allclose(-1 * u_generator, numpy.conj(u_generator).T)
    unitary = scipy.linalg.expm(u_generator)

    original_unitary = unitary.copy()

    right_rotations = []
    right_full_rotations = []
    left_rotations = []
    left_full_rotations = []
    for i in range(1, N):
        if i % 2 == 1:
            for j in range(0, i):
                # eliminate U[N - j, i - j] by mixing U[N - j, i - j], U[N - j, i - j - 1] by right multiplication
                # of a givens rotation matrix in column [i - j, i - j + 1]
                print((N - j, i - j), i - j, i - j + 1, "col rotation")
                gmat = givens_matrix_elements(unitary[N - j - 1, i - j - 1], unitary[N - j - 1, i - j - 1 + 1], which='left')
                right_rotations.append((gmat.T, (i - j - 1, i - j)))
                right_full_rotations.append(create_givens(gmat.T, i - j - 1, i - j, N))
                # unitary = unitary @ fullgmat.T
                givens_rotate(unitary, gmat.conj(), i - j - 1, i - j, which='col')
        else:
            for j in range(1, i + 1):
                # elimination of U[N + j - i, j] by mixing U[N + j - i, j] and U[N + j - i - 1, j] by left multiplication
                # of a givens rotation that rotates row space [N + j - i - 1, N + j - i
                print((N + j - i, j), N + j - i - 1, N + j - i, "row rotation")
                gmat = givens_matrix_elements(unitary[N + j - i - 1 - 1, j - 1], unitary[N + j - i - 1, j - 1],  which='right')
                left_rotations.append((gmat, (N + j - i - 2, N + j - i - 1)))
                left_full_rotations.append(create_givens(gmat, N + j - i - 1 - 1, N + j - i - 1, N))
                # unitary = fullgmat @ unitary
                givens_rotate(unitary, gmat, N + j - i - 2, N + j - i - 1, which='row')

    assert numpy.allclose(numpy.diag(numpy.diag(unitary)) - unitary, 0.0)

    # now check to reconstruct u
    test_left_unitary = numpy.eye(N, dtype=complex)
    for left_gmat in left_full_rotations:
        test_left_unitary = test_left_unitary @ left_gmat.conj().T
    # assert numpy.allclose(left_full_rotations[0].conj().T @ left_full_rotations[1].conj().T, test_left_unitary)

    test_right_unitary = numpy.eye(N, dtype=complex)
    for right_gmat in reversed(right_full_rotations):
        test_right_unitary = test_right_unitary @ right_gmat.conj().T

    # assert numpy.allclose(right_full_rotations[0].conj().T, test_right_unitary)
    assert numpy.allclose(original_unitary, test_left_unitary @ unitary @ test_right_unitary)

    # now check to reconstruct u from gmats
    test_left_unitary = numpy.eye(N, dtype=complex)
    for (left_gmat, (i, j)) in left_rotations:
        lgmat = create_givens(left_gmat.conj().T, i, j, N)
        test_left_unitary = test_left_unitary @ lgmat
    # assert numpy.allclose(left_full_rotations[0].conj().T @ left_full_rotations[1].conj().T, test_left_unitary)

    test_right_unitary = numpy.eye(N, dtype=complex)
    for (right_gmat, (i, j)) in reversed(right_rotations):
        rgmat = create_givens(right_gmat.conj().T, i, j, N)
        test_right_unitary = test_right_unitary @ rgmat

    # assert numpy.allclose(right_full_rotations[0].conj().T, test_right_unitary)
    assert numpy.allclose(original_unitary, test_left_unitary @ unitary @ test_right_unitary)

    # new_left = []
    # # # now check if we can move the phase all the way to the front
    # left_gmat, (i, j) = left_rotations[-1]
    # phase_matrix = numpy.diag([unitary[i, i], unitary[j, j]])
    # matrix_to_decompose = left_gmat.conj().T @ phase_matrix
    # new_givens_matrix = givens_matrix_elements(matrix_to_decompose[1, 0], matrix_to_decompose[1, 1], which='left')
    # new_phase_matrix = matrix_to_decompose @ new_givens_matrix.T
    # assert numpy.allclose(new_phase_matrix @ new_givens_matrix.conj(), matrix_to_decompose)
    # unitary[i, i] = new_phase_matrix[0, 0]
    # unitary[j, j] = new_phase_matrix[1, 1]
    # new_left.append((new_givens_matrix.conj(), (i, j)))

    # left_gmat, (i, j) = left_rotations[-2]
    # phase_matrix = numpy.diag([unitary[i, i], unitary[j, j]])
    # matrix_to_decompose = left_gmat.conj().T @ phase_matrix
    # new_givens_matrix = givens_matrix_elements(matrix_to_decompose[1, 0], matrix_to_decompose[1, 1], which='left')
    # new_phase_matrix = matrix_to_decompose @ new_givens_matrix.T
    # assert numpy.allclose(new_phase_matrix @ new_givens_matrix.conj(), matrix_to_decompose)
    # unitary[i, i] = new_phase_matrix[0, 0]
    # unitary[j, j] = new_phase_matrix[1, 1]
    # new_left.append((new_givens_matrix.conj(), (i, j)))

    # print(new_left)
    # test_left = numpy.eye(N, dtype=complex)
    # test_left = test_left @ create_givens(new_left[1][0], new_left[1][1][0], new_left[1][1][1], N)
    # test_left = test_left @ create_givens(new_left[0][0], new_left[0][1][0], new_left[0][1][1], N)
    # print(unitary @ test_left @ test_right_unitary)
    # print(original_unitary)
    # exit()

    # # now use 2x2 givens rotation to decompose the unitary into a phase piece and a givens piece
    # updater = givens_matrix_elements(utest[1, 0], utest[1, 1], which='left')
    # phase_piece, givens_piece = utest @ updater.T, updater.T

    new_left_rotations = []
    for (left_gmat, (i, j)) in reversed(left_rotations):
        phase_matrix = numpy.diag([unitary[i, i], unitary[j, j]])
        matrix_to_decompose = left_gmat.conj().T @ phase_matrix
        new_givens_matrix = givens_matrix_elements(matrix_to_decompose[1, 0], matrix_to_decompose[1, 1], which='left')
        new_phase_matrix = matrix_to_decompose @ new_givens_matrix.T

        full_new_givens = create_givens(new_givens_matrix.T, i, j, N)
        full_new_phase = create_givens(left_gmat.conj().T, i, j, N) @ unitary @ full_new_givens

        assert numpy.allclose(full_new_phase - numpy.diag(numpy.diag(full_new_phase)), 0.0) # should always be diagonal

        # check if T_{m,n}^{-1}D  = D T
        assert numpy.allclose(new_phase_matrix @ new_givens_matrix.conj(), matrix_to_decompose)

        unitary[i, i], unitary[j, j] = new_phase_matrix[0, 0], new_phase_matrix[1, 1]
        new_left_rotations.append((new_givens_matrix.conj(), (i, j)))

    # now reconstruct the original matrix with the new givens matrices.

    test_left_unitary = numpy.eye(N, dtype=complex)
    for (left_gmat, (i, j)) in reversed(new_left_rotations):
        lgmat = create_givens(left_gmat, i, j, N)
        test_left_unitary = test_left_unitary @ lgmat

    assert numpy.allclose(unitary @ test_left_unitary @ test_right_unitary, original_unitary)


def test_generate_givens_circuits():

    from openfermioncirq.primitives import prepare_slater_determinant
    from openfermion import slater_determinant_preparation_circuit
    from openfermion.ops._givens_rotations import givens_decomposition
    N = 4
    dim = N
    u_generator = numpy.random.random((dim, dim))
    u_generator = u_generator - numpy.conj(u_generator).T
    # make sure the generator is actually antihermitian
    assert numpy.allclose(-1 * u_generator, numpy.conj(u_generator).T)
    unitary = scipy.linalg.expm(u_generator)
    # alpha_beta_unitary = numpy.kron(numpy.eye(2), unitary)
    qubits = cirq.LineQubit.range(4)

    print(unitary)
    circuit = cirq.Circuit()
    # circuit_description = slater_determinant_preparation_circuit(
    #         unitary.conj().T[:2, :])
    circuit.append(prepare_slater_determinant(qubits, unitary.conj().T[:2, :]))
    print(circuit)



if __name__ == "__main__":
    numpy.random.seed(4)
    # test_givens_inverse()
    # test_row_eliminate()
    # test_col_eliminate()
    # test_front_back_iteration()
    # test_front_back_elimination()
    test_generate_givens_circuits()