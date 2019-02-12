# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
utilities for contstucting givesn rotations
"""
from openfermion.ops._givens_rotations import (givens_matrix_elements, givens_rotate)


def front_and_back_givens_decomposition(unitary):
    """
    Perform the Givens diagonalization
    """
    N = unitary.shape[0]
    for i in range(1, N):
        if i % 2 == 1:
            for j in range(0, i):
                print((N - j, i - j), i - j, i - j  + 1, "col rotation")
        else:
            for j in range(1, i + 1):
                print((N + j - i, j), N + j - i - 1, N + j - i, "row rotation")


