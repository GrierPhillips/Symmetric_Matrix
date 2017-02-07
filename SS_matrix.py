"""Sparse Symmetrical Matrix"""

import numpy as np
import scipy.sparse as ss
from scipy.sparse.compressed import _cs_matrix
from scipy.sparse.sputils import IndexMixin



class SparseSymmetricMatrix(_cs_matrix, IndexMixin):
    def __init__(self, matrix, offset=0):
        if offset not in [0,1]:
            raise ValueError(("Sparse Symmetrical Matrix can only be constructed"
                             " with a value of 0 or 1 for offset."))
        self.matrix = matrix
        self.k = offset
        if not self.check_sym():
            raise ValueError('Matrix must be symmetric.')

    def check_sym(self):
        '''Check if the input matrix is symmetric.'''
        return (self.matrix.T == self.matrix).all()

    def sym_to_sparse(self):
        '''Convert a symmetric matrix into a sparse matrix by removing the lower
        triangle.

        Input:
            matrix: A symmetrical 2 dimensional numpy array
            k: The offset value for the triangle; 1 or 0 only
        Output:
            sparse_matrix: A sparse matrix in Compressed Sparse Row format
        '''
        out = self.matrix[triu_indices_from(self.matrix, k=self.k)].flatten()
        self.matrix = ss.csr_matrix(out)

    def __getitem__(self, idx):
        '''Return proper values from the matrix given either a tuple of indices
        or a single value for row.

        It is important to note that when a user calls a single row from the
        converted array we will need to reconstruct the missing values for that
        row.

        e.g. A = [[0,1,2],  --->    SparseSymmetricMatrix(A) = [0,1,2,0,2,0]
                  [1,0,2],
                  [2,2,0]]

            A[0,1] = 1      --->    SparseSymmetricMatrix(A)[1] = 1

            This first conversion can be handled by the formula:

                A[i,j] = SparseSymmetricMatrix[k]
                l = (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - offset

            A[0] = [0,1,2]  --->    SparseSymmetricMatrix(A)?

            To do: Determine how to reconstruct row from 1d array.
        '''
        i, j = self._unpack_index(idx)
        n = self.matrix.shape[0]
        l = (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - self.k
        return self.matrix[l]
