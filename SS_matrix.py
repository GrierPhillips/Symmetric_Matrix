"""Sparse Symmetrical Matrix"""

import numpy as np
import scipy.sparse as ss
from scipy.sparse.compressed import _cs_matrix
from scipy.sparse.sputils import IndexMixin


class SparseSymmetricMatrix(_cs_matrix, IndexMixin):
    def __init__(self, matrix, offset=0, diag=None):
        if offset not in [0,1]:
            raise ValueError(("Sparse Symmetrical Matrix can only be constructed"
                             " with a value of 0 or 1 for offset."))
        if not self.check_sym():
            raise ValueError('Matrix must be symmetric.')
        else:
            self.matrix = self.sym_to_sparse(matrix)
        self.n = matrix.shape[0]
        if offset == 0:
            self.diag = np.squeeze(np.asarray(ss.csr_matrix(np.diag(matrix)).todense()))
        else:
            if diag:
                self.diag = diag
            else:
                self.diag = 0
        self.k = offset

    def check_sym(self):
        '''Check if the input matrix is symmetric.'''
        return (self.matrix.T == self.matrix).all()

    def sym_to_sparse(self, matrix):
        '''Convert a symmetric matrix into a sparse matrix by removing the lower
        triangle.

        Input:
            matrix: A symmetrical 2 dimensional numpy array
            self.k: The offset value for the triangle; 1 or 0 only
        Output:
            sparse_matrix: A sparse matrix in Compressed Sparse Row format
        '''
        out = matrix[np.triu_indices_from(matrix, k=self.k)].flatten()
        sparse_matrix = ss.csr_matrix(out)
        return sparse_matrix

    def __getitem__(self, idx):
        '''Return proper values from the matrix given a tuple of indices. The
        values of the linear array will be populated from the upper triangle
        with k=1.

        e.g. A = [[0,1,2],  --->    SparseSymmetricMatrix(A) = [[1,2,2]]
                  [1,0,2],
                  [2,2,0]]

            A[2,1] = 2      --->    SparseSymmetricMatrix(A)[0,2] = 2

            This conversion can be handled by the formula:

                A[i,j] = SparseSymmetricMatrix[k]
                l = (n * (n - 1) / 2) - (n - i) * ((n - i) - 1) / 2 + j - i - 1

        Input:
            idx: Index of original matrix
            self.diag: Value of original matrix diagonal
            self.n: One of the square dimensions as matrix is n x n
            self.matrix: Sparse linear array of upper triangle values
        Output:
            Value of matrix at original index location
        '''
        i, j = self._unpack_index(idx)
        if i == j:
            if type(self.diag) == numpy.ndarray:
                return self.diag[i]
            else:
                return self.diag
        l = ((self.n * (self.n - 1)) / 2) - ((self.n -i) * (self.n - i - 1) / 2) + j
        return self.matrix[0, l]
