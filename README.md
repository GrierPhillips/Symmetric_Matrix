# Symmetric Matrices
Storage of large symmetric matrices can pose a problem for a deployment system when trying to optimize cost. This results from the memory used to store these matrices in full as the data can be reconstructed from either the upper or lower triangle of the matrix. The purpose of this project is to explore options for storing such matrices in an efficient way that still allows for quick access to all of the elements as well as mathematical operations. 

## Sparse storage
Sparse storage is inefficient and actually results in a larger memory consumption than condensing the matrix to a 1d array. 

The ss_matrix.py module contains a class that will convert any size symmetric matrix into a linear representation. This simple implementation utilizes at most 50% of the memory required for the original matrix and can still be indexed the same as the original matrix. 

To do: Properly implement all methods required for mimicking an subclass of Scipy.sparse.compressed._cs_matrix.
