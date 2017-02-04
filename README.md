# Symmetric Matrices
Storage of large symmetric matrices can pose a problem for a deployment system when trying to optimize cost. This results from the memory used to store these matrices in full as the data can be reconstructed from either the upper or lower triangle of the matrix. The purpose of this project is to explore options for storing such matrices in an efficient way that still allows for quick access to all of the elements as well as mathematical operations. 

## Sparse storage
Sparse storage is inefficient and actually results in a larger memory consumption than condensing the matrix to a 1d array. 

To do: Write class for converting sym-mat to 1d array of upper triangle only (option for including or excluding diagonal) by converting lower triangle to all zeros, flattening, and returning only the nonzero 1d array. Write custom __getitem__ method for returning the proper value for (i,j) == (j,i).
