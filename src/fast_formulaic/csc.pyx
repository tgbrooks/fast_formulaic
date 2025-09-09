# distutils: language = c++
# # cython: profile=True
# # cython: linetrace=True
# # distutils: define_macros=CYTHON_TRACE=1
# # cython: binding=True
import numpy as np
cimport numpy as cnp
cimport cython
from scipy.sparse import csc_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
def csc_column_product(A,B):
    # Method that works for canonical CSC matrices
    # Assumes A and B have compatible shape (same number of rows)
    # pairwise products of all columns

    cdef:
        cnp.int32_t nnz = 0
        cnp.int32_t A_pos, B_pos, A_end, B_end
        cnp.float64_t[:] C_data = np.empty(min(A.nnz * B.shape[1], B.nnz * A.shape[1]), np.float64)
        cnp.int32_t[:] C_indices = np.empty(min(A.nnz * B.shape[1], B.nnz * A.shape[1]), np.int32)
        cnp.int32_t[:] C_indptr = np.empty(A.shape[1] * B.shape[1]+1, np.int32)

        cnp.int32_t[:] A_indptr = A.indptr
        cnp.int32_t[:] A_indices = A.indices  
        cnp.float64_t[:] A_data = A.data
        cnp.int32_t[:] B_indptr = B.indptr
        cnp.int32_t[:] B_indices = B.indices
        cnp.float64_t[:] B_data = B.data

        cnp.int32_t n_row = A.shape[0]
        cnp.int32_t n_col_A = A.shape[1]
        cnp.int32_t n_col_B = B.shape[1]
        cnp.int32_t i, k, A_j, B_j
        cnp.float64_t result
        cnp.int32_t C_col = 0

    C_indptr[0] = 0

    for k in range(n_col_B):
        for i in range(n_col_A):
            A_pos = A_indptr[i]
            B_pos = B_indptr[k]
            B_end = B_indptr[k+1]
            A_end = A_indptr[i+1]

            #while not finished with either row
            while (A_pos < A_end and B_pos < B_end):
                A_j = A_indices[A_pos]
                B_j = B_indices[B_pos]

                if(A_j == B_j):
                    C_data[nnz] = A_data[A_pos] * B_data[B_pos]
                    C_indices[nnz] = A_j
                    nnz += 1
                    A_pos += 1
                    B_pos += 1
                elif (A_j < B_j):
                    A_pos += 1
                else:
                    B_pos += 1
            C_col += 1
            C_indptr[C_col] = nnz


    return csc_matrix((C_data[:nnz], C_indices[:nnz], C_indptr), shape=(A.shape[0], A.shape[1] * B.shape[1]))

