#ifndef _MATRIX_OPERATIONS_
#define _MATRIX_OPERATIONS_

#include<iostream>
#include <algorithm>
#include "mkl.h"

#define EPSILON 1.0e-9 //1.11e-16

/**
 * Calculates Moore-Penrose pseudoinverse of a square matrix
 * pinv(A) = V * S^-1 * U
 **/
inline int pinv(double* A, int n, double* pinvA, double* S, double* U, double* VT, double* work, int lwork) {
    constexpr double alpha{1.0f}, beta{0.0f};
    // A = S U Vt
    LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, 'S', 'S', n, n, A, n, S, U, n, VT, n, work, lwork);

    for (int i = 0; i < n; i++) {
        // S^-1
        double s = (S[i] > EPSILON) ? 1.0 / S[i] : S[i];
        // Vt = Vt * S^-1
        cblas_dscal(n, s, &VT[i*n], 1);
    }

    // pinv(A) = (Vt)t * Ut
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, n, n, n, alpha, VT, n, U, n, beta, pinvA, n);
    return 0;
}

#endif