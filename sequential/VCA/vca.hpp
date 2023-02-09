#ifndef _VCA_SEQUENTIAL_
#define _VCA_SEQUENTIAL_

#include "../../common/interfaces/vca_interface.hpp"
#include "mkl.h"

#define EPSILON 1.0e-9

class SequentialVCA: I_VCA {
    public:
        SequentialVCA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers);
        ~SequentialVCA();
        void run(float SNR, const double* image);
        double* getEndmembers() { return endmembers; };
        void clearMemory();
};


/**
 * Calculates Moore-Penrose pseudoinverse of a square matrix
 * pinv(A) = V * S^-1 * U
 **/
inline int pinv(double* A, int n, double* S, double* U, double* VT, double* work, int lwork) {
    constexpr double alpha{1.0f}, beta{0.0f};
    // A = S U Vt
    LAPACKE_dgesvd_work(LAPACK_COL_MAJOR, 'S', 'S', n, n, A, n, S, U, n, VT, n, work, lwork);

    for (int i = 0; i < n; i++) {
        // S^-1
        double s = (S[i] > EPSILON) ? 1.0 / S[i] : S[i];
        // Vt = Vt * S^-1
        cblas_dscal(n, s, &VT[i * n], 1);
    }

    // pinv(A) = (Vt)t * Ut
    cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, n, n, n, alpha, VT, n, U, n, beta, A, n);
    return 0;
}

#endif