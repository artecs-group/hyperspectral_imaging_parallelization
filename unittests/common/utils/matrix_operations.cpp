#include <gtest/gtest.h>
#include <cmath>
#include "mkl.h"
#include "../../../common/utils/matrix_operations.hpp"

/**
 * This test checks the property that states: A = A * pinv(A) *A
*/
TEST(SquarePinvTest, CommonCase) {
    constexpr int n = 3;
    double A[n*n] = {2,1,3,2,3,2,3,2,1};
    double A2[n*n] = {2,1,3,2,3,2,3,2,1};
    double pinvA[n*n] = {0};
    double S[n] = {0};
    double U[n*n] = {0};
    double VT[n*n] = {0};
    double I[n*n] = {0};

    int lwork = -1;
    double work_query;

    LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, 'S', 'S', n, n, A2, n, S, U, n, VT, n, &work_query, lwork);
    lwork = (int) work_query;
    double *work = new double[lwork];

    pinv(A2, n, pinvA, S, U, VT, work, lwork);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, A, n, pinvA, n, 0.0, I, n);
    
    constexpr double eps{1e-9};
    bool is_correct{true};
    double alpha{0.0f};
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            alpha = (i == j) ? 1.0 : 0.0;
            is_correct &= std::fabs(I[i * n + j] - alpha) < eps;
        }
    }

    delete[] work;
    ASSERT_EQ(is_correct, true);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}