#ifndef _VCA_SYCL_
#define _VCA_SYCL_

#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

#include "../utils/sycl_selector.hpp"
#include "../../common/interfaces/vca_interface.hpp"

#define EPSILON 1.0e-9

class SYCL_VCA: I_VCA {
    public:
        SYCL_VCA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers);
        ~SYCL_VCA();
        void run(float SNR, const double* image);
        double* getEndmembers() { return endmembers; };
        void clearMemory();
    private:
        sycl::queue _queue;
        float* dSNR{nullptr};
        int64_t scrach_size, pinv_size;
        int64_t* imax{nullptr}; 
        double *gesvd_scratchpad{nullptr}, 
               *pinv_scratchpad{nullptr}, 
               *dImage{nullptr}, 
               *redVars{nullptr};
};


/**
 * Calculates Moore-Penrose pseudoinverse of a square matrix
 * pinv(A) = V * S^-1 * U
 **/
inline int pinv(sycl::queue q, double* A, int n, double* S, double* U, double* VT, double* work, int lwork) {
    constexpr double alpha{1.0f}, beta{0.0f};
    // A = S U Vt
    oneapi::mkl::lapack::gesvd(q, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::somevec, n, n, A, n, S, U, n, VT, n, work, lwork);
    q.wait();

    // S^-1
    q.parallel_for<class pinv_10>(sycl::range(n), [=](auto index) {
		auto i = index[0];
		if(S[i] > EPSILON)
            S[i] = 1.0 / S[i];
    }).wait();

    // Vt = Vt * S^-1
    for (int i = 0; i < n; i++) 
        oneapi::mkl::blas::column_major::scal(q, n, S[i], VT + i * n, 1);
    q.wait();

    // pinv(A) = (Vt)t * Ut
    oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::trans, n, n, n, alpha, VT, n, U, n, beta, A, n);
    q.wait();
    return 0;
}

#endif