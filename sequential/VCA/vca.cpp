#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>
#include <limits>
#include "mkl.h"

#include "vca.hpp"

SequentialVCA::SequentialVCA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers) {
    lines = _lines;
    samples = _samples;
    bands = _bands;
    targetEndmembers = _targetEndmembers;

    Ud = new double[bands * targetEndmembers]();
    x_p = new double[lines * samples * targetEndmembers]();
    y = new double[lines * samples * targetEndmembers]();
    meanImage = new double[bands * lines * samples]();
    mean = new double[bands]();
    svdMat = new double[bands * bands]();
    D = new double[bands]();//eigenvalues
    U = new double[bands * bands]();//eigenvectors
    VT = new double[bands * bands]();//eigenvectors
    endmembers = new double[targetEndmembers * bands]();
    Rp = new double[bands * lines * samples]();
    u = new double[targetEndmembers]();
    sumxu = new double[lines * samples]();
    w = new double[targetEndmembers]();
    A = new double[targetEndmembers * targetEndmembers]();
    A_copy = new double[targetEndmembers * targetEndmembers]();
    aux = new double[targetEndmembers * targetEndmembers]();
    f = new double[targetEndmembers]();
    index = new unsigned int[targetEndmembers]();
    pinvS = new double[targetEndmembers]();
    pinvU = new double[targetEndmembers * targetEndmembers]();
    pinvVT = new double[targetEndmembers * targetEndmembers]();

    double work_query;
    pinv_lwork = -1;
    LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, 'S', 'S', targetEndmembers, targetEndmembers, A_copy, targetEndmembers, pinvS, pinvU, targetEndmembers, pinvVT, targetEndmembers, &work_query, pinv_lwork);
    pinv_lwork = (int)work_query;
    pinv_work = new double[pinv_lwork]();

    lwork = -1;
    LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, 'S', 'N', bands, bands, svdMat, bands, D, U, bands, VT, bands, &work_query, lwork);
    lwork = (int)work_query;
    work = new double[lwork]();
}


SequentialVCA::~SequentialVCA() {
    if (endmembers != nullptr) { delete[] endmembers; endmembers = nullptr; }
    if (Rp != nullptr) { delete[] Rp; Rp = nullptr; }
    clearMemory();
}


void SequentialVCA::clearMemory() {
    if (Ud != nullptr) { delete[] Ud; Ud = nullptr; }
    if (x_p != nullptr) { delete[] x_p; x_p = nullptr; }
    if (y != nullptr) { delete[] y; y = nullptr; }
    if (meanImage != nullptr) { delete[] meanImage; meanImage = nullptr; }
    if (mean != nullptr) { delete[] mean; mean = nullptr; }
    if (svdMat != nullptr) { delete[] svdMat; svdMat = nullptr; }
    if (D != nullptr) { delete[] D; D = nullptr; }
    if (U != nullptr) { delete[] U; U = nullptr; }
    if (VT != nullptr) { delete[] VT; VT = nullptr; }
    if (u != nullptr) { delete[] u; u = nullptr; }
    if (sumxu != nullptr) { delete[] sumxu; sumxu = nullptr; }
    if (w != nullptr) { delete[] w; w = nullptr; }
    if (A != nullptr) { delete[] A; A = nullptr; }
    if (A_copy != nullptr) { delete[] A_copy; A_copy = nullptr; }
    if (aux != nullptr) { delete[] aux; aux = nullptr; }
    if (f != nullptr) { delete[] f; f = nullptr; }
    if (index != nullptr) { delete[] index; index = nullptr; }
    if (pinvS != nullptr) { delete[] pinvS; pinvS = nullptr; }
    if (pinvU != nullptr) { delete[] pinvU; pinvU = nullptr; }
    if (pinvVT != nullptr) { delete[] pinvVT; pinvVT = nullptr; }
    if (pinv_work != nullptr) { delete[] pinv_work; pinv_work = nullptr; }
    if (work != nullptr) { delete[] work; work = nullptr; }
}


void SequentialVCA::run(float SNR, const double* image) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVca{0.f};
    const unsigned int N{lines * samples};
    const double inv_N{1/static_cast<double>(N)};
    int imax{0};
    double sum1{0}, sum2{0}, powery, powerx, mult{0}, alpha{1.0f}, beta{0.f};
    double SNR_th{15 + 10 * std::log10(targetEndmembers)};

    std::uint64_t seed{0};
    VSLStreamStatePtr rdstream;
    vslNewStream(&rdstream, VSL_BRNG_MRG32K3A, seed);

    start = std::chrono::high_resolution_clock::now();
    /***********
     * SNR estimation
     ***********/
    for (size_t i = 0; i < bands; i++)
        mean[i] = cblas_dasum(N, &image[i*N], 1);
    
    cblas_dscal(bands, inv_N, mean, 1);

    for (int i = 0; i < bands; i++) {
        for (int j = 0; j < N; j++)
            meanImage[i * N + j] = image[i * N + j] - mean[i];
    }

    // svdMat[bands, bands] = meanImg[bands, N] * transpose(meanImg[bands, N])
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, bands, N, alpha, meanImage, N, meanImage, N, beta, svdMat, bands);

    cblas_dscal(bands*bands, inv_N, svdMat, 1);

    LAPACKE_dgesvd_work(LAPACK_COL_MAJOR, 'S', 'N', bands, bands, svdMat, bands, D, U, bands, VT, bands, work, lwork);
    
    // x_p[target, N] = transpose(U[bands, bands]) * meanImg[bands, N]
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, targetEndmembers, N, bands, alpha, U, bands, meanImage, N, beta, x_p, targetEndmembers);

	sum1 = cblas_ddot(bands*N, image, 1, image, 1);
	sum2 = cblas_ddot(N*targetEndmembers, x_p, 1, x_p, 1);
	mult = cblas_ddot(bands, mean, 1, mean, 1);

    powery = sum1 / N;
    powerx = sum2 / N + mult;

    SNR = (SNR < 0) ? 10 * std::log10((powerx - targetEndmembers / bands * powery) / (powery - powerx)) : SNR;
    /**********************/

#if defined(DEBUG)
    std::cout << "SNR    = " << SNR << std::endl
              << "SNR_th = " << SNR_th << std::endl;
#endif

    /***************
     * Choosing Projective Projection or projection to p-1 subspace
     ***************/
    if (SNR < SNR_th) {
#if defined(DEBUG)
        std::cout << "Select proj. to p-1" << std::endl;
#endif
        for (size_t i = 0; i < bands; i++)
            for (size_t j = targetEndmembers-1; j < bands; j++)
                U[i*bands + j] = 0;

        for (int j = 0; j < N; j++)
            x_p[(targetEndmembers-1) * N + j] = 0;

        // for(int i{0}; i < targetEndmembers; i++)
        //     u[i] = cblas_ddot(N, &x_p[i*N], 1, &x_p[i*N], 1);

        for (int i = 0; i < targetEndmembers; i++) {
            for (int j = 0; j < N; j++) {
                u[i] += x_p[i * N + j] * x_p[i * N + j];
            }
        }
        imax = cblas_idamax(targetEndmembers, u, 1);
        sum1 = std::sqrt(u[imax]);

        // Rp[bands, N] = U[bands, bands] * x_p[target, N]
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, N, targetEndmembers, alpha, U, bands, x_p, targetEndmembers, beta, Rp, bands);

        for (int i = 0; i < bands; i++)
            for (int j = 0; j < N; j++)
                Rp[i * N + j] += mean[i];

        for (int i = 0; i < targetEndmembers; i++)
            for (int j = 0; j < N; j++)
                y[i * N + j] = (i < targetEndmembers - 1) ? x_p[i * N + j] : sum1;
    }
    else {
#if defined(DEBUG)
        std::cout << "Select the projective proj." << std::endl;
#endif
        // svdMat[bands, bands] = image[bands, N] * traspose(image[bands, N])
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, bands, N, alpha, image, N, image, N, beta, svdMat, bands);
        cblas_dscal(bands*bands, inv_N, svdMat, 1);

        LAPACKE_dgesvd_work(LAPACK_COL_MAJOR, 'S', 'N', bands, bands, svdMat, bands, D, U, bands, VT, bands, work, lwork);
        
        // x_p[target, N] = traspose(U[bands, bands]) * image[bands, N]
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, targetEndmembers, N, bands, alpha, U, bands, image, N, beta, x_p, targetEndmembers);
        
        // Rp[bands, N] = U[bands, bands] * x_p[target, N]
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, N, targetEndmembers, alpha, U, bands, x_p, targetEndmembers, beta, Rp, bands);

        for (size_t i = 0; i < targetEndmembers; i++) {
            for(int j{0}; j < N; j++)
                u[i] += x_p[i*N + j];
            u[i] *= inv_N;
        }

        for (int i = 0; i < targetEndmembers; i++) {
            for (int j = 0; j < N; j++)
                y[i * N + j] = x_p[i * N + j] * u[i];
        }

        for (int i = 0; i < N; i++)
            for (int j = 0; j < targetEndmembers; j++)
                sumxu[i] += y[j * N + i];


        for (int i = 0; i < targetEndmembers; i++)
            for (int j = 0; j < N; j++)
                y[i * N + j] = x_p[i * N + j] / sumxu[j];
    }
    /******************/

    /*******************
     * VCA algorithm
     *******************/
    A[(targetEndmembers - 1) * targetEndmembers] = 1;

    for (int i = 0; i < targetEndmembers; i++) {
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, rdstream, targetEndmembers, w, 0.0, 1.0);

        std::copy(A, A + targetEndmembers * targetEndmembers, A_copy);
        pinv(A_copy, targetEndmembers, pinvS, pinvU, pinvVT, pinv_work, pinv_lwork);

        //aux[target, target] = A[target, target] * pinvA[target, target]
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, targetEndmembers, targetEndmembers, targetEndmembers, alpha, A, targetEndmembers, A_copy, targetEndmembers, beta, aux, targetEndmembers);
        
        //f[target, 1] = aux[target, target] * w[target, 1]
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, targetEndmembers, 1, targetEndmembers, alpha, aux, targetEndmembers, w, targetEndmembers, beta, f, targetEndmembers);

        cblas_daxpy(targetEndmembers, -1.0f, w, 1, f, 1);
        //mkl_domatadd('C', 'N', 'N', 1, targetEndmembers, 1.0f, w, 1, -1.0f, f, 1, f, 1);
        sum1 = cblas_ddot(targetEndmembers, f, 1, f, 1);
        sum1 = std::sqrt(sum1);

        for(int j{0}; j < targetEndmembers; j++)
            f[j] /= sum1;

        // sumxu[1, N] = f[1, target] * y[target, N]
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, 1, N, targetEndmembers, alpha, f, 1, y, N, beta, sumxu, 1);

        for (int j = 0; j < N; j++)
            sumxu[j] = std::abs(sumxu[j]);

        index[i] = cblas_idamax(N, sumxu, 1);
        
        for (int j = 0; j < targetEndmembers; j++)
            A[j * targetEndmembers + i] = y[j * N + index[i]];
    }

    for (size_t i = 0; i < targetEndmembers; i++) {
        for (int j = 0; j < bands; j++)
            endmembers[j*targetEndmembers + i] = Rp[j * N + index[i]];
    }
    
    /******************/

    end = std::chrono::high_resolution_clock::now();
    tVca += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
#if defined(DEBUG)
    int test = std::accumulate(endmembers, endmembers + (targetEndmembers * bands), 0);
    std::cout << "Test = " << test << std::endl;
#endif
    std::cout << std::endl << "VCA took = " << tVca << " (s)" << std::endl;
    vslDeleteStream(&rdstream);
}