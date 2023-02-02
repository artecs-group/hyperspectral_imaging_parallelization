#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>
#include <limits>
#include "mkl.h"

#include "vca.hpp"
#include "../../common/utils/matrix_operations.hpp"

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
    pinvA = new double[targetEndmembers * targetEndmembers]();
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
    LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, 'S', 'S', bands, bands, svdMat, bands, D, U, bands, VT, bands, &work_query, lwork);
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
    if (pinvA != nullptr) { delete[] pinvA; pinvA = nullptr; }
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
    double sum1{0}, sum2{0}, powery, powerx, mult{0}, alpha{1.0f}, beta{0.f};
    double SNR_th{15 + 10 * std::log10(targetEndmembers)};

    std::uint64_t seed{0};
    VSLStreamStatePtr rdstream;
    vslNewStream(&rdstream, VSL_BRNG_MRG32K3A, seed);

    start = std::chrono::high_resolution_clock::now();
    /***********
     * SNR estimation
     ***********/
    for (int i = 0; i < bands; i++) {
        for (int j = 0; j < N; j++)
            mean[i] += image[i * N + j];

        mean[i] /= N;

        for (int j = 0; j < N; j++)
            meanImage[i * N + j] = image[i * N + j] - mean[i];
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, bands, bands, N, alpha, meanImage, N, meanImage, N, beta, svdMat, bands);

    for (int i = 0; i < bands * bands; i++)
        svdMat[i] /= N;

    LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, 'S', 'S', bands, bands, svdMat, bands, D, U, bands, VT, bands, work, lwork);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, targetEndmembers, N, bands, alpha, VT, bands, meanImage, N, beta, x_p, N);

    for (int i = 0; i < bands * N; i++) {
        sum1 += image[i] * image[i];
        if (i < targetEndmembers * N)
            sum2 += x_p[i] * x_p[i];
        if (i < bands)
            mult += mean[i] * mean[i];
    }

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
        std::cout << "Select the projective proj." << std::endl;
#endif
        sum1 = std::numeric_limits<double>::lowest();
        for (int i = 0; i < targetEndmembers; i++) {
            for (int j = 0; j < N; j++) {
                if (i == targetEndmembers - 1)
                    x_p[i * N + j] = 0;
                u[i] += x_p[i * N + j] * x_p[i * N + j];
            }

            if (sum1 < u[i])
                sum1 = u[i];
        }
        sum1 = std::sqrt(sum1);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, bands, N, targetEndmembers, alpha, VT, targetEndmembers, x_p, targetEndmembers, beta, Rp, N);

        for (int i = 0; i < bands; i++)
            for (int j = 0; j < N; j++)
                Rp[i * N + j] += mean[i];

        for (int i = 0; i < targetEndmembers; i++)
            for (int j = 0; j < N; j++)
                y[i * N + j] = (i < targetEndmembers - 1) ? x_p[i * N + j] : sum1;
    }
    else {
#if defined(DEBUG)
        std::cout << "Select proj. to p-1" << std::endl;
#endif
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, bands, bands, N, alpha, image, N, image, N, beta, svdMat, bands);

        for (int i = 0; i < bands * bands; i++)
            svdMat[i] /= N;

        LAPACKE_dgesvd_work(LAPACK_ROW_MAJOR, 'S', 'S', bands, bands, svdMat, bands, D, U, bands, VT, bands, work, lwork);
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, targetEndmembers, N, bands, alpha, VT, bands, image, N, beta, x_p, N);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, bands, N, targetEndmembers, alpha, VT, bands, x_p, N, beta, Rp, N);

        for (int i = 0; i < targetEndmembers; i++) {
            for (int j = 0; j < N; j++)
                u[i] += x_p[i * N + j];

            u[i] /= N;

            for (int j = 0; j < N; j++)
                y[i * N + j] = x_p[i * N + j] * u[i];
        }

        for (int i = 0; i < N; i++)
            for (int j = 0; j < targetEndmembers; j++)
                sumxu[i] += y[j * N + i];


        for (int i = 0; i < targetEndmembers; i++)
            for (int j = 0; j < N; j++)
                y[i * N + j] /= sumxu[j];
    }
    /******************/

    /*******************
     * VCA algorithm
     *******************/
    A[(targetEndmembers - 1) * targetEndmembers] = 1;

    for (int i = 0; i < targetEndmembers; i++) {
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, rdstream, targetEndmembers, w, 0.0, 1.0);

        std::copy(A, A + targetEndmembers * targetEndmembers, A_copy);
        pinv(A_copy, targetEndmembers, pinvA, pinvS, pinvU, pinvVT, pinv_work, pinv_lwork);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, targetEndmembers, targetEndmembers, targetEndmembers, alpha, pinvA, targetEndmembers, A, targetEndmembers, beta, aux, targetEndmembers);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, targetEndmembers, 1, targetEndmembers, alpha, aux, targetEndmembers, w, 1, beta, f, 1);

        sum1 = 0;
        for (int j = 0; j < targetEndmembers; j++) {
            f[j] = w[j] - f[j];
            sum1 += f[j] * f[j];
        }

        for (int j = 0; j < targetEndmembers; j++)
            f[j] /= std::sqrt(sum1);

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 1, N, targetEndmembers, alpha, f, targetEndmembers, y, N, beta, sumxu, N);

        sum2 = 0;
        for (int j = 0; j < N; j++) {
            if (sumxu[j] < 0)
                sumxu[j] *= -1;
            if (sum2 < sumxu[j]) {
                sum2 = sumxu[j];
                index[i] = j;
            }
        }

        for (int j = 0; j < targetEndmembers; j++)
            A[j * targetEndmembers + i] = y[j * N + index[i]];

        for (int j = 0; j < bands; j++)
            endmembers[j * targetEndmembers + i] = Rp[j + bands * index[i]];
    }
    /******************/
    for (int j = 0; j < 10; j++)
        std::cout << endmembers[j] << ", ";
    std::cout << std::endl;

    end = std::chrono::high_resolution_clock::now();
    tVca += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
#if defined(DEBUG)
    int test = std::accumulate(endmembers, endmembers + (targetEndmembers * bands), 0);
    std::cout << "Test = " << test << std::endl;
#endif
    std::cout << std::endl << "VCA took = " << tVca << " (s)" << std::endl;
    vslDeleteStream(&rdstream);
}