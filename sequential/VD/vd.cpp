#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include "mkl.h"

#include "vd.hpp"

SequentialVD::SequentialVD(int _lines, int _samples, int _bands) {
    lines = _lines;
    samples = _samples;
    bands = _bands;
    meanSpect = new double[bands];
    Cov = new double[bands * bands];
    Corr = new double[bands * bands];
    CovEigVal = new double[bands];
    CorrEigVal = new double[bands];
    U = new double[bands * bands];
    VT = new double[bands * bands];
    count = new unsigned int[FPS];
    estimation = new double[FPS];
    meanImage = new double[lines * samples * bands];

    // table where find the estimation by FPS
    estimation[0] = 0.906193802436823;
    estimation[1] = 1.644976357133188;
    estimation[2] = 2.185124219133003;
    estimation[3] = 2.629741776210312;
    estimation[4] = 3.015733201402701;
}


SequentialVD::~SequentialVD() {
    clearMemory();
}


void SequentialVD::clearMemory() {
    if (meanSpect != nullptr) { delete[] meanSpect; meanSpect = nullptr; }
    if (Cov != nullptr) { delete[] Cov; Cov = nullptr; }
    if (Corr != nullptr) { delete[] Corr; Corr = nullptr; }
    if (CovEigVal != nullptr) { delete[] CovEigVal; CovEigVal = nullptr; }
    if (CorrEigVal != nullptr) { delete[] CorrEigVal; CorrEigVal = nullptr; }
    if (U != nullptr) { delete[] U; U = nullptr; }
    if (VT != nullptr) { delete[] VT; VT = nullptr; }
    if (count != nullptr) { delete[] count; count = nullptr; }
    if (estimation != nullptr) { delete[] estimation; estimation = nullptr; }
    if (meanImage != nullptr) { delete[] meanImage; meanImage = nullptr; }
}


void SequentialVD::run(const int approxVal, const double* image) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVd{0.f};
    double TaoTest{0.f}, sigmaTest{0.f}, sigmaSquareTest{0.f};
    const unsigned int N{lines * samples};
    const double inv_N{1 / static_cast<double>(N)};
    const double alpha{(double)1 / N}, beta{0};
    double superb[bands - 1];

    start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < bands; i++)
        meanSpect[i] = cblas_dasum(N, &image[i*N], 1);
    
    cblas_dscal(bands, inv_N, meanSpect, 1);

    for (int i = 0; i < bands; i++) {
        for (int j = 0; j < N; j++)
            meanImage[i * N + j] = image[i * N + j] - meanSpect[i];
    }

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, bands, N, alpha, meanImage, N, meanImage, N, beta, Cov, bands);

    //correlation
    for (int j = 0; j < bands; j++)
        for (int i = 0; i < bands; i++)
            Corr[i * bands + j] = Cov[i * bands + j] + (meanSpect[i] * meanSpect[j]);

    //SVD
    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'N', bands, bands, Cov, bands, CovEigVal, U, bands, VT, bands, superb);
    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'N', bands, bands, Corr, bands, CorrEigVal, U, bands, VT, bands, superb);

    //estimation
    std::fill(count, count + FPS, 0);

    const double k = 2 / static_cast<double>(samples) / static_cast<double>(lines);
    for (int i = 0; i < bands; i++) {
        sigmaSquareTest = (CovEigVal[i] * CovEigVal[i] + CorrEigVal[i] * CorrEigVal[i]) * k;
        sigmaTest = std::sqrt(sigmaSquareTest);

        for (int j = 1; j <= FPS; j++) {
            TaoTest = M_SQRT2 * sigmaTest * estimation[j - 1];

            if ((CorrEigVal[i] - CovEigVal[i]) > TaoTest)
                count[j - 1]++;
        }
    }

    endmembers = count[approxVal - 1];
    end = std::chrono::high_resolution_clock::now();
    tVd += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
#if defined(DEBUG)
    std::cout << "Test = " << endmembers << std::endl;
#endif
    std::cout << std::endl << "VD took = " << tVd << " (s)" << std::endl;
}