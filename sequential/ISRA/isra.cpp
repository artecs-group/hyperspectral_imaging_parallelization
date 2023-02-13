#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include "mkl.h"

#include "./isra.hpp"

SequentialISRA::SequentialISRA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers) {
    lines = _lines;
    samples = _samples;
    bands = _bands;
    targetEndmembers = _targetEndmembers;

    abundanceMatrix = new double[targetEndmembers * lines * samples]();
    numerator = new double[lines * samples * targetEndmembers]();
    aux = new double[targetEndmembers * targetEndmembers]();
    denominator = new double[lines * samples * targetEndmembers]();
    Et_E = new double[targetEndmembers*targetEndmembers];
    comput = new double[targetEndmembers*bands];
    ipiv = new lapack_int[targetEndmembers];
}


SequentialISRA::~SequentialISRA() {
    if (abundanceMatrix != nullptr) { delete[] abundanceMatrix; abundanceMatrix = nullptr; }
    clearMemory();
}

void SequentialISRA::clearMemory() {
    if (numerator != nullptr) { delete[] numerator; numerator = nullptr; }
    if (denominator != nullptr) { delete[] denominator; denominator = nullptr; }
    if (aux != nullptr) { delete[] aux; aux = nullptr; }
    if (Et_E != nullptr) { delete[] Et_E; Et_E = nullptr; }
    if (comput != nullptr) { delete[] comput; comput = nullptr; }
    if (ipiv != nullptr) { delete[] ipiv; ipiv = nullptr; }
}


void SequentialISRA::preProcessAbundance(const double* image, double* Ab,  const double* e, int targetEndmembers, int lines, int samples, int bands) {
	double alpha{1.0}, beta{0.0};

    // Et_E[target * target] = e[bands * target] * e[bands * target]
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, targetEndmembers, targetEndmembers, bands, alpha, e, targetEndmembers, e, targetEndmembers, beta, Et_E, targetEndmembers);
	invTR(Et_E, targetEndmembers);

    //comput[target * bands] = Et_E[target * target] * e[bands * target]
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, targetEndmembers, bands, targetEndmembers, alpha, Et_E, targetEndmembers, e, targetEndmembers, beta, comput, targetEndmembers);

    // Ab[N * target] = image[bands * N] * comput[target * bands]
	const int N = lines*samples;
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, targetEndmembers, bands, alpha, image, N, comput, targetEndmembers, beta, Ab, N);
	
    // remove negatives
	for (int i = 0; i < N*targetEndmembers; i++)
        Ab[i] = (Ab[i] < 0.0) ? 0.00001 : Ab[i];
}


void SequentialISRA::invTR(double* A, int targetEndmembers) {
    LAPACKE_dgetrf(LAPACK_COL_MAJOR, targetEndmembers, targetEndmembers, A, targetEndmembers, ipiv);
    LAPACKE_dgetri(LAPACK_COL_MAJOR, targetEndmembers, A, targetEndmembers, ipiv);
}


void SequentialISRA::run(int maxIter, const double* image, const double* endmembers) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tIsra{0.f};
    unsigned int N{lines * samples};
    double alpha{1}, beta{0};

    start = std::chrono::high_resolution_clock::now();

    preProcessAbundance(image, abundanceMatrix,  endmembers, targetEndmembers, lines, samples, bands);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, targetEndmembers, bands, alpha, image, N, endmembers, targetEndmembers, beta, numerator, N);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, targetEndmembers, targetEndmembers, bands, alpha, endmembers, targetEndmembers, endmembers, targetEndmembers, beta, aux, targetEndmembers);

    for (int i = 0; i < 1; i++) {       
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, targetEndmembers, targetEndmembers, alpha, abundanceMatrix, N, aux, targetEndmembers, beta, denominator, N);

        for (int j = 0; j < N * targetEndmembers; j++)
            abundanceMatrix[j] = abundanceMatrix[j] * (numerator[j] / denominator[j]);
    }

    end = std::chrono::high_resolution_clock::now();
    tIsra += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();

#if defined(DEBUG)
    int ntest = (targetEndmembers * N) < 10 ? targetEndmembers * N : 10;
    std::cout << "Abundance matrix first " << ntest << " elements: " << std::endl << "      -> ";
    for (size_t i = 0; i < ntest; i++)
        std::cout << abundanceMatrix[i] << ", ";
    std::cout << std::endl;
#endif
    std::cout << std::endl << "ISRA took = " << tIsra << " (s)" << std::endl;
}