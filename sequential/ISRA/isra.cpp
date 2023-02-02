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
    aux = new double[lines * samples * bands]();
    denominator = new double[lines * samples * targetEndmembers]();
}


SequentialISRA::~SequentialISRA() {
    if (abundanceMatrix != nullptr) { delete[] abundanceMatrix; abundanceMatrix = nullptr; }
    clearMemory();
}

void SequentialISRA::clearMemory() {
    if (numerator != nullptr) { delete[] numerator; numerator = nullptr; }
    if (denominator != nullptr) { delete[] denominator; denominator = nullptr; }
    if (aux != nullptr) { delete[] aux; aux = nullptr; }
}


void SequentialISRA::run(int maxIter, const double* image, const double* endmembers) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tIsra{0.f};
    unsigned int N{lines * samples};
    double alpha{1}, beta{0};

    start = std::chrono::high_resolution_clock::now();

    std::fill(abundanceMatrix, abundanceMatrix + (N * targetEndmembers), 1);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, N, targetEndmembers, bands, alpha, image, N, endmembers, bands, beta, numerator, targetEndmembers);

    for (int i = 0; i < maxIter; i++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, bands, targetEndmembers, alpha, abundanceMatrix, targetEndmembers, endmembers, bands, beta, aux, bands);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, targetEndmembers, bands, alpha, aux, bands, endmembers, bands, beta, denominator, targetEndmembers);

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