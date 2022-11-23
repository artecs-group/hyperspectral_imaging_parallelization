#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include "mkl.h"

#include "./isra.hpp"

SequentialISRA::SequentialISRA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers){
    lines            = _lines;
    samples          = _samples;
    bands            = _bands;
    targetEndmembers = _targetEndmembers;

    abundanceMatrix = new double[targetEndmembers * lines * samples]();
    numerator       = new double[targetEndmembers * lines * samples]();
    aux             = new double[lines * samples * bands]();
    denominator     = new double[targetEndmembers * lines * samples]();
}


SequentialISRA::~SequentialISRA() {
    delete[] abundanceMatrix;
    delete[] numerator;
    delete[] denominator;
    delete[] aux;
}


void SequentialISRA::run(int maxIter, const double* image, const double* endmembers) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tIsra{0.f};
    unsigned int N{lines * samples};
    double alpha{1}, beta{0};

    start = std::chrono::high_resolution_clock::now();
    
    std::fill(abundanceMatrix, abundanceMatrix + (N*targetEndmembers), 1);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, targetEndmembers, bands, alpha, image, N, endmembers, targetEndmembers, beta, numerator, N);
    
    for(int i = 0; i < maxIter; i++) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, bands, targetEndmembers, alpha, abundanceMatrix, N, endmembers, targetEndmembers, beta, aux, N);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, targetEndmembers, bands, alpha, aux, N, endmembers, targetEndmembers, beta, denominator, N);

        for(int j = 0; j < N * targetEndmembers; j++)
            abundanceMatrix[j] = abundanceMatrix[j] * (numerator[j] / denominator[j]);
    }

    end = std::chrono::high_resolution_clock::now();
    tIsra += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
    
    double result = std::accumulate(abundanceMatrix, abundanceMatrix + (targetEndmembers * N), 0);
    std::cout << "Sum abundance = " << result << std::endl;
    std::cout << std::endl << "Sequential ISRA time = " << tIsra << " (s)" << std::endl;
}