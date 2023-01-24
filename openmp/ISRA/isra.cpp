#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <omp.h>

#if defined(NVIDIA_GPU)
#include "cublas.h"
#else
#include "mkl.h"
#include "mkl_omp_offload.h"
#endif

#include "./isra.hpp"

OpenMP_ISRA::OpenMP_ISRA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers){
    lines            = _lines;
    samples          = _samples;
    bands            = _bands;
    targetEndmembers = _targetEndmembers;

    abundanceMatrix = new double[targetEndmembers * lines * samples]();
    numerator       = new double[targetEndmembers * lines * samples]();
    aux             = new double[lines * samples * bands]();
    denominator     = new double[targetEndmembers * lines * samples]();
}


OpenMP_ISRA::~OpenMP_ISRA() {
    if(abundanceMatrix != nullptr) {delete[] abundanceMatrix; abundanceMatrix = nullptr; }
    clearMemory();
}


void OpenMP_ISRA::clearMemory() {
    if(numerator != nullptr) {delete[] numerator; numerator = nullptr; }
    if(denominator != nullptr) {delete[] denominator; denominator = nullptr; }
    if(aux != nullptr) {delete[] aux; aux = nullptr; }
}


void OpenMP_ISRA::runOnCPU(int maxIter, const double* image, const double* endmembers) {
    unsigned int N{lines * samples};
    double alpha{1}, beta{0};
    
    #pragma omp target teams distribute parallel for simd
    for(int i = 0; i < N*targetEndmembers; i++)
        abundanceMatrix[i] = 1;

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, targetEndmembers, bands, alpha, image, N, endmembers, targetEndmembers, beta, numerator, N);
    
    for(int i = 0; i < maxIter; i++) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, bands, targetEndmembers, alpha, abundanceMatrix, N, endmembers, targetEndmembers, beta, aux, N);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, targetEndmembers, bands, alpha, aux, N, endmembers, targetEndmembers, beta, denominator, N);

        #pragma omp teams distribute parallel for simd
        for(int j = 0; j < N * targetEndmembers; j++)
            abundanceMatrix[j] = abundanceMatrix[j] * (numerator[j] / denominator[j]);
    }
}


void OpenMP_ISRA::runOnGPU(int maxIter, const double* image, const double* endmembers) {
    unsigned int N{lines * samples};
    double alpha{1}, beta{0};
    const int default_dev = omp_get_default_device();

    double* abundanceMatrix = this->abundanceMatrix;
    double* numerator = this->numerator;
    double* denominator = this->denominator;
    double* aux = this->aux;
    unsigned int lines = this->lines;
	unsigned int samples = this->samples;
	unsigned int bands = this->bands;
	unsigned int targetEndmembers = this->targetEndmembers;
    
    #pragma omp target enter data \
    map(to: image[0:bands*N], endmembers[0:targetEndmembers*bands])\
    map(alloc: abundanceMatrix[0:targetEndmembers*N], numerator[0:targetEndmembers*N], \
    denominator[0:targetEndmembers*N], aux[0:bands*N]) device(default_dev)
    {
        #pragma omp target teams distribute parallel for
        for(int i = 0; i < N*targetEndmembers; i++)
            abundanceMatrix[i] = 1;

        #pragma omp target variant dispatch use_device_ptr(image, endmembers, numerator)
        {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, targetEndmembers, bands, alpha, image, N, endmembers, targetEndmembers, beta, numerator, N);
        }
        
        for(int i = 0; i < maxIter; i++) {
            #pragma omp target variant dispatch use_device_ptr(abundanceMatrix, endmembers, aux)
            {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, bands, targetEndmembers, alpha, abundanceMatrix, N, endmembers, targetEndmembers, beta, aux, N);
            }

            #pragma omp target variant dispatch use_device_ptr(endmembers, aux, denominator)
            {
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, N, targetEndmembers, bands, alpha, aux, N, endmembers, targetEndmembers, beta, denominator, N);
            }

            #pragma omp target teams distribute parallel for
            for(int j = 0; j < N * targetEndmembers; j++)
                abundanceMatrix[j] = abundanceMatrix[j] * (numerator[j] / denominator[j]);
        }
    }
    #pragma omp target exit data map(from: abundanceMatrix[0:targetEndmembers*N]) device(default_dev)
}


void OpenMP_ISRA::run(int maxIter, const double* image, const double* endmembers) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tIsra{0.f};

    start = std::chrono::high_resolution_clock::now();
#if defined(GPU)
    runOnGPU(maxIter, image, endmembers);
#else
    runOnCPU(maxIter, image, endmembers);
#endif
    end = std::chrono::high_resolution_clock::now();
    tIsra += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
    
#if defined(DEBUG)
    int ntest = (targetEndmembers * lines * samples) < 10 ? targetEndmembers * lines * samples : 10;
    std::cout << "Abundance matrix first " << ntest << " elements: " << std::endl << "      -> ";
    for (size_t i = 0; i < ntest; i++)
        std::cout << abundanceMatrix[i] << ", ";
    std::cout << std::endl;
#endif
    std::cout << std::endl << "ISRA took = " << tIsra << " (s)" << std::endl;
}