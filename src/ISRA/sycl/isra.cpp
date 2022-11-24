#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

#include "./isra.hpp"

constexpr oneapi::mkl::transpose trans = oneapi::mkl::transpose::trans;
constexpr oneapi::mkl::transpose nontrans = oneapi::mkl::transpose::nontrans;

SYCL_ISRA::SYCL_ISRA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers){
    _queue           = get_queue();
    lines            = _lines;
    samples          = _samples;
    bands            = _bands;
    targetEndmembers = _targetEndmembers;

    abundanceMatrix = sycl::malloc_shared<double>(targetEndmembers * lines * samples, _queue);
    numerator       = sycl::malloc_device<double>(targetEndmembers* lines * samples, _queue);
    aux             = sycl::malloc_device<double>(lines * samples * bands, _queue);
    denominator     = sycl::malloc_device<double>(targetEndmembers* lines * samples, _queue);
    image           = sycl::malloc_device<double>(bands * lines * samples, _queue);
    endmembers      = sycl::malloc_device<double>(targetEndmembers * bands, _queue);
}


SYCL_ISRA::~SYCL_ISRA() {
    sycl::free(abundanceMatrix, _queue);
    sycl::free(numerator, _queue);
    sycl::free(denominator, _queue);
    sycl::free(aux, _queue);
    sycl::free(image, _queue);
    sycl::free(endmembers, _queue);
}


void SYCL_ISRA::run(int maxIter, const double* hImage, const double* hEndmembers) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tIsra{0.f};
    unsigned int N{lines * samples};
    double alpha{1}, beta{0};

    double* abundanceMatrix = this->abundanceMatrix;
    double* numerator = this->numerator;
    double* denominator = this->denominator;
	unsigned int targetEndmembers = this->targetEndmembers;

    start = std::chrono::high_resolution_clock::now();
    _queue.memcpy(image, hImage, sizeof(double) * N * bands);
    _queue.memcpy(endmembers, hEndmembers, sizeof(double) * targetEndmembers * bands);
    _queue.memset(abundanceMatrix, 1, sizeof(abundanceMatrix) * N*targetEndmembers);
    _queue.wait();
    
    oneapi::mkl::blas::column_major::gemm(_queue, nontrans, trans, N, targetEndmembers, bands, alpha, image, N, endmembers, targetEndmembers, beta, numerator, N);
    _queue.wait();

    for(int i = 0; i < maxIter; i++) {
        oneapi::mkl::blas::column_major::gemm(_queue, nontrans, nontrans, N, bands, targetEndmembers, alpha, abundanceMatrix, N, endmembers, targetEndmembers, beta, aux, N);
        _queue.wait();

        oneapi::mkl::blas::column_major::gemm(_queue, nontrans, trans, N, targetEndmembers, bands, alpha, aux, N, endmembers, targetEndmembers, beta, denominator, N);
        _queue.wait();

        _queue.parallel_for<class isra_10>(cl::sycl::range<1> (N * targetEndmembers), [=] (auto j){
            abundanceMatrix[j] = abundanceMatrix[j] * (numerator[j] / denominator[j]);
        }).wait();
    }

    end = std::chrono::high_resolution_clock::now();
    tIsra += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
    
    double test = std::accumulate(abundanceMatrix, abundanceMatrix + (targetEndmembers * N), 0);
    std::cout << "Test = " << test << std::endl;
    std::cout << std::endl << "SYCL ISRA time = " << tIsra << " (s)" << std::endl;
}