#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

#include "./vd.hpp"

constexpr oneapi::mkl::transpose trans = oneapi::mkl::transpose::trans;
constexpr oneapi::mkl::transpose nontrans = oneapi::mkl::transpose::nontrans;

SYCL_VD::SYCL_VD(int _lines, int _samples, int _bands){
    _queue     = get_queue();
    lines      = _lines;
    samples    = _samples;
    bands      = _bands;
    Cov		   = sycl::malloc_device<double>(bands*bands, _queue);
    Corr	   = sycl::malloc_device<double>(bands*bands, _queue);
    CovEigVal  = sycl::malloc_device<double>(bands, _queue);
    CorrEigVal = sycl::malloc_device<double>(bands, _queue);
    U		   = sycl::malloc_device<double>(bands*bands, _queue);
    VT	       = sycl::malloc_device<double>(bands*bands, _queue);
    count      = sycl::malloc_shared<unsigned int>(FPS, _queue);
    meanImage  = sycl::malloc_device<double>(lines*samples*bands, _queue);
    estimation = sycl::malloc_device<double>(FPS, _queue);
    mean       = sycl::malloc_device<double>(bands, _queue);
    _scrach_size = oneapi::mkl::lapack::gesvd_scratchpad_size<double>(
                    _queue, 
                    oneapi::mkl::jobsvd::novec, 
                    oneapi::mkl::jobsvd::novec, 
                    bands, bands, bands, bands, bands
                );
    _queue.wait();

    gesvd_scratchpad = sycl::malloc_device<double>(_scrach_size, _queue);

    _queue.memset(count, 0, FPS*sizeof(unsigned int));
    _queue.memset(mean, 0, bands*sizeof(double));

    // table where find the estimation by FPS
    _queue.submit([&](auto &h){
        double* estimation = this->estimation;
        h.single_task([=](){
            estimation[0] = 0.906193802436823;
            estimation[1] = 1.644976357133188;
            estimation[2] = 2.185124219133003;
            estimation[3] = 2.629741776210312;
            estimation[4] = 3.015733201402701;
        });
    }).wait();
}


SYCL_VD::~SYCL_VD() {
	if(!isQueueInit())
		return;

    if(Cov != nullptr) sycl::free(Cov, _queue);
    if(Corr != nullptr) sycl::free(Corr, _queue);
    if(CovEigVal != nullptr) sycl::free(CovEigVal, _queue);
    if(CorrEigVal != nullptr) sycl::free(CorrEigVal, _queue);
    if(U != nullptr) sycl::free(U, _queue);
    if(VT != nullptr) sycl::free(VT, _queue);
    if(estimation != nullptr) sycl::free(estimation, _queue);
    if(count != nullptr) sycl::free(count, _queue);
    if(meanImage != nullptr) sycl::free(meanImage, _queue);
    if(mean != nullptr) sycl::free(mean, _queue);
    if(gesvd_scratchpad != nullptr) sycl::free(gesvd_scratchpad, _queue);
}


void SYCL_VD::run(const int approxVal, const double* h_image) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVd{0.f};
    const unsigned int N{lines*samples};
    const double alpha{(double) 1/N}, beta{0};

    double* CovEigVal    = this->CovEigVal;
    double* CorrEigVal   = this->CorrEigVal;
    unsigned int* count  = this->count;
    double* estimation   = this->estimation;
    double* meanImage = this->meanImage;
    double* mean      = this->mean;
    double* Cov        = this->Cov;
    double* Corr       = this->Corr;
    unsigned int bands   = this->bands;
    unsigned int samples = this->samples;
    unsigned int lines   = this->lines;

    start = std::chrono::high_resolution_clock::now();

    _queue.memcpy(meanImage, h_image, sizeof(double)*lines*samples*bands);
    _queue.wait();

    _queue.parallel_for<class vd_10>(sycl::range(bands), [=](auto i) {
        for(int j = 0; j < N; j++)
            mean[i] += meanImage[(i*N) + j];
        mean[i] /= N;
        for(int j = 0; j < N; j++)
            meanImage[(i*N) + j] -= mean[i];
    }).wait();

    oneapi::mkl::blas::column_major::gemm(_queue, trans, nontrans, bands, bands, N, alpha, meanImage, N, meanImage, N, beta, Cov, bands);
    _queue.wait();

    _queue.parallel_for<class vd_20>(sycl::range<2>(bands, bands), [=](auto index) {
        int i = index[1];
        int j = index[0];
        Corr[(i*bands) + j] = Cov[(i*bands) + j] + (mean[i] * mean[j]);
    }).wait();

    // SVD
    oneapi::mkl::lapack::gesvd(_queue, oneapi::mkl::jobsvd::novec, oneapi::mkl::jobsvd::novec, bands, bands, Cov, bands, CovEigVal, U, bands, VT, bands, gesvd_scratchpad, _scrach_size);
    oneapi::mkl::lapack::gesvd(_queue, oneapi::mkl::jobsvd::novec, oneapi::mkl::jobsvd::novec, bands, bands, Corr, bands, CorrEigVal, U, bands, VT, bands, gesvd_scratchpad, _scrach_size);
    _queue.wait();

    // Estimation
    _queue.single_task<class vd_30>([=]() {
        double TaoTest{0.f}, sigmaTest{0.f}, sigmaSquareTest{0.f};
        
        for(int i = 0; i < bands; i++) {
            sigmaSquareTest = (CovEigVal[i]*CovEigVal[i] + CorrEigVal[i]*CorrEigVal[i]) * 2 / samples / lines;
            sigmaTest = sycl::sqrt(sigmaSquareTest);

            for(int j = 1; j <= FPS; j++) {
                TaoTest = M_SQRT2 * sigmaTest * estimation[j-1];

                if((CorrEigVal[i] - CovEigVal[i]) > TaoTest)
                    count[j-1]++;
            }
        }
    }).wait();

    endmembers = count[approxVal-1];
    end = std::chrono::high_resolution_clock::now();
    tVd += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
    
    std::cout << "Test = " << endmembers << std::endl;
    std::cout << std::endl << "VD took = " << tVd << " (s)" << std::endl;
}
