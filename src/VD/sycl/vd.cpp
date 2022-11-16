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
    _queue     = _get_queue();
    lines      = _lines;
    samples    = _samples;
    bands      = _bands;
    meanSpect  = sycl::malloc_device<double>(bands, _queue);
    Cov		   = sycl::malloc_device<double>(bands*bands, _queue);
    Corr	   = sycl::malloc_device<double>(bands*bands, _queue);
    CovEigVal  = sycl::malloc_device<double>(bands, _queue);
    CorrEigVal = sycl::malloc_device<double>(bands, _queue);
    U		   = sycl::malloc_device<double>(bands*bands, _queue);
    VT	       = sycl::malloc_device<double>(bands*bands, _queue);
    count      = sycl::malloc_shared<unsigned int>(FPS, _queue);
    image	   = sycl::malloc_device<double>(lines*samples*bands, _queue);
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
    sycl::free(meanSpect, _queue);
    sycl::free(Cov, _queue);
    sycl::free(Corr, _queue);
    sycl::free(CovEigVal, _queue);
    sycl::free(CorrEigVal, _queue);
    sycl::free(U, _queue);
    sycl::free(VT, _queue);
    sycl::free(estimation, _queue);
    sycl::free(count, _queue);
    sycl::free(image, _queue);
    sycl::free(mean, _queue);
    sycl::free(gesvd_scratchpad, _queue);
}

sycl::queue SYCL_VD::_get_queue(){
#if defined(INTEL_GPU)
	IntelGPUSelector selector{};
#elif defined(NVIDIA_GPU)
	NvidiaGPUSelector selector{};
#elif defined(CPU)
	sycl::cpu_selector selector{};
#else
	default_selector selector{};
#endif

	sycl::queue queue{selector};
    std::cout << "Running on: "
              << queue.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    return queue;
}


void SYCL_VD::run(const int approxVal, double* h_image) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVd{0.f};

    const unsigned int N{lines*samples};
    const double alpha{(double) 1/N}, beta{0};

    _queue.memcpy(image, h_image, sizeof(double)*lines*samples*bands);
    _queue.wait();

    start = std::chrono::high_resolution_clock::now();

    _queue.submit([&](sycl::handler &h) {
        double* meanSpect = this->meanSpect;
        double* image     = this->image;
        double* mean      = this->mean;

        h.parallel_for<class image_mean>(sycl::range(bands), [=](auto i) {
            mean[i] = 0;
            for(int j = 0; j < N; j++)
                mean[i] += image[(i*N) + j];

            mean[i] /= N;
            meanSpect[i] = mean[i];

            for(int j = 0; j < N; j++)
                image[(i*N) + j] -= mean[i];
        });
    }).wait();

    oneapi::mkl::blas::column_major::gemm(_queue, trans, nontrans, bands, bands, N, alpha, image, N, image, N, beta, Cov, bands);
    _queue.wait();

    _queue.submit([&](sycl::handler &h) {
        double* meanSpect  = this->meanSpect;
        double* Cov        = this->Cov;
        double* Corr       = this->Corr;
        unsigned int bands = this->bands;
        h.parallel_for<class correlation>(sycl::range<2>(bands, bands), [=](auto index) {
            int i = index[1];
            int j = index[0];
            Corr[(i*bands) + j] = Cov[(i*bands) + j] + (meanSpect[i] * meanSpect[j]);
        });
    }).wait();

    // SVD
    oneapi::mkl::lapack::gesvd(_queue, oneapi::mkl::jobsvd::novec, oneapi::mkl::jobsvd::novec, bands, bands, Cov, bands, CovEigVal, U, bands, VT, bands, gesvd_scratchpad, _scrach_size);
    oneapi::mkl::lapack::gesvd(_queue, oneapi::mkl::jobsvd::novec, oneapi::mkl::jobsvd::novec, bands, bands, Corr, bands, CorrEigVal, U, bands, VT, bands, gesvd_scratchpad, _scrach_size);
    _queue.wait();

    // Estimation
    _queue.submit([&](sycl::handler &h) {
        double* CovEigVal    = this->CovEigVal;
        double* CorrEigVal   = this->CorrEigVal;
        unsigned int* count  = this->count;
        double* estimation   = this->estimation;
        unsigned int bands   = this->bands;
        unsigned int samples = this->samples;
        unsigned int lines   = this->lines;

        h.single_task<class foo_estimation>([=]() {
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
        });
    }).wait();

    result = count[approxVal-1];
    end = std::chrono::high_resolution_clock::now();
    tVd += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
    
    std::cout << "Result = " << result << std::endl;
    std::cout << std::endl << "SYCL VD time = " << tVd << " (s)" << std::endl;
}
