#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <omp.h>
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

    _queue.memset(count, 0, FPS*sizeof(unsigned int));
    // table where find the estimation by FPS
    _queue.submit([&](handler& h){
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
}

sycl::queue SYCL_VD::_get_queue(){
#if defined(INTEL_GPU)
	IntelGpuSelector selector{};
#elif defined(NVIDIA)
	CudaGpuSelector selector{};
#elif defined(CPU)	
	cpu_selector selector{};
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

    double TaoTest{0.f}, sigmaTest{0.f}, sigmaSquareTest{0.f};
    const int N{lines*samples};
    const double alpha{(double) 1/N}, beta{0};
    double superb[bands-1];

    _queue.memcpy(image, h_image, sizeof(double)*lines*samples*bands);
    _queue.wait();

    start = std::chrono::high_resolution_clock::now();

    _queue.submit([&](auto &h) {
        double* meanSpect = this->meanSpect;
        double* image = this->image;
        double* mean  = this->mean;

        h.parallel_for(sycl::range(bands), [=](auto i) {
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

    _queue.submit([&](auto &h) {
        double* meanSpect = this->meanSpect;
        double* Cov = this->Cov;
        double* Corr = this->Corr;
        h.parallel_for(sycl::range<2>(bands, bands), [=](auto index) {
            int i = index[1];
            int j = index[0];
            Corr[(i*bands) + j] = Cov[(i*bands) + j] + (meanSpect[i] * meanSpect[j]);
        });
    }).wait();

    result = count[approxVal-1];
    end = std::chrono::high_resolution_clock::now();
    tVd += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
    
    std::cout << "Result = " << result << std::endl;
    std::cout << std::endl << "SYCL VD time = " << tVd << " (s)" << std::endl;
}
