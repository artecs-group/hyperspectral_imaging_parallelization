#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

#include "./vca.hpp"

constexpr oneapi::mkl::transpose trans = oneapi::mkl::transpose::trans;
constexpr oneapi::mkl::transpose nontrans = oneapi::mkl::transpose::nontrans;

SYCL_VCA::SYCL_VCA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers){
	_queue  = get_queue();
    lines   = _lines;
    samples = _samples;
    bands   = _bands;
    targetEndmembers = _targetEndmembers;

    Ud         = sycl::malloc_device<double>(bands * targetEndmembers, _queue);
	x_p        = sycl::malloc_device<double>(lines * samples * targetEndmembers, _queue);
	y          = sycl::malloc_device<double>(lines * samples * targetEndmembers, _queue);
	meanImage  = sycl::malloc_device<double>(bands * lines * samples, _queue);
	mean       = sycl::malloc_device<double>(bands, _queue);
	svdMat     = sycl::malloc_device<double>(bands * bands, _queue);
	D          = sycl::malloc_device<double>(bands, _queue);//eigenvalues
	U          = sycl::malloc_device<double>(bands * bands, _queue);//eigenvectors
	VT         = sycl::malloc_device<double>(bands * bands, _queue);//eigenvectors
	endmembers = sycl::malloc_shared<double>(targetEndmembers * bands, _queue);
	Rp         = sycl::malloc_device<double>(bands * lines * samples, _queue);
	u          = sycl::malloc_device<double>(targetEndmembers, _queue);
	sumxu      = sycl::malloc_device<double>(lines * samples, _queue);
	w          = sycl::malloc_device<double>(targetEndmembers, _queue);
	A          = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	A2         = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	aux        = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	f          = sycl::malloc_device<double>(targetEndmembers, _queue);
    index      = sycl::malloc_device<unsigned int>(targetEndmembers, _queue);
	pinvS	   = sycl::malloc_device<double>(targetEndmembers, _queue);
	pinvU	   = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	pinvVT	   = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	Utranstmp  = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);

    _scrach_size = oneapi::mkl::lapack::gesvd_scratchpad_size<double>(
                    _queue, 
                    oneapi::mkl::jobsvd::somevec, 
                    oneapi::mkl::jobsvd::somevec, 
                    bands, bands, bands, bands, bands
                );
    _queue.wait();
    gesvd_scratchpad = sycl::malloc_device<double>(_scrach_size, _queue);

	_queue.memset(mean, 0, bands*sizeof(double));
	_queue.memset(u, 0, targetEndmembers*sizeof(double));
	_queue.memset(sumxu, 0, lines * samples*sizeof(double));
	_queue.wait();
}


SYCL_VCA::~SYCL_VCA(){
    sycl::free(Ud, _queue);
	sycl::free(x_p, _queue);
	sycl::free(y, _queue);
	sycl::free(meanImage, _queue);
	sycl::free(mean, _queue);
	sycl::free(svdMat, _queue);
	sycl::free(D, _queue);
	sycl::free(U, _queue);
	sycl::free(VT, _queue);
	sycl::free(endmembers, _queue);
	sycl::free(Rp, _queue);
	sycl::free(u, _queue);
	sycl::free(sumxu, _queue);
	sycl::free(w, _queue);
	sycl::free(A, _queue);
	sycl::free(A2, _queue);
	sycl::free(aux, _queue);
	sycl::free(f, _queue);
    sycl::free(index, _queue);
	sycl::free(gesvd_scratchpad, _queue);
	sycl::free(pinvS, _queue);
	sycl::free(pinvU, _queue);
	sycl::free(pinvVT, _queue);
	sycl::free(Utranstmp, _queue);
}


void SYCL_VCA::run(float SNR, const double* image) {
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVca{0.f};
    const unsigned int N{lines*samples}; 
    int info{0};
	double sum1{0}, sum2{0}, powery, powerx, mult{0}, sum1Sqrt{0}, alpha{1.0f}, beta{0.f};
    double SNR_th{15 + 10 * std::log10(targetEndmembers)};

    double* Ud = this->Ud;
	double* x_p = this->x_p;
	double* y = this->y;
	double* meanImage = this->meanImage;
	double* mean = this->mean;
	double* svdMat = this->svdMat;
	double* D = this->D;
	double* U = this->U;
	double* VT = this->VT;
	double* endmembers = this->endmembers;
	double* Rp = this->Rp;
	double* u = this->u;
	double* sumxu = this->sumxu;
	double* w = this->w;
	double* A = this->A;
	double* A2 = this->A2;
	double* aux = this->aux;
	double* f = this->f;
	double* pinvS = this->pinvS;;
	double* pinvU = this->pinvU;;
	double* pinvVT = this->pinvVT;;
	double* Utranstmp = this->Utranstmp;;
    unsigned int* index = this->index;
	unsigned int lines = this->lines;
	unsigned int samples = this->samples;
	unsigned int bands = this->bands;
	unsigned int targetEndmembers = this->targetEndmembers;

    start = std::chrono::high_resolution_clock::now();

    _queue.memcpy(meanImage, image, sizeof(double)*lines*samples*bands);
    _queue.wait();



    end = std::chrono::high_resolution_clock::now();
    tVca += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
    int result = std::accumulate(endmembers, endmembers + (targetEndmembers * bands), 0);
    std::cout << "Endmembers sum = " << result << std::endl;
    std::cout << std::endl << "SYCL VCA time = " << tVca << " (s)" << std::endl;
}