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
	endmembers = sycl::malloc_device<double>(targetEndmembers * bands, _queue);
	Rp         = sycl::malloc_device<double>(bands * lines * samples, _queue);
	u          = sycl::malloc_device<double>(targetEndmembers, _queue);
	sumxu      = sycl::malloc_device<double>(lines * samples, _queue);
	w          = sycl::malloc_device<double>(targetEndmembers, _queue);
	A          = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	A2         = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	aux        = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	f          = sycl::malloc_device<double>(targetEndmembers, _queue);
    index      = sycl::malloc_device<unsigned int>(targetEndmembers, _queue);

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
}


void SYCL_VCA::run(float SNR, const double* image) {

}