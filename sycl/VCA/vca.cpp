#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

#include "vca.hpp"

constexpr oneapi::mkl::transpose trans = oneapi::mkl::transpose::trans;
constexpr oneapi::mkl::transpose nontrans = oneapi::mkl::transpose::nontrans;

SYCL_VCA::SYCL_VCA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers){
	_queue  = get_queue();
    lines   = _lines;
    samples = _samples;
    bands   = _bands;
    targetEndmembers = _targetEndmembers;

	x_p        = sycl::malloc_device<double>(lines * samples * targetEndmembers, _queue);
	y          = sycl::malloc_device<double>(lines * samples * targetEndmembers, _queue);
	dImage     = sycl::malloc_device<double>(bands * lines * samples, _queue);
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
	A_copy     = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	aux        = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	f          = sycl::malloc_device<double>(targetEndmembers, _queue);
	pinvS	   = sycl::malloc_shared<double>(targetEndmembers, _queue);
	pinvU	   = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	pinvVT	   = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	redVars    = sycl::malloc_shared<double>(3, _queue);
	imax       = sycl::malloc_device<int64_t>(1, _queue);

    scrach_size = oneapi::mkl::lapack::gesvd_scratchpad_size<double>(
                    _queue, 
                    oneapi::mkl::jobsvd::somevec, 
                    oneapi::mkl::jobsvd::novec, 
                    bands, bands, bands, bands, bands
                );
	pinv_size   = oneapi::mkl::lapack::gesvd_scratchpad_size<double>(
					_queue, 
					oneapi::mkl::jobsvd::somevec, 
					oneapi::mkl::jobsvd::somevec, 
					targetEndmembers, targetEndmembers, targetEndmembers, targetEndmembers, targetEndmembers
				);
    _queue.wait();
    gesvd_scratchpad = sycl::malloc_device<double>(scrach_size, _queue);
	pinv_scratchpad  = sycl::malloc_device<double>(pinv_size, _queue);

	_queue.memset(mean, 0, bands*sizeof(double));
	_queue.memset(u, 0, targetEndmembers*sizeof(double));
	_queue.memset(sumxu, 0, lines * samples*sizeof(double));
	_queue.memset(A, 0, targetEndmembers * targetEndmembers*sizeof(double));
	_queue.memset(redVars, 0, 3*sizeof(double));

	_queue.submit([&](cl::sycl::handler &h) {
		double* A = this->A;
		unsigned int targetEndmembers = this->targetEndmembers;
		h.single_task<class vca_10>([=]() {
			A[(targetEndmembers-1) * targetEndmembers] = 1;
		});
	});
	_queue.wait();
}


SYCL_VCA::~SYCL_VCA(){
	if(!isQueueInit())
		return;

	if(endmembers != nullptr) {sycl::free(endmembers, _queue); endmembers = nullptr;}
	if(Rp != nullptr) {sycl::free(Rp, _queue); Rp = nullptr;}
	clearMemory();
}

void SYCL_VCA::clearMemory() {
	if(!isQueueInit())
		return;
	
	if(x_p != nullptr) {sycl::free(x_p, _queue); x_p = nullptr;}
	if(y != nullptr) {sycl::free(y, _queue); y = nullptr;}
	if(dImage != nullptr) {sycl::free(dImage, _queue); dImage = nullptr;}
	if(meanImage != nullptr) {sycl::free(meanImage, _queue); meanImage = nullptr;}
	if(mean != nullptr) {sycl::free(mean, _queue); mean = nullptr;}
	if(svdMat != nullptr) {sycl::free(svdMat, _queue); svdMat = nullptr;}
	if(D != nullptr) {sycl::free(D, _queue); D = nullptr;}
	if(U != nullptr) {sycl::free(U, _queue); U = nullptr;}
	if(VT != nullptr) {sycl::free(VT, _queue); VT = nullptr;}
	if(u != nullptr) {sycl::free(u, _queue); u = nullptr;}
	if(sumxu != nullptr) {sycl::free(sumxu, _queue); sumxu = nullptr;}
	if(w != nullptr) {sycl::free(w, _queue); w = nullptr;}
	if(A != nullptr) {sycl::free(A, _queue); A = nullptr;}
	if(A_copy != nullptr) {sycl::free(A_copy, _queue); A_copy = nullptr;}
	if(aux != nullptr) {sycl::free(aux, _queue); aux = nullptr;}
	if(f != nullptr) {sycl::free(f, _queue); f = nullptr;}
	if(gesvd_scratchpad != nullptr) {sycl::free(gesvd_scratchpad, _queue); gesvd_scratchpad = nullptr;}
	if(pinv_scratchpad != nullptr) {sycl::free(pinv_scratchpad, _queue); pinv_scratchpad = nullptr;}
	if(pinvS != nullptr) {sycl::free(pinvS, _queue); pinvS = nullptr;}
	if(pinvU != nullptr) {sycl::free(pinvU, _queue); pinvU = nullptr;}
	if(pinvVT != nullptr) {sycl::free(pinvVT, _queue); pinvVT = nullptr;}
	if(redVars != nullptr) {sycl::free(redVars, _queue); redVars = nullptr;}
	if(imax != nullptr) {sycl::free(imax, _queue); imax = nullptr;}
}


void SYCL_VCA::run(float SNR, const double* image) {
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVca{0.f};
    unsigned int N{lines*samples};
	double inv_N{1/static_cast<double>(N)};
	double alpha{1.0f}, beta{0.f}, powerx{0}, powery{0};
    const double SNR_th{15 + 10 * std::log10(targetEndmembers)};

	std::uint64_t seed{0};
	oneapi::mkl::rng::mrg32k3a engine(_queue, seed);
	oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::box_muller2> distr(0.0, 1.0);

	double* x_p = this->x_p;
	double* y = this->y;
	double* dImage = this->dImage;
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
	double* A_copy = this->A_copy;
	double* aux = this->aux;
	double* f = this->f;
	double* pinvS = this->pinvS;
	double* pinvU = this->pinvU;
	double* pinvVT = this->pinvVT;
	double* redVars = this->redVars;
	unsigned int lines = this->lines;
	unsigned int samples = this->samples;
	unsigned int bands = this->bands;
	unsigned int targetEndmembers = this->targetEndmembers;
	int64_t* imax = this->imax;

    start = std::chrono::high_resolution_clock::now();

    /***********
	 * SNR estimation
	 ***********/
    _queue.memcpy(dImage, image, sizeof(double)*lines*samples*bands);
    _queue.wait();

	for (size_t i = 0; i < bands; i++)
		oneapi::mkl::blas::column_major::asum(_queue, N, &dImage[i*N], 1, &mean[i]);
	_queue.wait();

	oneapi::mkl::blas::column_major::scal(_queue, bands, inv_N, mean, 1).wait();

    _queue.parallel_for<class vca_20>(sycl::range(bands, N), [=](auto index) {
		auto i = index[0];
		auto j = index[1];
		meanImage[i*N + j] = dImage[i*N + j] - mean[i];
    }).wait();

	oneapi::mkl::blas::column_major::gemm(_queue, trans, nontrans, bands, bands, N, alpha, meanImage, N, meanImage, N, beta, svdMat, bands);
	_queue.wait();

	oneapi::mkl::blas::column_major::scal(_queue, bands*bands, inv_N, svdMat, 1).wait();

	oneapi::mkl::lapack::gesvd(_queue, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::novec, bands, bands, svdMat, bands, D, U, bands, VT, bands, gesvd_scratchpad, scrach_size);
	_queue.wait();

	oneapi::mkl::blas::column_major::gemm(_queue, nontrans, trans, targetEndmembers, N, bands, alpha, U, bands, meanImage, N, beta, x_p, targetEndmembers);
	_queue.wait();

	oneapi::mkl::blas::column_major::dot(_queue, bands*N, dImage, 1, dImage, 1, &redVars[0]);
	oneapi::mkl::blas::column_major::dot(_queue, N*targetEndmembers, x_p, 1, x_p, 1, &redVars[1]);
	oneapi::mkl::blas::column_major::dot(_queue, bands, mean, 1, mean, 1, &redVars[2]);
	_queue.wait();

	powery = redVars[0] / N; 
	powerx = redVars[1] / N + redVars[2];
	SNR = (SNR < 0) ? 10 * cl::sycl::log10((powerx - targetEndmembers / bands * powery) / (powery - powerx)) : SNR;
	/**********************/

#if defined(DEBUG)
		std::cout << "SNR    = " << SNR << std::endl 
				  << "SNR_th = " << SNR_th << std::endl;
#endif

/***************
 * Choosing Projective Projection or projection to p-1 subspace
 ***************/
	if(SNR < SNR_th) {
#if defined(DEBUG)
		std::cout << "Select proj. to p-1"<< std::endl;
#endif
		_queue.parallel_for<class vca_30>(cl::sycl::range<2>(bands, bands - targetEndmembers), [=](auto index) {
			int i = index[0];
			int j = index[1] + (targetEndmembers-1);
			U[i*bands + j] = 0;
		});

		_queue.parallel_for<class vca_40>(cl::sycl::range<1>(N), [=](auto j) {
			x_p[(targetEndmembers-1)*N + j] = 0;
		});
		_queue.wait();

		_queue.parallel_for<class vca_50>(cl::sycl::range<1>(targetEndmembers), [=](auto index) {
			int i = index[0];
			for(int j{0}; j < N; j++)
				u[i] += x_p[i * N + j] * x_p[i * N + j];
		}).wait();

		oneapi::mkl::blas::column_major::iamax(_queue, targetEndmembers, u, 1, &imax[0]).wait();

		_queue.single_task<class vca_60>([=]() {
			redVars[0] = cl::sycl::sqrt(u[imax[0]]);
		});

		oneapi::mkl::blas::column_major::gemm(_queue, trans, nontrans, bands, N, targetEndmembers, alpha, U, bands, x_p, targetEndmembers, beta, Rp, bands);
		_queue.wait();

		_queue.parallel_for<class vca_70>(cl::sycl::range(bands), [=](auto i) {
			for(int j = 0; j < N; j++)
				Rp[i*N + j] += mean[i];
		});

		_queue.parallel_for<class vca_80>(cl::sycl::range<2>(targetEndmembers-1, N), [=](auto index) {
			int i = index[0];
			int j = index[1];
			y[i*N + j] = x_p[i*N + j];
		});

		_queue.parallel_for<class vca_90>(cl::sycl::range<1>(N), [=](auto index) {
			int j = index[0];
			y[(targetEndmembers-1) * N + j] = redVars[0];
		});
		_queue.wait();
	}

	else {
#if defined(DEBUG)
		std::cout << "Select the projective proj."<< std::endl;
#endif
		oneapi::mkl::blas::column_major::gemm(_queue, trans, nontrans, bands, bands, N, alpha, dImage, N, dImage, N, beta, svdMat, bands);
		_queue.wait();

		oneapi::mkl::blas::column_major::scal(_queue, bands*bands, inv_N, svdMat, 1).wait();

		oneapi::mkl::lapack::gesvd(_queue, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::novec, bands, bands, svdMat, bands, D, U, bands, VT, bands, gesvd_scratchpad, scrach_size);
		_queue.wait();

		oneapi::mkl::blas::column_major::gemm(_queue, nontrans, trans, targetEndmembers, N, bands, alpha, U, bands, dImage, N, beta, x_p, targetEndmembers);
		_queue.wait();

		oneapi::mkl::blas::column_major::gemm(_queue, trans, nontrans, bands, N, targetEndmembers, alpha, U, bands, x_p, targetEndmembers, beta, Rp, bands);
		_queue.wait();

		_queue.parallel_for<class vca_95>(cl::sycl::range<1>(targetEndmembers), [=](auto i) {
			for(int j = 0; j < N; j++)
				u[i] += x_p[i*N + j];
			u[i] *= inv_N;
		}).wait();

		_queue.parallel_for<class vca_100>(cl::sycl::range<1>(targetEndmembers), [=](auto i) {
			for(int j = 0; j < N; j++)
				y[i*N + j] = x_p[i*N + j] * u[i];
		}).wait();

		_queue.parallel_for<class vca_110>(cl::sycl::range<1>(targetEndmembers), [=](auto j) {
			for(int i = 0; i < N; i++)
				sumxu[i] += y[j*N + i];
		}).wait();

		_queue.parallel_for<class vca_120>(cl::sycl::range<1>(targetEndmembers), [=](auto i) {
			for(int j = 0; j < N; j++)
				y[i*N + j] /= sumxu[j];
		}).wait();
	}
	/******************/

	/*******************
	 * VCA algorithm
	 *******************/
	for(int i = 0; i < targetEndmembers; i++) {
		oneapi::mkl::rng::generate(distr, engine, targetEndmembers, w);

		_queue.memcpy(A_copy, A, sizeof(double)*targetEndmembers*targetEndmembers);
    	_queue.wait();

		pinv(_queue, A_copy, targetEndmembers, pinvS, pinvU, pinvVT, pinv_scratchpad, pinv_size);
		
		oneapi::mkl::blas::column_major::gemm(_queue, nontrans, nontrans, targetEndmembers, targetEndmembers, targetEndmembers, alpha, A, targetEndmembers, A_copy, targetEndmembers, beta, aux, targetEndmembers);
		_queue.wait();

		oneapi::mkl::blas::column_major::gemm(_queue, nontrans, nontrans, targetEndmembers, 1, targetEndmembers, alpha, aux, targetEndmembers, w, targetEndmembers, beta, f, targetEndmembers);
		_queue.wait();

		oneapi::mkl::blas::column_major::axpy(_queue, targetEndmembers, -1.0f, w, 1, f, 1).wait();
		oneapi::mkl::blas::column_major::dot(_queue, targetEndmembers, f, 1, f, 1, &redVars[0]).wait();

		_queue.parallel_for<class vca_130>(cl::sycl::range{targetEndmembers}, [=](auto j) {
			f[j] /= cl::sycl::sqrt(redVars[0]);
		}).wait();

		oneapi::mkl::blas::column_major::gemm(_queue, nontrans, trans, 1, N, targetEndmembers, alpha, f, 1, y, N, beta, sumxu, 1);
		_queue.wait();

		oneapi::mkl::blas::column_major::iamax(_queue, N, sumxu, 1, &imax[0]);
		_queue.wait();

		_queue.parallel_for<class vca_150>(cl::sycl::range(targetEndmembers), [=](auto j) {
			A[j*targetEndmembers + i] = y[j*N + imax[0]];
		});

		_queue.parallel_for<class vca_160>(cl::sycl::range(bands), [=](auto j) {
			endmembers[j*targetEndmembers + i] = Rp[j * N + imax[0]];
		});
		_queue.wait();
	}
	/******************/

    end = std::chrono::high_resolution_clock::now();
    tVca += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
#if defined(DEBUG)
	int test = std::accumulate(endmembers, endmembers + (targetEndmembers * bands), 0);
    std::cout << "Test = " << test << std::endl;
#endif
    std::cout << std::endl << "VCA took = " << tVca << " (s)" << std::endl;
}