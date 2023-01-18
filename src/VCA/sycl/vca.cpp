#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"
//#include "oneapi/mkl/rng.hpp"

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
	A2         = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	aux        = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	f          = sycl::malloc_device<double>(targetEndmembers, _queue);
	pinvS	   = sycl::malloc_device<double>(targetEndmembers, _queue);
	pinvU	   = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	pinvVT	   = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	Utranstmp  = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	dSNR	   = sycl::malloc_shared<float>(1, _queue);
	redVars    = sycl::malloc_device<double>(3, _queue);
	maxIdx     = sycl::malloc_shared<pair<double, int>>(1, _queue);

    scrach_size = oneapi::mkl::lapack::gesvd_scratchpad_size<double>(
                    _queue, 
                    oneapi::mkl::jobsvd::somevec, 
                    oneapi::mkl::jobsvd::somevec, 
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
		h.single_task<class vca_5>([=]() {
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
	
    if(Ud != nullptr) {sycl::free(Ud, _queue); Ud = nullptr;}
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
	if(A2 != nullptr) {sycl::free(A2, _queue); A2 = nullptr;}
	if(aux != nullptr) {sycl::free(aux, _queue); aux = nullptr;}
	if(f != nullptr) {sycl::free(f, _queue); f = nullptr;}
	if(gesvd_scratchpad != nullptr) {sycl::free(gesvd_scratchpad, _queue); gesvd_scratchpad = nullptr;}
	if(pinv_scratchpad != nullptr) {sycl::free(pinv_scratchpad, _queue); pinv_scratchpad = nullptr;}
	if(pinvS != nullptr) {sycl::free(pinvS, _queue); pinvS = nullptr;}
	if(pinvU != nullptr) {sycl::free(pinvU, _queue); pinvU = nullptr;}
	if(pinvVT != nullptr) {sycl::free(pinvVT, _queue); pinvVT = nullptr;}
	if(Utranstmp != nullptr) {sycl::free(Utranstmp, _queue); Utranstmp = nullptr;}
	if(dSNR != nullptr) {sycl::free(dSNR, _queue); dSNR = nullptr;}
	if(redVars != nullptr) {sycl::free(redVars, _queue); redVars = nullptr;}
	if(maxIdx != nullptr) {sycl::free(maxIdx, _queue); maxIdx = nullptr;}
}


void SYCL_VCA::run(float SNR, const double* image) {
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVca{0.f};
    unsigned int N{lines*samples};
	double alpha{1.0f}, beta{0.f};
    const double SNR_th{15 + 10 * std::log10(targetEndmembers)};

	std::uint64_t seed{0};
	oneapi::mkl::rng::mrg32k3a engine(_queue, seed);
	oneapi::mkl::rng::gaussian<double, oneapi::mkl::rng::gaussian_method::box_muller2> distr(0.0, 1.0);


    double* Ud = this->Ud;
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
	double* A2 = this->A2;
	double* aux = this->aux;
	double* f = this->f;
	double* pinvS = this->pinvS;
	double* pinvU = this->pinvU;
	double* pinvVT = this->pinvVT;
	double* Utranstmp = this->Utranstmp;
	float* dSNR = this->dSNR;
	double* redVars = this->redVars;
	unsigned int lines = this->lines;
	unsigned int samples = this->samples;
	unsigned int bands = this->bands;
	unsigned int targetEndmembers = this->targetEndmembers;
	pair<double, int>* maxIdx = this->maxIdx;

    start = std::chrono::high_resolution_clock::now();

    /***********
	 * SNR estimation
	 ***********/
    _queue.memcpy(dImage, image, sizeof(double)*lines*samples*bands);
	_queue.memcpy(dSNR, &SNR, sizeof(float));
    _queue.wait();

    _queue.parallel_for<class vca_10>(sycl::range(bands), [=](auto i) {
		for(int j = 0; j < N; j++)
			mean[i] += dImage[i*N + j];
		mean[i] /= N;
    }).wait();

    _queue.parallel_for<class vca_15>(sycl::range(bands, N), [=](auto index) {
		auto i = index[0];
		auto j = index[1];
		meanImage[i*N + j] = dImage[i*N + j] - mean[i];
    }).wait();

	oneapi::mkl::blas::column_major::gemm(_queue, trans, nontrans, bands, bands, N, alpha, meanImage, N, meanImage, N, beta, svdMat, bands);
	_queue.wait();

	const size_t max_wgs = _queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
	size_t wgs = (bands*bands < max_wgs) ? bands*bands : max_wgs;
	size_t blocks = bands*bands / wgs + (bands*bands % wgs == 0 ? 0 : 1);
	_queue.parallel_for<class vca_20>(sycl::nd_range<1>{blocks*wgs, wgs}, [=](sycl::nd_item<1> it) {
		auto i = it.get_global_id(0);
		if(i < bands*bands)
			svdMat[i] /= N;
	}).wait();

	oneapi::mkl::lapack::gesvd(_queue, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::somevec, bands, bands, svdMat, bands, D, U, bands, VT, bands, gesvd_scratchpad, scrach_size);
	_queue.wait();

	_queue.parallel_for<class vca_30>(sycl::range<2>{bands, targetEndmembers}, [=](sycl::item<2> it) {
		auto i = it[0];
		auto j = it[1];
		
		Ud[i*targetEndmembers + j] = VT[i*bands + j];
	}).wait();

	oneapi::mkl::blas::column_major::gemm(_queue, nontrans, trans, targetEndmembers, N, bands, alpha, Ud, targetEndmembers, meanImage, N, beta, x_p, targetEndmembers);
	_queue.wait();

	wgs = (N*bands < max_wgs) ? N*bands : max_wgs;
	blocks = (N*bands + wgs - 1) / wgs;
	_queue.parallel_for(sycl::nd_range<1>{blocks*wgs, wgs}, sycl::reduction(&redVars[0], sycl::plus<>()), [=](sycl::nd_item<1> it, auto& red) {
		auto id = it.get_global_id(0);
		if(id < N*bands)
			red.combine(dImage[id]*dImage[id]);
	});

	wgs = (N*targetEndmembers < max_wgs) ? N*targetEndmembers : max_wgs;
	blocks = (N*targetEndmembers + wgs - 1) / wgs;
	_queue.parallel_for(sycl::nd_range<1>{blocks*wgs, wgs}, sycl::reduction(&redVars[1], sycl::plus<>()), [=](sycl::nd_item<1> it, auto& red) {
		auto id = it.get_global_id(0);
		if(id < N*targetEndmembers)
			red.combine(x_p[id]*x_p[id]);
	});

	wgs = (bands < max_wgs) ? bands : max_wgs;
	blocks = (bands + wgs - 1) / wgs;
	_queue.parallel_for(sycl::nd_range<1>{blocks*wgs, wgs}, sycl::reduction(&redVars[2], sycl::plus<>()), [=](sycl::nd_item<1> it, auto& red) {
		auto id = it.get_global_id(0);
		if(id < bands)
			red.combine(mean[id]*mean[id]);
	});
	_queue.wait();

    _queue.single_task<class vca_40>([=]() {
		double powerx{0}, powery{0};

		powery = redVars[0] / N; 
		powerx = redVars[1] / N + redVars[2];
		dSNR[0] = (dSNR[0] == 0) ? 
					10 * cl::sycl::log10((powerx - targetEndmembers / bands * powery) / (powery - powerx)) :
					dSNR[0];
    }).wait();
	/**********************/

#if defined(DEBUG)
		std::cout << "SNR    = " << dSNR[0] << std::endl 
				  << "SNR_th = " << SNR_th << std::endl;
#endif

/***************
 * Choosing Projective Projection or projection to p-1 subspace
 ***************/
	if(dSNR[0] < SNR_th) {
#if defined(DEBUG)
		std::cout << "Select the projective proj."<< std::endl;
#endif
		_queue.parallel_for<class vca_50>(cl::sycl::range<2>(bands, targetEndmembers), [=](auto index) {
			int i = index[0];
			int j = index[1];
			Ud[i*targetEndmembers + j] = VT[i*bands + j];
		}).wait();

		_queue.parallel_for<class vca_53>(cl::sycl::range<1>(bands), [=](auto index) {
			int i = index[0];
			Ud[i*targetEndmembers + (targetEndmembers-1)] = 0;
		});

		_queue.parallel_for<class vca_55>(cl::sycl::range<1>(N), [=](auto index) {
			int j = index[0];
			x_p[(targetEndmembers-1)*N + j] = 0;
		});
		_queue.wait();

		_queue.parallel_for<class vca_57>(cl::sycl::range<2>(targetEndmembers, N), [=](auto index) {
			int i = index[0];
			int j = index[1];
			u[i] += x_p[i*N + j] * x_p[i*N + j];
		});
		_queue.wait();

		wgs = (targetEndmembers < max_wgs) ? targetEndmembers : max_wgs;
		blocks = (targetEndmembers + wgs - 1) / wgs;
		_queue.parallel_for(sycl::nd_range<1>{blocks*wgs, wgs}, sycl::reduction(&redVars[0], sycl::maximum<>()), [=](sycl::nd_item<1> it, auto& red) {
			auto id = it.get_global_id(0);
			if(id < targetEndmembers)
				red.combine(u[id]);
		}).wait();

		_queue.single_task<class vca_60>([=]() {
			redVars[0] = cl::sycl::sqrt(redVars[0]);
		}).wait();

		oneapi::mkl::blas::column_major::gemm(_queue, trans, nontrans, bands, N, targetEndmembers, alpha, Ud, targetEndmembers, x_p, targetEndmembers, beta, Rp, bands);
		_queue.wait();

		_queue.parallel_for<class vca_70>(cl::sycl::range(bands), [=](auto i) {
			for(int j = 0; j < N; j++)
				Rp[i*N + j] += mean[i];
		}).wait();

		_queue.parallel_for<class vca_80>(cl::sycl::range<2>(targetEndmembers, N), [=](auto index) {
			int i = index[0];
			int j = index[1];

			if(i < targetEndmembers-1) 
				y[i*N + j] = x_p[i*N + j];
			else 
				y[i*N + j] = redVars[0];
		}).wait();
	}

	else {
#if defined(DEBUG)
		std::cout << "Select proj. to p-1"<< std::endl;
#endif
		oneapi::mkl::blas::column_major::gemm(_queue, trans, nontrans, bands, bands, N, alpha, dImage, N, dImage, N, beta, svdMat, bands);
		_queue.wait();

		wgs = (bands*bands < max_wgs) ? bands*bands : max_wgs;
		blocks = (bands*bands + wgs - 1) / wgs;
		_queue.parallel_for<class vca_90>(sycl::nd_range<1>{blocks*wgs, wgs}, [=](sycl::nd_item<1> it) {
			auto i = it.get_global_id(0);
			if(i < bands*bands)
				svdMat[i] /= N;
    	}).wait();

		oneapi::mkl::lapack::gesvd(_queue, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::somevec, bands, bands, svdMat, bands, D, U, bands, VT, bands, gesvd_scratchpad, scrach_size);
		_queue.wait();

		_queue.parallel_for<class vca_100>(cl::sycl::range<2>(bands, targetEndmembers), [=](auto index) {
			int i = index[0];
			int j = index[1];
			Ud[i*targetEndmembers + j] = VT[i*bands + j];
		}).wait();

		oneapi::mkl::blas::column_major::gemm(_queue, nontrans, trans, targetEndmembers, N, bands, alpha, Ud, targetEndmembers, dImage, N, beta, x_p, targetEndmembers);
		_queue.wait();

		oneapi::mkl::blas::column_major::gemm(_queue, trans, nontrans, bands, N, targetEndmembers, alpha, Ud, targetEndmembers, x_p, targetEndmembers, beta, Rp, bands);
		_queue.wait();

		_queue.parallel_for<class vca_110>(cl::sycl::range(targetEndmembers), [=](auto i) {
			for(int j = 0; j < N; j++)
				u[i] += x_p[i*N + j];
			u[i] /= N;
		}).wait();

		wgs = (targetEndmembers < max_wgs) ? targetEndmembers : max_wgs;
		blocks = (targetEndmembers + wgs - 1) / wgs;
		_queue.parallel_for<class vca_115>(sycl::nd_range<1>{blocks*wgs, wgs}, [=](sycl::nd_item<1> it) {
			size_t i = it.get_global_id(0);
			if(i < targetEndmembers) {
				for(int j = 0; j < N; j++)
					y[i*N + j] = x_p[i*N + j] * u[i];
			}
		}).wait();

		_queue.parallel_for<class vca_120>(cl::sycl::range(N), [=](auto i) {
			for(int j = 0; j < targetEndmembers; j++)
				sumxu[i] += y[j*N + i];
		}).wait();

		wgs = (targetEndmembers < max_wgs) ? targetEndmembers : max_wgs;
		blocks = (targetEndmembers + wgs - 1) / wgs;
		_queue.parallel_for<class vca_130>(sycl::nd_range<1>{blocks*wgs, wgs}, [=](sycl::nd_item<1> it) {
			size_t i = it.get_global_id(0);
			if(i < targetEndmembers) {
				for(int j = 0; j < N; j++)
					y[i*N + j] /= sumxu[j];
			}
		}).wait();
	}
	/******************/

	/*******************
	 * VCA algorithm
	 *******************/
	for(int i = 0; i < targetEndmembers; i++) {
		oneapi::mkl::rng::generate(distr, engine, targetEndmembers, w);
		_queue.wait();

		_queue.memcpy(A2, A, sizeof(double)*targetEndmembers*targetEndmembers);
    	_queue.wait();

		// Start of computation of the pseudo inverse A
		oneapi::mkl::lapack::gesvd(_queue, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::somevec, targetEndmembers, targetEndmembers, A2, targetEndmembers, pinvS, pinvU, targetEndmembers, pinvVT, targetEndmembers, pinv_scratchpad, pinv_size);
		_queue.wait();

		wgs = (targetEndmembers < max_wgs) ? targetEndmembers : max_wgs;
		blocks = (targetEndmembers + wgs - 1) / wgs;
		_queue.parallel_for(sycl::nd_range<1>{blocks*wgs, wgs}, sycl::reduction(&redVars[1], sycl::maximum<>()), [=](sycl::nd_item<1> it, auto& red) {
			auto id = it.get_global_id(0);
			if(id < targetEndmembers)
				red.combine(pinvS[id]);
		}).wait();

		wgs = (targetEndmembers < max_wgs) ? targetEndmembers : max_wgs;
		blocks = (targetEndmembers + wgs - 1) / wgs;
		_queue.parallel_for<class vca_150>(sycl::nd_range<1>{blocks*wgs, wgs}, [=](sycl::nd_item<1> it) {
			size_t j = it.get_global_id(0);
			const double tolerance = EPSILON * targetEndmembers * redVars[1];
			if (pinvS[j] > tolerance && j < targetEndmembers)
				pinvS[j] = 1.0 / pinvS[j];
		}).wait();

		_queue.parallel_for<class vca_160>(cl::sycl::range<2>(targetEndmembers, targetEndmembers), [=](auto index) {
			int i = index[0];
			int j = index[1];
			Utranstmp[j*targetEndmembers + i] = pinvS[i] * pinvU[i*targetEndmembers + j];
		}).wait();

		oneapi::mkl::blas::column_major::gemm(_queue, trans, nontrans, targetEndmembers, targetEndmembers, targetEndmembers, alpha, pinvVT, targetEndmembers, Utranstmp, targetEndmembers, beta, A2, targetEndmembers);
		_queue.wait();
		// End of computation of the pseudo inverse A

		oneapi::mkl::blas::column_major::gemm(_queue, nontrans, nontrans, targetEndmembers, targetEndmembers, targetEndmembers, alpha, A2, targetEndmembers, A, targetEndmembers, beta, aux, targetEndmembers);
		_queue.wait();

		oneapi::mkl::blas::column_major::gemm(_queue, nontrans, nontrans, targetEndmembers, alpha, targetEndmembers, alpha, aux, targetEndmembers, w, targetEndmembers, beta, f, targetEndmembers);
		_queue.wait();

		_queue.parallel_for<class vca_163>(cl::sycl::range{targetEndmembers}, [=](auto j) {
			f[j] = w[j] - f[j];
		}).wait();

		wgs = (targetEndmembers < max_wgs) ? targetEndmembers : max_wgs;
		blocks = (targetEndmembers + wgs - 1) / wgs;
		_queue.parallel_for(sycl::nd_range<1>{blocks*wgs, wgs}, sycl::reduction(&redVars[0], sycl::plus<>()), [=](sycl::nd_item<1> it, auto& red) {
			auto id = it.get_global_id(0);
			if(id < targetEndmembers)
				red.combine(f[id]*f[id]);
		}).wait();

		_queue.single_task<class vca_167>([=]() {
			redVars[0] = cl::sycl::sqrt(redVars[0]);
		}).wait();

		_queue.parallel_for<class vca_170>(cl::sycl::range{targetEndmembers}, [=](auto j) {
			f[j] /= redVars[0];
		}).wait();

		oneapi::mkl::blas::column_major::gemm(_queue, nontrans, trans, alpha, N, targetEndmembers, alpha, f, alpha, y, N, beta, sumxu, alpha);
		_queue.wait();

		wgs = (N < max_wgs) ? N : max_wgs;
		blocks = (N + wgs - 1) / wgs;
		_queue.parallel_for<class vca_173>(sycl::nd_range<1>{blocks*wgs, wgs}, [=](sycl::nd_item<1> it) {
			auto id = it.get_global_id(0);
			if(sumxu[id] < 0 && id < N)
				sumxu[id] *= -1;
		}).wait();

		pair<double, int> operator_identity = {std::numeric_limits<int>::min(), std::numeric_limits<int>::min()};
		*maxIdx = operator_identity;
		auto reduction_object = sycl::reduction(maxIdx, operator_identity, sycl::maximum<pair<double, int>>());
		//The implementation handling parallel_for with reduction requires work group size not bigger than 256
		wgs = (N < 256) ? N : 256;
		blocks = (N + wgs - 1) / wgs;
		_queue.parallel_for<class vca_180>(sycl::nd_range<1>{blocks*wgs, wgs}, reduction_object, [=](sycl::nd_item<1> it, auto& red) {
			int id = it.get_global_id(0);
			if(id < N)
				red.combine({sumxu[id], id});
		}).wait();

		_queue.parallel_for<class vca_190>(cl::sycl::range(targetEndmembers), [=](auto j) {
			A[j*targetEndmembers + i] = y[j*N + maxIdx->idx];
		}).wait();

		_queue.parallel_for<class vca_200>(cl::sycl::range(bands), [=](auto j) {
			endmembers[j*targetEndmembers + i] = Rp[bands * maxIdx->idx + j];
		}).wait();
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