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
    index      = sycl::malloc_device<unsigned int>(targetEndmembers, _queue);
	pinvS	   = sycl::malloc_device<double>(targetEndmembers, _queue);
	pinvU	   = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	pinvVT	   = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	Utranstmp  = sycl::malloc_device<double>(targetEndmembers * targetEndmembers, _queue);
	dSNR	   = sycl::malloc_shared<float>(1, _queue);
	sum1Sqrt   = sycl::malloc_shared<double>(1, _queue);

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
    sycl::free(Ud, _queue);
	sycl::free(x_p, _queue);
	sycl::free(y, _queue);
	sycl::free(dImage, _queue);
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
	sycl::free(pinv_scratchpad, _queue);
	sycl::free(pinvS, _queue);
	sycl::free(pinvU, _queue);
	sycl::free(pinvVT, _queue);
	sycl::free(Utranstmp, _queue);
	sycl::free(dSNR, _queue);
	sycl::free(sum1Sqrt, _queue);
}


void SYCL_VCA::run(float SNR, const double* image) {
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVca{0.f};
    unsigned int N{lines*samples}; 
    int info{0};
	double sum1{0}, sum2{0}, mult{0}, alpha{1.0f}, beta{0.f};
    const double SNR_th{15 + 10 * std::log10(targetEndmembers)};

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
	double* sum1Sqrt = this->sum1Sqrt;
    unsigned int* index = this->index;
	unsigned int lines = this->lines;
	unsigned int samples = this->samples;
	unsigned int bands = this->bands;
	unsigned int targetEndmembers = this->targetEndmembers;

    start = std::chrono::high_resolution_clock::now();

    _queue.memcpy(dImage, image, sizeof(double)*lines*samples*bands);
	_queue.memcpy(dSNR, &SNR, sizeof(float));
    _queue.wait();

	_queue.submit([&](cl::sycl::handler &h) {
		h.parallel_for<class vca_10>(cl::sycl::range(bands), [=](auto i) {
			for(int j = 0; j < N; j++)
				mean[i] += dImage[i*N + j];
			
			mean[i] /= N;

			for(int j = 0; j < N; j++)
				meanImage[i*N + j] = dImage[i*N + j] - mean[i];
		});
	}).wait();

	oneapi::mkl::blas::column_major::gemm(_queue, trans, nontrans, bands, bands, N, alpha, meanImage, N, meanImage, N, beta, svdMat, bands);
	_queue.wait();

	_queue.submit([&](cl::sycl::handler &h) {
		h.parallel_for<class vca_20>(cl::sycl::range(bands*bands), [=](auto i) {
			svdMat[i] /= N;
		});
	}).wait();

	oneapi::mkl::lapack::gesvd(_queue, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::somevec, bands, bands, svdMat, bands, D, U, bands, VT, bands, gesvd_scratchpad, scrach_size);
	_queue.wait();

	_queue.submit([&](cl::sycl::handler &h) {
		h.parallel_for<class vca_30>(cl::sycl::range<2>(bands, targetEndmembers), [=](auto index) {
			int i = index[0];
			int j = index[1];
			Ud[i*targetEndmembers + j] = VT[i*bands + j];
		});
	}).wait();

	oneapi::mkl::blas::column_major::gemm(_queue, nontrans, trans, targetEndmembers, N, bands, alpha, Ud, targetEndmembers, meanImage, N, beta, x_p, targetEndmembers);
	_queue.wait();

    _queue.submit([&](sycl::handler &h) {
        h.single_task<class vca_40>([=]() {
			double dSum1{0}, dSum2{0}, dMult{0}, powerx{0}, powery{0};

            for(int i = 0; i < N*bands; i++) {
				dSum1 += dImage[i] * dImage[i];
				if(i < N*targetEndmembers)
					dSum2 += x_p[i] * x_p[i];
				if(i < bands)
					dMult += mean[i] * mean[i];
			}
			powery = dSum1 / N; 
			powerx = dSum2 / N + dMult;
			dSNR[0] = (dSNR[0] == 0) ? 
						10 * cl::sycl::log10((powerx - targetEndmembers / bands * powery) / (powery - powerx)) :
						dSNR[0];
        });
    }).wait();

	if(dSNR[0] < SNR_th) {
		_queue.submit([&](cl::sycl::handler &h) {
			h.parallel_for<class vca_50>(cl::sycl::range<2>(bands, targetEndmembers), [=](auto index) {
				int i = index[0];
				int j = index[1];
				if(j < targetEndmembers-1) 
					Ud[i*targetEndmembers + j] = VT[i*bands + j];
				else 
					Ud[i*targetEndmembers + j] = 0;
			});
		}).wait();

		_queue.submit([&](sycl::handler &h) {
			h.single_task<class vca_60>([=]() {
				double dSum1{0};
				for(int i = 0; i < targetEndmembers; i++) {
					for(int j = 0; j < N; j++) {
						if(i == (targetEndmembers-1))
							x_p[i*N + j] = 0;

						u[i] += x_p[i*N + j] * x_p[i*N + j];
					}
					if(dSum1 < u[i])
						dSum1 = u[i];
				}
				sum1Sqrt[0] = cl::sycl::sqrt(dSum1);
			});
		}).wait();

		oneapi::mkl::blas::column_major::gemm(_queue, trans, nontrans, bands, N, targetEndmembers, alpha, Ud, targetEndmembers, x_p, targetEndmembers, beta, Rp, bands);
		_queue.wait();

		_queue.submit([&](cl::sycl::handler &h) {
			h.parallel_for<class vca_70>(cl::sycl::range(bands), [=](auto i) {
				for(int j = 0; j < N; j++)
					Rp[i*N + j] += mean[i];
			});
		}).wait();

		_queue.submit([&](cl::sycl::handler &h) {
			h.parallel_for<class vca_80>(cl::sycl::range<2>(targetEndmembers, N), [=](auto index) {
				int i = index[0];
				int j = index[1];

				if(i < targetEndmembers-1) 
					y[i*N + j] = x_p[i*N + j];
				else 
					y[i*N + j] = sum1Sqrt[0];
			});
		}).wait();
	}

	else {
		oneapi::mkl::blas::column_major::gemm(_queue, trans, nontrans, bands, bands, N, alpha, dImage, N, dImage, N, beta, svdMat, bands);
		_queue.wait();

		_queue.submit([&](cl::sycl::handler &h) {
			h.parallel_for<class vca_90>(cl::sycl::range(bands*bands), [=](auto i) {
				svdMat[i] /= N;
			});
    	}).wait();

		oneapi::mkl::lapack::gesvd(_queue, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::somevec, bands, bands, svdMat, bands, D, U, bands, VT, bands, gesvd_scratchpad, scrach_size);
		_queue.wait();

		_queue.submit([&](cl::sycl::handler &h) {
			h.parallel_for<class vca_100>(cl::sycl::range<2>(bands, targetEndmembers), [=](auto index) {
				int i = index[0];
				int j = index[1];
				Ud[i*targetEndmembers + j] = VT[i*bands + j];
			});
		}).wait();

		oneapi::mkl::blas::column_major::gemm(_queue, nontrans, trans, targetEndmembers, N, bands, alpha, Ud, targetEndmembers, dImage, N, beta, x_p, targetEndmembers);
		_queue.wait();

		oneapi::mkl::blas::column_major::gemm(_queue, trans, nontrans, bands, N, targetEndmembers, alpha, Ud, targetEndmembers, x_p, targetEndmembers, beta, Rp, bands);
		_queue.wait();

		_queue.submit([&](cl::sycl::handler &h) {
			h.parallel_for<class vca_110>(cl::sycl::range(targetEndmembers), [=](auto i) {
					for(int j = 0; j < N; j++)
						u[i] += x_p[i*N + j];

					for(int j = 0; j < N; j++)
						y[i*N + j] = x_p[i*N + j] * u[i];
			});
		}).wait();

		_queue.submit([&](cl::sycl::handler &h) {
			h.parallel_for<class vca_120>(cl::sycl::range(N), [=](auto i) {
				for(int j = 0; j < targetEndmembers; j++)
					sumxu[i] += y[j*N + i];
			});
		}).wait();

		_queue.submit([&](cl::sycl::handler &h) {
			h.parallel_for<class vca_130>(cl::sycl::range(targetEndmembers), [=](auto i) {
				for(int j = 0; j < N; j++)
						y[i*N + j] /= sumxu[j];
			});
		}).wait();
	}

	for(int i = 0; i < targetEndmembers; i++) {
		_queue.submit([&](cl::sycl::handler &h) {
			int imax = std::numeric_limits<int>::max();
			h.parallel_for<class vca_140>(cl::sycl::range(targetEndmembers), [=](auto j) {
				w[j]  = 16000 % imax; // Cambiamos el valor rand() por un valor fijo 16000
				w[j] /= imax;
			});
		}).wait();
		
		_queue.memcpy(A2, A, sizeof(double)*targetEndmembers*targetEndmembers);
    	_queue.wait();

		// Start of computation of the pseudo inverse A
		oneapi::mkl::lapack::gesvd(_queue, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::somevec, targetEndmembers, targetEndmembers, A2, targetEndmembers, pinvS, pinvU, targetEndmembers, pinvVT, targetEndmembers, pinv_scratchpad, pinv_size);
		_queue.wait();

		_queue.submit([&](cl::sycl::handler &h) {
			h.single_task<class vca_150>([=]() {
				int maxi = std::numeric_limits<double>::min();
				int rank{0};
				double tolerance{0};

				for(int i = 0; i < targetEndmembers; i++)
					if(maxi < pinvS[i]) 
						maxi = pinvS[i];

				tolerance = EPSILON * targetEndmembers * maxi;

				for (int i = 0; i < targetEndmembers; i++) {
					if (pinvS[i] > tolerance) {
						rank += 1;
						pinvS[i] = 1.0 / pinvS[i];
					}
				}
			});
		}).wait();

		_queue.submit([&](cl::sycl::handler &h) {
			h.parallel_for<class vca_160>(cl::sycl::range<2>(targetEndmembers, targetEndmembers), [=](auto index) {
				int i = index[0];
				int j = index[1];
				Utranstmp[j*targetEndmembers + i] = pinvS[i] * pinvU[i*targetEndmembers + j];
			});
		}).wait();

		oneapi::mkl::blas::column_major::gemm(_queue, trans, nontrans, targetEndmembers, targetEndmembers, targetEndmembers, alpha, pinvVT, targetEndmembers, Utranstmp, targetEndmembers, beta, A2, targetEndmembers);
		_queue.wait();
		// End of computation of the pseudo inverse A

		oneapi::mkl::blas::column_major::gemm(_queue, nontrans, nontrans, targetEndmembers, targetEndmembers, targetEndmembers, alpha, A2, targetEndmembers, A, targetEndmembers, beta, aux, targetEndmembers);
		_queue.wait();

		oneapi::mkl::blas::column_major::gemm(_queue, nontrans, nontrans, targetEndmembers, alpha, targetEndmembers, alpha, aux, targetEndmembers, w, targetEndmembers, beta, f, targetEndmembers);
		_queue.wait();

		_queue.submit([&](cl::sycl::handler &h) {
			h.single_task<class vca_170>([=]() {
				double dSum{0};
				for(int j = 0; j < targetEndmembers; j++) {
					f[j] = w[j] - f[j];
					dSum += f[j] * f[j];
				}

				for(int j = 0; j < targetEndmembers; j++) 
					f[j] /= cl::sycl::sqrt(dSum); 
			});
		}).wait();

		oneapi::mkl::blas::column_major::gemm(_queue, nontrans, trans, alpha, N, targetEndmembers, alpha, f, alpha, y, N, beta, sumxu, alpha);
		_queue.wait();

		_queue.submit([&](cl::sycl::handler &h) {
			h.single_task<class vca_180>([=]() {
				double dSum{0};
				for(int j = 0; j < N; j++) {
					if(sumxu[j] < 0) 
						sumxu[j] *= -1;
					if(dSum < sumxu[j]) {
						dSum = sumxu[j];
						index[i] = j;
					}
				}
			});
		}).wait();

		_queue.submit([&](cl::sycl::handler &h) {
			h.parallel_for<class vca_190>(cl::sycl::range(targetEndmembers), [=](auto j) {
				A[j*targetEndmembers + i] = y[j*N + index[i]];
			});
		});

		_queue.submit([&](cl::sycl::handler &h) {
			h.parallel_for<class vca_200>(cl::sycl::range(bands), [=](auto j) {
				endmembers[j*targetEndmembers + i] = Rp[bands * index[i] + j];
			});
		});
		_queue.wait();
	}

    end = std::chrono::high_resolution_clock::now();
    tVca += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
    int result = std::accumulate(endmembers, endmembers + (targetEndmembers * bands), 0);
    std::cout << "Endmembers sum = " << result << std::endl;
    std::cout << std::endl << "SYCL VCA time = " << tVca << " (s)" << std::endl;
}