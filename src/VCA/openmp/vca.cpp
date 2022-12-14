#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <omp.h>
#include <algorithm>

#if defined(NVIDIA_GPU)
#include "cublas.h"
#include <curand.h>
#else
#include "mkl.h"
#include "mkl_omp_offload.h"
#endif

#include "./vca.hpp"

OpenMP_VCA::OpenMP_VCA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers){
    lines   = _lines;
    samples = _samples;
    bands   = _bands;
    targetEndmembers = _targetEndmembers;

    Ud         = new double [bands * targetEndmembers]();
	x_p        = new double [lines * samples * targetEndmembers]();
	y          = new double [lines * samples * targetEndmembers]();
	meanImage  = new double [bands * lines * samples]();
	mean       = new double [bands]();
	svdMat     = new double [bands * bands]();
	D          = new double [bands]();//eigenvalues
	U          = new double [bands * bands]();//eigenvectors
	VT         = new double [bands * bands]();//eigenvectors
	endmembers = new double [targetEndmembers * bands]();
	Rp         = new double [bands * lines * samples]();
	u          = new double [targetEndmembers]();
	sumxu      = new double [lines * samples]();
	w          = new double [targetEndmembers]();
	A          = new double [targetEndmembers * targetEndmembers]();
	A2         = new double [targetEndmembers * targetEndmembers]();
	aux        = new double [targetEndmembers * targetEndmembers]();
	f          = new double [targetEndmembers]();
    index      = new unsigned int [targetEndmembers]();
    pinvS      = new double[targetEndmembers]();
    pinvU      = new double[targetEndmembers * targetEndmembers]();
    pinvVT     = new double[targetEndmembers * targetEndmembers]();
    Utranstmp  = new double[targetEndmembers * targetEndmembers]();
}


OpenMP_VCA::~OpenMP_VCA(){
    if(Ud != nullptr) delete[] Ud;
	if(x_p != nullptr) delete[] x_p;
	if(y != nullptr) delete[] y;
	if(meanImage != nullptr) delete[] meanImage;
	if(mean != nullptr) delete[] mean;
	if(svdMat != nullptr) delete[] svdMat;
	if(D != nullptr) delete[] D;
	if(U != nullptr) delete[] U;
	if(VT != nullptr) delete[] VT;
	if(endmembers != nullptr) delete[] endmembers;
	if(Rp != nullptr) delete[] Rp;
	if(u != nullptr) delete[] u;
	if(sumxu != nullptr) delete[] sumxu;
	if(w != nullptr) delete[] w;
	if(A != nullptr) delete[] A;
	if(A2 != nullptr) delete[] A2;
	if(aux != nullptr) delete[] aux;
	if(f != nullptr) delete[] f;
    if(index != nullptr) delete[] index;
    if(pinvS != nullptr) delete[] pinvS;
    if(pinvU != nullptr) delete[] pinvU;
    if(pinvVT != nullptr) delete[] pinvVT;
    if(Utranstmp != nullptr) delete[] Utranstmp;
}


void OpenMP_VCA::_runOnCPU(float SNR, const double* image) {
    const unsigned int N{lines*samples};
	double sum1{0}, sum2{0}, powery, powerx, mult{0}, sum1Sqrt{0}, alpha{1.0f}, beta{0.f};
    double SNR_th{15 + 10 * std::log10(targetEndmembers)};
    double superb[bands-1];
    double scarch_pinv[targetEndmembers-1];
	std::uint64_t seed{0};
	VSLStreamStatePtr generator;
	vslNewStream(&generator, VSL_BRNG_MT19937, seed);

    // get mean image
	#pragma omp parallel for
	for(int i = 0; i < bands; i++) {
		double sum{0.0f};
		#pragma omp simd reduction(+: sum)
		for(int j = 0; j < N; j++)
			sum += image[i*N + j];

		mean[i] = sum / N;
	}

	#pragma omp parallel for simd
	for(int i = 0; i < bands; i++) {
		for(int j = 0; j < N; j++)
			meanImage[i*N + j] = image[i*N + j] - mean[i];
	}

	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, bands, N, alpha, meanImage, N, meanImage, N, beta, svdMat, bands);

	#pragma omp parallel for simd
	for(int i = 0; i < bands * bands; i++) 
        svdMat[i] /= N;

    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', bands, bands, svdMat, bands, D, U, bands, VT, bands, superb);

	#pragma omp parallel for simd collapse(2)
	for(int i = 0; i < bands; i++)
		for(int j = 0; j < targetEndmembers; j++)
			Ud[i*targetEndmembers + j] = VT[i*bands + j];

	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, targetEndmembers, N, bands, alpha, Ud, targetEndmembers, meanImage, N, beta, x_p, targetEndmembers);

	#pragma omp parallel for reduction(+: sum1)
	for(int i = 0; i < N*bands; i++)
		sum1 += image[i] * image[i];

	#pragma omp parallel for reduction(+: sum2)
	for(int i = 0; i < N*targetEndmembers; i++)
		sum2 += x_p[i] * x_p[i];

	#pragma omp parallel for reduction(+: mult)
	for(int i = 0; i < bands; i++) { 
        mult += mean[i] * mean[i];
	}

	powery = sum1 / N;
	powerx = sum2 / N + mult;

	SNR = (SNR == 0) ? 
                    10 * std::log10((powerx - targetEndmembers / bands * powery) / (powery - powerx)) :
                    SNR;

	if(SNR < SNR_th) {
		#pragma omp teams distribute parallel for collapse(2)
		for(int i = 0; i < bands; i++)
			for(int j = 0; j < targetEndmembers; j++)
                Ud[i*targetEndmembers + j] = (j < targetEndmembers-1) ? VT[i*bands + j] : 0;

		#pragma omp parallel for simd
		for(int j = 0; j < N; j++)
        	x_p[(targetEndmembers-1)*N + j] = 0;

		#pragma omp parallel for
		for(int i = 0; i < targetEndmembers; i++) {
			#pragma omp simd reduction(+: u[i])
			for(int j = 0; j < N; j++)
				u[i] += x_p[i*N + j] * x_p[i*N + j];
		}

		sum1 = 0;
		#pragma omp parallel for reduction(max: sum1)
		for(int i = 0; i < targetEndmembers; i++) {
			if(sum1 < u[i]) 
                sum1 = u[i];
		}

		sum1Sqrt = std::sqrt(sum1);

		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, N, targetEndmembers, alpha, Ud, targetEndmembers, x_p, targetEndmembers, beta, Rp, bands);

		#pragma omp teams distribute parallel for simd collapse(2)
		for(int i = 0; i < bands; i++)
			for(int j = 0; j < N; j++)
				Rp[i*N + j] += mean[i];

		#pragma omp teams distribute parallel for collapse(2)
		for(int i = 0; i < targetEndmembers; i++)
			for(int j = 0; j < N; j++)
                y[i*N + j] = (i < targetEndmembers-1) ? x_p[i*N + j] : sum1Sqrt;
	}
    else {

		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, bands, N, alpha, image, N, image, N, beta, svdMat, bands);

		#pragma omp teams distribute parallel for simd
		for(int i = 0; i < bands*bands; i++)
            svdMat[i] /= N;

		LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', bands, bands, svdMat, bands, D, U, bands, VT, bands, superb);

		#pragma omp teams distribute parallel for simd collapse(2)
		for(int i = 0; i < bands; i++)
			for(int j = 0; j < targetEndmembers; j++)
				Ud[i*targetEndmembers + j] = VT[i*bands + j];

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, targetEndmembers, N, bands, alpha, Ud, targetEndmembers, image, N, beta, x_p, targetEndmembers);
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, N, targetEndmembers, alpha, Ud, targetEndmembers, x_p, targetEndmembers, beta, Rp, bands);

		#pragma omp parallel for
		for(int i = 0; i < targetEndmembers; i++) {
			#pragma omp simd reduction(+: u[i])
			for(int j = 0; j < N; j++)
				u[i] += x_p[i*N + j];
		}

		#pragma omp parallel for simd collapse(2)
		for(int i = 0; i < targetEndmembers; i++) {
			for(int j = 0; j < N; j++)
				y[i*N + j] = x_p[i*N + j] * u[i];
		}

		#pragma omp parallel for
		for(int i = 0; i < N; i++)
			#pragma omp simd reduction(+: sumxu[i])
			for(int j = 0; j < targetEndmembers; j++)
				sumxu[i] += y[j*N + i];

		#pragma omp teams distribute parallel for simd collapse(2)
		for(int i = 0; i < targetEndmembers; i++)
			for(int j = 0; j < N; j++)
				y[i*N + j] /= sumxu[j];
	}

	A[(targetEndmembers-1) * targetEndmembers] = 1;

	for(int i = 0; i < targetEndmembers; i++) {

		#pragma omp parallel
		{
			const int nthreads = omp_get_num_threads();
			const int tid = omp_get_thread_num();
			vdRngGaussian(
				VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, 
				generator, 
				targetEndmembers/nthreads, 
				w + tid * targetEndmembers / nthreads, 
				0.0, 
				1.0);
		}

        #pragma omp parallel for simd
		for (size_t i = 0; i < targetEndmembers*targetEndmembers; i++)
			A2[i] = A[i];

		// Start of computation of the pseudo inverse A
        LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', targetEndmembers, targetEndmembers, A2, targetEndmembers, pinvS, pinvU, targetEndmembers, pinvVT, targetEndmembers, scarch_pinv);

        double maxi = std::numeric_limits<double>::min();
		
		#pragma omp prallel for reduction(max: maxi)
		for(int i = 0; i < targetEndmembers; i++)
			if(maxi < pinvS[i]) 
				maxi = pinvS[i];

		double tolerance = EPSILON * targetEndmembers * maxi;
        int rank = 0;

		#pragma omp parallel for reduction(+: rank)
        for (int i = 0; i < targetEndmembers; i++) {
            if (pinvS[i] > tolerance) {
                rank++;
                pinvS[i] = 1.0 / pinvS[i];
            }
        }

		#pragma omp teams for simd collapse(2)
        for (int i = 0; i < targetEndmembers; i++)
            for (int j = 0; j < targetEndmembers; j++) 
                Utranstmp[i + j * targetEndmembers] = pinvS[i] * pinvU[j + i * targetEndmembers];

        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, targetEndmembers, targetEndmembers, targetEndmembers, alpha, pinvVT, targetEndmembers, Utranstmp, targetEndmembers, beta, A2, targetEndmembers);
        // End of computation of the pseudo inverse A

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, targetEndmembers, targetEndmembers, targetEndmembers, alpha, A2, targetEndmembers, A, targetEndmembers, beta, aux, targetEndmembers);
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, targetEndmembers, alpha, targetEndmembers, alpha, aux, targetEndmembers, w, targetEndmembers, beta, f, targetEndmembers);

		#pragma omp parallel for simd
	    for(int j = 0; j < targetEndmembers; j++)
	    	f[j] = w[j] - f[j];

	    sum1 = 0;
		#pragma omp parallel for reduction(+: sum1)
	    for(int j = 0; j < targetEndmembers; j++)
	    	sum1 += f[j] * f[j];

		const double sqrt_s1 = std::sqrt(sum1);
		#pragma omp parallel for simd
	    for(int j = 0; j < targetEndmembers; j++)
            f[j] /= sqrt_s1;

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, alpha, N, targetEndmembers, alpha, f, alpha, y, N, beta, sumxu, alpha);

		#pragma omp parallel for simd
	    for(int j = 0; j < N; j++) {
	    	if(sumxu[j] < 0) 
                sumxu[j] *= -1;
	    }

		#pragma omp parallel for reduction(max: index[i])
	    for(int j = 0; j < N; j++) {
	    	if(sumxu[j] > sumxu[index[i]])
	    		index[i] = j;
	    }

		#pragma omp parallel for simd
	    for(int j = 0; j < targetEndmembers; j++)
	    	A[j*targetEndmembers + i] = y[j*N + index[i]];

		#pragma omp parallel for simd
	    for(int j = 0; j < bands; j++)
	    	endmembers[j*targetEndmembers + i] = Rp[j + bands * index[i]];
	}
}


void OpenMP_VCA::_runOnGPU(float SNR, const double* image) {
    const unsigned int N{lines*samples}; 
	double sum1{0}, sum2{0}, powery, powerx, mult{0}, sum1Sqrt{0}, alpha{1.0f}, beta{0.f};
    double SNR_th{15 + 10 * std::log10(targetEndmembers)};
    double superb[bands-1];
	const int default_dev = omp_get_default_device();
    double scarch_pinv[targetEndmembers-1];
	std::uint64_t seed{0};
#if defined(NVIDIA_GPU)
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MT19937);
    curandSetPseudoRandomGeneratorSeed(generator, seed);
#else
	VSLStreamStatePtr generator;
	vslNewStream(&generator, VSL_BRNG_MT19937, seed);
#endif

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

	#pragma omp target enter data \
	map(to: image[0:bands*lines*samples], mean[0:bands], u[0:targetEndmembers], sumxu[0:lines*samples],\
		A[0:targetEndmembers*targetEndmembers]) \
	map(alloc: endmembers[0:targetEndmembers*bands], meanImage[0:bands*lines*samples],\
		Ud[0:bands*targetEndmembers], x_p[0:lines*samples*targetEndmembers], y[0:lines*samples*targetEndmembers],\
		svdMat[0:bands*bands], Rp[0:bands*lines*samples], w[0:targetEndmembers],\
		A2[0:targetEndmembers*targetEndmembers], aux[0:targetEndmembers*targetEndmembers],\
		f[0:targetEndmembers], index[0:targetEndmembers], pinvS[0:targetEndmembers],\
		pinvU[0:targetEndmembers*targetEndmembers], pinvVT[0:targetEndmembers*targetEndmembers],\
		Utranstmp[0:targetEndmembers*targetEndmembers], scarch_pinv[0:targetEndmembers-1],\
		D[0:bands], U[0:bands*bands], VT[0:bands*bands]) \
	device(default_dev)
	{
		// get mean image
		#pragma omp target distribute teams parallel for
		for(int i = 0; i < bands; i++) {
			double sum{0.0f};
			#pragma omp simd reduction(+: sum)
			for(int j = 0; j < N; j++)
				sum += image[i*N + j];

			mean[i] = sum / N;
		}

		#pragma omp target distribute teams parallel for simd
		for(int i = 0; i < bands; i++) {
			for(int j = 0; j < N; j++)
				meanImage[i*N + j] = image[i*N + j] - mean[i];
		}

		#pragma omp target data use_device_ptr(meanImage, svdMat)
		{
			cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, bands, N, alpha, meanImage, N, meanImage, N, beta, svdMat, bands);
		}

		#pragma omp target teams distribute parallel for
		for(int i = 0; i < bands * bands; i++) 
			svdMat[i] /= N;

		#pragma omp target data use_device_ptr(svdMat, VT, D, U) 
		{
			LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', bands, bands, svdMat, bands, D, U, bands, VT, bands, superb);
		}

		#pragma omp target teams distribute parallel for collapse(2)
		for(int i = 0; i < bands; i++)
			for(int j = 0; j < targetEndmembers; j++)
				Ud[i*targetEndmembers + j] = VT[i*bands + j];

		#pragma omp target data use_device_ptr(meanImage, Ud, x_p)
		{
			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, targetEndmembers, N, bands, alpha, Ud, targetEndmembers, meanImage, N, beta, x_p, targetEndmembers);
		}

		#pragma omp target teams distribute parallel for reduction(+: sum1) map(sum1)
		for(int i = 0; i < N*bands; i++){
			sum1 += image[i] * image[i];
		}
		#pragma omp target update from(sum1)

		#pragma omp target teams distribute parallel for reduction(+: sum2) map(sum2)
		for(int i = 0; i < N*targetEndmembers; i++){
			sum2 += x_p[i] * x_p[i];
		}
		#pragma omp target update from(sum2)

		#pragma omp target teams distribute parallel for reduction(+: mult) map(mult)
		for(int i = 0; i < bands; i++) { 
			mult += mean[i] * mean[i];
		}
		#pragma omp target update from(mult)

		powery = sum1 / N;
		powerx = sum2 / N + mult;

		SNR = (SNR == 0) ? 
						10 * std::log10((powerx - targetEndmembers / bands * powery) / (powery - powerx)) :
						SNR;

		if(SNR < SNR_th) {

			#pragma omp target teams distribute parallel for collapse(2)
			for(int i = 0; i < bands; i++)
				for(int j = 0; j < targetEndmembers; j++)
					Ud[i*targetEndmembers + j] = (j < targetEndmembers-1) ? VT[i*bands + j] : 0;

			#pragma omp target teams distribute parallel for simd
			for(int j = 0; j < N; j++)
				x_p[(targetEndmembers-1)*N + j] = 0;

			#pragma omp target teams distribute parallel for
			for(int i = 0; i < targetEndmembers; i++) {
				#pragma omp simd reduction(+: u[i])
				for(int j = 0; j < N; j++)
					u[i] += x_p[i*N + j] * x_p[i*N + j];
			}

			#pragma omp target map(sum1)
			{sum1 = 0;}

			#pragma omp target teams distribute parallel for reduction(max: sum1) map(sum1)
			for(int i = 0; i < targetEndmembers; i++) {
				if(sum1 < u[i]) 
					sum1 = u[i];
			}

			#pragma omp target update from(sum1)
			{sum1Sqrt = std::sqrt(sum1);}

			#pragma omp target data use_device_ptr(Ud, x_p, Rp)
			{
				cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, N, targetEndmembers, alpha, Ud, targetEndmembers, x_p, targetEndmembers, beta, Rp, bands);
			}

			#pragma omp target teams distribute parallel for collapse(2)
			for(int i = 0; i < bands; i++)
				for(int j = 0; j < N; j++)
					Rp[i*N + j] += mean[i];

			#pragma omp target teams distribute parallel for collapse(2)
			for(int i = 0; i < targetEndmembers; i++)
				for(int j = 0; j < N; j++)
					y[i*N + j] = (i < targetEndmembers-1) ? x_p[i*N + j] : sum1Sqrt;
		}
		else {
			
			#pragma omp target data use_device_ptr(image, svdMat)
			{
				cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, bands, N, alpha, image, N, image, N, beta, svdMat, bands);
			}

			#pragma omp target teams distribute parallel for
			for(int i = 0; i < bands*bands; i++)
				svdMat[i] /= N;

			#pragma omp target data use_device_ptr(svdMat, VT, U, D)
			{
				LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', bands, bands, svdMat, bands, D, U, bands, VT, bands, superb);
			}

			#pragma omp target teams distribute parallel for collapse(2)
			for(int i = 0; i < bands; i++)
				for(int j = 0; j < targetEndmembers; j++)
					Ud[i*targetEndmembers + j] = VT[i*bands + j];

			#pragma omp target data use_device_ptr(image, Ud, x_p)
			{
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, targetEndmembers, N, bands, alpha, Ud, targetEndmembers, image, N, beta, x_p, targetEndmembers);
			}

			#pragma omp target data use_device_ptr(Ud, x_p, Rp)
			{
				cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, N, targetEndmembers, alpha, Ud, targetEndmembers, x_p, targetEndmembers, beta, Rp, bands);
			}

			#pragma omp target teams distribute parallel for
			for(int i = 0; i < targetEndmembers; i++) {
				#pragma omp simd reduction(+: u[i])
				for(int j = 0; j < N; j++)
					u[i] += x_p[i*N + j];
			}

			#pragma omp target teams distribute parallel for simd collapse(2)
			for(int i = 0; i < targetEndmembers; i++) {
				for(int j = 0; j < N; j++)
					y[i*N + j] = x_p[i*N + j] * u[i];
			}

			#pragma omp target teams distribute parallel for
			for(int i = 0; i < N; i++)
				#pragma omp reduction(+: sumxu[i])
				for(int j = 0; j < targetEndmembers; j++)
					sumxu[i] += y[j*N + i];

			#pragma omp target teams distribute parallel for collapse(2)
			for(int i = 0; i < targetEndmembers; i++)
				for(int j = 0; j < N; j++)
					y[i*N + j] /= sumxu[j];
		}

		#pragma omp target map(A[:targetEndmembers*targetEndmembers])
		{
			A[(targetEndmembers-1) * targetEndmembers] = 1;
		}

		for(int i = 0; i < targetEndmembers; i++) {
			#pragma omp target teams distribute parallel for
			for (size_t i = 0; i < targetEndmembers; i++) {
#if defined(NVIDIA_GPU)
				w[i] = curand_normal_double(generator, 0.0, 1.0);
#else
				vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, generator, 1, w + i, 0.0, 1.0);
#endif
			}

			#pragma omp target teams distribute parallel for
			for (size_t i = 0; i < targetEndmembers*targetEndmembers; i++)
				A2[i] = A[i];

			// Start of computation of the pseudo inverse A
			#pragma omp target data use_device_ptr(A2, pinvS, pinvU, pinvVT)
			{
				LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', targetEndmembers, targetEndmembers, A2, targetEndmembers, pinvS, pinvU, targetEndmembers, pinvVT, targetEndmembers, scarch_pinv);
			}

			double maxi = std::numeric_limits<double>::min();

			#pragma omp target parallel for reduction(max: maxi)
			for(int i = 0; i < targetEndmembers; i++)
				if(maxi < pinvS[i]) 
					maxi = pinvS[i];
			#pragma omp target update from(maxi)

			double tolerance = EPSILON * targetEndmembers * maxi;
			int rank = 0;

			#pragma omp target teams distribute parallel for reduction(+: rank)
			for (int i = 0; i < targetEndmembers; i++) {
				if (pinvS[i] > tolerance) {
					rank++;
					pinvS[i] = 1.0 / pinvS[i];
				}
			}

			#pragma omp target teams distribute parallel for collapse(2)
			for (int i = 0; i < targetEndmembers; i++)
				for (int j = 0; j < targetEndmembers; j++) 
					Utranstmp[i + j * targetEndmembers] = pinvS[i] * pinvU[j + i * targetEndmembers];

			#pragma omp target data use_device_ptr(A2, pinvVT, Utranstmp)
			{
				cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, targetEndmembers, targetEndmembers, targetEndmembers, alpha, pinvVT, targetEndmembers, Utranstmp, targetEndmembers, beta, A2, targetEndmembers);
			}
			// End of computation of the pseudo inverse A

			#pragma omp target data use_device_ptr(A2, A, aux)
			{
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, targetEndmembers, targetEndmembers, targetEndmembers, alpha, A2, targetEndmembers, A, targetEndmembers, beta, aux, targetEndmembers);
			}
			
			#pragma omp target data use_device_ptr(aux, w, f)
			{
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, targetEndmembers, alpha, targetEndmembers, alpha, aux, targetEndmembers, w, targetEndmembers, beta, f, targetEndmembers);
			}

			#pragma omp target map(sum1)
			{sum1 = 0;}

			#pragma omp target teams distribute parallel for reduction(+: sum1)
			for(int j = 0; j < targetEndmembers; j++) {
				f[j] = w[j] - f[j];
				sum1 += f[j] * f[j];
			}

			#pragma omp target teams distribute parallel for
			for(int j = 0; j < targetEndmembers; j++)
				f[j] /= sqrt(sum1);

			#pragma omp target data use_device_ptr(f, y, sumxu, index)
			{
				cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, alpha, N, targetEndmembers, alpha, f, alpha, y, N, beta, sumxu, alpha);
			}

			#pragma omp target teams distribute parallel for
			for(int j = 0; j < N; j++) {
				if(sumxu[j] < 0) 
					sumxu[j] *= -1;
			}

			#pragma omp target teams distribute parallel for reduction(max: index[i])
			for(int j = 0; j < N; j++) {
				if(sumxu[j] > sumxu[index[i]])
					index[i] = j;
			}

			#pragma omp target teams distribute parallel for
			for(int j = 0; j < targetEndmembers; j++)
				A[j*targetEndmembers + i] = y[j*N + index[i]];

			#pragma omp target teams distribute parallel for
			for(int j = 0; j < bands; j++)
				endmembers[j*targetEndmembers + i] = Rp[j + bands * index[i]];
		}
	}
	#pragma omp target exit data map(from: endmembers[0: targetEndmembers * bands]) device(default_dev)
#if defined(NVIDIA_GPU)
	curandDestroyGenerator(generator);
#else
	vslDeleteStream(&generator);
#endif
}


void OpenMP_VCA::run(float SNR, const double* image) {
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVca{0.f};

	start = std::chrono::high_resolution_clock::now();
#if defined(GPU)
    _runOnGPU(SNR, image);
#else
    _runOnCPU(SNR, image);
#endif
    end = std::chrono::high_resolution_clock::now();
    tVca += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();

    int test = std::accumulate(endmembers, endmembers + (targetEndmembers * bands), 0);
    std::cout << "Test = " << test << std::endl;
    std::cout << std::endl << "VCA took = " << tVca << " (s)" << std::endl;
}