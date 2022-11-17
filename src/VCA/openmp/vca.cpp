#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <omp.h>
#include <algorithm>

#if defined(NVIDIA_GPU)
#include "cublas.h"
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
}


OpenMP_VCA::~OpenMP_VCA(){
    delete[] Ud;
	delete[] x_p;
	delete[] y;
	delete[] meanImage;
	delete[] mean;
	delete[] svdMat;
	delete[] D;
	delete[] U;
	delete[] VT;
	delete[] endmembers;
	delete[] Rp;
	delete[] u;
	delete[] sumxu;
	delete[] w;
	delete[] A;
	delete[] A2;
	delete[] aux;
	delete[] f;
    delete[] index;
}


void OpenMP_VCA::_runOnCPU(float SNR, const double* image) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVca{0.f};
    const unsigned int N{lines*samples}; 
    int info{0};
	double sum1{0}, sum2{0}, powery, powerx, mult{0}, sum1Sqrt{0}, alpha{1.0f}, beta{0.f};
    double SNR_th{15 + 10 * std::log10(targetEndmembers)};
    double superb[bands-1];

    // Aux variables to compute the pseudo inverse of A
    double* pinvS  = new double[targetEndmembers];
    double* pinvU  = new double[targetEndmembers * targetEndmembers];
    double* pinvVT = new double[targetEndmembers * targetEndmembers];
    double* Utranstmp = new double[targetEndmembers * targetEndmembers];
    double scarch_pinv[targetEndmembers-1];

    start = std::chrono::high_resolution_clock::now();

    // get mean image
	#pragma omp teams distribute
	for(int i = 0; i < bands; i++) {
		#pragma omp single
		for(int j = 0; j < N; j++)
			mean[i] += image[i*N + j];

		mean[i] /= N;

		#pragma omp parallel for
		for(int j = 0; j < N; j++)
			meanImage[i*N + j] = image[i*N + j] - mean[i];
	}

	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, bands, N, alpha, meanImage, N, meanImage, N, beta, svdMat, bands);

	#pragma omp teams distribute parallel for
	for(int i = 0; i < bands * bands; i++) 
        svdMat[i] /= N;

    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', bands, bands, svdMat, bands, D, U, bands, VT, bands, superb);

	#pragma omp teams distribute parallel for collapse(2)
	for(int i = 0; i < bands; i++)
		for(int j = 0; j < targetEndmembers; j++)
			Ud[i*targetEndmembers + j] = VT[i*bands + j];

	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, targetEndmembers, N, bands, alpha, Ud, targetEndmembers, meanImage, N, beta, x_p, targetEndmembers);

	#pragma omp parallel for reduction(+:sum1, sum2, mult)
	for(int i = 0; i < N*bands; i++) {
		sum1 += image[i] * image[i];
		if(i < N * targetEndmembers) 
            sum2 += x_p[i] * x_p[i];
		if(i < bands) 
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

		sum1 = 0;
		#pragma omp teams distribute parallel for
		for(int i = 0; i < targetEndmembers; i++) {
			for(int j = 0; j < N; j++) {
				if(i == targetEndmembers-1) 
                    x_p[i*N + j] = 0;
				u[i] += x_p[i*N + j] * x_p[i*N + j];
			}

			if(sum1 < u[i]) 
                sum1 = u[i];
		}

		sum1Sqrt = std::sqrt(sum1);

		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, N, targetEndmembers, alpha, Ud, targetEndmembers, x_p, targetEndmembers, beta, Rp, bands);

		#pragma omp teams distribute parallel for collapse(2)
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

		#pragma omp teams distribute parallel for
		for(int i = 0; i < bands*bands; i++)
            svdMat[i] /= N;

		LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', bands, bands, svdMat, bands, D, U, bands, VT, bands, superb);

		#pragma omp teams distribute parallel for collapse(2)
		for(int i = 0; i < bands; i++)
			for(int j = 0; j < targetEndmembers; j++)
				Ud[i*targetEndmembers + j] = VT[i*bands + j];

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, targetEndmembers, N, bands, alpha, Ud, targetEndmembers, image, N, beta, x_p, targetEndmembers);
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, N, targetEndmembers, alpha, Ud, targetEndmembers, x_p, targetEndmembers, beta, Rp, bands);

		#pragma omp teams distribute
		for(int i = 0; i < targetEndmembers; i++) {

			#pragma omp single
			for(int j = 0; j < N; j++)
				u[i] += x_p[i*N + j];

			#pragma omp parallel for
			for(int j = 0; j < N; j++)
				y[i*N + j] = x_p[i*N + j] * u[i];
		}

		#pragma omp teams distribute parallel for
		for(int i = 0; i < N; i++)
			#pragma omp parallel for
			for(int j = 0; j < targetEndmembers; j++)
				sumxu[i] += y[j*N + i];

		#pragma omp teams distribute parallel for collapse(2)
		for(int i = 0; i < targetEndmembers; i++)
			for(int j = 0; j < N; j++)
				y[i*N + j] /= sumxu[j];
	}

	A[(targetEndmembers-1) * targetEndmembers] = 1;

	for(int i = 0; i < targetEndmembers; i++) {

		#pragma omp teams distribute parallel for
		for(int j = 0; j < targetEndmembers; j++) {
			w[j] = 16000 % std::numeric_limits<int>::max(); // Cambiamos el valor rand() por un valor fijo 16000
			w[j] /= std::numeric_limits<int>::max();
		}

        #pragma omp teams distribute parallel for
		for (size_t i = 0; i < targetEndmembers*targetEndmembers; i++)
			A2[i] = A[i];

        LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', targetEndmembers, targetEndmembers, A2, targetEndmembers, pinvS, pinvU, targetEndmembers, pinvVT, targetEndmembers, scarch_pinv);

        double maxi = std::numeric_limits<double>::min();
		#pragma omp parallel
		{
			double pMax = std::numeric_limits<double>::min();
			#pragma omp parallel for
			for(int i = 0; i < targetEndmembers; i++)
				if(pMax < pinvS[i]) 
					pMax = pinvS[i];

			#pragma omp critical
			{
				if(pMax > maxi)
					maxi = pMax;
			}
		}

		double tolerance = EPSILON * targetEndmembers * maxi;

        int rank = 0;

		#pragma omp teams distribute parallel for reduction(+: rank)
        for (int i = 0; i < targetEndmembers; i++) {
            if (pinvS[i] > tolerance) {
                rank++;
                pinvS[i] = 1.0 / pinvS[i];
            }
        }

		#pragma omp teams distribute parallel for collapse(2)
        for (int i = 0; i < targetEndmembers; i++)
            for (int j = 0; j < targetEndmembers; j++) 
                Utranstmp[i + j * targetEndmembers] = pinvS[i] * pinvU[j + i * targetEndmembers];

        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, targetEndmembers, targetEndmembers, targetEndmembers, alpha, pinvVT, targetEndmembers, Utranstmp, targetEndmembers, beta, A2, targetEndmembers);
        // End of computation of the pseudo inverse A

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, targetEndmembers, targetEndmembers, targetEndmembers, alpha, A2, targetEndmembers, A, targetEndmembers, beta, aux, targetEndmembers);
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, targetEndmembers, alpha, targetEndmembers, alpha, aux, targetEndmembers, w, targetEndmembers, beta, f, targetEndmembers);

	    sum1 = 0;

		#pragma omp teams distribute parallel for reduction(+: sum1)
	    for(int j = 0; j < targetEndmembers; j++) {
	    	f[j] = w[j] - f[j];
	    	sum1 += f[j] * f[j];
	    }

		#pragma omp teams distribute parallel for
	    for(int j = 0; j < targetEndmembers; j++)
            f[j] /= sqrt(sum1);

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, alpha, N, targetEndmembers, alpha, f, alpha, y, N, beta, sumxu, alpha);

	    sum2 = 0;
		#pragma omp teams distribute parallel for
	    for(int j = 0; j < N; j++) {
	    	if(sumxu[j] < 0) 
                sumxu[j] *= -1;
	    }

	    for(int j = 0; j < N; j++) {
	    	if(sum2 < sumxu[j]) {
	    		sum2 = sumxu[j];
	    		index[i] = j;
	    	}
	    }

		#pragma omp teams distribute parallel for
	    for(int j = 0; j < targetEndmembers; j++)
	    	A[j*targetEndmembers + i] = y[j*N + index[i]];

		#pragma omp teams distribute parallel for
	    for(int j = 0; j < bands; j++)
	    	endmembers[j*targetEndmembers + i] = Rp[j + bands * index[i]];
	}

    end = std::chrono::high_resolution_clock::now();
    tVca += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();

    int result = std::accumulate(endmembers, endmembers + (targetEndmembers * bands), 0);

    std::cout << "Endmembers sum = " << result << std::endl;
    std::cout << std::endl << "OpenMP over CPU, VCA time = " << tVca << " (s)" << std::endl;

    delete[] pinvS;
    delete[] pinvU;
    delete[] pinvVT;
    delete[] Utranstmp;
}


void OpenMP_VCA::_runOnGPU(float SNR, const double* image) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVca{0.f};
    const unsigned int N{lines*samples}; 
    int info{0};
	double sum1{0}, sum2{0}, powery, powerx, mult{0}, sum1Sqrt{0}, alpha{1.0f}, beta{0.f};
    double SNR_th{15 + 10 * std::log10(targetEndmembers)};
    double superb[bands-1];
	const int default_dev = omp_get_default_device();

    // Aux variables to compute the pseudo inverse of A
    double* pinvS  = new double[targetEndmembers]();
    double* pinvU  = new double[targetEndmembers * targetEndmembers]();
    double* pinvVT = new double[targetEndmembers * targetEndmembers]();
    double* Utranstmp = new double[targetEndmembers * targetEndmembers]();
    double scarch_pinv[targetEndmembers-1];

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
    unsigned int* index = this->index;
	unsigned int lines = this->lines;
	unsigned int samples = this->samples;
	unsigned int bands = this->bands;
	unsigned int targetEndmembers = this->targetEndmembers;

    start = std::chrono::high_resolution_clock::now();

	#pragma omp target enter data \
	map(to: image[0:bands*lines*samples], mean[0:bands], u[0:targetEndmembers], sumxu[0:lines*samples]) \
	map(alloc: endmembers[0:targetEndmembers*bands], meanImage[0:bands*lines*samples],\
		Ud[0:bands*targetEndmembers], x_p[0:lines*samples*targetEndmembers], y[0:lines*samples*targetEndmembers],\
		svdMat[0:bands*bands], Rp[0:bands*lines*samples], w[0:targetEndmembers],\
		A2[0:targetEndmembers*targetEndmembers], aux[0:targetEndmembers*targetEndmembers],\
		f[0:targetEndmembers], index[0:targetEndmembers], pinvS[0:targetEndmembers],\
		pinvU[0:targetEndmembers*targetEndmembers], pinvVT[0:targetEndmembers*targetEndmembers],\
		Utranstmp[0:targetEndmembers*targetEndmembers], scarch_pinv[0:targetEndmembers-1],\
		D[0:bands], U[0:bands*bands], VT[0:bands*bands], A[0:targetEndmembers*targetEndmembers]) \
	device(default_dev)
	{
		// get mean image
		#pragma omp target teams distribute
		for(int i = 0; i < bands; i++) {
			#pragma omp single
			for(int j = 0; j < N; j++)
				mean[i] += image[i*N + j];

			mean[i] /= N;

			#pragma omp parallel for
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

		#pragma omp target teams distribute map(sum1, sum2, mult) reduction(+: sum1, sum2, mult)
		for(int i = 0; i < N*bands; i++) {
			sum1 += image[i] * image[i];
			if(i < N * targetEndmembers) 
				sum2 += x_p[i] * x_p[i];
			if(i < bands) 
				mult += mean[i] * mean[i];
		}
		#pragma omp target update from(sum1, sum2, mult)

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

			#pragma omp target map(sum1)
			{sum1 = 0;}

			#pragma omp target teams distribute map(sum1)
			for(int i = 0; i < targetEndmembers; i++) {

				#pragma omp parallel for
				for(int j = 0; j < N; j++) {
					if(i == targetEndmembers-1) 
						x_p[i*N + j] = 0;
					u[i] += x_p[i*N + j] * x_p[i*N + j];
				}

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

			#pragma omp target teams distribute
			for(int i = 0; i < targetEndmembers; i++) {

				#pragma omp single
				for(int j = 0; j < N; j++)
					u[i] += x_p[i*N + j];

				#pragma omp parallel for
				for(int j = 0; j < N; j++)
					y[i*N + j] = x_p[i*N + j] * u[i];
			}

			#pragma omp target teams distribute parallel for
			for(int i = 0; i < N; i++)
				#pragma omp parallel for
				for(int j = 0; j < targetEndmembers; j++)
					sumxu[i] += y[j*N + i];

			#pragma omp target teams distribute
			for(int i = 0; i < targetEndmembers; i++)
				#pragma omp single
				for(int j = 0; j < N; j++)
					y[i*N + j] /= sumxu[j];
		}

		#pragma omp target map(A[:targetEndmembers*targetEndmembers])
		{
			A[(targetEndmembers-1) * targetEndmembers] = 1;
		}

		for(int i = 0; i < targetEndmembers; i++) {

			#pragma omp target teams distribute parallel for
			for(int j = 0; j < targetEndmembers; j++) {
				w[j] = 16000 % std::numeric_limits<int>::max(); // Cambiamos el valor rand() por un valor fijo 16000
				w[j] /= std::numeric_limits<int>::max();
			}

			#pragma omp target teams distribute parallel for
			for (size_t i = 0; i < targetEndmembers*targetEndmembers; i++)
				A2[i] = A[i];

			#pragma omp target data use_device_ptr(A2, pinvS, pinvU, pinvVT)
			{
				LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', targetEndmembers, targetEndmembers, A2, targetEndmembers, pinvS, pinvU, targetEndmembers, pinvVT, targetEndmembers, scarch_pinv);
			}

			double maxi = std::numeric_limits<double>::min();

			#pragma omp target parallel for reduction(max: maxi)
			for(int i = 0; i < targetEndmembers; i++)
				maxi = (maxi < pinvS[i]) ? pinvS[i] : maxi;
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

			#pragma omp target map(sum2)
			{sum2 = std::numeric_limits<double>::min();}

			// #pragma omp target parallel
			// {
			// 	double pSum = std::numeric_limits<double>::min();
			// 	unsigned int pI{0};
			// 	#pragma omp parallel for
			// 	for(int j = 0; j < N; j++) {
			// 		if(pSum < sumxu[j]) {
			// 			pSum = sumxu[j];
			// 			pI = j;
			// 		}
			// 	}
			// 	#pragma omp critical
			// 	{
			// 		if(pSum > sum2){
			// 			sum2 = pSum;
			// 			index[i] = pI;
			// 		}
			// 	}
			// }

			#pragma omp target
			for(int j = 0; j < N; j++) {
				if(sum2 < sumxu[j]) {
					sum2 = sumxu[j];
					index[i] = j;
				}
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

    end = std::chrono::high_resolution_clock::now();
    tVca += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();

    int result = std::accumulate(endmembers, endmembers + (targetEndmembers * bands), 0);

    std::cout << "Endmembers sum = " << result << std::endl;
    std::cout << std::endl << "OpenMP over GPU, VCA time = " << tVca << " (s)" << std::endl;

    delete[] pinvS;
    delete[] pinvU;
    delete[] pinvVT;
    delete[] Utranstmp;
}


void OpenMP_VCA::run(float SNR, const double* image) {
#if defined(GPU)
    _runOnGPU(SNR, image);
#else
    _runOnCPU(SNR, image);
#endif
}