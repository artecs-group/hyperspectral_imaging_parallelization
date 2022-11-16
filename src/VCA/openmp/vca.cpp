#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <omp.h>

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


void OpenMP_VCA::runCPU(float SNR, const double* image) {
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
	for(int i = 0; i < bands; i++) {
		for(int j = 0; j < N; j++)
			mean[i] += image[i*N + j];

		mean[i] /= N;

		for(int j = 0; j < N; j++)
			meanImage[i*N + j] = image[i*N + j] - mean[i];
	}

	cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, bands, N, alpha, meanImage, N, meanImage, N, beta, svdMat, bands);

	for(int i = 0; i < bands * bands; i++) 
        svdMat[i] /= N;

    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', bands, bands, svdMat, bands, D, U, bands, VT, bands, superb);

	for(int i = 0; i < bands; i++)
		for(int j = 0; j < targetEndmembers; j++)
			Ud[i*targetEndmembers + j] = VT[i*bands + j];

	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, targetEndmembers, N, bands, alpha, Ud, targetEndmembers, meanImage, N, beta, x_p, targetEndmembers);

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
		for(int i = 0; i < bands; i++)
			for(int j = 0; j < targetEndmembers; j++)
                Ud[i*targetEndmembers + j] = (j < targetEndmembers-1) ? VT[i*bands + j] : 0;

		sum1 = 0;
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

		for(int i = 0; i < bands; i++)
			for(int j = 0; j < N; j++)
				Rp[i*N + j] += mean[i];

		for(int i = 0; i < targetEndmembers; i++)
			for(int j = 0; j < N; j++)
                y[i*N + j] = (i < targetEndmembers-1) ? x_p[i*N + j] : sum1Sqrt;
	}
    else {

		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, bands, N, alpha, image, N, image, N, beta, svdMat, bands);

		for(int i = 0; i < bands*bands; i++)
            svdMat[i] /= N;

		LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', bands, bands, svdMat, bands, D, U, bands, VT, bands, superb);

		for(int i = 0; i < bands; i++)
			for(int j = 0; j < targetEndmembers; j++)
				Ud[i*targetEndmembers + j] = VT[i*bands + j];

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, targetEndmembers, N, bands, alpha, Ud, targetEndmembers, image, N, beta, x_p, targetEndmembers);
		cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, N, targetEndmembers, alpha, Ud, targetEndmembers, x_p, targetEndmembers, beta, Rp, bands);

		for(int i = 0; i < targetEndmembers; i++) {
			for(int j = 0; j < N; j++)
				u[i] += x_p[i*N + j];

			for(int j = 0; j < N; j++)
				y[i*N + j] = x_p[i*N + j] * u[i];
		}

		for(int i = 0; i < N; i++)
			for(int j = 0; j < targetEndmembers; j++)
				sumxu[i] += y[j*N + i];


		for(int i = 0; i < targetEndmembers; i++)
			for(int j = 0; j < N; j++)
				y[i*N + j] /= sumxu[j];
	}

	A[(targetEndmembers-1) * targetEndmembers] = 1;

	for(int i = 0; i < targetEndmembers; i++) {
		for(int j = 0; j < targetEndmembers; j++) {
			w[j] = 16000 % std::numeric_limits<int>::max(); // Cambiamos el valor rand() por un valor fijo 16000
			w[j] /= std::numeric_limits<int>::max();
		}

        // Compute the pseudo inverse of A
        double* pinvS  = new double[targetEndmembers];
        double* pinvU  = new double[targetEndmembers * targetEndmembers];
        double* pinvVT = new double[targetEndmembers * targetEndmembers];
        double scarch_pinv[targetEndmembers-1];

        std::copy(A, A + targetEndmembers*targetEndmembers, A2);

        LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'S', targetEndmembers, targetEndmembers, A2, targetEndmembers, pinvS, pinvU, targetEndmembers, pinvVT, targetEndmembers, scarch_pinv);

        double maxi = *std::max_element(pinvS, pinvS + targetEndmembers);
        double tolerance = EPSILON * targetEndmembers * maxi;

        int rank = 0;
        for (int i = 0; i < targetEndmembers; i++) {
            if (pinvS[i] > tolerance) {
                rank ++;
                pinvS[i] = 1.0 / pinvS[i];
            }
        }

        for (int i = 0; i < targetEndmembers; i++)
            for (int j = 0; j < targetEndmembers; j++) 
                Utranstmp[i + j * targetEndmembers] = pinvS[i] * pinvU[j + i * targetEndmembers];

        for (int i = targetEndmembers; i < targetEndmembers; i++)
            for (int j = 0; j < targetEndmembers; j++) 
                Utranstmp[i + j * targetEndmembers] = 0.0;

        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, targetEndmembers, targetEndmembers, targetEndmembers, alpha, pinvVT, targetEndmembers, Utranstmp, targetEndmembers, beta, A2, targetEndmembers);
        // End of computation of the pseudo inverse A

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, targetEndmembers, targetEndmembers, targetEndmembers, alpha, A2, targetEndmembers, A, targetEndmembers, beta, aux, targetEndmembers);
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, targetEndmembers, alpha, targetEndmembers, alpha, aux, targetEndmembers, w, targetEndmembers, beta, f, targetEndmembers);

	    sum1 = 0;
	    for(int j = 0; j < targetEndmembers; j++) {
	    	f[j] = w[j] - f[j];
	    	sum1 += f[j] * f[j];
	    }

	    for(int j = 0; j < targetEndmembers; j++)
            f[j] /= sqrt(sum1);

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, alpha, N, targetEndmembers, alpha, f, alpha, y, N, beta, sumxu, alpha);

	    sum2 = 0;
	    for(int j = 0; j < N; j++) {
	    	if(sumxu[j] < 0) 
                sumxu[j] *= -1;
	    	if(sum2 < sumxu[j]) {
	    		sum2 = sumxu[j];
	    		index[i] = j;
	    	}
	    }

	    for(int j = 0; j < targetEndmembers; j++)
	    	A[j*targetEndmembers + i] = y[j*N + index[i]];

	    for(int j = 0; j < bands; j++)
	    	endmembers[j*targetEndmembers + i] = Rp[j + bands * index[i]];
	}

    end = std::chrono::high_resolution_clock::now();
    tVca += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();

    int result = std::accumulate(endmembers, endmembers + (targetEndmembers * bands), 0);

    std::cout << "Endmembers sum = " << result << std::endl;
    std::cout << std::endl << "OpenMP over CPU VCA time = " << tVca << " (s)" << std::endl;

    delete[] pinvS;
    delete[] pinvU;
    delete[] pinvVT;
    delete[] Utranstmp;
}


void OpenMP_VCA::runGPU(float SNR, const double* image) {

}


void OpenMP_VCA::run(float SNR, const double* image) {
#if defined(GPU)
    runGPU(SNR, image);
#else
    runCPU(SNR, image);
#endif
}