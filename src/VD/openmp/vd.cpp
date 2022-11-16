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

#include "./vd.hpp"

OpenMP_VD::OpenMP_VD(int _lines, int _samples, int _bands){
    lines      = _lines;
    samples    = _samples;
    bands      = _bands;
    meanSpect  = new double[bands];
    Cov		   = new double[bands * bands];
    Corr	   = new double[bands * bands];
    CovEigVal  = new double[bands];
    CorrEigVal = new double[bands];
    U		   = new double[bands * bands];
    VT	       = new double[bands * bands];
    count      = new unsigned int[FPS];
    estimation = new double[FPS];
    meanImage  = new double [lines * samples * bands];

    // table where find the estimation by FPS
    estimation[0] = 0.906193802436823;
    estimation[1] = 1.644976357133188;
    estimation[2] = 2.185124219133003;
    estimation[3] = 2.629741776210312;
    estimation[4] = 3.015733201402701;
}


OpenMP_VD::~OpenMP_VD() {
    delete[] meanSpect;
    delete[] Cov;
    delete[] Corr;
    delete[] CovEigVal;
    delete[] CorrEigVal;
    delete[] U;
    delete[] VT;
    delete[] count;
    delete[] count;
    delete[] estimation;
    delete[] meanImage;
}


void OpenMP_VD::runOnCPU(const int approxVal, const double* image) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVd{0.f};
    double* mean = new double[bands];
    const unsigned int N{lines*samples};
    double TaoTest{0.f}, sigmaTest{0.f}, sigmaSquareTest{0.f};
    const double alpha{(double) 1/N}, beta{0};
    double superb[bands-1];

    start = std::chrono::high_resolution_clock::now();

    #pragma omp teams distribute
    for(int i = 0; i < bands; i++) {
		mean[i] = 0.f;
        #pragma omp single
        for(int j = 0; j < N; j++)
			mean[i] += image[(i*N) + j];

		mean[i] /= N;
        meanSpect[i] = mean[i];

        #pragma omp parallel for
        for(int j = 0; j < N; j++)
			meanImage[i*N + j] = image[i*N + j] - mean[i];
	}

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, bands, N, alpha, meanImage, N, meanImage, N, beta, Cov, bands);

	//correlation
    #pragma omp teams distribute parallel for collapse(2)
    for(int j = 0; j < bands; j++)
        for(int i = 0; i < bands; i++)
        	Corr[i*bands + j] = Cov[i*bands + j]+(meanSpect[i] * meanSpect[j]);

	//SVD
    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'N', bands, bands, Cov, bands, CovEigVal, U, bands, VT, bands, superb);
    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'N', bands, bands, Corr, bands, CorrEigVal, U, bands, VT, bands, superb);

    //estimation
    std::fill(count, count+FPS, 0);

    for(int i = 0; i < bands; i++) {
    	sigmaSquareTest = (CovEigVal[i]*CovEigVal[i] + CorrEigVal[i]*CorrEigVal[i]) * 2 / samples / lines;
    	sigmaTest = sqrt(sigmaSquareTest);

    	for(int j = 1; j <= FPS; j++) {
            TaoTest = M_SQRT2 * sigmaTest * estimation[j-1];

            if((CorrEigVal[i] - CovEigVal[i]) > TaoTest)
                count[j-1]++;
        }
    }

    result = count[approxVal-1];
    end = std::chrono::high_resolution_clock::now();
    tVd += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
    
    std::cout << "Result = " << result << std::endl;
    std::cout << std::endl << "OpenMP over CPU VD time = " << tVd << " (s)" << std::endl;

    delete[] mean;
}


void OpenMP_VD::runOnGPU(const int approxVal, const double* image) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVd{0.f};
    double mean{0.f}, TaoTest{0.f}, sigmaTest{0.f}, sigmaSquareTest{0.f};
    unsigned int* count = new unsigned int[FPS];
    const unsigned int N{lines*samples};
    const double alpha{(double) 1/N}, beta{0};
    double superb[bands-1];
    const int default_dev = omp_get_default_device();

    //reassign "this" values
    double* estimation = this->estimation;
    double* meanSpect = this->meanSpect;
    double* Cov = this->Cov;
    double* Corr = this->Corr;
    double* CovEigVal = this->CovEigVal;
    double* CorrEigVal = this->CorrEigVal;
    double* U = this->U;
    double* VT = this->VT;
    double* meanImage = this->meanImage;

    start = std::chrono::high_resolution_clock::now();

    std::fill(count, count+FPS, 0);

    #pragma omp target enter data \
    map(to: estimation[0:FPS], count[0:FPS], \
            image[0:lines*samples*bands]) \
    map(alloc: meanSpect[0:bands], Cov[0:bands*bands], \
        Corr[0:bands*bands], CovEigVal[0:bands], CorrEigVal[0:bands], \
        U[0:bands*bands], VT[0:bands*bands], superb[0:bands-1], \
        meanImage[0:lines*samples*bands]) device(default_dev)
    {    
        #pragma omp target teams distribute private(mean) device(default_dev)
        for(int i = 0; i < bands; i++) {
            mean = 0.f;
            #pragma omp single
            for(int j = 0; j < N; j++)
                mean += image[(i*N) + j];

            mean /= N;
            meanSpect[i] = mean;

            #pragma omp parallel for
            for(int j = 0; j < N; j++)
                meanImage[i*N + j] = image[i*N + j] - mean;
        }

        #pragma omp target data use_device_ptr(meanImage, Cov) device(default_dev)
        {
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, bands, N, alpha, meanImage, N, meanImage, N, beta, Cov, bands);
        }

        //correlation
        #pragma omp target teams distribute parallel for collapse(2) device(default_dev)
        for(int j = 0; j < bands; j++)
            for(int i = 0; i < bands; i++)
                Corr[i*bands + j] = Cov[i*bands + j]+(meanSpect[i] * meanSpect[j]);

        //SVD
        #pragma omp target data use_device_ptr(Cov, CovEigVal, U, VT) device(default_dev)
        {
            LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'N', bands, bands, Cov, bands, CovEigVal, U, bands, VT, bands, superb);
        }

        #pragma omp target data use_device_ptr(Corr, CorrEigVal, U, VT) device(default_dev)
        {
            LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'N', bands, bands, Corr, bands, CorrEigVal, U, bands, VT, bands, superb);
        }

        //estimation
        #pragma omp target device(default_dev)
        for(int i = 0; i < bands; i++) {
            sigmaSquareTest = (CovEigVal[i]*CovEigVal[i] + CorrEigVal[i]*CorrEigVal[i]) * 2 / samples / lines;
            sigmaTest = sqrt(sigmaSquareTest);

            for(int j = 1; j <= FPS; j++) {
                TaoTest = M_SQRT2 * sigmaTest * estimation[j-1];

                if((CorrEigVal[i] - CovEigVal[i]) > TaoTest)
                    count[j-1]++;
            }
        }
    }
    #pragma omp target exit data map(from: count[0: FPS]) device(default_dev)

    result = count[approxVal-1];
    end = std::chrono::high_resolution_clock::now();
    tVd += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
    
    std::cout << "Result = " << result << std::endl;
    std::cout << std::endl << "OpenMP over GPU VD time = " << tVd << " (s)" << std::endl;
}


void OpenMP_VD::run(const int approxVal, const double* image) {
#if defined(GPU)
    runOnGPU(approxVal, image);
#else
    runOnCPU(approxVal, image);
#endif
}
