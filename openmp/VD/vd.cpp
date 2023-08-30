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

#include "vd.hpp"

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
    clearMemory();
}


void OpenMP_VD::clearMemory() {
    if(meanSpect != nullptr) {delete[] meanSpect; meanSpect = nullptr; }
    if(Cov != nullptr) {delete[] Cov; Cov = nullptr; }
    if(Corr != nullptr) {delete[] Corr; Corr = nullptr; }
    if(CovEigVal != nullptr) {delete[] CovEigVal; CovEigVal = nullptr; }
    if(CorrEigVal != nullptr) {delete[] CorrEigVal; CorrEigVal = nullptr; }
    if(U != nullptr) {delete[] U; U = nullptr; }
    if(VT != nullptr) {delete[] VT; VT = nullptr; }
    if(count != nullptr) {delete[] count; count = nullptr; }
    if(estimation != nullptr) {delete[] estimation; estimation = nullptr; }
    if(meanImage != nullptr) {delete[] meanImage; meanImage = nullptr; }
}


void OpenMP_VD::runOnCPU(const int approxVal, const double* image) {
    const unsigned int N{lines*samples};
    const double inv_N{1 / static_cast<double>(N)};
    double TaoTest{0.f}, sigmaTest{0.f}, sigmaSquareTest{0.f};
    const double alpha{(double) 1/N}, beta{0};
    double superb[bands-1];
    const double k = 2 / static_cast<double>(samples) / static_cast<double>(lines);

    #pragma omp parallel for
    for (size_t i = 0; i < bands; i++)
        meanSpect[i] = cblas_dasum(N, &image[i*N], 1);
    
    cblas_dscal(bands, inv_N, meanSpect, 1);

    #pragma omp parallel for simd
    for (int i = 0; i < bands; i++) {
        for (int j = 0; j < N; j++)
            meanImage[i * N + j] = image[i * N + j] - meanSpect[i];
    }

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, bands, N, alpha, meanImage, N, meanImage, N, beta, Cov, bands);

	//correlation
    #pragma omp teams distribute parallel for simd collapse(2)
    for(int j = 0; j < bands; j++)
        for(int i = 0; i < bands; i++)
        	Corr[i*bands + j] = Cov[i*bands + j] + (meanSpect[i] * meanSpect[j]);

	//SVD
    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'N', bands, bands, Cov, bands, CovEigVal, U, bands, VT, bands, superb);
    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'N', bands, bands, Corr, bands, CorrEigVal, U, bands, VT, bands, superb);

    //estimation
    #pragma omp parallel for simd
    for (size_t i = 0; i < FPS; i++)
        count[i] = 0;

    #pragma omp parallel for
    for(int i = 0; i < bands; i++) {
        const double testChecker = CorrEigVal[i] - CovEigVal[i];
    	sigmaSquareTest = (CovEigVal[i]*CovEigVal[i] + CorrEigVal[i]*CorrEigVal[i]) * k;
    	sigmaTest = std::sqrt(sigmaSquareTest) * M_SQRT2;

    	for(int j = 1; j <= FPS; j++) {
            TaoTest = sigmaTest * estimation[j-1];
            if(TaoTest < testChecker)
                count[j-1]++;
        }
    }
    endmembers = count[approxVal-1];
}


void OpenMP_VD::runOnGPU(const int approxVal, const double* image) {
    double TaoTest{0.f}, sigmaTest{0.f}, sigmaSquareTest{0.f};
    unsigned int* count = new unsigned int[FPS];
    const unsigned int N{lines*samples};
    const double inv_N{1 / static_cast<double>(N)};
    const double alpha{(double) 1/N}, beta{0};
    const double k = 2 / static_cast<double>(samples) / static_cast<double>(lines);
    double superb[bands-1];
    const int ompDevice = 1;//omp_get_default_device();

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

    #pragma omp target enter data \
    map(to: estimation[0:FPS], image[0:lines*samples*bands]) \
    map(alloc: meanSpect[0:bands], Cov[0:bands*bands], \
        Corr[0:bands*bands], CovEigVal[0:bands], CorrEigVal[0:bands], \
        U[0:bands*bands], VT[0:bands*bands], superb[0:bands-1], \
        meanImage[0:lines*samples*bands], count[0:FPS]) device(ompDevice)
    {
        for (size_t i = 0; i < bands; i++) {
            #pragma omp target variant dispatch use_device_ptr(image) device(ompDevice)
            {meanSpect[i] = cblas_dasum(N, &image[i*N], 1);}
        }
        
        #pragma omp target variant dispatch use_device_ptr(meanSpect) device(ompDevice)
        {cblas_dscal(bands, inv_N, meanSpect, 1);}

        #pragma omp target teams distribute parallel for simd device(ompDevice)
        for(int i = 0; i < bands; i++) {
            for(int j = 0; j < N; j++)
                meanImage[i*N + j] = image[i*N + j] - meanSpect[i];
        }

        #pragma omp target variant dispatch use_device_ptr(meanImage, Cov) device(ompDevice)
        {
            cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, bands, N, alpha, meanImage, N, meanImage, N, beta, Cov, bands);
        }

        //correlation
        #pragma omp target teams distribute parallel for collapse(2) device(ompDevice)
        for(int j = 0; j < bands; j++)
            for(int i = 0; i < bands; i++)
                Corr[i*bands + j] = Cov[i*bands + j]+(meanSpect[i] * meanSpect[j]);

        //SVD
        #pragma omp target variant dispatch use_device_ptr(Cov, CovEigVal, U, VT) device(ompDevice)
        {
            LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'N', bands, bands, Cov, bands, CovEigVal, U, bands, VT, bands, superb);
        }

        #pragma omp target variant dispatch use_device_ptr(Corr, CorrEigVal, U, VT) device(ompDevice)
        {
            LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'S', 'N', bands, bands, Corr, bands, CorrEigVal, U, bands, VT, bands, superb);
        }

        #pragma omp target teams distribute parallel for simd device(ompDevice)
        for (size_t i = 0; i < FPS; i++)
            count[i] = 0;

        //estimation
        #pragma omp target parallel for device(ompDevice)
        for(int i = 0; i < bands; i++) {
            const double testChecker = CorrEigVal[i] - CovEigVal[i];
            sigmaSquareTest = (CovEigVal[i]*CovEigVal[i] + CorrEigVal[i]*CorrEigVal[i]) * k;
            sigmaTest = std::sqrt(sigmaSquareTest) * M_SQRT2;

            for(int j = 1; j <= FPS; j++) {
                TaoTest = sigmaTest * estimation[j-1];
                if(TaoTest < testChecker)
                    count[j-1]++;
            }
        }
    }
    #pragma omp target exit data map(from: count[0: FPS], CorrEigVal[0:10]) device(ompDevice)

    endmembers = count[approxVal-1];
    for (size_t i = 0; i < 10; i++)
        std::cout << CorrEigVal[i] << ", ";
    std::cout << std::endl;
    
}


void OpenMP_VD::run(const int approxVal, const double* image) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVd{0.f};

    start = std::chrono::high_resolution_clock::now();
#if defined(GPU)
    runOnGPU(approxVal, image);
#else
    runOnCPU(approxVal, image);
#endif
    end = std::chrono::high_resolution_clock::now();
    tVd += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
    
#if defined(DEBUG)
    std::cout << "Test = " << endmembers << std::endl;
#endif
    std::cout << std::endl << "VD took = " << tVd << " (s)" << std::endl;
}
