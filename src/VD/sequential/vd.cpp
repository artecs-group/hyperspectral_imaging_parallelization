#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include "mkl.h"

#include "./vd.hpp"

SequentialVD::SequentialVD(int _lines, int _samples, int _bands){
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

    // table where find the estimation by FPS
    estimation[0] = 0.906193802436823;
    estimation[1] = 1.644976357133188;
    estimation[2] = 2.185124219133003;
    estimation[3] = 2.629741776210312;
    estimation[4] = 3.015733201402701;
}


SequentialVD::~SequentialVD() {
    delete[] meanSpect;
    delete[] Cov;
    delete[] Corr;
    delete[] CovEigVal;
    delete[] CorrEigVal;
    delete[] U;
    delete[] VT;
    delete[] count;
    delete[] estimation;
}


void SequentialVD::run(const int approxVal, double* image) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVd{0.f};
    double mean{0.f}, TaoTest{0.f}, sigmaTest{0.f}, sigmaSquareTest{0.f};
    const unsigned int N{lines*samples};
    const double alpha{(double) 1/N}, beta{0};
    double superb[bands-1];

    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < bands; i++) {
		mean = 0.f;
        mean = std::accumulate(image + (i*N), image + (i*N + N), 0);
		mean /= N;
        meanSpect[i] = mean;

        for(int j = 0; j < N; j++)
			image[i*N + j] -= mean;
	}

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, bands, N, alpha, image, N, image, N, beta, Cov, bands);

	//correlation
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
    std::cout << std::endl << "Sequential VD time = " << tVd << " (s)" << std::endl;
}