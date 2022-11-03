#include "vd.hpp"

#include <cmath>
#include "mkl.h"

SequentialVD::SequentialVD(int _bands, int _N){
    bands = _bands;
    N = _N;
    meanSpect  = new double[bands * sizeof(double)];
    Cov		   = new double[bands * bands * sizeof(double)];
    Corr	   = new double[bands * bands * sizeof(double)];
    CovEigVal  = new double[bands * sizeof(double)];
    CorrEigVal = new double[bands * sizeof(double)];
    U		   = new double[bands * bands * sizeof(double)];
    VT	       = new double[bands * bands * sizeof(double)];
    estimation = new double[FPS * sizeof(double)];

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
    delete[] estimation;
}


void SequentialVD::run(int approxVal, double* image) {
    float mean{0.f};

    for(int i = 0; i < bands; i++) {
		mean = 0.f;
        for(int j = 0; j < N; j++)
			mean += (image[i*N + j]);

		mean /= N;
        meanSpect[i] = mean;

        for(int j = 0; j < N; j++)
			image[i*N + j] = image[i*N + j] - mean;
	}

    double alpha = (double)1/N, beta = 0;
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, bands, bands, N, alpha, image, N, image, N, beta, Cov, bands);

	//CORRELATION
    for(int j = 0; j < bands; j++)
        for(int i = 0; i < bands; i++)
        	Corr[i*bands + j] = Cov[i*bands + j]+(meanSpect[i] * meanSpect[j]);

	//SVD
    double superb[bands-1];
    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'N', bands, bands, Cov, bands, CovEigVal, U, bands, VT, bands, superb);
    LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'N', 'N', bands, bands, Corr, bands, CorrEigVal, U, bands, VT, bands, superb);

    //ESTIMATION
    int* count = new int[FPS * sizeof(int)];
    std::fill(count, count+FPS, 0);

    for(int i = 0; i < bands; i++) {
    	sigmaSquareTest = (CovEigVal[i]*CovEigVal[i] + CorrEigVal[i]*CorrEigVal[i]) * 2 / samples / lines;
    	sigmaTest = sqrt(sigmaSquareTest);

    	for(int j = 0; j < FPS; j++) {
            TaoTest = sqrt(2) * sigmaTest * estimation[j];

            if((CorrEigVal[i] - CovEigVal[i]) > TaoTest)
                count[j-1]++;
        }
    }

    result = count[approxVal-1];
    
    delete[] count;
}