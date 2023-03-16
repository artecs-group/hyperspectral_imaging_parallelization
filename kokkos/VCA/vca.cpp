#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>
#include <limits>

#include "vca.hpp"

KokkosVCA::KokkosVCA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers) {
    lines = _lines;
    samples = _samples;
    bands = _bands;
    targetEndmembers = _targetEndmembers;
    x_p          = Kokkos::View<double**, Layout, MemSpace>("x_p", targetEndmembers, lines * samples);
    y            = Kokkos::View<double**, Layout, MemSpace>("y", targetEndmembers, lines * samples);
    meanImage    = Kokkos::View<double**, Layout, MemSpace>("meanImage", bands, lines*samples);
    mean         = Kokkos::View<double*, Layout, MemSpace>("mean", bands);
    svdMat       = Kokkos::View<double**, Layout, MemSpace>("svdMat", bands, bands);
    D            = Kokkos::View<double*, Layout, MemSpace>("D", bands);
    U            = Kokkos::View<double**, Layout, MemSpace>("U", bands, bands);
    VT           = Kokkos::View<double**, Layout, MemSpace>("VT", bands, bands);
    endmembers   = Kokkos::View<double**, Layout, MemSpace>("endmembers", bands, targetEndmembers);
    h_endmembers = Kokkos::View<double**, Layout, MemSpace>("h_endmembers", bands, targetEndmembers);
    Rp           = Kokkos::View<double**, Layout, MemSpace>("Rp", bands, lines*samples);
    u            = Kokkos::View<double*, Layout, MemSpace>("u", targetEndmembers);
    sumxu        = Kokkos::View<double*, Layout, MemSpace>("sumxu", lines*samples);
    w            = Kokkos::View<double*, Layout, MemSpace>("w", targetEndmembers);
    A            = Kokkos::View<double**, Layout, MemSpace>("A", targetEndmembers, targetEndmembers);
    A_copy       = Kokkos::View<double**, Layout, MemSpace>("A_copy", targetEndmembers, targetEndmembers);
    aux          = Kokkos::View<double**, Layout, MemSpace>("aux", targetEndmembers, targetEndmembers);
    f            = Kokkos::View<double*, Layout, MemSpace>("f", targetEndmembers);
    pinvS        = Kokkos::View<double*, Layout, MemSpace>("pinvS", targetEndmembers);
    pinvU        = Kokkos::View<double**, Layout, MemSpace>("pinvU", targetEndmembers, targetEndmembers);
    pinvVT       = Kokkos::View<double**, Layout, MemSpace>("pinvVT", targetEndmembers, targetEndmembers);
    pinv_work    = Kokkos::View<double*, Layout, MemSpace>("pinv_work", targetEndmembers);
    work         = Kokkos::View<double*, Layout, MemSpace>("work", bands);
}

double* KokkosVCA::getEndmembers() {
    Kokkos::deep_copy(h_endmembers, endmembers);
    return h_endmembers.data();
}

void KokkosVCA::clearMemory() {
    Kokkos::realloc(Kokkos::WithoutInitializing, x_p, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, y, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, meanImage, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, mean, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, svdMat, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, D, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, U, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, VT, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, Rp, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, u, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, sumxu, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, w, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, A, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, A_copy, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, aux, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, f, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, pinvS, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, pinvU, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, pinvVT, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, pinv_work, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, work, 0);
}


void KokkosVCA::run(float SNR, const double* image) {
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVca{0.f};
    unsigned int N{lines*samples};
	double inv_N{1/static_cast<double>(N)};
	double alpha{1.0f}, beta{0.f}, powerx{0}, powery{0};
    const double SNR_th{15 + 10 * std::log10(targetEndmembers)};

    Kokkos::View<double**, Layout, MemSpace> x_p = this->x_p;
    Kokkos::View<double**, Layout, MemSpace> y = this->y;
    Kokkos::View<double**, Layout, MemSpace> meanImage = this->meanImage;
    Kokkos::View<double*, Layout, MemSpace> mean = this->mean;
    Kokkos::View<double**, Layout, MemSpace> svdMat = this->svdMat;
    Kokkos::View<double*, Layout, MemSpace> D = this->D;
    Kokkos::View<double**, Layout, MemSpace> U = this->U;
    Kokkos::View<double**, Layout, MemSpace> VT = this->VT;
    Kokkos::View<double**, Layout, MemSpace> endmembers = this->endmembers;
    Kokkos::View<double**, Layout, MemSpace> h_endmembers = this->h_endmembers;
    Kokkos::View<double**, Layout, MemSpace> Rp = this->Rp;
    Kokkos::View<double*, Layout, MemSpace> u = this->u;
    Kokkos::View<double*, Layout, MemSpace> sumxu = this->sumxu;
    Kokkos::View<double*, Layout, MemSpace> w = this->w;
    Kokkos::View<double**, Layout, MemSpace> A = this->A;
    Kokkos::View<double**, Layout, MemSpace> A_copy = this->A_copy;
    Kokkos::View<double**, Layout, MemSpace> aux = this->aux;
    Kokkos::View<double*, Layout, MemSpace> f = this->f;
    Kokkos::View<double*, Layout, MemSpace> pinvS = this->pinvS;
    Kokkos::View<double**, Layout, MemSpace> pinvU = this->pinvU;
    Kokkos::View<double**, Layout, MemSpace> pinvVT = this->pinvVT;
    Kokkos::View<double*, Layout, MemSpace> pinv_work = this->pinv_work;
    Kokkos::View<double*, Layout, MemSpace> work = this->work;
	unsigned int lines = this->lines;
	unsigned int samples = this->samples;
	unsigned int bands = this->bands;
	unsigned int targetEndmembers = this->targetEndmembers;

    Kokkos::parallel_for("vca_10", 
    Kokkos::RangePolicy<ExecSpace>(0, 1), 
    KOKKOS_LAMBDA(const int i){
        A(targetEndmembers-1, 0) = 1;
    });

    start = std::chrono::high_resolution_clock::now();

    /***********
	 * SNR estimation
	 ***********/
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
	}

	else {
#if defined(DEBUG)
		std::cout << "Select the projective proj."<< std::endl;
#endif
	}
	/******************/

	/*******************
	 * VCA algorithm
	 *******************/
	for(int i = 0; i < targetEndmembers; i++) {

	}
	/******************/

    end = std::chrono::high_resolution_clock::now();
    tVca += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
#if defined(DEBUG)
    double* ends = getEndmembers();
	int test = std::accumulate(ends, ends + (targetEndmembers * bands), 0);
    std::cout << "Test = " << test << std::endl;
#endif
    std::cout << std::endl << "VCA took = " << tVca << " (s)" << std::endl;
}