#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>
#include <limits>

#include <Kokkos_Random.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_iamax.hpp>
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBatched_Gemm_Decl.hpp>

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
    h_endmembers = Kokkos::View<double**, Layout, Kokkos::HostSpace>("h_endmembers", bands, targetEndmembers);
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
    image        = Kokkos::View<double**, Layout, MemSpace>("image", bands, lines*samples);
    redVar         = Rank0Type("redVar");
}

double* KokkosVCA::getEndmembers() {
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
    Kokkos::realloc(Kokkos::WithoutInitializing, image, 0, 0);
}


void KokkosVCA::run(float SNR, const double* _image) {
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVca{0.f};
    unsigned int N{lines*samples};
	double inv_N{1/static_cast<double>(N)};
	double alpha{1.0f}, beta{0.f}, powerx{0}, powery{0};
    const double SNR_th{15 + 10 * std::log10(targetEndmembers)};
    Kokkos::Random_XorShift64_Pool<ExecSpace> pool(0);

    Kokkos::View<double**, Layout, MemSpace> x_p = this->x_p;
    Kokkos::View<double**, Layout, MemSpace> y = this->y;
    Kokkos::View<double**, Layout, MemSpace> meanImage = this->meanImage;
    Kokkos::View<double*, Layout, MemSpace> mean = this->mean;
    Kokkos::View<double**, Layout, MemSpace> svdMat = this->svdMat;
    Kokkos::View<double*, Layout, MemSpace> D = this->D;
    Kokkos::View<double**, Layout, MemSpace> U = this->U;
    Kokkos::View<double**, Layout, MemSpace> VT = this->VT;
    Kokkos::View<double**, Layout, MemSpace> endmembers = this->endmembers;
    Kokkos::View<double**, Layout, Kokkos::HostSpace> h_endmembers = this->h_endmembers;
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
    Kokkos::View<double**, Layout, MemSpace> image = this->image;
    Rank0Type redVar = this->redVar;
	unsigned int bands = this->bands;
	unsigned int targetEndmembers = this->targetEndmembers;

    Kokkos::View<const double**, Layout, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hImage(_image, bands, N);

    Kokkos::parallel_for("vca_10", 
    Kokkos::RangePolicy<ExecSpace>(0, 1), 
    KOKKOS_LAMBDA(const int i){
        A(targetEndmembers-1, 0) = 1;
    });

    start = std::chrono::high_resolution_clock::now();
    Kokkos::deep_copy(image, hImage);

    /***********
	 * SNR estimation
	 ***********/
    Kokkos::parallel_for("vca_20", 
    Kokkos::RangePolicy<ExecSpace>(0, bands), 
    KOKKOS_LAMBDA(const int i){
        for(int j{0}; j < N; j++)
            mean(i) += image(i, j);
    });

    KokkosBlas::scal(mean, inv_N, mean);

    Kokkos::parallel_for("vca_30", 
    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> ({0,0}, {bands, N}), 
    KOKKOS_LAMBDA(const int i, const int j){
        meanImage(i, j) = image(i, j) - mean(i);
    });

    KokkosBlas::gemm("N", "T", alpha, meanImage, meanImage, beta, svdMat);
    KokkosBlas::scal(svdMat, inv_N, svdMat);

    Kokkos::parallel_for("vca_40", 
    Kokkos::RangePolicy<ExecSpace>(0, 1), 
    KOKKOS_LAMBDA(const int i){
        KokkosBatched::SerialSVD::invoke(KokkosBatched::SVD_USV_Tag(), svdMat, U, D, VT, work);
    });

    //Hint: Kokkos::Subview does not work, since gemm definition just accept Kokkos::View
    Kokkos::resize(U, bands, targetEndmembers);
    KokkosBlas::gemm("T", "N", alpha, U, meanImage, beta, x_p);

    Kokkos::View<double*, Layout, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> flatImage(image.data(), bands*N);
    double redI = KokkosBlas::dot(flatImage, flatImage);

    Kokkos::View<double*, Layout, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> flatXp(x_p.data(), targetEndmembers*N);
    double redX = KokkosBlas::dot(flatXp, flatXp);

    double redM = KokkosBlas::dot(mean, mean);

	powery = redI / N; 
	powerx = redX / N + redM;
	SNR = (SNR < 0) ? 10 * std::log10((powerx - targetEndmembers / bands * powery) / (powery - powerx)) : SNR;

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

        Kokkos::parallel_for("vca_50", 
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> ({0, targetEndmembers-1}, {bands, bands - targetEndmembers}), 
        KOKKOS_LAMBDA(const int i, const int j){
            U(i, j) = 0;
        });

        Kokkos::parallel_for("vca_60", 
        Kokkos::RangePolicy<ExecSpace> (0, N), 
        KOKKOS_LAMBDA(const int j){
            x_p(targetEndmembers-1, j) = 0;
        });

        Kokkos::parallel_for("vca_70", 
        Kokkos::RangePolicy<ExecSpace> (0, targetEndmembers), 
        KOKKOS_LAMBDA(const int i){
            for(int j{0}; j < N; j++)
                u(i) += x_p(i, j) * x_p(i, j);
        });

        KokkosBlas::iamax(redVar, u);

        Kokkos::parallel_for("vca_80", 
        Kokkos::RangePolicy<ExecSpace>(0, 1), 
        KOKKOS_LAMBDA(const int i){
            redVar() = Kokkos::sqrt(u(redVar()));
        });

        KokkosBlas::gemm("N", "N", alpha, U, x_p, beta, Rp);

        Kokkos::parallel_for("vca_90", 
        Kokkos::RangePolicy<ExecSpace>(0, bands), 
        KOKKOS_LAMBDA(const int i){
            for (size_t j = 0; j < N; j++)
                Rp(i, j) += mean(i);
        });

        Kokkos::parallel_for("vca_100", 
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> ({0, 0}, {targetEndmembers-1, N}), 
        KOKKOS_LAMBDA(const int i, const int j){
            y(i, j) = x_p(i, j);
        });

        Kokkos::parallel_for("vca_110", 
        Kokkos::RangePolicy<ExecSpace>(0, N), 
        KOKKOS_LAMBDA(const int i){
            y(targetEndmembers-1, i) = redVar();
        });
	}
	else {
#if defined(DEBUG)
		std::cout << "Select the projective proj."<< std::endl;
#endif
        KokkosBlas::gemm("N", "T", alpha, meanImage, meanImage, beta, svdMat);
        KokkosBlas::scal(svdMat, inv_N, svdMat);

        Kokkos::resize(U, bands, bands);
        Kokkos::parallel_for("vca_120", 
        Kokkos::RangePolicy<ExecSpace>(0, 1), 
        KOKKOS_LAMBDA(const int i){
            KokkosBatched::SerialSVD::invoke(KokkosBatched::SVD_USV_Tag(), svdMat, U, D, VT, work);
        });

        Kokkos::resize(U, bands, targetEndmembers);
        KokkosBlas::gemm("N", "T", alpha, U, meanImage, beta, x_p);
        KokkosBlas::gemm("N", "N", alpha, U, x_p, beta, Rp);

        Kokkos::parallel_for("vca_120", 
        Kokkos::RangePolicy<ExecSpace>(0, targetEndmembers), 
        KOKKOS_LAMBDA(const int i){
            for (size_t j = 0; j < N; j++)
                u(i) += x_p(i, j);
            u(i) *= inv_N;
        });

        Kokkos::parallel_for("vca_130", 
        Kokkos::RangePolicy<ExecSpace>(0, targetEndmembers), 
        KOKKOS_LAMBDA(const int i){
            for (size_t j = 0; j < N; j++)
                y(i, j) += x_p(i, j) * u(i);
        });

        Kokkos::parallel_for("vca_140", 
        Kokkos::RangePolicy<ExecSpace>(0, targetEndmembers), 
        KOKKOS_LAMBDA(const int i){
            for (size_t j = 0; j < N; j++)
                sumxu(i) += y(i, j);
        });

        Kokkos::parallel_for("vca_150", 
        Kokkos::RangePolicy<ExecSpace>(0, targetEndmembers), 
        KOKKOS_LAMBDA(const int i){
            for (size_t j = 0; j < N; j++)
                y(i, j) /= sumxu(j);
        });
	}
	/******************/

	/*******************
	 * VCA algorithm
	 *******************/
    using namespace KokkosBatched;
	for(int i = 0; i < targetEndmembers; i++) {
        Kokkos::fill_random(w, pool, 0.0, 1.0);
        Kokkos::deep_copy(A_copy, A);
        pinv(A_copy, targetEndmembers, pinvS, pinvU, pinvVT, pinv_work);
        KokkosBlas::gemm("N", "N", alpha, A, A_copy, beta, aux);

        Kokkos::parallel_for("vca_160", 
        Kokkos::RangePolicy<ExecSpace>(0, 1), 
        KOKKOS_LAMBDA(const int j){
            SerialGemm<Trans::NoTranspose, Trans::NoTranspose, Algo::Gemm::Unblocked>
                ::invoke(alpha, aux, w, beta, f);
        });

        KokkosBlas::axpy(-1.0, w, f);
        KokkosBlas::dot(redVar, f, f);

        Kokkos::parallel_for("vca_170", 
        Kokkos::RangePolicy<ExecSpace>(0, targetEndmembers), 
        KOKKOS_LAMBDA(const int j){
            f(j) /= Kokkos::sqrt(redVar());
        });

        Kokkos::parallel_for("vca_180", 
        Kokkos::RangePolicy<ExecSpace>(0, 1), 
        KOKKOS_LAMBDA(const int j){
            SerialGemm<Trans::Transpose, Trans::NoTranspose, Algo::Gemm::Unblocked>
                ::invoke(alpha, y, f, beta, sumxu);
        });

        KokkosBlas::iamax(redVar, sumxu);

        Kokkos::parallel_for("vca_190", 
        Kokkos::RangePolicy<ExecSpace>(0, targetEndmembers), 
        KOKKOS_LAMBDA(const int j){
            A(j, i) = y(j, redVar());
        });

        Kokkos::parallel_for("vca_200", 
        Kokkos::RangePolicy<ExecSpace>(0, bands), 
        KOKKOS_LAMBDA(const int j){
            endmembers(j, i) = Rp(j, redVar());
        });
	}
	/******************/

    end = std::chrono::high_resolution_clock::now();
    tVca += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
    Kokkos::deep_copy(h_endmembers, endmembers);
#if defined(DEBUG)
    double* ends = h_endmembers.data();
	int test = std::accumulate(ends, ends + (targetEndmembers * bands), 0);
    std::cout << "Test = " << test << std::endl;
#endif
    std::cout << std::endl << "VCA took = " << tVca << " (s)" << std::endl;
}