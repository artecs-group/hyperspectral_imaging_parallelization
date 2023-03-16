#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>

#include <Kokkos_Core.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <KokkosBatched_SVD_Decl.hpp>

#include "vd.hpp"

KokkosVD::KokkosVD(int _lines, int _samples, int _bands) {
    lines = _lines;
    samples = _samples;
    bands = _bands;
    Cov		   = Kokkos::View<double**, Layout, MemSpace>("Cov", bands, bands);
    Corr	   = Kokkos::View<double**, Layout, MemSpace>("Corr", bands, bands);
    CovEigVal  = Kokkos::View<double*, Layout, MemSpace>("CovEigVal", bands);
    CorrEigVal = Kokkos::View<double*, Layout, MemSpace>("CorrEigVal", bands);
    count      = Kokkos::View<unsigned int[FPS], Layout, MemSpace>("count");
    estimation = Kokkos::View<double[FPS], Layout, MemSpace>("estimation");
    mean       = Kokkos::View<double*, Layout, MemSpace>("mean", bands);
    svdWork    = Kokkos::View<double*, Layout, MemSpace>("svdWork", bands);
    d_endmembers = Kokkos::View<unsigned int[1], Layout, MemSpace>("d_endmembers");
    meanImage  = Kokkos::View<double**, Layout, MemSpace>("meanImage", bands, lines*samples);
}


void KokkosVD::initAllocMem() {
    // table where find the estimation by FPS
    Kokkos::View<double*, Layout, MemSpace> estimation = this->estimation;
    Kokkos::RangePolicy<ExecSpace> singleTask(0, 1);
    Kokkos::parallel_for("vd_10", singleTask, KOKKOS_LAMBDA(const int i){
        estimation(0) = 0.906193802436823;
        estimation(1) = 1.644976357133188;
        estimation(2) = 2.185124219133003;
        estimation(3) = 2.629741776210312;
        estimation(4) = 3.015733201402701;
    });
}


void KokkosVD::clearMemory() {
    Kokkos::realloc(Kokkos::WithoutInitializing, meanImage, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, Cov, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, Corr, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, CovEigVal, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, CorrEigVal, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, estimation, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, count, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, mean, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, svdWork, 0);
}


void KokkosVD::run(const int approxVal, const double* _image) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVd{0.f};
    const unsigned int N{lines*samples};
    const double inv_N{1 / static_cast<double>(N)};
    const double alpha{(double) 1/N}, beta{0};
    double* hImage = new double[bands*N];

    std::copy(_image, _image + bands*N, hImage);
    Kokkos::View<double**, Layout, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> vImage(hImage, bands, N);

    Kokkos::View<double*, Layout, MemSpace> CovEigVal = this->CovEigVal;
    Kokkos::View<double*, Layout, MemSpace> CorrEigVal = this->CorrEigVal;
    Kokkos::View<unsigned int[FPS], Layout, MemSpace> count = this->count;
    Kokkos::View<double[FPS], Layout, MemSpace> estimation = this->estimation;
    Kokkos::View<double*, Layout, MemSpace> mean = this->mean;
    Kokkos::View<double**, Layout, MemSpace> Cov = this->Cov;
    Kokkos::View<double**, Layout, MemSpace> Corr = this->Corr;
    Kokkos::View<double**, Layout, MemSpace> meanImage = this->meanImage;
    Kokkos::View<unsigned int[1], Layout, MemSpace> d_endmembers = this->d_endmembers;
    Kokkos::View<double*, Layout, MemSpace> svdWork = this->svdWork;
    unsigned int bands   = this->bands;
    unsigned int samples = this->samples;
    unsigned int lines   = this->lines;

    Kokkos::deep_copy(meanImage, vImage);

    start = std::chrono::high_resolution_clock::now();
    Kokkos::parallel_for("vd_15", 
    Kokkos::RangePolicy<ExecSpace>(0, bands), 
    KOKKOS_LAMBDA(const int i){
        for(int j{0}; j < N; j++)
            mean(i) += meanImage(i, j);
    });

    KokkosBlas::scal(mean, inv_N, mean);

    Kokkos::parallel_for("vd_20", 
    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> ({0,0}, {bands, N}), 
    KOKKOS_LAMBDA(const int i, const int j){
        meanImage(i, j) -= mean(i);
    });

    KokkosBlas::gemm("N", "T", alpha, meanImage, meanImage, beta, Cov);

    Kokkos::parallel_for("vd_30", 
    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> ({0,0}, {bands, bands}), 
    KOKKOS_LAMBDA(const int i, const int j){
        Corr(i, j) = Cov(i, j) + (mean(i) * mean(j));
    });

    Kokkos::parallel_for("vd_35", 
    Kokkos::RangePolicy<ExecSpace>(0, 1), 
    KOKKOS_LAMBDA(const int i){
        KokkosBatched::SerialSVD::invoke(KokkosBatched::SVD_S_Tag(), Cov, CovEigVal, svdWork);
    });

    Kokkos::parallel_for("vd_37", 
    Kokkos::RangePolicy<ExecSpace>(0, 1), 
    KOKKOS_LAMBDA(const int i){
        KokkosBatched::SerialSVD::invoke(KokkosBatched::SVD_S_Tag(), Corr, CorrEigVal, svdWork);
    });

    // Estimation
    const double k = 2 / static_cast<double>(samples) / static_cast<double>(lines);
    Kokkos::parallel_for("vd_40", 
    Kokkos::RangePolicy<ExecSpace>(0, FPS+1), 
    KOKKOS_LAMBDA(const int index){
        const int j = index + 1;
        double TaoTest{0.f}, sigmaTest{0.f}, sigmaSquareTest{0.f};

        for(int i{0}; i < bands; i++) {
            sigmaSquareTest = (CovEigVal(i)*CovEigVal(i) + CorrEigVal(i)*CorrEigVal(i)) * k;
            sigmaTest = Kokkos::sqrt(sigmaSquareTest);

            TaoTest = M_SQRT2 * sigmaTest * estimation(j-1);

            if((CorrEigVal(i) - CovEigVal(i)) > TaoTest)
                count(j-1)++;
        }

    });

    Kokkos::parallel_for("vd_50", 
    Kokkos::RangePolicy<ExecSpace>(0, 1), 
    KOKKOS_LAMBDA(const int i){
        d_endmembers(0) = count(approxVal-1);
    });
    {
        Kokkos::View<unsigned int[1], Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ends(&endmembers);
        Kokkos::deep_copy(ends, d_endmembers);
    }
    end = std::chrono::high_resolution_clock::now();
    tVd += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();
    
#if defined(DEBUG)
    std::cout << "Test = " << endmembers << std::endl;
#endif
    std::cout << std::endl << "VD took = " << tVd << " (s)" << std::endl;
    
    delete[] hImage;
}