#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>

#include "vd.hpp"

KokkosVD::KokkosVD(int _lines, int _samples, int _bands) {
    lines = _lines;
    samples = _samples;
    bands = _bands;
    Cov		   = Kokkos::View<double*, Layout, MemSpace>("Cov", bands*bands);
    Corr	   = Kokkos::View<double*, Layout, MemSpace>("Corr", bands*bands);
    CovEigVal  = Kokkos::View<double*, Layout, MemSpace>("CovEigVal", bands);
    CorrEigVal = Kokkos::View<double*, Layout, MemSpace>("CorrEigVal", bands);
    U		   = Kokkos::View<double*, Layout, MemSpace>("U", bands*bands);
    VT	       = Kokkos::View<double*, Layout, MemSpace>("V", bands*bands);
    count      = Kokkos::View<unsigned int[FPS], Layout, MemSpace>("count");
    meanImage  = Kokkos::View<double*, Layout, MemSpace>("meanImage", lines*samples*bands);
    estimation = Kokkos::View<double[FPS], Layout, MemSpace>("estimation");
    mean       = Kokkos::View<double*, Layout, MemSpace>("mean", bands);
    d_endmembers = Kokkos::View<unsigned int[1], Layout, MemSpace>("d_endmembers");
    // _scrach_size = oneapi::mkl::lapack::gesvd_scratchpad_size<double>(
    //                 _queue, 
    //                 oneapi::mkl::jobsvd::somevec, 
    //                 oneapi::mkl::jobsvd::novec, 
    //                 bands, bands, bands, bands, bands
    //             );
    // _queue.wait();

    // gesvd_scratchpad = sycl::malloc_device<double>(_scrach_size, _queue);
    //initAllocMem();
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
    Kokkos::realloc(Kokkos::WithoutInitializing, Cov, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, Corr, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, CovEigVal, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, CorrEigVal, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, U, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, VT, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, estimation, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, count, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, meanImage, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, mean, 0);
}


void KokkosVD::run(const int approxVal, const double* _image) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tVd{0.f};
    const unsigned int N{lines*samples};
    const double inv_N{1 / static_cast<double>(N)};
    const double alpha{(double) 1/N}, beta{0};

    double* raw_image = new double[N*bands];
    std::copy(_image, _image + N*bands, raw_image);
    Kokkos::View<double*, Layout, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> image(raw_image, N*bands);

    Kokkos::View<double*, Layout, MemSpace> CovEigVal = this->CovEigVal;
    Kokkos::View<double*, Layout, MemSpace> CorrEigVal = this->CorrEigVal;
    Kokkos::View<unsigned int[FPS], Layout, MemSpace> count = this->count;
    Kokkos::View<double[FPS], Layout, MemSpace> estimation = this->estimation;
    Kokkos::View<double*, Layout, MemSpace> meanImage = this->meanImage;
    Kokkos::View<double*, Layout, MemSpace> mean = this->mean;
    Kokkos::View<double*, Layout, MemSpace> Cov = this->Cov;
    Kokkos::View<double*, Layout, MemSpace> Corr = this->Corr;
    Kokkos::View<unsigned int[1], Layout, MemSpace> d_endmembers = this->d_endmembers;
    unsigned int bands   = this->bands;
    unsigned int samples = this->samples;
    unsigned int lines   = this->lines;

    start = std::chrono::high_resolution_clock::now();
	// for (size_t i = 0; i < bands; i++)
	// 	oneapi::mkl::blas::column_major::asum(_queue, N, &meanImage[i*N], 1, &mean[i]);
	// _queue.wait();

	// oneapi::mkl::blas::column_major::scal(_queue, bands, inv_N, mean, 1).wait();

    Kokkos::parallel_for("vd_20", 
    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> ({0,0}, {bands, N}), 
    KOKKOS_LAMBDA(const int i, const int j){
        meanImage((i*N) + j) -= mean(i);
    });

    // oneapi::mkl::blas::column_major::gemm(_queue, trans, nontrans, bands, bands, N, alpha, meanImage, N, meanImage, N, beta, Cov, bands);
    // _queue.wait();

    Kokkos::parallel_for("vd_30", 
    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> ({0,0}, {bands, bands}), 
    KOKKOS_LAMBDA(const int i, const int j){
        Corr((i*bands) + j) = Cov((i*bands) + j) + (mean(i) * mean(j));
    });

    // SVD
    // oneapi::mkl::lapack::gesvd(_queue, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::novec, bands, bands, Cov, bands, CovEigVal, U, bands, VT, bands, gesvd_scratchpad, _scrach_size);
    // oneapi::mkl::lapack::gesvd(_queue, oneapi::mkl::jobsvd::somevec, oneapi::mkl::jobsvd::novec, bands, bands, Corr, bands, CorrEigVal, U, bands, VT, bands, gesvd_scratchpad, _scrach_size);
    // _queue.wait();

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
    delete[] raw_image;
}