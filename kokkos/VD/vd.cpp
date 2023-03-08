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
    count      = Kokkos::View<unsigned int*, Layout, MemSpace>("count", FPS);
    meanImage  = Kokkos::View<double*, Layout, MemSpace>("meanImage", lines*samples*bands);
    estimation = Kokkos::View<double*, Layout, MemSpace>("estimation", FPS);
    mean       = Kokkos::View<double*, Layout, MemSpace>("mean", bands);
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
    Kokkos::RangePolicy<ExecSpace> policy(0, 1);
    Kokkos::parallel_for("init_estimation", policy, KOKKOS_LAMBDA(const int i){
        estimation(0) = 0.906193802436823;
        estimation(1) = 1.644976357133188;
        estimation(2) = 2.185124219133003;
        estimation(3) = 2.629741776210312;
        estimation(4) = 3.015733201402701;
    });
}


KokkosVD::~KokkosVD() {
}


void KokkosVD::clearMemory() {}


void KokkosVD::run(const int approxVal, const double* image) {
}