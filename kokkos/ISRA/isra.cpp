#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>

#include<KokkosBlas1_fill.hpp>
#include <KokkosBlas3_gemm.hpp>

#include "./isra.hpp"

KokkosISRA::KokkosISRA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers) {
    lines            = _lines;
    samples          = _samples;
    bands            = _bands;
    targetEndmembers = _targetEndmembers;

    abundanceMatrix   = Kokkos::View<double**, Layout, MemSpace>("abundanceMatrix", lines * samples, targetEndmembers);
    h_abundanceMatrix = Kokkos::View<double**, Layout, Kokkos::HostSpace>("h_abundanceMatrix", lines * samples, targetEndmembers);
    numerator         = Kokkos::View<double**, Layout, MemSpace>("numerator", lines * samples, targetEndmembers);
    denominator       = Kokkos::View<double**, Layout, MemSpace>("denominator", lines * samples, targetEndmembers);
    aux               = Kokkos::View<double**, Layout, MemSpace>("aux", targetEndmembers, targetEndmembers);
    comput            = Kokkos::View<double**, Layout, MemSpace>("comput", targetEndmembers, bands);
    image             = Kokkos::View<double**, Layout, MemSpace>("image", bands, lines * samples);
    endmembers        = Kokkos::View<double**, Layout, MemSpace>("endmembers", bands, targetEndmembers);
}

double* KokkosISRA::getAbundanceMatrix() {
    return h_abundanceMatrix.data();
}

void KokkosISRA::clearMemory() {
    Kokkos::realloc(Kokkos::WithoutInitializing, abundanceMatrix, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, numerator, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, denominator, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, aux, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, comput, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, image, 0, 0);
    Kokkos::realloc(Kokkos::WithoutInitializing, endmembers, 0, 0);
}


void KokkosISRA::run(int maxIter, const double* _image, const double* _endmembers) {
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    float tIsra{0.f};
    unsigned int N{lines * samples};
    double alpha{1.0}, beta{0.0};

    Kokkos::View<double**, Layout, MemSpace> abundanceMatrix = this->abundanceMatrix;
    Kokkos::View<double**, Layout, MemSpace> numerator = this->numerator;
    Kokkos::View<double**, Layout, MemSpace> denominator = this->denominator;

    Kokkos::View<const double**, Layout, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hImage(_image, bands, N);
    Kokkos::View<const double**, Layout, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> hEnds(_endmembers, bands, targetEndmembers);

    start = std::chrono::high_resolution_clock::now();
    Kokkos::deep_copy(image, hImage);
    Kokkos::deep_copy(endmembers, hEnds);
    
    KokkosBlas::fill(abundanceMatrix, alpha);
    KokkosBlas::gemm("T", "N", alpha, image, endmembers, beta, numerator);
    KokkosBlas::gemm("T", "N", alpha, endmembers, endmembers, beta, aux);

    for(int i = 0; i < maxIter; i++) {
        KokkosBlas::gemm("N", "N", alpha, abundanceMatrix, aux, beta, denominator);

        Kokkos::parallel_for("isra_10", 
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> ({0, 0}, {N, targetEndmembers}), 
        KOKKOS_LAMBDA(const int i, const int j){
            abundanceMatrix(i, j) = abundanceMatrix(i, j) * (numerator(i, j) / denominator(i, j));
        });
    }

    end = std::chrono::high_resolution_clock::now();
    tIsra += std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();

    Kokkos::deep_copy(h_abundanceMatrix, abundanceMatrix);
#if defined(DEBUG)
    double* abd = getAbundanceMatrix();
    int ntest = (targetEndmembers * N) < 10 ? targetEndmembers * N : 10;
    std::cout << "Abundance matrix first " << ntest << " elements: " << std::endl << "      -> ";
    for (size_t i = 0; i < ntest; i++)
        std::cout << abd[i] << ", ";
    std::cout << std::endl;
#endif
    std::cout << std::endl << "ISRA took = " << tIsra << " (s)" << std::endl;
}