#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <random>
#include <limits>

#include "vca.hpp"

KokkosVCA::KokkosVCA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers) {
}


KokkosVCA::~KokkosVCA() {
}

double* KokkosVCA::getEndmembers() {
    Kokkos::deep_copy(h_endmembers, endmembers);
    return h_endmembers.data();
}

void KokkosVCA::clearMemory() {
}


void KokkosVCA::run(float SNR, const double* image) {
}