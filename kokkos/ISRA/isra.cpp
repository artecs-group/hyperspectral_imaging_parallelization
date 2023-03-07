#include <iostream>
#include <cmath>
#include <chrono>
#include <numeric>
#include <algorithm>

#include "./isra.hpp"

KokkosISRA::KokkosISRA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers) {
}


KokkosISRA::~KokkosISRA() {
}

void KokkosISRA::clearMemory() {
}


void KokkosISRA::preProcessAbundance(const double* image, double* Ab,  const double* e, int targetEndmembers, int lines, int samples, int bands) {
}


void KokkosISRA::invTR(double* A, int targetEndmembers) {
}


void KokkosISRA::run(int maxIter, const double* image, const double* endmembers) {
}