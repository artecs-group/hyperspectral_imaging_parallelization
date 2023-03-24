#ifndef _ISRA_KOKKOS_
#define _ISRA_KOKKOS_

#include "../kokkos_conf.hpp"
#include "../../common/interfaces/isra_interface.hpp"

class KokkosISRA: I_ISRA {
    public:
        KokkosISRA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers);
        ~KokkosISRA(){};
        void run(int maxIter, const double* image, const double* endmembers);
        double* getAbundanceMatrix();
        void clearMemory();
    protected:
        void preProcessAbundance(const double* image, double* Ab, const double* e, int targetEndmembers, int lines, int samples, int bands){};
        void invTR(double* A, int p){};
    private:
    Kokkos::View<double**, Layout, Kokkos::HostSpace> h_abundanceMatrix;
        Kokkos::View<double**, Layout, MemSpace> abundanceMatrix, numerator, denominator, aux, comput, image, endmembers;
};

#endif