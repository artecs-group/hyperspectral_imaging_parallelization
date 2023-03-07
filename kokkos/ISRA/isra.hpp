#ifndef _ISRA_KOKKOS_
#define _ISRA_KOKKOS_

#include "../../common/interfaces/isra_interface.hpp"

class KokkosISRA: I_ISRA {
    public:
        KokkosISRA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers);
        ~KokkosISRA();
        void run(int maxIter, const double* image, const double* endmembers);
        double* getAbundanceMatrix() { return abundanceMatrix; };
        void clearMemory();
    protected:
        void preProcessAbundance(const double* image, double* Ab, const double* e, int targetEndmembers, int lines, int samples, int bands);
        void invTR(double* A, int p);
};

#endif