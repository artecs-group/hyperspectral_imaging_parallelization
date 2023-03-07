#ifndef _VCA_KOKKOS_
#define _VCA_KOKKOS_

#include "../../common/interfaces/vca_interface.hpp"

#define EPSILON 1.0e-9

class KokkosVCA: I_VCA {
    public:
        KokkosVCA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers);
        ~KokkosVCA();
        void run(float SNR, const double* image);
        double* getEndmembers() { return endmembers; };
        void clearMemory();
};

#endif