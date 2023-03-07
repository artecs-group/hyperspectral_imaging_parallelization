#ifndef _VD_KOKKOS_
#define _VD_KOKKOS_

#include "../../common/interfaces/vd_interface.hpp"

class KokkosVD: I_VD {
    public:
        KokkosVD(int _lines, int _samples, int _bands);
        ~KokkosVD();
        void run(const int approxVal, const double* image);
        unsigned int getNumberEndmembers() { return endmembers; };
        void clearMemory();
};

#endif