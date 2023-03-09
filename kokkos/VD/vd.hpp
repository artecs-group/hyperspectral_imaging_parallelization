#ifndef _VD_KOKKOS_
#define _VD_KOKKOS_

#include "../kokkos_conf.hpp"
#include "../../common/interfaces/vd_interface.hpp"

class KokkosVD: I_VD {
    public:
        KokkosVD(int _lines, int _samples, int _bands);
        ~KokkosVD(){};
        void run(const int approxVal, const double* image);
        unsigned int getNumberEndmembers() { return endmembers; };
        void clearMemory();
        void initAllocMem();// Require for nvcc, not callable from constructor.
    private:
        Kokkos::View<double[FPS], Layout, MemSpace> estimation;
        Kokkos::View<double**, Layout, MemSpace> Cov, Corr, U, VT, meanImage;
        Kokkos::View<double*, Layout, MemSpace> CovEigVal, CorrEigVal, mean;
        Kokkos::View<unsigned int[FPS], Layout, MemSpace> count;
        Kokkos::View<unsigned int[1], Layout, MemSpace> d_endmembers;
};

#endif