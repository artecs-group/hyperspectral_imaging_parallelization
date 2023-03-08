#ifndef _VCA_KOKKOS_
#define _VCA_KOKKOS_

#include "../kokkos_conf.hpp"
#include "../../common/interfaces/vca_interface.hpp"

#define EPSILON 1.0e-9

class KokkosVCA: I_VCA {
    public:
        KokkosVCA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers);
        ~KokkosVCA();
        void run(float SNR, const double* image);
        double* getEndmembers();
        void clearMemory();
    private:
        Kokkos::View<unsigned int*, Layout, MemSpace> index;
        Kokkos::View<double*, Layout, MemSpace> Ud, x_p, y, meanImage, mean, svdMat, D, U, VT, Rp, u, sumxu,
             w, A, A_copy, pinvA, aux, f, endmembers, h_endmembers, pinvS, pinvU, pinvVT, pinv_work, work,
             Utranstmp;
};

#endif