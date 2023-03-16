#ifndef _VCA_KOKKOS_
#define _VCA_KOKKOS_

#include "../kokkos_conf.hpp"
#include "../../common/interfaces/vca_interface.hpp"

#define EPSILON 1.0e-9

class KokkosVCA: I_VCA {
    public:
        KokkosVCA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers);
        ~KokkosVCA() {};
        void run(float SNR, const double* image);
        double* getEndmembers();
        void clearMemory();
    private:
        Kokkos::View<unsigned int*, Layout, MemSpace> index;
        Kokkos::View<double**, Layout, MemSpace> x_p, y, meanImage, svdMat, U, VT, endmembers, h_endmembers,
            Rp, A, A_copy, aux, pinvU, pinvVT;
        Kokkos::View<double*, Layout, MemSpace> mean, D, u, sumxu, w, f, pinvS, pinv_work, work;
};

#endif