#ifndef _VCA_KOKKOS_
#define _VCA_KOKKOS_

#include <KokkosBlas3_gemm.hpp>
#include <KokkosBatched_SVD_Decl.hpp>

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
        Kokkos::View<double**, Layout, Kokkos::HostSpace> h_endmembers;
        Kokkos::View<double**, Layout, MemSpace> x_p, y, meanImage, svdMat, U, VT, endmembers,
            Rp, A, A_copy, aux, pinvU, pinvVT, image;
        Kokkos::View<double*, Layout, MemSpace> mean, D, u, sumxu, w, f, pinvS, pinv_work, work;
        //Kokkos::View<double[1], Layout, MemSpace> redVar;
        typedef Kokkos::View<typename decltype(u)::size_type, MemSpace> Rank0Type;
        Rank0Type redVar;
};


/**
 * Calculates Moore-Penrose pseudoinverse of a square matrix
 * pinv(A) = V * S^-1 * U
 **/
inline int pinv(Kokkos::View<double**, Layout, MemSpace> A, 
int n, Kokkos::View<double*, Layout, MemSpace> S, 
Kokkos::View<double**, Layout, MemSpace> U, 
Kokkos::View<double**, Layout, MemSpace> VT, 
Kokkos::View<double*, Layout, MemSpace> work) {
    constexpr double alpha{1.0f}, beta{0.0f};
    // A = S U Vt
    Kokkos::parallel_for("pinv_SVD1", 
    Kokkos::RangePolicy<ExecSpace>(0, 1), 
    KOKKOS_LAMBDA(const int i){
        KokkosBatched::SerialSVD::invoke(KokkosBatched::SVD_USV_Tag(), A, U, S, VT, work);
    });

    // S^-1
    Kokkos::parallel_for("pinv_SVD2", 
    Kokkos::RangePolicy<ExecSpace>(0, n), 
    KOKKOS_LAMBDA(const int i){
        if(S(i) > EPSILON)
            S(i) = 1.0 / S(i);
    });

    // Vt = Vt * S^-1
    Kokkos::parallel_for("pinv_SVD3", 
    Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>> ({0, 0}, {n, n}), 
    KOKKOS_LAMBDA(const int i, const int j){
        double scal = S(i);
        VT(i, j) *= scal;
    });

    // pinv(A) = (Vt)t * Ut
    KokkosBlas::gemm("T", "T", alpha, VT, U, beta, A);
    return 0;
}

#endif