#ifndef _VD_OPENMP_
#define _VD_OPENMP_

#include "../../common/interfaces/vd_interface.hpp"

class OpenMP_VD: I_VD {
    public:
        OpenMP_VD(int _lines, int _samples, int _bands);
        ~OpenMP_VD();
        void run(const int approxVal, const double* image);
        unsigned int getNumberEndmembers() { return endmembers; };
        void clearMemory();
    private:
        void runOnCPU(const int approxVal, const double* image);
        void runOnGPU(const int approxVal, const double* image);

        double *meanSpect{nullptr}, 
               *Cov{nullptr}, 
               *Corr{nullptr}, 
               *CovEigVal{nullptr}, 
               *CorrEigVal{nullptr}, 
               *U{nullptr}, 
               *VT{nullptr}, 
               *estimation{nullptr}, 
               *meanImage{nullptr};
        unsigned int* count{nullptr};
};

#endif