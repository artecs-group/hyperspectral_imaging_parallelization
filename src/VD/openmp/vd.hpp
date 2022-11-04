#ifndef _VD_OPENMP_
#define _VD_OPENMP_

#include "../vd_interface.hpp"

class OpenMP_VD: I_VD {
    public:
        OpenMP_VD(int _lines, int _samples, int _bands);
        ~OpenMP_VD();
        void run(int approxVal, double* image);
    private:
        void runOnCPU(int approxVal, double* image);
        void runOnGPU(int approxVal, double* image);
};

#endif