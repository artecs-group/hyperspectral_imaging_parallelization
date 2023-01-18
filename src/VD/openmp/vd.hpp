#ifndef _VD_OPENMP_
#define _VD_OPENMP_

#include "../vd_interface.hpp"

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
        double* mean;
};

#endif