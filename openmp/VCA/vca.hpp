#ifndef _VCA_OPENMP_
#define _VCA_OPENMP_

#include "../../common/interfaces/vca_interface.hpp"

class OpenMP_VCA: I_VCA {
    public:
        OpenMP_VCA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers);
        ~OpenMP_VCA();
        void run(float SNR, const double* image);
        double* getEndmembers() { return endmembers; };
        void clearMemory();
    private:
        void _runOnCPU(float SNR, const double* image);
        void _runOnGPU(float SNR, const double* image);
};

#endif