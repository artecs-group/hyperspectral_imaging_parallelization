#ifndef _ISRA_OPENMP_
#define _ISRA_OPENMP_

#include "../isra_interface.hpp"

class OpenMP_ISRA: I_ISRA {
    public:
        OpenMP_ISRA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers);
        ~OpenMP_ISRA();
        void run(int maxIter, const double* image, const double* endmembers);
        double* getAbundanceMatrix() { return abundanceMatrix; };
    private:
        void runOnCPU(int maxIter, const double* image, const double* endmembers);
        void runOnGPU(int maxIter, const double* image, const double* endmembers);
};

#endif