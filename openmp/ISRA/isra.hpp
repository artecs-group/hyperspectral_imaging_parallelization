#ifndef _ISRA_OPENMP_
#define _ISRA_OPENMP_

#include "../../common/interfaces/isra_interface.hpp"

class OpenMP_ISRA: I_ISRA {
    public:
        OpenMP_ISRA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers);
        ~OpenMP_ISRA();
        void run(int maxIter, const double* image, const double* endmembers);
        double* getAbundanceMatrix() { return abundanceMatrix; };
        void clearMemory();
    protected:
        void preProcessAbundance(const double* image, double* Ab, const double* e, int targetEndmembers, int lines, int samples, int bands);
        void invTR(double* A, int p);
    private:
        void runOnCPU(int maxIter, const double* image, const double* endmembers);
        void runOnGPU(int maxIter, const double* image, const double* endmembers);
        double *abundanceMatrix{nullptr}, 
                *numerator{nullptr}, 
                *denominator{nullptr}, 
                *aux{nullptr},
                *Et_E{nullptr},
                *comput{nullptr};
        long long* ipiv{nullptr};
};

#endif