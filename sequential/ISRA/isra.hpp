#ifndef _ISRA_SEQUENTIAL_
#define _ISRA_SEQUENTIAL_

#include "../../common/interfaces/isra_interface.hpp"

class SequentialISRA: I_ISRA {
    public:
        SequentialISRA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers);
        ~SequentialISRA();
        void run(int maxIter, const double* image, const double* endmembers);
        double* getAbundanceMatrix() { return abundanceMatrix; };
        void clearMemory();
    protected:
        void preProcessAbundance(const double* image, double* Ab, const double* e, int targetEndmembers, int lines, int samples, int bands);
        void invTR(double* A, int p);
    private:
        double *abundanceMatrix{nullptr}, 
                *numerator{nullptr}, 
                *denominator{nullptr}, 
                *aux{nullptr},
                *Et_E{nullptr},
                *comput{nullptr};
        long long* ipiv{nullptr};
};

#endif