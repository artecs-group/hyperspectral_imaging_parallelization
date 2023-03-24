#ifndef _ISRA_SYCL_
#define _ISRA_SYCL_

#include "../utils/sycl_selector.hpp"
#include "../../common/interfaces/isra_interface.hpp"

class SYCL_ISRA: I_ISRA {
    public:
        SYCL_ISRA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers);
        ~SYCL_ISRA();
        void run(int maxIter, const double* hImage, const double* hEndmembers);
        double* getAbundanceMatrix() { return abundanceMatrix; };
        void clearMemory();
    protected:
        void preProcessAbundance(const double* image, double* Ab, const double* e, int targetEndmembers, int lines, int samples, int bands);
        void invTR(double* A, int p);
    private:
        sycl::queue _queue;
        int64_t getrf_size, getri_size;
        double *image{nullptr}, 
               *endmembers{nullptr},
               *getrf_scratchpad{nullptr},
               *getri_scratchpad{nullptr},
               *abundanceMatrix{nullptr}, 
                *numerator{nullptr}, 
                *denominator{nullptr}, 
                *aux{nullptr},
                *Et_E{nullptr},
                *comput{nullptr};
        int64_t* ipiv{nullptr};
};

#endif