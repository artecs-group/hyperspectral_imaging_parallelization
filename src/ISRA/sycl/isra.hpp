#ifndef _ISRA_SYCL_
#define _ISRA_SYCL_

#include "../../utils/sycl_selector.hpp"
#include "../isra_interface.hpp"

class SYCL_ISRA: I_ISRA {
    public:
        SYCL_ISRA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers);
        ~SYCL_ISRA();
        void run(int maxIter, const double* hImage, const double* hEndmembers);
        double* getAbundanceMatrix() { return abundanceMatrix; };
    private:
        sycl::queue _queue;
        double *image{nullptr}, 
               *endmembers{nullptr};
};

#endif