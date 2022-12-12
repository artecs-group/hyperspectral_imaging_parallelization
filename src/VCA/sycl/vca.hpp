#ifndef _VCA_SYCL_
#define _VCA_SYCL_

#include <CL/sycl.hpp>

#include "../../utils/sycl_selector.hpp"
#include "../vca_interface.hpp"

class SYCL_VCA: I_VCA {
    public:
        SYCL_VCA(int _lines, int _samples, int _bands, unsigned int _targetEndmembers);
        ~SYCL_VCA();
        void run(float SNR, const double* image);
        double* getEndmembers() { return endmembers; };
    private:
        sycl::queue _queue;
        float* dSNR{nullptr};
        int64_t scrach_size, pinv_size;
        double *gesvd_scratchpad{nullptr}, 
               *pinv_scratchpad{nullptr}, 
               *dImage{nullptr}, 
               *sum1Sqrt{nullptr};
};

#endif