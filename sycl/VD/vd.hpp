#ifndef _VD_SYCL_
#define _VD_SYCL_

#include "../utils/sycl_selector.hpp"
#include "../../common/interfaces/vd_interface.hpp"

class SYCL_VD: I_VD {
    public:
        SYCL_VD(int _lines, int _samples, int _bands);
        ~SYCL_VD();
        void run(const int approxVal, const double* h_image);
        unsigned int getNumberEndmembers() { return endmembers; };
        void clearMemory();
    private:
        sycl::queue _queue;
        int64_t _scrach_size;
        unsigned int* count{nullptr};
        double *meanSpect{nullptr}, 
               *Cov{nullptr}, 
               *Corr{nullptr}, 
               *CovEigVal{nullptr}, 
               *CorrEigVal{nullptr}, 
               *U{nullptr}, 
               *VT{nullptr}, 
               *estimation{nullptr}, 
               *meanImage{nullptr},
               *mean{nullptr},
               *gesvd_scratchpad{nullptr};
};

#endif