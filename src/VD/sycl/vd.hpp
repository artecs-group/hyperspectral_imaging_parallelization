#ifndef _VD_SYCL_
#define _VD_SYCL_

#include <CL/sycl.hpp>
#include "../vd_interface.hpp"

class SYCL_VD: I_VD {
    public:
        SYCL_VD(int _lines, int _samples, int _bands);
        ~SYCL_VD();
        void run(int approxVal, double* h_image);
    private:
        sycl::queue _get_queue();
        sycl::queue _queue;
        double *image, *mean;
};

#endif