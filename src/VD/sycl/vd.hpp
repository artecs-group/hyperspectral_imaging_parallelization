#ifndef _VD_SYCL_
#define _VD_SYCL_

#include <CL/sycl.hpp>
#include "../vd_interface.hpp"

class SYCL_VD: I_VD {
    public:
        SYCL_VD(int _lines, int _samples, int _bands);
        ~SYCL_VD();
        void run(const int approxVal, const double* h_image);
        unsigned int getNumberEndmembers() { return endmembers; };
    private:
        sycl::queue _get_queue();
        sycl::queue _queue;
        double *mean, *gesvd_scratchpad;
        int64_t _scrach_size;
};

// CUDA device selector
class NvidiaGPUSelector : public sycl::device_selector {
    public:
        int operator()(const sycl::device &Device) const override {
            const std::string DriverVersion = Device.get_info<sycl::info::device::driver_version>();

            if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos))
                return 1;

            return 0;
        }
};

// Intel GPU
class IntelGPUSelector : public sycl::device_selector {
    public:
        int operator()(const sycl::device &Device) const override {
            const std::string vendor = Device.get_info<sycl::info::device::vendor>();

            if (Device.is_gpu() && (vendor.find("Intel(R) Corporation") != std::string::npos))
                return 1;

            return 0;
        }
};

#endif