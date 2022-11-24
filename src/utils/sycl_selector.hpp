#ifndef _SYCL_SELECTOR_
#define _SYCL_SELECTOR_

#include <CL/sycl.hpp>
#include <iostream>

sycl::queue get_queue();
bool isQueueInit();

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