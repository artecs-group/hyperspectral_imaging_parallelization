#include "sycl_selector.hpp"

static sycl::queue* dev_queue{nullptr};

sycl::queue get_queue(){
    if(dev_queue != nullptr)
        return *dev_queue;

#if defined(INTEL_GPU)
    auto intel_gpu_selector = [](const sycl::device &Device) {
        const std::string vendor = Device.get_info<sycl::info::device::vendor>();

        if (Device.is_gpu() && (vendor.find("Intel(R) Corporation") != std::string::npos))
            return 1;

        return -1;
    };
    dev_queue = new sycl::queue{intel_gpu_selector};
#elif defined(NVIDIA_GPU)
    auto nvidia_gpu_selector = [](const sycl::device &Device) {
        const std::string DriverVersion = Device.get_info<sycl::info::device::driver_version>();

        if (Device.is_gpu() && (DriverVersion.find("CUDA") != std::string::npos))
            return 1;

        return -1;
    };
    dev_queue = new sycl::queue{nvidia_gpu_selector};
#elif defined(CPU)
    dev_queue = new sycl::queue{sycl::cpu_selector_v};
#else
    dev_queue = new sycl::queue{sycl::default_selector_v};
#endif

    std::cout << std::endl << "Running on: "
            << dev_queue->get_device().get_info<sycl::info::device::name>()
            << std::endl << std::endl;

    return *dev_queue;
}


bool isQueueInit(){
    return dev_queue != nullptr;
}