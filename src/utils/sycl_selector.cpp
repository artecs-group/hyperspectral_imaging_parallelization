#include "sycl_selector.hpp"

static sycl::queue* dev_queue;

sycl::queue get_queue(){
    if(dev_queue)
        return *dev_queue;

#if defined(INTEL_GPU)
	IntelGPUSelector selector{};
#elif defined(NVIDIA_GPU)
	NvidiaGPUSelector selector{};
#elif defined(CPU)
	sycl::cpu_selector selector{};
#else
	default_selector selector{};
#endif

    dev_queue = new sycl::queue{selector};
    std::cout << std::endl << "Running on: "
            << dev_queue->get_device().get_info<sycl::info::device::name>()
            << std::endl << std::endl;

    return *dev_queue;
}