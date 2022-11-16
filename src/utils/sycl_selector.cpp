#include "sycl_selector.hpp"

sycl::queue get_queue(){
#if defined(INTEL_GPU)
	IntelGPUSelector selector{};
#elif defined(NVIDIA_GPU)
	NvidiaGPUSelector selector{};
#elif defined(CPU)
	sycl::cpu_selector selector{};
#else
	default_selector selector{};
#endif

	sycl::queue queue{selector};
    std::cout << "Running on: "
              << queue.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    return queue;
}