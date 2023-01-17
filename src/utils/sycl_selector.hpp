#ifndef _SYCL_SELECTOR_
#define _SYCL_SELECTOR_

#include <CL/sycl.hpp>
#include <iostream>

sycl::queue get_queue();
bool isQueueInit();

#endif