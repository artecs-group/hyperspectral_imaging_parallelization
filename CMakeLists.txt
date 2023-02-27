cmake_minimum_required(VERSION 3.13)

set(This hyperspectral_imaging_parallelization)

project(${This} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(supported_implementations
    sequential
    sycl
    openmp
)

set(supported_devices
    cpu
    igpu # intel GPU
    ngpu # nvidia GPU
)

list(FIND supported_implementations ${IMPL} impl_idx)
if(NOT IMPL OR impl_idx EQUAL -1)
    message(FATAL_ERROR "Please specify IMPL (one of: ${supported_implementations})")
endif()

if(DEVICE)
    list(FIND supported_devices ${DEVICE} device_idx)
    if(device_idx EQUAL -1)
        message(FATAL_ERROR "Please specify DEVICE (one of: ${supported_devices})")
    endif()
else()
    set(DEVICE cpu)
endif()


if(IMPL STREQUAL "sycl" AND DEVICE STREQUAL "ngpu")
    find_program(CMAKE_CXX_COMPILER NAMES clang++)
    set(CMAKE_CXX_COMPILER clang++)
else()
    find_program(CMAKE_CXX_COMPILER NAMES icpx)
    set(CMAKE_CXX_COMPILER icpx)
endif()

if(PDEBUG AND PDEBUG STREQUAL "yes")
    set(DEBUG DEBUG)
else()
    set(DEBUG NO_DEBUG)
endif()

#find_package(GTest REQUIRED)
#enable_testing()

if(DEVICE STREQUAL "igpu")
    set(DEV "GPU")
    set(DEV_TYPE "INTEL_GPU")
elseif(DEVICE STREQUAL "ngpu")
    set(DEV "GPU")
    set(DEV_TYPE "NVIDIA_GPU")
else() # CPU
    set(DEV "CPU")
    set(DEV_TYPE "X86")
endif()

if(IMPL STREQUAL "sequential")
    set(IMPLEMENTATION "SEQUENTIAL")
    add_subdirectory(sequential)
elseif(IMPL STREQUAL "openmp")
    set(IMPLEMENTATION "OPENMP")
    add_subdirectory(openmp)
else() # SYCL
    set(IMPLEMENTATION "SYCL")
    add_subdirectory(sycl)
endif()

add_subdirectory(common)
#add_subdirectory(unittests)