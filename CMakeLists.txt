cmake_minimum_required(VERSION 3.13)

set(This hyperspectral_imaging_parallelization)

project(${This} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(supported_implementations
    Sequential
    SYCL
    OpenMP
)

set(supported_devices
    CPU
    IGPU # intel GPU
    NGPU # nvidia GPU
)

list(FIND supported_implementations ${IMPL} impl_idx)
if(NOT IMPL OR impl_idx EQUAL -1)
    message(FATAL_ERROR "Please specify IMPL (one of: ${supported_implementations})")
endif()

list(FIND supported_devices ${DEVICE} device_idx)
if(device_idx EQUAL -1)
    message(FATAL_ERROR "Please specify DEVICE (one of: ${supported_devices})")
endif()


if(IMPL STREQUAL "SYCL" AND DEVICE STREQUAL "NGPU")
    find_program(CMAKE_CXX_COMPILER NAMES clang++)
    set(CMAKE_CXX_COMPILER clang++)
else()
    find_program(CMAKE_CXX_COMPILER NAMES icpx)
    set(CMAKE_CXX_COMPILER icpx)
endif()

find_package(GTest REQUIRED)
enable_testing()

if(DEVICE STREQUAL "IGPU")
    set(DEV "GPU" PARENT_SCOPE)
    set(DEV_TYPE "INTEL_GPU" PARENT_SCOPE)
elseif(DEVICE STREQUAL "NGPU")
    set(DEV "GPU" PARENT_SCOPE)
    set(DEV_TYPE "NVIDIA_GPU" PARENT_SCOPE)
else() # CPU
    set(DEV "CPU" PARENT_SCOPE)
    set(DEV_TYPE "X86" PARENT_SCOPE)
endif()

if(IMPL STREQUAL "Sequential")
    set(IMPLEMENTATION "SEQUENTIAL" PARENT_SCOPE)
    add_subdirectory(sequential)
elseif(IMPL STREQUAL "OpenMP")
    set(IMPLEMENTATION "OPENMP" PARENT_SCOPE)
    add_subdirectory(openmp)
else() # SYCL
    set(IMPLEMENTATION "SYCL" PARENT_SCOPE)
    add_subdirectory(sycl)
endif()

add_subdirectory(common)
add_subdirectory(unittests)