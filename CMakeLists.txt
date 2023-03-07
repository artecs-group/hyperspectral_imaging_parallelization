cmake_minimum_required(VERSION 3.13)

set(This hyperspectral_imaging_parallelization)

project(${This} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(supported_implementations
    sequential
    sycl
    openmp
    kokkos
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
    set(CMAKE_CXX_COMPILER icpx)
    #find_package(IntelDPCPP REQUIRED)
endif()

if(IMPL STREQUAL "kokkos")
    if(NOT DEFINED ENV{KOKKOS_INSTALL_DIR})
        set(KOKKOS_INSTALL_DIR "/opt/kokkos/build/" CACHE PATH "Path to which Kokkos has been installed") 
    endif()
    
    set(CMAKE_CXX_COMPILER ${KOKKOS_INSTALL_DIR}/bin/nvcc_wrapper)
    set(CMAKE_CXX_EXTENSIONS Off)
    set(Kokkos_ROOT ${KOKKOS_INSTALL_DIR}/lib/cmake/Kokkos)
    find_package(Kokkos REQUIRED)
endif()

if(PDEBUG AND PDEBUG STREQUAL "yes")
    set(DEBUG DEBUG)
else()
    set(DEBUG NO_DEBUG)
endif()

find_package(GTest REQUIRED)
enable_testing()

if(DEVICE STREQUAL "igpu")
    set(DEV "GPU")
    set(DEV_TYPE "INTEL_GPU")
elseif(DEVICE STREQUAL "ngpu")
    find_package(CUDA REQUIRED)
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
elseif(IMPL STREQUAL "kokkos")
    set(IMPLEMENTATION "KOKKOS")
    add_subdirectory(kokkos)
else() # SYCL
    set(IMPLEMENTATION "SYCL")
    add_subdirectory(sycl)
endif()

add_subdirectory(common)
#add_subdirectory(unittests)