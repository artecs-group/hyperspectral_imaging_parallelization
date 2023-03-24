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

if(DEVICE STREQUAL "igpu")
    set(DEV "GPU")
    set(DEV_TYPE "INTEL_GPU")
    message(STATUS "Chosen Intel GPU as the device to run.")
elseif(DEVICE STREQUAL "ngpu")
    find_package(CUDA REQUIRED)
    set(DEV "GPU")
    set(DEV_TYPE "NVIDIA_GPU")
    message(STATUS "Chosen NVIDIA GPU as the device to run.")
else() # CPU
    set(DEV "CPU")
    set(DEV_TYPE "X86")
    message(STATUS "Chosen CPU as the device to run.")
endif()

if(IMPL STREQUAL "sycl" AND DEVICE STREQUAL "ngpu")
    find_program(CMAKE_CXX_COMPILER NAMES clang++)
    set(CMAKE_CXX_COMPILER clang++)
elseif(IMPL STREQUAL "sequential" OR IMPL STREQUAL "openmp" OR IMPL STREQUAL "sycl")
    set(CMAKE_CXX_COMPILER icpx)
    #find_package(IntelDPCPP REQUIRED)
elseif(IMPL STREQUAL "kokkos")
    if(NOT DEFINED ENV{KOKKOS_INSTALL_DIR})
        set(KOKKOS_INSTALL_DIR "/opt/kokkos/build/" CACHE PATH "Path to which Kokkos has been installed") 
    endif()
    if(NOT DEFINED ENV{KOKKOS_KERNELS_INSTALL_DIR})
        set(KOKKOS_KERNELS_INSTALL_DIR "/opt/kokkos-kernels/build/" CACHE PATH "Path to which Kokkos Kernels has been installed") 
    endif()
    
    set(CMAKE_CXX_EXTENSIONS Off)
    if(DEVICE STREQUAL "igpu")
        set(Kokkos_ARCH_INTEL_GEN9 ON)
        set(Kokkos_ENABLE_CXX11_DISPATCH_LAMBDA ON)
        set(CMAKE_CXX_COMPILER icpx)
    else()
        set(Kokkos_ENABLE_CUDA_LAMBDA ON)
        set(Kokkos_ARCH_PASCAL61 ON)
        set(Kokkos_ENABLE_CXX11_DISPATCH_LAMBDA ON)
        set(CMAKE_CXX_COMPILER ${KOKKOS_INSTALL_DIR}/bin/nvcc_wrapper)
    endif()

    set(Kokkos_ROOT ${KOKKOS_INSTALL_DIR}/lib/cmake/Kokkos)
    set(KokkosKernels_ROOT ${KOKKOS_KERNELS_INSTALL_DIR}/lib/cmake/KokkosKernels)
    find_package(KokkosKernels REQUIRED)
endif()

if(PDEBUG AND PDEBUG STREQUAL "yes")
    set(DEBUG DEBUG)
    message(STATUS "Activated application debug info.")
else()
    set(DEBUG NO_DEBUG)
endif()

find_package(GTest REQUIRED)
enable_testing()

if(IMPL STREQUAL "sequential")
    message(STATUS "Chosen Sequential implementation to run.")
    set(IMPLEMENTATION "SEQUENTIAL")
    add_subdirectory(sequential)
elseif(IMPL STREQUAL "openmp")
    message(STATUS "Chosen OpenMP implementation to run.")
    set(IMPLEMENTATION "OPENMP")
    add_subdirectory(openmp)
elseif(IMPL STREQUAL "kokkos")
    message(STATUS "Chosen Kokkos implementation to run.")
    set(IMPLEMENTATION "KOKKOS")
    add_subdirectory(kokkos)
else() # SYCL
    message(STATUS "Chosen SYCL implementation to run.")
    set(IMPLEMENTATION "SYCL")
    add_subdirectory(sycl)
endif()

add_subdirectory(common)
#add_subdirectory(unittests)