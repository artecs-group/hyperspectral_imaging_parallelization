cmake_minimum_required(VERSION 3.13)

set(This vca)

set(Headers
    vca.hpp
)

set(Sources
    vca.cpp
)

add_library(${This} STATIC ${Sources} ${Headers})
add_compile_options(-g $<$<BOOL:${WITH_DEBUG}>)
add_definitions(-D"${IMPLEMENTATION}" -D"${DEV}" -D"${DEV_TYPE}" -D"${DEBUG}")
target_include_directories(${This} PRIVATE $ENV{MKLROOT}/include)
target_compile_options(${This} PRIVATE "-qmkl" "-DMKL_ILP64" "-O3" "-qopenmp")
if(DEV_TYPE STREQUAL "X86")
    target_compile_options(${This} PRIVATE "-xhost")
elseif(DEV_TYPE STREQUAL "INTEL_GPU")
    target_compile_options(${This} PRIVATE "-fsycl" "-fopenmp-targets=spir64")
else()# NVIDIA_GPU
endif()