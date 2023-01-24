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
if(DEV_TYPE STREQUAL "X86")
    target_compile_options(${This} PRIVATE "-I\"${MKLROOT}/include\"" "-qmkl" "-DMKL_ILP64" "-O3" "-qopenmp" "-xhost")
elseif(DEV_TYPE STREQUAL "INTEL_GPU")
    target_compile_options(${This} PRIVATE "-I\"${MKLROOT}/include\"" "-qmkl" "-DMKL_ILP64" "-O3" "-qopenmp" "-fopenmp-targets=spir64")
else()# NVIDIA_GPU
endif()