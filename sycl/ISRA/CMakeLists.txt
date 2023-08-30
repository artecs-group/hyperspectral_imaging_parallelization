cmake_minimum_required(VERSION 3.20.5)

set(This isra)

set(Headers
    isra.hpp
)

set(Sources
    isra.cpp
)

add_library(${This} ${Sources} ${Headers})
add_compile_options(-g $<$<BOOL:${WITH_DEBUG}>)
add_definitions(-D"${IMPLEMENTATION}" -D"${DEV}" -D"${DEV_TYPE}" -D"${DEBUG}")
target_compile_options(${This} PRIVATE "-O3" "-fsycl-unnamed-lambda" "-fsycl")
if(DEV_TYPE STREQUAL "NVIDIA_GPU")
    target_compile_options(${This} PRIVATE "-fsycl-targets=nvptx64-nvidia-cuda")
    target_include_directories(${This} PRIVATE $ENV{ONEMKL}/include)
else()
    target_compile_options(${This} PRIVATE "-qmkl" "-DMKL_ILP64")
    target_include_directories(${This} PRIVATE $ENV{MKLROOT}/include)
endif()