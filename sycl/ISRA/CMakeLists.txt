cmake_minimum_required(VERSION 3.13)

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
if(DEV_TYPE STREQUAL "NVIDIA_GPU")
    target_compile_options(${This} PRIVATE "-I${ONEMKL}/include" "-O3" "-fsycl-unnamed-lambda" "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda")
else()
    target_compile_options(${This} PRIVATE "-I${MKLROOT}/include" "-qmkl" "-DMKL_ILP64" "-O3" "-fsycl-unnamed-lambda" "-fsycl")
endif()