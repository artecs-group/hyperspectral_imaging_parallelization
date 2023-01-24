cmake_minimum_required(VERSION 3.13)

set(This vd)

set(Headers
    vd.hpp
)

set(Sources
    vd.cpp
)

add_library(${This} STATIC ${Sources} ${Headers})
add_compile_options(-g $<$<BOOL:${WITH_DEBUG}>)
add_definitions(-D"${IMPLEMENTATION}" -D"${DEV}" -D"${DEV_TYPE}" -D"${DEBUG}")
target_compile_options(${This} PRIVATE "-I${MKLROOT}/include" "-qmkl" "-DMKL_ILP64" "-O3")