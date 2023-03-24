cmake_minimum_required(VERSION 3.13)

set(This isra)

set(Headers
    isra.hpp
)

set(Sources
    isra.cpp
)

add_library(${This} STATIC ${Sources} ${Headers})
add_compile_options(-g $<$<BOOL:${WITH_DEBUG}>)
add_definitions(-D"${IMPLEMENTATION}" -D"${DEV}" -D"${DEV_TYPE}" -D"${DEBUG}")
target_compile_options(${This} PRIVATE "-O3")
target_link_libraries(${This} Kokkos::kokkoskernels)