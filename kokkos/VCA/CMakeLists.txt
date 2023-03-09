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
target_compile_options(${This} PRIVATE "-O3")
target_link_libraries(${This} Kokkos::kokkos Kokkos::kokkoskernels)