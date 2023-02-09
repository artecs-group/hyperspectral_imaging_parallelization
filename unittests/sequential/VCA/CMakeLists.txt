cmake_minimum_required(VERSION 3.13)

set(This vca.exe)

set(Headers
    ${PROJECT_SOURCE_DIR}/common/utils/matrix_operations.hpp
    ${PROJECT_SOURCE_DIR}/sequential/VCA/vca.hpp
)

set(Sources
    vca.cpp
    ${PROJECT_SOURCE_DIR}/sequential/VCA/vca.cpp
)

add_executable(${This} ${Sources} ${Headers})
add_definitions(-DCMAKE_PROJECT_DIR="${PROJECT_SOURCE_DIR}")
target_include_directories(${This} PRIVATE $ENV{MKLROOT}/include)
target_link_directories(${This} PRIVATE $ENV{MKLROOT}/lib/intel64)
target_compile_options(${This} PRIVATE "-qmkl" "-DMKL_ILP64" "-O3")
target_link_libraries(${This} GTest::GTest GTest::Main "-qmkl" "-DMKL_ILP64" "-O3" "-Wl,--start-group" mkl_intel_lp64 mkl_sequential mkl_core "-Wl,--end-group" pthread m dl)

add_test(
    NAME ${This} 
    COMMAND ${This}
)