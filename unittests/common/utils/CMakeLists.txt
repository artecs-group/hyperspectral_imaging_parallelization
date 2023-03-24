cmake_minimum_required(VERSION 3.13)

set(This matrix_operations.exe)

set(Headers
    ${PROJECT_SOURCE_DIR}/common/utils/matrix_operations.hpp
)

set(Sources
    matrix_operations.cpp
)

add_executable(${This} ${Sources} ${Headers})
target_compile_options(${This} PRIVATE "-qmkl" "-DMKL_ILP64" "-O3")
target_include_directories(${This} PRIVATE $ENV{MKLROOT}/include)
target_link_directories(${This} PRIVATE $ENV{MKLROOT}/lib/intel64)
target_link_libraries(${This} GTest::GTest GTest::Main "-qmkl" "-DMKL_ILP64" "-O3" "-Wl,--start-group" mkl_intel_lp64 mkl_sequential mkl_core "-Wl,--end-group" pthread m dl)

add_test(
    NAME ${This} 
    COMMAND ${This}
)