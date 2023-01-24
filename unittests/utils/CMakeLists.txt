cmake_minimum_required(VERSION 3.13)

set(This matrix_operations.exe)

set(Headers
    ${PROJECT_SOURCE_DIR}/src/utils/matrix_operations.hpp
)

set(Sources
    matrix_operations.cpp
)

add_executable(${This} ${Sources} ${Headers})
target_compile_options(${This} PRIVATE "-I\"${MKLROOT}/include\"" "-qmkl" "-DMKL_ILP64" "-O3")
target_link_libraries(${This} GTest::GTest GTest::Main)
link_directories(${MKLROOT}/lib/intel64)
target_link_libraries(${This} -Wl,--start-group libmkl_intel_lp64.a libmkl_sequential.a libmkl_core.a -Wl,--end-group -lpthread -lm -ldl)

add_test(
    NAME ${This} 
    COMMAND ${This}
)