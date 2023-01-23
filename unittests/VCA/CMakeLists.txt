cmake_minimum_required(VERSION 3.13)

set(This vca)

set(Headers
    ${PROJECT_SOURCE_DIR}/src/utils/file_utils.hpp
    ${PROJECT_SOURCE_DIR}/src/VCA/sequential/vca.hpp
)

set(Sources
    vca.cpp
    ${PROJECT_SOURCE_DIR}/src/VCA/sequential/vca.cpp
)

add_executable(${This} ${Sources})
target_compile_options(${This} PRIVATE "-I\"${MKLROOT}/include\"" "-qmkl" "-DMKL_ILP64")
target_link_libraries(${This} GTest::GTest GTest::Main)
link_directories(${MKLROOT}/lib/intel64)
target_link_libraries(${This} -Wl,--start-group libmkl_intel_lp64.a libmkl_sequential.a libmkl_core.a -Wl,--end-group -lpthread -lm -ldl)

add_custom_target(run
    COMMAND ${This}
    WORKING_DIRECTORY .
    DEPENDS ${This})