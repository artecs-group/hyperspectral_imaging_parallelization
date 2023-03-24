cmake_minimum_required(VERSION 3.13)

if(CMAKE_CXX_COMPILER STREQUAL icpx)
    add_subdirectory(common)
    add_subdirectory(sequential)
endif()
