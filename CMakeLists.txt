cmake_minimum_required(VERSION 3.13)

set(This hyperspectral_imaging_parallelization)

project(${This} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_program(CMAKE_CXX_COMPILER NAMES icpx)
set(CMAKE_CXX_COMPILER icpx)

find_package(GTest REQUIRED)
enable_testing()

add_subdirectory(unittests)