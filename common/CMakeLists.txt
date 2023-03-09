cmake_minimum_required(VERSION 3.13)

set(This main.exe)

set(Headers
    utils/file_utils.hpp
    utils/matrix_operations.hpp
    interfaces/isra_interface.hpp
    interfaces/vca_interface.hpp
    interfaces/vd_interface.hpp
)

set(Sources
    main.cpp
)

if(IMPLEMENTATION STREQUAL "SEQUENTIAL")
    set(Lib_path ${CMAKE_BINARY_DIR}/sequential)
elseif(IMPLEMENTATION STREQUAL "OPENMP")
    set(Lib_path ${CMAKE_BINARY_DIR}/openmp)
elseif(IMPLEMENTATION STREQUAL "KOKKOS")
    set(Lib_path ${CMAKE_BINARY_DIR}/kokkos)
else() # SYCL
    set(Lib_path ${CMAKE_BINARY_DIR}/sycl)
endif()

set(Libraries
    ${Lib_path}/ISRA/libisra.a
    ${Lib_path}/VCA/libvca.a
    ${Lib_path}/VD/libvd.a
)

add_executable(${This} ${Sources} ${Headers})
add_compile_options(-g $<$<BOOL:${WITH_DEBUG}>)
add_definitions(-D"${IMPLEMENTATION}" -D"${DEV}" -D"${DEV_TYPE}" -D"${DEBUG}")

if(IMPLEMENTATION STREQUAL "SEQUENTIAL")
    set(CFLAGS "-qmkl" "-DMKL_ILP64" "-O3")
    target_compile_options(${This} PRIVATE ${CFLAGS})
    target_include_directories(${This} PRIVATE $ENV{MKLROOT}/include)
    target_link_directories(${This} PRIVATE $ENV{MKLROOT}/lib/intel64)
    target_link_libraries(${This} ${CFLAGS} "-Wl,--start-group" mkl_intel_lp64 mkl_sequential mkl_core "-Wl,--end-group" pthread m dl ${Libraries})
elseif(IMPLEMENTATION STREQUAL "OPENMP")
    set(CFLAGS "-qmkl" "-DMKL_ILP64" "-O3" "-qopenmp")
    target_compile_options(${This} PRIVATE ${CFLAGS})
    target_include_directories(${This} PRIVATE $ENV{MKLROOT}/include)
    target_link_directories(${This} PRIVATE $ENV{MKLROOT}/lib/intel64 $ENV{CMPLR_ROOT}/linux/lib)
    if(DEV_TYPE STREQUAL "X86")
        list(APPEND CFLAGS "-xhost")
        target_compile_options(${This} PRIVATE ${CFLAGS})
        target_link_libraries(${This} ${CFLAGS} "-Wl,--start-group" mkl_intel_lp64 mkl_intel_thread mkl_core "-Wl,--end-group" iomp5 pthread m dl ${Libraries})
    elseif(DEV_TYPE STREQUAL "INTEL_GPU")
        list(APPEND CFLAGS "-fsycl" "-fopenmp-targets=spir64" "-qopenmp")
        target_compile_options(${This} PRIVATE ${CFLAGS})
        target_link_libraries(${This} ${CFLAGS} mkl_sycl mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 sycl OpenCL stdc++ pthread m dl ${Libraries})
    else()# NVIDIA_GPU
    # CC = nvc++
    # CFLAGS = -DGPU -DNVIDIA_GPU -I/usr/local/cuda/include -lcuda -lcublas -lcusolver -fopenmp -mp=gpu
    # CLINK = -L/usr/local/cuda/lib64 -lcudart
    endif()
elseif(IMPLEMENTATION STREQUAL "KOKKOS")
    set(CFLAGS "-O3")
    target_compile_options(${This} PRIVATE ${CFLAGS})
    target_link_libraries(${This} ${CFLAGS} Kokkos::kokkos Kokkos::kokkoskernels ${Libraries})
else() #SYCL
    set(Sycl_libs 
        ${Libraries}
        ${Lib_path}/utils/libsycl_selector.a
    )
    set(CFLAGS "-O3" "-fsycl-unnamed-lambda" "-fsycl")
    target_compile_options(${This} PRIVATE ${CFLAGS})
    if(DEV_TYPE STREQUAL "NVIDIA_GPU")
        list(APPEND CFLAGS "-fsycl-targets=nvptx64-nvidia-cuda")
        target_compile_options(${This} PRIVATE ${CFLAGS})
        target_include_directories(${This} PRIVATE $ENV{ONEMKL}/include)
        target_link_directories(${This} PRIVATE $ENV{ONEMKL}/lib)
        target_link_libraries(${This} ${CFLAGS} "-fsycl-device-code-split=per_kernel" onemkl sycl ${Sycl_libs})
    else()
        list(APPEND CFLAGS "-qmkl" "-DMKL_ILP64")
        target_compile_options(${This} PRIVATE ${CFLAGS})
        target_include_directories(${This} PRIVATE $ENV{MKLROOT}/include)
        target_link_directories(${This} PRIVATE $ENV{MKLROOT}/lib/intel64 $ENV{TBBROOT}/lib/intel64/gcc4.8)
        target_link_libraries(${This} ${CFLAGS} "-fsycl-device-code-split=per_kernel" mkl_sycl "-Wl,-export-dynamic -Wl,--start-group" mkl_intel_ilp64 mkl_tbb_thread mkl_core "-Wl,--end-group" tbb sycl OpenCL pthread m dl ${Sycl_libs})
    endif()
endif()


add_custom_target(run
    COMMAND ./${This} ${PROJECT_SOURCE_DIR}/data/Cuprite 5 0 5
    DEPENDS ${PROJECT_BINARY_DIR}/common/${This}
    WORKING_DIRECTORY .
)
