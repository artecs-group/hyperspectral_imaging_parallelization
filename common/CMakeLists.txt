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
    target_compile_options(${This} PRIVATE "-qmkl" "-DMKL_ILP64" "-O3")
    target_include_directories(${This} PRIVATE $ENV{MKLROOT}/include)
    target_link_directories(${This} PRIVATE $ENV{MKLROOT}/lib/intel64)
    target_link_libraries(${This} ${Libraries} "-Wl,--start-group" mkl_intel_lp64 mkl_sequential mkl_core "-Wl,--end-group" pthread m dl)
elseif(IMPLEMENTATION STREQUAL "OPENMP")
    target_compile_options(${This} PRIVATE "-qmkl" "-DMKL_ILP64" "-O3" "-qopenmp")
    target_include_directories(${This} PRIVATE $ENV{MKLROOT}/include)
    target_link_directories(${This} PRIVATE $ENV{MKLROOT}/lib/intel64 $ENV{CMPLR_ROOT}/linux/lib)
    if(DEV_TYPE STREQUAL "X86")
        target_compile_options(${This} PRIVATE "-xhost")
        target_link_libraries(${This} ${Libraries} "-Wl,--start-group" mkl_intel_lp64 mkl_intel_thread mkl_core "-Wl,--end-group" iomp5 pthread m dl)
    elseif(DEV_TYPE STREQUAL "INTEL_GPU")
        target_compile_options(${This} PRIVATE "-fsycl" "-fopenmp-targets=spir64")
        target_link_libraries(${This} ${Libraries} "-fsycl" "-fopenmp-targets=spir64" "-qopenmp" mkl_sycl mkl_intel_ilp64 mkl_intel_thread mkl_core iomp5 sycl OpenCL stdc++ pthread m dl)
    else()# NVIDIA_GPU
    # CC = nvc++
    # CFLAGS = -DGPU -DNVIDIA_GPU -I/usr/local/cuda/include -lcuda -lcublas -lcusolver -fopenmp -mp=gpu
    # CLINK = -L/usr/local/cuda/lib64 -lcudart
    endif()
else() #SYCL
    set(Sycl_libs 
        ${Libraries}
        ${Lib_path}/utils/libsycl_selector.a
    )
    target_compile_options(${This} PRIVATE "-O3" "-fsycl-unnamed-lambda" "-fsycl")
    if(DEV_TYPE STREQUAL "NVIDIA_GPU")
        target_compile_options(${This} PRIVATE "-fsycl-targets=nvptx64-nvidia-cuda")
        target_include_directories(${This} PRIVATE $ENV{ONEMKL}/include)
        target_link_directories(${This} PRIVATE $ENV{ONEMKL}/lib)
        target_link_libraries(${This} ${Sycl_libs} "-fsycl-device-code-split=per_kernel" onemkl sycl)
    else()
        target_compile_options(${This} PRIVATE "-qmkl" "-DMKL_ILP64")
        target_include_directories(${This} PRIVATE $ENV{MKLROOT}/include)
        target_link_directories(${This} PRIVATE $ENV{MKLROOT}/lib/intel64 $ENV{TBBROOT}/lib/intel64/gcc4.8)
        target_link_libraries(${This} ${Sycl_libs} "-fsycl-device-code-split=per_kernel" mkl_sycl "-Wl,-export-dynamic -Wl,--start-group" mkl_intel_ilp64 mkl_tbb_thread mkl_core "-Wl,--end-group" tbb sycl OpenCL pthread m dl)
    endif()
endif()


add_custom_target(run
    COMMAND ./${This} ${PROJECT_SOURCE_DIR}/data/Cuprite 5 0 5
    DEPENDS ${This}
    WORKING_DIRECTORY .
)
