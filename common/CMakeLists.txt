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
    target_compile_options(${This} PRIVATE "-I${MKLROOT}/include" "-qmkl" "-DMKL_ILP64" "-O3")
    link_directories(${MKLROOT}/lib/intel64)
    target_link_libraries(${This} ${Libraries} -Wl,--start-group libmkl_intel_lp64.a libmkl_sequential.a libmkl_core.a -Wl,--end-group -lpthread -lm -ldl)
elseif(IMPLEMENTATION STREQUAL "OPENMP")
    if(DEV_TYPE STREQUAL "X86")
        target_compile_options(${This} PRIVATE "-I${MKLROOT}/include" "-qmkl" "-DMKL_ILP64" "-O3" "-qopenmp" "-xhost")
        link_directories(${MKLROOT}/lib/intel64)
        target_link_libraries(${This} ${Libraries} -Wl,--start-group libmkl_intel_lp64.a libmkl_intel_thread.a libmkl_core.a -Wl,--end-group -liomp5 -lpthread -lm -ldl)
    elseif(DEV_TYPE STREQUAL "INTEL_GPU")
        target_compile_options(${This} PRIVATE "-I${MKLROOT}/include" "-qmkl" "-DMKL_ILP64" "-O3" "-qopenmp" "-fopenmp-targets=spir64")
        link_directories(${MKLROOT}/lib/intel64)
        target_link_libraries(${This} ${Libraries} -fsycl -lmkl_sycl -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lsycl -lOpenCL -lstdc++ -lpthread -lm -ldl)
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
    if(DEV_TYPE STREQUAL "NVIDIA_GPU")
        target_compile_options(${This} PRIVATE "-I${ONEMKL}/include" "-O3" "-fsycl-unnamed-lambda" "-fsycl" "-fsycl-targets=nvptx64-nvidia-cuda")
        link_directories(${ONEMKL}/lib)
        target_link_libraries(${This} ${Sycl_libs} -fsycl-device-code-split=per_kernel -lonemkl)
    else()
        target_compile_options(${This} PRIVATE "-I${MKLROOT}/include" "-qmkl" "-DMKL_ILP64" "-O3" "-fsycl-unnamed-lambda" "-fsycl")
        link_directories(${MKLROOT}/lib/intel64)
        target_link_libraries(${This} ${Sycl_libs} -fsycl-device-code-split=per_kernel libmkl_sycl.a -Wl,-export-dynamic -Wl,--start-group libmkl_intel_ilp64.a libmkl_tbb_thread.a libmkl_core.a -Wl,--end-group -L${TBBROOT}/lib/intel64/gcc4.8 -ltbb -lsycl -lOpenCL -lpthread -lm -ldl)
    endif()
endif()


add_custom_target(run
    COMMAND ./${This} ${PROJECT_SOURCE_DIR}/data/Cuprite 5 0 5
    DEPENDS ${This}
    WORKING_DIRECTORY .
)