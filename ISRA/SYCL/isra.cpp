#include <CL/sycl.hpp>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "oneapi/mkl.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <sys/time.h>
#include <sys/resource.h>

void ISRA(double *image_vector, double *endmembers, double *abundanceVector, int it, int lines, int samples, int bands, int targets)
{
    int lines_samples = lines*samples;
    
    sycl::queue my_queue{sycl::default_selector{}};

    std::cout << "Device: "
              << my_queue.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    
    double* image_vector_usm = sycl::malloc_device<double>(lines_samples*bands, my_queue);
    double* endmembers_usm = sycl::malloc_device<double>(targets*bands, my_queue);
    double* abundanceVector_usm = sycl::malloc_device<double>(lines_samples*targets, my_queue);
    double* h_Num_usm = sycl::malloc_device<double>(lines_samples*targets, my_queue);
    double* h_aux_usm = sycl::malloc_device<double>(lines_samples*bands, my_queue);
    double* h_Den_usm = sycl::malloc_device<double>(lines_samples*targets, my_queue);
    auto nonTrans = oneapi::mkl::transpose::nontrans;
    auto yesTrans = oneapi::mkl::transpose::trans;

    int i,j;

    my_queue.submit([&] (sycl::handler& h) {
      h.memcpy(image_vector_usm, image_vector,sizeof(double) * lines_samples*bands);
    }).wait();
    
    my_queue.submit([&] (sycl::handler& h) {
      h.memcpy(endmembers_usm, endmembers,sizeof(double) * targets*bands);
    }).wait();
    
    my_queue.parallel_for(sycl::range<1> (lines_samples*targets), [=] (sycl::id<1> i){
        abundanceVector_usm[i]=1;
    }).wait();

    double alpha = 1.0;
    double beta = 0.0;
    
	  oneapi::mkl::blas::column_major::gemm(my_queue, nonTrans, yesTrans, lines_samples, targets, bands, alpha, image_vector_usm,lines_samples, endmembers_usm, targets, beta, h_Num_usm, lines_samples); // Revisar las llamadas, hacer un codigo a parte y ver resultados, comprobar primera practica de GPUs, correspondencia row y column major, ejemplo 4, pagina 24; A * B = (B T * A T ) T, cuidado al trasponer
    
    my_queue.wait();

    for(i=0; i<it; i++)
    {
        oneapi::mkl::blas::column_major::gemm(my_queue, nonTrans, nonTrans, lines_samples, bands, targets, alpha, abundanceVector_usm, lines_samples, endmembers_usm, targets, beta, h_aux_usm, lines_samples);
        
        my_queue.wait();

        oneapi::mkl::blas::column_major::gemm(my_queue, nonTrans, yesTrans, lines_samples, targets, bands, alpha, h_aux_usm, lines_samples, endmembers_usm, targets, beta, h_Den_usm, lines_samples);

        my_queue.wait();

    		my_queue.parallel_for(sycl::range<1> (lines_samples*targets), [=] (sycl::id<1> j){
                abundanceVector_usm[j] = abundanceVector_usm[j]*(h_Num_usm[j]/h_Den_usm[j]);
    		}).wait();
    }
    
    my_queue.submit([&] (sycl::handler& h) {
      h.memcpy(abundanceVector, abundanceVector_usm,sizeof(double) * lines_samples*targets);
    }).wait();

#ifdef DEBUG
    printf("abundanceVector %f %f %f %f\n", 
            abundanceVector[0], abundanceVector[lines_samples], abundanceVector[lines_samples*targets/2], abundanceVector[3*lines_samples*targets/4] );

#endif

	free(image_vector_usm, my_queue);
	free(endmembers_usm, my_queue);
	free(abundanceVector_usm, my_queue);
	free(h_Num_usm, my_queue);
	free(h_aux_usm, my_queue);
	free(h_Den_usm, my_queue);
}
