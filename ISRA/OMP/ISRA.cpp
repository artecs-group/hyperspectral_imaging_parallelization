#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <sys/time.h>
#include <sys/resource.h>

#ifdef _OPENMP
#include <omp.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif


void ISRA(double *image_vector, double *endmembers, double *abundanceVector, int it, int lines, int samples, int bands, int targets)
{
    double *h_Num;
    double *h_aux;
    double *h_Den;
    int lines_samples = lines*samples;

    h_Num = (double*) malloc(lines_samples * targets * sizeof(double));
    h_aux = (double*) malloc(lines_samples * bands * sizeof(double));
    h_Den = (double*) malloc(lines_samples * targets * sizeof(double));

    int i,j;
    
    int deviceNum;
    deviceNum = 1; // 0 es RTX 3090 y 1 es V100
    
    cudaSetDevice(deviceNum);
    
    cudaStream_t stream;
    cublasHandle_t handle_gemm1, handle_gemm2, handle_gemm3;
    cublasCreate(&handle_gemm1);
    cublasCreate(&handle_gemm2);
    cublasCreate(&handle_gemm3);

    for(i=0; i<lines_samples*targets; i++)
        abundanceVector[i]=1;

    double alpha = 1, beta = 0;
    int cublas_error;

    #pragma omp target data device(deviceNum)                 \
    map(to: image_vector[0:lines_samples * bands])            \
    map(to: endmembers[0:bands * targets])                    \
    map(tofrom: abundanceVector[0:lines_samples * targets] )  \
    map(alloc: h_Num[0:lines_samples * targets])              \
    map(alloc: h_aux[0:lines_samples * bands])                \
    map(alloc: h_Den[0:lines_samples * targets])
    {
        #pragma omp target data use_device_ptr(image_vector,endmembers,h_Num) device(deviceNum)
        {
            cublas_error = cublasDgemm(handle_gemm1, CUBLAS_OP_N, CUBLAS_OP_T, lines_samples,targets, bands, &alpha, image_vector, lines_samples, endmembers,targets, &beta, h_Num, lines_samples);
            
            if( cublas_error != CUBLAS_STATUS_SUCCESS )
            {
                printf( "failed cuBLAS execution %d\n", cublas_error );
                exit(1);
            }
        }
        
        cublasGetStream(handle_gemm1, &stream);
        cudaStreamSynchronize(stream);

        for (i = 0; i < it; i++)
        {
            #pragma omp target data use_device_ptr(abundanceVector,endmembers,h_aux) device(deviceNum)
            {
                  cublas_error = cublasDgemm(handle_gemm2, CUBLAS_OP_N, CUBLAS_OP_N, lines_samples, bands, targets, &alpha, abundanceVector, lines_samples, endmembers, targets, &beta, h_aux, lines_samples);
                  
                  if( cublas_error != CUBLAS_STATUS_SUCCESS )
                  {
                      printf( "failed cuBLAS execution %d\n", cublas_error );
                      exit(1);
                  }
            }
            
            cublasGetStream(handle_gemm2, &stream);
            cudaStreamSynchronize(stream);

            #pragma omp target data use_device_ptr(h_aux,endmembers,h_Den) device(deviceNum)
            {
                  cublas_error = cublasDgemm(handle_gemm3, CUBLAS_OP_N, CUBLAS_OP_T, lines_samples, targets, bands, &alpha, h_aux, lines_samples, endmembers, targets, &beta, h_Den, lines_samples);
                  
                  if( cublas_error != CUBLAS_STATUS_SUCCESS )
                  {
                      printf( "failed cuBLAS execution %d\n", cublas_error );
                      exit(1);
                  }
            }
            
            cublasGetStream(handle_gemm3, &stream);
            cudaStreamSynchronize(stream);
            
            #pragma omp target teams distribute parallel for device(deviceNum)
            for (j = 0; j < lines_samples * targets; j++) {
                abundanceVector[j] = abundanceVector[j] * (h_Num[j] / h_Den[j]);
            }
        }
    }
    
#ifdef DEBUG
    printf("abundanceVector %f %f %f %f\n", 
            abundanceVector[0], abundanceVector[lines_samples], abundanceVector[lines_samples*targets/2], abundanceVector[3*lines_samples*targets/4] );
#endif

    cublasDestroy(handle_gemm1);
    cublasDestroy(handle_gemm2);
    cublasDestroy(handle_gemm3);
    
    free(h_Den);
    free(h_Num);
    free(h_aux);
}
