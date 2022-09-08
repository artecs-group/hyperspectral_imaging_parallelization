#include <cstdlib>
#include <iostream>
#include <vector>

#include "mkl.h"
#include "mkl_omp_offload.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include <sys/time.h>
#include <sys/resource.h>

#include <omp.h>

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

    for(i=0; i<lines_samples*targets; i++)
      abundanceVector[i]=1;

    double alpha = 1.0;
    double beta = 0.0;

    #pragma omp target data                                   \
    map(to: image_vector[0:lines_samples * bands])            \
    map(to: endmembers[0:bands * targets])                    \
    map(tofrom: abundanceVector[0:lines_samples * targets] )  \
    map(to: h_Num[0:lines_samples * targets])                 \
    map(to: h_aux[0:lines_samples * bands])                   \
    map(to: h_Den[0:lines_samples * targets])
    {
      #pragma omp target variant dispatch use_device_ptr(image_vector,endmembers,h_Num)
      {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, lines_samples, targets, bands, alpha, image_vector,lines_samples, endmembers, targets, beta, h_Num, lines_samples);
      }

      for(i=0; i<it; i++)
      {
        #pragma omp target variant dispatch use_device_ptr(abundanceVector,endmembers,h_aux)
        {
          cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, lines_samples, bands, targets, alpha, abundanceVector, lines_samples, endmembers, targets, beta, h_aux, lines_samples);
        }

        #pragma omp target variant dispatch use_device_ptr(h_aux,endmembers,h_Den)
        {
          cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, lines_samples, targets, bands, alpha, h_aux, lines_samples, endmembers, targets, beta, h_Den, lines_samples);
        }

        #pragma omp target teams distribute parallel for
        for (j = 0; j < lines_samples * targets; j++) {
          abundanceVector[j] = abundanceVector[j]*(h_Num[j]/h_Den[j]);
        }
      }
    }

#ifdef DEBUG
    printf("abundanceVector %f %f %f %f\n", 
            abundanceVector[0], abundanceVector[lines_samples], abundanceVector[lines_samples*targets/2], abundanceVector[3*lines_samples*targets/4] );
#endif

  free(h_Den);
  free(h_Num);
  free(h_aux);
}
